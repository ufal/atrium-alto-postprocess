#!/usr/bin/env python3
"""
langID_classify.py

Step 2: Read TXT files → Merge Split Words → Batch classify.

Architecture (CPU/GPU Split Queue):
1. ONE dedicated GPU Worker loops continuously, holding the only perplexity-model
   instance to prevent VRAM OOM errors.
2. MULTIPLE CPU Workers read files, run Regex/FastText, and place texts into a
   multiprocessing Task Queue.
3. CPU Workers poll a Result Dictionary until the GPU returns their Perplexity
   scores.

Robustness (#6): if the GPU worker dies (e.g. model load failure or crash), it
raises a shared `gpu_dead` event. CPU workers detect that — and a hard wall-clock
timeout — instead of spinning forever, so a dead GPU worker fails the run loudly
rather than hanging it.

Input directory (#4): the text source defaults to [CLASSIFY] TEXT_DIR but is
overridden by the LANGID_TEXT_DIR env var, which run_pipeline.py sets to the
selected extraction method's output directory.
"""

import os
import pandas as pd
import torch
from pathlib import Path
import csv
import sys
import time
import queue
from itertools import groupby
import configparser
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from text_util_langID import *
# `from ... import *` skips underscore-prefixed names, so _lang_base (used by the
# language-remapping logic below) must be imported explicitly.
from text_util_langID import _lang_base
from atrium_paradata import ParadataLogger

# Hard ceiling on how long a CPU worker waits for a batch's perplexity before
# declaring the GPU worker unresponsive (#6). Generous so legitimate large
# batches are never killed, but finite so a crash cannot hang the pipeline.
GPU_WAIT_TIMEOUT = 600.0  # seconds

CSV_HEADER = [
    "file", "page_num", "line_num", "text",
    "split_ws", "split_we",
    "lang", "lang_score", "perplex",
    "word_count", "char_count",
    "garbage_density",
    "symbol", "upper", "repeated",
    "ldl_fuses", "fused_words", "gibberish",
    "word_weird", "vowel_ratio", "rot_ratio",
    "quality_score",
    "categ", "caps_header",
    "allcaps_novowel", "lowppl_clear", "cleanprose_clear",
    "trash_threshold", "noisy_threshold", "clear_threshold",
    "pp_dedup", "pp_surrounded_trash", "pp_inverted_run"
]


def gpu_inference_worker(task_queue: mp.Queue, result_dict: dict, model_name: str, gpu_dead=None):
    """
    Standalone background loop that consumes line batches and generates Perplexity
    scores from the Language Model running on a unified GPU instance.

    On fatal model-load failure it sets `gpu_dead` so waiting CPU workers can
    abort instead of polling an empty result dictionary forever.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPU Engine] Initializing {model_name} on {device.upper()}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
        ).to(device)
        model.eval()
        print(f"[GPU Engine] {model_name} ready. Waiting for text batches...")
    except Exception as e:
        print(f"[GPU Engine] Failed to load model: {e}")
        if gpu_dead is not None:
            gpu_dead.set()  # (#6) signal CPU workers so they don't hang
        return

    while True:
        try:
            msg = task_queue.get(timeout=1.0)
            if msg == "STOP":
                print("[GPU Engine] Received STOP signal. Shutting down.")
                break

            batch_id, texts = msg
            ppls = calculate_perplexity_batch(texts, model, tokenizer, device)
            try:
                result_dict[batch_id] = ppls
            except (BrokenPipeError, OSError) as e:
                print(f"[GPU Engine] Dropped result for batch {batch_id}: {e}")
                continue

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU Engine Error] Processing batch: {e}")
            if 'msg' in locals() and msg != "STOP":
                result_dict[msg[0]] = [0.0] * len(msg[1])


worker_models = {}


def init_cpu_worker():
    """Initializes the CPU-bound FastText model once per spawned process."""
    import fasttext
    worker_models['ft'] = fasttext.load_model("lid.176.bin")


def write_rows_to_doc(output_dir: Path, file_id: str, rows: list):
    """Safely appends classified rows to a specific document's CSV file."""
    out_path = output_dir / f"{file_id}.csv"
    file_exists = out_path.exists()

    with open(out_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def process_and_write_batch_cpu(batch_id: str, lines: list, meta: list, out_dir: Path,
                                task_queue: mp.Queue, result_dict: dict, expected_langs: list = None,
                                trusted_langs: list = None, gpu_dead=None):
    """Evaluates heuristical bounds, queries FastText, and fetches PPL to finalize scores."""
    ft = worker_models['ft']

    task_queue.put((batch_id, lines))

    lines_lower = [line.lower() for line in lines]
    labels, scores = ft.predict(lines_lower, k=1)
    langs = [l[0].replace("__label__", "") for l in labels]
    scores = [s[0] for s in scores]

    # FastText returns ISO 639-3 codes with a script suffix (e.g. "deu_Latn").
    # The expected/trusted lists are bare base codes — compare on base only.
    _known_lang_bases = frozenset(
        _lang_base(l) for l in ((trusted_langs or []) + (expected_langs or []))
    )

    # (#6) Bounded wait: abort if the GPU worker died or never answers.
    waited = 0.0
    while batch_id not in result_dict:
        if gpu_dead is not None and gpu_dead.is_set():
            raise RuntimeError(
                f"GPU inference worker is down; cannot score batch {batch_id}"
            )
        time.sleep(0.01)
        waited += 0.01
        if waited >= GPU_WAIT_TIMEOUT:
            raise RuntimeError(
                f"Timed out after {GPU_WAIT_TIMEOUT:.0f}s waiting for perplexity "
                f"of batch {batch_id}; GPU worker unresponsive."
            )

    ppls = result_dict.pop(batch_id)

    results = []
    for i in range(len(lines)):
        file_id, page_id, line_num, text_content, split_ws, split_we = meta[i]

        original_lang_score = scores[i]

        wc = len(text_content.split())
        cc = len(text_content)

        if _lang_base(langs[i]) not in _known_lang_bases:
            # Remap to the collection default, but PRESERVE any script suffix.
            suffix = langs[i][len(_lang_base(langs[i])):]
            langs[i] = expected_langs[0] + suffix
            scores[i] = max(scores[i], LANG_SCORE_CLEAR)

        ppl_val = ppls[i]

        if wc <= 2 and ppl_val > SHORT_PPL_CAP:
            ppl_val = SHORT_PPL_CAP

        g_density = compute_garbage_density(text_content)

        sym_count = detect_strange_symbols(text_content)
        upper_count = detect_mid_uppercase(text_content)
        rep_count = detect_repeated_chars(text_content)
        fuse_count = detect_letter_digit_letter(text_content)
        fused_words = detect_fused_words(text_content)
        gibb_count = detect_gibberish_words(text_content)

        vowel_ratio = compute_vowel_ratio(text_content)
        rot_ratio = compute_rotatable_ratio(text_content)
        caps_header = is_all_caps_line(text_content)

        word_scores = score_words_in_line(text_content)
        weird_ratio = compute_word_weird_ratio(word_scores)

        q_score = compute_quality_score(
            valid_word_ratio=compute_valid_ratio(text_content),
            symbol_ratio=compute_symbol_ratio(text_content),
            perplexity=ppl_val,
            text_length=cc,
            weird_ratio=weird_ratio,
            vowel_ratio=vowel_ratio,
            garbage_density=g_density,
            lang_score=original_lang_score,
            gibberish_ratio=gibb_count / max(wc, 1),
            fused_ratio=fused_words / max(wc, 1),
            rot_ratio=rot_ratio,
        )

        categ, q_score, reason = categorize_line(
            q_score, text_content, wc, vowel_ratio, ppl_val,
            rot_ratio=rot_ratio,
            weird_ratio=weird_ratio,
            return_reason=True
        )

        flags = {
            "allcaps_novowel": reason == "allcaps_novowel",
            "lowppl_clear": reason == "lowppl_clear",
            "cleanprose_clear": reason == "cleanprose_clear",
            "trash_threshold": reason == "trash_threshold",
            "noisy_threshold": reason == "noisy_threshold",
            "clear_threshold": reason == "clear_threshold",
        }

        row = [
            file_id, page_id, line_num, text_content,
            split_ws, split_we, langs[i], f"{scores[i]:.4f}", f"{ppl_val:.2f}",
            wc, cc, f"{g_density:.4f}",
            sym_count, upper_count, rep_count,
            fuse_count, fused_words, gibb_count,
            f"{weird_ratio:.4f}", f"{vowel_ratio:.4f}", f"{rot_ratio:.4f}",
            f"{q_score:.4f}", categ, caps_header,
            flags["allcaps_novowel"], flags["lowppl_clear"], flags["cleanprose_clear"],
            flags["trash_threshold"], flags["noisy_threshold"], flags["clear_threshold"],
            False, False, False
        ]
        results.append(row)

    results.sort(key=lambda x: x[0])
    for doc_id, group in groupby(results, key=lambda x: x[0]):
        write_rows_to_doc(out_dir, doc_id, list(group))


def process_document(task):
    """
    Worker function executed by CPU pool.
    Processes a single document's groups, handles split words, queues to GPU, and saves CSV.
    """
    (file_id, group, text_dir, output_dir, batch_size, task_queue, result_dict,
     expected_langs, trusted_bases, gpu_dead) = task

    try:
        out_path = Path(output_dir) / f"{file_id}.csv"
        if out_path.exists():
            return {
                "status": "skipped",
                "file_id": file_id,
                "lines": 0,
                "reason": "output already exists (resume)",
            }

        batch_lines = []
        batch_meta = []
        processed_count = 0
        batch_counter = 0

        for _, row in group.iterrows():
            page_id = str(row["page"])
            txt_path = Path(text_dir) / file_id / f"{file_id}-{page_id}.txt"
            if not txt_path.exists():
                txt_path = Path(text_dir) / file_id / f"{file_id}_{page_id}.txt"

            if not txt_path.exists():
                continue

            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            expected_incoming_suffix = ""

            for i, line in enumerate(lines, 1):
                merged_text, outgoing_prefix, outgoing_suffix = parse_line_splits(line)
                current_split_ws = outgoing_prefix
                current_split_we = ""

                if expected_incoming_suffix:
                    stripped = merged_text.lstrip()
                    if stripped.startswith(expected_incoming_suffix):
                        merged_text = merged_text.replace(expected_incoming_suffix, "", 1).strip()
                        current_split_we = expected_incoming_suffix

                expected_incoming_suffix = outgoing_suffix
                cat, clean_merged = pre_filter_line(merged_text)

                if cat != "Process":
                    write_rows_to_doc(Path(output_dir), file_id, [[
                        file_id, page_id, i, clean_merged, current_split_ws, current_split_we,
                        "N/A", "0.0000", "0.00", 0, len(clean_merged), "0.0000",
                        0, 0, 0, 0, 0, 0, "0.0000", "0.0000", "0.0000", "0.0000", cat, False,
                        False, False, False, False, False, False, False, False, False
                    ]])
                    continue

                batch_lines.append(clean_merged)
                batch_meta.append((file_id, page_id, i, clean_merged, current_split_ws, current_split_we))
                processed_count += 1

                if len(batch_lines) >= batch_size:
                    b_id = f"{file_id}_{batch_counter}"
                    process_and_write_batch_cpu(b_id, batch_lines, batch_meta, Path(output_dir), task_queue,
                                                result_dict,
                                                expected_langs, trusted_bases, gpu_dead=gpu_dead)
                    batch_lines.clear()
                    batch_meta.clear()
                    batch_counter += 1

        if batch_lines:
            b_id = f"{file_id}_{batch_counter}"
            process_and_write_batch_cpu(b_id, batch_lines, batch_meta, Path(output_dir), task_queue, result_dict,
                                        expected_langs, trusted_bases, gpu_dead=gpu_dead)

        if out_path.exists():
            df = pd.read_csv(out_path, dtype={
                "text": str,
                "split_ws": str,
                "split_we": str,
                "lang": str,
                "categ": str,
            })
            df = df.sort_values(by=["page_num", "line_num"], ascending=True)

            if not df.empty:
                text_modes = df.groupby("text", dropna=False)["categ"].transform(
                    lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
                changed_by_dedup = df["categ"] != text_modes
                df["categ"] = text_modes
                if "pp_dedup" not in df.columns:
                    df["pp_dedup"] = False
                df.loc[changed_by_dedup, "pp_dedup"] = True

                if len(df) >= 5:
                    prev_cat = df["categ"].shift(1)
                    next_cat = df["categ"].shift(-1)
                    prev2_cat = df["categ"].shift(2)
                    next2_cat = df["categ"].shift(-2)

                    surrounded_by_trash = (
                            (prev_cat == "Trash") & (next_cat == "Trash") &
                            (prev2_cat == "Trash") & (next2_cat == "Trash") &
                            (df["categ"] == "Noisy") &
                            (df["quality_score"].astype(float) < CATEG_TRASH_SCORE_MAX + 0.15)
                    )
                    df.loc[surrounded_by_trash, "categ"] = "Trash"
                    if "pp_surrounded_trash" not in df.columns:
                        df["pp_surrounded_trash"] = False
                    df.loc[surrounded_by_trash, "pp_surrounded_trash"] = True

                CZ_DIACS = set("áčďéěíňóřšťůúýžÁČĎÉĚÍŇÓŘŠŤŮÚÝŽ")
                MIN_RUN = 4

                def _has_cz_diacs(text):
                    return any(c in CZ_DIACS for c in str(text))

                if "pp_inverted_run" not in df.columns:
                    df["pp_inverted_run"] = False

                for page_id, page_df in df.groupby("page_num"):
                    candidates = page_df[~page_df["categ"].isin(["Empty", "Non-text"])].copy()
                    if candidates.empty:
                        continue

                    no_diacs = ~candidates["text"].apply(_has_cz_diacs)
                    low_lang = candidates["lang_score"].astype(float) < LANG_SCORE_ROUGH

                    high_rot = candidates["rot_ratio"].astype(float) >= ROT_RATIO_INVERTED_MIN
                    high_ppl = candidates["perplex"].astype(float) >= PPL_INVERTED_MIN

                    suspicious = (no_diacs & low_lang) | (high_rot & high_ppl)

                    run_len = 0
                    run_indices = []
                    for idx, flag in suspicious.items():
                        if flag:
                            run_len += 1
                            run_indices.append(idx)
                        else:
                            if run_len >= MIN_RUN:
                                df.loc[run_indices, "categ"] = "Trash"
                                df.loc[run_indices, "pp_inverted_run"] = True
                            run_len = 0
                            run_indices = []
                    if run_len >= MIN_RUN:
                        df.loc[run_indices, "categ"] = "Trash"
                        df.loc[run_indices, "pp_inverted_run"] = True

            df.to_csv(out_path, index=False)

        return {
            "status": "success",
            "file_id": file_id,
            "lines": processed_count
        }

    except Exception as e:
        return {
            "status": "error",
            "file_id": file_id,
            "reason": str(e)
        }


def main():
    """Initializes queue managers, sets up models, and maps CPU document tasks."""
    config_path = os.getenv("LANGID_CONFIG", "config_langID.txt")
    config = configparser.ConfigParser()
    config.read(config_path)

    INPUT_CSV = config.get("CLASSIFY", "INPUT_CSV")
    # (#4) LANGID_TEXT_DIR (set by run_pipeline for the chosen extraction method)
    # takes precedence over the config default.
    TEXT_DIR = os.getenv("LANGID_TEXT_DIR") or config.get("CLASSIFY", "TEXT_DIR")
    OUTPUT_DIR = config.get("CLASSIFY", "OUTPUT_LINES_LOG")
    BATCH_SIZE = config.getint("CLASSIFY", "BATCH_SIZE")
    WORKERS_MAX = config.getint("CLASSIFY", "WORKERS_MAX", fallback=32)

    MODEL_NAME = config.get("CLASSIFY", "MODEL_NAME", fallback="Qwen/Qwen2.5-0.5B")

    EXPECTED_LANGS_STR = config.get("CLASSIFY", "EXPECTED_LANGS", fallback="ces,deu,eng")
    EXPECTED_LANGS = [lang.strip() for lang in EXPECTED_LANGS_STR.split(",") if lang.strip()]

    TRUSTED_FOREIGN_LANG_BASES = config.get("CLASSIFY", "TRUSTED_FOREIGN_LANGS", fallback="deu,eng,fra,pol,ita")
    _TRUSTED_FOREIGN_LANG_BASES = [lang.strip() for lang in TRUSTED_FOREIGN_LANG_BASES.split(",") if lang.strip()]

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Main] Classifying text from: {TEXT_DIR}")

    df = pd.read_csv(INPUT_CSV)
    sort_cols = (["file", "page", "line_order"] if "line_order" in df.columns else ["file", "page"])
    df = df.sort_values(by=sort_cols)

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_dict = manager.dict()
    gpu_dead = manager.Event()  # (#6) shared liveness signal for the GPU worker

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[Main] Ensuring {MODEL_NAME} is cached...")
    AutoTokenizer.from_pretrained(MODEL_NAME)
    AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto")
    print(f"[Main] Cache OK.")

    gpu_process = mp.Process(target=gpu_inference_worker,
                             args=(task_queue, result_dict, MODEL_NAME, gpu_dead))
    gpu_process.start()

    grouped_tasks = []
    for file_id, group in df.groupby("file"):
        grouped_tasks.append(
            (str(file_id), group, TEXT_DIR, OUTPUT_DIR, BATCH_SIZE, task_queue, result_dict, EXPECTED_LANGS,
             _TRUSTED_FOREIGN_LANG_BASES, gpu_dead))

    logger = ParadataLogger(
        program="langID-classify",
        config={
            "batch_size": BATCH_SIZE,
            "max_workers": WORKERS_MAX,
            "text_dir": TEXT_DIR,
            "output_dir": OUTPUT_DIR,
            "model_name": MODEL_NAME,
        },
        paradata_dir="paradata",
        output_types=["csv"],
    )

    # ── paradata: record the licensed components this step exercises ──────────
    logger.log_component("fasttext")
    _ppl_component = "distilgpt2" if "distilgpt2" in MODEL_NAME.lower() else "qwen2.5_0.5b"
    logger.log_component(_ppl_component)

    max_cores = min(mp.cpu_count(), WORKERS_MAX)
    print(f"Starting {max_cores} CPU Document Processors...")

    total_processed = 0
    total_tasks = len(grouped_tasks)

    try:
        with ProcessPoolExecutor(max_workers=max_cores, initializer=init_cpu_worker) as executor:
            futures = {executor.submit(process_document, task): task[0] for task in grouped_tasks}

            for future in tqdm(as_completed(futures), total=total_tasks, desc="Classifying Documents"):
                file_id = futures[future]

                try:
                    result = future.result()

                    if not isinstance(result, dict):
                        logger.log_skip(file_id, f"Unexpected return type: {type(result).__name__} = {result!r}")
                        tqdm.write(f"Warning: unexpected result for {file_id}: {result!r}")
                        continue

                    status = result.get("status", "error")
                    if status == "success":
                        total_processed += result["lines"]
                        logger.log_success("csv")
                    elif status == "skipped":
                        # (#11) record resume-skips so paradata reflects real work
                        logger.log_skip(result["file_id"],
                                        result.get("reason", "output already exists (resume)"))
                        tqdm.write(f"Skipped (already exists): {result['file_id']}")
                    else:
                        logger.log_skip(result["file_id"], result["reason"])
                        tqdm.write(f"Skipped {result['file_id']}: {result['reason']}")

                except Exception as e:
                    logger.log_skip(file_id, f"Worker crashed unexpectedly: {e}")
                    tqdm.write(f"Error processing {file_id}: {e}")

    finally:
        task_queue.put("STOP")
        gpu_process.join(timeout=30)
        if gpu_process.is_alive():
            gpu_process.terminate()

        logger.finalize(input_total=total_tasks)

    print(f"All done! Processed {total_processed} total lines.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()