#!/usr/bin/env python3
"""
langID_classify.py

Step 2: Read TXT files → Merge Split Words → Batch classify.

Architecture (Solution B - CPU/GPU Split Queue):
1. ONE dedicated GPU Worker loops continuously, holding the only Qwen2.5-0.5B instance to prevent VRAM OOM errors.
2. MULTIPLE CPU Workers read files, run Regex/FastText, and place texts into a multiprocessing Task Queue.
3. CPU Workers poll a Result Dictionary until the GPU returns their Perplexity scores.
"""

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
from atrium_paradata import ParadataLogger

CSV_HEADER = [
    "file", "page_num", "line_num", "text",
    "split_ws", "split_we",
    "lang", "lang_score", "perplex",
    "word_count", "char_count",
    "garbage_density",
    "symbol", "upper", "repeated",
    "ldl_fuses", "gibberish",
    "word_weird", "vowel_ratio",
    "quality_score",
    "categ", "caps_header"
]


def gpu_inference_worker(task_queue: mp.Queue, result_dict: dict, model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPU Engine] Initializing {model_name} on {device.upper()}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Safely assign pad_token if it doesn't exist (needed for distilgpt2, etc.)
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
    import fasttext
    worker_models['ft'] = fasttext.load_model("lid.176.bin")


def write_rows_to_doc(output_dir: Path, file_id: str, rows: list):
    out_path = output_dir / f"{file_id}.csv"
    file_exists = out_path.exists()

    with open(out_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def process_and_write_batch_cpu(batch_id: str, lines: list[str], meta: list[tuple], out_dir: Path,
                                task_queue: mp.Queue, result_dict: dict, expected_langs: list[str] = None, trusted_langs: list[str] = None):
    ft = worker_models['ft']

    task_queue.put((batch_id, lines))

    lines_lower = [line.lower() for line in lines]
    labels, scores = ft.predict(lines_lower, k=1)
    langs = [l[0].replace("__label__", "") for l in labels]
    scores = [s[0] for s in scores]

    while batch_id not in result_dict:
        time.sleep(0.01)

    ppls = result_dict.pop(batch_id)

    results = []
    for i in range(len(lines)):
        file_id, page_id, line_num, text_content, split_ws, split_we = meta[i]

        # Force language remapping and fix the score
        if langs[i] not in trusted_langs + expected_langs:
            langs[i] = expected_langs[0]  # ces
            scores[i] = max(scores[i], LANG_SCORE_CLEAR)

        ppl_val = ppls[i]

        wc = len(text_content.split())
        cc = len(text_content)
        g_density = compute_garbage_density(text_content)

        sym_count = detect_strange_symbols(text_content)
        upper_count = detect_mid_uppercase(text_content)
        rep_count = detect_repeated_chars(text_content)
        fuse_count = detect_letter_digit_letter(text_content)
        gibb_count = detect_gibberish_words(text_content)

        vowel_ratio = compute_vowel_ratio(text_content)
        caps_header = is_all_caps_line(text_content)

        word_scores = score_words_in_line(text_content)
        weird_ratio = compute_word_weird_ratio(word_scores)

        q_score = compute_quality_score(
            valid_word_ratio=compute_valid_ratio(text_content),
            symbol_ratio=compute_symbol_ratio(text_content),
            perplexity=ppl_val,
            text_length=cc,
            weird_ratio=weird_ratio
        )

        categ = categorize_line(q_score, text_content, wc, weird_ratio, vowel_ratio, ppl_val)

        row = [
            file_id, page_id, line_num, text_content,
            split_ws, split_we, langs[i], f"{scores[i]:.4f}", f"{ppl_val:.2f}",
            wc, cc, f"{g_density:.4f}",
            sym_count, upper_count, rep_count,
            fuse_count, gibb_count,
            f"{weird_ratio:.4f}", f"{vowel_ratio:.4f}",
            f"{q_score:.4f}", categ, caps_header
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
    file_id, group, text_dir, output_dir, batch_size, task_queue, result_dict, expected_langs, trusted_bases = task

    try:
        out_path = Path(output_dir) / f"{file_id}.csv"
        if out_path.exists():
            return {
                "status": "skipped",
                "file_id": file_id,
                "lines": 0,
                "reason": "output already exists",
            }

        batch_lines = []
        batch_meta = []
        processed_count = 0
        batch_counter = 0

        for _, row in group.iterrows():
            page_id = str(row["page"])
            txt_path = Path(text_dir) / file_id / f"{file_id}-{page_id}.txt"

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
                    # Preserve the line number (i) strictly so the output CSV matches
                    # input ALTO lines 1-to-1.  String-formatted placeholders prevent
                    # pandas dtype drift when the CSV is later read back.
                    write_rows_to_doc(Path(output_dir), file_id, [[
                        file_id, page_id, i, clean_merged, current_split_ws, current_split_we,
                        "N/A", "0.0000", "0.00", 0, len(clean_merged), "0.0000",
                        0, 0, 0, 0, 0, "0.0000", "0.0000", "0.0000", cat, False
                    ]])
                    continue

                batch_lines.append(clean_merged)
                batch_meta.append((file_id, page_id, i, clean_merged, current_split_ws, current_split_we))
                processed_count += 1

                if len(batch_lines) >= batch_size:
                    b_id = f"{file_id}_{batch_counter}"
                    process_and_write_batch_cpu(b_id, batch_lines, batch_meta, Path(output_dir), task_queue, result_dict,
                                                expected_langs, trusted_bases)
                    batch_lines.clear()
                    batch_meta.clear()
                    batch_counter += 1

        if batch_lines:
            b_id = f"{file_id}_{batch_counter}"
            process_and_write_batch_cpu(b_id, batch_lines, batch_meta, Path(output_dir), task_queue, result_dict,
                                        expected_langs, trusted_bases)

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
                # Force Pandas to include empty lines in the groupby by specifying dropna=False
                text_modes = df.groupby("text", dropna=False)["categ"].transform(
                    lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
                df["categ"] = text_modes

                if len(df) >= 3:
                    prev_cat = df["categ"].shift(1)
                    next_cat = df["categ"].shift(-1)

                    surrounded_by_trash = (prev_cat == "Trash") & (next_cat == "Trash") & (df["categ"] == "Noisy")
                    df.loc[surrounded_by_trash, "categ"] = "Trash"

            df.to_csv(out_path, index=False)

        # Output successful processing metrics to the main process
        return {
            "status": "success",
            "file_id": file_id,
            "lines": processed_count
        }

    except Exception as e:
        # Catch errors so the future doesn't just crash,
        # but safely returns the failure reason back to the main thread.
        return {
            "status": "error",
            "file_id": file_id,
            "reason": str(e)
        }


def main():
    config = configparser.ConfigParser()
    config.read("config_langID.txt")

    INPUT_CSV = config.get("CLASSIFY", "INPUT_CSV")
    TEXT_DIR = config.get("CLASSIFY", "TEXT_DIR")
    OUTPUT_DIR = config.get("CLASSIFY", "OUTPUT_LINES_LOG")
    BATCH_SIZE = config.getint("CLASSIFY", "BATCH_SIZE")
    WORKERS_MAX = config.getint("CLASSIFY", "WORKERS_MAX", fallback=32)

    # Read the model name, falling back to Qwen if not found
    MODEL_NAME = config.get("CLASSIFY", "MODEL_NAME", fallback="Qwen/Qwen2.5-0.5B")

    EXPECTED_LANGS_STR = config.get("CLASSIFY", "EXPECTED_LANGS", fallback="ces,deu,eng")
    EXPECTED_LANGS = [lang.strip() for lang in EXPECTED_LANGS_STR.split(",") if lang.strip()]

    TRUSTED_FOREIGN_LANG_BASES = config.get("CLASSIFY", "TRUSTED_FOREIGN_LANGS", fallback="deu,eng,fra,pol,ita")
    _TRUSTED_FOREIGN_LANG_BASES = [lang.strip() for lang in TRUSTED_FOREIGN_LANG_BASES.split(",") if lang.strip()]

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    sort_cols = (["file", "page", "line_order"] if "line_order" in df.columns else ["file", "page"])
    df = df.sort_values(by=sort_cols)

    manager = mp.Manager()
    task_queue = manager.Queue()
    result_dict = manager.dict()

    # Pass the MODEL_NAME into the worker args
    gpu_process = mp.Process(target=gpu_inference_worker, args=(task_queue, result_dict, MODEL_NAME))
    gpu_process.start()

    # ... (Rest of your main function remains completely unchanged)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()