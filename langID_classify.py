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

Output columns (#3): rows are emitted dict-keyed against CSV_HEADER so the scored
and fast-track (Empty/Non-text) writers can never drift out of column alignment.
`categ`/`quality_score` lead the file; `original_text`, `original_lang`,
`orig_lang_score` and `weird_wx` are recorded alongside the cleaned/remapped
values. The page aggregator reads strictly by column NAME, so this reorder is
safe; every aggregate-consumed name is retained verbatim.
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
# language-remapping logic below) must be imported explicitly. has_cz_diacs is
# imported explicitly too so the extracted page-postprocess helper can reference
# it by name (#3 A3).
from text_util_langID import _lang_base, has_cz_diacs, TRASH_REASONS
from atrium_paradata import ParadataLogger

# Hard ceiling on how long a CPU worker waits for a batch's perplexity before
# declaring the GPU worker unresponsive (#6). Generous so legitimate large
# batches are never killed, but finite so a crash cannot hang the pipeline.
# GPU_WAIT_TIMEOUT = 600.0  # seconds

# (#3) Single source of truth for the per-line CSV schema. Rows are built as a
# dict keyed by these names and emitted in this exact order, so the scored path
# and the fast-track Empty/Non-text path cannot drift. `categ`/`quality_score`
# lead; the aggregator reads by NAME so column ORDER is free to change.
CSV_HEADER = [
    "categ", "quality_score",
    "file", "page_num", "line_num",
    "text", "original_text",
    "split_ws", "split_we",
    "lang", "lang_score", "original_lang", "orig_lang_score",
    "perplex",
    "word_count", "char_count",
    "garbage_density",
    "upper", "repeated",
    "ldl_fuses", "fused_words", "gibberish", "weird_wx",
    "word_weird", "vowel_ratio", "rot_ratio",
    "caps_header",
    "allcaps_novowel", "lowppl_clear", "cleanprose_clear",
    "trash_threshold", "noisy_threshold", "clear_threshold",
    "pp_dedup", "pp_surrounded_trash", "pp_inverted_run", "pp_page_context",
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


def _row_from_dict(d: dict) -> list:
    """Emit a CSV row in CSV_HEADER order from a column-keyed dict (#3)."""
    return [d[c] for c in CSV_HEADER]


def process_and_write_batch_cpu(batch_id: str, lines: list, meta: list, out_dir: Path,
                                task_queue: mp.Queue, result_dict: dict, expected_langs: list = None,
                                trusted_langs: list = None, gpu_dead=None, gpu_time_out=600.0):
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
        if waited >= gpu_time_out:
            raise RuntimeError(
                f"Timed out after {gpu_time_out:.0f}s waiting for perplexity "
                f"of batch {batch_id}; GPU worker unresponsive."
            )

    ppls = result_dict.pop(batch_id)

    results = []
    for i in range(len(lines)):
        # (#3) meta now carries the pre-repair text as a 7th element so the
        # original line can be recorded and fed to the density/vowel signals.
        file_id, page_id, line_num, text_content, split_ws, split_we, original_text = meta[i]

        # Pre-remap FastText prediction — stored verbatim and (the score) fed to QS.
        original_lang = langs[i]
        original_lang_score = scores[i]

        wc = len(text_content.split())
        cc = len(text_content)

        # (#15, #3) Language remap via the pure helper: slk keeps its score, every
        # other unknown base is relabelled to the collection default and its score
        # CAPPED at LANG_SCORE_REMAP / LANG_SCORE_REMAP_FAR. The stored lang/
        # lang_score reflect the remap; the ORIGINAL score still drives the
        # QS_WEIGHT_LANG component.
        langs[i], scores[i] = remap_lang(
            langs[i], scores[i], _known_lang_bases, expected_langs[0]
        )

        ppl_val = ppls[i]

        if wc <= 2 and ppl_val > SHORT_PPL_CAP:
            ppl_val = SHORT_PPL_CAP

        # (#5/#11/#8) garbage density and vowel ratio are computed on the ORIGINAL
        # (pre-repair) line so cleaning never hides noise; char_count and the QS
        # length signal stay on the cleaned text_content.
        g_density = compute_garbage_density(original_text)
        vowel_ratio = compute_vowel_ratio(original_text)

        # sym_count = detect_strange_symbols(text_content)
        upper_count = detect_mid_uppercase(text_content)
        rep_count = detect_repeated_chars(text_content)
        fuse_count = detect_letter_digit_letter(text_content)
        fused_words = detect_fused_words(text_content)
        gibb_count = detect_gibberish_words(text_content)
        wx_count = detect_wx_words(text_content)  # (#13) standalone weird_wx column

        rot_ratio = compute_rotatable_ratio(text_content)

        is_upright_czech, ghost_dominated = analyze_rotation_signals(text_content)
        caps_header = is_all_caps_line(text_content)
        word_scores = score_words_in_line(text_content)
        weird_ratio = compute_word_weird_ratio(word_scores)
        valid_ratio = compute_valid_ratio(text_content)

        # Two-tier Trust System over flat remapping
        base_lang = _lang_base(original_lang)
        if base_lang in _known_lang_bases:
            if base_lang in expected_langs:
                trust_lang_score = original_lang_score
            else:
                trust_lang_score = original_lang_score * 0.85
        else:
            trust_lang_score = original_lang_score * 0.50

        q_score = compute_quality_score(
            valid_word_ratio=valid_ratio,
            perplexity=ppl_val,
            text_length=cc,
            weird_ratio=weird_ratio,
            vowel_ratio=vowel_ratio,
            garbage_density=g_density,
            lang_score=trust_lang_score,  # Feed trust-tier into QS natively
            gibberish_ratio=(gibb_count + wx_count) / max(wc, 1),
            fused_ratio=fused_words / max(wc, 1),
            is_upright_czech=is_upright_czech,
        )

        categ, q_score, reason = categorize_line(
            q_score, text_content, wc, vowel_ratio, ppl_val,
            weird_ratio=weird_ratio,
            return_reason=True,
            valid_word_ratio=valid_ratio,
            lang_score=trust_lang_score,  # Structural guard uses tier trust
            orig_lang_score=original_lang_score,
            gibberish_present=(gibb_count + wx_count) > 0,
            garbage_density=g_density,
            is_upright_czech=is_upright_czech,
            ghost_dominated=ghost_dominated,
        )

        row_dict = {
            "categ": categ,
            "quality_score": f"{q_score:.4f}",
            "file": file_id,
            "page_num": page_id,
            "line_num": line_num,
            "text": text_content,
            "original_text": original_text,
            "split_ws": split_ws,
            "split_we": split_we,
            "lang": langs[i],
            "lang_score": f"{scores[i]:.4f}",
            "original_lang": original_lang,
            "orig_lang_score": f"{original_lang_score:.4f}",
            "perplex": f"{ppl_val:.2f}",
            "word_count": wc,
            "char_count": cc,
            "garbage_density": f"{g_density:.4f}",
            # "symbol": sym_count,
            "upper": upper_count,
            "repeated": rep_count,
            "ldl_fuses": fuse_count,
            "fused_words": fused_words,
            "gibberish": gibb_count,
            "weird_wx": wx_count,
            "word_weird": f"{weird_ratio:.4f}",
            "vowel_ratio": f"{vowel_ratio:.4f}",
            "rot_ratio": f"{rot_ratio:.4f}",
            "caps_header": caps_header,
            "allcaps_novowel": reason == "allcaps_novowel",
            "lowppl_clear": reason == "lowppl_clear",
            "cleanprose_clear": reason == "cleanprose_clear",
            "trash_threshold": reason in TRASH_REASONS,
            "noisy_threshold": reason == "noisy_threshold",
            "clear_threshold": reason == "clear_threshold",
            "pp_dedup": False,
            "pp_surrounded_trash": False,
            "pp_inverted_run": False,
        }
        results.append(_row_from_dict(row_dict))

    # Column 2 of CSV_HEADER is "file"; group by it to write per-document.
    _file_idx = CSV_HEADER.index("file")
    results.sort(key=lambda x: x[_file_idx])
    for doc_id, group in groupby(results, key=lambda x: x[_file_idx]):
        write_rows_to_doc(out_dir, doc_id, list(group))


def _fast_track_row(file_id, page_id, line_num, clean_text, original_text,
                    split_ws, split_we, categ) -> list:
    """Build a dict-keyed CSV row for a pre-filtered Empty/Non-text line (#3)."""
    d = {
        "categ": categ,
        "quality_score": "0.0000",
        "file": file_id,
        "page_num": page_id,
        "line_num": line_num,
        "text": clean_text,
        "original_text": original_text,
        "split_ws": split_ws,
        "split_we": split_we,
        "lang": "N/A",
        "lang_score": "0.0000",
        "original_lang": "N/A",
        "orig_lang_score": "0.0000",
        "perplex": "0.00",
        "word_count": 0,
        "char_count": len(clean_text),
        "garbage_density": "0.0000",
        # "symbol": 0,
        "upper": 0,
        "repeated": 0,
        "ldl_fuses": 0,
        "fused_words": 0,
        "gibberish": 0,
        "weird_wx": 0,
        "word_weird": "0.0000",
        "vowel_ratio": "0.0000",
        "rot_ratio": "0.0000",
        "caps_header": False,
        "allcaps_novowel": False,
        "lowppl_clear": False,
        "cleanprose_clear": False,
        "trash_threshold": False,
        "noisy_threshold": False,
        "clear_threshold": False,
        "pp_dedup": False,
        "pp_surrounded_trash": False,
        "pp_inverted_run": False,
        "pp_page_context": False,
    }
    return _row_from_dict(d)


def apply_document_postprocessing(df: "pd.DataFrame") -> "pd.DataFrame":
    """Document-level smoothing — pure pandas, GPU-free, idempotent-safe.

    Runs the three post-passes in order on a per-document line dataframe and
    sets the matching ``pp_*`` flags:

      1. ``pp_dedup``            — harmonise identical text to its modal category.
      2. ``pp_surrounded_trash`` — a low-scoring Noisy island fully enclosed by a
                                   4-line Trash window is downgraded to Trash.
      3. ``pp_inverted_run``     — page-level inverted-scan sweep. A line is
         "suspicious" when (no Czech diacritics AND stored ``lang_score`` <
         ``LANG_SCORE_ROUGH``) OR (``rot_ratio`` >= ``ROT_RATIO_INVERTED_MIN``
         AND ``perplex`` >= ``PPL_INVERTED_MIN`` AND ``word_weird`` > 0.0 AND
         ``lang_score`` < ``ROT_HIGH_LANG_CONF``). Suspicious lines are Trashed
         when they form a contiguous run >= ``INVERTED_RUN_MIN`` *or* (#3 A3
         page-MAJORITY arm) when they make up >= ``INVERTED_PAGE_MAJORITY`` of
         the page's scoreable lines — catching garbage that is broken up by
         Empty / Non-text / short fragments and so never forms a long run.

    Extracted out of ``process_document`` (#3 A3) so the Step-5 re-scorer
    (``tools/recategorize_from_csv.py``) reuses byte-identical logic: zero drift
    between production output and offline re-measurement.
    """
    if df.empty:
        return df

    df = df.sort_values(by=["page_num", "line_num"], ascending=True).copy()

    # 1. header/footer dedup -> modal category
    text_modes = df.groupby("text", dropna=False)["categ"].transform(
        lambda x: x.mode()[0] if not x.mode().empty else x.iloc[0])
    changed_by_dedup = df["categ"] != text_modes
    df["categ"] = text_modes
    if "pp_dedup" not in df.columns:
        df["pp_dedup"] = False
    df.loc[changed_by_dedup, "pp_dedup"] = True

    # 2. rolling-window surrounded-Trash smoothing
    if "pp_surrounded_trash" not in df.columns:
        df["pp_surrounded_trash"] = False
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
        df.loc[surrounded_by_trash, "pp_surrounded_trash"] = True

    # 3. page-level inverted-scan sweep (run-based + page-majority)
    if "pp_inverted_run" not in df.columns:
        df["pp_inverted_run"] = False

    if "pp_page_context" not in df.columns:
        df["pp_page_context"] = False

    for _page_id, page_df in df.groupby("page_num"):
        scoreable_idx = page_df[~page_df["categ"].isin(["Empty", "Non-text"])].index
        if len(scoreable_idx) == 0:
            continue

        page_scoreable = df.loc[scoreable_idx]
        median_qs = page_scoreable["quality_score"].astype(float).median()
        clear_ratio = (page_scoreable["categ"] == "Clear").mean()

        # Determine ratio of document leaning towards trusted languages based strictly on origin
        trusted_bases = set(["ces", "deu", "eng", "fra", "pol", "ita", "slk"])
        is_trusted = page_scoreable["original_lang"].apply(lambda x: str(x).split("_")[0] in trusted_bases)
        decent_lang_ratio = is_trusted.mean()

        # Symmetric Rule 1: Page is heavily garbage (Pull borderline Noisy down)
        if clear_ratio <= 0.05 and decent_lang_ratio < 0.50 and median_qs < 0.55:
            sus_idx = page_scoreable[
                (page_scoreable["categ"] == "Noisy") & (page_scoreable["quality_score"].astype(float) < 0.80)].index
            if len(sus_idx) > 0:
                df.loc[sus_idx, "categ"] = "Trash"
                df.loc[sus_idx, "pp_page_context"] = True

        # Symmetric Rule 2: Page is predominantly clean (Promote edge-case recoverable Trash)
        elif clear_ratio > 0.60 and median_qs > 0.80:
            sus_idx = page_scoreable[(page_scoreable["categ"] == "Trash") & (
                        page_scoreable["quality_score"].astype(float) >= 0.45) & is_trusted].index
            if len(sus_idx) > 0:
                df.loc[sus_idx, "categ"] = "Noisy"
                df.loc[sus_idx, "pp_page_context"] = True

    for _page_id, page_df in df.groupby("page_num"):
        candidates = page_df[~page_df["categ"].isin(["Empty", "Non-text"])].copy()
        if candidates.empty:
            continue

        no_diacs = ~candidates["text"].apply(has_cz_diacs)
        low_lang = candidates["lang_score"].astype(float) < LANG_SCORE_ROUGH
        high_rot = candidates["rot_ratio"].astype(float) >= ROT_RATIO_INVERTED_MIN
        high_ppl = candidates["perplex"].astype(float) >= PPL_INVERTED_MIN
        has_weird = candidates["word_weird"].astype(float) > 0.0
        high_lang_conf = candidates["lang_score"].astype(float) >= ROT_HIGH_LANG_CONF

        suspicious = (
                (no_diacs & low_lang)
                | (high_ppl & has_weird & ~high_lang_conf)
                | (no_diacs & high_rot & high_ppl & ~high_lang_conf)  # (#3 Problem 1) rot arm
        )

        # (#3 A3) Page-MAJORITY arm: a page that is mostly suspicious is an
        # inverted/garbage scan; Trash every suspicious line regardless of run
        # length. Catches garbage broken up by Empty/Non-text/short lines.
        if len(candidates) > 0 and (suspicious.sum() / len(candidates)) >= INVERTED_PAGE_MAJORITY:
            idx = suspicious[suspicious].index
            df.loc[idx, "categ"] = "Trash"
            df.loc[idx, "pp_inverted_run"] = True
            continue



        # Otherwise fall back to the contiguous-run rule for mixed pages.
        run_len = 0
        run_indices = []
        for idx, flag in suspicious.items():
            if flag:
                run_len += 1
                run_indices.append(idx)
            else:
                if run_len >= INVERTED_RUN_MIN:
                    df.loc[run_indices, "categ"] = "Trash"
                    df.loc[run_indices, "pp_inverted_run"] = True
                run_len = 0
                run_indices = []
        if run_len >= INVERTED_RUN_MIN:
            df.loc[run_indices, "categ"] = "Trash"
            df.loc[run_indices, "pp_inverted_run"] = True

    return df


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
                # merged_text is the pre-repair line (post split-merge); clean_merged
                # is what pre_filter_line cleaned. We keep BOTH: original_text records
                # merged_text, text records clean_merged (#3).
                cat, clean_merged = pre_filter_line(merged_text)

                if cat != "Process":
                    write_rows_to_doc(
                        Path(output_dir), file_id,
                        [_fast_track_row(file_id, page_id, i, clean_merged, merged_text,
                                         current_split_ws, current_split_we, cat)]
                    )
                    continue

                batch_lines.append(clean_merged)
                batch_meta.append(
                    (file_id, page_id, i, clean_merged, current_split_ws, current_split_we, merged_text)
                )
                processed_count += 1

                if len(batch_lines) >= batch_size:
                    b_id = f"{file_id}_{batch_counter}"
                    process_and_write_batch_cpu(b_id, batch_lines, batch_meta, Path(output_dir), task_queue,
                                                result_dict,
                                                expected_langs, trusted_bases, gpu_dead=gpu_dead, gpu_time_out=GPU_WAIT_TIMEOUT)
                    batch_lines.clear()
                    batch_meta.clear()
                    batch_counter += 1

        if batch_lines:
            b_id = f"{file_id}_{batch_counter}"
            process_and_write_batch_cpu(b_id, batch_lines, batch_meta, Path(output_dir), task_queue, result_dict,
                                        expected_langs, trusted_bases, gpu_dead=gpu_dead, gpu_time_out=GPU_WAIT_TIMEOUT)

        if out_path.exists():
            df = pd.read_csv(out_path, dtype={
                "text": str,
                "original_text": str,
                "split_ws": str,
                "split_we": str,
                "lang": str,
                "original_lang": str,
                "categ": str,
            })

            if not df.empty:
                df = apply_document_postprocessing(df)

            # (#3.7) UTF-8 keeps Czech diacritics intact on the finalised file.
            df = df[CSV_HEADER]  # guard: write columns in canonical order
            df.to_csv(out_path, index=False, encoding='utf-8')

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
    GPU_WAIT_TIMEOUT = config.getint("CLASSIFY", "GPU_WAIT_TIMEOUT", fallback=600.0)

    MODEL_NAME = config.get("CLASSIFY", "MODEL_NAME", fallback="Qwen/Qwen2.5-0.5B")

    # [NEW] Guard against scale mismatch
    if "qwen" in MODEL_NAME.lower() and PERPLEXITY_THRESHOLD_MAX > 500.0:
        print(f"\n[WARNING] Configuration Mismatch: You are using '{MODEL_NAME}'.")
        print(f"          PERPLEXITY_THRESHOLD_MAX is set to {PERPLEXITY_THRESHOLD_MAX}.")
        print("          Qwen generally produces much lower perplexities than DistilGPT2.")
        print("          Consider lowering the threshold to avoid false 'Clear' categorizations.\n")

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