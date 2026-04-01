#!/usr/bin/env python3
"""
langID_classify.py

Step 2: Read TXT files → Merge Split Words → Batch classify.

Architecture (Solution B - CPU/GPU Split Queue):
1. ONE dedicated GPU Worker loops continuously, holding the only distilgpt2 instance to prevent VRAM OOM errors.
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

# Import from our refined utility script
from text_util_langID import *
from atrium_paradata import ParadataLogger

# Standardized output headers for the final CSV files
CSV_HEADER = [
    "file", "page_num", "line_num", "text",
    "split_ws", "split_we",
    "lang", "lang_score", "perplex",
    "word_count", "char_count",
    "garbage_density",
    "symbol", "upper", "repeated",
    "ldl_fuses", "gibberish",
    "word_weird",
    "quality_score",
    "categ",
]


# ---------------------------------------------------------------------------
# GPU Worker (Consumer)
# ---------------------------------------------------------------------------
def gpu_inference_worker(task_queue: mp.Queue, result_dict: dict):
    """
    Runs entirely on the GPU/primary device. Holds the ONLY copy of distilgpt2 in memory.
    Consumes batches of text from CPU workers via a Queue, computes Perplexity,
    and returns results via a shared dictionary.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GPU Engine] Initializing DistilGPT2 on {device.upper()}...")

    try:
        # Load the DistilGPT2 tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
        model.eval()
        print(f"[GPU Engine] Ready. Waiting for text batches...")
    except Exception as e:
        print(f"[GPU Engine] Failed to load model: {e}")
        return

    while True:
        try:
            # Pull a batch from the queue. Timeout allows graceful shutdown checks.
            msg = task_queue.get(timeout=1.0)
            if msg == "STOP":
                print("[GPU Engine] Received STOP signal. Shutting down.")
                break

            batch_id, texts = msg

            # Compute perplexity scores for the text batch
            ppls = calculate_perplexity_batch(texts, model, tokenizer, device)

            # Post results back to the shared memory dictionary using the batch_id as the key
            result_dict[batch_id] = ppls

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[GPU Engine Error] Processing batch: {e}")
            # Ensure CPU workers don't hang infinitely if a batch fails by returning zeros
            if 'msg' in locals() and msg != "STOP":
                result_dict[msg[0]] = [0.0] * len(msg[1])


# ---------------------------------------------------------------------------
# CPU Worker Pool (Producers)
# ---------------------------------------------------------------------------
worker_models = {}


def init_cpu_worker():
    """
    Initializes lightweight CPU models (FastText) once per CPU core/worker.
    Since FastText is purely CPU-bound, loading it per-worker scales linearly and safely.
    """
    import fasttext
    worker_models['ft'] = fasttext.load_model("lid.176.bin")


def write_rows_to_doc(output_dir: Path, file_id: str, rows: list):
    """
    Appends classified rows to the respective document's CSV log.
    """
    out_path = output_dir / f"{file_id}.csv"
    file_exists = out_path.exists()

    with open(out_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def process_and_write_batch_cpu(batch_id: str, lines: list[str], meta: list[tuple], out_dir: Path,
                                task_queue: mp.Queue, result_dict: dict, expected_langs: list[str] = None):
    """
    Core CPU task logic. Submits heavy text sequences to the GPU, processes
    Regex/FastText concurrently while waiting, extracts structure data, and finalizes categorization.
    """
    ft = worker_models['ft']

    # 1. Dispatch heavy text sequences to the GPU Engine immediately
    task_queue.put((batch_id, lines))

    # 2. Concurrently run Language ID (FastText) on the CPU while waiting for the GPU
    lines_lower = [line.lower() for line in lines]
    labels, scores = ft.predict(lines_lower, k=1)
    langs = [l[0].replace("__label__", "") for l in labels]
    scores = [s[0] for s in scores]

    # 3. Poll the shared dictionary until the GPU engine finishes this specific batch
    while batch_id not in result_dict:
        time.sleep(0.01)

    # Extract and clean up the result to free shared memory
    ppls = result_dict.pop(batch_id)

    # 4. Finalize Structural Extraction & Apply Refined Categorization Logic
    results = []
    for i in range(len(lines)):
        file_id, page_id, line_num, text_content, split_ws, split_we = meta[i]

        ppl_val = ppls[i]
        lang = langs[i]
        score = scores[i]

        # --- Extracted Metrics (For CSV Logging) ---
        wc = len(text_content.split())
        cc = len(text_content)
        g_density = compute_garbage_density(text_content)

        sym_count = detect_strange_symbols(text_content)
        upper_count = detect_mid_uppercase(text_content)
        rep_count = detect_repeated_chars(text_content)
        fuse_count = detect_letter_digit_letter(text_content)
        gibb_count = detect_gibberish_words(text_content)

        word_scores = score_words_in_line(text_content)
        weird_ratio = compute_word_weird_ratio(word_scores)

        q_score = compute_quality_score(
            valid_word_ratio=compute_valid_ratio(text_content),
            symbol_ratio=compute_symbol_ratio(text_content),
            perplexity=ppl_val,
            text_length=cc,
        )

        # Apply the refined, unified penalty-based categorization passing down the dynamic language allowlist
        categ = categorize_line(ppl_val, text_content, lang, score, expected_langs)

        # Construct final output row matching CSV_HEADER
        row = [
            file_id, page_id, line_num, text_content,
            split_ws, split_we, lang, f"{score:.4f}", f"{ppl_val:.2f}",
            wc, cc, f"{g_density:.4f}",
            sym_count, upper_count, rep_count,
            fuse_count, gibb_count,
            f"{weird_ratio:.4f}", f"{q_score:.4f}", categ,
        ]
        results.append(row)

    # Sort results by document ID to group them safely before writing
    results.sort(key=lambda x: x[0])
    for doc_id, group in groupby(results, key=lambda x: x[0]):
        write_rows_to_doc(out_dir, doc_id, list(group))


def process_document(args) -> int:
    """
    Task mapped to the ProcessPool. Reads a specific document's pages,
    repairs split words, and batches the lines for ML processing.
    """
    file_id, file_rows, text_dir, out_dir, batch_size, task_queue, result_dict, expected_langs = args

    # Check if this document has already been processed to allow crash-recovery
    out_path = Path(out_dir) / f"{file_id}.csv"
    if out_path.exists():
        return 0

    batch_lines = []
    batch_meta = []
    processed_count = 0
    batch_counter = 0

    # Iterate over pages associated with this document ID
    for _, row in file_rows.iterrows():
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

            # Re-attach fragments split across page/line breaks
            if expected_incoming_suffix:
                stripped = merged_text.lstrip()
                if stripped.startswith(expected_incoming_suffix):
                    merged_text = merged_text.replace(expected_incoming_suffix, "", 1).strip()
                    current_split_we = expected_incoming_suffix

            expected_incoming_suffix = outgoing_suffix
            cat, clean_merged = pre_filter_line(merged_text)

            # Bypassing ML ops if entirely non-text
            if cat != "Process":
                write_rows_to_doc(Path(out_dir), file_id, [[
                    file_id, page_id, i, clean_merged, current_split_ws, current_split_we,
                    "N/A", 0, 0, 0, 0, "0.0000", 0, 0, 0, 0, 0, "0.0000", "0.0000", cat
                ]])
                continue

            batch_lines.append(clean_merged)
            batch_meta.append((file_id, page_id, i, clean_merged, current_split_ws, current_split_we))
            processed_count += 1

            if len(batch_lines) >= batch_size:
                b_id = f"{file_id}_{batch_counter}"
                process_and_write_batch_cpu(b_id, batch_lines, batch_meta, Path(out_dir), task_queue, result_dict,
                                            expected_langs)
                batch_lines.clear()
                batch_meta.clear()
                batch_counter += 1

    # Flush remaining trailing lines
    if batch_lines:
        b_id = f"{file_id}_{batch_counter}"
        process_and_write_batch_cpu(b_id, batch_lines, batch_meta, Path(out_dir), task_queue, result_dict,
                                    expected_langs)

    # Sort
    if out_path.exists():
        df = pd.read_csv(out_path)
        df = df.sort_values(by=["page_num", "line_num"], ascending=True)
        df.to_csv(out_path, index=False)

    return processed_count


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def main():
    """
    Entry point. Reads the config, sets up multiprocessing queues, starts the GPU engine,
    and maps the documents to the CPU Worker Pool.
    """
    config = configparser.ConfigParser()
    config.read("config_langID.txt")

    INPUT_CSV = config.get("CLASSIFY", "INPUT_CSV")
    TEXT_DIR = config.get("CLASSIFY", "TEXT_DIR")
    OUTPUT_DIR = config.get("CLASSIFY", "OUTPUT_LINES_LOG")
    BATCH_SIZE = config.getint("CLASSIFY", "BATCH_SIZE")
    WORKERS_MAX = config.getint("CLASSIFY", "WORKERS_MAX", fallback=32)

    # Pull expected languages configuration, falling back to 'ces,deu,eng'
    EXPECTED_LANGS_STR = config.get("CLASSIFY", "EXPECTED_LANGS", fallback="ces,deu,eng")
    EXPECTED_LANGS = [lang.strip() for lang in EXPECTED_LANGS_STR.split(",") if lang.strip()]

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    sort_cols = (["file", "page", "line_order"] if "line_order" in df.columns else ["file", "page"])
    df = df.sort_values(by=sort_cols)

    # 1. Set up Inter-Process Shared Memory
    manager = mp.Manager()
    task_queue = manager.Queue()
    result_dict = manager.dict()

    # 2. Spin up the dedicated GPU Engine
    gpu_process = mp.Process(target=gpu_inference_worker, args=(task_queue, result_dict))
    gpu_process.start()

    # 3. Create tasks mapped with the Proxies AND the custom language config
    grouped_tasks = []
    for file_id, group in df.groupby("file"):
        grouped_tasks.append(
            (str(file_id), group, TEXT_DIR, OUTPUT_DIR, BATCH_SIZE, task_queue, result_dict, EXPECTED_LANGS))

    # CPU Cores handle standard multiprocessing. Because DistilGPT2 is out of the picture,
    # we can safely max out the CPU cores to WORKERS_MAX without triggering an OOM.
    max_cores = min(mp.cpu_count(), WORKERS_MAX)
    print(f"Starting {max_cores} CPU Document Processors...")

    total_processed = 0
    with ProcessPoolExecutor(max_workers=max_cores, initializer=init_cpu_worker) as executor:
        futures = {executor.submit(process_document, task): task[0] for task in grouped_tasks}

        for count, future in enumerate(as_completed(futures), 1):
            file_id = futures[future]
            try:
                lines_proc = future.result()
                total_processed += lines_proc
                print(f"[{count}/{len(grouped_tasks)}] Finished {file_id}")
            except Exception as e:
                print(f"Error processing {file_id}: {e}")

    # 4. Graceful Shutdown
    print("All documents processed. Shutting down GPU Engine...")
    task_queue.put("STOP")
    gpu_process.join()
    print("Pipeline Complete.")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()