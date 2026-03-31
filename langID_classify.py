#!/usr/bin/env python3
"""
langID_classify.py

Step 2: Read TXT files → Merge Split Words → Batch classify.

Architecture:
This script uses concurrent.futures.ProcessPoolExecutor to distribute documents
across multiple CPU cores.
Because `fasttext` and `distilgpt2` models consume significant memory, they are
initialized ONCE per worker instance using the `init_worker()` function and stored
in the worker's local global dictionary `worker_models`.
"""

import pandas as pd
import torch
import fasttext
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import csv
import sys
from itertools import groupby
import configparser
from text_util_langID import *
from atrium_paradata import ParadataLogger
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

CSV_HEADER = [
    "file", "page_num", "line_num", "text",
    "split_ws", "split_we",
    "lang", "lang_score", "perplex",
    "symbol", "upper",
    "word_weird",
    "quality_score",
    "categ",
]

# Global dictionary to hold models for the specific CPU worker process
worker_models = {}


def init_worker():
    """
    Worker Initializer.
    Called once when a new CPU process spins up. Loads the Heavy ML models into RAM/VRAM
    so they don't have to be passed repeatedly through memory pipelines between processes.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load FastText for Language ID
    ft = fasttext.load_model("lid.176.bin")

    # 2. Load DistilGPT2 for Perplexity evaluation
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(device)
    model.eval()  # Set model to evaluation mode (no gradients)

    # Cache in the worker's unique memory space
    worker_models['ft'] = ft
    worker_models['tokenizer'] = tokenizer
    worker_models['model'] = model
    worker_models['device'] = device


def write_rows_to_doc(output_dir: Path, file_id: str, rows: list):
    """
    Writes a list of formatted rows to the document's specific CSV file.
    Uses 'append' mode to support batch writing.
    """
    out_path = output_dir / f"{file_id}.csv"
    file_exists = out_path.exists()

    with open(out_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header if file is being created for the first time
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def process_and_write_batch(lines: list[str], meta: list[tuple], out_dir: Path):
    """
    Retrieves the loaded models from the worker's memory, performs batched inference,
    calculates structural flaws, categorizes the lines, and writes the results.

    Args:
        lines (list[str]): The raw text lines.
        meta (list[tuple]): Metadata linking the line to its file_id, page_id, and line_num.
        out_dir (Path): Output directory for the CSVs.
    """
    # Fetch models cached locally in this worker process
    ft = worker_models['ft']
    ppl_model = worker_models['model']
    tokenizer = worker_models['tokenizer']
    device = worker_models['device']

    # Batch compute perplexity
    ppls = calculate_perplexity_batch(lines, ppl_model, tokenizer, device)

    # Batch compute language ID predictions
    lines_lower = [line.lower() for line in lines]
    labels, scores = ft.predict(lines_lower, k=1)

    # Clean up fasttext '__label__' prefix
    langs = [l[0].replace("__label__", "") for l in labels]
    scores = [s[0] for s in scores]

    results = []

    # Iterate over the processed batch to compute granular metrics
    for i in range(len(lines)):
        file_id, page_id, line_num, text_content, split_ws, split_we = meta[i]

        ppl_val = ppls[i]
        lang = langs[i]
        score = scores[i]

        # Structural Checks
        sym_count = detect_strange_symbols(text_content)
        upper_count = detect_mid_uppercase(text_content)

        # Weirdness computation
        word_scores = score_words_in_line(text_content)
        weird_ratio = compute_word_weird_ratio(word_scores)

        # Linear Quality Score computation
        q_score = compute_quality_score(
            valid_word_ratio=compute_valid_ratio(text_content),
            symbol_ratio=compute_symbol_ratio(text_content),
            perplexity=ppl_val,
            text_length=len(text_content),
        )

        # Evaluate logic categories and fallbacks
        struct_cat = categorize_line(ppl_val, text_content, sym_count, upper_count, lang, score)
        pipe_cat = classify_pipeline(text_content)
        categ = struct_cat if struct_cat == pipe_cat else classify_by_score(q_score)

        row = [
            file_id, page_id, line_num, text_content,
            split_ws, split_we, lang, f"{score:.4f}", f"{ppl_val:.2f}",
            sym_count, upper_count, f"{weird_ratio:.4f}", f"{q_score:.4f}", categ,
        ]
        results.append(row)

    # Sort results by file to ensure grouping logic holds
    results.sort(key=lambda x: x[0])

    # Write to respective CSVs
    for file_id, group in groupby(results, key=lambda x: x[0]):
        write_rows_to_doc(out_dir, file_id, list(group))


def process_document(args) -> int:
    """
    The target function executed by a CPU worker.
    It parses an entire document line-by-line, resolves hyphenated splits,
    and groups lines into batches for the models to process.

    Args:
        args (tuple): Contains (file_id, file_rows_dataframe, text_dir, out_dir, batch_size)
    """
    file_id, file_rows, text_dir, out_dir, batch_size = args

    out_path = Path(out_dir) / f"{file_id}.csv"
    if out_path.exists():
        # Document previously processed. Skip to prevent duplicates.
        return 0

    batch_lines = []
    batch_meta = []
    processed_count = 0

    # Iterate through all pages of this specific document
    for _, row in file_rows.iterrows():
        page_id = str(row["page"])
        txt_path = Path(text_dir) / file_id / f"{file_id}-{page_id}.txt"

        if not txt_path.exists():
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        expected_incoming_suffix = ""

        # Read line by line, merging split words across line breaks
        for i, line in enumerate(lines, 1):
            merged_text, outgoing_prefix, outgoing_suffix = parse_line_splits(line)
            current_split_ws = outgoing_prefix
            current_split_we = ""

            if expected_incoming_suffix:
                stripped = merged_text.lstrip()
                if stripped.startswith(expected_incoming_suffix):
                    # Remove the completed suffix to avoid double-printing
                    merged_text = merged_text.replace(expected_incoming_suffix, "", 1).strip()
                    current_split_we = expected_incoming_suffix

            expected_incoming_suffix = outgoing_suffix

            # Fast CPU pre-filter
            cat, clean_merged = pre_filter_line(merged_text)

            if cat != "Process":
                # Immediately write Non-text or Empty lines to CSV, bypassing the GPU batch
                write_rows_to_doc(Path(out_dir), file_id, [[
                    file_id, page_id, i, clean_merged, current_split_ws, current_split_we,
                    "N/A", 0, 0, 0, 0, "0.0000", "0.0000", cat
                ]])
                continue

            # Append valid lines to the current batch
            batch_lines.append(clean_merged)
            batch_meta.append((file_id, page_id, i, clean_merged, current_split_ws, current_split_we))
            processed_count += 1

            # Dispatch batch when full
            if len(batch_lines) >= batch_size:
                process_and_write_batch(batch_lines, batch_meta, Path(out_dir))
                batch_lines.clear()
                batch_meta.clear()

    # Flush remaining lines
    if batch_lines:
        process_and_write_batch(batch_lines, batch_meta, Path(out_dir))

    # Sort the final CSV file to guarantee correct reading order downstream
    if out_path.exists():
        df = pd.read_csv(out_path)
        df = df.sort_values(by=["page_num", "line_num"], ascending=True)
        df.to_csv(out_path, index=False)

    return processed_count


def main():
    # Setup Paths via config
    config = configparser.ConfigParser()
    config.read("config_langID.txt")

    INPUT_CSV = config.get("CLASSIFY", "INPUT_CSV")
    TEXT_DIR = config.get("CLASSIFY", "TEXT_DIR")
    OUTPUT_DIR = config.get("CLASSIFY", "OUTPUT_LINES_LOG")
    BATCH_SIZE = config.getint("CLASSIFY", "BATCH_SIZE")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read the master index
    df = pd.read_csv(INPUT_CSV)
    sort_cols = (["file", "page", "line_order"] if "line_order" in df.columns else ["file", "page"])
    df = df.sort_values(by=sort_cols)

    # Prepare chunks by document. Each tuple becomes an argument mapped to a worker.
    grouped_tasks = []
    for file_id, group in df.groupby("file"):
        grouped_tasks.append((str(file_id), group, TEXT_DIR, OUTPUT_DIR, BATCH_SIZE))

    print(f"Starting classification on {len(grouped_tasks)} documents using Multiprocessing. Output → {OUTPUT_DIR}/")

    # Determine maximum number of workers. Limits to 12 or your hardware's CPU count.
    # Note: 12 instantiations of distilgpt2 will consume approx 3-4GB total VRAM/RAM.
    max_cores = min(multiprocessing.cpu_count(), 12)

    total_processed = 0

    # Spin up the worker pool. `init_worker` runs automatically in the new process.
    with ProcessPoolExecutor(max_workers=max_cores, initializer=init_worker) as executor:
        # Submit tasks asynchronously
        futures = {executor.submit(process_document, task): task[0] for task in grouped_tasks}

        # Track completion as workers return results
        for count, future in enumerate(as_completed(futures), 1):
            file_id = futures[future]
            try:
                lines_proc = future.result()
                total_processed += lines_proc
                print(f"[{count}/{len(grouped_tasks)}] Finished {file_id}")
            except Exception as e:
                print(f"Error processing {file_id}: {e}")


if __name__ == "__main__":
    # Required for proper handling of Torch/CUDA across multithreaded bounds in Python
    multiprocessing.set_start_method('spawn', force=True)
    main()