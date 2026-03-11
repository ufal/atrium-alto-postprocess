#!/usr/bin/env python3
"""
2_classify.py
Step 2: Read TXT files, Merge Split Words, Batch, Classify on GPU.
Output: Individual CSV files per document.
"""
import pandas as pd
import torch
import fasttext
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import csv
import sys
from tqdm import tqdm
from itertools import groupby
import configparser
from text_util_langID import *  # Config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    print(f"Loading models on {DEVICE}...")
    ft = fasttext.load_model("lid.176.bin")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE)
    model.eval()

    return ft, model, tokenizer


def write_rows_to_doc(output_dir, file_id, rows):
    """
    Appends rows to the specific document CSV.
    """
    out_path = Path(output_dir) / f"{file_id}.csv"
    file_exists = out_path.exists()

    with open(out_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # HEADER: split_ws (start/prefix), split_we (end/suffix)
            writer.writerow([
                "file", "page_num", "line_num", "text",
                "split_we", "split_ws",
                "lang", "lang_score", "perplex", "categ"
            ])
        writer.writerows(rows)


def main():
    config = configparser.ConfigParser()
    config.read('config_langID.txt')

    INPUT_CSV = config.get('CLASSIFY', 'INPUT_CSV')
    TEXT_DIR = config.get('CLASSIFY', 'TEXT_DIR')
    OUTPUT_DIR = config.get('CLASSIFY', 'OUTPUT_LINES_LOG')
    BATCH_SIZE = config.getint('CLASSIFY', 'BATCH_SIZE')

    out_dir_path = Path(OUTPUT_DIR)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    ft_model, ppl_model, ppl_tok = load_models()

    df = pd.read_csv(INPUT_CSV)

    batch_lines = []
    batch_meta = []

    current_file_id = None
    skipping_current_file = False
    session_files = set()

    print(f"Starting classification loop. Outputting to {OUTPUT_DIR}/...")

    # Sort to ensure we process lines in order
    df = df.sort_values(by=['file', 'page', 'line_order'] if 'line_order' in df.columns else ['file', 'page'])

    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_id = str(row['file'])
        page_id = str(row['page'])

        # File Skip Logic
        if file_id != current_file_id:
            # Document finalized: Flush the batch for the previous file before switching
            if batch_lines:
                process_and_write_batch(batch_lines, batch_meta, out_dir_path, ft_model, ppl_model, ppl_tok)
                batch_lines = []
                batch_meta = []

            # Document finalized: Sort the previous document's CSV
            if current_file_id is not None and not skipping_current_file:
                sort_document_csv(out_dir_path, current_file_id)

            current_file_id = file_id
            out_path = out_dir_path / f"{file_id}.csv"

            if out_path.exists() and file_id not in session_files:
                skipping_current_file = True
            else:
                skipping_current_file = False
                session_files.add(file_id)

        if skipping_current_file:
            continue

        txt_path = Path(TEXT_DIR) / file_id / f"{file_id}-{page_id}.txt"
        if not txt_path.exists():
            continue

        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # --- WORD SPLIT STATE RESET ---
        # "ded" expected at start of next line
        expected_incoming_suffix = ""

        for i, line in enumerate(lines, 1):

            # 1. Parse Splits (Outgoing)
            merged_text, outgoing_prefix, outgoing_suffix = parse_line_splits(line)

            # 2. Current Line's Split State
            current_split_ws = outgoing_prefix
            current_split_we = ""

            if expected_incoming_suffix != "":
                stripped_candidate = merged_text.lstrip()
                if stripped_candidate.startswith(expected_incoming_suffix):
                    merged_text = merged_text.replace(expected_incoming_suffix, "", 1).strip()
                    current_split_we = expected_incoming_suffix

            # 3. Update state for NEXT line
            expected_incoming_suffix = outgoing_suffix

            # 4. Pre-filter (now includes quote balancing)
            cat, clean_merged = pre_filter_line(merged_text)

            if cat != "Process":
                row_data = [
                    file_id, page_id, i, clean_merged,
                    current_split_we, current_split_ws,
                    "N/A", 0, 0, cat
                ]
                write_rows_to_doc(out_dir_path, file_id, [row_data])
                continue

            # 6. Add to Batch
            batch_lines.append(clean_merged)
            batch_meta.append((file_id, page_id, i, clean_merged, current_split_we, current_split_ws))

            # PROCESS BATCH
            if len(batch_lines) >= BATCH_SIZE:
                process_and_write_batch(batch_lines, batch_meta, out_dir_path, ft_model, ppl_model, ppl_tok)
                batch_lines = []
                batch_meta = []

    # Final Batch (catch-all at the end of the script)
    if batch_lines:
        process_and_write_batch(batch_lines, batch_meta, out_dir_path, ft_model, ppl_model, ppl_tok)

    # Sort the very last document processed in the loop
    if current_file_id is not None and not skipping_current_file:
        sort_document_csv(out_dir_path, current_file_id)


def process_and_write_batch(lines, meta, out_dir, ft, ppl_model, tokenizer):
    """
    Runs models, aggregates results, writes to CSV.
    """
    ppls = calculate_perplexity_batch(lines, ppl_model, tokenizer, DEVICE)

    lines_lower = [line.lower() for line in lines]
    labels, scores = ft.predict(lines_lower, k=1)
    langs = [l[0].replace("__label__", "") for l in labels]
    scores = [s[0] for s in scores]

    results = []
    for i in range(len(lines)):
        file_id, page_id, line_num, text_content, split_we, split_ws = meta[i]

        row = [
            file_id,
            page_id,
            line_num,
            text_content,
            split_we,
            split_ws,
            langs[i],
            f"{scores[i]:.4f}",
            f"{ppls[i]:.2f}",
            categorize_line(langs[i], scores[i], ppls[i], text_content)
        ]
        results.append(row)

    results.sort(key=lambda x: x[0])

    for file_id, group in groupby(results, key=lambda x: x[0]):
        rows_for_file = list(group)
        write_rows_to_doc(out_dir, file_id, rows_for_file)


def sort_document_csv(output_dir, file_id):
    """
    Sorts the finalized document CSV by page number and line number.
    This is called only when the document is completely finished processing.
    """
    out_path = Path(output_dir) / f"{file_id}.csv"
    if out_path.exists():
        df_csv = pd.read_csv(out_path)
        # Sort values ascendingly by page_num and line_num
        df_csv = df_csv.sort_values(by=['page_num', 'line_num'], ascending=[True, True])
        df_csv.to_csv(out_path, index=False)

if __name__ == "__main__":
    main()