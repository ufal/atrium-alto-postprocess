#!/usr/bin/env python3
"""
langID_classify.py  (formerly 2_classify.py)

Step 2: Read TXT files → Merge Split Words → Batch classify on GPU.
Output: One CSV file per document in OUTPUT_LINES_LOG directory.
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
from text_util_langID import *
from atrium_paradata import ParadataLogger
import configparser as _cp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_HEADER = [
    "file", "page_num", "line_num", "text",
    "split_ws", "split_we",
    "lang", "lang_score", "perplex",
    "symbol", "upper",
    "word_weird",
    "quality_score",
    "categ",
]

def load_models():
    print(f"Loading models on {DEVICE}...")
    ft = fasttext.load_model("lid.176.bin")

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("distilgpt2").to(DEVICE)
    model.eval()

    return ft, model, tokenizer

def write_rows_to_doc(output_dir: Path, file_id: str, rows: list):
    tmp_path = output_dir / f"{file_id}.csv.tmp"
    file_exists = tmp_path.exists()

    with open(tmp_path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        writer.writerows(rows)

def sort_document_csv(output_dir: Path, file_id: str):
    tmp_path = output_dir / f"{file_id}.csv.tmp"
    out_path = output_dir / f"{file_id}.csv"
    if tmp_path.exists():
        df = pd.read_csv(tmp_path)
        df = df.sort_values(by=["page_num", "line_num"], ascending=True)
        df.to_csv(out_path, index=False)
        tmp_path.unlink()

def process_and_write_batch(
    lines: list[str],
    meta: list[tuple],
    out_dir: Path,
    ft,
    ppl_model,
    tokenizer,
):
    ppls = calculate_perplexity_batch(lines, ppl_model, tokenizer, DEVICE)

    lines_lower = [line.lower() for line in lines]
    labels, scores = ft.predict(lines_lower, k=1)
    langs  = [l[0].replace("__label__", "") for l in labels]
    scores = [s[0] for s in scores]

    results = []
    for i in range(len(lines)):
        file_id, page_id, line_num, text_content, split_ws, split_we = meta[i]

        ppl_val     = ppls[i]
        lang        = langs[i]
        score       = scores[i]
        sym_count   = detect_strange_symbols(text_content)
        upper_count = detect_mid_uppercase(text_content)

        word_scores = score_words_in_line(text_content)
        weird_ratio = compute_word_weird_ratio(word_scores)

        # FIXED: Pass explicitly computed valid ratio, not inverse weirdness
        q_score = compute_quality_score(
            valid_word_ratio=compute_valid_ratio(text_content),
            symbol_ratio=compute_symbol_ratio(text_content),
            perplexity=ppl_val,
            text_length=len(text_content),
        )

        # FIXED: Pass language ID predictions into the rule-based categorizer
        struct_cat = categorize_line(ppl_val, text_content, sym_count, upper_count, lang, score)
        pipe_cat   = classify_pipeline(text_content)

        if struct_cat == pipe_cat:
            categ = struct_cat
        else:
            categ = classify_by_score(q_score)

        row = [
            file_id,
            page_id,
            line_num,
            text_content,
            split_ws,
            split_we,
            lang,
            f"{score:.4f}",
            f"{ppl_val:.2f}",
            sym_count,
            upper_count,
            f"{weird_ratio:.4f}",
            f"{q_score:.4f}",
            categ,
        ]
        results.append(row)

    results.sort(key=lambda x: x[0])

    for file_id, group in groupby(results, key=lambda x: x[0]):
        write_rows_to_doc(out_dir, file_id, list(group))

def main():
    config = configparser.ConfigParser()
    config.read("config_langID.txt")

    _cfg_p = config
    _logger = ParadataLogger(
        program="alto-postprocess",
        config={
            "script": "langID_classify",
            "input_txt_dir": _cfg_p.get("CLASSIFY", "TEXT_DIR", fallback=""),
            "input_csv": _cfg_p.get("CLASSIFY", "INPUT_CSV", fallback=""),
            "output_dir": _cfg_p.get("CLASSIFY", "OUTPUT_LINES_LOG", fallback=""),
            "fasttext_model": _cfg_p.get("CLASSIFY", "fasttext_model", fallback="lid.176.bin"),
            "gpt2_model": "distilgpt2",
            "ppl_threshold": _cfg_p.get("CLASSIFY", "ppl_threshold", fallback="1500"),
            "batch_size": _cfg_p.get("CLASSIFY", "BATCH_SIZE", fallback=""),
        },
        paradata_dir="paradata",
        output_types=["csv"],
    )
    _total_inputs = 0

    INPUT_CSV  = config.get("CLASSIFY", "INPUT_CSV")
    TEXT_DIR   = config.get("CLASSIFY", "TEXT_DIR")
    OUTPUT_DIR = config.get("CLASSIFY", "OUTPUT_LINES_LOG")
    BATCH_SIZE = config.getint("CLASSIFY", "BATCH_SIZE")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ft_model, ppl_model, ppl_tok = load_models()

    df = pd.read_csv(INPUT_CSV)

    batch_lines: list[str]   = []
    batch_meta:  list[tuple] = []

    current_file_id      = None
    skipping_current_file = False
    session_files:  set   = set()

    print(f"Starting classification. Output → {OUTPUT_DIR}/")

    try:
        sort_cols = (
            ["file", "page", "line_order"]
            if "line_order" in df.columns
            else ["file", "page"]
        )
        df = df.sort_values(by=sort_cols)

        page_id = 0
        for _, row in tqdm(df.iterrows(), total=len(df)):
            file_id = str(row["file"])
            prev_pi = page_id
            page_id = str(row["page"])

            if file_id != current_file_id:
                if batch_lines:
                    process_and_write_batch(
                        batch_lines, batch_meta, out_dir, ft_model, ppl_model, ppl_tok
                    )
                    batch_lines.clear()
                    batch_meta.clear()

                if current_file_id is not None and not skipping_current_file:
                    sort_document_csv(out_dir, current_file_id)
                    _logger.log_success("csv")

                current_file_id = file_id
                out_path = out_dir / f"{file_id}.csv"

                if out_path.exists() and file_id not in session_files:
                    skipping_current_file = True
                else:
                    skipping_current_file = False
                    session_files.add(file_id)

            if skipping_current_file:
                continue

            txt_path = Path(TEXT_DIR) / file_id / f"{file_id}-{page_id}.txt"

            if not txt_path.exists():
                _logger.log_skip(str(txt_path), "not found")
                continue

            _total_inputs += 1

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
                        merged_text = merged_text.replace(
                            expected_incoming_suffix, "", 1
                        ).strip()
                        current_split_we = expected_incoming_suffix

                expected_incoming_suffix = outgoing_suffix

                cat, clean_merged = pre_filter_line(merged_text)

                if cat != "Process":
                    write_rows_to_doc(
                        out_dir,
                        file_id,
                        [[
                            file_id, page_id, i, clean_merged,
                            current_split_ws, current_split_we,
                            "N/A", 0, 0,
                            0, 0,
                            "0.0000", "0.0000",
                            cat,
                        ]],
                    )
                    continue

                batch_lines.append(clean_merged)
                batch_meta.append(
                    (file_id, page_id, i, clean_merged, current_split_ws, current_split_we)
                )

                if len(batch_lines) >= BATCH_SIZE:
                    process_and_write_batch(
                        batch_lines, batch_meta, out_dir, ft_model, ppl_model, ppl_tok
                    )
                    batch_lines.clear()
                    batch_meta.clear()

        if batch_lines:
            process_and_write_batch(
                batch_lines, batch_meta, out_dir, ft_model, ppl_model, ppl_tok
            )

        if current_file_id is not None and not skipping_current_file:
            sort_document_csv(out_dir, current_file_id)
            _logger.log_success("csv")

    finally:
        _logger.finalize(input_total=_total_inputs)


if __name__ == "__main__":
    main()