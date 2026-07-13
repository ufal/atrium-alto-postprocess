#!/usr/bin/env python3
"""
langID_aggregate_STAT.py

Step 4.2: Aggregate raw lines into page statistics.

Reads a directory of per-document CSVs produced by the classification step
and compiles final page-level stats, including:
  - Counts of each line category (Clear / Noisy / Trash / Non-text / Empty)
  - 'total_word_count'  - Sum of words in valid text lines
  - 'total_char_count'  - Sum of characters in valid text lines
  - 'avg_quality_score' - Mean composite quality score for relevant lines
  - 'avg_word_weird'    - Mean per-word weirdness ratio for relevant lines
  - 'avg_lang_score'    - Mean FastText confidence score
  - 'avg_perplex'       - Mean perplexity score
  - 'main_lang'         - The statistical mode (most frequent) language per page.
  - 'avg_vowel_ratio'   - Mean vowel ratio
  - 'ch_ratio'          - The ratio of caps_header lines to valid lines

This process is parallelized using concurrent.futures.

(#3) The per-document line CSVs now carry additional columns for transparency
(`original_text`, `original_lang`, `orig_lang_score`, `weird_wx`) alongside the nine
diagnostic boolean columns after `caps_header` (six categoriser-rule flags + three
post-pass flags). This aggregation reads strictly by column name, safely ignoring
the new fields — page stats are unchanged — but a future revision could emit
per-page rule-frequency sums from them.
"""

import argparse
import configparser
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from atrium_paradata import ParadataLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# STANDARD_COLS = ["Clear", "Noisy", "Trash", "Non-text", "Empty"]
DEFAULT_CONFIG = "config_langID.txt"


def load_config(config_path):
    """Loads configuration fields mapped to required system paths.

    (#14) Defaults are aligned with the documented artifact names
    (DOC_LINE_CATEG / DOC_LINE_STATS) and the shipped config so a missing
    config file does not silently introduce a different directory layout.
    """
    config = configparser.ConfigParser()
    if not Path(config_path).exists():
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return {
            "input_dir": "data_samples/DOC_LINE_CATEG",
            "output_dir": "data_samples/DOC_LINE_STATS",
            "output_stats": "samples_page_stats.csv",
            "standard_cols": "Clear,Noisy,Trash,Non-text,Empty",
        }
    config.read(config_path)

    # (#7 Phase 0) STANDARD_COLS was documented in [AGGREGATE] but never read:
    # main() called .get("standard_cols", ...) on a dict that lacked the key,
    # so the fallback default always won. Now wired through load_config().
    return {
        "input_dir": config.get("AGGREGATE", "RAW_LINES_CSV", fallback="data_samples/DOC_LINE_CATEG"),
        "output_dir": config.get("AGGREGATE", "OUTPUT_DOC_DIR", fallback="data_samples/DOC_LINE_STATS"),
        "output_stats": config.get("AGGREGATE", "OUTPUT_STATS", fallback="samples_page_stats.csv"),
        "standard_cols": config.get("AGGREGATE", "STANDARD_COLS", fallback="Clear,Noisy,Trash,Non-text,Empty"),
    }


def _sum_metrics(df, STANDARD_COLS):
    """Groups line data by page and aggregates the statistics."""
    if df.empty:
        return pd.DataFrame()

    valid_lines = df[df["categ"].isin(["Clear", "Noisy"])].copy()

    cat_counts = df.groupby(["file", "page_num", "categ"]).size().unstack(fill_value=0).reset_index()
    for col in STANDARD_COLS:
        if col not in cat_counts.columns:
            cat_counts[col] = 0

    if valid_lines.empty:
        stats = df[["file", "page_num"]].drop_duplicates().copy()
        for col in ["total_word_count", "total_char_count"]:
            stats[col] = 0
        for col in [
            "avg_quality_score",
            "avg_word_weird",
            "avg_lang_score",
            "avg_perplex",
            "avg_vowel_ratio",
            "avg_rot_ratio",
            "ch_ratio",
        ]:
            stats[col] = float("nan")
        stats["main_lang"] = "None"

        final_page_df = pd.merge(cat_counts, stats, on=["file", "page_num"], how="left")
        final_page_df["num_lines"] = final_page_df[STANDARD_COLS].sum(axis=1)
        ordered_cols = [
            "file",
            "page_num",
            "num_lines",
            "Clear",
            "Noisy",
            "Trash",
            "Non-text",
            "Empty",
            "total_word_count",
            "total_char_count",
            "avg_quality_score",
            "avg_word_weird",
            "avg_lang_score",
            "avg_perplex",
            "avg_vowel_ratio",
            "avg_rot_ratio",
            "ch_ratio",
            "main_lang",
        ]
        return final_page_df[ordered_cols]

    if "caps_header" in valid_lines.columns:
        valid_lines["caps_header"] = (
            valid_lines["caps_header"].map({"True": 1.0, "False": 0.0, True: 1.0, False: 0.0}).astype(float)
        )

    stats = (
        valid_lines.groupby(["file", "page_num"])
        .agg(
            total_word_count=("word_count", "sum"),
            total_char_count=("char_count", "sum"),
            avg_quality_score=("quality_score", "mean"),
            avg_word_weird=("word_weird", "mean"),
            avg_lang_score=("lang_score", "mean"),
            avg_perplex=("perplex", "mean"),
            # avg_symbol=('symbol', 'mean'),
            avg_vowel_ratio=("vowel_ratio", "mean"),
            avg_rot_ratio=("rot_ratio", "mean"),
        )
        .reset_index()
    )

    if "caps_header" in valid_lines.columns:
        ch_stats = valid_lines.groupby(["file", "page_num"])["caps_header"].mean().reset_index(name="ch_ratio")
        stats = pd.merge(stats, ch_stats, on=["file", "page_num"], how="left")
    else:
        stats["ch_ratio"] = 0.0

    if "lang" in valid_lines.columns:

        def mode_lang(x):
            return x.mode().iloc[0] if not x.empty else "None"

        lang_stats = valid_lines.groupby(["file", "page_num"])["lang"].apply(mode_lang).reset_index(name="main_lang")
        stats = pd.merge(stats, lang_stats, on=["file", "page_num"], how="left")
    else:
        stats["main_lang"] = "None"

    final_page_df = pd.merge(cat_counts, stats, on=["file", "page_num"], how="left")

    for count_col in ["total_word_count", "total_char_count", "word_count", "char_count"]:
        if count_col in final_page_df.columns:
            final_page_df[count_col] = final_page_df[count_col].fillna(0).astype(int)

    final_page_df["num_lines"] = final_page_df[STANDARD_COLS].sum(axis=1)

    ordered_cols = [
        "file",
        "page_num",
        "num_lines",
        "Clear",
        "Noisy",
        "Trash",
        "Non-text",
        "Empty",
        "total_word_count",
        "total_char_count",
        "avg_quality_score",
        "avg_word_weird",
        "avg_lang_score",
        "avg_perplex",
        "avg_vowel_ratio",
        "avg_rot_ratio",
        "ch_ratio",
        "main_lang",
    ]
    return final_page_df[ordered_cols]


def process_csv_file(file_path, STANDARD_COLS):
    """Reads a single CSV file and returns aggregated page metrics."""
    try:
        dtype_map = {
            "split_ws": str,
            "split_we": str,
            "word_count": "float64",
            "char_count": "float64",
            "quality_score": "float64",
            "word_weird": "float64",
            "lang_score": "float64",
            "perplex": "float64",
            "garbage_density": "float64",
            # 'symbol': 'float64',
            "vowel_ratio": "float64",
            "rot_ratio": "float64",
        }

        df = pd.read_csv(file_path, dtype=dtype_map, on_bad_lines="skip")

        if df.empty:
            return None

        df.columns = df.columns.str.strip()

        return _sum_metrics(df, STANDARD_COLS)

    except pd.errors.EmptyDataError:
        return None
    except Exception as exc:
        return exc


def main():
    parser = argparse.ArgumentParser(description="Aggregate post-classification line metrics into page stats.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    output_stats_path = Path(config["output_stats"])
    STANDARD_COLS = frozenset(config.get("standard_cols", "Clear,Noisy,Trash,Non-text,Empty").split(","))

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stats_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        sys.exit(0)

    logger = ParadataLogger(
        program="langID-aggregate",
        config=vars(args),
        paradata_dir="paradata",
        output_types=["csv"],
    )

    print(f"Aggregating {len(csv_files)} documents using Multiprocessing...")
    all_page_stats = []

    max_cores = min(multiprocessing.cpu_count(), 12)

    try:
        with ProcessPoolExecutor(max_workers=max_cores) as executor:
            # futures = {executor.submit(process_csv_file, f): f for f in csv_files}
            futures = {executor.submit(process_csv_file, f, STANDARD_COLS): f for f in csv_files}

            for future in tqdm(as_completed(futures), total=len(csv_files), desc="Aggregating Page Stats"):
                original_file = futures[future]

                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        tqdm.write(f"Error processing file {original_file.name}: {result}")
                        logger.log_skip(original_file.name, f"Processing Error: {result}")
                    elif result is not None and not result.empty:
                        all_page_stats.append(result)
                        doc_out = output_dir / f"stats_{original_file.stem}.csv"
                        result.to_csv(doc_out, index=False, encoding="utf-8")
                        logger.log_success("csv")
                    else:
                        logger.log_skip(original_file.name, "Empty or invalid CSV structure")
                except Exception as exc:
                    tqdm.write(f"Hard crash while processing {original_file.name}: {exc}")
                    logger.log_skip(original_file.name, f"Hard Crash: {exc}")

        if all_page_stats:
            print("Consolidating final page stats ...")
            final_df = pd.concat(all_page_stats, ignore_index=True)

            if "file" in final_df.columns and "page_num" in final_df.columns:
                final_df.sort_values(by=["file", "page_num"], inplace=True)

            final_df.to_csv(output_stats_path, index=False, encoding="utf-8")
            print(f"Done. Final stats saved to {output_stats_path}")
        else:
            print("No valid page stats could be aggregated.")

    finally:
        logger.finalize(input_total=len(csv_files))


if __name__ == "__main__":
    main()
