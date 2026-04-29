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
  - 'avg_perplex'       - Mean DistilGPT2 perplexity score
  - 'avg_symbol'        - Mean structural strange symbol count
  - 'main_lang'         - The statistical mode (most frequent) language per page.
  - 'avg_vowel_ratio'   - Mean vowel ratio
  - 'ch_ratio'          - The ratio of caps_header lines to valid lines

This process is parallelized using concurrent.futures to handle massive directories quickly.
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

STANDARD_COLS = ["Clear", "Noisy", "Trash", "Non-text", "Empty"]
DEFAULT_CONFIG = "config_langID.txt"


def load_config(config_path):
    config = configparser.ConfigParser()
    if not Path(config_path).exists():
        print(f"Warning: Configuration file {config_path} not found. Using defaults.")
        return {"input_dir": "data_samples/DOC_LINE_LANG_CLASS", "output_dir": "data_samples/DOC_PAGE_STAT"}
    config.read(config_path)
    return {
        "input_dir": config.get("Paths", "INPUT_DIR", fallback="data_samples/DOC_LINE_LANG_CLASS"),
        "output_dir": config.get("Paths", "OUTPUT_DIR", fallback="data_samples/DOC_PAGE_STAT"),
    }


def _sum_metrics(df):
    """
    Groups line data by page and aggregates the statistics.
    """
    if df.empty:
        return pd.DataFrame()

    # Isolate relevant lines for metrics (Clear and Noisy)
    valid_lines = df[df['category'].isin(["Clear", "Noisy"])].copy()

    # Count categories per page
    cat_counts = df.groupby(['file', 'page_num', 'category']).size().unstack(fill_value=0).reset_index()
    for col in STANDARD_COLS:
        if col not in cat_counts.columns:
            cat_counts[col] = 0

    if valid_lines.empty:
        # If no valid lines, return zeros/NaNs for stats
        stats = df[['file', 'page_num']].drop_duplicates().copy()
        for col in ['total_word_count', 'total_char_count']:
            stats[col] = 0
        for col in ['avg_quality_score', 'avg_word_weird', 'avg_lang_score',
                    'avg_perplex', 'avg_symbol', 'avg_vowel_ratio', 'ch_ratio']:
            stats[col] = pd.NA
        stats['main_lang'] = "None"

        final_page_df = pd.merge(cat_counts, stats, on=['file', 'page_num'], how='left')
        return final_page_df

    # --- MAIN FIX 1: NaN cascades on boolean columns ---
    if 'caps_header' in valid_lines.columns:
        # csv.writer formats bools as 'True' / 'False'. Coerce back to float for .mean()
        valid_lines['caps_header'] = valid_lines['caps_header'].replace(
            {'True': 1.0, 'False': 0.0, True: 1.0, False: 0.0}
        ).astype(float)

    # Calculate main stats for valid lines
    stats = valid_lines.groupby(['file', 'page_num']).agg(
        total_word_count=('word_count', 'sum'),
        total_char_count=('char_count', 'sum'),
        avg_quality_score=('quality_score', 'mean'),
        avg_word_weird=('word_weirdness', 'mean'),
        avg_lang_score=('lang_score', 'mean'),
        avg_perplex=('perplex', 'mean'),
        avg_symbol=('symbol_count', 'mean'),
        avg_vowel_ratio=('vowel_ratio', 'mean')
    ).reset_index()

    # Calculate caps header ratio
    if 'caps_header' in valid_lines.columns:
        ch_stats = valid_lines.groupby(['file', 'page_num'])['caps_header'].mean().reset_index(name='ch_ratio')
        stats = pd.merge(stats, ch_stats, on=['file', 'page_num'], how='left')
    else:
        stats['ch_ratio'] = 0.0

    # Calculate statistical mode for language
    if 'lang' in valid_lines.columns:
        def mode_lang(x):
            return x.mode().iloc[0] if not x.empty else "None"

        lang_stats = valid_lines.groupby(['file', 'page_num'])['lang'].apply(mode_lang).reset_index(name='main_lang')
        stats = pd.merge(stats, lang_stats, on=['file', 'page_num'], how='left')
    else:
        stats['main_lang'] = "None"

    # Merge category counts with the stats
    final_page_df = pd.merge(cat_counts, stats, on=['file', 'page_num'], how='left')

    # --- MAIN FIX 3 & 4: int casting NaN errors on count columns ---
    for count_col in ['total_word_count', 'total_char_count', 'word_count', 'char_count']:
        if count_col in final_page_df.columns:
            final_page_df[count_col] = final_page_df[count_col].fillna(0).astype(int)

    return final_page_df


def process_csv_file(args):
    """
    Reads a single CSV file, defines proper data types to prevent edge-case
    inferences, and returns aggregated page metrics.
    """
    file_path, _ = args
    try:
        # --- MAIN FIX 2: Explicit dtypes to stop format string parsing as 'object' ---
        dtype_map = {
            'word_count': 'float64',
            'char_count': 'float64',
            'quality_score': 'float64',
            'word_weirdness': 'float64',
            'lang_score': 'float64',
            'perplex': 'float64',
            'garbage_density': 'float64',
            'symbol_count': 'float64',
            'vowel_ratio': 'float64'
            # Caps header handled specifically later due to bool string mix
        }

        # Handle empty/missing rows gracefully
        df = pd.read_csv(file_path, dtype=dtype_map, on_bad_lines='skip')

        if df.empty:
            return None

        # Strip any whitespace from column names just in case
        df.columns = df.columns.str.strip()

        return _sum_metrics(df)

    except pd.errors.EmptyDataError:
        return None
    except Exception as exc:
        # Return the exception to the main thread so we can handle it safely without a loop crash
        return exc


def main():
    parser = argparse.ArgumentParser(description="Aggregate post-classification line metrics into page stats.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        sys.exit(0)

    # Initialize Paradata Logger
    logger = ParadataLogger(
        program="langID-aggregate",
        config=vars(args),
        paradata_dir="paradata",
        output_types=["csv"],
    )

    print(f"Aggregating {len(csv_files)} documents using Multiprocessing...")
    all_page_stats = []

    # Aggregation is CPU light, max out thread count safely.
    max_cores = min(multiprocessing.cpu_count(), 12)

    try:
        with ProcessPoolExecutor(max_workers=max_cores) as executor:
            tasks = [(f, output_dir) for f in csv_files]

            # Keep track of which future maps to which file for accurate error reporting
            futures = {executor.submit(process_csv_file, t): t[0] for t in tasks}

            for future in tqdm(as_completed(futures), total=len(csv_files), desc="Aggregating Page Stats"):
                original_file = futures[future]

                # --- MAIN FIX 2: Guard future.result() and surface failing filename ---
                try:
                    result = future.result()
                    if isinstance(result, Exception):
                        tqdm.write(f"Error processing file {original_file.name}: {result}")
                        logger.log_skip(original_file.name, f"Processing Error: {result}")
                    elif result is not None and not result.empty:
                        all_page_stats.append(result)
                        logger.log_success("csv")
                    else:
                        logger.log_skip(original_file.name, "Empty or invalid CSV structure")
                except Exception as exc:
                    tqdm.write(f"Hard crash while processing {original_file.name}: {exc}")
                    logger.log_skip(original_file.name, f"Hard Crash: {exc}")

        if all_page_stats:
            print("Consolidating final page stats ...")
            final_df = pd.concat(all_page_stats, ignore_index=True)

            if 'file' in final_df.columns and 'page_num' in final_df.columns:
                final_df.sort_values(by=["file", "page_num"], inplace=True)

            output_file = output_dir / "AGGREGATED_PAGE_STATS.csv"
            final_df.to_csv(output_file, index=False)
            print(f"Done. Final stats saved to {output_file}")
        else:
            print("No valid page stats could be aggregated.")

    finally:
        # Finalize paradata logging regardless of crashes
        logger.finalize(input_total=len(csv_files))


if __name__ == "__main__":
    main()