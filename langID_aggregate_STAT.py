#!/usr/bin/env python3
"""
langID_aggregate_STAT.py

Step 4.2: Aggregate raw lines into page statistics.

Reads a directory of per-document CSVs produced by the classification step
and compiles final page-level stats, including:
  - Counts of each line category (Clear / Noisy / Trash / Non-text / Empty)
  - 'avg_quality_score' – Mean composite quality score for relevant lines
  - 'avg_word_weird'    – Mean per-word weirdness ratio for relevant lines
  - 'avg_lang_score'    – Mean FastText confidence score
  - 'avg_perplex'       – Mean DistilGPT2 perplexity score
  - 'avg_symbol'        – Mean structural strange symbol count
  - 'avg_upper'         – Mean structural mid-word uppercase error count
  - 'main_lang'         – The statistical mode (most frequent) language per page.

This process is parallelized using concurrent.futures to handle massive directories quickly.
"""

import pandas as pd
from pathlib import Path
import configparser
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The expected outcome columns in the final stats pivot table
STANDARD_COLS = ["Clear", "Trash", "Noisy", "Empty", "Non-text"]

# Only these categories represent "actual attempts at text".
# Empty and Non-text are ignored when computing average quality scores.
SCORED_CATEGS = {"Clear", "Noisy", "Trash"}

# The numeric metric columns outputted by the classify step that we want to average
METRIC_COLS = [
    "word_count", "char_count", "garbage_density",
    "lang_score", "perplex",
    "symbol", "upper", "repeated", "ldl_fuses", "gibberish",
    "word_weird", "quality_score"
]

# ---------------------------------------------------------------------------
# Aggregation Helpers
# ---------------------------------------------------------------------------

def _category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the DataFrame to count how many lines fall into each Category per Page.

    Args:
        df: The raw lines dataframe.
    Returns:
        pd.DataFrame: A frame indexed by (file, page_num) with STANDARD_COLS as columns.
    """
    counts = (df.groupby(["file", "page_num"])["categ"].value_counts().unstack(fill_value=0))

    for col in STANDARD_COLS:
        if col not in counts.columns:
            counts[col] = 0
    return counts[STANDARD_COLS]


def _mean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the average structural and ML metrics per Page.
    Restricts calculation strictly to valid text lines (SCORED_CATEGS).
    """
    scored = df[df["categ"].isin(SCORED_CATEGS)].copy()

    # Identify which of our target metrics actually exist in this CSV
    agg = {col: "mean" for col in METRIC_COLS if col in scored.columns}

    if not agg:
        # Return an empty placeholder preserving the index
        idx = df.set_index(["file", "page_num"]).index.unique()
        return pd.DataFrame(index=idx)

    # Compute means based on the dictionary
    means = scored.groupby(["file", "page_num"])[list(agg.keys())].mean()

    # Rename for final output (e.g. 'perplex' -> 'avg_perplex')
    rename = {col: f"avg_{col}" for col in agg.keys()}
    return means.rename(columns=rename)


def _prevailing_lang(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the single most prevailing language per page.
    Filters out lines categorized as "N/A" or "unknown" to prevent OCR noise
    from skewing the document's true language profile.
    """
    valid_langs = df[~df['lang'].isin(['N/A', 'unknown'])].copy()

    if valid_langs.empty:
        idx = df.set_index(["file", "page_num"]).index.unique()
        return pd.DataFrame({'main_lang': 'unknown'}, index=idx)

    main_langs = valid_langs.groupby(["file", "page_num"])['lang'].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else 'unknown'
    )
    return main_langs.to_frame(name='main_lang')


def _build_page_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines Category Counts, Averages, and Main Language into a single flat DataFrame.
    """
    counts = _category_counts(df)
    means = _mean_metrics(df)
    langs = _prevailing_lang(df)

    stats = counts.join(means, how="left").join(langs, how="left")

    # Round all calculated average columns to 4 decimal places
    for col in stats.columns:
        if col.startswith("avg_"):
            stats[col] = stats[col].round(4)

    stats.reset_index(inplace=True)
    return stats


def process_csv_file(args):
    """Worker function for parallel mapping."""
    csv_file, output_doc_dir = args
    try:
        df = pd.read_csv(csv_file)
        if df.empty or "categ" not in df.columns:
            return None

        stats = _build_page_stats(df)
        stats_out_path = Path(output_doc_dir) / f"stats_{csv_file.name}"
        stats.to_csv(stats_out_path, index=False)
        return stats
    except Exception as exc:
        print(f"Error processing file {csv_file.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    config = configparser.ConfigParser()
    config.read("config_langID.txt")

    INPUT_DIR_PATH = config.get("AGGREGATE", "RAW_LINES_CSV")
    OUTPUT_STATS = config.get("AGGREGATE", "OUTPUT_STATS")
    OUTPUT_DOC_DIR = config.get("AGGREGATE", "OUTPUT_DOC_DIR")

    input_dir = Path(INPUT_DIR_PATH)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    Path(OUTPUT_DOC_DIR).mkdir(parents=True, exist_ok=True)

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        sys.exit(0)

    print(f"Aggregating {len(csv_files)} documents using Multiprocessing...")
    all_page_stats = []

    # Aggregation is CPU light, max out thread count.
    max_cores = min(multiprocessing.cpu_count(), 12)

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        tasks = [(f, OUTPUT_DOC_DIR) for f in csv_files]
        futures = {executor.submit(process_csv_file, t): t for t in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                all_page_stats.append(result)

            if i % 50 == 0:
                print(f"Aggregated {i}/{len(csv_files)} documents...")

    if all_page_stats:
        print("Consolidating final page stats ...")
        final_df = pd.concat(all_page_stats, ignore_index=True)
        final_df.sort_values(by=["file", "page_num"], inplace=True)

        final_df.to_csv(OUTPUT_STATS, index=False)
        print(f"Done. Global stats saved to {OUTPUT_STATS} (Includes averaged metrics and 'main_lang' column)")
    else:
        print("No statistics were generated.")


if __name__ == "__main__":
    main()