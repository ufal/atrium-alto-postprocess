#!/usr/bin/env python3
"""
langID_aggregate_STAT.py

Step 4.2: Aggregate raw lines into page statistics.

Reads a directory of per-document CSVs produced by the classification step
and compiles final page-level stats, including:
  - Counts of each line category (Clear / Noisy / Trash / Non-text / Empty)
  - 'avg_quality_score' – Mean composite quality score for relevant lines
  - 'avg_word_weird'    – Mean per-word weirdness ratio for relevant lines
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
    # unstack() converts the "categ" values into distinct columns
    counts = (df.groupby(["file", "page_num"])["categ"].value_counts().unstack(fill_value=0))

    # Ensure all standard columns exist even if no lines fell into that category
    for col in STANDARD_COLS:
        if col not in counts.columns:
            counts[col] = 0
    return counts[STANDARD_COLS]


def _mean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the average structural quality scores per Page.
    Restricts calculation strictly to valid text lines (SCORED_CATEGS).
    """
    scored = df[df["categ"].isin(SCORED_CATEGS)].copy()
    agg = {}

    if "quality_score" in scored.columns: agg["quality_score"] = "mean"
    if "word_weird" in scored.columns: agg["word_weird"] = "mean"

    if not agg:
        # Return an empty placeholder preserving the index
        idx = df.set_index(["file", "page_num"]).index.unique()
        return pd.DataFrame(index=idx)

    # Compute means based on the provided aggregate dict
    means = scored.groupby(["file", "page_num"])[list(agg.keys())].mean()

    # Rename for final output
    rename = {}
    if "quality_score" in means.columns: rename["quality_score"] = "avg_quality_score"
    if "word_weird" in means.columns: rename["word_weird"] = "avg_word_weird"

    return means.rename(columns=rename)


def _prevailing_lang(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the single most prevailing language per page.
    Filters out lines categorized as "N/A" or "unknown" to prevent OCR noise
    from skewing the document's true language profile.
    """
    # Filter out empty/broken rows that lack language identifiers
    valid_langs = df[~df['lang'].isin(['N/A', 'unknown'])].copy()

    if valid_langs.empty:
        # If the page was entirely empty/symbols, default to unknown
        idx = df.set_index(["file", "page_num"]).index.unique()
        return pd.DataFrame({'main_lang': 'unknown'}, index=idx)

    # Group by page, use .mode() to find the most frequent language string.
    # .iloc[0] extracts the top result if there's a tie.
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

    # Combine frames based on their shared index (file, page_num)
    stats = counts.join(means, how="left").join(langs, how="left")

    # Round floats to keep the CSV clean
    for col in ("avg_quality_score", "avg_word_weird"):
        if col in stats.columns:
            stats[col] = stats[col].round(4)

    stats.reset_index(inplace=True)
    return stats


def process_csv_file(args):
    """
    Worker function mapping. Reads a document's CSV, computes its statistics,
    and writes the individual summary file.

    Args:
        args (tuple): (csv_file_path, output_doc_directory)
    """
    csv_file, output_doc_dir = args
    try:
        df = pd.read_csv(csv_file)
        if df.empty or "categ" not in df.columns:
            return None

        # Build the table
        stats = _build_page_stats(df)

        # Save specific summary
        stats_out_path = Path(output_doc_dir) / f"stats_{csv_file.name}"
        stats.to_csv(stats_out_path, index=False)

        # Return the frame so the main thread can consolidate everything globally
        return stats
    except Exception as exc:
        print(f"Error processing file {csv_file.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    # Setup Paths via Config
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

    # Grab all CSV files inside the targeted directory
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        sys.exit(0)

    print(f"Aggregating {len(csv_files)} documents using Multiprocessing...")
    all_page_stats = []

    # Allocate workers. (Aggregation is CPU light, so we can maximize thread count).
    max_cores = min(multiprocessing.cpu_count(), 12)

    with ProcessPoolExecutor(max_workers=max_cores) as executor:
        # Create execution argument tuples
        tasks = [(f, OUTPUT_DOC_DIR) for f in csv_files]

        # Launch pool
        futures = {executor.submit(process_csv_file, t): t for t in tasks}

        # Retrieve results
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                # Store the completed DataFrame
                all_page_stats.append(result)

            # Simple UI Progress Tracker
            if i % 50 == 0:
                print(f"Aggregated {i}/{len(csv_files)} documents...")

    # Consolidate memory into one massive CSV summary file
    if all_page_stats:
        print("Consolidating final page stats ...")
        final_df = pd.concat(all_page_stats, ignore_index=True)
        final_df.sort_values(by=["file", "page_num"], inplace=True)

        # Output the master file
        final_df.to_csv(OUTPUT_STATS, index=False)
        print(f"Done. Global stats saved to {OUTPUT_STATS} (Includes 'main_lang' column)")
    else:
        print("No statistics were generated.")


if __name__ == "__main__":
    main()