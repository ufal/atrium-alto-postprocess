#!/usr/bin/env python3
"""
langID_aggregate_STAT.py
Step 4.2: Aggregate raw lines into page statistics.

Reads a directory of per-document CSVs produced by the classification step
and compiles final page-level stats, including:
  - counts of each line category  (Clear / Noisy / Trash / Non-text / Empty)
  - avg_quality_score  – mean composite quality score for scored lines
  - avg_word_weird     – mean per-word weirdness ratio for scored lines

"Scored lines" means lines whose categ is not Empty and not Non-text;
those two categories bypass the GPU pipeline and carry no meaningful
quality_score / word_weird values.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import configparser
import sys
from atrium_paradata import ParadataLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_COLS = ["Clear", "Trash", "Noisy", "Empty", "Non-text"]

# Categories that go through the quality-scoring pipeline.
# Empty and Non-text are assigned by the fast CPU pre-filter before any
# quality metrics are computed, so their scores are not meaningful.
SCORED_CATEGS = {"Clear", "Noisy", "Trash"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _category_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group *df* by (file, page_num) and pivot categ value-counts into columns.

    Returns a DataFrame indexed by (file, page_num) with one column per
    STANDARD_COLS entry (missing categories filled with 0).
    """
    counts = (
        df.groupby(["file", "page_num"])["categ"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for col in STANDARD_COLS:
        if col not in counts.columns:
            counts[col] = 0
    return counts[STANDARD_COLS]


def _mean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-(file, page_num) mean of quality_score and word_weird,
    restricted to lines in SCORED_CATEGS.

    Returns a DataFrame with columns avg_quality_score and avg_word_weird,
    indexed by (file, page_num).  Pages that have no scored lines receive NaN.
    """
    scored = df[df["categ"].isin(SCORED_CATEGS)].copy()

    agg: dict[str, str] = {}
    if "quality_score" in scored.columns:
        agg["quality_score"] = "mean"
    if "word_weird" in scored.columns:
        agg["word_weird"] = "mean"

    if not agg:
        # Neither column present – return an empty placeholder
        idx = df.set_index(["file", "page_num"]).index.unique()
        return pd.DataFrame(index=idx)

    means = scored.groupby(["file", "page_num"])[list(agg.keys())].mean()

    rename = {}
    if "quality_score" in means.columns:
        rename["quality_score"] = "avg_quality_score"
    if "word_weird" in means.columns:
        rename["word_weird"] = "avg_word_weird"

    return means.rename(columns=rename)


def _build_page_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine category counts and mean metrics for a single document DataFrame.

    Returns a flat DataFrame (reset index) ready to be written to CSV.
    """
    counts = _category_counts(df)
    means  = _mean_metrics(df)

    stats = counts.join(means, how="left")

    # Round computed averages for readability (4 decimal places)
    for col in ("avg_quality_score", "avg_word_weird"):
        if col in stats.columns:
            stats[col] = stats[col].round(4)

    stats.reset_index(inplace=True)
    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    config = configparser.ConfigParser()
    config.read("config_langID.txt")

    _cfg_p = config
    _logger = ParadataLogger(
        program="alto-postprocess",
        config={
            "script": "langID_aggregate_STAT",
            "input_dir":       _cfg_p.get("AGGREGATE", "RAW_LINES_CSV",  fallback=""),
            "output_csv":      _cfg_p.get("AGGREGATE", "OUTPUT_STATS",   fallback="final_page_stats.csv"),
            "per_doc_output_dir": _cfg_p.get("AGGREGATE", "OUTPUT_DOC_DIR", fallback=""),
        },
        paradata_dir="paradata",
        output_types=["csv"],
    )
    _total_inputs = 0

    INPUT_DIR_PATH = config.get("AGGREGATE", "RAW_LINES_CSV")
    OUTPUT_STATS   = config.get("AGGREGATE", "OUTPUT_STATS")
    OUTPUT_DOC_DIR = config.get("AGGREGATE", "OUTPUT_DOC_DIR")

    # ------------------------------------------------------------------
    # Path setup
    # ------------------------------------------------------------------
    input_dir = Path(INPUT_DIR_PATH)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    Path(OUTPUT_DOC_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV files from {input_dir} ...")

    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in the directory.")
        sys.exit(0)

    all_page_stats: list[pd.DataFrame] = []

    # ------------------------------------------------------------------
    # Per-document aggregation
    # ------------------------------------------------------------------
    try:
        for csv_file in tqdm(csv_files):
            _total_inputs += 1

            try:
                # Expected columns (minimum required):
                #   file, page_num, line_num, text, lang, lang_score,
                #   perplex, categ
                # Optional (added by classify step):
                #   quality_score, word_weird
                df = pd.read_csv(csv_file)

                if df.empty or "categ" not in df.columns:
                    continue

                stats = _build_page_stats(df)

                # Save per-document stats
                stats_out_path = Path(OUTPUT_DOC_DIR) / f"stats_{csv_file.name}"
                stats.to_csv(stats_out_path, index=False)

                all_page_stats.append(stats)

            except Exception as exc:
                print(f"Error processing file {csv_file.name}: {exc}")
                _logger.log_skip(csv_file, exc)

        # ------------------------------------------------------------------
        # Global consolidation
        # ------------------------------------------------------------------
        if all_page_stats:
            print("Consolidating final page stats ...")
            final_df = pd.concat(all_page_stats, ignore_index=True)
            final_df.sort_values(by=["file", "page_num"], inplace=True)

            final_df.to_csv(OUTPUT_STATS, index=False)
            print(f"Done. Global stats saved to {OUTPUT_STATS}")
            print(f"Per-document stats saved to {OUTPUT_DOC_DIR}")
        else:
            print("No statistics were generated.")

    finally:
        _logger.finalize(_total_inputs)


if __name__ == "__main__":
    main()