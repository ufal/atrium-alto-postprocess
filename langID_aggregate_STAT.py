#!/usr/bin/env python3
"""
3_aggregate.py
Step 3: Aggregate raw lines into page statistics.
Reads a directory of per-document CSVs and compiles final stats.
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import configparser
import sys
from atrium_paradata import ParadataLogger

def main():
    # Initialize the parser
    config = configparser.ConfigParser()
    config.read('config_langID.txt')

    _cfg_p = config
    _logger = ParadataLogger(
        program="alto-postprocess",
        config={
            "script": "langID_aggregate_STAT",
            "input_dir": _cfg_p.get("AGGREGATE", "RAW_LINES_CSV", fallback=""),
            "output_csv": _cfg_p.get("AGGREGATE", "OUTPUT_STATS", fallback="final_page_stats.csv"),
            "per_doc_output_dir": _cfg_p.get("AGGREGATE", "OUTPUT_DOC_DIR", fallback="")
        },
        paradata_dir="paradata",
        output_types=["csv"],
    )
    _total_inputs = 0

    # This is now treated as a Directory containing {file_id}.csv files
    INPUT_DIR_PATH = config.get('AGGREGATE', 'RAW_LINES_CSV')
    OUTPUT_STATS = config.get('AGGREGATE', 'OUTPUT_STATS')
    OUTPUT_DOC_DIR = config.get('AGGREGATE', 'OUTPUT_DOC_DIR')

    # 1. Setup paths
    input_dir = Path(INPUT_DIR_PATH)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    Path(OUTPUT_DOC_DIR).mkdir(parents=True, exist_ok=True)

    print(f"Reading CSV files from {input_dir}...")

    # Get list of all csv files in the input directory
    csv_files = list(input_dir.glob("*.csv"))

    if not csv_files:
        print("No CSV files found in the directory.")
        sys.exit(0)

    all_page_stats = []

    try:
        # 2. Iterate through each document file
        for csv_file in tqdm(csv_files):
            _total_inputs += 1

            try:
                # Read individual document CSV
                # Expected columns: "file", "page_num", "line_num", "text", "lang", "lang_score", "perplex", "categ"
                df = pd.read_csv(csv_file)

                # Check for empty dataframe or missing columns
                if df.empty or 'categ' not in df.columns:
                    continue


                # 3. Aggregate Stats for this document
                # Group by file and page_num, then count the categories
                stats = df.groupby(['file', 'page_num'])['categ'].value_counts().unstack(fill_value=0)

                # Ensure standard columns exist even if count is 0
                standard_cols = ["Clear", "Trash", "Noisy", "Empty", "Non-text"]
                for col in standard_cols:
                    if col not in stats.columns:
                        stats[col] = 0

                # Reorder columns for consistency
                stats = stats[standard_cols]
                stats.reset_index(inplace=True)

                # 4. Save per-document STATS (Optional but useful replacement for the old logic)
                # This saves stats_docName.csv to OUTPUT_DOC_DIR
                stats_out_path = Path(OUTPUT_DOC_DIR) / f"stats_{csv_file.name}"
                stats.to_csv(stats_out_path, index=False)

                # Collect for the final summary
                all_page_stats.append(stats)


            except Exception as e:
                print(f"Error processing file {csv_file.name}: {e}")
                _logger.log_skip(csv_file, e)

        # 5. Consolidate all page stats
        if all_page_stats:
            print("Consolidating final page stats...")
            final_df = pd.concat(all_page_stats, ignore_index=True)

            final_df.sort_values(by=['file', 'page_num'], inplace=True)

            # Save global stats
            final_df.to_csv(OUTPUT_STATS, index=False)
            print(f"Done. Global stats saved to {OUTPUT_STATS}")
            print(f"Per-document stats saved to {OUTPUT_DOC_DIR}")
        else:
            print("No statistics were generated.")
    finally:
        _logger.finalize(_total_inputs)


if __name__ == "__main__":
    main()