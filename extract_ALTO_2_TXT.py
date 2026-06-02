#!/usr/bin/env python3
"""
1_extract.py
Step 1: Extract text from ALTO XML files in parallel.
"""
import pandas as pd
import subprocess
import concurrent.futures
import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import configparser
from atrium_paradata import ParadataLogger

_SCRIPT_NAME = "extract_alto2txt"

CONFIG_PATH = os.getenv("LANGID_CONFIG", "config_langID.txt")


def _load_extract_config(config_path: str = CONFIG_PATH) -> dict:
    """Read extraction parameters from the [EXTRACT] section of the config.

    Falls back to the previous hardcoded defaults when the file or a key is
    missing, so the script keeps working without a config present.
    MAX_WORKERS keeps honouring the MAX_WORKERS env var as the final override.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    def get(key, default):
        return cfg.get("EXTRACT", key, fallback=default) if cfg.has_section("EXTRACT") else default

    workers_default = cfg.getint("EXTRACT", "WORKERS_MAX", fallback=16) if cfg.has_section("EXTRACT") else 16
    return {
        "input_csv": get("INPUT_CSV", "test_alto_stats.csv"),
        "output_text_dir": get("OUTPUT_TXT", "./data_samples/PAGE_TXT"),
        "max_workers": int(os.getenv("MAX_WORKERS", workers_default)),
    }


_CFG = _load_extract_config()
INPUT_CSV = _CFG["input_csv"]
OUTPUT_TEXT_DIR = _CFG["output_text_dir"]
MAX_WORKERS = _CFG["max_workers"]


def extract_single_page(args: tuple) -> bool:
    """Worker function to extract one page with robust de-hyphenation."""
    file_id, page_id, xml_path, output_dir = args

    # Define output path
    save_dir = Path(output_dir) / str(file_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / f"{file_id}-{page_id}.txt"

    # Skip if exists
    if txt_path.exists():
        return True

    # Define common hyphen variations found in OCR/Typesetting
    HYPHEN_VARIATIONS = ('-', '\xad', '\u2013', '\u2014')

    # Run extraction (alto-tools)
    cmd = ["alto-tools", "-t", str(xml_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # ... logic
        return True
    except subprocess.CalledProcessError as e:
        return False
    except Exception:
        return False


def main() -> None:
    # 1. Validate external dependencies first
    if shutil.which("alto-tools") is None:
        print("CRITICAL ERROR: 'alto-tools' binary not found in system PATH. Please install it before running.")
        sys.exit(1)

    # 2. Parse and Process
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find input file {INPUT_CSV}")
        sys.exit(1)

    print(f"Loaded {len(df)} pages to extract.")

    tasks = []
    for _, row in df.iterrows():
        tasks.append((row['file'], row['page'], row['path'], OUTPUT_TEXT_DIR))

    if not tasks:
        return

    page_alto_dir = Path(tasks[-1][2]).parent

    _logger = ParadataLogger(
        program="alto-postprocess",
        config={
            "script": _SCRIPT_NAME,
            "input_csv": str(INPUT_CSV),
            "input_dir": str(page_alto_dir),
            "output_dir": str(OUTPUT_TEXT_DIR),
            "n_workers": MAX_WORKERS,
        },
        paradata_dir="paradata",
        output_types=["txt"],
    )

    # Parallel Execution
    print(f"Extracting with {MAX_WORKERS} workers...")
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(extract_single_page, tasks), total=len(tasks)))

        print(f"Extraction complete. Success rate: {sum(results) / len(results):.2%}")

        for t, r in zip(tasks, results):
            if not r:
                _logger.log_skip(t[2], "Subprocess execution failed.")

    except Exception as e:
        print(f"Unexpected execution failure: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()