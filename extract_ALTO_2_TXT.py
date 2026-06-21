#!/usr/bin/env python3
"""
extract_ALTO_2_TXT.py
Step 3 (alto-tools method): Extract text from ALTO XML files in parallel.

Uses the `alto-tools -t` CPU extractor. Output text lines are written verbatim
except for end-of-line hyphenation, which is repaired by joining a word split
across two lines back into its full form.

History / fixes
---------------
* (#1) extract_single_page previously ran alto-tools but never wrote the result;
  it now captures stdout, de-hyphenates, and writes the .txt file.
* (#2) main() now wraps execution in try/finally, records every produced file via
  log_success("txt"), logs failures via log_skip, and always finalize()s so the
  alto-tools stage emits a paradata JSON like the other extraction methods.
"""

import concurrent.futures
import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from atrium_paradata import ParadataLogger

_SCRIPT_NAME = "extract_alto2txt"

CONFIG_PATH = os.getenv("LANGID_CONFIG", "config_langID.txt")

# Common hyphen variations found in OCR/typesetting at a line break.
HYPHEN_VARIATIONS = ("-", "\xad", "\u2013", "\u2014")


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


def _dehyphenate(text: str) -> str:
    """Join words split by a trailing hyphen at a line break into their full form.

    A line whose last non-space character is one of HYPHEN_VARIATIONS is merged
    with the following line: the hyphen is dropped and the two fragments are
    concatenated with no space. Lines without a trailing hyphen keep their break.
    """
    raw_lines = text.splitlines()
    out_lines: list[str] = []
    carry = ""
    for line in raw_lines:
        stripped = line.rstrip()
        if stripped and stripped[-1] in HYPHEN_VARIATIONS:
            # Drop the hyphen and hold the fragment to fuse with the next line.
            carry += stripped[:-1]
            continue
        out_lines.append(carry + line)
        carry = ""
    if carry:
        out_lines.append(carry)
    return "\n".join(out_lines).strip() + "\n"


def extract_single_page(args: tuple) -> bool:
    """Worker: extract one page with robust de-hyphenation. Returns success."""
    file_id, page_id, xml_path, output_dir = args

    save_dir = Path(output_dir) / str(file_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / f"{file_id}-{page_id}.txt"

    # Resume support: skip pages already extracted.
    if txt_path.exists():
        return True

    # Run extraction (alto-tools); -t prints the page text to stdout.
    cmd = ["alto-tools", "-t", str(xml_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False

    # (#1) Persist the result — previously the output was discarded.
    page_text = _dehyphenate(result.stdout or "")
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(page_text)
    except OSError:
        return False
    return True


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
        tasks.append((row["file"], row["page"], row["path"], OUTPUT_TEXT_DIR))

    if not tasks:
        print("No pages to extract.")
        return

    page_alto_dir = Path(tasks[-1][2]).parent

    _logger = ParadataLogger(
        program="alto-postprocess",
        config={
            "script": "extract_ALTO_2_TXT",
            "method": "alto-tools",
            "input_csv": str(INPUT_CSV),
            "input_dir": str(page_alto_dir),
            "output_dir": str(OUTPUT_TEXT_DIR),
            "n_workers": MAX_WORKERS,
        },
        paradata_dir="paradata",
        output_types=["txt"],
    )
    # alto_tools is already seeded as an "always" component, so its Apache-2.0
    # license is recorded automatically; FastText (CC BY-NC 4.0, also "always")
    # keeps the effective license at the project baseline. No explicit
    # log_component call is needed for the alto-tools method.
    _total_inputs = len(tasks)

    # (#2) Always finalize, and record per-file successes/skips.
    try:
        print(f"Extracting with {MAX_WORKERS} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = list(tqdm(executor.map(extract_single_page, tasks), total=len(tasks)))

        if results:
            print(f"Extraction complete. Success rate: {sum(results) / len(results):.2%}")

        for t, r in zip(tasks, results, strict=True):
            if r:
                _logger.log_success("txt")
            else:
                _logger.log_skip(t[2], "alto-tools extraction failed")
        print("Done.")
    finally:
        _logger.finalize(input_total=_total_inputs)


if __name__ == "__main__":
    main()
