#!/usr/bin/env python3
"""
extract_JSON_2_TXT.py
Step 3 (json-keys method): Extract text from generic JSON OCR-engine output
in parallel, CSV-driven like the ALTO extractors.

Walks a whitelist of informative keys (TARGET_KEYS) and yields every string
leaf whose parent key matches, in document order. This makes no assumption
about a particular OCR engine's JSON schema beyond "text lives under a
key named roughly 'text'/'line'/'word'/etc." — see TARGET_KEYS below.
"""

import argparse
import concurrent.futures
import configparser
import json
import os
from pathlib import Path
from typing import Any, Generator, Optional, Set

import pandas as pd
from tqdm import tqdm

import document_hook
from atrium_paradata import ParadataLogger

CONFIG_PATH = os.getenv("LANGID_CONFIG", "setup/config.txt")

# Whitelist of informative text keys
TARGET_KEYS: Set[str] = {
    "content",
    "text",
    "string",
    "textline",
    "line",
    "word",
    "lines",
    "words",
    "strings",
    "textlines",
    "textstring",
    "textstrings",
    "contents",
    "data",
    "texts",
    "pagetext",
    "page_text",
    "text_string",
    "text_line",
    "text_strings",
    "page_texts",
    "text_lines",
}


def _yield_json_text_by_keys(data: Any, target_keys: Set[str], current_key: str = None) -> Generator[str, None, None]:
    """Recursively yields text from string leaves if their parent key is in the target_keys."""
    if isinstance(data, dict):
        for k, v in data.items():
            yield from _yield_json_text_by_keys(v, target_keys, current_key=k)
    elif isinstance(data, list):
        for item in data:
            yield from _yield_json_text_by_keys(item, target_keys, current_key=current_key)
    elif isinstance(data, str):
        text = data.strip()
        if text and current_key and current_key.lower() in target_keys:
            yield text


def process_json_to_txt(input_file: Path, output_file: Path, join_char: str = "\n") -> None:
    """Reads a JSON file and writes its ordered plain text to the output file."""
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    extracted_leaves = list(_yield_json_text_by_keys(data, TARGET_KEYS))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(join_char.join(extracted_leaves))


def _load_extract_config(config_path: str = CONFIG_PATH) -> dict:
    """Read extraction parameters from the [EXTRACT] section of the config.

    Mirrors extract_ALTO_2_TXT.py's _load_extract_config so all extractors
    are configured identically.
    """
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")

    def get(key, default):
        return cfg.get("EXTRACT", key, fallback=default) if cfg.has_section("EXTRACT") else default

    workers_default = cfg.getint("EXTRACT", "WORKERS_MAX_JSON", fallback=16) if cfg.has_section("EXTRACT") else 16
    force_single_page_default = (
        cfg.getboolean("EXTRACT", "FORCE_SINGLE_PAGE_JSON", fallback=False) if cfg.has_section("EXTRACT") else False
    )
    return {
        "input_csv": get("INPUT_CSV", "test_alto_stats.csv"),
        "output_text_dir": get("OUTPUT_TXT_JSON", "./data_samples/PAGE_TXT_JSON"),
        "max_workers": int(os.getenv("MAX_WORKERS", workers_default)),
        "force_single_page": force_single_page_default,
    }


_CFG = _load_extract_config()
INPUT_CSV = _CFG["input_csv"]
OUTPUT_TEXT_DIR = _CFG["output_text_dir"]
MAX_WORKERS = _CFG["max_workers"]
FORCE_SINGLE_PAGE_DEFAULT = _CFG["force_single_page"]


def extract_single_page(args: tuple) -> bool:
    """Worker: extract one JSON file's ordered text. Returns success."""
    file_id, page_id, json_path, output_dir = args

    save_dir = Path(output_dir) / str(file_id)
    save_dir.mkdir(parents=True, exist_ok=True)
    txt_path = save_dir / f"{file_id}-{page_id}.txt"

    # Resume support: skip pages already extracted.
    if txt_path.exists():
        return True

    try:
        process_json_to_txt(Path(json_path), txt_path)
    except Exception:
        return False
    return True


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    """CLI surface for extract_JSON_2_TXT.py (issue #37).

    Every existing invocation (bare `python3 extract_JSON_2_TXT.py`, as run_pipeline.py
    calls it) keeps working unchanged: with no flags, `--force-single-page` is left
    unset (None) and main() falls back to `[EXTRACT].FORCE_SINGLE_PAGE_JSON` from
    config.txt, so orchestrated pipeline runs can still opt in via config alone.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract text from generic JSON OCR-engine output (--method json-keys) and, "
            "when [DOCUMENT].JSON_DIR/DOCUMENT_JSON_DIR is configured, accrete it into "
            "each document's <doc_id>.document.json (pages[]/content blocks)."
        )
    )
    parser.add_argument(
        "--force-single-page",
        dest="force_single_page",
        action="store_true",
        default=None,
        help=(
            "Document-assembly policy switch (issue #37 / D4): collapse every source page "
            'extracted for a document into a SINGLE schema-valid pages[] row (page: "1") '
            "whose ocr.source_pages lists the original page labels in concatenation order. "
            "content.text is unaffected either way — it is always the full joined document "
            "text. Overrides [EXTRACT].FORCE_SINGLE_PAGE_JSON in config.txt; if omitted, "
            "that config value (default: false) is used instead."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = _parse_args(argv)
    force_single_page = args.force_single_page if args.force_single_page is not None else FORCE_SINGLE_PAGE_DEFAULT

    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: Could not find input file {INPUT_CSV}")
        raise SystemExit(1) from e

    print(f"Loaded {len(df)} pages to extract.")

    tasks = []
    for _, row in df.iterrows():
        tasks.append((row["file"], row["page"], row["path"], OUTPUT_TEXT_DIR))

    if not tasks:
        print("No pages to extract.")
        return

    input_dir = Path(tasks[-1][2]).parent

    _logger = ParadataLogger(
        program="alto-postprocess",
        config={
            "script": "extract_JSON_2_TXT",
            "method": "json-keys",
            "input_csv": str(INPUT_CSV),
            "input_dir": str(input_dir),
            "output_dir": str(OUTPUT_TEXT_DIR),
            "n_workers": MAX_WORKERS,
            "force_single_page": force_single_page,
        },
        paradata_dir="paradata",
        output_types=["txt"],
        config_dir=str(Path(__file__).resolve().parent / "setup"),
    )
    # No log_component() call needed: stdlib json parsing only, no licensed
    # external component (same as extract_ALTO_2_TXT.py's alto-tools method).
    _total_inputs = len(tasks)

    _doc_cfg = configparser.ConfigParser()
    _doc_cfg.read(CONFIG_PATH)
    _document_json_dir = document_hook.resolve_document_json_dir(_doc_cfg.get("DOCUMENT", "JSON_DIR", fallback=""))
    _doc_paradata_ref = document_hook.paradata_ref_for(_logger)

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
                _logger.log_skip(t[2], "json extraction failed")

        for doc_id, page_ids in document_hook.group_tasks_by_doc(tasks).items():
            pages, content = document_hook.pages_and_content_from_text(
                OUTPUT_TEXT_DIR, doc_id, page_ids, engine="json-keys", force_single_page=force_single_page
            )
            document_hook.write_document_block(
                _document_json_dir,
                doc_id,
                _logger.run_id,
                _doc_paradata_ref,
                merge_blocks={"pages": pages} if pages else None,
                set_blocks={"content": content} if pages else None,
            )
        print("Done.")
    finally:
        _logger.finalize(input_total=_total_inputs)


if __name__ == "__main__":
    main()
