#!/usr/bin/env python3
"""
extract_TEXT_2_TXT.py — Step 3 of the text-lines method (#31): classify-ready page text.

The 5th extraction method (`run_pipeline.py --method text-lines`), CSV-driven like the
ALTO and JSON extractors: it reads the page statistics CSV (`file,page,path`, built by
text_stats_create.py from text_split.py's output) and writes, per page,

    <OUTPUT_TXT_TEXT>/<file>/<file>-<page>.txt     what classify_TEXT.py reads

after the one policy the categorizer needs (text_formats.shape_lines): lines are
normalized (NFC, control/zero-width characters removed, soft hyphens resolved), blank
lines dropped unless [TEXT_INGEST].KEEP_BLANK_LINES, and any line longer than
[TEXT_INGEST].MAX_LINE_CHARS wrapped at a word boundary — a single 30,000-character
paragraph would otherwise pad a whole perplexity batch to the model's full context.

It also writes one **line table** per document,

    <OUTPUT_LINES_TEXT>/<file>.csv     file,page_num,line_num,text,page_label

— "the ordered text lines of the input as CSV rows", before any categorization. It is
built from the very list written to the page file, so its (page_num, line_num) are the
numbers classify_TEXT.py will assign to the same lines (it enumerates `readlines()`
from 1 per page); the columns are the key columns of DOC_LINE_CATEG, so nlp-enrich and
llm-enrich can read it too. Page files in any encoding are accepted (hand-made input
with --skip-split): each is decoded through text_formats.decode_bytes.

With [DOCUMENT].JSON_DIR set, pages/content are accreted into `<file>.document.json`
exactly like the other extractors (engine "text-lines"); for a born-digital record the
document_hook guard keeps them out (see docs/text_inputs.md).

Usage:
    python extract_TEXT_2_TXT.py [--input-csv CSV] [--output-dir DIR] [--lines-dir DIR]
"""

from __future__ import annotations

import argparse
import configparser
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import document_hook
import text_formats
from atrium_paradata import ParadataLogger

CONFIG_PATH = os.getenv("LANGID_CONFIG", "setup/config.txt")
LINE_TABLE_COLUMNS = ["file", "page_num", "line_num", "text", "page_label"]
DEFAULT_OUTPUT_DIR = "./data_samples/PAGE_TXT_TEXT"
DEFAULT_LINES_DIR = "./data_samples/DOC_LINES_TEXT"


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3 of the text-lines method (#31): write classify-ready page text and a per-document "
        "line table (file,page_num,line_num,text,page_label) from the page statistics CSV."
    )
    parser.add_argument("--input-csv", default=None, help="Page statistics CSV (default: [EXTRACT].INPUT_CSV).")
    parser.add_argument(
        "--output-dir",
        default=None,
        help=f"Page text output dir (default: [EXTRACT].OUTPUT_TXT_TEXT, {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--lines-dir",
        default=None,
        help=f"Line table output dir (default: [EXTRACT].OUTPUT_LINES_TEXT, {DEFAULT_LINES_DIR}).",
    )
    return parser.parse_args(argv)


def _same_dir(a: str, b: str) -> bool:
    return bool(a and b) and os.path.abspath(a) == os.path.abspath(b)


def read_page_lines(path: str, fallbacks) -> List[str]:
    """Decoded, newline-normalized lines of one page text file (form feeds become breaks)."""
    with open(path, "rb") as fh:
        data = fh.read()
    if not data:
        return []
    text, _enc, _flags = text_formats.decode_bytes(data, fallbacks)
    text = text_formats.normalize_newlines(text).replace("\f", "\n")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _page_labels(page_csv_dir: str) -> Dict[Tuple[str, str], str]:
    """{(file, page): original label} from text_split's pages_report.csv, if it is there."""
    path = os.path.join(page_csv_dir, "pages_report.csv")
    labels: Dict[Tuple[str, str], str] = {}
    if not os.path.isfile(path):
        return labels
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            labels[(row.get("file", ""), str(row.get("page", "")))] = row.get("page_label", "")
    return labels


def main(argv: Optional[list] = None) -> int:
    args = _parse_args(argv)
    cfg = configparser.ConfigParser(inline_comment_prefixes=None)
    cfg.read(CONFIG_PATH, encoding="utf-8")
    try:
        _limits, options = text_formats.load_settings(cfg)
    except ValueError as exc:
        print(f"CRITICAL ERROR: invalid configuration in {CONFIG_PATH}: {exc}", file=sys.stderr)
        return 2

    input_csv = args.input_csv or cfg.get("EXTRACT", "INPUT_CSV", fallback="./data_samples/test_alto_stats.csv")
    output_dir = args.output_dir or cfg.get("EXTRACT", "OUTPUT_TXT_TEXT", fallback=DEFAULT_OUTPUT_DIR)
    lines_dir = args.lines_dir or cfg.get("EXTRACT", "OUTPUT_LINES_TEXT", fallback=DEFAULT_LINES_DIR)

    # classify_TEXT.py skips any document whose <OUTPUT_LINES_LOG>/<file>.csv already
    # exists (resume), and aggregate_STAT.py globs every *.csv there: a line table
    # written into that directory would make both treat it as categorized output.
    for section, key in (("CLASSIFY", "OUTPUT_LINES_LOG"), ("AGGREGATE", "RAW_LINES_CSV")):
        if _same_dir(lines_dir, cfg.get(section, key, fallback="")):
            print(
                f"CRITICAL ERROR: [EXTRACT].OUTPUT_LINES_TEXT ({lines_dir}) is the same directory as "
                f"[{section}].{key}; line tables there would be mistaken for categorized output.",
                file=sys.stderr,
            )
            return 2

    # Imported here, not at module level: classify_TEXT pulls in pandas/numpy, and
    # `--help` should not pay for that.
    from classify_TEXT import load_page_index

    try:
        df = load_page_index(input_csv)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Could not find input file {input_csv}", file=sys.stderr)
        return 1
    missing = {"file", "page", "path"} - set(df.columns)
    if missing:
        print(f"CRITICAL ERROR: {input_csv} lacks column(s) {', '.join(sorted(missing))}", file=sys.stderr)
        return 1
    df = df.dropna(subset=["file", "page", "path"])
    print(f"Loaded {len(df)} pages to extract.")
    if df.empty:
        print("No pages to extract.")
        return 0

    tasks = []
    for _, row in df.iterrows():
        try:
            page = int(row["page"])
        except (TypeError, ValueError):
            continue
        tasks.append((str(row["file"]), page, str(row["path"])))

    labels = _page_labels(str(Path(tasks[0][2]).parent.parent)) if tasks else {}
    _logger = ParadataLogger(
        program=document_hook.PROGRAM_NAME,
        config={
            "script": "extract_TEXT_2_TXT",
            "method": "text-lines",
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "lines_dir": str(lines_dir),
            "max_line_chars": options.max_line_chars,
            "keep_blank_lines": options.keep_blank_lines,
        },
        paradata_dir="paradata",
        output_types=["txt", "csv"],
        config_dir=str(Path(__file__).resolve().parent / "setup"),
    )
    # No log_component(): stdlib + text_formats' pure-Python normalization only.

    doc_json_dir = document_hook.resolve_document_json_dir(cfg.get("DOCUMENT", "JSON_DIR", fallback=""))
    doc_ref = document_hook.paradata_ref_for(_logger)
    ok = failed = 0
    table_rows: Dict[str, List[Dict[str, object]]] = {}
    try:
        for file_id, page, path in tasks:
            out_path = Path(output_dir) / file_id / f"{file_id}-{page}.txt"
            try:
                raw = read_page_lines(path, options.fallback_encodings)
                lines = text_formats.shape_lines(raw, options.max_line_chars, options.keep_blank_lines)
                while lines and not lines[-1]:
                    lines.pop()  # a trailing blank line is invisible to readlines(): keep the table in step
                out_path.parent.mkdir(parents=True, exist_ok=True)
                # No final newline, like the other extractors: document_hook joins pages
                # into content.text verbatim, and classify's readlines() does not need it.
                with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write("\n".join(lines))
            except (OSError, text_formats.IngestError) as exc:
                failed += 1
                _logger.log_skip(path, f"text-lines extraction failed: {exc}")
                print(f"  failed {path}: {exc}", file=sys.stderr)
                continue
            ok += 1
            _logger.log_success("txt")
            label = labels.get((file_id, str(page)), str(page))
            rows = table_rows.setdefault(file_id, [])
            rows.extend(
                {"file": file_id, "page_num": page, "line_num": n, "text": text, "page_label": label}
                for n, text in enumerate(lines, 1)
            )

        os.makedirs(lines_dir, exist_ok=True)
        for file_id, rows in table_rows.items():
            rows.sort(key=lambda r: (r["page_num"], r["line_num"]))
            with open(os.path.join(lines_dir, f"{file_id}.csv"), "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=LINE_TABLE_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            _logger.log_success("csv")

        for doc_id, page_ids in document_hook.group_tasks_by_doc(tasks).items():
            pages, content = document_hook.pages_and_content_from_text(
                output_dir, doc_id, page_ids, engine="text-lines"
            )
            document_hook.write_document_block(
                doc_json_dir,
                doc_id,
                _logger.run_id,
                doc_ref,
                merge_blocks={"pages": pages} if pages else None,
                set_blocks={"content": content} if pages else None,
            )
        total = ok + failed
        print(f"Extraction complete. Success rate: {ok / total:.2%}" if total else "Nothing extracted.")
    finally:
        _logger.finalize(input_total=len(tasks), processed_total=ok)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
