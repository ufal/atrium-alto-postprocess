#!/usr/bin/env python3
"""
text_stats_create.py — Step 2 of the text-lines method (#31): page statistics CSV.

Scans the per-page text files text_split.py wrote (`<dir>/<doc_id>/<doc_id>-<n>.txt`,
the root of `<dir>` and one level of subdirectories, like alto_stats_create.py and
json_stats_create.py) and writes the same 7-column page CSV those two produce:

    file, page, textlines, illustrations, graphics, strings, path

so extract_TEXT_2_TXT.py, classify_TEXT.py and aggregate_STAT.py consume it unchanged.
`textlines` counts non-blank lines, `strings` whitespace-separated tokens, and
`illustrations` the image objects text_split recorded per PDF page (pages_report.csv);
`graphics` has no text-format equivalent and is 0.

Hyphen-safe ids: `file` is the parent directory's name whenever the file is named
`<parent>-<digits>.txt` — the layout text_split writes — so a document called
`my-doc.pdf` stays `my-doc` (the ALTO/JSON stats scripts split on every "-" and would
report file `my`, page `doc`). Other names fall back to the LAST "-". A file with no
numeric page suffix cannot be addressed by classify_TEXT and is skipped with a warning.
Hand-written page text files work too (`--skip-split`), in any encoding: the extract
stage decodes them (with [TEXT_INGEST].FALLBACK_ENCODINGS, as this stage counts them).
A damaged pages_report.csv or an unreadable subdirectory costs its counts or its files
(reported), never the stage.

Usage:
    python text_stats_create.py <input_folder> [-o <output_csv>]
"""

from __future__ import annotations

import argparse
import configparser
import csv
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from atrium_paradata import ParadataLogger
from page_split import doc_page_from_path
from text_formats import IngestError, ReaderOptions, decode_bytes, load_settings

CONFIG_PATH = os.getenv("LANGID_CONFIG", os.path.join("setup", "config.txt"))
COLUMNS = ["file", "page", "textlines", "illustrations", "graphics", "strings", "path"]
PAGES_REPORT = "pages_report.csv"


def file_page_from_path(path: str) -> Tuple[str, str]:
    """(file, page) for one page text file; page is "" when there is no numeric suffix.
    The split stages' one derivation (page_split.doc_page_from_path), digits only."""
    return doc_page_from_path(path, numeric=True)


def _page_images(folder: str) -> Dict[Tuple[str, str], int]:
    """{(file, page): images} from text_split's pages_report.csv, when present."""
    path = os.path.join(folder, PAGES_REPORT)
    out: Dict[Tuple[str, str], int] = {}
    if not os.path.isfile(path):
        return out
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    out[(row["file"], str(row["page"]))] = int(row.get("images") or 0)
                except (KeyError, ValueError):
                    continue
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        print(f"WARNING: ignoring unreadable {path} ({exc}); illustrations are counted as 0", file=sys.stderr)
        return {}
    return out


def _process_single_txt(
    path: str, images: Dict[Tuple[str, str], int], fallbacks: Sequence[str] = ReaderOptions.fallback_encodings
) -> Tuple[Optional[Dict], Optional[str]]:
    """(row, None) for a usable page file, (None, reason) otherwise."""
    file_id, page = file_page_from_path(path)
    if not page:
        return None, "no numeric page suffix (expected <doc_id>-<n>.txt)"
    try:
        with open(path, "rb") as fh:
            text, _enc, _flags = decode_bytes(fh.read(), fallbacks)
    except (OSError, IngestError) as exc:
        return None, f"unreadable ({exc})"
    lines = text.splitlines()
    return (
        {
            "file": file_id,
            "page": int(page),
            "textlines": sum(1 for ln in lines if ln.strip()),
            "illustrations": images.get((file_id, page), 0),
            "graphics": 0,
            "strings": sum(len(ln.split()) for ln in lines),
            "path": path,
        },
        None,
    )


def _page_files(input_folder: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    """(`*.txt` in the root and in each non-hidden immediate subdirectory, sorted;
    [(subdirectory, reason)] that could not be listed)."""
    found: List[str] = []
    skipped: List[Tuple[str, str]] = []
    with os.scandir(input_folder) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        if entry.name.startswith("."):
            continue  # hidden files and text_split's .tmp-/.old-<doc> directories
        if entry.is_file() and entry.name.lower().endswith(".txt"):
            found.append(entry.path)
        elif entry.is_dir(follow_symlinks=False):
            try:
                with os.scandir(entry.path) as sub:
                    for f in sorted(sub, key=lambda e: e.name):
                        if f.is_file() and not f.name.startswith(".") and f.name.lower().endswith(".txt"):
                            found.append(f.path)
            except OSError as exc:
                skipped.append((entry.path, f"unreadable directory ({exc})"))
    return found, skipped


def process_text_files(
    input_folder: str, fallbacks: Sequence[str] = ReaderOptions.fallback_encodings
) -> Tuple[List[Dict], List[Tuple[str, str]]]:
    """(rows sorted by file then page, [(path, reason)] skipped)."""
    images = _page_images(input_folder)
    rows = []
    paths, skipped = _page_files(input_folder)
    for path in paths:
        row, reason = _process_single_txt(path, images, fallbacks)
        if row is None:
            skipped.append((path, reason))
        else:
            rows.append(row)
    rows.sort(key=lambda r: (r["file"], r["page"]))
    return rows, skipped


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Step 2 of the text-lines method (#31): build the page statistics CSV "
        "(file,page,textlines,illustrations,graphics,strings,path) from per-page text files.",
    )
    parser.add_argument(
        "input_folder", help="Directory holding <doc_id>/<doc_id>-<n>.txt page files (text_split output)."
    )
    parser.add_argument("-o", "--output", default="text_stats.csv", help="Output CSV path (default: text_stats.csv).")
    args = parser.parse_args(argv)

    if not os.path.isdir(args.input_folder):
        print(f"Error: input folder not found: {args.input_folder}", file=sys.stderr)
        return 1
    cfg = configparser.ConfigParser(inline_comment_prefixes=None)
    cfg.read(CONFIG_PATH, encoding="utf-8")
    try:
        _limits, options = load_settings(cfg)
    except ValueError as exc:
        print(f"Error: invalid configuration in {CONFIG_PATH}: {exc}", file=sys.stderr)
        return 2

    logger = ParadataLogger(
        program="alto-postprocess",
        config={"script": "text_stats_create", "input_folder": str(args.input_folder), "output": str(args.output)},
        paradata_dir="paradata",
        output_types=["csv"],
        config_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup"),
    )
    rows: List[Dict] = []
    skipped: List[Tuple[str, str]] = []
    try:
        try:
            rows, skipped = process_text_files(args.input_folder, options.fallback_encodings)
        except OSError as exc:
            print(f"Error: cannot list {args.input_folder}: {exc}", file=sys.stderr)
            return 1
        for path, reason in skipped:
            logger.log_skip(path, reason)
            print(f"  skipped {path}: {reason}", file=sys.stderr)
        try:
            out_dir = os.path.dirname(os.path.abspath(args.output))
            os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            print(f"Error: cannot write {args.output}: {exc}", file=sys.stderr)
            return 1
        logger.log_success("csv", count=len(rows))
        print(f"Wrote {len(rows)} page rows to {args.output} ({len(skipped)} file(s) skipped).")
    finally:
        logger.finalize(input_total=len(rows) + len(skipped), processed_total=len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
