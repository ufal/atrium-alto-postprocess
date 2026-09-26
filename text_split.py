#!/usr/bin/env python3
"""
text_split.py — Step 1 of the text-lines method (#31): any text-bearing file → pages.

The `<format>-2-txt` step for every input that is not ALTO XML or the generic JSON of
the json-keys method: PDF, DOCX, ODT, XLSX/ODS, PPTX/ODP, EPUB, RTF, HTML/hOCR,
PAGE XML, TEI, generic XML, ABBYY FineReader XML, DjVuXML, Tesseract TSV, JSON/JSONL,
CSV/TSV, Markdown, SRT/WebVTT subtitles, EML/MBOX e-mail and plain text in any common
encoding — also gzip/bzip2/xz-compressed, and a ZIP of per-page files as one document.
Formats are recognised by content (text_formats.sniff_kind), so a misnamed file is
still read correctly and an unsupported one is refused with a reason.

For every input file this writes, like page_split.py does for ALTO:

    <output_dir>/<doc_id>/<doc_id>-<n>.txt     one file per page, n = 1..N in reading
                                               order; UTF-8, "\\n" line ends, one text
                                               line per line (blank lines kept as block
                                               separators; the extract stage drops them)

and, once per run:

    <output_dir>/ingest_report.csv   one row per input file: status (ok | partial |
                                     error | ignored), reason code(s), detected kind,
                                     encoding, page/line counts, the source.origin
                                     recorded, notes
    <output_dir>/pages_report.csv    one row per written page: its original label
                                     (PDF page label, sheet name, JSON page number…),
                                     PDF text-layer class and needs-OCR reason

A page with no text is still written (an empty file), so page numbers stay faithful
to the source. Pages that are not native to the format are "blocks" (a sheet, a slide,
a JSON child object, a form-feed section) — see docs/text_inputs.md for the matrix.

Robustness: every file is processed independently and a failure costs that file
only (its row in ingest_report.csv names the reason code); PDFs are read in an
isolated child process with a timeout. A file read with a possible loss (recovered
XML, a skipped bundle member, a bad JSONL record) is `partial`: written and processed
downstream, its reasons listed. Each document's pages are written into
`.tmp-<doc_id>/` and renamed in place of `<doc_id>/` (the old one set aside as
`.old-<doc_id>/` and removed); only a directory that holds nothing but this
document's page files is ever replaced or removed, so a document id can never delete
anything else. When a document fails, its pages from an earlier run are removed
(`stale_pages_removed`), so later stages never read pages its report calls failed.
Exit status is 0 unless --strict (or [TEXT_INGEST].STRICT) is set and a file failed
or was read partially.

Usage:
    python text_split.py <input_dir> <output_dir> [--source-origin ORIGIN] [--strict | --no-strict]
"""

from __future__ import annotations

import argparse
import configparser
import csv
import os
import re
import shutil
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

import document_hook
import text_formats
from atrium_document import resolve_originator
from atrium_paradata import ParadataLogger
from page_split import _doc_id_from_filename, _sha256_of

CONFIG_PATH = os.getenv("LANGID_CONFIG", os.path.join("setup", "config.txt"))

INGEST_REPORT = "ingest_report.csv"
PAGES_REPORT = "pages_report.csv"

INGEST_COLUMNS = [
    "filename", "doc_id", "status", "reason", "kind", "media_type", "encoding", "pages", "lines", "chars",
    "pages_no_text", "pages_garbled", "pages_ocr_layer", "origin", "sha256", "notes",
]  # fmt: skip
PAGES_COLUMNS = ["file", "page", "page_label", "text_layer", "needs_ocr_reason", "lines", "images", "flags"]

#: Office lock files, OS metadata and editor droppings — reported as `ignored`,
#: never read (a lock file's bytes are not the document).
_IGNORED_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}
_MAX_DOC_ID_BYTES = 200


def _ignore_reason(name: str) -> Optional[str]:
    lower = name.lower()
    if lower in _IGNORED_NAMES:
        return "OS metadata file"
    if name.startswith("~$") or name.startswith(".~lock."):
        return "office lock file"
    if name.startswith("._"):
        return "AppleDouble metadata file"
    if name.startswith("."):
        return "hidden file"
    if name in (INGEST_REPORT, PAGES_REPORT):
        return "ingest report"
    return None


def scan_input_dir(input_dir: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """Top-level entries of `input_dir`, sorted: ([(name, path)] to read, [(name, why)] ignored).

    Only regular files are read — never symlinks (they may point outside the input
    tree) and never FIFOs/devices (reading one blocks forever). Subdirectories are
    not descended into, matching page_split.py.
    """
    candidates, ignored = [], []
    with os.scandir(input_dir) as it:
        entries = sorted(it, key=lambda e: e.name)
    for entry in entries:
        why = _ignore_reason(entry.name)
        if why is None:
            if entry.is_symlink():
                why = "symbolic link (not followed)"
            elif entry.is_dir(follow_symlinks=False):
                why = "directory (not scanned)"
            elif not entry.is_file(follow_symlinks=False):
                why = "not a regular file"
        if why is None:
            candidates.append((entry.name, entry.path))
        else:
            ignored.append((entry.name, why))
    return candidates, ignored


def iter_candidate_files(input_dir: str) -> List[Tuple[str, str]]:
    """The (name, path) pairs text_split would read — used by run_pipeline.py too."""
    return scan_input_dir(input_dir)[0]


def validate_doc_id(doc_id: str) -> Optional[str]:
    """None when `doc_id` is usable as a directory/file stem, else why not."""
    if not doc_id or doc_id in (".", ".."):
        return "empty document id"
    if any(unicodedata.category(ch) == "Cc" for ch in doc_id):
        return "control characters in the file name"
    if "/" in doc_id or "\\" in doc_id or os.sep in doc_id:
        return "path separator in the document id"
    if len(doc_id.encode("utf-8")) > _MAX_DOC_ID_BYTES:
        return f"document id longer than {_MAX_DOC_ID_BYTES} bytes"
    return None


def resolve_text_source_origin(
    doc: text_formats.TextDocument,
    configured: str = "",
    override: str = "",
    by_kind: Optional[Dict[str, str]] = None,
) -> str:
    """`source.origin` for one text-lines document (#31, "truthful per class").

    Precedence: CLI flag > DOCUMENT_SOURCE_ORIGIN env var > [DOCUMENT].SOURCE_ORIGIN_BY_KIND
    for the document's kind > [DOCUMENT].SOURCE_ORIGIN > the per-class default
    (document_hook.resolve_input_origin). The default is truthful: OCR-bearing formats
    get an `ocr:`/`ABBYY-ALTO` origin (this repo's to originate), born-digital ones
    `digital-born-<kind>` (llm-enrich's digital-convert's). The document_hook guard then
    keeps this repo's positional writes out of a digital-born record; the CSV outputs
    are produced either way.
    """
    return document_hook.resolve_input_origin(
        doc.kind,
        text_formats.default_source_origin(doc),
        override=override,
        env=os.getenv("DOCUMENT_SOURCE_ORIGIN", ""),
        configured=configured,
        by_kind=by_kind,
    )


def _page_file(out_dir: str, doc_id: str, n: int) -> str:
    return os.path.join(out_dir, f"{doc_id}-{n}.txt")


def _is_page_dir(path: str, doc_id: str) -> bool:
    """True for a real directory holding nothing but `<doc_id>-<n>.txt` regular files —
    the only kind of directory this stage ever replaces or removes."""
    if os.path.islink(path) or not os.path.isdir(path):
        return False
    pattern = re.compile(re.escape(doc_id) + r"-[1-9][0-9]*\.txt")
    with os.scandir(path) as it:
        return all(pattern.fullmatch(e.name) and e.is_file(follow_symlinks=False) for e in it)


def _clear_owned(path: str, doc_id: str) -> None:
    """Remove `path` when it is this document's page directory; refuse anything else."""
    if not os.path.lexists(path):
        return
    if not _is_page_dir(path, doc_id):
        raise text_formats.IngestError(
            "output_failed", f"{path} exists and holds files text_split did not write — not replacing it"
        )
    shutil.rmtree(path)


def _stage_pages(output_dir: str, doc_id: str, doc: text_formats.TextDocument) -> str:
    """Write every page into `.tmp-<doc_id>/` and return it (nothing replaced yet)."""
    final_dir = os.path.join(output_dir, doc_id)
    if os.path.lexists(final_dir) and not _is_page_dir(final_dir, doc_id):
        raise text_formats.IngestError(
            "output_failed", f"{final_dir} exists and holds files text_split did not write — not replacing it"
        )
    tmp_dir = os.path.join(output_dir, f".tmp-{doc_id}")
    _clear_owned(tmp_dir, doc_id)
    os.makedirs(tmp_dir)
    try:
        for n, page in enumerate(doc.pages, 1):
            with open(_page_file(tmp_dir, doc_id, n), "w", encoding="utf-8", newline="\n") as fh:
                if page.lines:
                    fh.write("\n".join(page.lines) + "\n")
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    return tmp_dir


def _keep_unchanged_times(staged: str, final_dir: str) -> None:
    """Give a staged page the file time of the page it replaces when the two are byte for
    byte the same. (#31 Phase 5) The extract and classify stages resume by file time, so
    a re-run that re-reads unchanged inputs must not make every page look new."""
    if not os.path.isdir(final_dir) or os.path.islink(final_dir):
        return
    for name in os.listdir(staged):
        new, old = os.path.join(staged, name), os.path.join(final_dir, name)
        try:
            if os.path.isfile(old) and not os.path.islink(old):
                with open(new, "rb") as a, open(old, "rb") as b:
                    if a.read() != b.read():
                        continue
                st = os.stat(old)
                os.utime(new, ns=(st.st_atime_ns, st.st_mtime_ns))
        except OSError:
            continue


def _swap_in(output_dir: str, doc_id: str, staged: str) -> None:
    """Put the staged pages in place: the old directory is renamed aside first, so a
    failure never leaves a half-replaced one (a POSIX directory swap is not atomic).
    Unchanged pages keep their file times (``_keep_unchanged_times``)."""
    final_dir = os.path.join(output_dir, doc_id)
    old_dir = os.path.join(output_dir, f".old-{doc_id}")
    try:
        _keep_unchanged_times(staged, final_dir)
        _clear_owned(old_dir, doc_id)
        if os.path.lexists(final_dir):
            os.replace(final_dir, old_dir)
        os.replace(staged, final_dir)
    except BaseException:
        shutil.rmtree(staged, ignore_errors=True)
        raise
    shutil.rmtree(old_dir, ignore_errors=True)


def _remove_stale_pages(output_dir: str, doc_id: str) -> bool:
    """Remove this document's page directories from an earlier run (only ones it
    owns); True when pages were removed."""
    removed = False
    for name in (doc_id, f".old-{doc_id}", f".tmp-{doc_id}"):
        path = os.path.join(output_dir, name)
        try:
            if os.path.lexists(path) and _is_page_dir(path, doc_id):
                shutil.rmtree(path)
                removed = removed or name == doc_id
        except OSError:
            continue
    return removed


def _append_note(row: Dict[str, object], note: str) -> None:
    row["notes"] = "; ".join(filter(None, [str(row.get("notes") or ""), note]))


def _write_csv(path: str, columns: List[str], rows: List[Dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1 of the text-lines method (#31): read any text-bearing file (PDF, DOCX, ODT, XLSX, "
        "PPTX, EPUB, RTF, HTML/hOCR, PAGE XML, TEI, XML, JSON/JSONL, CSV/TSV, Markdown, plain text) and write "
        "its ordered pages as <output_dir>/<doc_id>/<doc_id>-<n>.txt, plus ingest_report.csv and pages_report.csv.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_dir", help="Directory of input files (top level only; hidden/lock files are ignored).")
    parser.add_argument("output_dir", help="Directory for the per-page text files and the two reports.")
    parser.add_argument(
        "--source-origin",
        default="",
        help=(
            "Override the source.origin recorded in the document record for EVERY input of this run\n"
            "(default: truthful per format — ocr:<kind>/ABBYY-ALTO for OCR outputs, digital-born-<kind>\n"
            "for born-digital documents; see docs/text_inputs.md). Use ocr:<engine> when the files are\n"
            "known OCR output, e.g. --source-origin ocr:pero. Precedence: this flag > the\n"
            "DOCUMENT_SOURCE_ORIGIN env var > [DOCUMENT].SOURCE_ORIGIN_BY_KIND > [DOCUMENT].SOURCE_ORIGIN."
        ),
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Exit with status 1 if any input file failed or was read partially\n"
        "(default: [TEXT_INGEST].STRICT, else false; --no-strict overrides the config).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    cfg = configparser.ConfigParser(inline_comment_prefixes=None)
    cfg.read(CONFIG_PATH, encoding="utf-8")
    try:
        limits, options = text_formats.load_settings(cfg)
        strict = (
            args.strict if args.strict is not None else text_formats.config_bool(cfg, "TEXT_INGEST", "STRICT", False)
        )
        origin_by_kind = document_hook.parse_origin_by_kind(
            cfg.get("DOCUMENT", "SOURCE_ORIGIN_BY_KIND", fallback=""), text_formats.READERS
        )
    except ValueError as exc:
        print(f"Error: invalid configuration in {CONFIG_PATH}: {exc}", file=sys.stderr)
        return 2
    document_json_dir = document_hook.resolve_document_json_dir(cfg.get("DOCUMENT", "JSON_DIR", fallback=""))
    configured_origin = cfg.get("DOCUMENT", "SOURCE_ORIGIN", fallback="")

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'", file=sys.stderr)
        return 1
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to '{os.path.abspath(args.output_dir)}'\n")

    candidates, ignored = scan_input_dir(args.input_dir)

    _logger = ParadataLogger(
        program=document_hook.PROGRAM_NAME,
        config={
            "script": "text_split",
            "method": "text-lines",
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
            "strict": bool(strict),
            "limits": {k: v for k, v in vars(limits).items()},
            "options": {k: list(v) if isinstance(v, tuple) else v for k, v in vars(options).items()},
            "source_origin_by_kind": origin_by_kind,
        },
        paradata_dir="paradata",
        output_types=["txt", "csv"],
        config_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup"),
    )
    _doc_paradata_ref = document_hook.paradata_ref_for(_logger)

    ingest_rows: List[Dict[str, object]] = [
        {"filename": name, "status": "ignored", "reason": why} for name, why in ignored
    ]
    page_rows: List[Dict[str, object]] = []
    reserved: Dict[str, str] = {}
    docs_ok = failures = partials = 0
    used_pdf = used_detection = False

    try:
        for name, path in candidates:
            doc_id = _doc_id_from_filename(name)
            row: Dict[str, object] = {"filename": name, "doc_id": doc_id}
            ingest_rows.append(row)
            print(f"Processing '{name}'...")
            owned_id = False
            try:
                problem = validate_doc_id(doc_id)
                if problem:
                    raise text_formats.IngestError("doc_id_invalid", problem)
                key = doc_id.casefold()
                if key in reserved:
                    raise text_formats.IngestError(
                        "doc_id_collision", f"{name!r} and {reserved[key]!r} both map to document id {doc_id!r}"
                    )
                reserved[key] = name
                owned_id = True

                doc = text_formats.read_document_isolated(path, limits, options)
                used_pdf |= doc.kind == "pdf"
                used_detection |= "encoding_detected" in doc.notes
                row.update(
                    kind=doc.kind,
                    media_type=doc.media_type,
                    encoding=doc.encoding or "",
                    pages=len(doc.pages),
                    lines=doc.line_count(),
                    chars=doc.char_count(),
                    pages_no_text=sum(1 for p in doc.pages if not any(ln.strip() for ln in p.lines)),
                    pages_garbled=sum(1 for p in doc.pages if p.text_layer == "garbled"),
                    pages_ocr_layer=sum(1 for p in doc.pages if p.text_layer == "ocr"),
                    notes="; ".join(doc.notes),
                )
                if doc.line_count() == 0:
                    raise text_formats.IngestError("no_text", text_formats.no_text_message(doc))

                origin = resolve_text_source_origin(doc, configured_origin, args.source_origin, origin_by_kind)
                sha256 = _sha256_of(path)
                row.update(origin=origin, sha256=sha256)
                staged = _stage_pages(args.output_dir, doc_id, doc)
                source = {"sha256": sha256, "filename": name, "media_type": doc.media_type, "origin": origin}
                if doc.native_pages:
                    # Only for formats whose pages are real pages: a DOCX "page" here is a
                    # break-delimited block, and how many there are depends on PAGE_BREAKS.
                    # llm-enrich counts DOCX pages with the same `auto` rules since its #18,
                    # but the two settings are separate and set_source() keeps the first
                    # writer's value, so the count is left to the plane's originator.
                    source["page_count"] = len(doc.pages)
                try:
                    # The record is written before the pages are swapped in, so a failed
                    # record never leaves pages that later stages would categorize.
                    document_hook.write_document_block(
                        document_json_dir, doc_id, _logger.run_id, _doc_paradata_ref, source=source
                    )
                except BaseException:
                    shutil.rmtree(staged, ignore_errors=True)
                    raise
                _swap_in(args.output_dir, doc_id, staged)
                for n, page in enumerate(doc.pages, 1):
                    page_rows.append(
                        {
                            "file": doc_id,
                            "page": n,
                            "page_label": page.label,
                            "text_layer": page.text_layer or "",
                            "needs_ocr_reason": page.needs_ocr_reason or "",
                            "lines": sum(1 for ln in page.lines if ln.strip()),
                            "images": page.images,
                            "flags": ";".join(page.flags),
                        }
                    )
                if document_json_dir and resolve_originator(origin) not in (None, document_hook.PROGRAM_NAME):
                    note = "born-digital origin: the document record gets source only"
                    if doc.kind not in document_hook.DIGITAL_CONVERT_KINDS:
                        note += (
                            f" (no ecosystem originator writes positional blocks for {doc.kind} yet — "
                            "see [DOCUMENT].SOURCE_ORIGIN_BY_KIND)"
                        )
                    _append_note(row, note)
                lossy = text_formats.lossy_reasons(doc)
                if lossy:
                    partials += 1
                    row.update(status="partial", reason=";".join(lossy))
                else:
                    row.update(status="ok", reason="")
                _logger.log_success("txt", count=len(doc.pages))
                docs_ok += 1
            except text_formats.IngestError as exc:
                failures += 1
                row.update(status="error", reason=exc.code)
                _append_note(row, exc.message)
                if owned_id and _remove_stale_pages(args.output_dir, doc_id):
                    _append_note(row, "stale_pages_removed")
                _logger.log_skip(name, f"{exc.code}: {exc.message}")
                print(f"  -> skipped: {exc.code} — {exc.message}", file=sys.stderr)
            except Exception as exc:  # writing pages or the record failed: this file only
                failures += 1
                row.update(status="error", reason="output_failed")
                _append_note(row, f"{type(exc).__name__}: {exc}")
                if owned_id and _remove_stale_pages(args.output_dir, doc_id):
                    _append_note(row, "stale_pages_removed")
                _logger.log_skip(name, f"output_failed: {exc}")
                print(f"  -> failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        _write_csv(os.path.join(args.output_dir, INGEST_REPORT), INGEST_COLUMNS, ingest_rows)
        _write_csv(os.path.join(args.output_dir, PAGES_REPORT), PAGES_COLUMNS, page_rows)
        _logger.log_success("csv", count=2)
        if used_pdf:
            _logger.log_component("pypdfium2")
        if used_detection:
            _logger.log_component("charset_normalizer")
        _logger.finalize(input_total=len(candidates), processed_total=docs_ok)

    partial_note = f", {partials} partial" if partials else ""
    print(
        f"\nDone: {docs_ok} document(s) split, {failures} failed, {len(ignored)} ignored{partial_note}. "
        f"See {os.path.join(args.output_dir, INGEST_REPORT)}."
    )
    return 1 if (strict and (failures or partials)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
