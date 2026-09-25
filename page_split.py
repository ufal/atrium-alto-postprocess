#!/usr/bin/env python3
"""
page_split.py

Purpose:
This script takes a multi-page document as input and splits it into multiple
single-page files, ordered plain-text-adjacent stage 1 of the pipeline for
BOTH input formats it supports:

- ALTO XML (any version: the namespace is the root's): a document-level
  `<file>.alto.xml` (multiple `<Page>` elements under one `<Layout>`) is split
  into one `<base>-<page>.alto.xml` per page (``split_alto_xml``). Each output
  file keeps the full header and style information from the original file, but
  the <Layout> section only contains the data for a single page.
- Generic JSON OCR/Doc-AI output (#31): a document-level JSON file is split
  into one `<base>-<page>.json` per page (``split_json_document``), detecting
  the page boundary heuristically since OCR/Doc-AI engines don't agree on how
  multi-page documents are represented (see that function's docstring).

A re-split replaces a document's page files: pages an earlier run wrote that the
current input no longer yields are removed (``remove_stale_pages``).

Usage:
    python page_split.py <input_directory> <output_directory>
"""

import argparse
import configparser
import hashlib
import io
import json
import os
import sys
import xml.etree.ElementTree as ET  # For parsing and creating XML
from typing import Iterable, Optional, Tuple

import document_hook
from atrium_document import canonical_doc_id, resolve_originator
from atrium_paradata import ParadataLogger

DOCUMENT_CONFIG_PATH = os.getenv("LANGID_CONFIG", os.path.join("setup", "config.txt"))

#: (atrium-project#10 D5) Default `source.origin` per input format. page_split is the
#: first stage to see the original input, so it is the record's first writer of
#: `source` — and since Issue #18 §1a that field is what AUTHORISES the positional
#: blocks (pages/content/lines/tables). Writing `source` without it left
#: `_assert_origin_consistent()` abstaining-and-deferring forever, i.e. the mixed-plane
#: guard was switched off for every real run of the repo it primarily protects.
#:
#: Each value MUST match an `ORIGIN_ORIGINATORS` prefix that resolves to
#: "alto-postprocess" or the check silently abstains again — that is the failure mode,
#: not a loud one, so `resolve_source_origin()` below checks the resolution rather
#: than trusting the spelling.
#:
#:   xml  — the ALTO path. "ABBYY-ALTO" is the table's only ALTO-specific value, and
#:          this repo's extractors already document ABBYY as the producer of the ALTO
#:          they consume (extract_LytRdr_ALTO_2_TXT.py's header: "Trusts ABBYY's
#:          <TextLine>"). A deployment whose ALTO came out of another engine sets
#:          `SOURCE_ORIGIN = ocr:<engine>` rather than leaving this default in place.
#:   json — the generic OCR/Doc-AI path (#31: Azure DI, docTR, Google Doc AI, AWS
#:          Textract, pero-ocr, OCR.space). Every one of those families is OCR
#:          output, so the `ocr:` prefix is the truthful one; the ENGINE is not
#:          recoverable from the JSON in the general case, hence "generic" — the
#:          same spelling tests/test_extract_json.py already uses for this path.
#:          A deployment that knows its engine sets `SOURCE_ORIGIN = ocr:pero`.
#:
#: `source.origin` describes how the ORIGINAL INPUT was acquired, never how this
#: stage read it — which is why an override belongs to the operator, not the code.
DEFAULT_SOURCE_ORIGIN = {"xml": "ABBYY-ALTO", "json": "ocr:generic"}


def _doc_id_from_filename(filename: str, fmt: str = "") -> str:
    """The document record's key for one input file (atrium-project#10 D3).

    Delegates to the hub's `canonical_doc_id()` — the one derivation every tool is
    meant to share. This used to hand-roll `splitext()` + `.replace(".alto", "")`,
    one of ~11 independent derivations across the five repos, and the class of bug
    that costs is a doc_id FORK: two stages keying the same document differently and
    each writing a record the other never sees.

    `fmt` is retained in the signature (and unused) so existing callers stay
    source-compatible: `canonical_doc_id()` strips both formats' suffixes from
    `KNOWN_PIPELINE_SUFFIXES`, longest first, so the caller no longer has to say
    which one it is holding.
    """
    del fmt  # (D3) no longer needed — canonical_doc_id() is format-agnostic.
    return canonical_doc_id(filename)


def resolve_source_origin(fmt: str, configured: str = "", override: str = "") -> Optional[str]:
    """The `source.origin` to record for one input format (D5).

    Precedence, matching every other setting in this repo: CLI flag > env var >
    config value > the per-format default. Empty strings are "not set", so an unset
    config key falls through rather than blanking the field.

    Returns None when nothing at all resolves — `set_source()` drops None values, so
    the field is then simply absent (the pre-D5 behaviour) rather than present and
    empty, which would look like a deliberate "unknown acquisition".

    An origin that resolves to no originator, or to a DIFFERENT one, is warned about
    loudly: both cases disable the §1a check (the first abstains, the second refuses
    every write this repo makes), and a silently-disabled guard is exactly what D5 is.
    """
    origin = override or os.getenv("DOCUMENT_SOURCE_ORIGIN", "") or configured or DEFAULT_SOURCE_ORIGIN.get(fmt, "")
    origin = origin.strip()
    if not origin:
        return None
    originator = resolve_originator(origin)
    if originator != document_hook.PROGRAM_NAME:
        print(
            f"[document] WARNING – source.origin {origin!r} resolves to {originator!r}, "
            f"not {document_hook.PROGRAM_NAME!r}: the Issue #18 §1a originator check will "
            f"{'abstain' if originator is None else 'REFUSE this repo every pages/lines write'} "
            f"for these documents. Use an ORIGIN_ORIGINATORS prefix owned by this repo "
            f"(ABBYY-ALTO, ocr:<engine>, vlm:<engine>).",
            file=sys.stderr,
        )
    return origin


def _sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


#: The page-file suffixes this stage writes: `<out>/<doc_id>/<doc_id>-<page><suffix>`.
ALTO_PAGE_SUFFIX = ".alto.xml"
JSON_PAGE_SUFFIX = ".json"


def doc_page_from_path(path: str, numeric: bool = False) -> Tuple[str, str]:
    """(doc_id, page) for one page file written by a split stage
    (`<out>/<doc_id>/<doc_id>-<page><suffix>`); page is "" when there is none.

    (#31 Phase 5) The page directory's name is the doc_id, so a doc_id with hyphens
    (`my-doc/my-doc-3.alto.xml` -> `my-doc`, `3`) survives; the old `split("-")` of
    the ALTO/JSON stats scripts gave `my`, `doc`. A file outside such a directory is
    split at its LAST hyphen. `numeric=True` accepts only a digit page (text-lines,
    whose pages are always 1..N); ALTO pages are `PHYSICAL_IMG_NR`s and JSON pages
    the engine's page numbers, so any non-empty label counts for them.
    """
    parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
    base = canonical_doc_id(os.path.basename(path))

    def ok(page: str) -> bool:
        return page.isdigit() if numeric else bool(page)

    if parent and base.startswith(parent + "-") and ok(base[len(parent) + 1 :]):
        return parent, base[len(parent) + 1 :]
    head, sep, tail = base.rpartition("-")
    if sep and head and ok(tail):
        return head, tail
    return base, ""


def page_sort_key(page) -> Tuple[int, int, str]:
    """Reading order of page labels: numeric labels by value (`2` before `10`), then any
    other label as text. (#31 Phase 5) The ALTO/JSON stats scripts sort their rows by it,
    so the extractors — which assemble a document's `content.text` in CSV row order — see
    the pages in order; the rows used to follow directory listing / thread completion
    order."""
    text = str(page).strip()
    return (0, int(text), text) if text.isdigit() else (1, 0, text)


def remove_stale_pages(page_output_dir: str, base_name: str, suffix: str, keep: Iterable[str] = ()) -> int:
    """(#31 Phase 5) Delete the page files an earlier split of this document left in
    its directory, so a re-split with fewer or renumbered pages (or an input that now
    yields none) leaves no old pages for the stats stage to pick up. Only regular files
    named `<doc_id>-<page><suffix>` directly in `<out>/<doc_id>/` are removed, except the
    names in `keep` (the pages this run wrote); other files, sub-directories and symlinks
    are left alone. Returns how many went."""
    keep = set(keep)
    prefix = f"{base_name}-"
    try:
        entries = list(os.scandir(page_output_dir))
    except (FileNotFoundError, NotADirectoryError):
        return 0
    removed = 0
    for entry in entries:
        name = entry.name
        if (
            name not in keep
            and name.startswith(prefix)
            and name.endswith(suffix)
            and len(name) > len(prefix) + len(suffix)
            and entry.is_file(follow_symlinks=False)
        ):
            os.remove(entry.path)
            removed += 1
    return removed


def _assert_no_doctype(input_file_path):
    """Reject any input containing a DOCTYPE declaration.

    (#5) ALTO documents may be untrusted, and legitimate ALTO never carries a
    DOCTYPE. The previous approach attached expat handlers via
    ``ET.XMLParser().parser`` — but the C-accelerated XMLParser exposes no such
    attribute (verified on CPython 3.12), so the ``except AttributeError``
    fallback silently disabled the hardening and internal entities were still
    expanded into the output. Scanning for a DOCTYPE before parsing fails
    closed: no entity declarations can exist without one, which rules out
    entity-expansion ("billion laughs") inputs entirely.
    """
    content = open(input_file_path, "rb").read()
    if b"<!doctype" in content.lower():
        raise ET.ParseError(f"DOCTYPE is not allowed in ALTO inputs: {input_file_path}")


def _alto_namespace(root) -> str:
    """The namespace URI of an ALTO root element (`{uri}alto`), or "" when the root is
    not a namespaced `<alto>` (PAGE XML, TEI, a namespace-less file, ...)."""
    tag = root.tag if isinstance(root.tag, str) else ""
    if not tag.startswith("{"):
        return ""
    uri, _, local = tag[1:].partition("}")
    return uri if local == "alto" else ""


def _local_name(tag) -> str:
    return tag.rpartition("}")[2] if isinstance(tag, str) else str(tag)


def split_alto_xml(input_file_path, output_dir):
    """
    Splits a single multi-page ALTO XML file into single-page files.

    (#31 Phase 5) Every ALTO version is split: the namespace is taken from the root
    element (v2, v3, v4, BnF, CCS ALTO 1), as `service/utils.py` and the LayoutReader
    extractor already do; it used to be hard-coded to v3, so a v2/v4 file printed
    "No <Page> elements found" and got no pages. A root that is not a namespaced
    `<alto>` (PAGE XML, TEI, namespace-less ALTO, which the stats stage's alto_tools
    rejects anyway) is skipped with its root named — the text-lines method reads those.

    Returns:
        int: The number of pages written (0 if no pages were found).
    """
    # --- Parse the Input XML (DOCTYPE rejected up front, see #5) ---
    _assert_no_doctype(input_file_path)
    tree = ET.parse(input_file_path)
    root = tree.getroot()

    ns_uri = _alto_namespace(root)
    if not ns_uri:
        print(
            f"  -> Not a namespaced ALTO document (root <{_local_name(root.tag)}>) in {input_file_path}. "
            f"Skipping; read it with --method text-lines."
        )
        return 0
    namespace = {"alto": ns_uri}
    # Serialise the pages with the document's own namespace as the default one, as
    # the v3-only code always did (so v3 output is byte-identical).
    ET.register_namespace("", ns_uri)

    description = root.find("alto:Description", namespace)
    styles = root.find("alto:Styles", namespace)

    pages = root.findall(".//alto:Page", namespace)

    if not pages:
        print(f"  -> No <Page> elements found in {input_file_path}. Skipping.")
        return 0

    # (#10 D3) Same derivation as the document record's key — literally the same
    # call — because this base name IS the doc_id: it names the per-document output
    # directory every later stage re-derives `file` from (alto_stats_create.py). Two
    # copies of one convention is how the split output and the record key drift apart.
    base_name = _doc_id_from_filename(os.path.basename(input_file_path))

    page_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(page_output_dir, exist_ok=True)
    written = []

    print(f"  -> Found {len(pages)} page(s). Splitting...")
    for i, page in enumerate(pages, 1):
        page_number = page.get("PHYSICAL_IMG_NR", str(i))
        output_filename = f"{base_name}-{page_number}{ALTO_PAGE_SUFFIX}"
        output_filepath = os.path.join(page_output_dir, output_filename)

        new_root = ET.Element(root.tag, root.attrib)
        if description is not None:
            new_root.append(description)
        if styles is not None:
            new_root.append(styles)

        # In the document's namespace: a bare "Layout" only re-parsed as ALTO while
        # the namespace was the registered default, i.e. for v3.
        new_layout = ET.SubElement(new_root, f"{{{ns_uri}}}Layout")
        new_layout.append(page)

        new_tree = ET.ElementTree(new_root)
        # (#31 Phase 5) The same bytes as writing to the path, written only when they changed:
        # the later stages resume by file time, so an unchanged page keeps its time.
        buffer = io.BytesIO()
        new_tree.write(buffer, encoding="UTF-8", xml_declaration=True)
        document_hook.write_bytes_if_changed(output_filepath, buffer.getvalue())
        written.append(output_filename)

    remove_stale_pages(page_output_dir, base_name, ALTO_PAGE_SUFFIX, keep=written)
    print(f"  -> Successfully split into {len(pages)} file(s) in '{page_output_dir}'.")
    return len(pages)


# ── (#31) Generic JSON page splitting ───────────────────────────────────────
#
# Real OCR/Doc-AI engines fall into two structural families for representing
# multi-page documents (no vendor-specific parsers — see 31.plan.md):
#
#   Family A — a nested "pages" array (Azure Document Intelligence, docTR,
#   Google Document AI, most PDF-text-layer dumps): one JSON per document, a
#   top-level (or one-level-nested) key holds an ordered list of page dicts.
#
#   Family B — a flat element list tagged with a per-item page field (AWS
#   Textract): no nesting, a single flat list where every item carries a
#   field pointing back to its page.
#
#   Family C — already one file per page (pero-ocr, OCR.space, and today's
#   pipeline default). Used as the fallback when neither A nor B fires, so
#   existing single-page JSON behaviour is fully preserved.

# Family A: top-level (or one-level-nested) key whose value is a non-empty list of dicts.
PAGE_LIST_KEYS = {"pages", "page_results", "parsedresults", "page_list", "document_pages", "documentpages"}

# Field checked inside each candidate page-dict / flat-list element (Family A page numbering,
# and Family B grouping key).
PAGE_NUMBER_FIELD_KEYS = {
    "pagenumber",
    "page_number",
    "page_num",
    "page",
    "pageid",
    "page_id",
    "pageno",
    "page_no",
    "physical_img_nr",
}


def _is_dict_list(value):
    """True if value is a non-empty list where every item is a dict."""
    return isinstance(value, list) and len(value) > 0 and all(isinstance(item, dict) for item in value)


def _get_field_ci(d, keys):
    """Case-insensitive lookup of the first key in `d` matching one of `keys`
    (already-lowercase). Returns the value, or None if no key matches."""
    for k, v in d.items():
        if isinstance(k, str) and k.lower() in keys:
            return v
    return None


def _find_family_a(data):
    """(D5) Depth <= 2 scan: top-level, or nested one level under a
    dict-valued top-level key. Returns (container_parent, key, page_list) for
    the first PAGE_LIST_KEYS match whose value is a non-empty list of dicts,
    by depth-then-declaration-order (top-level before nested, first-seen key
    order) — or None if nothing matches.
    """
    if not isinstance(data, dict):
        return None

    for k, v in data.items():
        if isinstance(k, str) and k.lower() in PAGE_LIST_KEYS and _is_dict_list(v):
            return data, k, v

    for _k, v in data.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(k2, str) and k2.lower() in PAGE_LIST_KEYS and _is_dict_list(v2):
                    return v, k2, v2

    return None


def _tagged_group(lst):
    """(D6) Try each PAGE_NUMBER_FIELD_KEYS candidate as a grouping field for
    a flat list of dicts. Returns (field_name, {value: [items]}) for the
    first field where a strict majority of items carry it (D6: fewer than
    half carrying the tag is not evidence of a multi-page document either)
    AND it has more than one distinct value — or None if no field qualifies.
    Group order follows first-seen value order.
    """
    n = len(lst)
    for field in PAGE_NUMBER_FIELD_KEYS:
        tagged = [(item, _get_field_ci(item, {field})) for item in lst]
        tagged = [(item, val) for item, val in tagged if val is not None]
        if len(tagged) * 2 <= n:  # not a strict majority
            continue

        distinct = []
        for _, val in tagged:
            if val not in distinct:
                distinct.append(val)
        if len(distinct) <= 1:
            continue

        groups = {val: [item for item, v in tagged if v == val] for val in distinct}
        return field, groups

    return None


def _find_family_b(data):
    """(D6) Depth <= 2 scan (same shape as Family A's) for any flat list of
    dicts that qualifies under `_tagged_group`. Returns
    (container_parent, key, list, tag_field, groups), or None.
    """
    if not isinstance(data, dict):
        return None

    candidates = []
    for k, v in data.items():
        if _is_dict_list(v):
            candidates.append((data, k, v))
    for _k, v in data.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if _is_dict_list(v2):
                    candidates.append((v, k2, v2))

    for parent, key, lst in candidates:
        result = _tagged_group(lst)
        if result is not None:
            tag_field, groups = result
            return parent, key, lst, tag_field, groups

    return None


def _replace_container_value(data, parent, key, new_value):
    """Return a shallow-copied `data` with `key` in the (possibly nested)
    `parent` dict replaced by `new_value`. Sibling keys at every level are
    preserved untouched — this is the "header" (Description/Styles-equivalent)
    that each split-out page keeps.
    """
    if parent is data:
        doc = dict(data)
        doc[key] = new_value
        return doc

    new_parent = dict(parent)
    new_parent[key] = new_value
    return {k: (new_parent if v is parent else v) for k, v in data.items()}


def _write_json_page(page_output_dir, base_name, page_number, doc) -> str:
    """Write one page file (only when its content changed, #31 Phase 5); returns its name."""
    output_filename = f"{base_name}-{page_number}{JSON_PAGE_SUFFIX}"
    document_hook.write_text_if_changed(
        os.path.join(page_output_dir, output_filename), json.dumps(doc, ensure_ascii=False)
    )
    return output_filename


def split_json_document(input_file_path, output_dir):
    """
    Splits a single JSON OCR/Doc-AI document into single-page JSON files.

    Detection order:
      1. Family A — a nested page-list container (D1-D3, D5).
      2. Family B — a flat list tagged with a per-item page field (D6).
      3. Family C — neither pattern found: fallback to today's behaviour.
    """
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_name = _doc_id_from_filename(os.path.basename(input_file_path))  # (#10 D3)
    page_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(page_output_dir, exist_ok=True)
    written = []

    # 1. Family A (Azure, docTR, Google Doc AI)
    family_a = _find_family_a(data)
    if family_a is not None:
        parent, key, page_list = family_a
        for i, page_obj in enumerate(page_list, 1):
            page_number = _get_field_ci(page_obj, PAGE_NUMBER_FIELD_KEYS)
            page_number = str(page_number) if page_number is not None else str(i)
            doc = _replace_container_value(data, parent, key, page_obj)
            written.append(_write_json_page(page_output_dir, base_name, page_number, doc))
    else:
        # 2. Family B (AWS Textract)
        family_b = _find_family_b(data)
        if family_b is not None:
            parent, key, _lst, _tag_field, groups = family_b
            for page_number, items in groups.items():
                doc = _replace_container_value(data, parent, key, items)
                written.append(_write_json_page(page_output_dir, base_name, str(page_number), doc))
        else:
            # 3. Family C fallback (pero-ocr, OCR.space)
            written.append(_write_json_page(page_output_dir, base_name, "1", data))

    remove_stale_pages(page_output_dir, base_name, JSON_PAGE_SUFFIX, keep=written)
    return len(written)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Split multi-page ALTO XML or generic JSON OCR files into single-page files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("input_dir", help="Path to the directory containing ALTO XML or JSON files to process.")
    parser.add_argument("output_dir", help="Path to the directory where split files will be saved.")
    parser.add_argument(
        "--source-origin",
        default="",
        help=(
            "How the ORIGINAL input was acquired, recorded as source.origin in the document\n"
            "record (default: ABBYY-ALTO for .xml input, ocr:generic for .json). Must start\n"
            "with an ORIGIN_ORIGINATORS prefix this repo owns — ABBYY-ALTO, ocr:<engine> or\n"
            "vlm:<engine> — e.g. --source-origin ocr:pero. Also settable as\n"
            "[DOCUMENT].SOURCE_ORIGIN or the DOCUMENT_SOURCE_ORIGIN env var."
        ),
    )
    args = parser.parse_args(argv)

    _doc_cfg = configparser.ConfigParser()
    _doc_cfg.read(DOCUMENT_CONFIG_PATH)
    document_json_dir = document_hook.resolve_document_json_dir(_doc_cfg.get("DOCUMENT", "JSON_DIR", fallback=""))
    _configured_origin = _doc_cfg.get("DOCUMENT", "SOURCE_ORIGIN", fallback="")

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at '{args.input_dir}'")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to '{os.path.abspath(args.output_dir)}'\n")

    all_files = sorted(os.listdir(args.input_dir))

    # (#31) output_types is format-aware: only declare the formats actually
    # present in input_dir, instead of hardcoding "xml". Falls back to
    # ["xml"] when the directory has neither, preserving prior behaviour.
    _formats_present = []
    if any(fname.lower().endswith(".xml") for fname in all_files):
        _formats_present.append("xml")
    if any(fname.lower().endswith(".json") for fname in all_files):
        _formats_present.append("json")
    if not _formats_present:
        _formats_present = ["xml"]

    _logger = ParadataLogger(
        program="alto-postprocess",
        config={
            "script": "page_split",
            "input_dir": str(args.input_dir),
            "output_dir": str(args.output_dir),
        },
        paradata_dir="paradata",
        output_types=_formats_present,
        config_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "setup"),
    )

    # (#10 D6) Resolved once, like every sibling stage script does, so the record
    # points at the paradata JSON this run will emit.
    _doc_paradata_ref = document_hook.paradata_ref_for(_logger)

    # (#10) Track documents (the unit of input) and pages (the unit of output)
    # separately. input_files_total counts source documents; the per-page count
    # feeds the per-format output total for throughput; successfully_processed
    # is the number of documents successfully split — so it can never exceed
    # inputs.
    _total_inputs = 0
    _docs_ok = 0

    try:
        for filename in all_files:
            lower_name = filename.lower()
            if lower_name.endswith(".xml"):
                fmt, split_fn = "xml", split_alto_xml
            elif lower_name.endswith(".json"):
                fmt, split_fn = "json", split_json_document
            else:
                continue

            input_file_path = os.path.join(args.input_dir, filename)
            print(f"Processing '{filename}'...")
            _total_inputs += 1
            page_suffix = ALTO_PAGE_SUFFIX if fmt == "xml" else JSON_PAGE_SUFFIX
            try:
                page_count = split_fn(input_file_path, args.output_dir)
                _logger.log_success(fmt, count=page_count)  # pages produced
                if page_count == 0:
                    _drop_stale_pages(args.output_dir, filename, page_suffix)
                if page_count > 0:
                    _docs_ok += 1
                    # (atrium-llm-enrich#13) page_split is the first stage to see the
                    # original input, so it is the natural first writer of `source`
                    # — set_source() is itself a no-op if a baseline already has one.
                    #
                    # (#10 D6) `_logger.run_id` and `paradata_ref_for(_logger)`, matching
                    # all seven sibling call sites. Passing the ParadataLogger OBJECT here
                    # was harmless ONLY while this call wrote `source=` and no block:
                    # set_source() never stamps, so the object was never serialised. The
                    # moment a block joined this call, _stamp() would embed it, json.dump()
                    # would raise TypeError, DocumentRecord.__exit__ would swallow that —
                    # and the ENTIRE record, `source` included, would silently never be
                    # written, leaving a stray .tmp behind.
                    doc_id = _doc_id_from_filename(filename, fmt)
                    document_hook.write_document_block(
                        document_json_dir,
                        doc_id,
                        _logger.run_id,
                        _doc_paradata_ref,
                        source={
                            "sha256": _sha256_of(input_file_path),
                            "filename": filename,
                            "media_type": "application/alto+xml" if fmt == "xml" else "application/json",
                            "page_count": page_count,
                            # (#10 D5) Without this the §1a mixed-plane guard defers forever.
                            "origin": resolve_source_origin(fmt, _configured_origin, args.source_origin),
                        },
                    )
            except Exception as e:
                print(f"  -> Failed: {e}. Skipping.")
                _logger.log_skip(str(filename), str(e))
                _drop_stale_pages(args.output_dir, filename, page_suffix)
    finally:
        _logger.finalize(input_total=_total_inputs, processed_total=_docs_ok)


def _drop_stale_pages(output_dir: str, filename: str, suffix: str) -> None:
    """(#31 Phase 5) An input that now fails or yields no pages must not leave the pages
    of an earlier run behind for the stats stage (text_split's `stale_pages_removed`)."""
    doc_id = _doc_id_from_filename(filename)
    removed = remove_stale_pages(os.path.join(output_dir, doc_id), doc_id, suffix)
    if removed:
        print(f"  -> Removed {removed} page file(s) left by an earlier run of '{doc_id}'.")


if __name__ == "__main__":
    main()
