#!/usr/bin/env python3
"""
text_formats.py — format detection and readers for the text-lines input path (#31).

Turns any sequentially readable, text-bearing file into an ordered list of pages,
each an ordered list of text lines — the `<format>-2-txt` step #31 asks for — so
the line-quality categorization can run on text of any shape, not only on ALTO XML.

Design rules (see docs/text_inputs.md for the full matrix and the reason codes):

* **Content decides, not the extension.** Binary containers are identified by their
  magic bytes (`%PDF-`, ZIP members, OLE2), XML by its root element, JSON by a parse.
  The extension only chooses between plain-text dialects (CSV/TSV/Markdown/JSONL,
  subtitles, e-mail). A gzip/bzip2/xz wrapper is looked through (`a.txt.gz`), and a
  ZIP of per-page files (a Transkribus/eScriptorium export) is ONE document.
* **Pages keep the reading order and the block structure.** A "page" is a real page
  where the format has one (PDF, ALTO, PAGE-XML, hOCR, DOCX/ODT page breaks) and
  otherwise the natural block: a JSON child object, a JSONL record, a sheet, a slide,
  an EPUB chapter, a form-feed section of a plain-text file.
* **Lines are the format's own units** — a physical line, a paragraph, a table cell
  or a spreadsheet row; an OCR engine's line where it exports words or characters
  (Tesseract TSV, ABBYY FineReader XML, DjVuXML, word-per-row CSV, JSON words).
  Very long lines are wrapped later by `shape_lines()`, the same helper the extract
  stage and the service use.
* **Light dependencies.** Standard library + lxml (already a repo dependency) for
  every XML/ZIP format — no python-docx, openpyxl or pdfplumber. PDF uses pypdfium2
  and non-UTF-8 plain text uses charset-normalizer; both are imported lazily, so a
  missing one only affects the files that need it (`dependency_missing`).
* **Fail closed, per file.** Every problem is an `IngestError` with a stable reason
  code; callers record it and move on to the next file. A document read with a
  possible loss (recovered XML, a skipped member, a bad record) carries a lossy note
  (`lossy_reasons()`), which text_split.py reports as status `partial`.
* **Report, never categorize.** Page flags (`mojibake_cp1252`, `mirrored_text=N`,
  `rotated_text=N`) describe the text layer; the categories stay classify's.

This module never prints (it is imported by the service, whose logging contract
forbids `print`); it logs through `logging` only.
"""

from __future__ import annotations

import csv
import html
import io
import json
import logging
import os
import posixpath
import re
import subprocess
import sys
import unicodedata
import zipfile
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

logger = logging.getLogger(__name__)

# ── reason codes ──────────────────────────────────────────────────────────────

#: Every stable reason code a reader can raise, with its one-line meaning. The
#: ingest report, the service's 4xx details and docs/text_inputs.md all use these.
REASON_CODES: Dict[str, str] = {
    "empty_file": "the file has zero bytes",
    "too_large": "a size, page or line limit from [TEXT_INGEST] was exceeded",
    "binary_content": "the bytes are not text and match no supported container",
    "legacy_office_unsupported": "legacy OLE2 Office file (.doc/.xls/.ppt) — save it as DOCX/XLSX/PPTX",
    "image_needs_ocr": "an image file — run OCR first, then feed the OCR output",
    "archive_unsupported": "an archive or ZIP container that is not DOCX/XLSX/PPTX/ODF/EPUB",
    "zip_limits_exceeded": "ZIP member count, size or compression ratio over the configured caps",
    "xml_entity_declaration": "XML declares entities (<!ENTITY>), refused as unsafe",
    "malformed": "the file is syntactically broken for its format",
    "corrupt": "the container could not be opened (damaged or truncated)",
    "encrypted": "the file is password-protected or DRM-encrypted",
    "timeout": "the isolated reader exceeded READER_TIMEOUT_S",
    "reader_crashed": "the isolated reader process died",
    "dependency_missing": "an optional reader dependency is not installed",
    "decode_failed": "the text could not be decoded with any configured encoding",
    "no_text": "the file was read but contains no text lines",
    "doc_id_collision": "another input file maps to the same document id",
    "doc_id_invalid": "the file name yields an empty or unusable document id",
    "output_failed": "writing the page files or the document record failed",
    "unreadable": "the file could not be opened or read (permissions or an I/O error)",
}


class IngestError(Exception):
    """A per-file ingest failure carrying a stable reason `code` (see REASON_CODES)."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message or REASON_CODES.get(code, code)


#: (#31 Phase 4) Reader notes meaning that some text may be missing or altered. A
#: document read with any of them is still processed, but text_split.py reports it
#: with status `partial` (and `--strict` counts it). Informational notes — the
#: encoding that was detected, a repaired `</n>`, an extension mismatch, a page that
#: overflowed onto `+1` pages — are deliberately not here.
LOSSY_NOTE_PREFIXES = (
    "xml_recovered",
    "decode_replacement",
    "jsonl_bad_records=",
    "csv_unbalanced_quote",
    "tsv_bad_rows=",
    "xlsx_bad_shared_string",
    "sheet_repeat_capped",
    "zip_members_skipped=",
    "epub_spine_skipped=",
    "email_bad_messages=",
    "lone_surrogates_dropped=",
)
#: Page flags with the same meaning (a PDF page PDFium could not load).
LOSSY_PAGE_FLAGS = frozenset({"page_load_failed"})


def lossy_reasons(doc: "TextDocument") -> List[str]:
    """The lossy note names of a read document (counts stripped), in first-seen order."""
    out: List[str] = []
    for note in doc.notes:
        if note.startswith(LOSSY_NOTE_PREFIXES):
            base = note.split("=", 1)[0]
            if base not in out:
                out.append(base)
    for page in doc.pages:
        for flag in page.flags:
            if flag in LOSSY_PAGE_FLAGS and flag not in out:
                out.append(flag)
    return out


def is_lossy(doc: "TextDocument") -> bool:
    return bool(lossy_reasons(doc))


def no_text_message(doc: "TextDocument") -> str:
    """The `no_text` message for a document that was read but holds no text line."""
    if doc.kind == "pdf":
        no_layer = sum(1 for p in doc.pages if p.text_layer == "none")
        return f"no extractable text layer on {no_layer} of {len(doc.pages)} PDF pages — run OCR first"
    return f"{doc.kind} file contains no text lines"


def config_bool(cfg, section: str, key: str, default: bool) -> bool:
    """A strict boolean from a ConfigParser: a typo raises ValueError naming the key."""
    if cfg is None or not cfg.has_section(section):
        return default
    raw = cfg.get(section, key, fallback="").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"[{section}] {key} = {raw!r} is not a boolean")


# ── settings ──────────────────────────────────────────────────────────────────

_MB = 1024 * 1024


@dataclass(frozen=True)
class Limits:
    """Hard caps applied before and while reading ([TEXT_INGEST] in setup/config.txt)."""

    max_file_mb: float = 256.0
    zip_max_members: int = 10000
    zip_max_total_mb: float = 1024.0
    zip_max_member_mb: float = 256.0
    zip_max_ratio: float = 200.0
    max_pages: int = 20000
    max_lines_per_page: int = 100000
    reader_timeout_s: float = 300.0


@dataclass(frozen=True)
class ReaderOptions:
    """Reader behaviour switches ([TEXT_INGEST] in setup/config.txt)."""

    fallback_encodings: Tuple[str, ...] = ("cp1250", "iso8859_2", "cp1252")
    page_breaks: str = "auto"  # DOCX/ODT: auto | explicit | none
    pdf_min_text_chars: int = 3
    pdf_garble_threshold: float = 0.15
    pdf_ocr_layer_min_ratio: float = 0.5
    max_line_chars: int = 1000
    keep_blank_lines: bool = False
    notes_placement: str = "page"  # DOCX/ODT footnotes and endnotes: page | end | skip


PAGE_BREAK_MODES = ("auto", "explicit", "none")
NOTES_MODES = ("page", "end", "skip")

#: Module-level defaults (frozen, so safe to share as argument defaults).
DEFAULT_LIMITS = Limits()
DEFAULT_OPTIONS = ReaderOptions()


def load_settings(cfg=None) -> Tuple[Limits, ReaderOptions]:
    """Build (Limits, ReaderOptions) from a ConfigParser's [TEXT_INGEST] section.

    Missing section or keys fall back to the dataclass defaults; malformed values
    raise ValueError naming the key, so a typo in the config fails the run loudly
    instead of silently disabling a cap.
    """
    section = "TEXT_INGEST"
    lim, opt = Limits(), ReaderOptions()
    if cfg is None or not cfg.has_section(section):
        return lim, opt

    def _num(key, default, cast, minimum, maximum=None):
        raw = cfg.get(section, key, fallback="").strip()
        if not raw:
            return default
        try:
            value = cast(raw)
        except ValueError as exc:
            raise ValueError(f"[{section}] {key} = {raw!r} is not a valid number") from exc
        if value < minimum:
            raise ValueError(f"[{section}] {key} = {raw!r} must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"[{section}] {key} = {raw!r} must be <= {maximum}")
        return value

    def _bool(key, default):
        return config_bool(cfg, section, key, default)

    lim = Limits(
        max_file_mb=_num("MAX_FILE_MB", lim.max_file_mb, float, 0.001),
        zip_max_members=_num("ZIP_MAX_MEMBERS", lim.zip_max_members, int, 1),
        zip_max_total_mb=_num("ZIP_MAX_TOTAL_MB", lim.zip_max_total_mb, float, 0.001),
        zip_max_member_mb=_num("ZIP_MAX_MEMBER_MB", lim.zip_max_member_mb, float, 0.001),
        zip_max_ratio=_num("ZIP_MAX_RATIO", lim.zip_max_ratio, float, 1.0),
        max_pages=_num("MAX_PAGES", lim.max_pages, int, 1),
        max_lines_per_page=_num("MAX_LINES_PER_PAGE", lim.max_lines_per_page, int, 1),
        reader_timeout_s=_num("READER_TIMEOUT_S", lim.reader_timeout_s, float, 1.0),
    )

    encodings_raw = cfg.get(section, "FALLBACK_ENCODINGS", fallback="").strip()
    encodings = tuple(e.strip() for e in encodings_raw.split(",") if e.strip()) or opt.fallback_encodings
    for enc in encodings:
        try:
            "".encode(enc)
        except LookupError as exc:
            raise ValueError(f"[{section}] FALLBACK_ENCODINGS: unknown encoding {enc!r}") from exc

    page_breaks = cfg.get(section, "PAGE_BREAKS", fallback="").strip().lower() or opt.page_breaks
    if page_breaks not in PAGE_BREAK_MODES:
        raise ValueError(f"[{section}] PAGE_BREAKS = {page_breaks!r} must be one of {', '.join(PAGE_BREAK_MODES)}")
    notes_placement = cfg.get(section, "NOTES", fallback="").strip().lower() or opt.notes_placement
    if notes_placement not in NOTES_MODES:
        raise ValueError(f"[{section}] NOTES = {notes_placement!r} must be one of {', '.join(NOTES_MODES)}")

    opt = ReaderOptions(
        fallback_encodings=encodings,
        page_breaks=page_breaks,
        pdf_min_text_chars=_num("PDF_MIN_TEXT_CHARS", opt.pdf_min_text_chars, int, 0),
        pdf_garble_threshold=_num("PDF_GARBLE_THRESHOLD", opt.pdf_garble_threshold, float, 0.0, 1.0),
        pdf_ocr_layer_min_ratio=_num("PDF_OCR_LAYER_MIN_RATIO", opt.pdf_ocr_layer_min_ratio, float, 0.0, 1.0),
        max_line_chars=_num("MAX_LINE_CHARS", opt.max_line_chars, int, 0),
        keep_blank_lines=_bool("KEEP_BLANK_LINES", opt.keep_blank_lines),
        notes_placement=notes_placement,
    )
    return lim, opt


# ── document model ────────────────────────────────────────────────────────────


@dataclass
class TextPage:
    """One page (or block) of a document: its ordered lines plus page-level facts."""

    lines: List[str]
    label: str = ""
    text_layer: Optional[str] = None  # PDF only: none | garbled | ocr | digital
    needs_ocr_reason: Optional[str] = None
    images: int = 0
    flags: List[str] = field(default_factory=list)


@dataclass
class TextDocument:
    """The reader output: kind, provenance facts and the ordered pages."""

    kind: str
    media_type: str
    pages: List[TextPage]
    encoding: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    native_pages: bool = False
    #: A reader-specific `source.origin` default (a ZIP bundle takes its members'),
    #: used by default_source_origin() ahead of the kind's registry default.
    origin_hint: Optional[str] = None

    def line_count(self) -> int:
        return sum(1 for p in self.pages for ln in p.lines if ln.strip())

    def char_count(self) -> int:
        return sum(len(ln) for p in self.pages for ln in p.lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextDocument":
        pages = [TextPage(**p) for p in data.get("pages", [])]
        rest = {k: v for k, v in data.items() if k != "pages"}
        return cls(pages=pages, **rest)


@dataclass(frozen=True)
class FormatSpec:
    """Registry row for one supported kind."""

    kind: str
    label: str
    extensions: Tuple[str, ...]
    media_type: str
    default_origin: str  # "" = computed per document (PDF)
    native_pages: bool
    page_unit: str
    line_unit: str


def _spec(kind, label, exts, media, origin, native, page_unit, line_unit):
    return FormatSpec(kind, label, tuple(exts), media, origin, native, page_unit, line_unit)


#: Every kind the text-lines path accepts. `default_origin` is the `source.origin`
#: text_split.py records unless the operator overrides it (#31, "truthful per class").
READERS: Dict[str, FormatSpec] = {
    s.kind: s
    for s in (
        _spec(
            "txt",
            "Plain text",
            (".txt", ".text", ".log"),
            "text/plain",
            "ocr:generic",
            False,
            "form-feed section",
            "physical line",
        ),
        _spec(
            "md",
            "Markdown",
            (".md", ".markdown", ".mdown"),
            "text/markdown",
            "ocr:generic",
            False,
            "form-feed section",
            "physical line (markup stripped)",
        ),
        _spec(
            "csv",
            "CSV",
            (".csv",),
            "text/csv",
            "ocr:generic",
            False,
            "file, or a page/page_num column",
            "row (text column, else cells)",
        ),
        _spec(
            "tsv",
            "TSV",
            (".tsv", ".tab"),
            "text/tab-separated-values",
            "ocr:generic",
            False,
            "file, or a page/page_num column",
            "row (text column, else cells)",
        ),
        _spec(
            "json",
            "JSON",
            (".json",),
            "application/json",
            "ocr:generic",
            False,
            "page list / page tag / top-level child",
            "text leaf",
        ),
        _spec(
            "jsonl",
            "JSON Lines",
            (".jsonl", ".ndjson"),
            "application/x-ndjson",
            "ocr:generic",
            False,
            "record",
            "text leaf",
        ),
        _spec("alto", "ALTO XML", (".xml",), "application/alto+xml", "ABBYY-ALTO", True, "Page", "TextLine"),
        _spec(
            "page-xml",
            "PAGE XML",
            (".xml",),
            "application/vnd.prima.page+xml",
            "ocr:page-xml",
            True,
            "Page",
            "TextLine (reading order)",
        ),
        _spec(
            "hocr", "hOCR", (".hocr", ".html", ".htm", ".xhtml"), "text/html", "ocr:hocr", True, "ocr_page", "ocr_line"
        ),
        _spec(
            "html",
            "HTML/XHTML",
            (".html", ".htm", ".xhtml"),
            "text/html",
            "digital-born-html",
            False,
            "CSS page break, else one page",
            "block element / <br>",
        ),
        _spec(
            "tei",
            "TEI / TEITOK XML",
            (".xml", ".tei"),
            "application/tei+xml",
            "ocr:generic",
            False,
            "<pb/>",
            "<lb/> and block end",
        ),
        _spec(
            "xml",
            "Generic XML",
            (".xml",),
            "application/xml",
            "ocr:generic",
            False,
            "root child element",
            "text-bearing element",
        ),
        _spec("pdf", "PDF", (".pdf",), "application/pdf", "", True, "PDF page", "text-layer line"),
        _spec(
            "docx",
            "Word DOCX",
            (".docx", ".docm", ".dotx"),
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "digital-born-docx",
            False,
            "page break (explicit / rendered)",
            "paragraph, table cell",
        ),
        _spec(
            "xlsx",
            "Excel XLSX",
            (".xlsx", ".xlsm"),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "digital-born-xlsx",
            False,
            "sheet",
            "row",
        ),
        _spec(
            "pptx",
            "PowerPoint PPTX",
            (".pptx", ".pptm"),
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "digital-born-pptx",
            False,
            "slide",
            "paragraph",
        ),
        _spec(
            "odt",
            "OpenDocument Text",
            (".odt",),
            "application/vnd.oasis.opendocument.text",
            "digital-born-odt",
            False,
            "page break (explicit / soft)",
            "paragraph, table cell",
        ),
        _spec(
            "ods",
            "OpenDocument Spreadsheet",
            (".ods",),
            "application/vnd.oasis.opendocument.spreadsheet",
            "digital-born-ods",
            False,
            "sheet",
            "row",
        ),
        _spec(
            "odp",
            "OpenDocument Presentation",
            (".odp",),
            "application/vnd.oasis.opendocument.presentation",
            "digital-born-odp",
            False,
            "slide",
            "paragraph",
        ),
        _spec(
            "epub",
            "EPUB",
            (".epub",),
            "application/epub+zip",
            "digital-born-epub",
            False,
            "spine chapter",
            "block element / <br>",
        ),
        _spec("rtf", "RTF", (".rtf",), "application/rtf", "digital-born-rtf", False, "\\page", "\\par / \\line"),
        # (#31 Phase 4) OCR engine exports with real pages.
        _spec(
            "tesseract-tsv",
            "Tesseract TSV",
            (".tsv",),
            "text/tab-separated-values",
            "ocr:tesseract",
            True,
            "page_num",
            "words of one (block, par, line), joined by spaces",
        ),
        _spec(
            "abbyy-xml",
            "ABBYY FineReader XML",
            (".xml",),
            "application/xml",
            "ocr:abbyy-finereader",
            True,
            "page",
            "line (charParams concatenated)",
        ),
        _spec("djvu-xml", "DjVuXML", (".xml",), "application/xml", "ocr:djvu", True, "OBJECT", "LINE (WORDs joined)"),
        # A ZIP of per-page files (Transkribus/eScriptorium exports, a folder of page
        # TXTs): one document; its origin and page nativeness come from its members.
        _spec(
            "zip-bundle",
            "ZIP bundle of page files",
            (".zip",),
            "application/zip",
            "",
            False,
            "member file (natural order), then its own pages",
            "the member format's line",
        ),
        _spec(
            "srt", "SubRip subtitles", (".srt",), "application/x-subrip", "ocr:generic", False, "file", "cue text line"
        ),
        _spec("vtt", "WebVTT subtitles", (".vtt",), "text/vtt", "ocr:generic", False, "file", "cue text line"),
        _spec(
            "eml",
            "E-mail message",
            (".eml",),
            "message/rfc822",
            "digital-born-eml",
            False,
            "message",
            "Subject, then body line",
        ),
        _spec(
            "mbox",
            "Mailbox",
            (".mbox", ".mbx"),
            "application/mbox",
            "digital-born-mbox",
            False,
            "message",
            "Subject, then body line",
        ),
    )
}

#: Single-file compression wrappers read transparently (`.txt.gz`, `.xml.bz2`, …).
COMPRESSION_SUFFIXES = (".gz", ".bz2", ".xz")

#: Extensions that are never content and are refused before reading (image scans
#: are the main case: they need OCR, not text ingest).
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp", ".jp2", ".j2k"}


def supported_extensions() -> List[str]:
    """Every extension some reader claims, plus the compression wrappers, sorted
    (for UIs and `accept=` lists)."""
    return sorted({ext for spec in READERS.values() for ext in spec.extensions} | set(COMPRESSION_SUFFIXES))


def default_source_origin(doc: TextDocument) -> str:
    """The truthful `source.origin` for a read document (#31 origin policy).

    A reader's `origin_hint` comes first (a ZIP bundle takes its members' origin).
    PDF is decided per document: a text layer that is mostly invisible text over the
    page image is an OCR layer (`ocr:pdf-text-layer`, this repo's to own); anything
    else is a born-digital PDF (`digital-born-pdf`, llm-enrich's digital-convert's).
    """
    if doc.origin_hint:
        return doc.origin_hint
    if doc.kind == "pdf":
        text_pages = [p for p in doc.pages if p.text_layer in ("ocr", "digital", "garbled")]
        ocr_pages = [p for p in text_pages if p.text_layer == "ocr"]
        if text_pages and len(ocr_pages) * 2 >= len(text_pages):
            return "ocr:pdf-text-layer"
        return "digital-born-pdf"
    spec = READERS.get(doc.kind)
    return (spec.default_origin if spec else "") or "ocr:generic"


# ── low-level text helpers ────────────────────────────────────────────────────

_BOMS = (
    (b"\x00\x00\xfe\xff", "utf-32-be"),
    (b"\xff\xfe\x00\x00", "utf-32-le"),
    (b"\xef\xbb\xbf", "utf-8"),
    (b"\xff\xfe", "utf-16-le"),
    (b"\xfe\xff", "utf-16-be"),
)

# Every separator str.splitlines() honours except "\n" and the page break "\f":
# classify_TEXT reads page files with readlines() (universal newlines: \n \r \r\n),
# so anything else left in a line would make line numbers drift between the line
# table and DOC_LINE_CATEG.
_LINE_SEPARATORS = re.compile("\r\n|[\r\v\x1c\x1d\x1e\x85\u2028\u2029]")
_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi", "\ufb04": "ffl",
              "\ufb05": "st", "\ufb06": "st"}  # fmt: skip
_STRIP_CHARS = dict.fromkeys(
    map(ord, "\u200b\u2060\ufeff\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"), None
)
_SPACE_CHARS = dict.fromkeys(map(ord, "\u00a0\u2007\u202f"), " ")
_HYPHEN_MARKS = ("\u00ad", "\x02")  # soft hyphen; PDFium's end-of-line hyphen marker
_WS_RUN = re.compile(r"\s+")
# A lone UTF-16 surrogate (an RTF \u pair split by a bad writer, a JSON "\ud83d" escape)
# cannot be written as UTF-8: left in, it failed the whole document's page write.
_SURROGATES = re.compile("[\ud800-\udfff]")


def _has_bom(data: bytes) -> bool:
    return any(data.startswith(bom) for bom, _ in _BOMS)


def _utf16_guess(data: bytes) -> Optional[str]:
    """UTF-16 without a BOM: NULs concentrated on one byte parity (Latin-script text)."""
    sample = data[:4096]
    if len(sample) < 4:
        return None
    even, odd = sample[0::2], sample[1::2]
    if odd.count(0) > 0.4 * len(odd) and even.count(0) < 0.1 * len(even):
        return "utf-16-le"
    if even.count(0) > 0.4 * len(even) and odd.count(0) < 0.1 * len(odd):
        return "utf-16-be"
    return None


def _looks_binary(data: bytes) -> bool:
    """NUL bytes or a high share of C0 control bytes (excluding \\t \\n \\r \\f \\v, ESC)."""
    sample = data[:65536]
    if not sample:
        return False
    if sample.count(b"\x00") > max(1, len(sample) // 100):
        return True
    controls = sum(1 for b in sample if b < 0x20 and b not in (0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x1B))
    return controls > len(sample) * 0.1


def decode_bytes(
    data: bytes, fallbacks: Sequence[str] = ReaderOptions.fallback_encodings
) -> Tuple[str, str, List[str]]:
    """Decode text bytes: BOM → strict UTF-8 → binary check → detection → fallbacks.

    Returns (text, encoding, flags). Detection uses charset-normalizer restricted to
    `fallbacks` (the archive's plausible legacy code pages — cp1250 first for Czech),
    so a short Czech text is not "detected" as some unrelated code page and turned
    into mojibake that the categorizer would then score as damaged OCR.
    """
    flags: List[str] = []
    for bom, enc in _BOMS:
        if data.startswith(bom):
            try:
                return data[len(bom) :].decode(enc), enc, flags
            except UnicodeDecodeError:
                flags.append("decode_replacement")
                return data[len(bom) :].decode(enc, errors="replace"), enc, flags
    if b"\x00" in data[:65536]:
        guess = _utf16_guess(data)
        if guess:
            try:
                return data[: len(data) - (len(data) % 2)].decode(guess), guess, flags + ["utf16_without_bom"]
            except UnicodeDecodeError:
                pass
        if _looks_binary(data):
            raise IngestError("binary_content", "bytes contain NULs/control codes and no BOM")
    try:
        return data.decode("utf-8"), "utf-8", flags
    except UnicodeDecodeError:
        pass
    if _looks_binary(data):
        raise IngestError("binary_content", "bytes contain NULs/control codes and no BOM")

    candidates = [e for e in fallbacks if e]
    try:
        from charset_normalizer import from_bytes  # optional, MIT

        best = from_bytes(data, cp_isolation=candidates or None).best()
        if best is not None and best.encoding:
            flags.append("encoding_detected")
            return str(best), best.encoding, flags
    except ImportError:
        flags.append("charset_normalizer_missing")
    except Exception as exc:  # detection is best-effort; fall through to the fallbacks
        logger.debug("charset detection failed: %s", exc)

    for enc in candidates:
        try:
            return data.decode(enc), enc, flags + ["encoding_fallback"]
        except (UnicodeDecodeError, LookupError):
            continue
    if not candidates:
        raise IngestError("decode_failed", "not UTF-8 and no FALLBACK_ENCODINGS configured")
    flags.append("decode_replacement")
    return data.decode(candidates[0], errors="replace"), candidates[0], flags


def normalize_newlines(text: str) -> str:
    """Map every line separator except the page break \\f to \\n."""
    return _LINE_SEPARATORS.sub("\n", text)


def normalize_line(line: str) -> str:
    """Canonicalise one line (no line separators left inside it).

    NFC; U+FB00–FB06 ligatures expanded (not full NFKC, which would also rewrite
    superscripts and fractions); a soft hyphen or PDFium's \\x02 at the end of the
    line becomes "-" and is dropped elsewhere; zero-width, BOM and bidi controls are
    removed (ZWJ/ZWNJ kept — they carry meaning in some scripts); NBSP-like spaces
    become plain spaces; remaining C0/C1 controls except \\t and lone surrogates are
    removed; stripped.
    """
    if not line:
        return ""
    if _SURROGATES.search(line):
        line = _SURROGATES.sub("", line)
    line = unicodedata.normalize("NFC", line)
    if any(ch in line for ch in _LIGATURES):
        line = "".join(_LIGATURES.get(ch, ch) for ch in line)
    line = line.translate(_STRIP_CHARS).translate(_SPACE_CHARS)
    stripped = line.rstrip()
    if stripped.endswith(_HYPHEN_MARKS):
        stripped = stripped[:-1] + "-"
    for mark in _HYPHEN_MARKS:
        stripped = stripped.replace(mark, "")
    cleaned = "".join(ch for ch in stripped if ch == "\t" or unicodedata.category(ch) != "Cc")
    return cleaned.strip()


def text_to_pages(text: str) -> List[List[str]]:
    """Split decoded text into pages at form feeds, then into physical lines."""
    text = normalize_newlines(text)
    pages = text.split("\f")
    out = []
    for page in pages:
        lines = page.split("\n")
        if lines and lines[-1] == "":
            lines.pop()  # the file's (or section's) final newline is not a line
        out.append(lines)
    while len(out) > 1 and not any(ln.strip() for ln in out[-1]):
        out.pop()  # a trailing form feed does not open a real page
    return out


def wrap_line(line: str, width: int) -> List[str]:
    """Split a line into chunks of at most `width` characters at word boundaries;
    an unbroken run longer than `width` is hard-split. `width <= 0` disables it."""
    if width <= 0 or len(line) <= width:
        return [line]
    out = []
    rest = line
    while len(rest) > width:
        cut = rest.rfind(" ", 0, width + 1)
        if cut <= 0:
            cut = rest.rfind("\t", 0, width + 1)
        if cut <= 0:
            out.append(rest[:width])
            rest = rest[width:]
        else:
            out.append(rest[:cut].rstrip())
            rest = rest[cut + 1 :].lstrip()
    if rest:
        out.append(rest)
    return out


def shape_lines(lines: Iterable[str], max_chars: int = 1000, keep_blank: bool = False) -> List[str]:
    """The classify-ready line list: blank lines dropped (unless `keep_blank`),
    every other line normalized and wrapped to `max_chars`.

    Shared by extract_TEXT_2_TXT.py and the service so a file yields the same lines
    through either path.
    """
    out: List[str] = []
    for raw in lines:
        for piece in normalize_newlines(raw).replace("\f", "\n").split("\n"):
            line = normalize_line(piece)
            if not line:
                if keep_blank:
                    out.append("")
                continue
            out.extend(wrap_line(line, max_chars))
    return out


def _collapse_ws(text: str) -> str:
    return _WS_RUN.sub(" ", text).strip()


def _local(tag: Any) -> str:
    """Local name of an lxml tag ('' for comments/PIs, whose tag is a function)."""
    if not isinstance(tag, str):
        return ""
    return tag.rpartition("}")[2]


def _ns(tag: str) -> str:
    return tag[1:].partition("}")[0] if tag.startswith("{") else ""


# ── ZIP and XML safety ────────────────────────────────────────────────────────


def open_zip(path: str, limits: Limits) -> zipfile.ZipFile:
    """Open a ZIP container after checking every declared size against the caps.

    zipfile never yields more than a member's declared size (and verifies the CRC),
    so checking the declared sizes up front is a sound zip-bomb guard.
    """
    try:
        zf = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise IngestError("corrupt", f"not a readable ZIP container ({exc})") from exc
    try:
        infos = zf.infolist()
        if len(infos) > limits.zip_max_members:
            raise IngestError("zip_limits_exceeded", f"{len(infos)} members > ZIP_MAX_MEMBERS={limits.zip_max_members}")
        total = sum(i.file_size for i in infos)
        if total > limits.zip_max_total_mb * _MB:
            raise IngestError("zip_limits_exceeded", f"{total / _MB:.1f} MB unpacked > ZIP_MAX_TOTAL_MB")
        for info in infos:
            if info.flag_bits & 0x1:
                raise IngestError("encrypted", f"ZIP member {info.filename!r} is encrypted")
            if info.file_size > limits.zip_max_member_mb * _MB:
                raise IngestError("zip_limits_exceeded", f"member {info.filename!r} > ZIP_MAX_MEMBER_MB")
            if (
                info.file_size > _MB
                and info.compress_size > 0
                and info.file_size / info.compress_size > limits.zip_max_ratio
            ):
                raise IngestError("zip_limits_exceeded", f"member {info.filename!r} compression ratio > ZIP_MAX_RATIO")
    except IngestError:
        zf.close()
        raise
    return zf


def _zip_read(zf: zipfile.ZipFile, name: str) -> bytes:
    try:
        return zf.read(name)
    except KeyError as exc:
        raise IngestError("malformed", f"container member {name!r} is missing") from exc
    except (zipfile.BadZipFile, OSError, EOFError, zipfile.LargeZipFile) as exc:
        raise IngestError("corrupt", f"container member {name!r} is damaged ({exc})") from exc
    except RuntimeError as exc:  # zipfile raises RuntimeError for encrypted members
        raise IngestError("encrypted", str(exc)) from exc


_ENTITY_DECL = re.compile(rb"<!ENTITY", re.IGNORECASE)
# Older TEITOK exports (atrium-nlp-enrich's format 1) close <name> with </n>: not well-formed.
_NAME_CLOSE = re.compile(rb"</n\s*>")
_XML_DECL = re.compile(r"^\s*<\?xml[^>]*\?>", re.IGNORECASE)


def _etree():
    try:
        from lxml import etree
    except ImportError as exc:  # declared in setup/requirements.txt
        raise IngestError("dependency_missing", "lxml is required for XML-based formats") from exc
    return etree


def _xml_parser(recover: bool = False):
    etree = _etree()
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
        remove_comments=True,
        remove_pis=True,
        recover=recover,
    )


def _utf8_xml_bytes(data: bytes, fallbacks: Sequence[str]) -> bytes:
    """Re-encode XML bytes as UTF-8 without a declaration (for UTF-16/legacy input)."""
    text, _enc, _flags = decode_bytes(data, fallbacks)
    return _XML_DECL.sub("", text, count=1).encode("utf-8")


def parse_xml_bytes(data: bytes, fallbacks: Sequence[str] = ReaderOptions.fallback_encodings, notes=None):
    """Parse untrusted XML: no entities, no DTD, no network, libxml2's size limits on.

    Entity declarations fail closed. A parse error gets one retry after re-decoding
    through the decode chain, and a last one in recover mode (flagged
    `xml_recovered`, since recovered text may be incomplete). Before recovering, a document
    with `<name>` elements closed by `</n>` (the quirk of older TEITOK exports) is repaired
    exactly instead (flagged `name_close_repaired`): recover mode would drop text.
    """
    etree = _etree()
    if _has_bom(data) or b"\x00" in data[:4096]:
        data = _utf8_xml_bytes(data, fallbacks)
    if _ENTITY_DECL.search(data):
        raise IngestError("xml_entity_declaration")
    try:
        return etree.fromstring(data, _xml_parser())
    except etree.XMLSyntaxError:
        pass
    try:
        data = _utf8_xml_bytes(data, fallbacks)
        return etree.fromstring(data, _xml_parser())
    except (etree.XMLSyntaxError, IngestError):
        pass
    if b"<name" in data and _NAME_CLOSE.search(data):
        try:
            root = etree.fromstring(_NAME_CLOSE.sub(b"</name>", data), _xml_parser())
        except etree.XMLSyntaxError:
            pass
        else:
            if notes is not None:
                notes.append("name_close_repaired")
            return root
    try:
        root = etree.fromstring(data, _xml_parser(recover=True))
    except etree.XMLSyntaxError as exc:
        raise IngestError("malformed", f"XML could not be parsed ({exc})") from exc
    if root is None:
        raise IngestError("malformed", "XML could not be parsed")
    if notes is not None:
        notes.append("xml_recovered")
    return root


def parse_html(text: str):
    """Parse (X)HTML leniently with lxml.html — no network, no comments/PIs."""
    try:
        import lxml.html
    except ImportError as exc:
        raise IngestError("dependency_missing", "lxml is required for HTML-based formats") from exc
    text = _XML_DECL.sub("", text, count=1)
    if "<!ENTITY" in text[:65536].upper():
        raise IngestError("xml_entity_declaration")
    if not text.strip():
        raise IngestError("no_text", "empty HTML document")
    parser = lxml.html.HTMLParser(remove_comments=True, remove_pis=True, no_network=True, huge_tree=False)
    try:
        return lxml.html.document_fromstring(text, parser=parser)
    except Exception as exc:  # lxml raises ParserError for documents with no elements
        raise IngestError("malformed", f"HTML could not be parsed ({exc})") from exc


# ── a small iterative tree walker + page/line assembler ──────────────────────


def _walk(root, on_start: Callable, on_end: Callable, on_text: Callable, skip: Callable[[Any], bool]) -> None:
    """Document-order walk over an lxml tree without recursion (deep HTML is safe).

    Calls on_start(el) → on_text(el.text) → children → on_end(el) → on_text(el.tail).
    A skipped element contributes only its tail (the text AFTER it).
    """
    stack: List[Tuple[str, Any]] = [("start", root)]
    while stack:
        op, el = stack.pop()
        if op == "tail":
            if el is not root and el.tail:
                on_text(el.tail)
            continue
        if op == "end":
            on_end(el)
            if el is not root and el.tail:
                on_text(el.tail)
            continue
        if not isinstance(el.tag, str) or skip(el):
            stack.append(("tail", el))
            continue
        on_start(el)
        if el.text:
            on_text(el.text)
        stack.append(("end", el))
        for child in reversed(list(el)):
            stack.append(("start", child))


class _Flow:
    """Accumulates text into lines and lines into pages, in reading order.

    `explicit_break()` starts a new page unless nothing was emitted yet (never an
    empty first page); `rendered_break()` (DOCX lastRenderedPageBreak, ODF
    soft-page-break) only starts one if the current page already has text — which
    is what deduplicates it against an explicit break just before it.
    """

    def __init__(self, collapse: bool = True):
        self.pages: List[TextPage] = [TextPage([])]
        self.buf: List[str] = []
        self.collapse = collapse
        self._any = False
        #: Lines that belong at the END of the current page (footnotes, #31 Phase 4):
        #: flushed onto it at the next page break, or when the flow finishes.
        self.tail: List[str] = []

    def text(self, s: str) -> None:
        if s:
            self.buf.append(s)

    def line_break(self) -> None:
        raw = "".join(self.buf)
        self.buf = []
        line = _collapse_ws(raw) if self.collapse else raw.strip()
        if line:
            self.pages[-1].lines.append(line)
            self._any = True

    def defer_to_page_end(self, lines: Iterable[str]) -> None:
        self.tail.extend(ln for ln in lines if ln)

    def _flush_tail(self) -> None:
        if self.tail:
            self.pages[-1].lines.extend(self.tail)
            self.tail = []
            self._any = True

    def explicit_break(self) -> None:
        self.line_break()
        self._flush_tail()
        if self._any:
            self.pages.append(TextPage([]))

    def rendered_break(self) -> None:
        self.line_break()
        self._flush_tail()
        if self.pages[-1].lines:
            self.pages.append(TextPage([]))

    def finish(self) -> List[TextPage]:
        self.line_break()
        self._flush_tail()
        while len(self.pages) > 1 and not self.pages[-1].lines:
            self.pages.pop()
        return self.pages


# ── sniffing ──────────────────────────────────────────────────────────────────

_IMAGE_MAGIC = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"II*\x00",
    b"MM\x00*",
    b"\x00\x00\x00\x0cjP  ",
    b"\xff\x4f\xff\x51",
)
#: 7z, RAR and zstd: refused (no stdlib reader on Python 3.11). gzip/bzip2/xz are
#: single-file wrappers and are read through (COMPRESSION_SUFFIXES).
_ARCHIVE_MAGIC = (b"7z\xbc\xaf\x27\x1c", b"Rar!\x1a\x07", b"\x28\xb5\x2f\xfd")
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_UTF8_BOM = b"\xef\xbb\xbf"
_ODF_MIMETYPES = {
    "application/vnd.oasis.opendocument.text": "odt",
    "application/vnd.oasis.opendocument.text-template": "odt",
    "application/vnd.oasis.opendocument.spreadsheet": "ods",
    "application/vnd.oasis.opendocument.spreadsheet-template": "ods",
    "application/vnd.oasis.opendocument.presentation": "odp",
    "application/vnd.oasis.opendocument.presentation-template": "odp",
    "application/epub+zip": "epub",
}
_ROOT_TAG = re.compile(rb"<\s*([A-Za-z_][\w.\-]*:)?([A-Za-z_][\w.\-]*)[\s/>]")
_XML_PROLOG_NOISE = re.compile(rb"^(\s|<\?.*?\?>|<!--.*?-->|<!DOCTYPE(?:[^\[>]|\[.*?\])*>)*", re.DOTALL | re.IGNORECASE)
_HOCR_HINT = re.compile(rb"ocr_page|ocrx?_line|ocrx_word|ocr-system|ocr-capabilities", re.IGNORECASE)

#: Extensions whose content is plain text with no structure of its own.
_PLAIN_EXTS = (".txt", ".text", ".log")
#: Plain-text dialects the extension chooses (content cannot tell prose with commas
#: from CSV). They also win over an unknown XML root and over the `{`/`[` JSON probe:
#: a README.md that starts with `<p align="center">` is still Markdown.
_EXT_DIALECTS = {
    ".md": "md",
    ".markdown": "md",
    ".mdown": "md",
    ".csv": "csv",
    ".tsv": "tsv",
    ".tab": "tsv",
    ".srt": "srt",
    ".vtt": "vtt",
    ".eml": "eml",
    ".mbox": "mbox",
    ".mbx": "mbox",
}
_HTML_EXTS = (".html", ".htm", ".xhtml", ".hocr")
#: Every extension that promises text — a `%PDF-` marker somewhere inside one of these
#: (a note that quotes a PDF header) does not make it a PDF.
_TEXT_EXTS = frozenset(
    set(_PLAIN_EXTS) | set(_EXT_DIALECTS) | set(_HTML_EXTS) | {".json", ".jsonl", ".ndjson", ".xml", ".tei", ".rtf"}
)
_TESSERACT_COLUMNS = (
    "level", "page_num", "block_num", "par_num", "line_num", "word_num",
    "left", "top", "width", "height", "conf", "text",
)  # fmt: skip
_CUE_TIMING = re.compile(
    r"^\s*(?:\d+:)?\d{1,2}:\d{2}[,.]\d{1,3}\s*-->\s*(?:\d+:)?\d{1,2}:\d{2}[,.]\d{1,3}", re.MULTILINE
)
_MAIL_FIELD = re.compile(rb"^([!-9;-~]+):")
_MAIL_ORIGIN_FIELDS = {b"from", b"received", b"return-path"}
_MAIL_TECH_FIELDS = {
    b"mime-version", b"message-id", b"received", b"return-path", b"content-type", b"dkim-signature",
    b"x-mailer", b"delivered-to",
}  # fmt: skip


def compression_of_bytes(head: bytes) -> Optional[str]:
    """`gzip` / `bz2` / `xz` for a single-file compression wrapper, else None."""
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if head.startswith(b"BZh") and head[3:4].isdigit() and head[4:10] == b"1AY&SY":
        return "bz2"
    return None


def compression_of(path: str) -> Optional[str]:
    """The compression wrapper of the file at `path` (None if it has none or is unreadable)."""
    try:
        with open(path, "rb") as fh:
            return compression_of_bytes(fh.read(10))
    except OSError:
        return None


_COMPRESSED_INNER = {".tgz": ".tar", ".tbz2": ".tar", ".tbz": ".tar", ".txz": ".tar"}
_COMPRESSION_EXTS = (".gz", ".gzip", ".bz2", ".bz", ".xz")


def inner_name(name: str) -> str:
    """The name of the file inside a compression wrapper: `a.txt.gz` → `a.txt`."""
    stem, ext = os.path.splitext(name)
    lower = ext.lower()
    if lower in _COMPRESSED_INNER:
        return stem + _COMPRESSED_INNER[lower]
    if lower in _COMPRESSION_EXTS:
        return stem
    return name


def _decompressor(fmt: str):
    import bz2
    import gzip
    import lzma

    return {"gzip": gzip.open, "bz2": bz2.open, "xz": lzma.open}[fmt]


def _decompress_errors():
    import lzma
    import zlib

    return (OSError, EOFError, ValueError, zlib.error, lzma.LZMAError)


def decompress_file(path: str, fmt: str, limits: Limits, max_bytes: Optional[int] = None) -> bytes:
    """Decompress a gzip/bz2/xz file with a bound on the output (a decompression bomb).

    The cap is MAX_FILE_MB of decompressed bytes, and at most ZIP_MAX_RATIO times the
    compressed size — the same two caps ZIP members get. `max_bytes` reads only a
    prefix (for sniffing) and never raises for size.
    """
    try:
        size = os.path.getsize(path)
        cap_file = int(limits.max_file_mb * _MB)
        cap_ratio = int(max(_MB, size * limits.zip_max_ratio))
        cap = min(cap_file, cap_ratio) if max_bytes is None else max_bytes
        with _decompressor(fmt)(path, "rb") as fh:
            data = fh.read(cap + 1 if max_bytes is None else max_bytes)
    except PermissionError as exc:
        raise IngestError("unreadable", f"cannot read file ({exc})") from exc
    except _decompress_errors() as exc:
        raise IngestError("corrupt", f"the {fmt} stream could not be decompressed ({exc})") from exc
    if max_bytes is None and len(data) > cap:
        if cap_ratio < cap_file:
            raise IngestError(
                "zip_limits_exceeded", f"{fmt} stream expands more than ZIP_MAX_RATIO={limits.zip_max_ratio:g}x"
            )
        raise IngestError("too_large", f"{fmt} stream expands beyond MAX_FILE_MB={limits.max_file_mb:g}")
    if not data:
        raise IngestError("empty_file", f"the {fmt} stream is empty")
    return data


def _rels_target(zf: zipfile.ZipFile, rels_name: str, type_suffix: str, base_dir: str = "") -> Optional[str]:
    """Resolve the first relationship whose Type ends with `type_suffix`."""
    if rels_name not in zf.namelist():
        return None
    root = parse_xml_bytes(_zip_read(zf, rels_name))
    for rel in root:
        if _local(rel.tag) != "Relationship":
            continue
        if rel.get("Type", "").endswith(type_suffix) and rel.get("TargetMode", "") != "External":
            return _resolve_part(base_dir, rel.get("Target", ""))
    return None


def _resolve_part(base_dir: str, target: str, names: Optional[Iterable[str]] = None) -> str:
    """A package-relative part name: fragment dropped, %-escapes decoded (EPUB and
    OOXML hrefs are URIs, so `Kapitola%201.xhtml` names `Kapitola 1.xhtml`). With
    `names`, a member stored under the still-escaped name is found too."""

    def join(t: str) -> str:
        return t.lstrip("/") if t.startswith("/") else posixpath.normpath(posixpath.join(base_dir, t))

    raw = target.split("#", 1)[0]
    decoded = join(unquote(raw))
    if names is not None and decoded not in names:
        literal = join(raw)
        if literal in names:
            return literal
    return decoded


def _rels_map(zf: zipfile.ZipFile, part: str) -> Dict[str, Tuple[str, str]]:
    """{rId: (Type, resolved target)} for one OOXML part's relationships."""
    base_dir = posixpath.dirname(part)
    rels_name = posixpath.join(base_dir, "_rels", posixpath.basename(part) + ".rels")
    out: Dict[str, Tuple[str, str]] = {}
    if rels_name not in zf.namelist():
        return out
    root = parse_xml_bytes(_zip_read(zf, rels_name))
    names = set(zf.namelist())
    for rel in root:
        if _local(rel.tag) == "Relationship" and rel.get("TargetMode", "") != "External":
            out[rel.get("Id", "")] = (rel.get("Type", ""), _resolve_part(base_dir, rel.get("Target", ""), names))
    return out


def _ooxml_main_part(zf: zipfile.ZipFile) -> Optional[str]:
    return _rels_target(zf, "_rels/.rels", "/officeDocument")


#: Member names that make a ZIP look like an office/EPUB package: without their
#: marker (`mimetype` / `[Content_Types].xml`) it is a damaged package, never a bundle.
_OFFICE_MARKERS = {"META-INF/container.xml", "word/document.xml", "xl/workbook.xml", "ppt/presentation.xml"}


def _sniff_zip(path: str, limits: Limits) -> str:
    with open_zip(path, limits) as zf:
        infos = zf.infolist()
        names = {i.filename for i in infos}
        if "mimetype" in names:
            mimetype = _zip_read(zf, "mimetype").decode("ascii", errors="replace").strip()
            if mimetype in _ODF_MIMETYPES:
                return _ODF_MIMETYPES[mimetype]
            raise IngestError("archive_unsupported", f"unsupported ODF/OCF type {mimetype!r}")
        if "[Content_Types].xml" in names:
            main = _ooxml_main_part(zf) or ""
            top = main.split("/", 1)[0]
            if top == "word":
                return "docx"
            if top == "xl":
                return "xlsx"
            if top == "ppt":
                return "pptx"
            raise IngestError("archive_unsupported", f"OOXML package with unsupported main part {main!r}")
        if names & _OFFICE_MARKERS or {"content.xml", "META-INF/manifest.xml"} <= names:
            raise IngestError("archive_unsupported", "damaged office/EPUB package (its type marker is missing)")
        kinds = {_bundle_member_kind(i) for i in infos}
        if "candidate" in kinds:
            return "zip-bundle"
        if "image" in kinds and not kinds & {"unknown", "refused"}:
            # Only images (besides metadata): scans. A package that merely carries a
            # preview image next to its own data (Apple .pages) is not one.
            raise IngestError("image_needs_ocr", "a ZIP of page images — run OCR first, then feed its output")
        if "refused" in kinds:
            raise IngestError(
                "archive_unsupported",
                "a ZIP of PDFs or nested containers — unpack it (PDFs are read in an isolated process, not inside "
                "bundles)",
            )
    raise IngestError(
        "archive_unsupported", "ZIP archive with no text-bearing page files (nor DOCX/XLSX/PPTX/ODF/EPUB)"
    )


def _xml_root_name(head: bytes) -> Tuple[str, bytes]:
    """(root local name, namespace-ish prefix bytes) from the first bytes of an XML file."""
    body = _XML_PROLOG_NOISE.sub(b"", head, count=1)
    match = _ROOT_TAG.match(body)
    if not match:
        return "", b""
    return match.group(2).decode("ascii", errors="replace"), body[:2048]


#: Root elements of the XML vocabularies with a reader of their own.
_TEI_ROOTS = ("tei", "teicorpus", "tei.2", "teicorpus.2")


def _xml_kind(head: bytes) -> str:
    name, start = _xml_root_name(head)
    lname = name.lower()
    if lname == "alto":
        return "alto"
    if lname == "pcgts":
        return "page-xml"
    if lname in _TEI_ROOTS:
        return "tei"
    if lname == "html":
        return "hocr" if _HOCR_HINT.search(head) else "html"
    if lname == "document" and b"abbyy" in start.lower():
        return "abbyy-xml"
    if lname == "djvuxml":
        return "djvu-xml"
    if lname:
        return "xml"
    return ""


def _looks_like_email(head: bytes) -> bool:
    """A strict RFC 822 header block: ≥3 fields, an origin field and a technical one.

    Strict on purpose — a YAML file or a typed memo ("From: … To: … Subject: …")
    must stay plain text.
    """
    fields: List[bytes] = []
    for line in head.replace(b"\r\n", b"\n").split(b"\n")[:400]:
        if not line.strip():
            break
        if line[:1] in (b" ", b"\t"):
            if not fields:
                return False
            continue
        match = _MAIL_FIELD.match(line)
        if not match:
            return False
        fields.append(match.group(1).lower())
    else:
        return False
    names = set(fields)
    return len(fields) >= 3 and bool(names & _MAIL_ORIGIN_FIELDS) and bool(names & _MAIL_TECH_FIELDS)


def _looks_like_mbox(head: bytes) -> bool:
    if not head.startswith(b"From "):
        return False
    lines = head.replace(b"\r\n", b"\n").split(b"\n")
    return len(lines) > 1 and bool(_MAIL_FIELD.match(lines[1]))


def _looks_like_srt(text: str) -> bool:
    lines = [ln.strip() for ln in text.lstrip().split("\n")[:3]]
    return len(lines) >= 2 and lines[0].isdigit() and bool(_CUE_TIMING.match(lines[1]))


def sniff_kind(path: str, limits: Limits = DEFAULT_LIMITS, notes: Optional[List[str]] = None) -> str:
    """Decide which reader handles `path` (see the module docstring's rules).

    Raises IngestError for files that are not text-bearing inputs (images, legacy
    Office, archives, binaries) with the reason code a user can act on. A gzip/bz2/xz
    wrapper is looked through: the kind returned is the inner file's.
    """
    notes = notes if notes is not None else []
    try:
        with open(path, "rb") as fh:
            head = fh.read(65536)
    except OSError as exc:
        raise IngestError("unreadable", f"cannot read file ({exc})") from exc
    if not head:
        raise IngestError("empty_file")
    name = os.path.basename(path)
    fmt = compression_of_bytes(head)
    if fmt:
        inner = inner_name(name)
        return sniff_bytes(
            decompress_file(path, fmt, limits, max_bytes=65536),
            inner,
            limits,
            notes,
            full=lambda: decompress_file(path, fmt, limits),
            container=f"a {fmt} file",
        )

    def full() -> Optional[bytes]:
        try:
            if os.path.getsize(path) > limits.max_file_mb * _MB:
                return None
            with open(path, "rb") as fh:
                return fh.read()
        except OSError as exc:
            raise IngestError("unreadable", f"cannot read file ({exc})") from exc

    return sniff_bytes(head, name, limits, notes, full=full, zip_path=path)


def sniff_bytes(
    head: bytes,
    name: str,
    limits: Limits,
    notes: List[str],
    *,
    full: Callable[[], Optional[bytes]],
    zip_path: Optional[str] = None,
    container: Optional[str] = None,
) -> str:
    """The detection rules over the first bytes of a file called `name`.

    `container` names what these bytes were found inside (a gzip file, a ZIP bundle):
    nested containers — a ZIP, an archive, another compressed stream, a PDF — are
    refused there. `full()` returns the whole content (None when it is too large to
    probe), used only for the JSON/JSONL decision.
    """
    ext = os.path.splitext(name)[1].lower()
    if not head:
        raise IngestError("empty_file")

    def nested(what: str) -> IngestError:
        return IngestError("archive_unsupported", f"{what} inside {container} — unpack it first")

    kind = ""
    body = head[3:] if head.startswith(_UTF8_BOM) else head
    if compression_of_bytes(head):
        raise nested("compressed data")
    # `%PDF-` at the start is a PDF; further into the first KiB (PDF readers tolerate
    # a junk prefix) only after the container signatures, and never in a file whose
    # extension promises text and whose bytes are text (a note quoting a PDF header).
    lenient_pdf = b"%PDF-" in head[:1024] and not (ext in _TEXT_EXTS and not _looks_binary(head))
    if body.lstrip().startswith(b"%PDF-") or (
        lenient_pdf and not head.startswith((_OLE2_MAGIC, b"PK\x03\x04", b"PK\x05\x06"))
    ):
        if container:
            raise nested("a PDF")
        kind = "pdf"
    elif head.startswith(_OLE2_MAGIC):
        raise IngestError("legacy_office_unsupported")
    elif head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        if container or zip_path is None:
            raise nested("a ZIP container")
        kind = _sniff_zip(zip_path, limits)
    elif head.startswith(_IMAGE_MAGIC) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        raise IngestError("image_needs_ocr")
    elif head[257:262] == b"ustar" or ext == ".tar":
        raise IngestError("archive_unsupported", "a tar archive — unpack it first")
    elif head.startswith(_ARCHIVE_MAGIC):
        raise IngestError("archive_unsupported", "7z/RAR/zstd archives are not read")
    elif body.lstrip().startswith(b"{\\rtf"):
        kind = "rtf"
    else:
        probe = head
        if _has_bom(head) or b"\x00" in head[:4096]:
            # decode_bytes raises binary_content for NUL-heavy non-UTF-16 bytes.
            probe_text, _enc, _fl = decode_bytes(head[: len(head) - (len(head) % 4)] or head)
            probe = probe_text.encode("utf-8", errors="replace")
        elif _looks_binary(head):
            raise IngestError("binary_content")
        stripped = probe.lstrip()
        if stripped.startswith(b"<"):
            kind = _xml_kind(stripped)
            if kind in ("xml", "") and ext in _HTML_EXTS:
                kind = "html"
            elif kind in ("xml", "") and ext in _EXT_DIALECTS:
                kind = _EXT_DIALECTS[ext]
            elif not kind:
                kind = "txt"
        elif stripped.startswith(b"WEBVTT") and stripped[6:7] in (b"", b" ", b"\t", b"\r", b"\n"):
            kind = "vtt"
        elif tuple(stripped.split(b"\n", 1)[0].rstrip(b"\r").decode("utf-8", "replace").lower().split("\t")) == (
            _TESSERACT_COLUMNS
        ):
            kind = "tesseract-tsv"
        elif ext in (".jsonl", ".ndjson"):
            kind = "jsonl"
        elif ext == ".json":
            kind = "json"
        elif ext in _EXT_DIALECTS:
            kind = _EXT_DIALECTS[ext]
        elif _looks_like_mbox(stripped):
            kind = "mbox"
        elif _looks_like_email(stripped):
            kind = "eml"
        elif _looks_like_srt(stripped.decode("utf-8", "replace").replace("\r\n", "\n")):
            kind = "srt"
        elif stripped.startswith((b"{", b"[")):
            kind = _json_or_text(full(), ext)
        else:
            kind = "txt"

    if ext in _IMAGE_EXTENSIONS and kind in ("txt",):
        raise IngestError("image_needs_ocr", f"{ext} file")
    spec = READERS.get(kind)
    if spec and ext and ext not in spec.extensions:
        notes.append(f"extension {ext} but content is {kind}")
    return kind


def _json_or_text(data: Optional[bytes], ext: str) -> str:
    """A `{`/`[`-leading file is JSON if it parses, JSONL if its first 50 records do."""
    if data is None:  # too large to probe
        return "json" if ext == ".json" else "txt"
    try:
        text, _enc, _fl = decode_bytes(data)
    except IngestError:
        return "txt"
    try:
        json.loads(text)
        return "json"
    except (ValueError, RecursionError):
        pass
    records = [ln for ln in text.splitlines() if ln.strip()]
    try:
        if records and all(isinstance(json.loads(r), (dict, list, str)) for r in records[:50]):
            return "jsonl"
    except (ValueError, RecursionError):
        pass
    return "txt"


# ── ZIP bundles: which members are page files ────────────────────────────────

#: Member extensions a bundle may carry as page files (content still decides the reader).
_BUNDLE_EXTS = frozenset(
    {".xml", ".txt", ".text", ".hocr", ".html", ".htm", ".xhtml", ".json", ".jsonl", ".ndjson", ".md", ".csv",
     ".tsv", ".tab", ".srt", ".vtt", ".eml", ".rtf", ".tei", ""}
)  # fmt: skip
_BUNDLE_METADATA_NAMES = frozenset({"mets.xml", "doc.xml", "metadata.xml", "manifest.xml", "mimetype"})
_IGNORED_BUNDLE_BASENAMES = frozenset({"thumbs.db", "desktop.ini", ".ds_store"})
_BUNDLE_METADATA_RE = re.compile(r"^(readme|licen[cs]e|copying|changelog|notice)(\..*)?$", re.IGNORECASE)
_BUNDLE_REFUSED_EXTS = {".pdf": "pdf_not_read_in_bundle"}
_BUNDLE_NESTED_EXTS = frozenset(
    {".zip", ".docx", ".docm", ".dotx", ".xlsx", ".xlsm", ".pptx", ".pptm", ".odt", ".ods", ".odp", ".epub",
     ".gz", ".gzip", ".bz2", ".xz", ".tgz", ".tar", ".7z", ".rar"}
)  # fmt: skip


def _bundle_member_kind(info: zipfile.ZipInfo) -> str:
    """By name only: `dir`, `metadata`, `image`, `refused` (a PDF or a nested
    container), `candidate` (a page file) or `unknown`."""
    if info.is_dir():
        return "dir"
    parts = [p for p in info.filename.split("/") if p]
    base = parts[-1] if parts else ""
    if (
        not base
        or parts[0] in ("__MACOSX", "META-INF")
        or any(p.startswith(".") for p in parts)
        or base.startswith(("._", "~$"))
        or base.lower() in _IGNORED_BUNDLE_BASENAMES
        or base.lower() in _BUNDLE_METADATA_NAMES
        or _BUNDLE_METADATA_RE.match(base)
    ):
        return "metadata"
    ext = posixpath.splitext(base)[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _BUNDLE_REFUSED_EXTS or ext in _BUNDLE_NESTED_EXTS:
        return "refused"
    if ext in _BUNDLE_EXTS:
        return "candidate"
    return "unknown"


def bundle_member_class(info: zipfile.ZipInfo) -> str:
    """`candidate` (a page file), `ignored` (metadata, images, unknown types) or
    `refused` (a PDF or a nested container: not read inside a bundle) — by name only."""
    kind = _bundle_member_kind(info)
    return kind if kind in ("candidate", "refused") else "ignored"


# ── readers: plain-text family ────────────────────────────────────────────────


def read_plain(text: str, ctx: "_Ctx") -> List[TextPage]:
    return [TextPage(lines) for lines in text_to_pages(text)]


_MD_FRONT_MATTER = re.compile(r"\A(?:---|\+\+\+)[ \t]*\n.*?\n(?:---|\+\+\+|\.\.\.)[ \t]*(?:\n|\Z)", re.DOTALL)
_MD_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_MD_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}(\s+|$)")
_MD_HEADING_TAIL = re.compile(r"\s+#+\s*$")
_MD_SETEXT = re.compile(r"^\s{0,3}(=+|-+)\s*$")
_MD_RULE = re.compile(r"^\s{0,3}([-*_])(\s*\1){2,}\s*$")
_MD_QUOTE = re.compile(r"^\s{0,3}(>\s?)+")
_MD_LIST = re.compile(r"^\s*(?:[-*+]|\d{1,9}[.)])\s+(?:\[[ xX]\]\s+)?")
_MD_REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+\S+")
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_MD_REFLINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_MD_AUTOLINK = re.compile(r"<((?:https?|ftp|mailto):[^>\s]+)>")
_MD_EMPH = re.compile(
    r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1|(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])|(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)"
)
_MD_CODE = re.compile(r"`+([^`]*)`+")
_MD_TAG = re.compile(r"</?[A-Za-z][^>]*>")
_MD_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")


def _md_inline(line: str) -> str:
    line = _MD_IMAGE.sub(r"\1", line)
    line = _MD_LINK.sub(r"\1", line)
    line = _MD_REFLINK.sub(r"\1", line)
    line = _MD_AUTOLINK.sub(r"\1", line)
    line = _MD_CODE.sub(r"\1", line)
    for _ in range(3):  # nested emphasis
        line = _MD_EMPH.sub(lambda m: m.group(2) or m.group(3) or m.group(4) or "", line)
    line = _MD_TAG.sub("", line)
    return _MD_ESCAPE.sub(r"\1", line)


def read_markdown(text: str, ctx: "_Ctx") -> List[TextPage]:
    """Markdown: front matter, fenced code, comments, rules and link definitions
    dropped; heading/list/quote markers and inline markup stripped; tables become
    tab-joined cells. Page breaks at form feeds, like plain text."""
    text = normalize_newlines(text)
    text = _MD_FRONT_MATTER.sub("", text, count=1)
    text = _MD_HTML_COMMENT.sub("", text)
    pages = []
    for page_lines in text_to_pages(text):
        out: List[str] = []
        fence = ""
        for line in page_lines:
            fence_match = _MD_FENCE.match(line)
            if fence:
                if fence_match and fence_match.group(1)[0] == fence[0] and len(fence_match.group(1)) >= len(fence):
                    fence = ""
                continue
            if fence_match:
                fence = fence_match.group(1)
                continue
            if _MD_SETEXT.match(line) and out and out[-1].strip():
                continue  # the underline of a setext heading
            if _MD_RULE.match(line) or _MD_REFDEF.match(line) or _MD_TABLE_SEP.match(line):
                continue
            line = _MD_QUOTE.sub("", line)
            if _MD_HEADING.match(line):
                line = _MD_HEADING_TAIL.sub("", _MD_HEADING.sub("", line))
            line = _MD_LIST.sub("", line)
            if line.strip().startswith("|") and line.strip().endswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                line = "\t".join(c for c in cells if c)
            out.append(_md_inline(line))
        pages.append(TextPage(out))
    return pages


_CSV_TEXT_COLUMNS = ("text", "line", "content", "transcription", "sentence", "string", "word", "token")
_CSV_PAGE_COLUMNS = ("page_num", "page", "page_number", "pagenumber", "page_no")
#: Numeric columns that address a line (and its block/paragraph): consecutive rows
#: that share one address are the words of one line and are joined by spaces.
_CSV_LINE_COLUMNS = ("line_num", "line_number", "line_no", "lineno", "linenumber", "line_id", "line")
_CSV_BLOCK_COLUMNS = ("block_num", "block")
_CSV_PAR_COLUMNS = ("par_num", "paragraph", "par")
_CSV_NUMERIC = re.compile(r"^[+-]?\d+(?:[.,]\d+)?$")
_CSV_PROBE_ROWS = 2000
_csv_field_limit_raised = False


def _raise_csv_field_limit() -> None:
    """Allow CSV cells up to the file-size cap. Raised once and never restored:
    restoring a process-wide setting per call raced between the service's threads."""
    global _csv_field_limit_raised
    if not _csv_field_limit_raised:
        csv.field_size_limit(max(csv.field_size_limit(), min(sys.maxsize, 2**31 - 1)))
        _csv_field_limit_raised = True


def _csv_rows(text: str, delimiter: str, ctx: "_Ctx", tsv: bool = False) -> List[List[str]]:
    """Rows of a delimited file, safe against a stray quote.

    A `"` that opens a field and never closes swallows every following row into one
    cell. CSV: an odd quote count, or a strict parse that fails, re-parses with no
    quoting at all (noted `csv_unbalanced_quote`, so the document is `partial`). TSV
    has no quoting convention (a lone `"` is a common OCR token), so it is read
    unquoted unless its quoting is well-formed.
    """
    _raise_csv_field_limit()
    quotes = text.count('"')
    if quotes and quotes % 2 == 0:
        try:
            rows = list(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
        except csv.Error:
            rows = None
        if rows is not None and not (tsv and any("\t" in c and "\n" in c for r in rows for c in r)):
            return rows
    elif not quotes:
        return list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if not tsv:
        ctx.notes.append("csv_unbalanced_quote")
    return list(csv.reader(io.StringIO(text), delimiter=delimiter, quoting=csv.QUOTE_NONE))


def read_csv_table(text: str, ctx: "_Ctx", delimiter: Optional[str] = None) -> List[TextPage]:
    """CSV/TSV: a recognised text column (text/line/content/transcription/…) gives
    one line per row, grouped into pages by a page/page_num column in first-seen
    order; without one, every row is a line of its non-empty cells joined by tab.

    A text column holding only numbers (a `line` column of line numbers) is not the
    text. Rows sharing one numeric (page, block, paragraph, line) address — one word
    per row, as OCR engines export — are joined into their line.
    """
    text = normalize_newlines(text).replace("\x00", "")
    tsv = delimiter == "\t"
    if delimiter is None:
        sample = text[:65536]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    try:
        rows = _csv_rows(text, delimiter, ctx, tsv=tsv)
    except csv.Error as exc:
        raise IngestError("malformed", f"CSV could not be parsed ({exc})") from exc
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return [TextPage([])]

    header = [c.strip().lower() for c in rows[0]]
    body = rows[1:]

    def numeric_only(col: int) -> bool:
        values = [r[col].strip() for r in body[:_CSV_PROBE_ROWS] if col < len(r) and r[col].strip()]
        return bool(values) and all(_CSV_NUMERIC.match(v) for v in values)

    text_col = next(
        (header.index(c) for c in _CSV_TEXT_COLUMNS if c in header and not numeric_only(header.index(c))), None
    )
    if text_col is None:
        return [TextPage(["\t".join(c.strip() for c in r if c.strip()) for r in rows])]

    def address_col(names) -> Optional[int]:
        return next(
            (
                header.index(c)
                for c in names
                if c in header and header.index(c) != text_col and numeric_only(header.index(c))
            ),
            None,
        )

    page_col = next((header.index(c) for c in _CSV_PAGE_COLUMNS if c in header), None)
    line_col = address_col(_CSV_LINE_COLUMNS)
    block_col = address_col(_CSV_BLOCK_COLUMNS) if line_col is not None else None
    par_col = address_col(_CSV_PAR_COLUMNS) if line_col is not None else None
    ctx.notes.append(f"csv text column {header[text_col]!r}")

    def cell(r: List[str], col: Optional[int]) -> str:
        return r[col].strip() if col is not None and col < len(r) else ""

    pages: Dict[str, List[str]] = {}
    last_key: Optional[Tuple[str, str, str, str]] = None
    joined = 0
    for r in body:
        value = r[text_col] if text_col < len(r) else ""
        page = cell(r, page_col) or "1"
        lines = pages.setdefault(page, [])
        key = (page, cell(r, block_col), cell(r, par_col), cell(r, line_col)) if line_col is not None else None
        if key is not None and key == last_key and lines:
            if value.strip():
                lines[-1] = f"{lines[-1].strip()} {value.strip()}".strip()
                joined += 1
        else:
            lines.append(value)
        last_key = key
    if joined:
        ctx.notes.append(f"csv_rows_joined={joined}")
    return [TextPage(lines, label=label) for label, lines in pages.items()]


def read_tesseract_tsv(text: str, ctx: "_Ctx") -> List[TextPage]:
    """Tesseract's TSV output: level-5 rows are words, grouped into lines by their
    (page, block, par, line) numbers; pages are `page_num` (every level-1 page row
    opens one, so a page without words is kept). Never quoted: `"` is a word there."""
    _raise_csv_field_limit()
    rows = list(csv.reader(io.StringIO(normalize_newlines(text).replace("\x00", "")), delimiter="\t",
                           quoting=csv.QUOTE_NONE))  # fmt: skip
    pages: Dict[str, List[str]] = {}
    words: Dict[Tuple[int, int, int, int], List[str]] = {}
    bad = 0
    for r in rows[1:]:
        if not any(c.strip() for c in r):
            continue
        r = r + [""] * (12 - len(r))
        try:
            level, page, block, par, line = (int(v) for v in r[:5])
        except ValueError:
            bad += 1
            continue
        pages.setdefault(str(page), [])
        if level != 5:
            continue
        word = "\t".join(r[11:]).strip()
        if word:
            words.setdefault((page, block, par, line), []).append(word)
    for (page, _b, _p, _l), line_words in words.items():
        pages.setdefault(str(page), []).append(" ".join(line_words))
    if bad:
        ctx.notes.append(f"tsv_bad_rows={bad}")
    return [TextPage(lines, label=label) for label, lines in pages.items()] or [TextPage([])]


# ── readers: JSON family ──────────────────────────────────────────────────────

_WORD_LEVEL_KEYS = {"words", "word", "strings", "string", "tokens", "token", "glyphs", "symbols", "chars", "characters"}
_LINE_LEVEL_KEYS = {"lines", "line", "textlines", "textline", "text_lines", "text_line"}

#: The text-key whitelist of extract_JSON_2_TXT.TARGET_KEYS, copied rather than
#: imported: that module imports pandas and reads the config at import time, which
#: made every JSON read fail as `malformed` where pandas is absent (the service
#: image). tests/test_text_formats.py pins the two sets equal.
_JSON_TEXT_KEYS = frozenset(
    {"content", "text", "string", "textline", "line", "word", "lines", "words", "strings", "textlines", "textstring",
     "textstrings", "contents", "data", "texts", "pagetext", "page_text", "text_string", "text_line", "text_strings",
     "page_texts", "text_lines"}
)  # fmt: skip
#: Keys whose string value is the text of the object that carries it (a page's full
#: text, a paragraph's content) — coarser than a line.
_JSON_COARSE_KEYS = frozenset(_JSON_TEXT_KEYS - _LINE_LEVEL_KEYS - _WORD_LEVEL_KEYS)
_JSON_OWN_TEXT_KEYS = frozenset(_JSON_COARSE_KEYS | _LINE_LEVEL_KEYS)
#: Fields that type the elements of a flat list (AWS Textract `BlockType`, …).
_JSON_TYPE_FIELDS = ("blocktype", "block_type", "type", "level", "kind", "granularity")
_JSON_LINE_TYPES = frozenset({"line", "textline", "text_line"})
_JSON_WORD_TYPES = frozenset({"word", "token", "symbol", "char", "character", "glyph"})
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_HEX_RE = re.compile(r"[0-9a-fA-F]+")


class _JsonStats:
    """What the JSON walk left out, reported once per document as notes."""

    __slots__ = ("words", "coarse", "blobs", "non_text")

    def __init__(self):
        self.words = self.coarse = self.blobs = self.non_text = 0

    def emit(self, notes: List[str]) -> None:
        for name, value in (
            ("json_word_leaves_skipped", self.words),
            ("json_coarse_leaves_skipped", self.coarse),
            ("json_blobs_skipped", self.blobs),
            ("json_non_text_strings", self.non_text),
        ):
            if value:
                notes.append(f"{name}={value}")


def _is_blob(text: str) -> bool:
    """Embedded binary rather than text: a data: URI, base64, a hex digest, or a long
    run without whitespace (page images and hashes some engines inline)."""
    if text.startswith("data:") and ";base64," in text[:200]:
        return True
    if len(text) >= 256 and not any(ch.isspace() for ch in text):
        return True
    if (
        len(text) >= 64
        and len(text) % 4 == 0
        and _BASE64_RE.fullmatch(text)
        and any(c.isdigit() for c in text)
        and any(c.isupper() for c in text)
        and any(c.islower() for c in text)
    ):
        return True
    return len(text) >= 32 and bool(_HEX_RE.fullmatch(text))


def _json_type(item: Dict[str, Any]) -> str:
    for k, v in item.items():
        if isinstance(k, str) and k.lower() in _JSON_TYPE_FIELDS and isinstance(v, str):
            return v.strip().lower()
    return ""


def _is_line_container(key: Any, value: Any) -> bool:
    return (
        isinstance(key, str)
        and key.lower() in _LINE_LEVEL_KEYS
        and isinstance(value, (list, dict, str))
        and bool(value)
    )


def _line_bearing(data: Any) -> set:
    """ids of the containers that hold a non-empty line-level container at or below them."""
    parent: Dict[int, Any] = {}
    marked: set = set()
    stack = [data]
    while stack:
        node = stack.pop()
        children = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
        for _k, v in children:
            if isinstance(v, (dict, list)):
                parent[id(v)] = node
                stack.append(v)
        if isinstance(node, dict) and any(_is_line_container(k, v) for k, v in node.items()):
            cur = node
            while cur is not None and id(cur) not in marked:
                marked.add(id(cur))
                cur = parent.get(id(cur))
    return marked


def _count_strings(value: Any) -> int:
    n, stack = 0, [value]
    while stack:
        node = stack.pop()
        if isinstance(node, str):
            n += bool(node.strip())
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return n


def _json_leaves(
    data: Any, keys: Optional[frozenset], current_key: Optional[str] = None, stats: Optional[_JsonStats] = None
) -> List[str]:
    """Ordered text leaves under whitelisted keys (`keys=None`: every string with a letter).

    Line granularity wins, so an engine that emits the same text at several levels
    yields it once:
    * a dict with a line-level container (`lines`) skips its word containers, and its
      own coarse text (a page's or paragraph's full `text`/`content`) — also when the
      lines sit deeper below it;
    * a dict with its own text skips its word containers (Azure Read's line objects);
    * a line object with its own text suppresses word containers anywhere below it;
      one without its own text becomes ONE line of its words (docTR's `value`s);
    * a flat list typed LINE and WORD (AWS Textract `Blocks`) keeps the non-word items.
    Base64, hex digests and other embedded binary strings are skipped.
    """
    stats = stats if stats is not None else _JsonStats()
    line_bearing = _line_bearing(data)
    out: List[str] = []

    def emit(text: str, key: Optional[str]) -> None:
        text = text.strip()
        if not text:
            return
        if keys is None:
            if not any(ch.isalpha() for ch in text):
                return
        elif key is None or str(key).lower() not in keys:
            if any(ch.isalpha() for ch in text):
                stats.non_text += 1
            return
        if _is_blob(text):
            stats.blobs += 1
            return
        out.extend(normalize_newlines(text).split("\n"))

    # (node, key, suppress_words_below, reached_through_a_line_key)
    stack: List[Tuple[Any, Optional[str], bool, bool]] = [(data, current_key, False, False)]
    while stack:
        node, key, suppress, via_line = stack.pop()
        if isinstance(node, str):
            emit(node, key)
            continue
        if isinstance(node, list):
            items = list(node)
            typed = [_json_type(x) for x in items if isinstance(x, dict)]
            if set(typed) & _JSON_LINE_TYPES and set(typed) & _JSON_WORD_TYPES:
                kept = [x for x in items if not (isinstance(x, dict) and _json_type(x) in _JSON_WORD_TYPES)]
                stats.words += len(items) - len(kept)
                items = kept
            stack.extend((item, key, suppress, via_line) for item in reversed(items))
            continue
        if not isinstance(node, dict):
            continue
        own_text = any(
            isinstance(k, str) and k.lower() in _JSON_OWN_TEXT_KEYS and isinstance(v, str) and v.strip()
            for k, v in node.items()
        )
        has_lines = any(_is_line_container(k, v) for k, v in node.items())
        if via_line and not own_text and not has_lines:
            words = _json_leaves_flat(node, keys, key, stats)
            if words:
                out.append(" ".join(words))
                continue
        inherit = suppress or (via_line and own_text)
        skip_words = inherit or has_lines or own_text
        skip_coarse = id(node) in line_bearing
        items = []
        for k, v in node.items():
            lk = k.lower() if isinstance(k, str) else ""
            if skip_words and lk in _WORD_LEVEL_KEYS and (has_lines or isinstance(v, (list, dict))):
                stats.words += _count_strings(v)
                continue
            if skip_coarse and lk in _JSON_COARSE_KEYS and isinstance(v, str) and v.strip():
                stats.coarse += 1
                continue
            items.append((v, k, inherit, lk in _LINE_LEVEL_KEYS))
        stack.extend(reversed(items))
    return out


def _json_leaves_flat(node: Any, keys: Optional[frozenset], key: Optional[str], stats: _JsonStats) -> List[str]:
    """Every emitted leaf below `node`, in order, with no granularity rules (a line's words)."""
    out: List[str] = []
    stack: List[Tuple[Any, Optional[str]]] = [(node, key)]
    while stack:
        cur, k = stack.pop()
        if isinstance(cur, dict):
            stack.extend(reversed([(v, kk) for kk, v in cur.items()]))
        elif isinstance(cur, list):
            stack.extend((item, k) for item in reversed(cur))
        elif isinstance(cur, str):
            text = cur.strip()
            if not text:
                continue
            if keys is None:
                if not any(ch.isalpha() for ch in text):
                    continue
            elif k is None or str(k).lower() not in keys:
                continue
            if _is_blob(text):
                stats.blobs += 1
                continue
            out.append(_collapse_ws(text))
    return out


def _json_pages(data: Any, ctx: "_Ctx", keys: Optional[frozenset]) -> List[TextPage]:
    """Pages for one parsed JSON value: Family A/B (same detection as json-keys),
    else top-level children as blocks. Header siblings are NOT copied into every
    page (json-keys does, because it re-serialises whole documents per page)."""
    from page_split import PAGE_NUMBER_FIELD_KEYS, _find_family_a, _find_family_b, _get_field_ci

    stats = ctx.json_stats
    if isinstance(data, list) and data and all(isinstance(x, str) for x in data):
        ctx.notes.append("json_string_array")
        return [TextPage([ln for x in data for ln in normalize_newlines(x).split("\n")], label="1")]

    family_a = _find_family_a(data)
    if family_a is not None:
        _parent, _key, page_list = family_a
        pages = []
        for i, page_obj in enumerate(page_list, 1):
            number = _get_field_ci(page_obj, PAGE_NUMBER_FIELD_KEYS)
            pages.append(
                TextPage(_json_leaves(page_obj, keys, stats=stats), label=str(number) if number is not None else str(i))
            )
        ctx.notes.append("json page list")
        return pages

    family_b = _find_family_b(data)
    if family_b is not None:
        _parent, _key, _lst, _field, groups = family_b
        ctx.notes.append("json page-tagged list")
        return [TextPage(_json_leaves(items, keys, stats=stats), label=str(value)) for value, items in groups.items()]

    blocks: List[Tuple[str, Any]] = []
    if isinstance(data, list) and len(data) >= 2:
        blocks = [(str(i), item) for i, item in enumerate(data, 1)]
    elif isinstance(data, dict):
        blocks = [(str(k), v) for k, v in data.items() if isinstance(v, (dict, list))]
    block_pages = []
    for label, value in blocks:
        leaves = _json_leaves(value, keys, stats=stats)
        if leaves:
            block_pages.append(TextPage(leaves, label=label))
    if len(block_pages) >= 2:
        ctx.notes.append("json top-level blocks")
        return block_pages
    ctx.json_stats = stats = _JsonStats()  # the blocks' counts are superseded by the whole-document walk
    return [TextPage(_json_leaves(data, keys, stats=stats), label="1")]


def _with_fallback_keys(build: Callable[[Optional[frozenset]], List[TextPage]], ctx: "_Ctx") -> List[TextPage]:
    ctx.json_stats = _JsonStats()
    pages = build(_JSON_TEXT_KEYS)
    if not any(ln.strip() for p in pages for ln in p.lines):
        ctx.notes.append("json_all_strings")
        ctx.json_stats = _JsonStats()
        pages = build(None)
    ctx.json_stats.emit(ctx.notes)
    return pages


def read_json(text: str, ctx: "_Ctx") -> List[TextPage]:
    try:
        data = json.loads(text)
    except RecursionError as exc:
        raise IngestError("malformed", "JSON nested too deeply") from exc
    except ValueError as exc:
        if "Extra data" in str(exc):
            # A .json file that is really JSON Lines (one record per line) — common.
            try:
                pages = read_jsonl(text, ctx)
            except IngestError:
                pages = None
            if pages is not None:
                ctx.notes.append("json_read_as_jsonl")
                ctx.kind = "jsonl"
                return pages
        raise IngestError("malformed", f"JSON could not be parsed ({str(exc)[:200]})") from exc
    if isinstance(data, str):
        return [TextPage(normalize_newlines(data).split("\n"), label="1")]
    return _with_fallback_keys(lambda keys: _json_pages(data, ctx, keys), ctx)


def read_jsonl(text: str, ctx: "_Ctx") -> List[TextPage]:
    """JSON Lines: one record = one page (block); bad records are counted and skipped."""
    records: List[Tuple[str, Any]] = []
    bad = 0
    for n, raw in enumerate(normalize_newlines(text).split("\n"), 1):
        if not raw.strip():
            continue
        try:
            records.append((str(n), json.loads(raw)))
        except (ValueError, RecursionError):
            bad += 1
    if not records:
        raise IngestError("malformed", "no JSON Lines record could be parsed")
    if bad:
        ctx.notes.append(f"jsonl_bad_records={bad}")

    def build(keys):
        pages = []
        for label, value in records:
            if isinstance(value, str):
                leaves = [value.strip()]
            elif isinstance(value, list) and value and all(isinstance(x, str) for x in value):
                leaves = [x.strip() for x in value]
            else:
                leaves = _json_leaves(value, keys, stats=ctx.json_stats)
            if any(leaves):
                pages.append(TextPage(leaves, label=label))
        return pages or [TextPage([])]

    return _with_fallback_keys(build, ctx)


# ── readers: XML / HTML family ────────────────────────────────────────────────


def read_alto(root, ctx: "_Ctx") -> List[TextPage]:
    """ALTO v2/v3/v4 or namespace-less: Page → TextLine → String@CONTENT, HYP → '-'.

    A deliberately small reader of its own (not alto_tools.alto_text, which keys
    blocks by ID and needs a document-level ReadingOrder): document order, one line
    per TextLine — the ALTO methods of the pipeline remain the way to get reading-
    order reconstruction and dehyphenation.
    """
    pages_el = list(root.iter("{*}Page")) or [root]
    pages = []
    for i, page in enumerate(pages_el, 1):
        lines = []
        for tl in page.iter("{*}TextLine"):
            parts: List[str] = []
            for child in tl:
                name = _local(child.tag)
                if name == "String":
                    parts.append(child.get("CONTENT", ""))
                elif name == "HYP" and parts:
                    parts[-1] = parts[-1] + "-"
            lines.append(" ".join(p for p in parts if p))
        label = page.get("PHYSICAL_IMG_NR") or page.get("ID") or str(i) if page is not root else "1"
        pages.append(TextPage(lines, label=str(label)))
    return pages


def _page_xml_text(el) -> str:
    """Text of a PAGE TextLine/Word: its first TextEquiv (lowest @index) Unicode."""
    equivs = [e for e in el if _local(e.tag) == "TextEquiv"]
    if not equivs:
        return ""

    def _index(e):
        try:
            return int(e.get("index", "0"))
        except ValueError:
            return 0

    best = min(equivs, key=_index)
    uni = next((u for u in best if _local(u.tag) == "Unicode"), None)
    return (uni.text or "") if uni is not None else ""


def _page_reading_order(page) -> List[str]:
    order: List[str] = []
    ro = next((e for e in page if _local(e.tag) == "ReadingOrder"), None)
    if ro is None:
        return order

    def _visit(group):
        refs = []
        for child in group:
            name = _local(child.tag)
            if name in ("RegionRefIndexed", "OrderedGroupIndexed", "UnorderedGroupIndexed"):
                try:
                    idx = int(child.get("index", "0"))
                except ValueError:
                    idx = 0
                refs.append((idx, child))
            elif name in ("RegionRef", "OrderedGroup", "UnorderedGroup"):
                refs.append((len(refs), child))
        for _idx, child in sorted(refs, key=lambda t: t[0]):
            name = _local(child.tag)
            if name.startswith("RegionRef"):
                order.append(child.get("regionRef", ""))
            else:
                if child.get("regionRef"):
                    order.append(child.get("regionRef", ""))
                _visit(child)

    for group in ro:
        _visit(group)
    return order


def _page_region_like(el) -> bool:
    """A PAGE region that can hold text lines: any `*Region`, or a Transkribus `TableCell`."""
    name = _local(el.tag)
    return name.endswith("Region") or name == "TableCell"


def _page_cell_position(el) -> Optional[Tuple[int, int]]:
    """(row, col) of a table cell: Transkribus `TableCell@row/@col`, or PAGE 2019's
    `Roles/TableCellRole@rowIndex/@columnIndex` on a TextRegion inside a TableRegion."""
    row, col = el.get("row"), el.get("col")
    if row is None or col is None:
        role = next((r for r in el.iter("{*}TableCellRole")), None)
        if role is not None:
            row, col = role.get("rowIndex"), role.get("columnIndex")
    try:
        return int(row), int(col)
    except (TypeError, ValueError):
        return None


def _page_parent_region(el, page):
    parent = el.getparent()
    while parent is not None and parent is not page:
        if _page_region_like(parent):
            return parent
        parent = parent.getparent()
    return None


def _page_region_lines(region, referenced: set, done: set, lines: List[str]) -> None:
    """Append one region's lines in reading order: its TextLines and the nested
    regions the ReadingOrder does not name, in document order (table cells
    row-major); a region with neither contributes its own TextEquiv.

    `done` holds the elements themselves, not their ids: lxml makes element proxies
    on demand and frees them, so an id can be reused by another element, and an
    id-keyed set skipped lines at random. A held proxy stays the node's only proxy.
    """
    if region in done:
        return
    done.add(region)
    children = [c for c in region if isinstance(c.tag, str)]
    items = [
        c for c in children if _local(c.tag) == "TextLine" or (_page_region_like(c) and c.get("id") not in referenced)
    ]
    if not items:
        if not any(_page_region_like(c) for c in children):
            text = _page_xml_text(region)
            lines.extend(normalize_newlines(text).split("\n") if text else [])
        return
    if _local(region.tag) == "TableRegion":
        positions = [_page_cell_position(c) if _page_region_like(c) else None for c in items]
        if all(pos is not None for pos in positions):
            items = [c for _pos, c in sorted(zip(positions, items, strict=True), key=lambda t: t[0])]
    for child in items:
        if _local(child.tag) != "TextLine":
            _page_region_lines(child, referenced, done, lines)
            continue
        if child in done:
            continue
        done.add(child)
        text = _page_xml_text(child)
        if not text:
            words = [_page_xml_text(w) for w in child if _local(w.tag) == "Word"]
            text = " ".join(w for w in words if w)
        lines.append(text)


def read_page_xml(root, ctx: "_Ctx") -> List[TextPage]:
    """PAGE XML: ReadingOrder → region → TextLine (TextEquiv/Unicode, else Words).

    Every region kind that holds lines is read — TextRegion, TableRegion and its cells
    (Transkribus `TableCell`, row-major), nested regions. A nested region that the
    ReadingOrder does not name is read in place, inside its parent; one it names is
    read at its ReadingOrder position.
    """
    pages = []
    for i, page in enumerate(root.iter("{*}Page"), 1):
        regions = [el for el in page.iter() if isinstance(el.tag, str) and _page_region_like(el)]
        by_id = {r.get("id"): r for r in regions if r.get("id")}
        ordered_ids = [rid for rid in _page_reading_order(page) if rid in by_id]
        referenced = set(ordered_ids)
        sequence = [by_id[rid] for rid in ordered_ids]
        in_sequence = set(sequence)
        sequence += [r for r in regions if _page_parent_region(r, page) is None and r not in in_sequence]
        lines: List[str] = []
        done: set = set()
        for region in sequence:
            _page_region_lines(region, referenced, done, lines)
        label = page.get("imageFilename") or str(i)
        pages.append(TextPage(lines, label=os.path.splitext(os.path.basename(label))[0] or str(i)))
    return pages or [TextPage([])]


_TEI_BLOCKS = {
    "p", "head", "l", "item", "cell", "ab", "div", "lg", "list", "table", "row", "note", "label", "trailer",
    "byline", "dateline", "opener", "closer", "salute", "signed", "fw", "argument", "epigraph", "docTitle",
    "titlePart", "docAuthor", "docDate", "castItem", "sp", "speaker", "stage", "figDesc", "bibl", "u", "s",
}  # fmt: skip
_TEI_SKIP = {"teiHeader", "facsimile", "standOff", "sourceDoc"}


def _flow_read(
    root, blocks, skip_names, line_break, page_break, collapse=True, extra_start=None, skip_fn=None
) -> List[TextPage]:
    flow = _Flow(collapse=collapse)

    def on_start(el):
        name = _local(el.tag)
        if extra_start is not None and extra_start(el, flow):
            return
        if name in page_break:
            flow.explicit_break()
        elif name in line_break:
            flow.line_break()
        elif name in blocks:
            flow.line_break()

    def on_end(el):
        if _local(el.tag) in blocks:
            flow.line_break()

    def skip(el):
        return skip_fn(el) if skip_fn is not None else _local(el.tag) in skip_names

    _walk(root, on_start, on_end, flow.text, skip)
    return flow.finish()


#: In `<choice>`, the editorial reading is kept and its source form skipped
#: (`<sic>`/`<corr>`, `<orig>`/`<reg>`, `<abbr>`/`<expan>`); either alone is text.
_TEI_CHOICE_PARTNERS = {"sic": "corr", "orig": "reg", "abbr": "expan"}


def _tei_skip(el) -> bool:
    name = _local(el.tag)
    if name in _TEI_SKIP:
        return True
    partner = _TEI_CHOICE_PARTNERS.get(name)
    if partner:
        parent = el.getparent()
        return parent is not None and _local(parent.tag) == "choice" and any(_local(c.tag) == partner for c in parent)
    return False


def read_tei(root, ctx: "_Ctx") -> List[TextPage]:
    """TEI/TEITOK: pages at <pb/>, lines at <lb/> and block ends; header skipped.

    Every `<pb/>` starts a page (text before the first one is page 1), empty pages
    included, labelled with `pb@n` ("I", "7a"; the page's ordinal when it has none) -- the
    pages atrium-nlp-enrich's TEITOK reader and layout reader count, so a table made from a
    TEITOK file lines up with that file's layout. In a tokenized TEITOK document
    (`<tok>` and `<lb/>`), `<s>` is a sentence, not a line: lines are the `<lb/>` ones, and a
    sentence running over a line or page break (nlp-enrich puts `<lb/>`/`<pb/>` inside it)
    is split there. A `teiCorpus` is read text by text (every outermost `<text>`, P4's
    `TEI.2` included); in `<choice>` the corrected/regularised/expanded form is read.
    """
    texts = [e for e in root.iter("{*}text") if not any(_local(a.tag) == "text" for a in e.iterancestors())]
    pages: List[TextPage] = []
    for scope in texts or [root]:
        names = {_local(e.tag) for e in scope.iter() if isinstance(e.tag, str)}
        blocks = _TEI_BLOCKS - {"s"} if {"tok", "lb"} <= names else _TEI_BLOCKS
        seen_pb = [False]

        def on_pb(el, flow, seen_pb=seen_pb):
            if _local(el.tag) != "pb":
                return False
            flow.line_break()
            if seen_pb[0] or flow.pages[-1].lines or len(flow.pages) > 1:
                flow.pages.append(TextPage([]))
            seen_pb[0] = True
            flow.pages[-1].label = (el.get("n") or "").strip()
            return True

        pages.extend(_flow_read(scope, blocks, _TEI_SKIP, {"lb"}, {"pb"}, extra_start=on_pb, skip_fn=_tei_skip))
    for n, page in enumerate(pages, 1):
        page.label = page.label or str(n)
    return pages


#: Element names that ARE a line in OCR-ish XML dialects without a reader of their
#: own; their word/character children are joined instead of each becoming a line.
_XML_LINE_NAMES = frozenset({"line", "textline", "text_line", "text-line"})
_XML_TEXT_ATTRS = ("CONTENT", "content", "text", "value")


def _xml_line_text(el) -> str:
    """One line from a line element: its mixed text, else its leaves — characters
    concatenated when every leaf is a single character, words joined by spaces."""
    if (el.text or "").strip() or any((c.tail or "").strip() for c in el):
        return _collapse_ws("".join(el.itertext()))
    leaves = []
    for node in el.iter():
        if node is el or not isinstance(node.tag, str) or any(isinstance(c.tag, str) for c in node):
            continue
        text = node.text if node.text is not None else next((node.get(a) for a in _XML_TEXT_ATTRS if node.get(a)), "")
        leaves.append(text or "")
    if leaves and all(len(t) <= 1 for t in leaves):
        return _collapse_ws("".join(leaves))
    return " ".join(t.strip() for t in leaves if t.strip())


def read_generic_xml(root, ctx: "_Ctx") -> List[TextPage]:
    """Generic XML: root children are blocks (pages) when two or more carry text;
    an element with its own (mixed) text is one line, containers descend; an element
    named line/textline is one line of its word or character children."""

    def lines_of(el) -> List[str]:
        out: List[str] = []
        stack = [el]
        while stack:
            node = stack.pop()
            if not isinstance(node.tag, str):
                continue
            if _local(node.tag).lower() in _XML_LINE_NAMES:
                out.append(_xml_line_text(node))
                continue
            own = (node.text or "").strip() or any((c.tail or "").strip() for c in node)
            if own:
                out.append(_collapse_ws("".join(t for t in node.itertext())))
                continue
            stack.extend(reversed([c for c in node if isinstance(c.tag, str)]))
        return [ln for ln in out if ln]

    children = [c for c in root if isinstance(c.tag, str)]
    if not (root.text or "").strip():
        blocks = []
        line_children = True
        for i, child in enumerate(children, 1):
            lines = lines_of(child)
            if lines:
                blocks.append(TextPage(lines, label=f"{_local(child.tag)}[{i}]"))
                line_children = line_children and _local(child.tag).lower() in _XML_LINE_NAMES
        if len(blocks) >= 2 and not line_children:
            return blocks
    return [TextPage(lines_of(root), label="1")]


def read_abbyy(root, ctx: "_Ctx") -> List[TextPage]:
    """ABBYY FineReader XML (6–12): `page` = page, `line` = line. A line's text is its
    `charParams` concatenated (spaces are characters there; a missing one is restored
    at `wordStart`), FineReader 6's plain `formatting` text otherwise; a line-final
    `¬` is FineReader's hyphen. Table blocks are read row by row, in document order."""
    pages = []
    for i, page in enumerate(root.iter("{*}page"), 1):
        lines = [t for t in (_abbyy_line_text(line) for line in page.iter("{*}line")) if t]
        pages.append(TextPage(lines, label=str(i)))
    return pages or [TextPage([])]


def _abbyy_line_text(line) -> str:
    parts: List[str] = []
    for child in line:
        name = _local(child.tag)
        if name == "charParams":
            chars = [child]
        elif name == "formatting":
            chars = [c for c in child if _local(c.tag) == "charParams"]
            if not chars:
                parts.append(child.text or "")
                continue
        else:
            continue
        for cp in chars:
            ch = cp.text or ""
            if (cp.get("wordStart") or "").lower() in ("true", "1") and parts and not parts[-1].endswith(" "):
                if ch != " ":
                    parts.append(" ")
            parts.append(ch)
    text = _collapse_ws("".join(parts))
    return text[:-1] + "-" if text.endswith("\u00ac") else text


def read_djvu(root, ctx: "_Ctx") -> List[TextPage]:
    """DjVuXML (djvutoxml): `OBJECT` = page (label from its PAGE param or usemap),
    `LINE` = line of its `WORD`s; a page without LINEs reads its PARAGRAPHs."""

    def named(el, name) -> bool:
        return isinstance(el.tag, str) and _local(el.tag).upper() == name

    pages = []
    for i, obj in enumerate((el for el in root.iter() if named(el, "OBJECT")), 1):
        label = ""
        for param in obj:
            if named(param, "PARAM") and (param.get("name") or "").upper() == "PAGE":
                label = os.path.splitext(posixpath.basename(param.get("value") or ""))[0]
        if not label and obj.get("usemap"):
            label = os.path.splitext(posixpath.basename(obj.get("usemap") or ""))[0]
        lines = []
        line_els = [el for el in obj.iter() if named(el, "LINE")]
        for el in line_els or [el for el in obj.iter() if named(el, "PARAGRAPH")]:
            words = [_collapse_ws("".join(w.itertext())) for w in el.iter() if named(w, "WORD")]
            text = " ".join(w for w in words if w) or _collapse_ws("".join(el.itertext()))
            if text:
                lines.append(text)
        pages.append(TextPage(lines, label=label or str(i)))
    return pages or [TextPage([])]


_HTML_BLOCKS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "td", "th", "tr", "blockquote", "section",
    "article", "header", "footer", "nav", "aside", "figcaption", "caption", "address", "table", "ul", "ol", "dl",
    "form", "fieldset", "legend", "main", "figure", "details", "summary", "title", "center", "hr", "option",
}  # fmt: skip
_HTML_SKIP = {"script", "style", "noscript", "template", "head", "svg", "canvas", "iframe", "object", "embed"}
_CSS_BREAK_BEFORE = re.compile(r"(?:page-break-before|break-before)\s*:\s*(?:always|page|left|right)", re.I)
_CSS_BREAK_AFTER = re.compile(r"(?:page-break-after|break-after)\s*:\s*(?:always|page|left|right)", re.I)
_HOCR_LINE_CLASSES = {"ocr_line", "ocrx_line", "ocr_caption", "ocr_header", "ocr_textfloat"}


def _classes(el) -> set:
    return set((el.get("class") or "").split())


def _html_flow(root) -> List[TextPage]:
    flow = _Flow()
    pre_depth = [0]

    def on_start(el):
        name = _local(el.tag).lower()
        style = el.get("style") or ""
        if style and _CSS_BREAK_BEFORE.search(style):
            flow.explicit_break()
        if name == "br":
            flow.line_break()
        elif name == "pre":
            flow.line_break()
            pre_depth[0] += 1
        elif name in _HTML_BLOCKS:
            flow.line_break()

    def on_end(el):
        name = _local(el.tag).lower()
        if name == "pre":
            pre_depth[0] -= 1
            flow.line_break()
        elif name in _HTML_BLOCKS:
            flow.line_break()
        style = el.get("style") or ""
        if style and _CSS_BREAK_AFTER.search(style):
            flow.explicit_break()

    def on_text(s):
        if pre_depth[0] > 0 and "\n" in s:
            parts = s.split("\n")
            for part in parts[:-1]:
                flow.text(part)
                flow.line_break()
            flow.text(parts[-1])
        else:
            flow.text(s)

    def skip(el):
        return _local(el.tag).lower() in _HTML_SKIP

    body = root.find(".//body")
    _walk(body if body is not None else root, on_start, on_end, on_text, skip)
    return flow.finish()


def read_html(text: str, ctx: "_Ctx") -> List[TextPage]:
    root = parse_html(text)
    pages_el = [el for el in root.iter() if isinstance(el.tag, str) and "ocr_page" in _classes(el)]
    if pages_el:
        ctx.kind = "hocr"
        pages = []
        for i, page in enumerate(pages_el, 1):
            lines = [
                _collapse_ws(" ".join(el.itertext()))
                for el in page.iter()
                if isinstance(el.tag, str) and _classes(el) & _HOCR_LINE_CLASSES
            ]
            if not lines:
                lines = [ln for p in _html_flow(page) for ln in p.lines]
            title = page.get("title") or ""
            match = re.search(r"ppageno\s+(\d+)", title)
            label = str(int(match.group(1)) + 1) if match else str(i)
            pages.append(TextPage(lines, label=label))
        return pages
    if ctx.kind == "hocr":
        ctx.kind = "html"
    return _html_flow(root)


def _is_abbyy_root(root) -> bool:
    if _local(root.tag).lower() != "document":
        return False
    if "abbyy" in _ns(root.tag).lower():
        return True
    return any(b.get("blockType") for b in root.iter("{*}block"))


def read_xml(data: bytes, ctx: "_Ctx") -> List[TextPage]:
    plain_ext = _ctx_ext(ctx) in _PLAIN_EXTS
    notes: List[str] = []
    try:
        root = parse_xml_bytes(data, ctx.options.fallback_encodings, notes)
    except IngestError as exc:
        if exc.code == "malformed" and plain_ext:
            return _xml_as_plain_text(data, ctx)
        raise
    name = _local(root.tag).lower()
    if name == "alto":
        ctx.kind = "alto"
        reader = read_alto
    elif name == "pcgts":
        ctx.kind = "page-xml"
        reader = read_page_xml
    elif name in _TEI_ROOTS:
        ctx.kind = "tei"
        reader = read_tei
    elif name == "html":
        text, _enc, _fl = decode_bytes(data, ctx.options.fallback_encodings)
        ctx.kind = "html"
        ctx.notes.extend(notes)
        return read_html(text, ctx)
    elif _is_abbyy_root(root):
        ctx.kind = "abbyy-xml"
        reader = read_abbyy
    elif name == "djvuxml":
        ctx.kind = "djvu-xml"
        reader = read_djvu
    else:
        if plain_ext and "xml_recovered" in notes:
            # A .txt that merely starts with "<" (a bracketed heading, a pasted tag):
            # recover mode would keep whatever parsed; the plain-text reader keeps it all.
            return _xml_as_plain_text(data, ctx)
        ctx.kind = "xml"
        reader = read_generic_xml
    ctx.notes.extend(notes)
    return reader(root, ctx)


def _ctx_ext(ctx: "_Ctx") -> str:
    return os.path.splitext(ctx.name or ctx.path)[1].lower()


def _xml_as_plain_text(data: bytes, ctx: "_Ctx") -> List[TextPage]:
    text, _enc, flags = decode_bytes(data, ctx.options.fallback_encodings)
    ctx.kind = "txt"
    ctx.notes.extend(flags + ["xml_parse_failed_read_as_text"])
    return read_plain(text, ctx)


# ── readers: PDF (pypdfium2, isolated) ────────────────────────────────────────

_PDF_OBJECT_CAP = 20000
#: A text object counts as rotated when its matrix turns more than about 1°
#: (tan 1° ≈ 0.0175): OCR layers of deskewed scans tilt by fractions of a degree.
_PDF_ROTATION_TOL = 0.0175


def _pdf_matrix(pdfium_c, obj, parent: Optional[Tuple[float, float, float, float]]):
    """The 2×2 part (a, b, c, d) of a page object's matrix, composed with the form
    XObject matrix it is nested in (PDF row-vector order: object × form)."""
    import ctypes

    m = pdfium_c.FS_MATRIX()
    if not pdfium_c.FPDFPageObj_GetMatrix(obj.raw, ctypes.byref(m)):
        return (1.0, 0.0, 0.0, 1.0)
    a, b, c, d = m.a, m.b, m.c, m.d
    if parent is not None:
        pa, pb, pc, pd = parent
        a, b, c, d = a * pa + b * pc, a * pb + b * pd, c * pa + d * pc, c * pb + d * pd
    return (a, b, c, d)


def classify_text_layer(raw: str, n_text_objs: int, n_invisible: int, opts: ReaderOptions) -> Tuple[str, Optional[str]]:
    """Per-page text-layer class, mirroring llm-enrich's pdf_to_md thresholds.

    none    — fewer than PDF_MIN_TEXT_CHARS visible characters: no text layer (OCR it)
    garbled — more than PDF_GARBLE_THRESHOLD of the characters are U+FFFD or
              control/format/private-use/unassigned code points (a subset font
              without /ToUnicode): the text layer does not decode
    ocr     — at least PDF_OCR_LAYER_MIN_RATIO of the text objects are invisible
              (render mode 3): the classic OCR layer under a page image
    digital — anything else: born-digital text
    """
    chars = [ch for ch in raw if not ch.isspace() and ch != "\x02"]
    if len(chars) < max(1, opts.pdf_min_text_chars):
        return "none", "no extractable text layer"
    bad = sum(1 for ch in chars if ch == "\ufffd" or unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cn"))
    if bad / len(chars) > opts.pdf_garble_threshold:
        return "garbled", "garbled text layer (subset font without /ToUnicode?)"
    if n_text_objs and n_invisible / n_text_objs >= opts.pdf_ocr_layer_min_ratio:
        return "ocr", None
    return "digital", None


def _pdfium():
    try:
        import pypdfium2
        import pypdfium2.raw as pdfium_c
    except ImportError as exc:
        raise IngestError("dependency_missing", "pypdfium2 is required for PDF input (setup/requirements.txt)") from exc
    return pypdfium2, pdfium_c


def read_pdf(path: str, ctx: "_Ctx") -> List[TextPage]:
    pdfium, pdfium_c = _pdfium()
    try:
        pdf = pdfium.PdfDocument(path)
    except pdfium.PdfiumError as exc:
        code = getattr(exc, "err_code", None)
        if code in (pdfium_c.FPDF_ERR_PASSWORD, pdfium_c.FPDF_ERR_SECURITY) or "password" in str(exc).lower():
            raise IngestError("encrypted", "password-protected PDF") from exc
        raise IngestError("corrupt", f"PDF could not be opened ({exc})") from exc
    try:
        n_pages = len(pdf)
        if n_pages > ctx.limits.max_pages:
            raise IngestError("too_large", f"{n_pages} pages > MAX_PAGES={ctx.limits.max_pages}")
        pages = []
        for i in range(n_pages):
            try:
                label = pdf.get_page_label(i) or str(i + 1)
            except Exception:
                label = str(i + 1)
            page = textpage = None
            try:
                page = pdf[i]
                textpage = page.get_textpage()
                raw = textpage.get_text_bounded()
                n_text = n_invisible = n_images = n_mirrored = n_rotated = 0
                forms: List[Tuple[float, float, float, float]] = []  # form matrices by nesting level
                for n, obj in enumerate(
                    page.get_objects(
                        filter=(pdfium_c.FPDF_PAGEOBJ_TEXT, pdfium_c.FPDF_PAGEOBJ_IMAGE, pdfium_c.FPDF_PAGEOBJ_FORM),
                        max_depth=4,
                    )
                ):
                    if n >= _PDF_OBJECT_CAP:
                        break
                    level = getattr(obj, "level", 0) or 0
                    if obj.type == pdfium_c.FPDF_PAGEOBJ_FORM:
                        del forms[level:]
                        forms.append(_pdf_matrix(pdfium_c, obj, forms[level - 1] if level and forms else None))
                    elif obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT:
                        n_text += 1
                        if pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw) == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE:
                            n_invisible += 1
                        a, b, c, d = _pdf_matrix(pdfium_c, obj, forms[level - 1] if 0 < level <= len(forms) else None)
                        if a < 0 or d < 0:
                            n_mirrored += 1
                        elif max(abs(b), abs(c)) > _PDF_ROTATION_TOL * max(abs(a), abs(b), abs(c), abs(d), 1e-9):
                            n_rotated += 1
                    else:
                        n_images += 1
            except pdfium.PdfiumError as exc:
                pages.append(
                    TextPage([], label=label, text_layer="none", needs_ocr_reason=f"page failed to load ({exc})",
                             flags=["page_load_failed"])
                )  # fmt: skip
                continue
            finally:
                if textpage is not None:
                    textpage.close()
                if page is not None:
                    page.close()
            layer, reason = classify_text_layer(raw, n_text, n_invisible, ctx.options)
            lines = normalize_newlines(raw).replace("\f", "\n").split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            flags = [
                f"{name}={value}"
                for name, value in (("mirrored_text", n_mirrored), ("rotated_text", n_rotated))
                if value
            ]
            pages.append(
                TextPage(lines, label=label, text_layer=layer, needs_ocr_reason=reason, images=n_images, flags=flags)
            )
        return pages
    finally:
        pdf.close()


# ── readers: OOXML (DOCX / XLSX / PPTX) ───────────────────────────────────────

_MC_FALLBACK = "Fallback"


def _truthy_val(el) -> bool:
    val = None
    for k, v in el.attrib.items():
        if _local(k) == "val":
            val = v
    return val is None or val.lower() not in ("0", "false", "off")


def _docx_handlers(flow: _Flow, mode: str, body, on_note: Optional[Callable[[str, str], None]]):
    """(on_start, on_end, skip) for walking WordprocessingML into `flow`: pages at
    explicit breaks (and at Word's rendered breaks in `auto` mode), tables row-major,
    text boxes after their anchor paragraph, `on_note(kind, id)` at each footnote or
    endnote reference. Shared by the document body and the note bodies."""
    deferred: List[Any] = []
    para_depth = [0]
    skip_names = {"del", "moveFrom", "instrText", "delText", "rPr", "sdtPr", "sdtEndPr", "fldData",
                  "commentRangeStart", "commentRangeEnd", "bookmarkStart", "bookmarkEnd"}  # fmt: skip

    def skip(el):
        name = _local(el.tag)
        if name == "txbxContent":
            deferred.append(el)
            return True
        if name == _MC_FALLBACK:
            return True
        if name == "pPr":
            return True
        if name == "sectPr" and body is not None and el.getparent() is body:
            return True
        return name in skip_names

    def page_break():
        if mode != "none":
            flow.explicit_break()
        else:
            flow.line_break()

    def on_start(el):
        # Only w:t carries document text, so it is read here; the walker's
        # on_text is a no-op because every other text node is pretty-printing.
        name = _local(el.tag)
        if name == "p":
            para_depth[0] += 1
            flow.line_break()
            ppr = next((c for c in el if _local(c.tag) == "pPr"), None)
            if ppr is not None and any(_local(c.tag) == "pageBreakBefore" and _truthy_val(c) for c in ppr):
                page_break()
        elif name == "t":
            if el.text:
                flow.text(el.text)
        elif name == "tab":
            flow.text("\t")
        elif name == "br":
            br_type = next((v for k, v in el.attrib.items() if _local(k) == "type"), "")
            if br_type == "page":
                page_break()
            else:
                flow.line_break()
        elif name == "cr":
            flow.line_break()
        elif name == "noBreakHyphen":
            flow.text("-")
        elif name == "lastRenderedPageBreak" and mode == "auto":
            flow.rendered_break()
        elif name in ("footnoteReference", "endnoteReference") and on_note is not None:
            on_note(name, next((v for k, v in el.attrib.items() if _local(k) == "id"), ""))

    def on_end(el):
        name = _local(el.tag)
        if name != "p":
            return
        flow.line_break()
        para_depth[0] -= 1
        ppr = next((c for c in el if _local(c.tag) == "pPr"), None)
        if ppr is not None:
            sect = next((c for c in ppr if _local(c.tag) == "sectPr"), None)
            if sect is not None:
                stype = next((c for c in sect if _local(c.tag) == "type"), None)
                kind = next((v for k, v in stype.attrib.items() if _local(k) == "val"), "") if stype is not None else ""
                if kind not in ("continuous", "nextColumn"):
                    page_break()
        if para_depth[0] == 0 and deferred:
            boxes = list(deferred)
            deferred.clear()
            for box in boxes:
                for child in box:  # the box itself would be deferred again by skip()
                    _walk(child, on_start, on_end, _ignore_text, skip)

    return on_start, on_end, skip


def _docx_notes(zf, rels, type_suffix: str, ctx: "_Ctx") -> Dict[str, Any]:
    """{w:id: note element} of the footnotes/endnotes part; separators (by `w:type`,
    not by the conventional ids -1/0) are not notes."""
    target = next((t for (typ, t) in rels.values() if typ.endswith(type_suffix)), None)
    if not target or target not in zf.namelist():
        return {}
    root = parse_xml_bytes(_zip_read(zf, target), ctx.options.fallback_encodings, ctx.notes)
    out: Dict[str, Any] = {}
    for el in root:
        if _local(el.tag) not in ("footnote", "endnote"):
            continue
        attrs = {_local(k): v for k, v in el.attrib.items()}
        if attrs.get("type", "normal") in ("", "normal"):
            out[attrs.get("id", "")] = el
    return out


def _docx_note_lines(el) -> List[str]:
    flow = _Flow()
    on_start, on_end, skip = _docx_handlers(flow, "none", None, None)
    for child in el:
        _walk(child, on_start, on_end, _ignore_text, skip)
    return [ln for p in flow.finish() for ln in p.lines]


def _count_parts_with_text(zf, rels, suffixes: Tuple[str, ...]) -> int:
    n = 0
    for typ, target in rels.values():
        if typ.endswith(suffixes) and target in zf.namelist():
            try:
                root = parse_xml_bytes(_zip_read(zf, target))
            except IngestError:
                continue
            n += any((t.text or "").strip() for t in root.iter("{*}t"))
    return n


def read_docx(path: str, ctx: "_Ctx") -> List[TextPage]:
    """DOCX via zipfile + lxml: body order, tables row-major, text boxes after their
    anchor paragraph, pages at explicit breaks (and at Word's rendered page breaks
    in `auto` mode). Footnotes and endnotes follow [TEXT_INGEST].NOTES: `page` puts a
    footnote at the end of the page that references it and endnotes at the end,
    `end` puts every note at the end, `skip` leaves them out (counted). Headers and
    footers are not read (counted)."""
    mode = ctx.options.page_breaks
    notes_mode = ctx.options.notes_placement
    with open_zip(path, ctx.limits) as zf:
        main = _ooxml_main_part(zf) or "word/document.xml"
        root = parse_xml_bytes(_zip_read(zf, main), ctx.options.fallback_encodings, ctx.notes)
        rels = _rels_map(zf, main)
        parts = {
            "footnoteReference": _docx_notes(zf, rels, "/footnotes", ctx),
            "endnoteReference": _docx_notes(zf, rels, "/endnotes", ctx),
        }
        headers = _count_parts_with_text(zf, rels, ("/header", "/footer"))
    body = next((el for el in root if _local(el.tag) == "body"), None)
    if body is None:
        raise IngestError("malformed", "DOCX main part has no <w:body>")

    flow = _Flow()
    end_lines: List[str] = []
    seen: set = set()
    counts = {"footnoteReference": 0, "endnoteReference": 0, "skipped": 0}

    def on_note(kind: str, note_id: str) -> None:
        if (kind, note_id) in seen or note_id not in parts[kind]:
            return
        seen.add((kind, note_id))
        if notes_mode == "skip":
            counts["skipped"] += 1
            return
        counts[kind] += 1
        lines = _docx_note_lines(parts[kind][note_id])
        if notes_mode == "page" and kind == "footnoteReference":
            flow.defer_to_page_end(lines)
        else:
            end_lines.extend(lines)

    on_start, on_end, skip = _docx_handlers(flow, mode, body, on_note)
    _walk(body, on_start, on_end, _ignore_text, skip)
    pages = flow.finish()
    if end_lines:
        pages[-1].lines.extend(end_lines)
    for note, value in (
        ("docx_footnotes", counts["footnoteReference"]),
        ("docx_endnotes", counts["endnoteReference"]),
        ("notes_not_read", counts["skipped"]),
        ("headers_footers_not_read", headers),
    ):
        if value:
            ctx.notes.append(f"{note}={value}")
    return pages


def _ignore_text(_s: str) -> None:
    return None


def _xlsx_shared_strings(zf, part_rels) -> List[str]:
    target = next((t for (typ, t) in part_rels.values() if typ.endswith("/sharedStrings")), None)
    if not target or target not in zf.namelist():
        return []
    root = parse_xml_bytes(_zip_read(zf, target))
    out = []
    for si in root:
        if _local(si.tag) != "si":
            continue
        parts = []
        for child in si:
            name = _local(child.tag)
            if name == "t":
                parts.append(child.text or "")
            elif name == "r":
                parts.extend(t.text or "" for t in child if _local(t.tag) == "t")
        out.append("".join(parts))
    return out


def _cell_inline_text(cell) -> str:
    parts = []
    for isel in cell:
        if _local(isel.tag) != "is":
            continue
        for child in isel:
            name = _local(child.tag)
            if name == "t":
                parts.append(child.text or "")
            elif name == "r":
                parts.extend(t.text or "" for t in child if _local(t.tag) == "t")
    return "".join(parts)


def read_xlsx(path: str, ctx: "_Ctx") -> List[TextPage]:
    """XLSX via zipfile + lxml iterparse: sheet = page (workbook order, label =
    sheet name, hidden sheets included and counted), row = line of its text cells
    joined by tab. Numbers, dates, booleans and errors are not text and are dropped.
    A sheet longer than MAX_LINES_PER_PAGE continues on `<name>+1`… pages."""
    etree = _etree()
    with open_zip(path, ctx.limits) as zf:
        workbook = _ooxml_main_part(zf) or "xl/workbook.xml"
        wb_root = parse_xml_bytes(_zip_read(zf, workbook))
        rels = _rels_map(zf, workbook)
        shared = _xlsx_shared_strings(zf, rels)
        sheets_el = next((el for el in wb_root if _local(el.tag) == "sheets"), None)
        pages = []
        hidden = 0
        for sheet in list(sheets_el) if sheets_el is not None else []:
            if (sheet.get("state") or "").lower() in ("hidden", "veryhidden"):
                hidden += 1  # read all the same: hidden is a view setting, the text is the document's
            rid = next((v for k, v in sheet.attrib.items() if _local(k) == "id" and _ns(k)), "")
            typ, target = rels.get(rid, ("", ""))
            if not typ.endswith("/worksheet") or target not in zf.namelist():
                continue
            data = _zip_read(zf, target)
            if _ENTITY_DECL.search(data):
                raise IngestError("xml_entity_declaration")
            lines: List[str] = []
            try:
                for _event, row in etree.iterparse(
                    io.BytesIO(data), events=("end",), tag="{*}row", resolve_entities=False, no_network=True,
                    load_dtd=False, huge_tree=False,
                ):  # fmt: skip
                    cells = []
                    for cell in row:
                        if _local(cell.tag) != "c":
                            continue
                        ctype = cell.get("t", "")
                        value_el = next((v for v in cell if _local(v.tag) == "v"), None)
                        value = value_el.text if value_el is not None else None
                        text = ""
                        if ctype == "s" and value is not None:
                            try:
                                text = shared[int(value)]
                            except (ValueError, IndexError):
                                ctx.notes.append("xlsx_bad_shared_string")
                        elif ctype == "inlineStr":
                            text = _cell_inline_text(cell)
                        elif ctype == "str" and value is not None:
                            text = value
                        text = _collapse_ws(text)
                        if text:
                            cells.append(text)
                    if cells:
                        line = "\t".join(cells)
                        _charge(ctx, len(line))
                        lines.append(line)
                    row.clear()
            except etree.XMLSyntaxError as exc:
                raise IngestError("malformed", f"worksheet {target!r} could not be parsed ({exc})") from exc
            if lines:
                pages.append(TextPage(lines, label=sheet.get("name") or str(len(pages) + 1)))
    if hidden:
        ctx.notes.append(f"xlsx_hidden_sheets={hidden}")
    return pages or [TextPage([])]


def _charge(ctx: "_Ctx", chars: int) -> None:
    """Bound the text a spreadsheet expands to (a sheet is continued on `+1` pages
    past MAX_LINES_PER_PAGE, so the page cap alone no longer bounds its memory)."""
    ctx.expanded_chars += chars
    if ctx.expanded_chars > ctx.limits.zip_max_total_mb * _MB:
        raise IngestError("too_large", "expanded sheet text > ZIP_MAX_TOTAL_MB")


def _drawingml_page(root) -> List[str]:
    """Every DrawingML paragraph (a:p) of a slide in document order; a:br breaks."""
    flow = _Flow()

    def is_dml(el) -> bool:
        return "drawingml" in _ns(el.tag)

    def on_start(el):
        name = _local(el.tag)
        if name in ("p", "br") and is_dml(el):
            flow.line_break()
        elif name == "t" and is_dml(el) and el.text:
            flow.text(el.text)

    def on_end(el):
        if _local(el.tag) == "p" and is_dml(el):
            flow.line_break()

    _walk(root, on_start, on_end, _ignore_text, lambda el: _local(el.tag) == _MC_FALLBACK)
    return [ln for p in flow.finish() for ln in p.lines]


def read_pptx(path: str, ctx: "_Ctx") -> List[TextPage]:
    """PPTX: slide = page in presentation (sldIdLst) order, hidden slides included
    and counted; paragraph = line. Speaker notes are not read."""
    with open_zip(path, ctx.limits) as zf:
        presentation = _ooxml_main_part(zf) or "ppt/presentation.xml"
        pres_root = parse_xml_bytes(_zip_read(zf, presentation))
        rels = _rels_map(zf, presentation)
        id_list = next((el for el in pres_root if _local(el.tag) == "sldIdLst"), None)
        pages = []
        hidden = 0
        for n, sld in enumerate(list(id_list) if id_list is not None else [], 1):
            rid = next((v for k, v in sld.attrib.items() if _local(k) == "id" and _ns(k)), "")
            typ, target = rels.get(rid, ("", ""))
            if not typ.endswith("/slide") or target not in zf.namelist():
                continue
            slide_root = parse_xml_bytes(_zip_read(zf, target), ctx.options.fallback_encodings, ctx.notes)
            if (slide_root.get("show") or "").lower() in ("0", "false"):
                hidden += 1
            pages.append(TextPage(_drawingml_page(slide_root), label=str(n)))
    if hidden:
        ctx.notes.append(f"pptx_hidden_slides={hidden}")
    return pages or [TextPage([])]


# ── readers: ODF (ODT / ODS / ODP) ────────────────────────────────────────────


def _odf_break_styles(roots) -> Tuple[set, set]:
    before, after = set(), set()
    for root in roots:
        for style in root.iter("{*}style"):
            name = next((v for k, v in style.attrib.items() if _local(k) == "name"), None)
            if not name:
                continue
            for props in style:
                if _local(props.tag) != "paragraph-properties":
                    continue
                for k, v in props.attrib.items():
                    if _local(k) == "break-before" and v == "page":
                        before.add(name)
                    if _local(k) == "break-after" and v == "page":
                        after.add(name)
    return before, after


def _odf_check_encrypted(zf) -> None:
    if "META-INF/manifest.xml" in zf.namelist():
        if b"encryption-data" in _zip_read(zf, "META-INF/manifest.xml"):
            raise IngestError("encrypted", "password-protected OpenDocument file")


def _odf_attr(el, local_name: str, default: str = "") -> str:
    return next((v for k, v in el.attrib.items() if _local(k) == local_name), default)


def _odf_text_flow(root_el, ctx, before=frozenset(), after=frozenset(), notes_mode=None) -> List[TextPage]:
    """ODF text into pages. `notes_mode` (page | end | skip; None = not read, as for
    slides) places `text:note` bodies like read_docx places Word's notes."""
    mode = ctx.options.page_breaks
    flow = _Flow()
    deferred: List[Any] = []
    para_depth = [0]
    end_lines: List[str] = []
    note_counts = {"footnote": 0, "endnote": 0, "skipped": 0}
    skip_names = {"note", "annotation", "tracked-changes", "notes", "sequence-decls", "forms", "deletion"}

    def read_note(el) -> None:
        if notes_mode == "skip":
            note_counts["skipped"] += 1
            return
        body = next((c for c in el if _local(c.tag) == "note-body"), None)
        if body is None:
            return
        note_ctx = _Ctx(ctx.path, ctx.limits, replace(ctx.options, page_breaks="none"))
        lines = [ln for p in _odf_text_flow(body, note_ctx) for ln in p.lines]
        note_class = "endnote" if _odf_attr(el, "note-class", "footnote") == "endnote" else "footnote"
        note_counts[note_class] += 1
        if notes_mode == "page" and note_class == "footnote":
            flow.defer_to_page_end(lines)
        else:
            end_lines.extend(lines)

    def skip(el):
        name = _local(el.tag)
        if name == "text-box":
            deferred.append(el)
            return True
        if name == "note" and notes_mode is not None:
            read_note(el)
            return True
        return name in skip_names

    def on_start(el):
        name = _local(el.tag)
        if name in ("p", "h"):
            para_depth[0] += 1
            flow.line_break()
            if mode != "none" and _odf_attr(el, "style-name") in before:
                flow.explicit_break()
        elif name == "line-break":
            flow.line_break()
        elif name == "tab":
            flow.text("\t")
        elif name == "s":
            try:
                flow.text(" " * max(1, min(int(_odf_attr(el, "c", "1")), 100)))
            except ValueError:
                flow.text(" ")
        elif name == "soft-page-break" and mode == "auto":
            flow.rendered_break()

    def on_end(el):
        name = _local(el.tag)
        if name not in ("p", "h"):
            return
        flow.line_break()
        para_depth[0] -= 1
        if mode != "none" and _odf_attr(el, "style-name") in after:
            flow.explicit_break()
        if para_depth[0] == 0 and deferred:
            boxes = list(deferred)
            deferred.clear()
            for box in boxes:
                for child in box:  # the box itself would be deferred again by skip()
                    _walk(child, on_start, on_end, flow.text, skip)

    _walk(root_el, on_start, on_end, flow.text, skip)
    pages = flow.finish()
    if end_lines:
        pages[-1].lines.extend(end_lines)
    for note, value in (
        ("odt_footnotes", note_counts["footnote"]),
        ("odt_endnotes", note_counts["endnote"]),
        ("notes_not_read", note_counts["skipped"]),
    ):
        if value:
            ctx.notes.append(f"{note}={value}")
    return pages


def _odf_headers_with_text(styles) -> int:
    """Master-page headers/footers (any variant) that hold text — counted, not read."""
    if styles is None:
        return 0
    n = 0
    for master in styles.iter("{*}master-page"):
        for part in master:
            if _local(part.tag).startswith(("header", "footer")) and "".join(part.itertext()).strip():
                n += 1
    return n


_ODF_REPEAT_CAP = 100
#: Cell content that is not the cell's text: comments and notes.
_ODF_CELL_SKIP = {"annotation", "note"}


def _odf_cell_text(cell) -> str:
    """A spreadsheet cell's paragraphs, comments (`office:annotation`) left out."""
    paras: List[str] = []
    buf: List[str] = []

    def flush():
        text = _collapse_ws("".join(buf))
        buf.clear()
        if text:
            paras.append(text)

    def on_start(el):
        name = _local(el.tag)
        if name in ("p", "h"):
            flush()
        elif name in ("s", "tab", "line-break"):
            buf.append(" ")

    def on_end(el):
        if _local(el.tag) in ("p", "h"):
            flush()

    _walk(cell, on_start, on_end, buf.append, lambda el: _local(el.tag) in _ODF_CELL_SKIP)
    flush()
    return " ".join(paras)


def _ods_sheet_lines(table, ctx) -> List[str]:
    """Rows of an ODS table: its string cells joined by tab. Cells typed as numbers,
    dates, times, currency, percentages or booleans are not text (parity with
    XLSX). Repeats are capped at _ODF_REPEAT_CAP (noted: the rest is not read)."""
    lines: List[str] = []
    capped = False
    for row in table.iter("{*}table-row"):
        cells = []
        for cell in row:
            if _local(cell.tag) not in ("table-cell", "covered-table-cell"):
                continue
            value_type = _odf_attr(cell, "value-type")
            if value_type and value_type != "string":
                continue
            text = _odf_cell_text(cell)
            if not text:
                continue
            try:
                repeat = int(_odf_attr(cell, "number-columns-repeated", "1"))
            except ValueError:
                repeat = 1
            capped |= repeat > _ODF_REPEAT_CAP
            cells.extend([text] * max(1, min(repeat, _ODF_REPEAT_CAP)))
        if not cells:
            continue
        try:
            repeat = int(_odf_attr(row, "number-rows-repeated", "1"))
        except ValueError:
            repeat = 1
        capped |= repeat > _ODF_REPEAT_CAP
        line = "\t".join(cells)
        for _ in range(max(1, min(repeat, _ODF_REPEAT_CAP))):
            _charge(ctx, len(line))
            lines.append(line)
    if capped and "sheet_repeat_capped" not in ctx.notes:
        ctx.notes.append("sheet_repeat_capped")
    return lines


def read_odf(path: str, ctx: "_Ctx") -> List[TextPage]:
    """ODT/ODS/ODP from content.xml (+ styles.xml for page-break styles)."""
    with open_zip(path, ctx.limits) as zf:
        _odf_check_encrypted(zf)
        content = parse_xml_bytes(_zip_read(zf, "content.xml"), ctx.options.fallback_encodings, ctx.notes)
        styles = None
        if "styles.xml" in zf.namelist():
            try:
                styles = parse_xml_bytes(_zip_read(zf, "styles.xml"))
            except IngestError:
                styles = None
    body = next((el for el in content if _local(el.tag) == "body"), None)
    if body is None:
        raise IngestError("malformed", "OpenDocument content.xml has no <office:body>")
    inner = next((el for el in body if isinstance(el.tag, str)), None)
    if inner is None:
        return [TextPage([])]
    kind = _local(inner.tag)
    if kind == "text":
        before, after = _odf_break_styles([content] + ([styles] if styles is not None else []))
        pages = _odf_text_flow(inner, ctx, before, after, notes_mode=ctx.options.notes_placement)
        headers = _odf_headers_with_text(styles)
        if headers:
            ctx.notes.append(f"headers_footers_not_read={headers}")
        return pages
    if kind == "spreadsheet":
        pages = []
        for n, table in enumerate(el for el in inner if _local(el.tag) == "table"):
            lines = _ods_sheet_lines(table, ctx)
            if lines:
                pages.append(TextPage(lines, label=_odf_attr(table, "name") or str(n + 1)))
        return pages or [TextPage([])]
    if kind == "presentation":
        pages = []
        for n, slide in enumerate((el for el in inner if _local(el.tag) == "page"), 1):
            slide_pages = _odf_text_flow(slide, ctx)
            lines = [ln for p in slide_pages for ln in p.lines]
            pages.append(TextPage(lines, label=_odf_attr(slide, "name") or str(n)))
        return pages or [TextPage([])]
    raise IngestError("archive_unsupported", f"OpenDocument body {kind!r} is not text/spreadsheet/presentation")


# ── readers: EPUB ─────────────────────────────────────────────────────────────


def read_epub(path: str, ctx: "_Ctx") -> List[TextPage]:
    """EPUB 2/3: spine chapter = page, lines through the HTML reader."""
    with open_zip(path, ctx.limits) as zf:
        names = set(zf.namelist())
        container = parse_xml_bytes(_zip_read(zf, "META-INF/container.xml"))
        rootfile = next((el for el in container.iter("{*}rootfile")), None)
        if rootfile is None:
            raise IngestError("malformed", "EPUB has no rootfile")
        opf_path = rootfile.get("full-path", "")
        opf = parse_xml_bytes(_zip_read(zf, opf_path))
        base = posixpath.dirname(opf_path)
        manifest = {
            item.get("id"): (_resolve_part(base, item.get("href", ""), names), item.get("media-type", ""))
            for item in opf.iter("{*}item")
        }
        encrypted = set()
        if "META-INF/encryption.xml" in names:
            enc_root = parse_xml_bytes(_zip_read(zf, "META-INF/encryption.xml"))
            for ref in enc_root.iter("{*}CipherReference"):
                encrypted.add(_resolve_part("", ref.get("URI", ""), names))
        pages = []
        missing = non_html = 0
        for itemref in opf.iter("{*}itemref"):
            href, media = manifest.get(itemref.get("idref"), ("", ""))
            if not href or href not in names:
                missing += 1
                continue
            if "html" not in media:
                non_html += 1
                continue
            if href in encrypted:
                raise IngestError("encrypted", "DRM-encrypted EPUB content")
            text, _enc, flags = decode_bytes(_zip_read(zf, href), ctx.options.fallback_encodings)
            ctx.notes.extend(f for f in flags if f not in ctx.notes)
            chapter = read_html(text, _Ctx(ctx.path, ctx.limits, ctx.options, kind="html"))
            lines = [ln for p in chapter for ln in p.lines]
            if lines:
                pages.append(TextPage(lines, label=posixpath.splitext(posixpath.basename(href))[0]))
    if missing:
        ctx.notes.append(f"epub_spine_skipped={missing}")
    if non_html:
        ctx.notes.append(f"epub_spine_non_html={non_html}")
    return pages or [TextPage([])]


# ── readers: RTF (stdlib stripper) ────────────────────────────────────────────

_RTF_TOKEN = re.compile(rb"\\([a-zA-Z]+)(-?\d+)? ?|\\'([0-9a-fA-F]{2})|\\(.)|([{}])|([^\\{}\r\n]+)|[\r\n]+", re.DOTALL)
_RTF_DESTINATIONS = {
    b"fonttbl", b"colortbl", b"stylesheet", b"info", b"pict", b"object", b"header", b"headerl", b"headerr",
    b"headerf", b"footer", b"footerl", b"footerr", b"footerf", b"footnote", b"fldinst", b"xmlnstbl",
    b"listtable", b"listoverridetable", b"revtbl", b"rsidtbl", b"generator", b"themedata",
    b"colorschememapping", b"latentstyles", b"datastore", b"mmathPr", b"filetbl", b"bkmkstart", b"bkmkend",
    b"field_instructions", b"annotation", b"atnid", b"atnauthor", b"comment", b"pgdsctbl", b"objdata",
    b"blipuid", b"shppict", b"nonshppict", b"sp", b"shpinst", b"template", b"userprops", b"docvar",
}  # fmt: skip
_RTF_SYMBOLS = {b"emdash": "—", b"endash": "–", b"bullet": "•", b"lquote": "‘",
                b"rquote": "’", b"ldblquote": "“", b"rdblquote": "”", b"tab": "\t"}  # fmt: skip
#: \fcharsetN → code page. 0 (ANSI), 1 (default) and 2 (symbol) mean "the document's
#: code page" (\ansicpgN), which keeps a CE document with \fcharset0 fonts decoding
#: as its \ansicpg says.
_RTF_CHARSETS = {
    77: "mac_roman", 128: "cp932", 129: "cp949", 130: "cp1361", 134: "cp936", 136: "cp950", 161: "cp1253",
    162: "cp1254", 163: "cp1258", 177: "cp1255", 178: "cp1256", 186: "cp1257", 204: "cp1251", 222: "cp874",
    238: "cp1250", 255: "cp437",
}  # fmt: skip


def _rtf_codepage(value: str) -> Optional[str]:
    try:
        "".encode(value)
    except LookupError:
        return None
    return value


def read_rtf(data: bytes, ctx: "_Ctx") -> List[TextPage]:
    """RTF via a small tokenizer: \\page → page, \\par/\\line/\\row → line, \\cell → tab;
    header/footer/footnote/picture/field-instruction destinations skipped; \\'xx
    decoded with the current font's code page (its \\fcharsetN or \\cpgN from the font
    table, else the document's \\ansicpgN); \\uN with its fallback skipped, surrogate
    pairs combined."""
    if data.startswith(_UTF8_BOM):
        data = data[len(_UTF8_BOM) :]
    flow = _Flow(collapse=False)
    stack: List[Tuple[bool, int, str]] = []
    ignorable = False
    uc = 1
    skip_chars = 0
    doc_cp = "cp1252"
    codepage = doc_cp
    fonts: Dict[int, str] = {}
    default_font: Optional[int] = None
    fonttbl_depth: Optional[int] = None
    table_font: Optional[int] = None
    pending = bytearray()
    high_surrogate: List[int] = []
    group_start = False

    def emit(text: str) -> None:
        if high_surrogate:  # a high surrogate not followed by its low half: dropped in _finalize
            flow.text(chr(high_surrogate.pop()))
        flow.text(text)

    def flush_bytes():
        if pending:
            emit(pending.decode(codepage, errors="replace"))
            pending.clear()

    pos = 0
    if len(data) > ctx.limits.max_file_mb * _MB:
        raise IngestError("too_large")
    for match in _RTF_TOKEN.finditer(data):
        if match.start() < pos:
            continue
        word, arg, hexbyte, symbol, brace, text = match.groups()
        if brace is not None:
            flush_bytes()
            if brace == b"{":
                stack.append((ignorable, uc, codepage))
                group_start = True
            else:
                ignorable, uc, codepage = stack.pop() if stack else (False, 1, doc_cp)
                if fonttbl_depth is not None and len(stack) < fonttbl_depth:
                    fonttbl_depth = None
                group_start = False
            continue
        starts_group, group_start = group_start, False
        if skip_chars and (hexbyte is not None or text is not None):
            if text is not None:
                if len(text) <= skip_chars:
                    skip_chars -= len(text)
                    continue
                text = text[skip_chars:]
                skip_chars = 0
            else:
                skip_chars -= 1
                continue
        if symbol is not None:
            if symbol == b"*":
                ignorable = True
            elif not ignorable:
                flush_bytes()
                if symbol in (b"\\", b"{", b"}"):
                    emit(symbol.decode())
                elif symbol == b"~":
                    emit(" ")
                elif symbol == b"_":
                    emit("-")
                elif symbol in (b"\n", b"\r"):
                    flow.line_break()
            continue
        if word is not None:
            if fonttbl_depth is not None:  # inside the font table: record each font's code page
                if word == b"f" and arg:
                    table_font = int(arg)
                elif word == b"fcharset" and arg and table_font is not None:
                    cp = _RTF_CHARSETS.get(int(arg))
                    if cp:
                        fonts[table_font] = cp
                elif word == b"cpg" and arg and table_font is not None:
                    cp = _rtf_codepage(f"cp{int(arg)}")
                    if cp:
                        fonts[table_font] = cp
                continue
            if word == b"fonttbl" and starts_group:
                fonttbl_depth = len(stack)
                ignorable = True
                continue
            if word in _RTF_DESTINATIONS and starts_group:
                ignorable = True
                continue
            if word == b"ansicpg" and arg:
                cp = _rtf_codepage(f"cp{int(arg)}")
                if cp:
                    doc_cp = codepage = cp
                continue
            if word == b"deff" and arg:
                default_font = int(arg)
                continue
            if word == b"uc" and arg:
                uc = max(0, int(arg))
                continue
            if word == b"bin" and arg:
                pos = match.end() + max(0, int(arg))  # skip the raw binary payload
                continue
            if word == b"f" and arg:
                flush_bytes()
                codepage = fonts.get(int(arg), doc_cp)
                continue
            if word == b"plain":
                flush_bytes()
                codepage = fonts.get(default_font, doc_cp) if default_font is not None else doc_cp
                continue
            if ignorable:
                continue
            if word == b"u" and arg:
                flush_bytes()
                code = int(arg)
                code = code + 65536 if code < 0 else code
                skip_chars = uc
                if 0xD800 <= code <= 0xDBFF:
                    if high_surrogate:
                        flow.text(chr(high_surrogate.pop()))
                    high_surrogate.append(code)
                elif 0xDC00 <= code <= 0xDFFF and high_surrogate:
                    high = high_surrogate.pop()
                    flow.text(chr(0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00)))
                else:
                    emit(chr(code))
            elif word in (b"par", b"line", b"row", b"sect"):
                flush_bytes()
                flow.line_break()
            elif word == b"page":
                flush_bytes()
                flow.explicit_break()
            elif word == b"cell":
                flush_bytes()
                emit("\t")
            elif word in _RTF_SYMBOLS:
                flush_bytes()
                emit(_RTF_SYMBOLS[word])
            continue
        if ignorable:
            continue
        if hexbyte is not None:
            pending.append(int(hexbyte, 16))
            continue
        if text is not None:
            flush_bytes()
            emit(text.decode(codepage, errors="replace"))
    flush_bytes()
    if high_surrogate:
        flow.text(chr(high_surrogate.pop()))
    pages = flow.finish()
    for page in pages:
        page.lines = [_collapse_ws(ln) for ln in page.lines]
    return pages


# ── readers: subtitles and e-mail ─────────────────────────────────────────────

_SUBTITLE_MARKUP = re.compile(r"</?[A-Za-z][^>]*>|<\d[^>]*>|\{\\[^}]*\}")
_VTT_META_BLOCKS = ("NOTE", "STYLE", "REGION")


def read_subtitles(text: str, ctx: "_Ctx") -> List[TextPage]:
    """SRT/WebVTT: one page; a line per cue text line. Cue numbers and identifiers,
    timings, the WEBVTT header, NOTE/STYLE/REGION blocks and inline markup (<i>,
    <c.x>, <v Speaker>, {\\an8}, inline timestamps) are dropped."""
    text = normalize_newlines(text).lstrip("﻿")
    lines: List[str] = []
    untimed = 0
    for n, block in enumerate(re.split(r"\n[ \t]*\n", text)):
        rows = block.strip("\n").split("\n")
        if not any(r.strip() for r in rows):
            continue
        first = rows[0].strip()
        if ctx.kind == "vtt" and (
            (n == 0 and first.startswith("WEBVTT")) or first.split(" ", 1)[0].split("\t", 1)[0] in _VTT_META_BLOCKS
        ):
            continue
        timing = next((i for i, r in enumerate(rows) if _CUE_TIMING.match(r)), None)
        if timing is None:
            if len(rows) == 1 and first.isdigit():
                continue
            untimed += 1
            cue = rows
        else:
            cue = rows[timing + 1 :]
        for row in cue:
            row = _SUBTITLE_MARKUP.sub("", row)
            if ctx.kind == "vtt":
                row = html.unescape(row)
            if row.strip():
                lines.append(row.strip())
    if untimed:
        ctx.notes.append(f"subtitle_blocks_without_timing={untimed}")
    return [TextPage(lines, label="1")]


_MBOX_SEPARATOR = re.compile(rb"(?m)^From [^\r\n]*\r?\n")
_MBOX_QUOTED_FROM = re.compile(rb"(?m)^>(>*From )")


def _email_part_text(part, ctx: "_Ctx") -> str:
    payload = part.get_payload(decode=True) or b""
    charset = part.get_content_charset() or ""
    if charset:
        try:
            return payload.decode(charset)
        except (LookupError, UnicodeDecodeError):
            ctx.notes.append("email_charset_fallback")
    text, _enc, flags = decode_bytes(payload, ctx.options.fallback_encodings)
    ctx.notes.extend(f for f in flags if f not in ctx.notes)
    return text


def read_email(data: bytes, ctx: "_Ctx") -> List[TextPage]:
    """EML (one message) / MBOX (one page per message): the decoded Subject, then the
    text/plain body (text/html through the HTML reader when there is no plain part).
    Other headers are metadata; attachments are counted, not read."""
    import email
    from email import policy

    if ctx.kind == "mbox":
        messages = [_MBOX_QUOTED_FROM.sub(rb"\1", m) for m in _MBOX_SEPARATOR.split(data) if m.strip()]
    else:
        messages = [data]
    pages: List[TextPage] = []
    bad = attachments = 0
    for n, raw in enumerate(messages, 1):
        lines: List[str] = []
        try:
            msg = email.message_from_bytes(raw, policy=policy.default)
            subject = _collapse_ws(str(msg.get("subject") or ""))
            if subject:
                lines.append(subject)
            body = msg.get_body(preferencelist=("plain", "html"))
            if body is not None:
                content = _email_part_text(body, ctx)
                if body.get_content_subtype() == "html":
                    if content.strip():
                        html_pages = read_html(content, _Ctx(ctx.path, ctx.limits, ctx.options, kind="html"))
                        lines.extend(ln for p in html_pages for ln in p.lines)
                else:
                    lines.extend(normalize_newlines(content).split("\n"))
            attachments += sum(1 for part in msg.walk() if part.is_attachment())
        except IngestError:
            raise
        except Exception as exc:  # a damaged message costs that message
            logger.debug("e-mail message %d unreadable: %s", n, exc)
            bad += 1
            continue
        if any(ln.strip() for ln in lines):
            pages.append(TextPage(lines, label=str(n)))
    if bad:
        ctx.notes.append(f"email_bad_messages={bad}")
    if attachments:
        ctx.notes.append(f"email_attachments_skipped={attachments}")
    return pages or [TextPage([])]


# ── readers: ZIP bundle of page files ─────────────────────────────────────────

#: When one page is exported in several formats (Transkribus: page/0001.xml and
#: alto/0001.xml), the first kind in this order is read, the others counted.
_BUNDLE_KIND_PRECEDENCE = (
    "page-xml", "alto", "abbyy-xml", "hocr", "djvu-xml", "tesseract-tsv", "tei", "json", "jsonl", "html", "xml",
    "md", "csv", "tsv", "srt", "vtt", "eml", "rtf", "txt",
)  # fmt: skip
#: XML roots of export metadata rather than pages (METS, Transkribus doc metadata, OPC/ODF manifests).
_BUNDLE_META_ROOTS = frozenset(
    {"mets", "trpdocmetadata", "relationships", "types", "manifest", "container", "package", "rdf"}
)
_BUNDLE_REFUSAL_CODES = ("pdf_not_read_in_bundle", "nested_container", "archive_unsupported")


def _natural_key(path: str):
    return ([int(t) if t.isdigit() else t.casefold() for t in re.split(r"(\d+)", path)], path)


def read_zip_bundle(path: str, ctx: "_Ctx") -> List[TextPage]:
    """A ZIP of per-page files as ONE document: members in natural path order
    (1, 2, …, 10), each read by content with the ordinary readers, their pages
    concatenated. Metadata, images and unknown types are ignored; PDFs and nested
    containers are not read here (a PDF is only read in its isolated process); a
    member that fails is skipped and noted (the document is then `partial`)."""
    with open_zip(path, ctx.limits) as zf:
        infos = sorted(zf.infolist(), key=lambda i: _natural_key(i.filename))
        skipped: List[Tuple[str, str]] = []
        ignored = 0
        candidates: List[Tuple[zipfile.ZipInfo, str]] = []
        for info in infos:
            cls = bundle_member_class(info)
            if cls == "ignored":
                ignored += not info.is_dir()
                continue
            if cls == "refused":
                ext = posixpath.splitext(info.filename)[1].lower()
                skipped.append((info.filename, _BUNDLE_REFUSED_EXTS.get(ext, "nested_container")))
                continue
            if info.file_size > ctx.limits.max_file_mb * _MB:
                skipped.append((info.filename, "too_large"))
                continue
            try:
                with zf.open(info) as fh:
                    head = fh.read(65536)
                kind = sniff_bytes(
                    head,
                    posixpath.basename(info.filename),
                    ctx.limits,
                    [],
                    full=lambda info=info: _zip_read(zf, info.filename),
                    container="a ZIP bundle",
                )
            except IngestError as exc:
                skipped.append((info.filename, exc.code))
                continue
            except (zipfile.BadZipFile, OSError, EOFError, RuntimeError) as exc:
                skipped.append((info.filename, "encrypted" if isinstance(exc, RuntimeError) else "corrupt"))
                continue
            if _xml_root_name(head.lstrip())[0].lower() in _BUNDLE_META_ROOTS:
                ignored += 1
                continue
            candidates.append((info, kind))

        by_stem: Dict[str, List[Tuple[zipfile.ZipInfo, str]]] = {}
        for info, kind in candidates:
            by_stem.setdefault(posixpath.splitext(posixpath.basename(info.filename))[0].casefold(), []).append(
                (info, kind)
            )
        keep: set = set()
        duplicates = 0
        for group in by_stem.values():
            kinds = {k for _i, k in group}
            best = min(kinds, key=lambda k: _BUNDLE_KIND_PRECEDENCE.index(k) if k in _BUNDLE_KIND_PRECEDENCE else 99)
            for info, kind in group:
                if len(kinds) == 1 or kind == best:
                    keep.add(info.filename)
                else:
                    duplicates += 1
        members = [(i, k) for i, k in candidates if i.filename in keep]
        try:
            prefix = posixpath.commonpath([posixpath.dirname(i.filename) for i, _k in members]) if members else ""
        except ValueError:
            prefix = ""

        pages: List[TextPage] = []
        kinds_read: List[str] = []
        encodings: set = set()
        for info, kind in members:
            mctx = _Ctx(
                f"{path}!{info.filename}", ctx.limits, ctx.options, kind=kind, name=posixpath.basename(info.filename)
            )
            try:
                member_pages, encoding = _run_reader(_read_bytes_kind, kind, _zip_read(zf, info.filename), mctx)
            except IngestError as exc:
                skipped.append((info.filename, exc.code))
                continue
            kinds_read.append(mctx.kind or kind)
            if encoding:
                encodings.add(encoding)
            ctx.notes.extend(n for n in mctx.notes if not n.startswith("extension ") and n not in ctx.notes)
            rel = info.filename[len(prefix) :].lstrip("/") if prefix else info.filename
            stem = posixpath.splitext(rel)[0] or rel
            for j, page in enumerate(member_pages, 1):
                page.label = stem if len(member_pages) == 1 else f"{stem}/{page.label or j}"
                pages.append(page)
            if len(pages) > ctx.limits.max_pages:
                raise IngestError("too_large", f"bundle pages > MAX_PAGES={ctx.limits.max_pages}")

    if not kinds_read:
        codes = [code for _name, code in skipped]
        if not codes:
            raise IngestError("archive_unsupported", "the ZIP holds no page files")
        if all(code in _BUNDLE_REFUSAL_CODES for code in codes):
            raise IngestError(
                "archive_unsupported",
                "the ZIP holds only PDFs or nested containers — unpack it (PDFs are read in an isolated process, "
                "not inside bundles)",
            )
        worst = max(set(codes), key=codes.count)
        names = ", ".join(name for name, code in skipped if code == worst)
        raise IngestError(worst, f"no member of the bundle could be read ({worst}: {names[:300]})")

    ctx.native_pages = all(READERS[k].native_pages for k in kinds_read)
    origins = {READERS[k].default_origin or "ocr:generic" for k in kinds_read}
    ctx.origin_hint = origins.pop() if len(origins) == 1 else "ocr:generic"
    ctx.encoding = encodings.pop() if len(encodings) == 1 else None
    ctx.notes.append(f"bundle_members={len(kinds_read)}")
    ctx.notes.append("bundle_kinds=" + ",".join(sorted(set(kinds_read))))
    if skipped:
        ctx.notes.append(f"zip_members_skipped={len(skipped)}")
        ctx.notes.extend(f"bundle_member_failed:{name}:{code}" for name, code in skipped[:5])
    if duplicates:
        ctx.notes.append(f"bundle_duplicates_skipped={duplicates}")
    if ignored:
        ctx.notes.append(f"bundle_ignored={ignored}")
    return pages


# ── report-only quality flags ─────────────────────────────────────────────────


def _cp1250_misreads() -> Dict[str, str]:
    """Characters a CP1250 byte becomes when read as CP1252, where CP1250 meant Czech.

    Ported from atrium-llm-enrich api_util/digital_to_json.py (`_build_cp1250_misreads`,
    MIT, same project): derived from the two codecs, so the table is complete for this
    failure — `á`/`é` decode identically in both, which is why "Zpráva" survives a
    misread while "sondě" becomes "sondì".
    """
    czech = frozenset("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")
    table: Dict[str, str] = {}
    for byte in range(0x80, 0x100):
        try:
            western = bytes([byte]).decode("cp1252")
            eastern = bytes([byte]).decode("cp1250")
        except UnicodeDecodeError:
            continue
        if western != eastern and eastern in czech:
            table[western] = eastern
    return table


CP1250_MISREADS: Dict[str, str] = _cp1250_misreads()
#: llm-enrich's GARBAGE_MIN_HITS / QUALITY_GARBAGE_BELOW: two misreads in a line, or
#: fewer than 90% of its letters clean, is a decode fault rather than a foreign word.
_MOJIBAKE_MIN_HITS = 2
_MOJIBAKE_SCORE_BELOW = 0.90
#: A page is flagged when at least this share of its letter-bearing lines is.
_MOJIBAKE_PAGE_SHARE = 0.2
#: Czech letters both code pages decode alike (so a misread keeps them): evidence the
#: line is Czech at all. French, Italian or Norwegian text has è/ì/ø/ù of its own.
_MOJIBAKE_CZECH_EVIDENCE = frozenset("áíúýšžÁÍÚÝŠŽ")
#: Letters a CP1250 misread of Czech cannot produce (they would need ŕ/ę/ű/ś):
#: evidence of Western text.
_MOJIBAKE_WESTERN_EVIDENCE = frozenset("àêûœÀÊÛŒ")


def mojibake_line(line: str) -> bool:
    """True for a line that reads like CP1250 Czech decoded as CP1252 ("sondì èíslo").

    llm-enrich's rule (≥2 misread characters, or <90% of the letters clean), made
    conservative for a report flag: the line must also hold a letter both code pages
    share with Czech (á, í, ú, ý, š, ž), none that only Western text has (à, ê, û,
    œ), and round-trip through CP1252. Correctly decoded Czech holds č/ř/ě, which
    CP1252 cannot encode, so it never matches.
    """
    letters = [c for c in line if c.isalpha()]
    hits = sum(1 for c in letters if c in CP1250_MISREADS)
    if not hits or not (hits >= _MOJIBAKE_MIN_HITS or 1 - hits / len(letters) < _MOJIBAKE_SCORE_BELOW):
        return False
    chars = set(letters)
    if not chars & _MOJIBAKE_CZECH_EVIDENCE or chars & _MOJIBAKE_WESTERN_EVIDENCE:
        return False
    try:
        line.encode("cp1252")
    except UnicodeEncodeError:
        return False
    return True


def _flag_mojibake(page: TextPage) -> bool:
    lettered = [ln for ln in page.lines if any(c.isalpha() for c in ln)]
    bad = sum(1 for ln in lettered if mojibake_line(ln))
    if bad and bad >= _MOJIBAKE_PAGE_SHARE * len(lettered):
        if "mojibake_cp1252" not in page.flags:
            page.flags.append("mojibake_cp1252")
        return True
    return False


# ── dispatcher ────────────────────────────────────────────────────────────────


@dataclass
class _Ctx:
    path: str
    limits: Limits
    options: ReaderOptions
    kind: str = ""
    notes: List[str] = field(default_factory=list)
    #: The file name that decides dialects when it is not the path's: the inner name
    #: of a compressed file, a bundle member's name.
    name: str = ""
    origin_hint: Optional[str] = None
    native_pages: Optional[bool] = None  # overrides the registry (a bundle of native pages)
    encoding: Optional[str] = None  # set by path readers that decode text themselves
    expanded_chars: int = 0
    json_stats: Any = None


_TEXT_READERS: Dict[str, Callable[[str, _Ctx], List[TextPage]]] = {
    "txt": read_plain,
    "md": read_markdown,
    "csv": lambda text, ctx: read_csv_table(text, ctx),
    "tsv": lambda text, ctx: read_csv_table(text, ctx, delimiter="\t"),
    "tesseract-tsv": read_tesseract_tsv,
    "json": read_json,
    "jsonl": read_jsonl,
    "html": read_html,
    "hocr": read_html,
    "srt": read_subtitles,
    "vtt": read_subtitles,
}
_PATH_READERS: Dict[str, Callable[[str, _Ctx], List[TextPage]]] = {
    "pdf": read_pdf,
    "docx": read_docx,
    "xlsx": read_xlsx,
    "pptx": read_pptx,
    "odt": read_odf,
    "ods": read_odf,
    "odp": read_odf,
    "epub": read_epub,
    "zip-bundle": read_zip_bundle,
}
_BYTES_READERS: Dict[str, Callable[[bytes, _Ctx], List[TextPage]]] = {
    "alto": read_xml,
    "page-xml": read_xml,
    "tei": read_xml,
    "xml": read_xml,
    "abbyy-xml": read_xml,
    "djvu-xml": read_xml,
    "rtf": read_rtf,
    "eml": read_email,
    "mbox": read_email,
}

#: Kinds read in a separate process with a timeout: native code (PDFium) can crash
#: or hang on hostile input, and that must cost one file, not the whole stage.
ISOLATED_KINDS = frozenset({"pdf"})


def _read_bytes_kind(kind: str, data: bytes, ctx: _Ctx) -> Tuple[List[TextPage], Optional[str]]:
    """(pages, encoding) for content already in memory — a file, a decompressed
    stream, a bundle member."""
    if kind in _BYTES_READERS:
        return _BYTES_READERS[kind](data, ctx), None
    if kind in _TEXT_READERS:
        text, encoding, flags = decode_bytes(data, ctx.options.fallback_encodings)
        ctx.notes.extend(flags)
        return _TEXT_READERS[kind](text, ctx), encoding
    raise IngestError("archive_unsupported", f"{kind} content cannot be read from inside a container")


def _run_reader(fn: Callable, *args):
    """Run one reader, mapping every failure onto a reason code: a reader bug or a
    hostile file costs this file, never the run."""
    try:
        return fn(*args)
    except IngestError:
        raise
    except RecursionError as exc:
        raise IngestError("malformed", "structure nested too deeply") from exc
    except MemoryError as exc:
        raise IngestError("too_large", "out of memory while reading") from exc
    except PermissionError as exc:
        raise IngestError("unreadable", f"cannot read file ({exc})") from exc
    except Exception as exc:
        raise IngestError("malformed", f"{type(exc).__name__}: {str(exc)[:300]}") from exc


def _finalize(doc_kind: str, pages: List[TextPage], ctx: _Ctx, encoding: Optional[str]) -> TextDocument:
    spec = READERS[doc_kind]
    native = spec.native_pages if ctx.native_pages is None else ctx.native_pages
    out_pages: List[TextPage] = []
    surrogates = 0
    for n, page in enumerate(pages, 1):
        raw_lines = [p for ln in page.lines for p in normalize_newlines(ln).replace("\f", "\n").split("\n")]
        surrogates += sum(1 for ln in raw_lines if _SURROGATES.search(ln))
        page.lines = [normalize_line(p) for p in raw_lines]
        if not page.label:
            page.label = str(n)
        if len(page.lines) > ctx.limits.max_lines_per_page:
            if native:
                raise IngestError("too_large", f"page {page.label!r} has {len(page.lines)} lines > MAX_LINES_PER_PAGE")
            step = ctx.limits.max_lines_per_page
            chunks = [page.lines[i : i + step] for i in range(0, len(page.lines), step)]
            if "page_overflow_split" not in ctx.notes:
                ctx.notes.append("page_overflow_split")
            for k, chunk in enumerate(chunks):
                label = page.label if k == 0 else f"{page.label}+{k}"
                out_pages.append(TextPage(chunk, label=label, flags=list(page.flags) + (["overflow"] if k else [])))
            continue
        out_pages.append(page)
    if len(out_pages) > ctx.limits.max_pages:
        raise IngestError("too_large", f"{len(out_pages)} pages > MAX_PAGES={ctx.limits.max_pages}")
    if surrogates:
        ctx.notes.append(f"lone_surrogates_dropped={surrogates}")
    flagged = sum(1 for page in out_pages if _flag_mojibake(page))
    if flagged:
        ctx.notes.append(f"mojibake_cp1252_pages={flagged}")
    return TextDocument(
        kind=doc_kind,
        media_type=spec.media_type,
        pages=out_pages or [TextPage([], label="1")],
        encoding=encoding if encoding is not None else ctx.encoding,
        notes=ctx.notes,
        native_pages=native,
        origin_hint=ctx.origin_hint,
    )


def read_document(
    path: str, limits: Limits = DEFAULT_LIMITS, options: ReaderOptions = DEFAULT_OPTIONS, kind: Optional[str] = None
) -> TextDocument:
    """Read one file into a TextDocument (in-process). Raises IngestError."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise IngestError("unreadable", f"cannot stat file ({exc})") from exc
    if size == 0:
        raise IngestError("empty_file")
    if size > limits.max_file_mb * _MB:
        raise IngestError("too_large", f"{size / _MB:.1f} MB > MAX_FILE_MB={limits.max_file_mb}")
    notes: List[str] = []
    kind = kind or sniff_kind(path, limits, notes)
    wrapper = compression_of(path)
    name = os.path.basename(path)
    ctx = _Ctx(path, limits, options, kind=kind, notes=notes, name=inner_name(name) if wrapper else name)
    if wrapper:
        ctx.notes.append(f"decompressed:{wrapper}")
        if kind in _PATH_READERS:
            raise IngestError("archive_unsupported", f"a compressed {kind} file — decompress it first")
        pages, encoding = _run_reader(_read_bytes_kind, kind, decompress_file(path, wrapper, limits), ctx)
    elif kind in _PATH_READERS:
        pages, encoding = _run_reader(_PATH_READERS[kind], path, ctx), None
    else:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            raise IngestError("unreadable", f"cannot read file ({exc})") from exc
        pages, encoding = _run_reader(_read_bytes_kind, kind, data, ctx)
    return _finalize(ctx.kind or kind, pages, ctx, encoding)


def read_document_isolated(
    path: str, limits: Limits = DEFAULT_LIMITS, options: ReaderOptions = DEFAULT_OPTIONS, kind: Optional[str] = None
) -> TextDocument:
    """Like read_document, but kinds in ISOLATED_KINDS run in a child process with
    READER_TIMEOUT_S. A plain `subprocess` of this file (not multiprocessing), so the
    child imports only this module — never the caller's __main__ (the service)."""
    notes: List[str] = []
    kind = kind or sniff_kind(path, limits, notes)
    if kind not in ISOLATED_KINDS:
        doc = read_document(path, limits, options, kind=kind)
        doc.notes = notes + doc.notes
        return doc
    payload = json.dumps(
        {"path": os.path.abspath(path), "kind": kind, "limits": asdict(limits), "options": asdict(options)}
    )
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--isolated-worker"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=limits.reader_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise IngestError("timeout", f"reader exceeded READER_TIMEOUT_S={limits.reader_timeout_s:g}s") from exc
    try:
        result = json.loads(proc.stdout) if proc.stdout.strip() else None
    except ValueError:
        result = None
    if not isinstance(result, dict):
        tail = (proc.stderr or "").strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
        raise IngestError("reader_crashed", tail[0][:300])
    if result.get("status") != "ok":
        raise IngestError(result.get("code", "reader_crashed"), result.get("message", ""))
    doc = TextDocument.from_dict(result["document"])
    doc.notes = notes + doc.notes
    return doc


def _isolated_worker_main() -> int:
    request = json.loads(sys.stdin.read())
    limits = Limits(**request["limits"])
    opt = request["options"]
    opt["fallback_encodings"] = tuple(opt.get("fallback_encodings") or ())
    options = ReaderOptions(**opt)
    try:
        doc = read_document(request["path"], limits, options, kind=request.get("kind"))
        # ASCII JSON: the child's stdout is decoded with the parent's locale encoding, and
        # Czech text written raw failed as `reader_crashed` on a non-UTF-8 locale.
        sys.stdout.write(json.dumps({"status": "ok", "document": doc.to_dict()}))
    except IngestError as exc:
        sys.stdout.write(json.dumps({"status": "error", "code": exc.code, "message": exc.message}))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--isolated-worker"]:
        # Make sibling modules (page_split, for the JSON page detection) importable when the
        # worker is launched by absolute path from any working directory.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        raise SystemExit(_isolated_worker_main())
    raise SystemExit("text_formats.py is a library; run text_split.py for the text-lines ingest stage.")
