#!/usr/bin/env python3
"""
text_formats.py — format detection and readers for the text-lines input path (#31).

Turns any sequentially readable, text-bearing file into an ordered list of pages,
each an ordered list of text lines — the `<format>-2-txt` step #31 asks for — so
the line-quality categorization can run on text of any shape, not only on ALTO XML.

Design rules (see docs/text_inputs.md for the full matrix and the reason codes):

* **Content decides, not the extension.** Binary containers are identified by their
  magic bytes (`%PDF-`, ZIP members, OLE2), XML by its root element, JSON by a parse.
  The extension only chooses between plain-text dialects (CSV/TSV/Markdown/JSONL).
* **Pages keep the reading order and the block structure.** A "page" is a real page
  where the format has one (PDF, ALTO, PAGE-XML, hOCR, DOCX/ODT page breaks) and
  otherwise the natural block: a JSON child object, a JSONL record, a sheet, a slide,
  an EPUB chapter, a form-feed section of a plain-text file.
* **Lines are the format's own units** — a physical line, a paragraph, a table cell
  or a spreadsheet row. Very long lines are wrapped later by `shape_lines()`, the
  same helper the extract stage and the service use.
* **Light dependencies.** Standard library + lxml (already a repo dependency) for
  every XML/ZIP format — no python-docx, openpyxl or pdfplumber. PDF uses pypdfium2
  and non-UTF-8 plain text uses charset-normalizer; both are imported lazily, so a
  missing one only affects the files that need it (`dependency_missing`).
* **Fail closed, per file.** Every problem is an `IngestError` with a stable reason
  code; callers record it and move on to the next file.

This module never prints (it is imported by the service, whose logging contract
forbids `print`); it logs through `logging` only.
"""

from __future__ import annotations

import csv
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
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
}


class IngestError(Exception):
    """A per-file ingest failure carrying a stable reason `code` (see REASON_CODES)."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message or REASON_CODES.get(code, code)


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


PAGE_BREAK_MODES = ("auto", "explicit", "none")

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

    def _num(key, default, cast, minimum):
        raw = cfg.get(section, key, fallback="").strip()
        if not raw:
            return default
        try:
            value = cast(raw)
        except ValueError as exc:
            raise ValueError(f"[{section}] {key} = {raw!r} is not a valid number") from exc
        if value < minimum:
            raise ValueError(f"[{section}] {key} = {raw!r} must be >= {minimum}")
        return value

    def _bool(key, default):
        raw = cfg.get(section, key, fallback="").strip().lower()
        if not raw:
            return default
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"[{section}] {key} = {raw!r} is not a boolean")

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

    opt = ReaderOptions(
        fallback_encodings=encodings,
        page_breaks=page_breaks,
        pdf_min_text_chars=_num("PDF_MIN_TEXT_CHARS", opt.pdf_min_text_chars, int, 0),
        pdf_garble_threshold=_num("PDF_GARBLE_THRESHOLD", opt.pdf_garble_threshold, float, 0.0),
        pdf_ocr_layer_min_ratio=_num("PDF_OCR_LAYER_MIN_RATIO", opt.pdf_ocr_layer_min_ratio, float, 0.0),
        max_line_chars=_num("MAX_LINE_CHARS", opt.max_line_chars, int, 0),
        keep_blank_lines=_bool("KEEP_BLANK_LINES", opt.keep_blank_lines),
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
    )
}

#: Extensions that are never content and are refused before reading (image scans
#: are the main case: they need OCR, not text ingest).
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".gif", ".bmp", ".webp", ".jp2", ".j2k"}


def supported_extensions() -> List[str]:
    """Every extension some reader claims, sorted (for UIs and `accept=` lists)."""
    return sorted({ext for spec in READERS.values() for ext in spec.extensions})


def default_source_origin(doc: TextDocument) -> str:
    """The truthful `source.origin` for a read document (#31 origin policy).

    PDF is decided per document: a text layer that is mostly invisible text over the
    page image is an OCR layer (`ocr:pdf-text-layer`, this repo's to own); anything
    else is a born-digital PDF (`digital-born-pdf`, llm-enrich's digital-convert's).
    """
    if doc.kind == "pdf":
        text_pages = [p for p in doc.pages if p.text_layer in ("ocr", "digital", "garbled")]
        ocr_pages = [p for p in text_pages if p.text_layer == "ocr"]
        if text_pages and len(ocr_pages) * 2 >= len(text_pages):
            return "ocr:pdf-text-layer"
        return "digital-born-pdf"
    spec = READERS.get(doc.kind)
    return spec.default_origin if spec else "ocr:generic"


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
    become plain spaces; remaining C0/C1 controls except \\t are removed; stripped.
    """
    if not line:
        return ""
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
    `xml_recovered`, since recovered text may be incomplete).
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

    def explicit_break(self) -> None:
        self.line_break()
        if self._any:
            self.pages.append(TextPage([]))

    def rendered_break(self) -> None:
        self.line_break()
        if self.pages[-1].lines:
            self.pages.append(TextPage([]))

    def finish(self) -> List[TextPage]:
        self.line_break()
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
_ARCHIVE_MAGIC = (b"\x1f\x8b", b"\xfd7zXZ\x00", b"7z\xbc\xaf\x27\x1c", b"Rar!\x1a\x07", b"\x28\xb5\x2f\xfd")
_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
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


def _resolve_part(base_dir: str, target: str) -> str:
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(base_dir, target))


def _rels_map(zf: zipfile.ZipFile, part: str) -> Dict[str, Tuple[str, str]]:
    """{rId: (Type, resolved target)} for one OOXML part's relationships."""
    base_dir = posixpath.dirname(part)
    rels_name = posixpath.join(base_dir, "_rels", posixpath.basename(part) + ".rels")
    out: Dict[str, Tuple[str, str]] = {}
    if rels_name not in zf.namelist():
        return out
    root = parse_xml_bytes(_zip_read(zf, rels_name))
    for rel in root:
        if _local(rel.tag) == "Relationship" and rel.get("TargetMode", "") != "External":
            out[rel.get("Id", "")] = (rel.get("Type", ""), _resolve_part(base_dir, rel.get("Target", "")))
    return out


def _ooxml_main_part(zf: zipfile.ZipFile) -> Optional[str]:
    return _rels_target(zf, "_rels/.rels", "/officeDocument")


def _sniff_zip(path: str, limits: Limits) -> str:
    with open_zip(path, limits) as zf:
        names = set(zf.namelist())
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
    raise IngestError("archive_unsupported", "ZIP archive that is not DOCX/XLSX/PPTX/ODF/EPUB")


def _xml_root_name(head: bytes) -> Tuple[str, bytes]:
    """(root local name, namespace-ish prefix bytes) from the first bytes of an XML file."""
    body = _XML_PROLOG_NOISE.sub(b"", head, count=1)
    match = _ROOT_TAG.match(body)
    if not match:
        return "", b""
    return match.group(2).decode("ascii", errors="replace"), body[:2048]


def _xml_kind(head: bytes) -> str:
    name, start = _xml_root_name(head)
    lname = name.lower()
    if lname == "alto":
        return "alto"
    if lname == "pcgts":
        return "page-xml"
    if lname in ("tei", "teicorpus"):
        return "tei"
    if lname == "html":
        return "hocr" if _HOCR_HINT.search(head) else "html"
    if lname:
        return "xml"
    return ""


def sniff_kind(path: str, limits: Limits = DEFAULT_LIMITS, notes: Optional[List[str]] = None) -> str:
    """Decide which reader handles `path` (see the module docstring's rules).

    Raises IngestError for files that are not text-bearing inputs (images, legacy
    Office, archives, binaries) with the reason code a user can act on.
    """
    notes = notes if notes is not None else []
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as fh:
        head = fh.read(65536)
    if not head:
        raise IngestError("empty_file")

    kind = ""
    if b"%PDF-" in head[:1024]:
        kind = "pdf"
    elif head.startswith(_OLE2_MAGIC):
        raise IngestError("legacy_office_unsupported")
    elif head.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        kind = _sniff_zip(path, limits)
    elif head.startswith(_IMAGE_MAGIC) or (head[:4] == b"RIFF" and head[8:12] == b"WEBP"):
        raise IngestError("image_needs_ocr")
    elif head.startswith(_ARCHIVE_MAGIC) or (head.startswith(b"BZh") and head[4:10] == b"1AY&SY"):
        raise IngestError("archive_unsupported")
    elif head.lstrip().startswith(b"{\\rtf"):
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
            if not kind and ext in (".html", ".htm", ".xhtml"):
                kind = "html"
            if not kind:
                kind = "txt"
        elif ext in (".jsonl", ".ndjson"):
            kind = "jsonl"
        elif ext == ".json":
            kind = "json"
        elif stripped.startswith((b"{", b"[")):
            kind = _json_or_text(path, ext, limits)
        elif ext in (".csv",):
            kind = "csv"
        elif ext in (".tsv", ".tab"):
            kind = "tsv"
        elif ext in (".md", ".markdown", ".mdown"):
            kind = "md"
        else:
            kind = "txt"

    if ext in _IMAGE_EXTENSIONS and kind in ("txt",):
        raise IngestError("image_needs_ocr", f"{ext} file")
    spec = READERS.get(kind)
    if spec and ext and ext not in spec.extensions:
        notes.append(f"extension {ext} but content is {kind}")
    return kind


def _json_or_text(path: str, ext: str, limits: Limits) -> str:
    """A `{`/`[`-leading file is JSON if it parses, JSONL if every record does."""
    size = os.path.getsize(path)
    if size > limits.max_file_mb * _MB:
        return "json" if ext == ".json" else "txt"
    with open(path, "rb") as fh:
        data = fh.read()
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


_CSV_TEXT_COLUMNS = ("text", "line", "content", "transcription", "sentence", "string")
_CSV_PAGE_COLUMNS = ("page_num", "page", "page_number", "pagenumber", "page_no")


def read_csv_table(text: str, ctx: "_Ctx", delimiter: Optional[str] = None) -> List[TextPage]:
    """CSV/TSV: a recognised text column (text/line/content/transcription/…) gives
    one line per row, grouped into pages by a page/page_num column in first-seen
    order; without one, every row is a line of its non-empty cells joined by tab."""
    text = normalize_newlines(text).replace("\x00", "")
    if delimiter is None:
        sample = text[:65536]
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    old_limit = csv.field_size_limit()
    csv.field_size_limit(max(old_limit, int(ctx.limits.max_file_mb * _MB)))
    try:
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    except csv.Error as exc:
        raise IngestError("malformed", f"CSV could not be parsed ({exc})") from exc
    finally:
        csv.field_size_limit(old_limit)
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return [TextPage([])]

    header = [c.strip().lower() for c in rows[0]]
    text_col = next((header.index(c) for c in _CSV_TEXT_COLUMNS if c in header), None)
    if text_col is None:
        return [TextPage(["\t".join(c.strip() for c in r if c.strip()) for r in rows])]

    page_col = next((header.index(c) for c in _CSV_PAGE_COLUMNS if c in header), None)
    ctx.notes.append(f"csv text column {header[text_col]!r}")
    pages: Dict[str, List[str]] = {}
    for r in rows[1:]:
        value = r[text_col] if text_col < len(r) else ""
        page = (r[page_col].strip() if page_col is not None and page_col < len(r) else "") or "1"
        pages.setdefault(page, []).append(value)
    return [TextPage(lines, label=label) for label, lines in pages.items()]


# ── readers: JSON family ──────────────────────────────────────────────────────

_WORD_LEVEL_KEYS = {"words", "word", "strings", "string", "tokens", "token", "glyphs", "symbols", "chars", "characters"}
_LINE_LEVEL_KEYS = {"lines", "line", "textlines", "textline", "text_lines", "text_line"}


def _json_target_keys():
    # Lazy: extract_JSON_2_TXT reads config and imports pandas at import time.
    from extract_JSON_2_TXT import TARGET_KEYS

    return TARGET_KEYS


def _json_leaves(data: Any, keys: Optional[set], current_key: Optional[str] = None) -> List[str]:
    """Ordered text leaves under whitelisted keys (`keys=None`: every string with a letter).

    One refinement over extract_JSON_2_TXT's walk: when a dict carries a line-level
    container (`lines`), its word-level siblings (`words`, `strings`, …) are skipped,
    so engines that emit both (Azure DI) do not produce every word twice.
    """
    out: List[str] = []
    stack: List[Tuple[Any, Optional[str]]] = [(data, current_key)]
    while stack:
        node, key = stack.pop()
        if isinstance(node, dict):
            has_lines = any(isinstance(k, str) and k.lower() in _LINE_LEVEL_KEYS and v for k, v in node.items())
            items = [
                (v, k)
                for k, v in node.items()
                if not (has_lines and isinstance(k, str) and k.lower() in _WORD_LEVEL_KEYS)
            ]
            stack.extend(reversed(items))
        elif isinstance(node, list):
            stack.extend((item, key) for item in reversed(node))
        elif isinstance(node, str):
            text = node.strip()
            if not text:
                continue
            if keys is None:
                if any(ch.isalpha() for ch in text):
                    out.extend(normalize_newlines(text).split("\n"))
            elif key is not None and str(key).lower() in keys:
                out.extend(normalize_newlines(text).split("\n"))
    return out


def _json_pages(data: Any, ctx: "_Ctx", keys: Optional[set]) -> List[TextPage]:
    """Pages for one parsed JSON value: Family A/B (same detection as json-keys),
    else top-level children as blocks. Header siblings are NOT copied into every
    page (json-keys does, because it re-serialises whole documents per page)."""
    from page_split import PAGE_NUMBER_FIELD_KEYS, _find_family_a, _find_family_b, _get_field_ci

    family_a = _find_family_a(data)
    if family_a is not None:
        _parent, _key, page_list = family_a
        pages = []
        for i, page_obj in enumerate(page_list, 1):
            number = _get_field_ci(page_obj, PAGE_NUMBER_FIELD_KEYS)
            pages.append(TextPage(_json_leaves(page_obj, keys), label=str(number) if number is not None else str(i)))
        ctx.notes.append("json page list")
        return pages

    family_b = _find_family_b(data)
    if family_b is not None:
        _parent, _key, _lst, _field, groups = family_b
        ctx.notes.append("json page-tagged list")
        return [TextPage(_json_leaves(items, keys), label=str(value)) for value, items in groups.items()]

    blocks: List[Tuple[str, Any]] = []
    if isinstance(data, list) and len(data) >= 2:
        blocks = [(str(i), item) for i, item in enumerate(data, 1)]
    elif isinstance(data, dict):
        blocks = [(str(k), v) for k, v in data.items() if isinstance(v, (dict, list))]
    block_pages = []
    for label, value in blocks:
        leaves = _json_leaves(value, keys)
        if leaves:
            block_pages.append(TextPage(leaves, label=label))
    if len(block_pages) >= 2:
        ctx.notes.append("json top-level blocks")
        return block_pages
    return [TextPage(_json_leaves(data, keys), label="1")]


def _with_fallback_keys(build: Callable[[Optional[set]], List[TextPage]], ctx: "_Ctx") -> List[TextPage]:
    pages = build(_json_target_keys())
    if not any(ln.strip() for p in pages for ln in p.lines):
        ctx.notes.append("json_all_strings")
        pages = build(None)
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
            leaves = [value.strip()] if isinstance(value, str) else _json_leaves(value, keys)
            if leaves:
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


def read_page_xml(root, ctx: "_Ctx") -> List[TextPage]:
    """PAGE XML: ReadingOrder → TextRegion → TextLine (TextEquiv/Unicode, else Words)."""
    pages = []
    for i, page in enumerate(root.iter("{*}Page"), 1):
        regions = {r.get("id", f"_r{n}"): r for n, r in enumerate(page.iter("{*}TextRegion"))}
        ordered = [rid for rid in _page_reading_order(page) if rid in regions]
        seen = set(ordered)
        ordered += [rid for rid in regions if rid not in seen]
        lines: List[str] = []
        done_lines = set()
        for rid in ordered:
            region = regions[rid]
            region_lines = [tl for tl in region if _local(tl.tag) == "TextLine"]
            if not region_lines and not any(_local(c.tag) == "TextRegion" for c in region):
                text = _page_xml_text(region)
                lines.extend(normalize_newlines(text).split("\n") if text else [])
                continue
            for tl in region_lines:
                if id(tl) in done_lines:
                    continue
                done_lines.add(id(tl))
                text = _page_xml_text(tl)
                if not text:
                    words = [_page_xml_text(w) for w in tl if _local(w.tag) == "Word"]
                    text = " ".join(w for w in words if w)
                lines.append(text)
        label = page.get("imageFilename") or str(i)
        pages.append(TextPage(lines, label=os.path.splitext(os.path.basename(label))[0] or str(i)))
    return pages or [TextPage([])]


_TEI_BLOCKS = {
    "p", "head", "l", "item", "cell", "ab", "div", "lg", "list", "table", "row", "note", "label", "trailer",
    "byline", "dateline", "opener", "closer", "salute", "signed", "fw", "argument", "epigraph", "docTitle",
    "titlePart", "docAuthor", "docDate", "castItem", "sp", "speaker", "stage", "figDesc", "bibl", "u", "s",
}  # fmt: skip
_TEI_SKIP = {"teiHeader", "facsimile", "standOff", "sourceDoc"}


def _flow_read(root, blocks, skip_names, line_break, page_break, collapse=True, extra_start=None) -> List[TextPage]:
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
        return _local(el.tag) in skip_names

    _walk(root, on_start, on_end, flow.text, skip)
    return flow.finish()


def read_tei(root, ctx: "_Ctx") -> List[TextPage]:
    """TEI/TEITOK: pages at <pb/>, lines at <lb/> and block ends; header skipped."""
    text_el = next((e for e in root.iter("{*}text")), None)
    return _flow_read(text_el if text_el is not None else root, _TEI_BLOCKS, _TEI_SKIP, {"lb"}, {"pb"})


def read_generic_xml(root, ctx: "_Ctx") -> List[TextPage]:
    """Generic XML: root children are blocks (pages) when two or more carry text;
    an element with its own text (mixed content) is one line, containers descend."""

    def lines_of(el) -> List[str]:
        out: List[str] = []
        stack = [el]
        while stack:
            node = stack.pop()
            if not isinstance(node.tag, str):
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
        for i, child in enumerate(children, 1):
            lines = lines_of(child)
            if lines:
                blocks.append(TextPage(lines, label=f"{_local(child.tag)}[{i}]"))
        if len(blocks) >= 2:
            return blocks
    return [TextPage(lines_of(root), label="1")]


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


def read_xml(data: bytes, ctx: "_Ctx") -> List[TextPage]:
    root = parse_xml_bytes(data, ctx.options.fallback_encodings, ctx.notes)
    name = _local(root.tag).lower()
    if name == "alto":
        ctx.kind = "alto"
        return read_alto(root, ctx)
    if name == "pcgts":
        ctx.kind = "page-xml"
        return read_page_xml(root, ctx)
    if name in ("tei", "teicorpus"):
        ctx.kind = "tei"
        return read_tei(root, ctx)
    if name == "html":
        text, _enc, _fl = decode_bytes(data, ctx.options.fallback_encodings)
        ctx.kind = "html"
        return read_html(text, ctx)
    ctx.kind = "xml"
    return read_generic_xml(root, ctx)


# ── readers: PDF (pypdfium2, isolated) ────────────────────────────────────────

_PDF_OBJECT_CAP = 20000


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
                n_text = n_invisible = n_images = 0
                for n, obj in enumerate(
                    page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_TEXT, pdfium_c.FPDF_PAGEOBJ_IMAGE), max_depth=4)
                ):
                    if n >= _PDF_OBJECT_CAP:
                        break
                    if obj.type == pdfium_c.FPDF_PAGEOBJ_TEXT:
                        n_text += 1
                        if pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw) == pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE:
                            n_invisible += 1
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
            pages.append(TextPage(lines, label=label, text_layer=layer, needs_ocr_reason=reason, images=n_images))
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


def read_docx(path: str, ctx: "_Ctx") -> List[TextPage]:
    """DOCX via zipfile + lxml: body order, tables row-major, text boxes after their
    anchor paragraph, pages at explicit breaks (and at Word's rendered page breaks
    in `auto` mode)."""
    mode = ctx.options.page_breaks
    with open_zip(path, ctx.limits) as zf:
        main = _ooxml_main_part(zf) or "word/document.xml"
        root = parse_xml_bytes(_zip_read(zf, main), ctx.options.fallback_encodings, ctx.notes)
    body = next((el for el in root if _local(el.tag) == "body"), None)
    if body is None:
        raise IngestError("malformed", "DOCX main part has no <w:body>")

    flow = _Flow()
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
        if name == "sectPr" and el.getparent() is body:
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

    _walk(body, on_start, on_end, _ignore_text, skip)
    return flow.finish()


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
    sheet name), row = line of its text cells joined by tab. Numbers, dates,
    booleans and errors are not text and are dropped."""
    etree = _etree()
    with open_zip(path, ctx.limits) as zf:
        workbook = _ooxml_main_part(zf) or "xl/workbook.xml"
        wb_root = parse_xml_bytes(_zip_read(zf, workbook))
        rels = _rels_map(zf, workbook)
        shared = _xlsx_shared_strings(zf, rels)
        sheets_el = next((el for el in wb_root if _local(el.tag) == "sheets"), None)
        pages = []
        for sheet in list(sheets_el) if sheets_el is not None else []:
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
                        lines.append("\t".join(cells))
                        if len(lines) > ctx.limits.max_lines_per_page:
                            raise IngestError("too_large", f"sheet {sheet.get('name')!r} > MAX_LINES_PER_PAGE")
                    row.clear()
            except etree.XMLSyntaxError as exc:
                raise IngestError("malformed", f"worksheet {target!r} could not be parsed ({exc})") from exc
            if lines:
                pages.append(TextPage(lines, label=sheet.get("name") or str(len(pages) + 1)))
    return pages or [TextPage([])]


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
    """PPTX: slide = page in presentation (sldIdLst) order; paragraph = line."""
    with open_zip(path, ctx.limits) as zf:
        presentation = _ooxml_main_part(zf) or "ppt/presentation.xml"
        pres_root = parse_xml_bytes(_zip_read(zf, presentation))
        rels = _rels_map(zf, presentation)
        id_list = next((el for el in pres_root if _local(el.tag) == "sldIdLst"), None)
        pages = []
        for n, sld in enumerate(list(id_list) if id_list is not None else [], 1):
            rid = next((v for k, v in sld.attrib.items() if _local(k) == "id" and _ns(k)), "")
            typ, target = rels.get(rid, ("", ""))
            if not typ.endswith("/slide") or target not in zf.namelist():
                continue
            slide_root = parse_xml_bytes(_zip_read(zf, target))
            pages.append(TextPage(_drawingml_page(slide_root), label=str(n)))
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


def _odf_text_flow(root_el, ctx, before=frozenset(), after=frozenset()) -> List[TextPage]:
    mode = ctx.options.page_breaks
    flow = _Flow()
    deferred: List[Any] = []
    para_depth = [0]
    skip_names = {"note", "annotation", "tracked-changes", "notes", "sequence-decls", "forms", "deletion"}

    def skip(el):
        name = _local(el.tag)
        if name == "text-box":
            deferred.append(el)
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
    return flow.finish()


_ODF_REPEAT_CAP = 100


def _ods_sheet_lines(table, ctx) -> List[str]:
    lines: List[str] = []
    rows = [r for r in table.iter("{*}table-row")]
    for row in rows:
        cells = []
        for cell in row:
            if _local(cell.tag) not in ("table-cell", "covered-table-cell"):
                continue
            paras = [_collapse_ws("".join(p.itertext())) for p in cell.iter("{*}p")]
            text = " ".join(p for p in paras if p)
            if not text:
                continue
            try:
                repeat = int(_odf_attr(cell, "number-columns-repeated", "1"))
            except ValueError:
                repeat = 1
            cells.extend([text] * max(1, min(repeat, _ODF_REPEAT_CAP)))
        if not cells:
            continue
        try:
            repeat = int(_odf_attr(row, "number-rows-repeated", "1"))
        except ValueError:
            repeat = 1
        lines.extend(["\t".join(cells)] * max(1, min(repeat, _ODF_REPEAT_CAP)))
        if len(lines) > ctx.limits.max_lines_per_page:
            raise IngestError("too_large", "sheet > MAX_LINES_PER_PAGE")
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
        return _odf_text_flow(inner, ctx, before, after)
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
            item.get("id"): (_resolve_part(base, item.get("href", "")), item.get("media-type", ""))
            for item in opf.iter("{*}item")
        }
        encrypted = set()
        if "META-INF/encryption.xml" in names:
            enc_root = parse_xml_bytes(_zip_read(zf, "META-INF/encryption.xml"))
            for ref in enc_root.iter("{*}CipherReference"):
                encrypted.add(_resolve_part("", ref.get("URI", "")))
        pages = []
        for itemref in opf.iter("{*}itemref"):
            href, media = manifest.get(itemref.get("idref"), ("", ""))
            if not href or href not in names or "html" not in media:
                continue
            if href in encrypted:
                raise IngestError("encrypted", "DRM-encrypted EPUB content")
            text, _enc, _fl = decode_bytes(_zip_read(zf, href), ctx.options.fallback_encodings)
            chapter = read_html(text, _Ctx(ctx.path, ctx.limits, ctx.options, kind="html"))
            lines = [ln for p in chapter for ln in p.lines]
            if lines:
                pages.append(TextPage(lines, label=posixpath.splitext(posixpath.basename(href))[0]))
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
_RTF_SYMBOLS = {b"emdash": "\u2014", b"endash": "\u2013", b"bullet": "\u2022", b"lquote": "\u2018",
                b"rquote": "\u2019", b"ldblquote": "\u201c", b"rdblquote": "\u201d", b"tab": "\t"}  # fmt: skip


def read_rtf(data: bytes, ctx: "_Ctx") -> List[TextPage]:
    """RTF via a small tokenizer: \\page → page, \\par/\\line/\\row → line, \\cell → tab;
    header/footer/footnote/picture/field-instruction destinations skipped; \\'xx
    decoded with the document code page (\\ansicpgN), \\uN with its fallback skip."""
    flow = _Flow(collapse=False)
    stack: List[Tuple[bool, int]] = []
    ignorable = False
    uc = 1
    skip_chars = 0
    codepage = "cp1252"
    pending = bytearray()
    group_start = False

    def flush_bytes():
        if pending:
            flow.text(pending.decode(codepage, errors="replace"))
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
                stack.append((ignorable, uc))
                group_start = True
            else:
                ignorable, uc = stack.pop() if stack else (False, 1)
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
                    flow.text(symbol.decode())
                elif symbol == b"~":
                    flow.text(" ")
                elif symbol == b"_":
                    flow.text("-")
                elif symbol in (b"\n", b"\r"):
                    flow.line_break()
            continue
        if word is not None:
            if word in _RTF_DESTINATIONS and starts_group:
                ignorable = True
                continue
            if word == b"ansicpg" and arg:
                try:
                    "".encode(f"cp{int(arg)}")
                    codepage = f"cp{int(arg)}"
                except (LookupError, ValueError):
                    pass
                continue
            if word == b"uc" and arg:
                uc = max(0, int(arg))
                continue
            if word == b"bin" and arg:
                pos = match.end() + max(0, int(arg))  # skip the raw binary payload
                continue
            if ignorable:
                continue
            if word == b"u" and arg:
                flush_bytes()
                code = int(arg)
                flow.text(chr(code + 65536 if code < 0 else code))
                skip_chars = uc
            elif word in (b"par", b"line", b"row", b"sect"):
                flush_bytes()
                flow.line_break()
            elif word == b"page":
                flush_bytes()
                flow.explicit_break()
            elif word == b"cell":
                flush_bytes()
                flow.text("\t")
            elif word in _RTF_SYMBOLS:
                flush_bytes()
                flow.text(_RTF_SYMBOLS[word])
            continue
        if ignorable:
            continue
        if hexbyte is not None:
            pending.append(int(hexbyte, 16))
            continue
        if text is not None:
            flush_bytes()
            flow.text(text.decode(codepage, errors="replace"))
    flush_bytes()
    pages = flow.finish()
    for page in pages:
        page.lines = [_collapse_ws(ln) for ln in page.lines]
    return pages


# ── dispatcher ────────────────────────────────────────────────────────────────


@dataclass
class _Ctx:
    path: str
    limits: Limits
    options: ReaderOptions
    kind: str = ""
    notes: List[str] = field(default_factory=list)


_TEXT_READERS: Dict[str, Callable[[str, _Ctx], List[TextPage]]] = {
    "txt": read_plain,
    "md": read_markdown,
    "csv": lambda text, ctx: read_csv_table(text, ctx),
    "tsv": lambda text, ctx: read_csv_table(text, ctx, delimiter="\t"),
    "json": read_json,
    "jsonl": read_jsonl,
    "html": read_html,
    "hocr": read_html,
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
}
_BYTES_READERS: Dict[str, Callable[[bytes, _Ctx], List[TextPage]]] = {
    "alto": read_xml,
    "page-xml": read_xml,
    "tei": read_xml,
    "xml": read_xml,
    "rtf": read_rtf,
}

#: Kinds read in a separate process with a timeout: native code (PDFium) can crash
#: or hang on hostile input, and that must cost one file, not the whole stage.
ISOLATED_KINDS = frozenset({"pdf"})


def _finalize(doc_kind: str, pages: List[TextPage], ctx: _Ctx, encoding: Optional[str]) -> TextDocument:
    spec = READERS[doc_kind]
    out_pages: List[TextPage] = []
    for n, page in enumerate(pages, 1):
        page.lines = [
            normalize_line(p) for ln in page.lines for p in normalize_newlines(ln).replace("\f", "\n").split("\n")
        ]
        if not page.label:
            page.label = str(n)
        if len(page.lines) > ctx.limits.max_lines_per_page:
            if spec.native_pages:
                raise IngestError("too_large", f"page {page.label!r} has {len(page.lines)} lines > MAX_LINES_PER_PAGE")
            step = ctx.limits.max_lines_per_page
            chunks = [page.lines[i : i + step] for i in range(0, len(page.lines), step)]
            ctx.notes.append("page_overflow_split")
            for k, chunk in enumerate(chunks):
                label = page.label if k == 0 else f"{page.label}+{k}"
                out_pages.append(TextPage(chunk, label=label, flags=list(page.flags) + (["overflow"] if k else [])))
            continue
        out_pages.append(page)
    if len(out_pages) > ctx.limits.max_pages:
        raise IngestError("too_large", f"{len(out_pages)} pages > MAX_PAGES={ctx.limits.max_pages}")
    return TextDocument(
        kind=doc_kind,
        media_type=spec.media_type,
        pages=out_pages or [TextPage([], label="1")],
        encoding=encoding,
        notes=ctx.notes,
        native_pages=spec.native_pages,
    )


def read_document(
    path: str, limits: Limits = DEFAULT_LIMITS, options: ReaderOptions = DEFAULT_OPTIONS, kind: Optional[str] = None
) -> TextDocument:
    """Read one file into a TextDocument (in-process). Raises IngestError."""
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise IngestError("corrupt", f"cannot stat file ({exc})") from exc
    if size == 0:
        raise IngestError("empty_file")
    if size > limits.max_file_mb * _MB:
        raise IngestError("too_large", f"{size / _MB:.1f} MB > MAX_FILE_MB={limits.max_file_mb}")
    notes: List[str] = []
    kind = kind or sniff_kind(path, limits, notes)
    ctx = _Ctx(path, limits, options, kind=kind, notes=notes)
    encoding = None
    try:
        if kind in _PATH_READERS:
            pages = _PATH_READERS[kind](path, ctx)
        else:
            with open(path, "rb") as fh:
                data = fh.read()
            if kind in _BYTES_READERS:
                pages = _BYTES_READERS[kind](data, ctx)
            else:
                text, encoding, flags = decode_bytes(data, options.fallback_encodings)
                ctx.notes.extend(flags)
                pages = _TEXT_READERS[kind](text, ctx)
    except IngestError:
        raise
    except RecursionError as exc:
        raise IngestError("malformed", "structure nested too deeply") from exc
    except MemoryError as exc:
        raise IngestError("too_large", "out of memory while reading") from exc
    except Exception as exc:  # any other reader failure is this file's, never the run's
        raise IngestError("malformed", f"{type(exc).__name__}: {str(exc)[:300]}") from exc
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
        sys.stdout.write(json.dumps({"status": "ok", "document": doc.to_dict()}, ensure_ascii=False))
    except IngestError as exc:
        sys.stdout.write(json.dumps({"status": "error", "code": exc.code, "message": exc.message}))
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--isolated-worker"]:
        # Make sibling modules (page_split, extract_JSON_2_TXT) importable when the
        # worker is launched by absolute path from any working directory.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        raise SystemExit(_isolated_worker_main())
    raise SystemExit("text_formats.py is a library; run text_split.py for the text-lines ingest stage.")
