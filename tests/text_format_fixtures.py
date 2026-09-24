"""
tests/text_format_fixtures.py — deterministic in-memory builders for the text-lines
input formats (#31). Not a test module (no `test_` prefix, so pytest does not collect
it); imported by tests/test_text_*.py and usable to regenerate data_samples/TEXT/.

Why hand-rolled rather than python-docx / openpyxl / reportlab: the same reason
atrium-llm-enrich's tests/fixtures/digital/make_fixtures.py gives — those writers stamp
dates and random ids, so their bytes are not reproducible, and none of them lets a test
build the edge cases the readers must survive (a rendered page break right after an
explicit one, Strict-namespace OOXML, a zip bomb, invisible OCR text). Every ZIP here
is STORED with a fixed 1980 timestamp and create_system=3, so the bytes are identical
on every machine; nothing is committed as a binary blob.

`pdf_bytes()` ports `_pdf_escape` / `_text_block` / `_build_pdf` from
atrium-llm-enrich tests/fixtures/digital/make_fixtures.py (MIT, same project), adding
an `invisible` switch (text render mode 3 — the OCR-layer construction), a text
`matrix` (mirrored / rotated text) and pages with no text at all.

(#31 Phase 4) Compressed wrappers (`compress_bytes`, fixed mtime), tar, OCR engine
exports (Tesseract TSV, ABBYY FineReader XML, DjVuXML, PAGE XML), e-mail and mailbox
builders, and DOCX notes/headers.
"""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import os
import tarfile
import zipfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W_NS_STRICT = "http://purl.oclc.org/ooxml/wordprocessingml/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_DOC = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
OFFICE_DOC_STRICT = "http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument"
SS_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"


def make_zip(members: Sequence[Tuple[str, Union[str, bytes]]], compression: int = zipfile.ZIP_STORED) -> bytes:
    """A reproducible ZIP: fixed timestamp, Unix create_system, members in the given order."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            info.create_system = 3
            info.compress_type = compression
            zf.writestr(info, data.encode("utf-8") if isinstance(data, str) else data)
    return buf.getvalue()


def _rels(entries: Iterable[Tuple[str, str, str]]) -> str:
    body = "".join(f'<Relationship Id="{rid}" Type="{typ}" Target="{target}"/>' for rid, typ, target in entries)
    return (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{REL_NS}">{body}</Relationships>'
    )


_CT = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/></Types>'
)


# ── DOCX ──────────────────────────────────────────────────────────────────────


def w_p(*runs: str, ppr: str = "") -> str:
    """A w:p with each run's inner XML (use w_t for plain text runs)."""
    return f"<w:p>{('<w:pPr>' + ppr + '</w:pPr>') if ppr else ''}{''.join(runs)}</w:p>"


def w_t(text: str) -> str:
    return f'<w:r><w:t xml:space="preserve">{text}</w:t></w:r>'


def docx_bytes(
    body_xml: str,
    *,
    main_part: str = "word/document.xml",
    strict: bool = False,
    footnotes: Optional[Dict[str, str]] = None,
    endnotes: Optional[Dict[str, str]] = None,
    header: Optional[str] = None,
) -> bytes:
    """A minimal DOCX whose <w:body> is `body_xml` (w: prefix bound to the chosen namespace).

    `footnotes`/`endnotes` map a note id to its text; each part also carries Word's
    separator (id -1) and continuationSeparator (id 0) notes, plus a separator-typed
    note under an ordinary id (7) — so a reader must filter by `w:type`, not by id.
    `header` adds a header part with that text.
    """
    ns = W_NS_STRICT if strict else W_NS
    rel_type = OFFICE_DOC_STRICT if strict else OFFICE_DOC
    document = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{ns}" xmlns:r="{R_NS}" '
        f'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        f'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        f'xmlns:v="urn:schemas-microsoft-com:vml">'
        f"<w:body>{body_xml}</w:body></w:document>"
    )
    members: List[Tuple[str, str]] = [
        ("[Content_Types].xml", _CT),
        ("_rels/.rels", _rels([("rId1", rel_type, main_part)])),
        (main_part, document),
    ]
    part_rels = []
    for kind, notes in (("footnote", footnotes), ("endnote", endnotes)):
        if notes is None:
            continue
        body = "".join(
            f'<w:{kind} w:type="{typ}" w:id="{nid}"><w:p><w:r><w:t>SEPARATOR</w:t></w:r></w:p></w:{kind}>'
            for typ, nid in (("separator", "-1"), ("continuationSeparator", "0"), ("separator", "7"))
        ) + "".join(
            f'<w:{kind} w:id="{nid}"><w:p><w:r><w:{kind}Ref/></w:r><w:r><w:t xml:space="preserve">{text}</w:t></w:r>'
            f"</w:p></w:{kind}>"
            for nid, text in notes.items()
        )
        members.append((f"word/{kind}s.xml", f'<w:{kind}s xmlns:w="{ns}">{body}</w:{kind}s>'))
        part_rels.append((f"r{kind}", f"{R_NS}/{kind}s", f"{kind}s.xml"))
    if header is not None:
        members.append(("word/header1.xml", f'<w:hdr xmlns:w="{ns}"><w:p><w:r><w:t>{header}</w:t></w:r></w:p></w:hdr>'))
        part_rels.append(("rhdr", f"{R_NS}/header", "header1.xml"))
    if part_rels:
        base, name = main_part.rsplit("/", 1)
        members.append((f"{base}/_rels/{name}.rels", _rels(part_rels)))
    return make_zip(members)


def w_note_ref(note_id: str, kind: str = "footnote") -> str:
    return f'<w:r><w:{kind}Reference w:id="{note_id}"/></w:r>'


# ── XLSX ──────────────────────────────────────────────────────────────────────


def xlsx_bytes(
    sheets: Sequence[Tuple[str, List[List[Optional[Union[str, int, float]]]]]],
    *,
    inline: bool = False,
    hidden: Sequence[str] = (),
) -> bytes:
    """A minimal XLSX: `sheets` = [(name, rows)], strings via sharedStrings (or inline);
    sheets named in `hidden` get `state="hidden"`."""
    shared: List[str] = []
    index: Dict[str, int] = {}

    def cell(ref: str, value) -> str:
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return f'<c r="{ref}"><v>{value}</v></c>'
        if inline:
            return f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>'
        if value not in index:
            index[value] = len(shared)
            shared.append(value)
        return f'<c r="{ref}" t="s"><v>{index[value]}</v></c>'

    members: List[Tuple[str, str]] = [
        ("[Content_Types].xml", _CT),
        ("_rels/.rels", _rels([("rId1", OFFICE_DOC, "xl/workbook.xml")])),
    ]
    sheet_entries, rel_entries = [], []
    for n, (name, rows) in enumerate(sheets, 1):
        row_xml = "".join(
            f'<row r="{r}">' + "".join(cell(f"{chr(64 + c)}{r}", v) for c, v in enumerate(row, 1)) + "</row>"
            for r, row in enumerate(rows, 1)
        )
        members.append(
            (f"xl/worksheets/sheet{n}.xml", f'<worksheet xmlns="{SS_NS}"><sheetData>{row_xml}</sheetData></worksheet>')
        )
        state = ' state="hidden"' if name in hidden else ""
        sheet_entries.append(f'<sheet name="{name}" sheetId="{n}"{state} r:id="rId{n}"/>')
        rel_entries.append((f"rId{n}", f"{R_NS}/worksheet", f"worksheets/sheet{n}.xml"))
    rel_entries.append(("rIdSS", f"{R_NS}/sharedStrings", "sharedStrings.xml"))
    members.append(
        (
            "xl/workbook.xml",
            f'<workbook xmlns="{SS_NS}" xmlns:r="{R_NS}"><sheets>{"".join(sheet_entries)}</sheets></workbook>',
        )
    )
    members.append(("xl/_rels/workbook.xml.rels", _rels(rel_entries)))
    sst = "".join(f"<si><t>{s}</t></si>" for s in shared)
    members.append(("xl/sharedStrings.xml", f'<sst xmlns="{SS_NS}" count="{len(shared)}">{sst}</sst>'))
    return make_zip(members)


# ── PPTX ──────────────────────────────────────────────────────────────────────


def pptx_bytes(
    slides: Sequence[List[str]], *, order: Optional[Sequence[int]] = None, hidden: Sequence[int] = ()
) -> bytes:
    """A minimal PPTX; `order` lists slide part numbers (1-based) in presentation order;
    slide parts in `hidden` get `show="0"`."""
    members: List[Tuple[str, str]] = [
        ("[Content_Types].xml", _CT),
        ("_rels/.rels", _rels([("rId1", OFFICE_DOC, "ppt/presentation.xml")])),
    ]
    rels = []
    for n, paragraphs in enumerate(slides, 1):
        paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in paragraphs)
        show = ' show="0"' if n in hidden else ""
        members.append(
            (
                f"ppt/slides/slide{n}.xml",
                f'<p:sld xmlns:p="{P_NS}" xmlns:a="{A_NS}"{show}>'
                f"<p:cSld><p:spTree><p:sp><p:txBody>{paras}"
                f"</p:txBody></p:sp></p:spTree></p:cSld></p:sld>",
            )
        )
        rels.append((f"rId{n}", f"{R_NS}/slide", f"slides/slide{n}.xml"))
    order = list(order) if order is not None else list(range(1, len(slides) + 1))
    ids = "".join(f'<p:sldId id="{255 + i}" r:id="rId{n}"/>' for i, n in enumerate(order, 1))
    members.append(
        (
            "ppt/presentation.xml",
            f'<p:presentation xmlns:p="{P_NS}" xmlns:r="{R_NS}"><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>',
        )
    )
    members.append(("ppt/_rels/presentation.xml.rels", _rels(rels)))
    return make_zip(members)


# ── ODF / EPUB ────────────────────────────────────────────────────────────────

ODF_NS = (
    'xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0" '
    'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
    'xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0" '
    'xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0" '
    'xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0" '
    'xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"'
)


def odf_bytes(kind: str, body_inner: str, *, automatic_styles: str = "", encrypted: bool = False) -> bytes:
    """ODT/ODS/ODP: kind in text|spreadsheet|presentation; `body_inner` goes inside it."""
    mimetype = {
        "text": "application/vnd.oasis.opendocument.text",
        "spreadsheet": "application/vnd.oasis.opendocument.spreadsheet",
        "presentation": "application/vnd.oasis.opendocument.presentation",
    }[kind]
    content = (
        f'<?xml version="1.0" encoding="UTF-8"?><office:document-content {ODF_NS}>'
        f"<office:automatic-styles>{automatic_styles}</office:automatic-styles>"
        f"<office:body><office:{kind}>{body_inner}</office:{kind}></office:body></office:document-content>"
    )
    manifest = (
        '<manifest:manifest xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0">'
        + (
            '<manifest:file-entry manifest:full-path="content.xml"><manifest:encryption-data/></manifest:file-entry>'
            if encrypted
            else ""
        )
        + "</manifest:manifest>"
    )
    return make_zip([("mimetype", mimetype), ("content.xml", content), ("META-INF/manifest.xml", manifest)])


def epub_bytes(
    chapters: Sequence[Tuple[str, str]], *, encrypt: Sequence[str] = (), missing_spine: Sequence[str] = ()
) -> bytes:
    """EPUB 3: chapters = [(member name, xhtml body inner)] in spine order. The
    manifest href is the member name %-escaped, as EPUB requires (a space → %20);
    `missing_spine` adds spine items whose member does not exist."""
    from urllib.parse import quote

    items = list(chapters) + [(h, None) for h in missing_spine]
    manifest = "".join(
        f'<item id="c{i}" href="{quote(h)}" media-type="application/xhtml+xml"/>' for i, (h, _b) in enumerate(items)
    )
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(items)))
    members: List[Tuple[str, str]] = [
        ("mimetype", "application/epub+zip"),
        (
            "META-INF/container.xml",
            '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles>'
            '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>',
        ),
        (
            "OEBPS/content.opf",
            f'<package xmlns="http://www.idpf.org/2007/opf" version="3.0"><manifest>{manifest}</manifest>'
            f"<spine>{spine}</spine></package>",
        ),
    ]
    for href, body in chapters:
        members.append(
            (
                f"OEBPS/{href}",
                '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE html><html xmlns="http://www.w3.org/1999/xhtml">'
                f"<head><title>t</title></head><body>{body}</body></html>",
            )
        )
    if encrypt:
        refs = "".join(
            f'<enc:EncryptedData xmlns:enc="http://www.w3.org/2001/04/xmlenc#"><enc:CipherData>'
            f'<enc:CipherReference URI="OEBPS/{h}"/></enc:CipherData></enc:EncryptedData>'
            for h in encrypt
        )
        members.append(
            (
                "META-INF/encryption.xml",
                f'<encryption xmlns="urn:oasis:names:tc:opendocument:xmlns:container">{refs}</encryption>',
            )
        )
    return make_zip(members)


# ── PDF (ported from atrium-llm-enrich, MIT) ──────────────────────────────────

PAGE_W, PAGE_H = 612, 792
FONT_CLEAN = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"


def _pdf_escape(raw: bytes) -> bytes:
    out = raw.replace(b"\\", b"\\\\")
    return out.replace(b"(", b"\\(").replace(b")", b"\\)")


def _text_block(
    x: int,
    y: int,
    lines: List[bytes],
    *,
    invisible: bool = False,
    leading: int = 14,
    matrix: Optional[Tuple[float, float, float, float]] = None,
) -> bytes:
    position = f"{matrix[0]} {matrix[1]} {matrix[2]} {matrix[3]} {x} {y} Tm" if matrix is not None else f"{x} {y} Td"
    parts = [b"BT", b"/F1 12 Tf", position.encode("ascii"), f"{leading} TL".encode("ascii")]
    if invisible:
        parts.append(b"3 Tr")
    for i, line in enumerate(lines):
        if i:
            parts.append(b"T*")
        parts.append(b"(" + _pdf_escape(line) + b") Tj")
    parts.append(b"ET")
    return b"\n".join(parts)


def _build_pdf(streams: List[bytes], font_obj: bytes = FONT_CLEAN) -> bytes:
    n_pages = len(streams)
    font_num = 3 + 2 * n_pages
    kid_nums = [3 + 2 * i for i in range(n_pages)]
    objects: Dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: b"<< /Type /Pages /Kids ["
        + b" ".join(f"{n} 0 R".encode("ascii") for n in kid_nums)
        + f"] /Count {n_pages} >>".encode("ascii"),
        font_num: font_obj,
    }
    for i, stream in enumerate(streams):
        page_num = 3 + 2 * i
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> /Contents {page_num + 1} 0 R >>".encode("ascii")
        )
        objects[page_num + 1] = f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: Dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("ascii") + objects[num] + b"\nendobj\n"
    xref_at = len(out)
    highest = max(objects)
    out += f"xref\n0 {highest + 1}\n".encode("ascii") + b"0000000000 65535 f \n"
    for num in range(1, highest + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode("ascii")
    out += f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode("ascii")
    return bytes(out)


def pdf_bytes(
    pages: Sequence[Sequence[str]],
    *,
    invisible: Union[bool, Sequence[bool]] = False,
    matrix: Optional[Tuple[float, float, float, float]] = None,
) -> bytes:
    """A PDF with one text block per page (an empty list = a page with no text).

    `invisible` (per page or for all) renders the text in mode 3 — how OCR engines
    lay their text layer under a scanned image. `matrix` (a, b, c, d) sets the text
    matrix: (-1, 0, 0, 1) mirrors, (0, 1, -1, 0) rotates by 90°. Text is encoded as
    cp1252 (WinAnsi).
    """
    flags = [invisible] * len(pages) if isinstance(invisible, bool) else list(invisible)
    streams = []
    for lines, inv in zip(pages, flags, strict=True):
        streams.append(
            # A transformed block starts mid-page so mirrored/rotated text stays on the page;
            # the default position is unchanged, which keeps the committed sample PDFs identical.
            _text_block(
                *((300, 400) if matrix is not None else (72, 720)),
                [ln.encode("cp1252") for ln in lines],
                invisible=inv,
                matrix=matrix,
            )
            if lines
            else b""
        )
    return _build_pdf(streams)


# ── compression wrappers and archives ─────────────────────────────────────────


def compress_bytes(data: Union[str, bytes], fmt: str = "gzip") -> bytes:
    """gzip (mtime 0, so reproducible), bzip2 or xz of `data`."""
    raw = data.encode("utf-8") if isinstance(data, str) else data
    if fmt == "gzip":
        return gzip.compress(raw, mtime=0)
    if fmt == "bz2":
        return bz2.compress(raw)
    if fmt == "xz":
        return lzma.compress(raw, format=lzma.FORMAT_XZ)
    raise ValueError(fmt)


def tar_bytes(members: Sequence[Tuple[str, Union[str, bytes]]]) -> bytes:
    """A reproducible ustar archive (fixed mtime, uid/gid 0)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name, data in members:
            raw = data.encode("utf-8") if isinstance(data, str) else data
            info = tarfile.TarInfo(name)
            info.size, info.mtime, info.uid, info.gid = len(raw), 0, 0, 0
            tf.addfile(info, io.BytesIO(raw))
    return buf.getvalue()


# ── OCR engine exports ────────────────────────────────────────────────────────

PAGE_NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
ABBYY_NS = "http://www.abbyy.com/FineReader_xml/FineReader10-schema-v1.xml"
TESSERACT_HEADER = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"


def page_xml(page_inner: str, *, image: str = "scan_0001.jpg") -> str:
    """A PAGE 2019 document with one Page whose content is `page_inner`."""
    return f'<PcGts xmlns="{PAGE_NS}"><Page imageFilename="{image}">{page_inner}</Page></PcGts>'


def tesseract_tsv(pages: Sequence[Sequence[Sequence[str]]]) -> str:
    """Tesseract's TSV: pages → lines → words (level 5 rows), with the level 1-4 rows
    Tesseract writes around them."""
    rows = [TESSERACT_HEADER]
    for p, lines in enumerate(pages, 1):
        rows.append(f"1\t{p}\t0\t0\t0\t0\t0\t0\t2480\t3508\t-1\t")
        rows.append(f"2\t{p}\t1\t0\t0\t0\t10\t10\t900\t400\t-1\t")
        rows.append(f"3\t{p}\t1\t1\t0\t0\t10\t10\t900\t400\t-1\t")
        for ln, words in enumerate(lines, 1):
            rows.append(f"4\t{p}\t1\t1\t{ln}\t0\t10\t{10 * ln}\t900\t40\t-1\t")
            for wn, word in enumerate(words, 1):
                rows.append(f"5\t{p}\t1\t1\t{ln}\t{wn}\t{10 * wn}\t{10 * ln}\t40\t30\t91.5\t{word}")
    return "\n".join(rows) + "\n"


def abbyy_xml(pages: Sequence[Sequence[str]], *, table: Optional[Sequence[Sequence[str]]] = None) -> str:
    """ABBYY FineReader 10 XML: one charParams per character (spaces included, the
    first letter of each word `wordStart="true"`); `table` rows go into a Table block
    on the first page."""

    def line(text: str) -> str:
        chars, start = [], True
        for ch in text:
            flag = ' wordStart="true"' if start and ch != " " else ' wordStart="false"'
            chars.append(f"<charParams{flag}>{ch}</charParams>")
            start = ch == " "
        return f'<line baseline="10"><formatting lang="Czech">{"".join(chars)}</formatting></line>'

    out = []
    for n, lines in enumerate(pages):
        body = f'<block blockType="Text"><text><par>{"".join(line(t) for t in lines)}</par></text></block>'
        if table is not None and n == 0:
            rows = "".join(
                "<row>" + "".join(f"<cell><text><par>{line(c)}</par></text></cell>" for c in r) + "</row>"
                for r in table
            )
            body += f'<block blockType="Table">{rows}</block>'
        out.append(f'<page width="2480" height="3508" resolution="300">{body}</page>')
    return f'<?xml version="1.0" encoding="UTF-8"?><document xmlns="{ABBYY_NS}" version="1.0">{"".join(out)}</document>'


def djvu_xml(pages: Sequence[Tuple[str, Sequence[str]]]) -> str:
    """DjVuXML (djvutoxml): pages = [(page file, lines)]."""
    objects = []
    for page_file, lines in pages:
        text = "".join("<LINE>" + "".join(f"<WORD>{w}</WORD>" for w in ln.split()) + "</LINE>" for ln in lines)
        objects.append(
            f'<OBJECT data="file://localhost/{page_file}" type="image/x.djvu" width="2480" height="3508">'
            f'<PARAM name="PAGE" value="{page_file}"/><HIDDENTEXT><PAGECOLUMN><REGION><PARAGRAPH>{text}'
            "</PARAGRAPH></REGION></PAGECOLUMN></HIDDENTEXT></OBJECT>"
        )
    return f'<?xml version="1.0"?><DjVuXML><HEAD/><BODY>{"".join(objects)}</BODY></DjVuXML>'


# ── e-mail ────────────────────────────────────────────────────────────────────


def eml_bytes(
    subject: str,
    plain: Optional[str] = None,
    html: Optional[str] = None,
    *,
    charset: str = "utf-8",
    attachments: int = 0,
) -> bytes:
    """An RFC 5322 message: Q-encoded subject, text/plain and/or text/html parts
    (quoted-printable), `attachments` base64 parts. Fixed boundary and ids."""
    import quopri
    from email.header import Header

    head = (
        "From: Eva Prochazkova <eva@example.org>\r\nTo: Jan Novotny <jan@example.org>\r\n"
        f"Subject: {Header(subject, 'utf-8').encode()}\r\nDate: Mon, 2 Sep 2024 10:00:00 +0200\r\n"
        "Message-ID: <sample-1@example.org>\r\nMIME-Version: 1.0\r\n"
    )
    parts = []
    for subtype, body in (("plain", plain), ("html", html)):
        if body is not None:
            encoded = quopri.encodestring(body.encode(charset)).decode("ascii")
            parts.append(
                f"Content-Type: text/{subtype}; charset={charset}\r\nContent-Transfer-Encoding: quoted-printable"
                f"\r\n\r\n{encoded}\r\n"
            )
    for n in range(attachments):
        parts.append(
            f'Content-Type: image/png\r\nContent-Disposition: attachment; filename="scan{n}.png"\r\n'
            "Content-Transfer-Encoding: base64\r\n\r\niVBORw0KGgo=\r\n"
        )
    if len(parts) == 1:
        return (head + parts[0]).encode("ascii")
    sub = "alternative" if not attachments else "mixed"
    boundary = "==atrium-boundary=="
    body = "".join(f"--{boundary}\r\n{p}" for p in parts) + f"--{boundary}--\r\n"
    return (head + f'Content-Type: multipart/{sub}; boundary="{boundary}"\r\n\r\n' + body).encode("ascii")


def mbox_bytes(messages: Sequence[bytes]) -> bytes:
    """An mboxo mailbox: a `From ` separator line per message, body `From ` lines quoted."""
    out = b""
    for n, msg in enumerate(messages, 1):
        quoted = b"\n".join(
            b">" + ln if ln.startswith(b"From ") else ln for ln in msg.replace(b"\r\n", b"\n").split(b"\n")
        )
        out += f"From sender{n}@example.org Mon Sep  2 10:00:00 2024\n".encode("ascii") + quoted + b"\n"
    return out


# ── sample set ────────────────────────────────────────────────────────────────


def write_samples(out_dir: str) -> List[str]:
    """Write the binary members of data_samples/TEXT/ into `out_dir`.

    Same fictional world as the rest of data_samples/ (the invented site "Hradiště u
    Horní Mezí", researchers Jan Novotný and Eva Procházková). The PDFs use a WinAnsi
    base font, which has no Czech carons, so their text is German/English — as the
    archive's own foreign-language reports are.
    """
    os.makedirs(out_dir, exist_ok=True)
    samples = {
        "CTX000000010.docx": docx_bytes(
            w_p(w_t("Zpráva o záchranném výzkumu — Hradiště u Horní Mezí"))
            + w_p(w_t("Výzkum vedla Eva Procházková v září 2024."))
            + w_p(w_t("Sonda III zachytila zásypový horizont valu."), '<w:r><w:br w:type="page"/></w:r>')
            + w_p(w_t("Tab. 1: Přehled vrstev"))
            + "<w:tbl><w:tr><w:tc>"
            + w_p(w_t("Vrstva 101"))
            + "</w:tc><w:tc>"
            + w_p(w_t("hnědá hlinitá, mocnost 30 cm"))
            + "</w:tc></w:tr><w:tr><w:tc>"
            + w_p(w_t("Vrstva 102"))
            + "</w:tc><w:tc>"
            + w_p(w_t("šedá jílovitá s uhlíky"))
            + "</w:tc></w:tr></w:tbl>"
        ),
        # Born-digital PDF: visible text on pages 1-2, page 3 has no text layer at all.
        "CTX000000011.pdf": pdf_bytes(
            [
                ["Grabungsbericht Hradiste u Horni Mezi", "Schnitt II wurde im Juni 2024 angelegt."],
                ["Funde: Keramikscherben und Tierknochen.", "Bearbeitet von Jan Novotny."],
                [],
            ]
        ),
        # Scanned PDF with an invisible OCR text layer (render mode 3), OCR noise included.
        "CTX000000012.pdf": pdf_bytes(
            [["Excavation diary, trench II, 12 June 2024.", "Th1s l1ne has OCR n0ise: ~~ |||| 3"]],
            invisible=True,
        ),
        "CTX000000013.xlsx": xlsx_bytes(
            [
                ("Nálezy", [["Č.", "Popis"], [1, "zlomek okraje nádoby"], [2, "zvířecí kost, žebro"]]),
                ("Vrstvy", [["Vrstva", "Popis"], [101, "hnědá hlinitá"], [102, "šedá jílovitá s uhlíky"]]),
            ]
        ),
        "CTX000000014.odt": odf_bytes(
            "text",
            "<text:h>Terénní deník — sonda III</text:h><text:p>Den první: skrývka ornice, Jan Novotný.</text:p>"
            '<text:p text:style-name="PB">Den druhý: dokumentace západního profilu.</text:p>',
            automatic_styles='<style:style style:name="PB" style:family="paragraph">'
            '<style:paragraph-properties fo:break-before="page"/></style:style>',
        ),
        # (#31 Phase 4) A gzip-wrapped page transcript, and a ZIP bundle of two per-page
        # PAGE XML files with the METS and doc metadata a Transkribus export carries.
        "CTX000000023.txt.gz": compress_bytes(
            "Terénní deník, sonda IV — den třetí.\nZačištění profilu, odběr vzorků na uhlíky.\n"
            "Zapsala Eva Procházková.\n"
        ),
        "CTX000000024.zip": make_zip(
            [
                ("CTX000000024/mets.xml", '<mets xmlns="http://www.loc.gov/METS/"/>'),
                ("CTX000000024/doc.xml", "<trpDocMetadata><title>CTX000000024</title></trpDocMetadata>"),
                (
                    "CTX000000024/page/0001.xml",
                    page_xml(
                        '<TextRegion id="r1"><TextLine id="l1"><TextEquiv><Unicode>Nálezová zpráva, strana 1.'
                        '</Unicode></TextEquiv></TextLine><TextLine id="l2"><TextEquiv><Unicode>Sonda V, vrstva'
                        " 201.</Unicode></TextEquiv></TextLine></TextRegion>",
                        image="0001.jpg",
                    ),
                ),
                (
                    "CTX000000024/page/0002.xml",
                    page_xml(
                        '<TextRegion id="r1"><TextLine id="l1"><TextEquiv><Unicode>Strana 2: keramika, kosti.'
                        "</Unicode></TextEquiv></TextLine></TextRegion>",
                        image="0002.jpg",
                    ),
                ),
            ]
        ),
    }
    written = []
    for name, data in samples.items():
        path = os.path.join(out_dir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        written.append(path)
    return written
