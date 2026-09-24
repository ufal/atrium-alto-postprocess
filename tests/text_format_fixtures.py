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
an `invisible` switch (text render mode 3 — the OCR-layer construction) and pages with
no text at all.
"""

from __future__ import annotations

import io
import os
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


def docx_bytes(body_xml: str, *, main_part: str = "word/document.xml", strict: bool = False) -> bytes:
    """A minimal DOCX whose <w:body> is `body_xml` (w: prefix bound to the chosen namespace)."""
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
    return make_zip(
        [
            ("[Content_Types].xml", _CT),
            ("_rels/.rels", _rels([("rId1", rel_type, main_part)])),
            (main_part, document),
        ]
    )


# ── XLSX ──────────────────────────────────────────────────────────────────────


def xlsx_bytes(
    sheets: Sequence[Tuple[str, List[List[Optional[Union[str, int, float]]]]]], *, inline: bool = False
) -> bytes:
    """A minimal XLSX: `sheets` = [(name, rows)], strings via sharedStrings (or inline)."""
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
        sheet_entries.append(f'<sheet name="{name}" sheetId="{n}" r:id="rId{n}"/>')
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


def pptx_bytes(slides: Sequence[List[str]], *, order: Optional[Sequence[int]] = None) -> bytes:
    """A minimal PPTX; `order` lists slide part numbers (1-based) in presentation order."""
    members: List[Tuple[str, str]] = [
        ("[Content_Types].xml", _CT),
        ("_rels/.rels", _rels([("rId1", OFFICE_DOC, "ppt/presentation.xml")])),
    ]
    rels = []
    for n, paragraphs in enumerate(slides, 1):
        paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in paragraphs)
        members.append(
            (
                f"ppt/slides/slide{n}.xml",
                f'<p:sld xmlns:p="{P_NS}" xmlns:a="{A_NS}"><p:cSld><p:spTree><p:sp><p:txBody>{paras}'
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


def epub_bytes(chapters: Sequence[Tuple[str, str]], *, encrypt: Sequence[str] = ()) -> bytes:
    """EPUB 3: chapters = [(href, xhtml body inner)] in spine order."""
    manifest = "".join(
        f'<item id="c{i}" href="{h}" media-type="application/xhtml+xml"/>' for i, (h, _b) in enumerate(chapters)
    )
    spine = "".join(f'<itemref idref="c{i}"/>' for i in range(len(chapters)))
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


def _text_block(x: int, y: int, lines: List[bytes], *, invisible: bool = False, leading: int = 14) -> bytes:
    parts = [b"BT", b"/F1 12 Tf", f"{x} {y} Td".encode("ascii"), f"{leading} TL".encode("ascii")]
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


def pdf_bytes(pages: Sequence[Sequence[str]], *, invisible: Union[bool, Sequence[bool]] = False) -> bytes:
    """A PDF with one text block per page (an empty list = a page with no text).

    `invisible` (per page or for all) renders the text in mode 3 — how OCR engines
    lay their text layer under a scanned image. Text is encoded as cp1252 (WinAnsi).
    """
    flags = [invisible] * len(pages) if isinstance(invisible, bool) else list(invisible)
    streams = []
    for lines, inv in zip(pages, flags, strict=True):
        streams.append(_text_block(72, 720, [ln.encode("cp1252") for ln in lines], invisible=inv) if lines else b"")
    return _build_pdf(streams)


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
    }
    written = []
    for name, data in samples.items():
        path = os.path.join(out_dir, name)
        with open(path, "wb") as fh:
            fh.write(data)
        written.append(path)
    return written
