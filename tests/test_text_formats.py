"""
tests/test_text_formats.py — the text-lines readers (#31): detection, decoding,
normalization, per-format page/line rules, and every refusal path.

Fast lane: no models, no network, no data_samples/. Binary inputs are built in memory
by tests/text_format_fixtures.py; PDF tests need pypdfium2 (setup/requirements-test.txt).
"""

import io
import json
import zipfile

import pytest

import text_formats as tf
from tests.text_format_fixtures import (
    docx_bytes,
    epub_bytes,
    make_zip,
    odf_bytes,
    pdf_bytes,
    pptx_bytes,
    w_p,
    w_t,
    xlsx_bytes,
)


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return str(path)


def _read(tmp_path, name, data, **kw):
    return tf.read_document(_write(tmp_path, name, data), **kw)


def _lines(doc):
    return [p.lines for p in doc.pages]


def _code(tmp_path, name, data, **kw):
    with pytest.raises(tf.IngestError) as info:
        tf.read_document(_write(tmp_path, name, data), **kw)
    return info.value.code


# ── detection: content decides ────────────────────────────────────────────────


def test_misnamed_pdf_is_read_as_pdf_and_noted(tmp_path):
    doc = _read(tmp_path, "notes.txt", pdf_bytes([["Hidden pdf"]]))
    assert doc.kind == "pdf"
    assert "extension .txt but content is pdf" in doc.notes


def test_docx_under_a_foreign_extension_is_sniffed_by_zip_members(tmp_path):
    assert _read(tmp_path, "report.dat", docx_bytes(w_p(w_t("Ahoj")))).kind == "docx"


@pytest.mark.parametrize(
    "name,data,code",
    [
        ("empty.txt", b"", "empty_file"),
        ("old.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 600, "legacy_office_unsupported"),
        ("scan.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image_needs_ocr"),
        ("scan.tif", b"II*\x00" + b"\x00" * 64, "image_needs_ocr"),
        ("pack.gz", b"\x1f\x8b\x08\x00" + b"\x00" * 32, "archive_unsupported"),
        ("pack.zip", None, "archive_unsupported"),
        ("blob.bin", bytes(range(256)) * 8, "binary_content"),
    ],
)
def test_non_text_inputs_are_refused_with_a_reason(tmp_path, name, data, code):
    if data is None:
        data = make_zip([("readme.txt", "x")])
    assert _code(tmp_path, name, data) == code


def test_unsupported_odf_type_is_refused(tmp_path):
    data = make_zip([("mimetype", "application/vnd.oasis.opendocument.graphics")])
    assert _code(tmp_path, "d.odg", data) == "archive_unsupported"


def test_unknown_extension_brace_file_is_json_or_jsonl(tmp_path):
    assert _read(tmp_path, "x.dat", '{"text": "a"}').kind == "json"
    assert _read(tmp_path, "y.dat", '{"text": "a"}\n{"text": "b"}\n').kind == "jsonl"


# ── decoding ──────────────────────────────────────────────────────────────────

_CZECH = "Příliš žluťoučký kůň úpěl ďábelské ódy.\nDruhý řádek textu.\n"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "utf-16-le"])
def test_unicode_encodings_with_and_without_bom(tmp_path, encoding):
    doc = _read(tmp_path, "a.txt", _CZECH.encode(encoding))
    assert _lines(doc) == [["Příliš žluťoučký kůň úpěl ďábelské ódy.", "Druhý řádek textu."]]


def test_legacy_czech_code_page_is_detected_not_mangled(tmp_path):
    doc = _read(tmp_path, "a.txt", _CZECH.encode("cp1250"))
    assert doc.encoding in ("cp1250", "iso8859_2", "windows-1250")
    assert doc.pages[0].lines[0] == "Příliš žluťoučký kůň úpěl ďábelské ódy."


@pytest.fixture
def no_detector(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _no_detector(name, *a, **k):
        if name == "charset_normalizer":
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_detector)


def test_decode_without_detector_falls_back_in_order(no_detector):
    text, enc, flags = tf.decode_bytes("čeština".encode("cp1250"), ("cp1250", "cp1252"))
    assert (text, enc) == ("čeština", "cp1250")
    assert "charset_normalizer_missing" in flags


def test_decode_with_no_fallbacks_and_no_detector_fails_closed(no_detector):
    """An empty fallback list means "let the detector choose any encoding"; with no
    detector installed either, nothing is left to try."""
    with pytest.raises(tf.IngestError) as info:
        tf.decode_bytes("čeština".encode("cp1250"), ())
    assert info.value.code == "decode_failed"


# ── normalization: the line invariant classify_TEXT relies on ────────────────


def test_every_line_separator_is_normalized_so_readlines_round_trips():
    raw = "a\r\nb\rc\x0bd\x1ce\x85f g h"
    lines = tf.shape_lines([raw])
    written = "\n".join(lines) + "\n"
    assert io.StringIO(written).readlines() == [ln + "\n" for ln in lines]
    assert lines == list("abcdefgh")


def test_normalize_line_rules():
    assert tf.normalize_line("é") == "é"  # NFC
    assert tf.normalize_line("ﬁnal") == "final"  # ligature expanded
    assert tf.normalize_line("zá­kon") == "zákon"  # soft hyphen mid-word removed
    assert tf.normalize_line("zá­") == "zá-"  # …and becomes "-" at the end of a line
    assert tf.normalize_line("wor\x02") == "wor-"  # PDFium's end-of-line hyphen marker
    assert tf.normalize_line("a​b﻿‮") == "ab"  # zero-width / BOM / bidi removed
    assert tf.normalize_line("a b") == "a b"  # NBSP
    assert tf.normalize_line("a\x07b\tc  ") == "ab\tc"  # control removed, tab kept, stripped
    assert tf.normalize_line("zero‍width joiner") == "zero‍width joiner"  # ZWJ kept


def test_wrap_line_and_shape_lines():
    assert tf.wrap_line("aaa bbb ccc", 7) == ["aaa bbb", "ccc"]
    assert tf.wrap_line("x" * 25, 10) == ["x" * 10, "x" * 10, "x" * 5]  # unbroken run hard-split
    assert tf.wrap_line("short", 0) == ["short"]
    assert tf.shape_lines(["one", "", "  ", "two"]) == ["one", "two"]
    assert tf.shape_lines(["one", "", "two"], keep_blank=True) == ["one", "", "two"]
    assert all(len(x) <= 1000 for x in tf.shape_lines(["word " * 3000]))


# ── plain-text family ─────────────────────────────────────────────────────────


def test_plain_text_pages_at_form_feeds_and_keeps_blank_separators(tmp_path):
    doc = _read(tmp_path, "a.txt", "one\n\ntwo\n\fthree\n\f")
    assert _lines(doc) == [["one", "", "two"], ["three"]]  # trailing \f opens no page
    assert doc.kind == "txt" and tf.default_source_origin(doc) == "ocr:generic"


def test_markdown_markup_is_stripped(tmp_path):
    md = (
        "---\ntitle: x\n---\n# Nadpis #\n\nText s **tučným**, _kurzívou_ a [odkazem](http://x).\n"
        "```python\nprint('code')\n```\n> citace\n- položka\n1. první\n* * *\n| a | b |\n|---|---|\n| c | d |\n"
        "Setext\n======\n[ref]: http://example.org\n![obrázek](img.png) snake_case_name\n"
    )
    lines = [ln for ln in _read(tmp_path, "a.md", md).pages[0].lines if ln]
    assert lines == [
        "Nadpis",
        "Text s tučným, kurzívou a odkazem.",
        "citace",
        "položka",
        "první",
        "a\tb",
        "c\td",
        "Setext",
        "obrázek snake_case_name",
    ]


def test_csv_with_text_and_page_columns_round_trips_a_line_table(tmp_path):
    doc = _read(tmp_path, "t.csv", "file,page_num,line_num,text\nX,2,1,Druhá\nX,1,1,První\nX,1,2,Další\n")
    assert [(p.label, p.lines) for p in doc.pages] == [("2", ["Druhá"]), ("1", ["První", "Další"])]


def test_csv_without_a_text_column_joins_cells(tmp_path):
    doc = _read(tmp_path, "t.csv", "a;b;c\n1;;x y\n")
    assert _lines(doc) == [["a\tb\tc", "1\tx y"]]


def test_tsv_and_embedded_newlines_and_nuls(tmp_path):
    doc = _read(tmp_path, "t.tsv", 'text\tpage\n"multi\nline"\t1\nplain\x00text\t1\n')
    assert _lines(doc) == [["multi", "line", "plaintext"]]


# ── JSON family ───────────────────────────────────────────────────────────────


def test_json_family_a_pages_without_header_duplication(tmp_path):
    data = {
        "analyzeResult": {
            "content": "FULL DOCUMENT TEXT",
            "pages": [
                {"pageNumber": 1, "lines": [{"content": "L1"}], "words": [{"content": "W1"}]},
                {"pageNumber": 2, "lines": [{"content": "L2"}]},
            ],
        }
    }
    doc = _read(tmp_path, "azure.json", json.dumps(data))
    assert [(p.label, p.lines) for p in doc.pages] == [("1", ["L1"]), ("2", ["L2"])]


def test_json_family_b_groups_a_page_tagged_list(tmp_path):
    data = {"Blocks": [{"Page": 1, "Text": "a"}, {"Page": 2, "Text": "b"}, {"Page": 1, "Text": "c"}]}
    doc = _read(tmp_path, "textract.json", json.dumps(data))
    assert [(p.label, p.lines) for p in doc.pages] == [("1", ["a", "c"]), ("2", ["b"])]


def test_json_family_c_top_level_children_become_blocks(tmp_path):
    doc = _read(tmp_path, "list.json", json.dumps([{"text": "one"}, {"meta": 1}, {"content": "three"}]))
    assert [(p.label, p.lines) for p in doc.pages] == [("1", ["one"]), ("3", ["three"])]
    doc = _read(tmp_path, "d.json", json.dumps({"header": {"text": "H"}, "body": {"lines": ["B1", "B2"]}}))
    assert [(p.label, p.lines) for p in doc.pages] == [("header", ["H"]), ("body", ["B1", "B2"])]


def test_json_single_page_and_all_strings_fallback(tmp_path):
    assert _lines(_read(tmp_path, "s.json", json.dumps({"page": {"lines": [{"text": "x"}]}}))) == [["x"]]
    doc = _read(tmp_path, "n.json", json.dumps({"foo": {"bar": "Some prose"}, "id": "123"}))
    assert _lines(doc) == [["Some prose"]] and "json_all_strings" in doc.notes


def test_json_pages_match_the_json_keys_split_for_header_free_documents(tmp_path):
    """Parity with json-keys (page_split + extract_JSON_2_TXT) where no header text exists."""
    import page_split
    from extract_JSON_2_TXT import process_json_to_txt

    data = {"pages": [{"page_number": 1, "lines": [{"text": "a"}, {"text": "b"}]}, {"page_number": 2, "text": "c"}]}
    src = tmp_path / "in" / "P.json"
    src.parent.mkdir()
    src.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "out"
    assert page_split.split_json_document(str(src), str(out)) == 2
    expected = []
    for n in (1, 2):
        txt = tmp_path / f"p{n}.txt"
        process_json_to_txt(out / "P" / f"P-{n}.json", txt)
        expected.append(txt.read_text(encoding="utf-8").split("\n"))
    assert _lines(tf.read_document(str(src))) == expected


def test_json_that_is_really_jsonl_and_broken_json(tmp_path):
    doc = _read(tmp_path, "l.json", '{"text":"a"}\n{"text":"b"}\n')
    assert doc.kind == "jsonl" and _lines(doc) == [["a"], ["b"]]
    assert _code(tmp_path, "bad.json", '{"a": [1, 2') == "malformed"
    assert _code(tmp_path, "deep.json", "[" * 100000 + "]" * 100000) == "malformed"


def test_jsonl_skips_and_counts_bad_records(tmp_path):
    doc = _read(tmp_path, "r.jsonl", '{"text": "ok"}\nnot json\n"bare string"\n')
    assert _lines(doc) == [["ok"], ["bare string"]]
    assert "jsonl_bad_records=1" in doc.notes


# ── XML / HTML family ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ns", ["http://www.loc.gov/standards/alto/ns-v2#", "http://www.loc.gov/standards/alto/ns-v4#", ""]
)
def test_alto_any_version(tmp_path, ns):
    xmlns = f' xmlns="{ns}"' if ns else ""
    xml = (
        f'<alto{xmlns}><Layout><Page ID="p1" PHYSICAL_IMG_NR="5"><PrintSpace><TextBlock>'
        '<TextLine><String CONTENT="Hello"/><SP/><String CONTENT="wor"/><HYP CONTENT="¬"/></TextLine>'
        '<TextLine><String CONTENT="ld"/></TextLine></TextBlock></PrintSpace></Page></Layout></alto>'
    )
    doc = _read(tmp_path, "a.xml", xml)
    assert doc.kind == "alto" and [(p.label, p.lines) for p in doc.pages] == [("5", ["Hello wor-", "ld"])]
    assert tf.default_source_origin(doc) == "ABBYY-ALTO"


def test_page_xml_follows_reading_order_and_falls_back_to_words(tmp_path):
    xml = (
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"><Page imageFilename="s/img7.jpg">'
        '<ReadingOrder><OrderedGroup id="g"><RegionRefIndexed index="1" regionRef="r1"/>'
        '<RegionRefIndexed index="0" regionRef="r2"/></OrderedGroup></ReadingOrder>'
        '<TextRegion id="r1"><TextLine id="l1"><TextEquiv index="1"><Unicode>alt</Unicode></TextEquiv>'
        '<TextEquiv index="0"><Unicode>second region</Unicode></TextEquiv></TextLine></TextRegion>'
        '<TextRegion id="r2"><TextLine id="l2"><Word><TextEquiv><Unicode>first</Unicode></TextEquiv></Word>'
        "<Word><TextEquiv><Unicode>region</Unicode></TextEquiv></Word></TextLine></TextRegion></Page></PcGts>"
    )
    doc = _read(tmp_path, "p.xml", xml)
    assert doc.kind == "page-xml" and [(p.label, p.lines) for p in doc.pages] == [
        ("img7", ["first region", "second region"])
    ]
    assert tf.default_source_origin(doc) == "ocr:page-xml"


def test_tei_pages_at_pb_lines_at_lb_header_skipped(tmp_path):
    xml = (
        '<TEI xmlns="http://www.tei-c.org/ns/1.0"><teiHeader><fileDesc>HEADER</fileDesc></teiHeader>'
        '<text><body><pb n="1"/><p>First <lb/>second\n   line</p><pb n="2"/><p>Page two</p></body></text></TEI>'
    )
    assert _lines(_read(tmp_path, "t.xml", xml)) == [["First", "second line"], ["Page two"]]


def test_teitok_pages_keep_their_labels_and_empty_pages(tmp_path):
    """Every <pb> is a page (a blank one too), labelled with pb@n -- the pages
    atrium-nlp-enrich's TEITOK readers count, so a table made from a TEITOK file lines up with
    its layout. Page labels used to be renumbered and a blank page dropped."""
    xml = (
        '<TEI><text><body><pb n="I"/><p>Titul</p><pb n="II"/><pb n="1"/><p>Text</p>'
        "<pb/><p>Konec</p></body></text></TEI>"
    )
    doc = _read(tmp_path, "t.teitok.xml", xml)
    assert [(p.label, p.lines) for p in doc.pages] == [
        ("I", ["Titul"]),
        ("II", []),
        ("1", ["Text"]),
        ("4", ["Konec"]),
    ]


def test_tokenized_teitok_lines_are_its_lb_lines(tmp_path):
    """nlp-enrich's format 2: sentences inline, <lb/> for physical lines, a sentence running
    over a page break with its <pb/> inside. <s> is not a line there; <lb/> and <pb/> are."""
    xml = (
        '<TEI xmlnsoff="http://www.tei-c.org/ns/1.0"><text><body><pb n="1" id="pb-1"/>'
        '<div type="TextBlock" id="b-1.1"><s id="s-1" text="Jedna. Dvě tři">'
        '<lb id="lb-1.1"/><tok id="w-1" join="right">Jedna</tok><tok id="w-2">.</tok></s> '
        '<s id="s-2"><tok id="w-3">Dvě</tok>\n<lb id="lb-1.2"/><tok id="w-4">tři</tok>\n'
        '<pb n="2" id="pb-2"/><lb id="lb-2.1"/><name id="n-1" type="LOC"><tok id="w-5">Praha</tok></name>'
        "</s></div></body></text></TEI>"
    )
    doc = _read(tmp_path, "CTX.teitok.xml", xml)
    assert [(p.label, p.lines) for p in doc.pages] == [
        ("1", ["Jedna. Dvě", "tři"]),
        ("2", ["Praha"]),
    ]


def test_legacy_teitok_name_close_quirk_is_repaired_not_recovered(tmp_path):
    xml = '<TEI><text><body><pb n="1"/><p>V <name type="LOC">Praze</n> a dál</p></body></text></TEI>'
    doc = _read(tmp_path, "old.teitok.xml", xml)
    assert doc.pages[0].lines == ["V Praze a dál"]
    assert "name_close_repaired" in doc.notes and "xml_recovered" not in doc.notes


def test_generic_xml_blocks_and_mixed_content(tmp_path):
    xml = "<records><record><title>R1</title><body>Body <i>one</i> text</body></record><record><title>R2</title></record></records>"
    doc = _read(tmp_path, "g.xml", xml)
    assert doc.kind == "xml" and [(p.label, p.lines) for p in doc.pages] == [
        ("record[1]", ["R1", "Body one text"]),
        ("record[2]", ["R2"]),
    ]


def test_xml_entity_declarations_are_refused_and_broken_xml_recovered(tmp_path):
    bomb = '<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;">]><r>&b;</r>'
    assert _code(tmp_path, "e.xml", bomb) == "xml_entity_declaration"
    doc = _read(tmp_path, "broken.xml", "<doc><p>kept text</p><p>unclosed</doc>")
    assert "xml_recovered" in doc.notes and "kept text" in doc.pages[0].lines


def test_xml_in_a_legacy_encoding_without_declaration(tmp_path):
    doc = _read(tmp_path, "c.xml", "<doc><p>Věta číslo jedna</p></doc>".encode("cp1250"))
    assert doc.pages[0].lines == ["Věta číslo jedna"]


def test_html_blocks_breaks_css_pages_and_dropped_elements(tmp_path):
    html = (
        "<html><head><title>T</title><style>p{}</style></head><body><h1>Head</h1><script>x=1</script>"
        '<p>Para one<br>line two</p><div style="page-break-before: always">Page two</div></body></html>'
    )
    doc = _read(tmp_path, "h.html", html)
    assert doc.kind == "html" and _lines(doc) == [["Head", "Para one", "line two"], ["Page two"]]
    assert tf.default_source_origin(doc) == "digital-born-html"


def test_hocr_pages_and_lines(tmp_path):
    html = (
        '<html><head><meta name="ocr-system" content="tesseract"/></head><body>'
        '<div class="ocr_page" title="bbox 0 0 10 10; ppageno 0"><span class="ocr_line">Line <b>A</b></span>'
        '<span class="ocr_line">Line B</span></div>'
        '<div class="ocr_page" title="ppageno 1"><span class="ocrx_line">Page 2</span></div></body></html>'
    )
    doc = _read(tmp_path, "s.hocr", html)
    assert doc.kind == "hocr" and [(p.label, p.lines) for p in doc.pages] == [
        ("1", ["Line A", "Line B"]),
        ("2", ["Page 2"]),
    ]
    assert tf.default_source_origin(doc) == "ocr:hocr"


def test_rtf_code_page_unicode_pages_and_skipped_destinations(tmp_path):
    rtf = (
        rb"{\rtf1\ansi\ansicpg1250{\fonttbl{\f0 Arial;}}{\header HEADER}{\*\generator X;}\f0 "
        rb"P\'f8\'edli\'9a \u382?lu\'9d\par Druh\'fd\tab \'f8\'e1dek\page Strana 2\par}"
    )
    doc = _read(tmp_path, "r.rtf", rtf)
    assert doc.kind == "rtf" and _lines(doc) == [["Příliš žluť", "Druhý řádek"], ["Strana 2"]]


# ── ZIP containers ────────────────────────────────────────────────────────────


def test_docx_page_breaks_textboxes_tracked_changes(tmp_path):
    body = (
        w_p(w_t("P1"))
        + w_p(w_t("P2"), '<w:r><w:br w:type="page"/></w:r>')
        + w_p("<w:r><w:lastRenderedPageBreak/><w:t>P3</w:t></w:r>")  # right after an explicit break: deduped
        + w_p(w_t("P4a"), "<w:r><w:lastRenderedPageBreak/><w:t>P4b</w:t></w:r>")  # mid-paragraph rendered break
        + w_p("<w:del><w:r><w:delText>GONE</w:delText></w:r></w:del>", w_t("kept"))
        + w_p(
            w_t("anchor"),
            '<w:r><mc:AlternateContent><mc:Choice Requires="wps"><w:drawing><wps:txbx><w:txbxContent>'
            + w_p(w_t("BOX"))
            + "</w:txbxContent></wps:txbx></w:drawing></mc:Choice><mc:Fallback><w:pict><v:textbox><w:txbxContent>"
            + w_p(w_t("DUPLICATE"))
            + "</w:txbxContent></v:textbox></w:pict></mc:Fallback></mc:AlternateContent></w:r>",
            w_t(" tail"),
        )
        + w_p(w_t("a"), "<w:r><w:br/><w:t>b</w:t><w:tab/><w:t>c</w:t></w:r>")
        + w_p(w_t("continuous"), ppr='<w:sectPr><w:type w:val="continuous"/></w:sectPr>')
        + w_p(w_t("next page section"), ppr="<w:sectPr/>")
        + w_p(w_t("before"), ppr='<w:pageBreakBefore w:val="0"/>')
        + w_p(w_t("forced"), ppr="<w:pageBreakBefore/>")
        + "<w:tbl><w:tr><w:tc>"
        + w_p(w_t("c1"))
        + "</w:tc><w:tc>"
        + w_p(w_t("c2"))
        + "</w:tc></w:tr></w:tbl>"
        + "<w:sectPr/>"
    )
    doc = _read(tmp_path, "d.docx", docx_bytes(body))
    assert _lines(doc) == [
        ["P1", "P2"],
        ["P3", "P4a"],
        ["P4b", "kept", "anchor tail", "BOX", "a", "b c", "continuous", "next page section"],
        ["before"],
        ["forced", "c1", "c2"],
    ]
    assert tf.default_source_origin(doc) == "digital-born-docx"


@pytest.mark.parametrize("mode,pages", [("explicit", 3), ("none", 1)])
def test_docx_page_break_modes(tmp_path, mode, pages):
    body = (
        w_p(w_t("A"), '<w:r><w:br w:type="page"/></w:r>')
        + w_p(w_t("B"), "<w:r><w:lastRenderedPageBreak/><w:t>C</w:t></w:r>")
        + w_p(w_t("D"), ppr="<w:pageBreakBefore/>")
    )
    doc = _read(tmp_path, "m.docx", docx_bytes(body), options=tf.ReaderOptions(page_breaks=mode))
    assert len(doc.pages) == pages


def test_docx_strict_namespace_and_main_part_from_rels(tmp_path):
    doc = _read(tmp_path, "s.docx", docx_bytes(w_p(w_t("strict ok")), strict=True, main_part="word/doc2.xml"))
    assert _lines(doc) == [["strict ok"]]


def test_xlsx_sheets_in_workbook_order_text_cells_only(tmp_path):
    data = xlsx_bytes(
        [("Nálezy", [["Číslo", "Popis"], [1, "Zlomek"], [None, None]]), ("Prázdný", [[1, 2]]), ("B", [["x"]])]
    )
    doc = _read(tmp_path, "w.xlsx", data)
    assert [(p.label, p.lines) for p in doc.pages] == [("Nálezy", ["Číslo\tPopis", "Zlomek"]), ("B", ["x"])]
    assert _lines(_read(tmp_path, "i.xlsx", xlsx_bytes([("S", [["inline"]])], inline=True))) == [["inline"]]


def test_pptx_slides_in_presentation_order(tmp_path):
    doc = _read(tmp_path, "p.pptx", pptx_bytes([["one A", "one B"], ["two"]], order=[2, 1]))
    assert _lines(doc) == [["two"], ["one A", "one B"]]


def test_odt_explicit_and_soft_page_breaks(tmp_path):
    styles = '<style:style style:name="PB" style:family="paragraph"><style:paragraph-properties fo:break-before="page"/></style:style>'
    body = (
        '<text:h>Deník</text:h><text:p>Den <text:s text:c="2"/>první<text:line-break/>zlom</text:p>'
        '<text:p text:style-name="PB">Den druhý<text:note><text:note-body><text:p>FOOTNOTE</text:p></text:note-body></text:note></text:p>'
        "<text:soft-page-break/><text:p>Po měkkém zlomu</text:p>"
    )
    doc = _read(tmp_path, "t.odt", odf_bytes("text", body, automatic_styles=styles))
    assert _lines(doc) == [["Deník", "Den první", "zlom"], ["Den druhý"], ["Po měkkém zlomu"]]


def test_ods_repeats_are_capped_and_empty_repeats_never_expanded(tmp_path):
    body = (
        '<table:table table:name="S1"><table:table-row><table:table-cell><text:p>A1</text:p></table:table-cell>'
        '<table:table-cell table:number-columns-repeated="16384"/><table:table-cell><text:p>Z1</text:p></table:table-cell>'
        '</table:table-row><table:table-row table:number-rows-repeated="1048576"><table:table-cell/></table:table-row>'
        '<table:table-row table:number-rows-repeated="5000"><table:table-cell><text:p>rep</text:p></table:table-cell></table:table-row>'
        "</table:table>"
    )
    doc = _read(tmp_path, "s.ods", odf_bytes("spreadsheet", body))
    assert doc.pages[0].label == "S1" and doc.pages[0].lines[0] == "A1\tZ1"
    assert len(doc.pages[0].lines) == 1 + 100  # the text row repeated up to the cap


def test_odp_slides(tmp_path):
    body = '<draw:page draw:name="s1"><text:p>Slide one</text:p></draw:page><draw:page draw:name="s2"><text:p>Two</text:p></draw:page>'
    assert [(p.label, p.lines) for p in _read(tmp_path, "p.odp", odf_bytes("presentation", body)).pages] == [
        ("s1", ["Slide one"]),
        ("s2", ["Two"]),
    ]


def test_epub_spine_chapters(tmp_path):
    doc = _read(
        tmp_path, "b.epub", epub_bytes([("c1.xhtml", "<h1>Ch 1</h1><p>Text&#160;one</p>"), ("c2.xhtml", "<p>Ch 2</p>")])
    )
    assert [(p.label, p.lines) for p in doc.pages] == [("c1", ["Ch 1", "Text one"]), ("c2", ["Ch 2"])]


@pytest.mark.parametrize(
    "name,builder",
    [
        ("enc.odt", lambda: odf_bytes("text", "<text:p>x</text:p>", encrypted=True)),
        ("drm.epub", lambda: epub_bytes([("c1.xhtml", "<p>x</p>")], encrypt=["c1.xhtml"])),
    ],
)
def test_encrypted_containers(tmp_path, name, builder):
    assert _code(tmp_path, name, builder()) == "encrypted"


def test_zip_limits(tmp_path):
    small = tf.Limits(zip_max_members=3, zip_max_total_mb=0.01, zip_max_member_mb=0.005, zip_max_ratio=5)
    many = make_zip([(f"f{i}", "x") for i in range(5)])
    assert _code(tmp_path, "many.docx", many, limits=small) == "zip_limits_exceeded"
    big = docx_bytes(w_p(w_t("x" * 20000)))
    assert _code(tmp_path, "big.docx", big, limits=small) == "zip_limits_exceeded"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", "x")
        zf.writestr("word/document.xml", b"\x00" * (4 * 1024 * 1024))
    assert _code(tmp_path, "bomb.docx", buf.getvalue(), limits=tf.Limits(zip_max_ratio=50)) == "zip_limits_exceeded"


def test_encrypted_zip_member_flag(tmp_path):
    data = bytearray(docx_bytes(w_p(w_t("x"))))
    # Set the "encrypted" general-purpose flag bit on every central-directory entry.
    pos = data.find(b"PK\x01\x02")
    while pos != -1:
        data[pos + 8] |= 0x01
        pos = data.find(b"PK\x01\x02", pos + 4)
    assert _code(tmp_path, "e.docx", bytes(data)) == "encrypted"


def test_corrupt_zip_is_corrupt(tmp_path):
    assert _code(tmp_path, "c.docx", b"PK\x03\x04" + b"garbage" * 20) == "corrupt"


# ── PDF (pypdfium2) ───────────────────────────────────────────────────────────


def test_pdf_pages_labels_and_text_layer_classes(tmp_path):
    pytest.importorskip("pypdfium2")
    doc = _read(tmp_path, "d.pdf", pdf_bytes([["Page one.", "Second line."], ["Page two."], []]))
    assert [(p.label, p.lines, p.text_layer) for p in doc.pages] == [
        ("1", ["Page one.", "Second line."], "digital"),
        ("2", ["Page two."], "digital"),
        ("3", [], "none"),
    ]
    assert doc.pages[2].needs_ocr_reason == "no extractable text layer"
    assert tf.default_source_origin(doc) == "digital-born-pdf"


def test_pdf_invisible_text_is_an_ocr_layer(tmp_path):
    pytest.importorskip("pypdfium2")
    doc = _read(tmp_path, "o.pdf", pdf_bytes([["Scanned text."], ["More."]], invisible=True))
    assert [p.text_layer for p in doc.pages] == ["ocr", "ocr"]
    assert tf.default_source_origin(doc) == "ocr:pdf-text-layer"


def test_pdf_corrupt_and_encrypted(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    import pypdfium2

    assert _code(tmp_path, "c.pdf", b"%PDF-1.4\n not really a pdf") == "corrupt"

    def _locked(*_a, **_k):
        raise pypdfium2.PdfiumError("Failed to load document (PDFium: Incorrect password error).", err_code=4)

    monkeypatch.setattr(pypdfium2, "PdfDocument", _locked)
    assert _code(tmp_path, "l.pdf", pdf_bytes([["x"]])) == "encrypted"


def test_classify_text_layer_thresholds():
    opts = tf.ReaderOptions()
    assert tf.classify_text_layer("  a ", 1, 0, opts) == ("none", "no extractable text layer")
    garbled = "abc" + "�" * 2 + ""
    assert tf.classify_text_layer(garbled, 5, 0, opts)[0] == "garbled"
    assert tf.classify_text_layer("normal text here", 4, 2, opts) == ("ocr", None)
    assert tf.classify_text_layer("normal text here", 4, 1, opts) == ("digital", None)


def test_isolated_reader_runs_pdf_out_of_process_and_times_out(tmp_path, monkeypatch):
    pytest.importorskip("pypdfium2")
    path = _write(tmp_path, "d.pdf", pdf_bytes([["Isolated."]]))
    assert tf.read_document_isolated(path).pages[0].lines == ["Isolated."]

    import subprocess

    def _hang(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(tf.subprocess, "run", _hang)
    with pytest.raises(tf.IngestError) as info:
        tf.read_document_isolated(path)
    assert info.value.code == "timeout"


def test_isolated_reader_reports_a_crashed_child(tmp_path, monkeypatch):
    path = _write(tmp_path, "d.pdf", b"%PDF-1.4\n")

    class _Dead:
        stdout, stderr, returncode = "", "Segmentation fault", -11

    monkeypatch.setattr(tf.subprocess, "run", lambda *a, **k: _Dead())
    with pytest.raises(tf.IngestError) as info:
        tf.read_document_isolated(path)
    assert info.value.code == "reader_crashed"


def test_non_pdf_kinds_are_read_in_process(tmp_path, monkeypatch):
    monkeypatch.setattr(tf.subprocess, "run", lambda *a, **k: pytest.fail("no subprocess for text"))
    assert tf.read_document_isolated(_write(tmp_path, "a.txt", "x\n")).pages[0].lines == ["x"]


# ── limits, settings, registry ────────────────────────────────────────────────


def test_file_size_and_line_limits(tmp_path):
    assert _code(tmp_path, "a.txt", "x" * 2048, limits=tf.Limits(max_file_mb=0.001)) == "too_large"
    doc = _read(tmp_path, "b.txt", "\n".join(["l"] * 7), limits=tf.Limits(max_lines_per_page=3))
    assert [len(p.lines) for p in doc.pages] == [3, 3, 1]  # a block overflows onto continuation pages
    assert [p.label for p in doc.pages] == ["1", "1+1", "1+2"]
    assert _code(tmp_path, "c.txt", "a\fb\fc", limits=tf.Limits(max_pages=2)) == "too_large"


def test_load_settings_reads_and_validates_the_config_section():
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read_dict(
        {
            "TEXT_INGEST": {
                "MAX_FILE_MB": "10",
                "PAGE_BREAKS": "explicit",
                "FALLBACK_ENCODINGS": "cp1252, latin-1",
                "KEEP_BLANK_LINES": "yes",
            }
        }
    )
    limits, opts = tf.load_settings(cfg)
    assert limits.max_file_mb == 10.0 and opts.page_breaks == "explicit"
    assert opts.fallback_encodings == ("cp1252", "latin-1") and opts.keep_blank_lines is True
    for key, bad in (
        ("MAX_PAGES", "lots"),
        ("MAX_PAGES", "0"),
        ("PAGE_BREAKS", "sometimes"),
        ("FALLBACK_ENCODINGS", "no-such-codec"),
        ("KEEP_BLANK_LINES", "maybe"),
    ):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"TEXT_INGEST": {key: bad}})
        with pytest.raises(ValueError, match=key):
            tf.load_settings(cfg)
    assert tf.load_settings(None) == (tf.Limits(), tf.ReaderOptions())


def test_registry_is_consistent():
    # Every registered kind has exactly one reader entry point.
    assert set(tf.READERS) == set(tf._TEXT_READERS) | set(tf._PATH_READERS) | set(tf._BYTES_READERS)
    for kind, spec in tf.READERS.items():
        assert spec.kind == kind and spec.extensions and spec.media_type
    assert ".pdf" in tf.supported_extensions() and ".docx" in tf.supported_extensions()


def test_document_serialization_round_trip(tmp_path):
    doc = _read(tmp_path, "a.txt", "x\fy\n")
    assert tf.TextDocument.from_dict(json.loads(json.dumps(doc.to_dict()))) == doc
