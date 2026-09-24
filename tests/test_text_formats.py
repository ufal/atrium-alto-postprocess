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
    abbyy_xml,
    compress_bytes,
    djvu_xml,
    docx_bytes,
    eml_bytes,
    epub_bytes,
    make_zip,
    mbox_bytes,
    odf_bytes,
    page_xml,
    pdf_bytes,
    pptx_bytes,
    tar_bytes,
    tesseract_tsv,
    w_note_ref,
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
        ("pack.tar.gz", "tar", "archive_unsupported"),
        ("pack.zip", None, "archive_unsupported"),
        ("nested.zip", "nested", "archive_unsupported"),
        ("blob.bin", bytes(range(256)) * 8, "binary_content"),
    ],
)
def test_non_text_inputs_are_refused_with_a_reason(tmp_path, name, data, code):
    if data is None:
        data = make_zip([("readme.txt", "x")])  # a README is metadata, not a page file
    elif data == "tar":
        data = compress_bytes(tar_bytes([("a.txt", "x")]))
    elif data == "nested":
        data = make_zip([("inner.zip", make_zip([("a.txt", "x")]))])
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
    # (#31 Phase 4) NOTES = page (default): the footnote ends the page that cites it.
    assert _lines(doc) == [["Deník", "Den první", "zlom"], ["Den druhý", "FOOTNOTE"], ["Po měkkém zlomu"]]
    skipped = _read(tmp_path, "t.odt", odf_bytes("text", body, automatic_styles=styles),
                    options=tf.ReaderOptions(notes_placement="skip"))  # fmt: skip
    assert _lines(skipped)[1] == ["Den druhý"] and "notes_not_read=1" in skipped.notes


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


# ── (#31 Phase 4) detection hardening ─────────────────────────────────────────


def test_pdf_marker_inside_a_text_file_is_text(tmp_path):
    doc = _read(tmp_path, "note.txt", "Header line\nA PDF starts with %PDF-1.4 and then objects.\n")
    assert doc.kind == "txt" and doc.pages[0].lines[1].startswith("A PDF starts")


def test_a_pdf_stored_in_a_zip_does_not_make_the_zip_a_pdf(tmp_path):
    # The member is STORED, so "%PDF-" sits in the ZIP's first KiB.
    data = make_zip([("1.txt", "Strana jedna"), ("2.pdf", pdf_bytes([["x"]]))])
    doc = _read(tmp_path, "b.zip", data)
    assert doc.kind == "zip-bundle" and _lines(doc) == [["Strana jedna"]]
    assert "bundle_member_failed:2.pdf:pdf_not_read_in_bundle" in doc.notes


@pytest.mark.parametrize(
    "name,content,kind",
    [
        ("readme.md", '<p align="center">Logo</p>\n\n# Nadpis\n', "md"),
        ("links.md", "[odkaz](http://x) na začátku\n", "md"),
        ("t.csv", "<b>text</b>\nřádek\n", "csv"),
        ("frag.html", "<div><p>Odstavec</p></div>", "html"),
        ("page.txt", '<alto><Layout><Page><TextLine><String CONTENT="A"/></TextLine></Page></Layout></alto>', "alto"),
    ],
)
def test_markup_leading_dialect_files_keep_their_extension(tmp_path, name, content, kind):
    assert _read(tmp_path, name, content).kind == kind


def test_a_txt_that_merely_starts_with_a_bracket_is_plain_text(tmp_path):
    doc = _read(tmp_path, "k.txt", "<Kapitola 1> Úvod\nDruhý řádek & konec\n")
    assert doc.kind == "txt" and _lines(doc) == [["<Kapitola 1> Úvod", "Druhý řádek & konec"]]
    assert "xml_parse_failed_read_as_text" in doc.notes


def test_an_unreadable_path_is_unreadable(tmp_path):
    (tmp_path / "adir").mkdir()
    with pytest.raises(tf.IngestError) as info:
        tf.read_document(str(tmp_path / "adir"))
    assert info.value.code == "unreadable"


@pytest.mark.parametrize(
    "data",
    [tar_bytes([("a.txt", "x")]), b"7z\xbc\xaf\x27\x1c" + b"\x00" * 64, b"Rar!\x1a\x07\x00" + b"\x00" * 64,
     b"\x28\xb5\x2f\xfd" + b"\x00" * 64],
    ids=["tar", "7z", "rar", "zstd"],
)  # fmt: skip
def test_archives_without_a_reader_are_refused(tmp_path, data):
    assert _code(tmp_path, "a.bin", data) == "archive_unsupported"


# ── compression wrappers ──────────────────────────────────────────────────────


@pytest.mark.parametrize("fmt,suffix", [("gzip", ".gz"), ("bz2", ".bz2"), ("xz", ".xz")])
def test_compressed_files_are_read_by_their_inner_name(tmp_path, fmt, suffix):
    doc = _read(tmp_path, "t.csv" + suffix, compress_bytes("text,page\nPrvní,1\nDruhá,2\n", fmt))
    assert doc.kind == "csv" and [(p.label, p.lines) for p in doc.pages] == [("1", ["První"]), ("2", ["Druhá"])]
    assert f"decompressed:{fmt}" in doc.notes


def test_compressed_alto_stays_alto_and_is_read_in_process(tmp_path, monkeypatch):
    monkeypatch.setattr(tf.subprocess, "run", lambda *a, **k: pytest.fail("no subprocess for text"))
    alto = (
        '<alto><Layout><Page PHYSICAL_IMG_NR="7"><TextLine><String CONTENT="Ahoj"/></TextLine></Page></Layout></alto>'
    )
    doc = tf.read_document_isolated(_write(tmp_path, "d.alto.xml.gz", compress_bytes(alto)))
    assert doc.kind == "alto" and [(p.label, p.lines) for p in doc.pages] == [("7", ["Ahoj"])]


@pytest.mark.parametrize(
    "name,inner",
    [("a.pdf.gz", "pdf"), ("a.docx.gz", "docx"), ("a.tgz", "tar"), ("a.txt.gz.gz", "gz")],
)
def test_compressed_containers_are_refused(tmp_path, name, inner):
    payload = {
        "pdf": pdf_bytes([["x"]]),
        "docx": docx_bytes(w_p(w_t("x"))),
        "tar": tar_bytes([("a.txt", "x")]),
        "gz": compress_bytes("x"),
    }[inner]
    assert _code(tmp_path, name, compress_bytes(payload)) == "archive_unsupported"


def test_damaged_compressed_streams_are_corrupt(tmp_path):
    assert _code(tmp_path, "t.txt.gz", compress_bytes("x" * 5000)[:-12]) == "corrupt"


def test_decompression_is_bounded(tmp_path):
    bomb = compress_bytes(b"a" * (3 * 1024 * 1024))
    assert _code(tmp_path, "bomb.txt.gz", bomb) == "zip_limits_exceeded"  # ratio > ZIP_MAX_RATIO
    lim = tf.Limits(max_file_mb=0.01, zip_max_ratio=10000)
    assert _code(tmp_path, "big.txt.gz", bomb, limits=lim) == "too_large"


# ── ZIP bundles of page files ─────────────────────────────────────────────────


def _page(text, image):
    return page_xml(f'<TextRegion id="r"><TextLine id="l"><TextEquiv><Unicode>{text}</Unicode></TextEquiv>'
                    "</TextLine></TextRegion>", image=image)  # fmt: skip


def test_zip_bundle_reads_page_files_in_natural_order_and_ignores_metadata(tmp_path):
    alto_dup = '<alto><Layout><Page><TextLine><String CONTENT="dup"/></TextLine></Page></Layout></alto>'
    data = make_zip(
        [
            ("doc/page/10.xml", _page("deset", "10.jpg")),
            ("doc/page/2.xml", _page("dva", "2.jpg")),
            ("doc/page/1.xml", _page("jedna", "1.jpg")),
            ("doc/alto/1.xml", alto_dup),  # the same page in a second format: counted, not read
            ("doc/mets.xml", '<mets xmlns="http://www.loc.gov/METS/"/>'),
            ("doc/doc.xml", "<trpDocMetadata/>"),
            ("__MACOSX/doc/._1.xml", "x"),
            ("doc/images/1.jpg", b"\xff\xd8\xff"),
            ("doc/README.txt", "export notes"),
        ]
    )
    doc = _read(tmp_path, "export.zip", data)
    assert doc.kind == "zip-bundle" and doc.native_pages is True
    # labels are member paths relative to the members' common folder, extension dropped
    assert [(p.label, p.lines) for p in doc.pages] == [("1", ["jedna"]), ("2", ["dva"]), ("10", ["deset"])]
    assert tf.default_source_origin(doc) == "ocr:page-xml"
    assert "bundle_duplicates_skipped=1" in doc.notes and not tf.is_lossy(doc)


def test_zip_bundle_of_mixed_kinds_is_generic_text_and_not_native(tmp_path):
    data = make_zip([("1.txt", "Strana jedna\fStrana dvě"), ("2.xml", _page("Tři", "3.jpg"))])
    doc = _read(tmp_path, "mix.zip", data)
    assert [(p.label, p.lines) for p in doc.pages] == [("1/1", ["Strana jedna"]), ("1/2", ["Strana dvě"]),
                                                      ("2", ["Tři"])]  # fmt: skip
    assert tf.default_source_origin(doc) == "ocr:generic" and doc.native_pages is False


def test_zip_bundle_member_failures_are_skipped_noted_and_lossy(tmp_path):
    data = make_zip([("1.txt", "Dobrá strana"), ("2.txt", bytes(range(256)) * 8), ("3.json", "{broken")])
    doc = _read(tmp_path, "b.zip", data)
    assert _lines(doc) == [["Dobrá strana"]]
    assert "zip_members_skipped=2" in doc.notes and tf.lossy_reasons(doc) == ["zip_members_skipped"]


def test_zip_bundles_that_cannot_be_read(tmp_path):
    assert _code(tmp_path, "p.zip", make_zip([("1.pdf", pdf_bytes([["x"]]))])) == "archive_unsupported"
    assert _code(tmp_path, "i.zip", make_zip([("1.jpg", b"\xff\xd8\xff"), ("2.png", b"x")])) == "image_needs_ocr"
    assert _code(tmp_path, "d.zip", make_zip([("word/document.xml", "<x/>")])) == "archive_unsupported"
    assert _code(tmp_path, "b.zip", make_zip([("1.txt", bytes(range(256)) * 8)])) == "binary_content"


# ── CSV/TSV and OCR tables ────────────────────────────────────────────────────


def test_tesseract_tsv_words_become_lines_on_their_pages(tmp_path):
    tsv = tesseract_tsv([[["Nálezová", "zpráva"], ['"Sonda', "V"]], [], [["Strana", "3"]]])
    doc = _read(tmp_path, "scan.tsv", tsv)
    assert doc.kind == "tesseract-tsv" and doc.native_pages
    assert [(p.label, p.lines) for p in doc.pages] == [("1", ["Nálezová zpráva", '"Sonda V']), ("2", []),
                                                      ("3", ["Strana 3"])]  # fmt: skip
    assert tf.default_source_origin(doc) == "ocr:tesseract"


def test_csv_word_rows_sharing_a_line_address_are_joined(tmp_path):
    doc = _read(tmp_path, "w.csv", "page,line,word\n1,1,Dobrý\n1,1,den\n1,2,Nashle\n2,1,Konec\n")
    assert [(p.label, p.lines) for p in doc.pages] == [("1", ["Dobrý den", "Nashle"]), ("2", ["Konec"])]
    assert "csv_rows_joined=1" in doc.notes


def test_a_numeric_line_column_is_never_the_text_column(tmp_path):
    doc = _read(tmp_path, "l.csv", "page,line,content\n1,1,První řádek\n1,2,Druhý řádek\n")
    assert _lines(doc) == [["První řádek", "Druhý řádek"]] and "csv text column 'content'" in doc.notes


def test_a_stray_quote_does_not_swallow_the_rest_of_a_csv(tmp_path):
    doc = _read(tmp_path, "q.csv", 'text,page\n"Uvozovka,1\nDalší,2\n')
    assert [(p.label, p.lines) for p in doc.pages] == [("1", ['"Uvozovka']), ("2", ["Další"])]
    assert tf.lossy_reasons(doc) == ["csv_unbalanced_quote"]


def test_a_quote_in_a_tsv_is_literal(tmp_path):
    doc = _read(tmp_path, "q.tsv", 'text\tpage\n"Uvozovka\t1\nDalší\t2\n')
    assert [(p.label, p.lines) for p in doc.pages] == [("1", ['"Uvozovka']), ("2", ["Další"])]
    assert not tf.is_lossy(doc)


def test_csv_field_limit_is_raised_once_and_not_restored(tmp_path):
    import csv

    _read(tmp_path, "a.csv", "text\nx\n")
    raised = csv.field_size_limit()
    _read(tmp_path, "b.csv", "text\ny\n")
    assert csv.field_size_limit() == raised >= 2**31 - 1


# ── JSON: one granularity, no binary ──────────────────────────────────────────


def test_json_line_objects_with_words_yield_the_line_once(tmp_path):
    azure_read = {"readResults": [{"page": 1, "lines": [
        {"text": "Dobrý den", "words": [{"text": "Dobrý"}, {"text": "den"}]},
        {"text": "Druhý řádek", "words": [{"text": "Druhý"}, {"text": "řádek"}]},
    ]}]}  # fmt: skip
    doc = _read(tmp_path, "a.json", json.dumps(azure_read))
    assert _lines(doc) == [["Dobrý den", "Druhý řádek"]] and "json_word_leaves_skipped=4" in doc.notes


def test_json_coarse_text_above_lines_is_skipped(tmp_path):
    data = {"text": "Dobrý den Druhý řádek", "blocks": [{"lines": [{"text": "Dobrý den"}, {"text": "Druhý řádek"}]}]}
    doc = _read(tmp_path, "c.json", json.dumps(data))
    assert _lines(doc) == [["Dobrý den", "Druhý řádek"]] and "json_coarse_leaves_skipped=1" in doc.notes


def test_textract_line_and_word_blocks_keep_the_lines(tmp_path):
    blocks = [
        {"BlockType": "PAGE", "Id": "p"},
        {"BlockType": "LINE", "Text": "Dobrý den", "Id": "l1"},
        {"BlockType": "WORD", "Text": "Dobrý", "Id": "w1"},
        {"BlockType": "WORD", "Text": "den", "Id": "w2"},
    ]
    doc = _read(tmp_path, "t.json", json.dumps({"DocumentMetadata": {"Pages": 1}, "Blocks": blocks}))
    assert _lines(doc) == [["Dobrý den"]]


def test_line_objects_without_their_own_text_are_one_line_of_words(tmp_path):
    doctr = {"pages": [{"blocks": [{"lines": [{"words": [{"value": "Dobrý"}, {"value": "den"}]}]}]}]}
    doc = _read(tmp_path, "d.json", json.dumps(doctr))
    assert _lines(doc) == [["Dobrý den"]] and "json_all_strings" in doc.notes


def test_a_json_array_of_strings_is_one_page(tmp_path):
    doc = _read(tmp_path, "s.json", json.dumps(["první řádek", "druhý řádek", "třetí"]))
    assert _lines(doc) == [["první řádek", "druhý řádek", "třetí"]] and "json_string_array" in doc.notes


def test_json_blobs_are_skipped_and_counted(tmp_path):
    b64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODk="
    data = {"pages": [{"text": "Skutečný text", "data": b64}, {"content": "data:image/png;base64,iVBORw0KGgo="}]}
    doc = _read(tmp_path, "b.json", json.dumps(data))
    assert _lines(doc) == [["Skutečný text"], []] and "json_blobs_skipped=2" in doc.notes


def test_json_text_keys_match_the_json_keys_extractor():
    extract = pytest.importorskip("extract_JSON_2_TXT")
    assert tf._JSON_TEXT_KEYS == frozenset(extract.TARGET_KEYS)


def test_reading_json_does_not_import_pandas(tmp_path):
    import subprocess
    import sys

    path = _write(tmp_path, "a.json", json.dumps({"lines": ["Ahoj"]}))
    code = (
        "import sys, text_formats as tf; d = tf.read_document(sys.argv[1]); "
        "assert d.pages[0].lines == ['Ahoj'], d; "
        "assert 'pandas' not in sys.modules and 'extract_JSON_2_TXT' not in sys.modules, sorted(sys.modules)"
    )
    root = str(__import__("pathlib").Path(tf.__file__).resolve().parent)
    proc = subprocess.run([sys.executable, "-c", code, path], cwd=root, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


# ── XML OCR dialects ──────────────────────────────────────────────────────────


def test_page_xml_table_cells_and_nested_regions_are_read_in_order(tmp_path):
    def cell(row, col, text):
        return (f'<TableCell id="c{row}{col}" row="{row}" col="{col}"><TextLine id="t{row}{col}"><TextEquiv>'
                f"<Unicode>{text}</Unicode></TextEquiv></TextLine></TableCell>")  # fmt: skip

    inner = (
        '<ReadingOrder><OrderedGroup id="g"><RegionRefIndexed index="0" regionRef="r1"/>'
        '<RegionRefIndexed index="1" regionRef="t"/></OrderedGroup></ReadingOrder>'
        '<TextRegion id="r1"><TextLine id="l1"><TextEquiv><Unicode>Nadpis</Unicode></TextEquiv></TextLine>'
        '<TextRegion id="r1a"><TextLine id="l2"><TextEquiv><Unicode>vnořený</Unicode></TextEquiv></TextLine>'
        "</TextRegion></TextRegion>"
        '<TableRegion id="t">'
        + cell(1, 0, "B1")
        + cell(0, 1, "A2")
        + cell(0, 0, "A1")
        + cell(1, 1, "B2")
        + "</TableRegion>"
    )
    doc = _read(tmp_path, "t.xml", page_xml(inner))
    assert _lines(doc) == [["Nadpis", "vnořený", "A1", "A2", "B1", "B2"]]


def test_abbyy_finereader_xml_pages_lines_words_and_tables(tmp_path):
    doc = _read(tmp_path, "f.xml", abbyy_xml([["Zpráva o sondě", "druhý řádek"], ["Strana dvě"]],
                                             table=[["Vrstva", "Popis"]]))  # fmt: skip
    assert doc.kind == "abbyy-xml" and doc.native_pages
    assert _lines(doc) == [["Zpráva o sondě", "druhý řádek", "Vrstva", "Popis"], ["Strana dvě"]]
    assert tf.default_source_origin(doc) == "ocr:abbyy-finereader"


def test_abbyy_restores_a_missing_space_at_word_start_and_its_hyphen():
    from lxml import etree

    line = etree.fromstring(
        '<line><formatting><charParams wordStart="true">d</charParams><charParams>o</charParams>'
        '<charParams wordStart="true">v</charParams><charParams>ý</charParams><charParams>¬</charParams>'
        "</formatting></line>"
    )
    assert tf._abbyy_line_text(line) == "do vý-"


def test_djvu_xml_objects_lines_and_words(tmp_path):
    doc = _read(tmp_path, "d.xml", djvu_xml([("p0001.djvu", ["Dobrý den", "Druhý"]), ("p0002.djvu", ["Dva"])]))
    assert doc.kind == "djvu-xml"
    assert [(p.label, p.lines) for p in doc.pages] == [("p0001", ["Dobrý den", "Druhý"]), ("p0002", ["Dva"])]


def test_generic_xml_line_elements_join_their_words_or_characters(tmp_path):
    xml = "<doc><line><c>a</c><c>b</c><c> </c><c>c</c></line><line><w>slovo</w><w>dvě</w></line></doc>"
    assert _lines(_read(tmp_path, "g.xml", xml)) == [["ab c", "slovo dvě"]]


def test_tei_p4_corpora_and_choice(tmp_path):
    p4 = (
        "<TEI.2><teiHeader><fileDesc>HLAVIČKA</fileDesc></teiHeader><text><body><p>Starý "
        "<choice><sic>teh</sic><corr>the</corr></choice> <choice><abbr>Dr</abbr><expan>Doktor</expan></choice> "
        "<abbr>atd.</abbr></p></body></text></TEI.2>"
    )
    assert _lines(_read(tmp_path, "p4.xml", p4)) == [["Starý the Doktor atd."]]
    corpus = (
        "<teiCorpus><teiHeader/><TEI><teiHeader/><text><body><p>Dokument jedna</p></body></text></TEI>"
        '<TEI><text><body><pb n="9"/><p>Dokument dvě</p></body></text></TEI></teiCorpus>'
    )
    assert [(p.label, p.lines) for p in _read(tmp_path, "c.xml", corpus).pages] == [
        ("1", ["Dokument jedna"]),
        ("9", ["Dokument dvě"]),
    ]


# ── office documents ──────────────────────────────────────────────────────────


def _docx_with_notes():
    body = (
        w_p(w_t("Odstavec jedna"), w_note_ref("1"), w_t(" pokračuje"))
        + w_p('<w:r><w:br w:type="page"/></w:r>')
        + w_p(w_t("Strana dvě"), w_note_ref("2", "endnote"), w_note_ref("7"))
    )
    return docx_bytes(body, footnotes={"1": "Poznámka pod čarou"}, endnotes={"2": "Vysvětlivka"}, header="Záhlaví")


def test_docx_footnotes_end_their_page_and_endnotes_the_document(tmp_path):
    doc = _read(tmp_path, "n.docx", _docx_with_notes())
    assert _lines(doc) == [["Odstavec jedna pokračuje", "Poznámka pod čarou"], ["Strana dvě", "Vysvětlivka"]]
    assert "SEPARATOR" not in str(_lines(doc))  # separator notes filtered by w:type, id 7 included
    assert "docx_footnotes=1" in doc.notes and "headers_footers_not_read=1" in doc.notes


@pytest.mark.parametrize(
    "mode,lines,note",
    [
        ("end", [["Odstavec jedna pokračuje"], ["Strana dvě", "Poznámka pod čarou", "Vysvětlivka"]], "docx_endnotes=1"),
        ("skip", [["Odstavec jedna pokračuje"], ["Strana dvě"]], "notes_not_read=2"),
    ],
)
def test_docx_notes_modes(tmp_path, mode, lines, note):
    doc = _read(tmp_path, "n.docx", _docx_with_notes(), options=tf.ReaderOptions(notes_placement=mode))
    assert _lines(doc) == lines and note in doc.notes


def test_odt_endnotes_go_to_the_end(tmp_path):
    body = (
        '<text:p>Jedna<text:note text:note-class="endnote"><text:note-body><text:p>KONEC</text:p>'
        "</text:note-body></text:note></text:p><text:p>Dva</text:p>"
    )
    doc = _read(tmp_path, "e.odt", odf_bytes("text", body))
    assert _lines(doc) == [["Jedna", "Dva", "KONEC"]] and "odt_endnotes=1" in doc.notes


def test_a_sheet_over_the_line_cap_continues_on_plus_pages(tmp_path):
    data = xlsx_bytes([("Nálezy", [[f"nález {i}"] for i in range(7)])])
    doc = _read(tmp_path, "big.xlsx", data, limits=tf.Limits(max_lines_per_page=3))
    assert [(p.label, len(p.lines)) for p in doc.pages] == [("Nálezy", 3), ("Nálezy+1", 3), ("Nálezy+2", 1)]
    body = "<table:table table:name='S'>" + "".join(
        f"<table:table-row><table:table-cell><text:p>r{i}</text:p></table:table-cell></table:table-row>" for i in range(5)
    ) + "</table:table>"  # fmt: skip
    ods = _read(tmp_path, "big.ods", odf_bytes("spreadsheet", body), limits=tf.Limits(max_lines_per_page=2))
    assert [p.label for p in ods.pages] == ["S", "S+1", "S+2"]


def test_ods_reads_string_cells_without_comments_and_notes_capped_repeats(tmp_path):
    body = (
        '<table:table table:name="T"><table:table-row>'
        '<table:table-cell office:value-type="float" office:value="3"><text:p>3</text:p></table:table-cell>'
        '<table:table-cell office:value-type="string"><text:p>popis</text:p>'
        "<office:annotation><text:p>KOMENTÁŘ</text:p></office:annotation></table:table-cell>"
        '</table:table-row><table:table-row table:number-rows-repeated="500"><table:table-cell office:value-type="string">'
        "<text:p>opak</text:p></table:table-cell></table:table-row></table:table>"
    )
    doc = _read(tmp_path, "s.ods", odf_bytes("spreadsheet", body))
    assert doc.pages[0].lines[0] == "popis" and len(doc.pages[0].lines) == 1 + 100
    assert tf.lossy_reasons(doc) == ["sheet_repeat_capped"]


def test_hidden_sheets_and_slides_are_read_and_counted(tmp_path):
    xl = _read(tmp_path, "h.xlsx", xlsx_bytes([("A", [["viditelný"]]), ("B", [["skrytý"]])], hidden=["B"]))
    assert _lines(xl) == [["viditelný"], ["skrytý"]] and "xlsx_hidden_sheets=1" in xl.notes
    pp = _read(tmp_path, "h.pptx", pptx_bytes([["jedna"], ["dvě"]], hidden=[2]))
    assert _lines(pp) == [["jedna"], ["dvě"]] and "pptx_hidden_slides=1" in pp.notes


def test_epub_escaped_hrefs_and_missing_spine_items(tmp_path):
    doc = _read(tmp_path, "b.epub", epub_bytes([("Kapitola 1.xhtml", "<p>Obsah</p>")], missing_spine=["Chybí.xhtml"]))
    assert [(p.label, p.lines) for p in doc.pages] == [("Kapitola 1", ["Obsah"])]
    assert tf.lossy_reasons(doc) == ["epub_spine_skipped"]


# ── RTF, normalisation, report flags ──────────────────────────────────────────


def test_rtf_font_code_pages_surrogate_pairs_and_a_leading_bom(tmp_path):
    rtf = (
        b"\xef\xbb\xbf{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0\\fcharset238 Arial;}{\\f1\\fcharset204 Times;}"
        b"{\\f2\\fcharset0 Courier;}}\\f0 \\'e8\\'ed\\'9a\\par \\f1 \\'c0\\'e1\\par \\f2 caf\\'e9\\par "
        b"\\u55357?\\u56832? smajl\\par osamocen\\u55357?\\'fd\\par}"
    )
    doc = _read(tmp_path, "f.rtf", rtf)
    assert _lines(doc) == [["číš", "Аб", "café", "\U0001f600 smajl", "osamocený"]]
    assert tf.lossy_reasons(doc) == ["lone_surrogates_dropped"]


def test_normalize_line_drops_lone_surrogates():
    assert tf.normalize_line("a\ud83db\udc00c") == "abc"


def test_mojibake_page_flag_is_conservative(tmp_path):
    bad = _read(tmp_path, "m.txt", "Zpráva o sondì èíslo 3. Nalezeny høeby, vrstva ornice mìla\n")
    assert bad.pages[0].flags == ["mojibake_cp1252"] and "mojibake_cp1252_pages=1" in bad.notes
    assert not tf.is_lossy(bad)  # a report flag, never a category or a loss
    good = _read(tmp_path, "g.txt", "Zpráva o sondě číslo 3. Nalezeny hřeby, vrstva ornice měla\n")
    french = _read(tmp_path, "f.txt", "Une très belle journée à la plage, près de la mer.\nIl était une fois.\n")
    assert good.pages[0].flags == [] and french.pages[0].flags == []


def test_pdf_mirrored_and_rotated_text_flags(tmp_path):
    pytest.importorskip("pypdfium2")
    mirrored = tf.read_document(_write(tmp_path, "m.pdf", pdf_bytes([["Mirror"]], matrix=(-1, 0, 0, 1))))
    rotated = tf.read_document(_write(tmp_path, "r.pdf", pdf_bytes([["Turned"]], matrix=(0, 1, -1, 0))))
    plain = tf.read_document(_write(tmp_path, "p.pdf", pdf_bytes([["Plain"]])))
    assert mirrored.pages[0].flags == ["mirrored_text=1"]
    assert rotated.pages[0].flags == ["rotated_text=1"]
    assert plain.pages[0].flags == []


# ── subtitles and e-mail ──────────────────────────────────────────────────────


def test_subtitles_keep_only_cue_text(tmp_path):
    srt = (
        "1\n00:00:01,000 --> 00:00:02,500\n<i>Dobrý</i> den\n\n2\n00:00:03,000 --> 00:00:04,000\n{\\an8}Druhá\nřádka\n"
    )
    assert _lines(_read(tmp_path, "s.srt", srt)) == [["Dobrý den", "Druhá", "řádka"]]
    vtt = (
        "WEBVTT\nKind: captions\n\nNOTE poznámka\nautora\n\nSTYLE\n::cue { color: red }\n\n"
        "uvod\n00:01.000 --> 00:02.000 align:start\n<v Jan>Ahoj &amp; <c.jmeno>Evo</c>\n"
    )
    doc = _read(tmp_path, "s.vtt", vtt)
    assert doc.kind == "vtt" and _lines(doc) == [["Ahoj & Evo"]]


def test_srt_content_in_a_txt_is_read_as_srt(tmp_path):
    doc = _read(tmp_path, "titulky.txt", "1\n00:00:01,000 --> 00:00:02,000\nText\n")
    assert doc.kind == "srt" and _lines(doc) == [["Text"]]


def test_eml_subject_then_plain_body_attachments_counted(tmp_path):
    msg = eml_bytes("Zpráva o sondě", plain="Dobrý den,\nposílám nálezy.", html="<p>HTML</p>", attachments=1)
    doc = _read(tmp_path, "m.eml", msg)
    assert doc.kind == "eml" and [ln for ln in doc.pages[0].lines if ln] == [
        "Zpráva o sondě",
        "Dobrý den,",
        "posílám nálezy.",
    ]
    assert "email_attachments_skipped=1" in doc.notes and tf.default_source_origin(doc) == "digital-born-eml"
    html_only = _read(tmp_path, "h.eml", eml_bytes("Předmět", html="<p>Jen <b>HTML</b></p><p>tělo</p>"))
    assert [ln for ln in html_only.pages[0].lines if ln] == ["Předmět", "Jen HTML", "tělo"]


def test_mbox_is_one_page_per_message(tmp_path):
    box = mbox_bytes([eml_bytes("První", plain="From the start"), eml_bytes("Druhá", plain="tělo")])
    doc = _read(tmp_path, "box.mbox", box)
    assert doc.kind == "mbox"
    assert [[ln for ln in p.lines if ln] for p in doc.pages] == [["První", "From the start"], ["Druhá", "tělo"]]


def test_memo_and_yaml_headers_stay_plain_text(tmp_path):
    assert _read(tmp_path, "memo.txt", "From: vedoucí\nTo: tým\nSubject: porada\n\nZítra v 9.\n").kind == "txt"
    assert _read(tmp_path, "c.yaml", "title: x\nauthor: y\ndate: z\n\nbody\n").kind == "txt"


# ── helpers and settings ──────────────────────────────────────────────────────


def test_lossy_reasons_strip_counts_and_include_page_flags():
    doc = tf.TextDocument(
        kind="pdf",
        media_type="application/pdf",
        pages=[tf.TextPage([], flags=["page_load_failed"])],
        notes=["jsonl_bad_records=3", "encoding_detected", "name_close_repaired", "xml_recovered"],
    )
    assert tf.lossy_reasons(doc) == ["jsonl_bad_records", "xml_recovered", "page_load_failed"]


def test_load_settings_notes_and_ratio_bounds():
    import configparser

    cfg = configparser.ConfigParser()
    cfg.read_dict({"TEXT_INGEST": {"NOTES": "end"}})
    assert tf.load_settings(cfg)[1].notes_placement == "end"
    for key, bad in (("NOTES", "margin"), ("PDF_GARBLE_THRESHOLD", "1.5"), ("PDF_OCR_LAYER_MIN_RATIO", "2")):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"TEXT_INGEST": {key: bad}})
        with pytest.raises(ValueError, match=key):
            tf.load_settings(cfg)
    with pytest.raises(ValueError, match="STRICT"):
        cfg = configparser.ConfigParser()
        cfg.read_dict({"TEXT_INGEST": {"STRICT": "ture"}})
        tf.config_bool(cfg, "TEXT_INGEST", "STRICT", False)


def test_origin_hint_and_native_pages_survive_the_worker_round_trip(tmp_path):
    doc = _read(tmp_path, "b.zip", make_zip([("1.xml", _page("Jedna", "1.jpg"))]))
    again = tf.TextDocument.from_dict(json.loads(json.dumps(doc.to_dict())))
    assert again == doc and again.origin_hint == "ocr:page-xml" and again.native_pages is True
