import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from atrium_document import DocumentRecord, canonical_doc_id, load_document
from service.text_api import app

client = TestClient(app)


def test_info_endpoint():
    response = client.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "active"
    assert "quality_categories" in data
    assert "Clear" in data["quality_categories"]


def test_info_version_matches_para_config():
    """The API version must come from para_config.txt [tool], never hardcoded."""
    import configparser
    from pathlib import Path

    config = configparser.ConfigParser()
    config.read(Path(__file__).resolve().parent.parent / "setup" / "para_config.txt", encoding="utf-8")
    expected = config.get("tool", "version").lstrip("v")

    response = client.get("/info")
    assert response.status_code == 200
    assert response.json()["version"] == expected
    assert app.version == expected


@patch("service.text_api.text_manager.process_text_file", create=True)
def test_process_text_auto_routing(mock_process):
    """Ensure text uploads correctly route to the text_manager text processor."""
    mock_process.return_value = {
        "type": "plain_text",
        "cleaned_lines": [{"line_num": 1, "text": "Mocked Line", "category": "Clear"}],
    }

    content = b"Mock line content"
    files = {"file": ("document.txt", content, "text/plain")}
    data = {"task_type": "auto"}

    response = client.post("/process", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["type"] == "plain_text"
    assert res_data["cleaned_lines"][0]["category"] == "Clear"
    assert res_data["filename"] == "document.txt"


@patch("service.text_api.text_manager.process_alto", create=True)
def test_process_alto_explicit_routing(mock_process):
    """Ensure ALTO XML uploads hit the alto pipeline specifically."""
    mock_process.return_value = {"type": "alto_xml", "cleaned_lines": []}

    content = b"<alto></alto>"
    files = {"file": ("document.xml", content, "application/xml")}
    data = {"task_type": "alto"}

    response = client.post("/process", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["type"] == "alto_xml"


@patch("service.text_api.text_manager.process_json", create=True)
def test_process_json_auto_routing(mock_process):
    """Ensure .json uploads auto-detect to task_type='json' and route to process_json."""
    mock_process.return_value = {
        "type": "json",
        "cleaned_lines": [{"line_num": 1, "text": "Mocked JSON Line", "category": "Clear"}],
    }

    content = b'{"text": "Mocked JSON Line"}'
    files = {"file": ("document.json", content, "application/json")}
    data = {"task_type": "auto"}

    response = client.post("/process", files=files, data=data)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["type"] == "json"
    assert res_data["cleaned_lines"][0]["category"] == "Clear"
    assert res_data["filename"] == "document.json"


@patch("service.text_api.text_manager.process_json", create=True)
def test_process_json_explicit_routing(mock_process):
    """Ensure task_type='json' routes to process_json regardless of filename."""
    mock_process.return_value = {"type": "json", "cleaned_lines": []}

    content = b'{"text": "irrelevant"}'
    files = {"file": ("upload.dat", content, "application/octet-stream")}
    data = {"task_type": "json"}

    response = client.post("/process", files=files, data=data)
    assert response.status_code == 200
    assert response.json()["type"] == "json"


def test_info_lists_json_as_supported_format():
    response = client.get("/info")
    assert response.status_code == 200
    assert any("JSON" in fmt for fmt in response.json()["supported_formats"])


def test_process_unrecognized_extension_still_rejected():
    """Auto-detect must still 400 on a file no reader supports (#8).

    (#31) PDF used to be the example here; it is a supported "document" now, so the
    unsupported case is plain binary — the 400 names the reason code."""
    files = {"file": ("document.bin", bytes(range(256)) * 4, "application/octet-stream")}
    data = {"task_type": "auto"}

    response = client.post("/process", files=files, data=data)
    assert response.status_code == 400
    assert "binary_content" in response.json()["detail"]


@pytest.mark.parametrize(
    "name, content",
    [
        ("scan.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64),
        ("letter.doc", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512),
    ],
)
def test_process_rejects_images_and_legacy_office_with_reason(name, content):
    response = client.post(
        "/process", files={"file": (name, content, "application/octet-stream")}, data={"task_type": "auto"}
    )
    assert response.status_code == 400
    assert ("image_needs_ocr" if name.endswith(".png") else "legacy_office_unsupported") in response.json()["detail"]


# ── (#31) any other text-bearing upload → task_type "document" ───────────────

_DOC_RESULT = {
    "type": "document",
    "format": "pdf",
    "media_type": "application/pdf",
    "origin": "ocr:pdf-text-layer",
    "pages": [{"page": "1", "page_label": "i", "lines": 1}, {"page": "2", "page_label": "ii", "lines": 1}],
    "cleaned_lines": [
        {"text": "Strana jedna", "lang": "ces", "quality_score": 0.9, "category": "Clear", "line_num": 1, "page": "1"},
        {"text": "Str4na dv4", "lang": "ces", "quality_score": 0.5, "category": "Trash", "line_num": 1, "page": "2"},
    ],
}


@patch("service.text_api.text_manager.process_document", create=True)
def test_process_pdf_auto_routes_to_document(mock_process):
    mock_process.return_value = dict(_DOC_RESULT)
    files = {"file": ("scan.pdf", b"%PDF-1.4\n%minimal", "application/pdf")}
    response = client.post("/process", files=files, data={"task_type": "auto"})
    assert response.status_code == 200
    assert response.json()["type"] == "document"
    assert mock_process.call_count == 1


@patch("service.text_api.text_manager.process_document", create=True)
def test_process_docx_is_sniffed_by_content_not_name(mock_process):
    """A DOCX uploaded under a wrong extension is still read as a document."""
    from tests.text_format_fixtures import docx_bytes, w_p, w_t

    mock_process.return_value = dict(_DOC_RESULT)
    files = {"file": ("report.bin", docx_bytes(w_p(w_t("Ahoj"))), "application/octet-stream")}
    response = client.post("/process", files=files, data={"task_type": "auto"})
    assert response.status_code == 200
    assert mock_process.call_count == 1


@patch("service.text_api.text_manager.process_document", create=True)
@patch("service.text_api.text_manager.process_alto", create=True)
def test_process_xml_routes_by_root_element(mock_alto, mock_document):
    """ALTO keeps the ALTO path; PAGE XML (also .xml) becomes a document."""
    mock_alto.return_value = {"type": "alto_xml", "cleaned_lines": []}
    mock_document.return_value = dict(_DOC_RESULT)
    alto = b'<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#"><Layout/></alto>'
    page = b'<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"><Page/></PcGts>'

    assert client.post("/process", files={"file": ("a.xml", alto, "application/xml")}).json()["type"] == "alto_xml"
    assert client.post("/process", files={"file": ("p.xml", page, "application/xml")}).json()["type"] == "document"
    assert (mock_alto.call_count, mock_document.call_count) == (1, 1)


@patch("service.text_api.text_manager.process_document", create=True)
def test_process_document_read_failure_is_422_with_reason(mock_process):
    from text_formats import IngestError

    mock_process.side_effect = IngestError("encrypted", "password-protected PDF")
    response = client.post("/process", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")})
    assert response.status_code == 422
    assert response.json()["detail"].startswith("encrypted")


def test_info_lists_document_formats():
    formats = client.get("/info").json()["supported_formats"]
    assert formats[:3] == ["ALTO XML (.xml)", "Plain Text (.txt)", "Generic JSON (.json)"]
    assert any(fmt.startswith("PDF") for fmt in formats)
    assert any("DOCX" in fmt for fmt in formats)


@patch("service.text_api.text_manager.process_document", create=True)
def test_process_document_accretes_per_page(mock_process, tmp_path, monkeypatch):
    """An OCR-class document (PDF with an OCR text layer) accretes each line under its
    own page, with per-page metrics computed from that page's lines."""
    monkeypatch.chdir(tmp_path)
    mock_process.return_value = json.loads(json.dumps(_DOC_RESULT))
    baseline_path = _real_baseline(tmp_path, pages=("1", "2"))

    response = client.post(
        "/process",
        files={
            "file": (f"{_DOC_ID}.pdf", b"%PDF-1.4", "application/pdf"),
            "document_record": (f"{_DOC_ID}.document.json", baseline_path.read_bytes(), "application/json"),
        },
        data={"task_type": "document"},
    )
    assert response.status_code == 200
    record = response.json()["document_json_out"]
    assert [(line["page"], line["line"], line["categ"]) for line in record["lines"]] == [
        ("1", 1, "Clear"),
        ("2", 1, "Trash"),
    ]
    by_page = {page["page"]: page for page in record["pages"]}
    assert by_page["1"]["quality_band"] == "Clear" and by_page["2"]["quality_band"] == "Trash"


@patch("service.text_api.text_manager.process_document", create=True)
def test_process_born_digital_document_contributes_nothing_to_the_record(mock_process, tmp_path, monkeypatch, caplog):
    """A born-digital upload (DOCX, visible-text PDF) is digital-convert's to originate
    (atrium_document §1a): the lines are returned, the record gets no pages/lines."""
    monkeypatch.chdir(tmp_path)
    result = json.loads(json.dumps(_DOC_RESULT))
    result["origin"] = "digital-born-docx"
    mock_process.return_value = result
    baseline_path = _real_baseline(tmp_path, pages=("1",))

    response = client.post(
        "/process",
        files={
            "file": (f"{_DOC_ID}.docx", b"PK\x03\x04", "application/octet-stream"),
            "document_record": (f"{_DOC_ID}.document.json", baseline_path.read_bytes(), "application/json"),
        },
        data={"task_type": "document"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["cleaned_lines"]) == 2
    assert "lines" not in body["document_json_out"]
    assert "born-digital upload" in caplog.text


# ── (atrium-project#10 J1 + D2) the `document_record` accretion parameter ─────
#
# This parameter had ZERO coverage, which is why two independent P0s shipped in one
# endpoint: `lines` (a key text_inference never returns, so the lines merge was skipped
# on every call) and a hardcoded `[{"page": "1", "quality_score":
# result.get("doc_quality", 1.0)}]` (also a key it never returns, so every record
# claimed a perfect single page). D2 rode along on `Path(filename.lower()).stem`, which
# re-keyed the whole record. The tests below post a REAL baseline, which is the only
# way any of that becomes visible.

_ALTO_NS = "http://www.loc.gov/standards/alto/ns-v3#"

# One page, labelled 7 — PHYSICAL_IMG_NR is what page_split.py names pages by, so a
# real document's labels are frequently NOT "1". That is what makes the old hardcoded
# row observable: it appended a page the document does not have.
_ONE_PAGE_ALTO = f"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="{_ALTO_NS}">
  <Layout>
    <Page ID="P1" PHYSICAL_IMG_NR="7" WIDTH="1000" HEIGHT="2000"><PrintSpace/></Page>
  </Layout>
</alto>
"""

_TWO_PAGE_ALTO = f"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="{_ALTO_NS}">
  <Layout>
    <Page ID="P1" PHYSICAL_IMG_NR="7" WIDTH="1000" HEIGHT="2000"><PrintSpace/></Page>
    <Page ID="P2" PHYSICAL_IMG_NR="8" WIDTH="1000" HEIGHT="2000"><PrintSpace/></Page>
  </Layout>
</alto>
"""

_CLASSIFIED_LINES = [
    {
        "text": "První řádek stránky",
        "lang": "ces",
        "lang_score": 0.99,
        "perplexity": 42.0,
        "quality_score": 0.9,
        "category": "Clear",
        "line_num": 1,
    },
    {
        "text": "|||| ~~ 3",
        "lang": "deu",
        "lang_score": 0.11,
        "perplexity": 9000.0,
        "quality_score": 0.1,
        # A load-bearing value: json_to_md's DROP_CATEGORIES keys off this exact
        # spelling, so it has to survive the mapping verbatim.
        "category": "Garbage",
        "line_num": 2,
    },
]

_DOC_ID = "CTX000000001"


def _real_baseline(tmp_path, pages=("7", "8", "9")):
    """A baseline as the pipeline would really hand it over: page-classification's own
    block plus its fields on the shared `pages` rows, and the immutable `source`."""
    with DocumentRecord(_DOC_ID, "page-classification", out_dir=str(tmp_path)) as doc:
        doc.set_source(sha256="a" * 64, filename=f"{_DOC_ID}.alto.xml", origin="ABBYY-ALTO")
        doc.merge_block("pages", [{"page": p, "category": "Text", "category_confidence": 0.91} for p in pages])
        doc.set_block("page_categories", {p: "Text" for p in pages})
    return tmp_path / f"{_DOC_ID}.document.json"


@patch("service.text_api.text_manager.process_alto", create=True)
def test_process_accretes_real_lines_and_pages_onto_a_baseline(mock_process, tmp_path, monkeypatch):
    """(J1 + D2) One POST, four separate regressions pinned."""
    monkeypatch.chdir(tmp_path)  # ParadataLogger writes ./paradata
    mock_process.return_value = {"type": "alto_xml", "cleaned_lines": _CLASSIFIED_LINES}
    baseline_path = _real_baseline(tmp_path)
    baseline_before = load_document(str(baseline_path))

    response = client.post(
        "/process",
        files={
            # Original case AND a multi-dot name, i.e. exactly the convention D2 broke.
            "file": (f"{_DOC_ID}.alto.xml", _ONE_PAGE_ALTO.encode("utf-8"), "application/xml"),
            "document_record": (
                f"{_DOC_ID}.document.json",
                baseline_path.read_bytes(),
                "application/json",
            ),
        },
        data={"task_type": "auto"},
    )
    assert response.status_code == 200
    record = response.json()["document_json_out"]

    # (d) D2: identity is unchanged. `Path("CTX000000001.alto.xml".lower()).stem` gave
    # "ctx000000001.alto", and DocumentRecord.__init__ sets doc_id unconditionally, so
    # the accreted output was re-keyed away from every upstream block.
    assert record["doc_id"] == baseline_before["doc_id"] == _DOC_ID
    assert canonical_doc_id(f"{_DOC_ID}.alto.xml") == _DOC_ID

    # (a) J1: per-line text actually arrives. `result["lines"]` was always empty, so
    # write_document_block's `if records:` guard skipped this merge every single call.
    lines = record["lines"]
    assert [line["text"] for line in lines] == [entry["text"] for entry in _CLASSIFIED_LINES]
    assert [line["line"] for line in lines] == [1, 2]
    # `category` in the inference layer, `categ` in the schema — and the load-bearing
    # value passes through unchanged.
    assert [line["categ"] for line in lines] == ["Clear", "Garbage"]
    assert lines[0]["lang"] == "ces"
    assert lines[0]["quality_score"] == 0.9
    # Attributed to the page the ALTO itself names, not to a fabricated "1".
    assert {line["page"] for line in lines} == {"7"}

    # (b) J1: the page count is the document's, not the literal 1 the stub wrote. The
    # old row named a page this document does not have, so it APPENDED a fourth.
    assert [page["page"] for page in record["pages"]] == ["7", "8", "9"]
    page7 = next(page for page in record["pages"] if page["page"] == "7")
    # Real metrics, computed from the real lines — mean of 0.9 and 0.1, never 1.0.
    assert page7["quality_score"] == 0.5
    assert page7["quality_band"] == "Clear"  # 1 Clear vs 0 Noisy / 0 Trash
    assert all("quality_score" not in page for page in record["pages"] if page["page"] != "7")

    # Co-owned fields on the row this run touched survive untouched.
    assert page7["category"] == "Text"
    assert page7["category_confidence"] == 0.91

    # (c) An upstream block passes through byte-for-byte.
    assert record["page_categories"] == baseline_before["page_categories"]
    assert record["source"] == baseline_before["source"]

    # Rule 4: this contribution is stamped, and the baseline is acknowledged.
    assert record["assembled"]["blocks"]["lines"]["program"] == "alto-postprocess"
    assert record["assembled"]["blocks"]["page_categories"]["program"] == "page-classification"
    assert record["assembled"]["had_baseline"] is True


@patch("service.text_api.text_manager.process_text_file", create=True)
def test_process_text_upload_accretes_under_the_single_page_label(mock_process, tmp_path, monkeypatch):
    """A .txt upload has no page identity of its own, so it accretes onto the label
    page_split gives a single-page document — and must still not invent a page row when
    the classifier returned nothing."""
    monkeypatch.chdir(tmp_path)
    mock_process.return_value = {"type": "plain_text", "cleaned_lines": []}
    baseline_path = _real_baseline(tmp_path, pages=("1",))

    response = client.post(
        "/process",
        files={
            "file": (f"{_DOC_ID}.txt", b"a line\n", "text/plain"),
            "document_record": (f"{_DOC_ID}.document.json", baseline_path.read_bytes(), "application/json"),
        },
        data={"task_type": "auto"},
    )
    assert response.status_code == 200
    record = response.json()["document_json_out"]
    # No lines classified -> no rows invented on either block, and the baseline's own
    # single page row is left exactly as it was.
    assert "lines" not in record
    assert record["pages"] == [{"page": "1", "category": "Text", "category_confidence": 0.91}]


@patch("service.text_api.text_manager.process_alto", create=True)
def test_process_refuses_to_attribute_a_multipage_upload(mock_process, tmp_path, monkeypatch, caplog):
    """The service's inference flattens every <Page> into one list scaled by the first
    page's geometry, so `result` cannot say which page a line came from. Guessing would
    write misattributed rows into a record other tools align their fields onto, so the
    accretion is skipped — loudly — while the response still carries the lines.

    The warning moved from print(file=sys.stderr) to logger.warning() under issue #61
    (logs as an event stream); assert on caplog rather than capsys accordingly."""
    monkeypatch.chdir(tmp_path)
    mock_process.return_value = {"type": "alto_xml", "cleaned_lines": _CLASSIFIED_LINES}
    baseline_path = _real_baseline(tmp_path, pages=("7", "8"))

    response = client.post(
        "/process",
        files={
            "file": (f"{_DOC_ID}.alto.xml", _TWO_PAGE_ALTO.encode("utf-8"), "application/xml"),
            "document_record": (f"{_DOC_ID}.document.json", baseline_path.read_bytes(), "application/json"),
        },
        data={"task_type": "alto"},
    )
    assert response.status_code == 200
    body = response.json()
    assert [entry["text"] for entry in body["cleaned_lines"]] == [e["text"] for e in _CLASSIFIED_LINES]
    record = body["document_json_out"]
    assert "lines" not in record
    assert record["pages"] == json.loads(baseline_path.read_text(encoding="utf-8"))["pages"]
    assert "2 <Page> elements" in caplog.text


# ── (#31 Phase 4) one status mapping: unsupported → 400, unreadable → 422 ─────


def test_a_damaged_container_found_by_sniffing_is_422():
    files = {"file": ("report.docx", b"PK\x03\x04" + b"\x00" * 64, "application/octet-stream")}
    response = client.post("/process", files=files, data={"task_type": "auto"})
    assert response.status_code == 422 and response.json()["detail"].startswith("corrupt")


def test_an_explicit_document_that_is_an_image_is_400():
    files = {"file": ("scan.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 64, "image/png")}
    response = client.post("/process", files=files, data={"task_type": "document"})
    assert response.status_code == 400 and response.json()["detail"].startswith("image_needs_ocr")


def test_a_document_without_text_is_422_no_text():
    files = {"file": ("blank.md", b"\n\n   \n", "text/markdown")}
    response = client.post("/process", files=files, data={"task_type": "document"})
    assert response.status_code == 422 and response.json()["detail"].startswith("no_text")


def test_a_text_upload_that_is_not_text_is_400():
    files = {"file": ("blob.txt", bytes(range(256)) * 8, "text/plain")}
    response = client.post("/process", files=files, data={"task_type": "auto"})
    assert response.status_code == 400 and response.json()["detail"].startswith("binary_content")


def test_every_reason_code_maps_to_400_or_422():
    from service.text_api import UNSUPPORTED_REASONS, _ingest_http_error
    from text_formats import REASON_CODES, IngestError

    assert UNSUPPORTED_REASONS <= set(REASON_CODES)
    for code in REASON_CODES:
        status = _ingest_http_error(IngestError(code)).status_code
        assert status == (400 if code in UNSUPPORTED_REASONS else 422), code


@patch("service.text_api.text_manager.process_document", create=True)
@patch("service.text_api.text_manager.process_alto", create=True)
def test_a_compressed_alto_upload_goes_to_the_document_reader(mock_alto, mock_document):
    from tests.text_format_fixtures import compress_bytes

    mock_document.return_value = dict(_DOC_RESULT)
    alto = b'<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#"><Layout/></alto>'
    files = {"file": ("a.alto.xml.gz", compress_bytes(alto), "application/gzip")}
    assert client.post("/process", files=files).json()["type"] == "document"
    assert (mock_alto.call_count, mock_document.call_count) == (0, 1)
