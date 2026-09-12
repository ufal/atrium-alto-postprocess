import json
from unittest.mock import patch

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
    """Auto-detect must still 400 on formats outside alto/text/json (#8)."""
    files = {"file": ("document.pdf", b"%PDF-1.4", "application/pdf")}
    data = {"task_type": "auto"}

    response = client.post("/process", files=files, data=data)
    assert response.status_code == 400


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
