import json
from pathlib import Path

import jsonschema
import pytest

import document_hook
from atrium_document import DocumentRecord, load_document
from extract_JSON_2_TXT import _parse_args, extract_single_page, process_json_to_txt

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "atrium_document.schema.json"


def test_extract_json_to_txt(tmp_path: Path):
    # Setup mock JSON
    mock_json = {
        "metadata": {"engine": "TestEngine"},
        "page": {
            "lines": [
                {"textline": "Hello World", "bbox": [0, 0, 10, 10]},
                {"textline": "This is a test.", "bbox": [10, 10, 20, 20]},
            ]
        },
    }

    input_file = tmp_path / "test.json"
    output_file = tmp_path / "test.txt"

    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(mock_json, f)

    # Run processor
    process_json_to_txt(input_file, output_file)

    # Assert
    with open(output_file, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Hello World" in content
    assert "This is a test." in content
    assert "TestEngine" not in content  # Metadata should be ignored


def test_extract_single_page_writes_alto_compatible_output_path(tmp_path: Path):
    """The CSV-driven worker follows the {file}/{file}-{page}.txt convention
    shared with the ALTO extractors, so classify_TEXT.py needs no changes."""
    mock_json = {"text": "Only line"}
    input_file = tmp_path / "doc7.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(mock_json, f)

    output_dir = tmp_path / "out"
    ok = extract_single_page(("doc7", 1, str(input_file), str(output_dir)))

    assert ok is True
    txt_path = output_dir / "doc7" / "doc7-1.txt"
    assert txt_path.exists()
    assert txt_path.read_text(encoding="utf-8") == "Only line"


def test_extract_single_page_resumes_existing_output(tmp_path: Path):
    """A pre-existing .txt is left untouched (resume support)."""
    output_dir = tmp_path / "out"
    save_dir = output_dir / "doc9"
    save_dir.mkdir(parents=True)
    txt_path = save_dir / "doc9-1.txt"
    txt_path.write_text("already extracted", encoding="utf-8")

    # Point at a nonexistent JSON input — if the worker tried to re-extract
    # it would fail; success here proves the resume short-circuit fired.
    ok = extract_single_page(("doc9", 1, str(tmp_path / "missing.json"), str(output_dir)))

    assert ok is True
    assert txt_path.read_text(encoding="utf-8") == "already extracted"


def test_extract_single_page_reports_failure_on_bad_json(tmp_path: Path):
    input_file = tmp_path / "broken.json"
    input_file.write_text("{not valid", encoding="utf-8")

    ok = extract_single_page(("broken", 1, str(input_file), str(tmp_path / "out")))

    assert ok is False


# ── CLI surface (issue #37 D4: --force-single-page) ──────────────────────────


def test_parse_args_force_single_page_defaults_to_none_when_absent():
    """Absent means 'use the config default', not 'false' — main() must be able
    to distinguish the two so [EXTRACT].FORCE_SINGLE_PAGE_JSON keeps working for
    orchestrated run_pipeline.py invocations that pass no extra CLI flags."""
    args = _parse_args([])
    assert args.force_single_page is None


def test_parse_args_force_single_page_flag_sets_true():
    args = _parse_args(["--force-single-page"])
    assert args.force_single_page is True


def test_parse_args_rejects_unknown_flag():
    with pytest.raises(SystemExit):
        _parse_args(["--not-a-real-flag"])


# ── Accretion contract (issue #37): ownership boundaries + preservation ──────


def test_accretion_preserves_unrelated_metadata_and_co_owned_fields(tmp_path: Path):
    """The json-keys extractor's accretion contribution must only ever touch its
    own `content` block and its own fields inside the field-split `pages` block —
    every other block, and every other tool's fields on a shared `pages` row,
    must come out exactly as they went in (issue #37 D2/D3)."""
    doc_dir = tmp_path / "doc"
    doc_dir.mkdir()

    # A baseline doc.json as it would exist after earlier pipeline stages have
    # already run: page-classification's block, nlp-enrich's entities, and the
    # first-writer-wins `source` block.
    with DocumentRecord("CTX99", "page-classification", out_dir=str(doc_dir)) as doc:
        doc.set_source(sha256="a" * 64, filename="CTX99.json", origin="ocr:generic")
        doc.merge_block("pages", [{"page": "1", "category": "Text", "category_confidence": 0.91}])
        doc.set_block("page_categories", {"1": "Text"})
    with DocumentRecord.open(
        "CTX99", "nlp-enrich", baseline=str(doc_dir / "CTX99.document.json"), out_dir=str(doc_dir)
    ) as doc:
        doc.set_block("entities", [{"surface": "Praha", "type_teitok": "LOC"}])

    baseline_path = doc_dir / "CTX99.document.json"
    baseline_before = load_document(str(baseline_path))

    # Now the json-keys extraction contributes its own share, as main() does.
    txt_dir = tmp_path / "txt"
    txt_dir.mkdir()
    (txt_dir / "CTX99").mkdir()
    (txt_dir / "CTX99" / "CTX99-1.txt").write_text("Extracted page text", encoding="utf-8")

    pages, content = document_hook.pages_and_content_from_text(str(txt_dir), "CTX99", [1], engine="json-keys")
    document_hook.write_document_block(
        str(doc_dir),
        "CTX99",
        run_id="r3",
        paradata_ref="paradata/r3_alto-postprocess.json",
        merge_blocks={"pages": pages},
        set_blocks={"content": content},
    )

    after = load_document(str(baseline_path))

    # Untouched blocks: byte-for-byte (semantically) identical to before this run.
    assert after["source"] == baseline_before["source"]
    assert after["page_categories"] == baseline_before["page_categories"]
    assert after["entities"] == baseline_before["entities"]

    # Co-owned `pages` row: page-classification's fields survive, alto-postprocess's
    # own field is added on the SAME row — neither overwrites the other.
    assert len(after["pages"]) == 1
    row = after["pages"][0]
    assert row["category"] == "Text"
    assert row["category_confidence"] == 0.91
    assert row["ocr"] == {"engine": "json-keys"}

    # This run's own block, freshly written.
    assert after["content"] == {"text": "Extracted page text"}


# ── Schema conformance (issue #37 D5) ─────────────────────────────────────────


@pytest.fixture(scope="module")
def document_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("force_single_page", [False, True])
def test_json_keys_accretion_output_is_schema_valid(tmp_path, document_schema, force_single_page):
    txt_dir = tmp_path / "txt"
    (txt_dir / "CTXschema").mkdir(parents=True)
    (txt_dir / "CTXschema" / "CTXschema-1.txt").write_text("Line one", encoding="utf-8")
    (txt_dir / "CTXschema" / "CTXschema-2.txt").write_text("Line two", encoding="utf-8")

    doc_dir = tmp_path / "doc"
    pages, content = document_hook.pages_and_content_from_text(
        str(txt_dir), "CTXschema", [1, 2], engine="json-keys", force_single_page=force_single_page
    )
    document_hook.write_document_block(
        str(doc_dir),
        "CTXschema",
        run_id="r1",
        paradata_ref="paradata/r1_alto-postprocess.json",
        merge_blocks={"pages": pages},
        set_blocks={"content": content},
    )

    record = load_document(str(doc_dir / "CTXschema.document.json"))
    jsonschema.validate(instance=record, schema=document_schema)


# ── (#31 Phase 5) one page, each text once ───────────────────────────────────

_AZURE_DOC = {
    "apiVersion": "2024-11-30",
    "analyzeResult": {
        "modelId": "prebuilt-read",
        "content": "Titulní strana\nZpráva o výzkumu\nDruhá strana",
        "pages": [
            {
                "pageNumber": 1,
                "lines": [{"content": "Titulní strana"}, {"content": "Zpráva o výzkumu"}],
                "words": [{"content": "Titulní"}, {"content": "strana"}, {"content": "Zpráva"}],
            },
            {"pageNumber": 2, "lines": [{"content": "Druhá strana"}], "words": [{"content": "Druhá"}]},
        ],
    },
}


def _split_and_extract(tmp_path: Path, name: str, data) -> list:
    import page_split

    src = tmp_path / "in" / f"{name}.json"
    src.parent.mkdir(exist_ok=True)
    src.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "split"
    count = page_split.split_json_document(str(src), str(out))
    pages = []
    for n in range(1, count + 1):
        txt = tmp_path / f"{name}-{n}.txt"
        process_json_to_txt(out / name / f"{name}-{n}.json", txt)
        pages.append(txt.read_text(encoding="utf-8").split("\n"))
    return pages


def test_split_azure_pages_carry_neither_the_header_text_nor_the_words(tmp_path: Path):
    """page_split keeps the document header in every page file; json-keys used to print
    Azure's whole-document `content` at the top of every page and each line's words
    after the lines. A page is now its own lines, once."""
    assert _split_and_extract(tmp_path, "az", _AZURE_DOC) == [
        ["Titulní strana", "Zpráva o výzkumu"],
        ["Druhá strana"],
    ]
    split_page = json.loads((tmp_path / "split" / "az" / "az-1.json").read_text(encoding="utf-8"))
    assert split_page["analyzeResult"]["content"] == _AZURE_DOC["analyzeResult"]["content"]  # header still kept


def test_json_keys_agrees_with_text_lines_on_an_azure_document(tmp_path: Path):
    import text_formats as tf

    src = tmp_path / "doc.json"
    src.write_text(json.dumps(_AZURE_DOC, ensure_ascii=False), encoding="utf-8")
    text_lines = [page.lines for page in tf.read_document(str(src)).pages]
    assert _split_and_extract(tmp_path, "az", _AZURE_DOC) == text_lines


def test_textract_flat_blocks_keep_the_lines_and_drop_the_words(tmp_path: Path):
    """Family B (AWS Textract): a LINE block's text is followed by its WORD blocks."""
    blocks = [
        {"BlockType": "PAGE", "Page": 1},
        {"BlockType": "LINE", "Page": 1, "Text": "První řádek"},
        {"BlockType": "WORD", "Page": 1, "Text": "První"},
        {"BlockType": "WORD", "Page": 1, "Text": "řádek"},
        {"BlockType": "PAGE", "Page": 2},
        {"BlockType": "LINE", "Page": 2, "Text": "Druhý řádek"},
        {"BlockType": "WORD", "Page": 2, "Text": "Druhý"},
    ]
    data = {"DocumentMetadata": {"Pages": 2}, "Blocks": blocks}
    assert _split_and_extract(tmp_path, "tx", data) == [["První řádek"], ["Druhý řádek"]]


def test_a_page_object_without_text_falls_back_to_the_whole_document(tmp_path: Path):
    """`pages` that is a dict without text (a page count) must not hide the text."""
    data = {"pages": {"count": 1}, "text": "Jediný řádek"}
    src = tmp_path / "d.json"
    src.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "d.txt"
    process_json_to_txt(src, out)
    assert out.read_text(encoding="utf-8") == "Jediný řádek"


def test_embedded_binary_strings_are_not_text(tmp_path: Path):
    b64 = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODk="
    src = tmp_path / "b.json"
    src.write_text(json.dumps({"text": "Skutečný text", "data": b64}), encoding="utf-8")
    out = tmp_path / "b.txt"
    process_json_to_txt(src, out)
    assert out.read_text(encoding="utf-8") == "Skutečný text"


def test_json_stats_counts_the_lines_that_will_be_extracted(tmp_path: Path):
    pytest.importorskip("pandas")
    import json_stats_create
    import page_split

    src = tmp_path / "in" / "az.json"
    src.parent.mkdir()
    src.write_text(json.dumps(_AZURE_DOC, ensure_ascii=False), encoding="utf-8")
    page_split.split_json_document(str(src), str(tmp_path / "split"))
    page = tmp_path / "split" / "az" / "az-1.json"
    record, skipped = json_stats_create._process_single_json(str(page), page.name)
    assert skipped is None
    assert (record["file"], record["page"], record["strings"]) == ("az", "1", 2)


# ── (#31 Phase 5) resume notices a changed page; `0001` ids; --input-csv ─────


def test_extract_single_page_redoes_a_page_whose_json_is_newer(tmp_path: Path):
    import os

    src = tmp_path / "doc9-1.json"
    src.write_text(json.dumps({"text": "new text"}), encoding="utf-8")
    txt = tmp_path / "out" / "doc9" / "doc9-1.txt"
    txt.parent.mkdir(parents=True)
    txt.write_text("old text", encoding="utf-8")
    st = txt.stat()
    os.utime(txt, ns=(st.st_atime_ns, st.st_mtime_ns - 60 * 10**9))  # older than its input

    assert extract_single_page(("doc9", 1, str(src), str(tmp_path / "out"))) is True
    assert txt.read_text(encoding="utf-8") == "new text"


def test_main_keeps_zero_padded_doc_ids_and_takes_input_csv(tmp_path: Path, monkeypatch):
    """`0001` used to be read as 1, so the text went to `1/1-1.txt` and classify, looking
    under `0001/`, found nothing. --input-csv now reaches this stage (run_pipeline)."""
    import extract_JSON_2_TXT as ext

    monkeypatch.chdir(tmp_path)
    page = tmp_path / "PAGE_JSON" / "0001" / "0001-1.json"
    page.parent.mkdir(parents=True)
    page.write_text(json.dumps({"text": "Řádek"}), encoding="utf-8")
    csv_path = tmp_path / "stats.csv"
    csv_path.write_text(f"file,page,textlines,illustrations,graphics,strings,path\n0001,1,0,0,0,1,{page}\n")
    out_dir = tmp_path / "PAGE_TXT_JSON"
    monkeypatch.setattr(ext, "OUTPUT_TEXT_DIR", str(out_dir))
    monkeypatch.setattr(ext, "MAX_WORKERS", 1)
    monkeypatch.setattr(ext, "INPUT_CSV", str(tmp_path / "not-this.csv"))

    ext.main(["--input-csv", str(csv_path)])

    assert (out_dir / "0001" / "0001-1.txt").read_text(encoding="utf-8") == "Řádek"
    assert not (out_dir / "1").exists()
