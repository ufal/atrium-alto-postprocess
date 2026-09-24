"""
tests/test_extract_text.py — stage 3 of the text-lines method (#31), in-process.
"""

import csv

import pytest

import extract_TEXT_2_TXT as ext
from atrium_document import load_document, validate_document


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCUMENT_JSON_DIR", raising=False)
    return tmp_path


def _setup(tmp_path, monkeypatch, pages, *, json_dir="", extra_ingest=""):
    """pages: {(file, page): bytes}. Writes page files + stats CSV + config."""
    stats = tmp_path / "stats.csv"
    rows = ["file,page,textlines,illustrations,graphics,strings,path"]
    for (file_id, page), data in pages.items():
        path = tmp_path / "PAGE_TEXT" / file_id / f"{file_id}-{page}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        rows.append(f"{file_id},{page},0,0,0,0,{path}")
    stats.write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = tmp_path / "config.txt"
    config.write_text(
        "\n".join(
            [
                "[EXTRACT]",
                f"INPUT_CSV = {stats}",
                f"OUTPUT_TXT_TEXT = {tmp_path / 'OUT'}",
                f"OUTPUT_LINES_TEXT = {tmp_path / 'LINES'}",
                "[CLASSIFY]",
                f"OUTPUT_LINES_LOG = {tmp_path / 'CATEG'}",
                "[DOCUMENT]",
                f"JSON_DIR = {json_dir}",
                "[TEXT_INGEST]",
                extra_ingest,
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ext, "CONFIG_PATH", str(config))
    return tmp_path / "OUT", tmp_path / "LINES"


def _table(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_blank_lines_dropped_long_lines_wrapped_and_line_table_matches_readlines(workdir, monkeypatch):
    long_line = ("slovo " * 400).strip()
    out, lines_dir = _setup(
        workdir,
        monkeypatch,
        {("0001", 1): f"První\n\n  \nDruhá­\n{long_line}\n".encode(), ("0001", 2): "Třetí\n".encode("cp1250")},
    )
    assert ext.main([]) == 0

    page1 = (out / "0001" / "0001-1.txt").read_text(encoding="utf-8").splitlines()
    assert page1[:2] == ["První", "Druhá-"] and all(len(x) <= 1000 for x in page1)
    assert (out / "0001" / "0001-2.txt").read_text(encoding="utf-8") == "Třetí"

    rows = _table(lines_dir / "0001.csv")
    assert list(rows[0]) == ["file", "page_num", "line_num", "text", "page_label"]
    for row in rows:  # the invariant classify_TEXT's numbering relies on
        lines = (out / row["file"] / f"{row['file']}-{row['page_num']}.txt").open(encoding="utf-8").readlines()
        assert lines[int(row["line_num"]) - 1].rstrip("\n") == row["text"]
    assert {r["file"] for r in rows} == {"0001"}  # a numeric-looking id survives


def test_keep_blank_lines_and_max_line_chars_from_config(workdir, monkeypatch):
    out, lines_dir = _setup(
        workdir,
        monkeypatch,
        {("d", 1): b"a\n\nbbbbbbbbbb\n\n"},
        extra_ingest="KEEP_BLANK_LINES = true\nMAX_LINE_CHARS = 4",
    )
    assert ext.main([]) == 0
    # Inner blank kept, trailing blank trimmed (readlines() could not see it).
    assert (out / "d" / "d-1.txt").read_text(encoding="utf-8") == "a\n\nbbbb\nbbbb\nbb"
    assert [r["text"] for r in _table(lines_dir / "d.csv")] == ["a", "", "bbbb", "bbbb", "bb"]


def test_page_labels_come_from_the_pages_report(workdir, monkeypatch):
    _out, lines_dir = _setup(workdir, monkeypatch, {("s", 1): b"x\n", ("s", 2): b"y\n"})
    (workdir / "PAGE_TEXT" / "pages_report.csv").write_text(
        "file,page,page_label,text_layer,needs_ocr_reason,lines,images,flags\ns,1,Nálezy,,,1,0,\ns,2,Vrstvy,,,1,0,\n",
        encoding="utf-8",
    )
    assert ext.main([]) == 0
    assert [r["page_label"] for r in _table(lines_dir / "s.csv")] == ["Nálezy", "Vrstvy"]


def test_refuses_to_write_line_tables_into_the_classify_output_dir(workdir, monkeypatch):
    _setup(workdir, monkeypatch, {("d", 1): b"x\n"})
    assert ext.main(["--lines-dir", str(workdir / "CATEG")]) == 2


def test_missing_csv_and_missing_columns(workdir, monkeypatch):
    _setup(workdir, monkeypatch, {("d", 1): b"x\n"})
    assert ext.main(["--input-csv", str(workdir / "nope.csv")]) == 1
    bad = workdir / "bad.csv"
    bad.write_text("file,page\nd,1\n", encoding="utf-8")
    assert ext.main(["--input-csv", str(bad)]) == 1


def test_unreadable_page_is_skipped_not_fatal(workdir, monkeypatch):
    out, _ = _setup(workdir, monkeypatch, {("d", 1): b"ok\n", ("d", 2): bytes(range(256)) * 4})
    assert ext.main([]) == 0
    assert (out / "d" / "d-1.txt").exists() and not (out / "d" / "d-2.txt").exists()


def test_document_record_is_schema_valid(workdir, monkeypatch):
    pytest.importorskip("jsonschema")
    docs = workdir / "docs"
    docs.mkdir()
    _setup(workdir, monkeypatch, {("d", 1): b"line one\n", ("d", 2): b"line two\n"}, json_dir=str(docs))
    assert ext.main([]) == 0
    record = load_document(str(docs / "d.document.json"))
    validate_document(record)
    assert [p["page"] for p in record["pages"]] == ["1", "2"]
    assert record["pages"][0]["ocr"] == {"engine": "text-lines"}
    assert record["content"]["text"] == "line one\n\nline two"


# ── (#31 Phase 4) per-document isolation, labels per directory, strictness ────


def test_a_failing_document_record_costs_that_document_only(workdir, monkeypatch):
    out, lines_dir = _setup(workdir, monkeypatch, {("a", 1): b"jedna\n", ("b", 1): b"dva\n"}, json_dir=str(workdir))
    calls = []

    def record(dir_, doc_id, *a, **k):
        calls.append(doc_id)
        if doc_id == "a":
            raise ValueError("does not validate")

    monkeypatch.setattr(ext.document_hook, "write_document_block", record)
    assert ext.main([]) == 0
    assert calls == ["a", "b"] and (lines_dir / "b.csv").exists() and (out / "b" / "b-1.txt").exists()
    assert ext.main(["--strict"]) == 1


def test_an_unreadable_stats_csv_stops_the_stage_with_a_message(workdir, monkeypatch, capsys):
    _setup(workdir, monkeypatch, {("a", 1): b"x\n"})
    bad = workdir / "bad.csv"
    bad.write_text('file,page,path\n"unterminated,1,x\n', encoding="utf-8")
    assert ext.main(["--input-csv", str(bad)]) == 1
    assert "cannot read" in capsys.readouterr().err


def test_page_labels_are_looked_up_next_to_each_documents_folder(workdir, monkeypatch):
    out, lines_dir = _setup(workdir, monkeypatch, {("a", 1): b"x\n"})
    other = workdir / "OTHER" / "b"
    other.mkdir(parents=True)
    (other / "b-1.txt").write_bytes(b"y\n")
    (workdir / "OTHER" / "pages_report.csv").write_text(
        "file,page,page_label,text_layer,needs_ocr_reason,lines,images,flags\nb,1,List B,,,1,0,\n", encoding="utf-8"
    )
    (workdir / "PAGE_TEXT" / "pages_report.csv").write_text(
        "file,page,page_label,text_layer,needs_ocr_reason,lines,images,flags\na,1,List A,,,1,0,\n", encoding="utf-8"
    )
    stats = workdir / "stats.csv"
    stats.write_text(stats.read_text(encoding="utf-8") + f"b,1,0,0,0,0,{other / 'b-1.txt'}\n", encoding="utf-8")
    assert ext.main([]) == 0
    assert _table(lines_dir / "a.csv")[0]["page_label"] == "List A"
    assert _table(lines_dir / "b.csv")[0]["page_label"] == "List B"


def test_a_damaged_pages_report_costs_only_the_labels(workdir, monkeypatch):
    out, lines_dir = _setup(workdir, monkeypatch, {("a", 1): b"x\n"})
    (workdir / "PAGE_TEXT" / "pages_report.csv").write_bytes(b"\xff\xfe\x00broken")
    assert ext.main([]) == 0
    assert _table(lines_dir / "a.csv")[0]["page_label"] == "1"


def test_strict_exit_code_from_the_flag_and_the_config(workdir, monkeypatch):
    _setup(workdir, monkeypatch, {("a", 1): b"x\n"})
    stats = workdir / "stats.csv"
    stats.write_text(stats.read_text(encoding="utf-8") + "gone,1,0,0,0,0,/nonexistent/gone-1.txt\n", encoding="utf-8")
    assert ext.main([]) == 0
    assert ext.main(["--strict"]) == 1
    _setup(workdir, monkeypatch, {("a", 1): b"x\n"}, extra_ingest="STRICT = true")
    stats.write_text(stats.read_text(encoding="utf-8") + "gone,1,0,0,0,0,/nonexistent/gone-1.txt\n", encoding="utf-8")
    assert ext.main([]) == 1
    assert ext.main(["--no-strict"]) == 0
