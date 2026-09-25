import json

import pytest

from json_stats_create import _process_single_json, main, process_json_files


def test_json_stats_cli_help(capsys):
    with pytest.raises(SystemExit) as e:  # argparse exits 0 on --help
        main(["--help"])
    assert e.value.code == 0
    assert "input_folder" in capsys.readouterr().out


def test_json_stats_missing_args(capsys):
    with pytest.raises(SystemExit) as e:  # missing required positional → exit 2
        main([])
    assert e.value.code == 2
    assert "required" in capsys.readouterr().err.lower()


def test_process_single_json_valid_file(tmp_path):
    """(#31/D7) file/page are now derived from the split filename
    ("<file>-<page>.json"), mirroring alto_stats_create's convention,
    instead of a hardcoded page=1."""
    mock_json = {"page": {"lines": [{"textline": "Hello World"}, {"textline": "Second line"}]}}
    json_path = tmp_path / "doc1-1.json"
    json_path.write_text(json.dumps(mock_json), encoding="utf-8")

    rec, reason = _process_single_json(str(json_path), "doc1-1.json")

    assert reason is None
    assert rec["file"] == "doc1"
    assert rec["page"] == "1"
    assert rec["textlines"] == 0
    assert rec["illustrations"] == 0
    assert rec["graphics"] == 0
    assert rec["strings"] == 2
    assert rec["path"] == str(json_path)


def test_process_single_json_no_page_suffix_yields_empty_page(tmp_path):
    """A filename with no '-<page>' suffix (no split occurred, e.g. a stray
    file dropped straight in) yields an empty page id rather than crashing —
    mirrors alto_stats_create._process_single_xml's fallback."""
    json_path = tmp_path / "doc1.json"
    json_path.write_text(json.dumps({"text": "solo"}), encoding="utf-8")

    rec, reason = _process_single_json(str(json_path), "doc1.json")

    assert reason is None
    assert rec["file"] == "doc1"
    assert rec["page"] == ""


def test_process_single_json_malformed_file_is_skipped(tmp_path):
    json_path = tmp_path / "broken-1.json"
    json_path.write_text("{not valid json", encoding="utf-8")

    rec, reason = _process_single_json(str(json_path), "broken-1.json")

    assert rec is None
    assert reason is not None


def test_process_json_files_walks_directory(tmp_path):
    (tmp_path / "a-1.json").write_text(json.dumps({"text": "A"}), encoding="utf-8")
    (tmp_path / "b-1.json").write_text(json.dumps({"text": "B"}), encoding="utf-8")
    (tmp_path / "not_json.txt").write_text("ignored", encoding="utf-8")

    results, total, skips = process_json_files(str(tmp_path))

    assert total == 2
    assert skips == []
    assert {r["file"] for r in results} == {"a", "b"}
    assert all(r["page"] == "1" for r in results)


def test_process_json_files_reports_skips(tmp_path):
    (tmp_path / "good-1.json").write_text(json.dumps({"text": "OK"}), encoding="utf-8")
    (tmp_path / "bad-1.json").write_text("{broken", encoding="utf-8")

    results, total, skips = process_json_files(str(tmp_path))

    assert total == 2
    assert len(results) == 1
    assert len(skips) == 1
    assert skips[0][0].endswith("bad-1.json")


def test_main_writes_csv_with_alto_compatible_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "doc1-1.json").write_text(
        json.dumps({"lines": [{"text": "One"}, {"text": "Two"}, {"text": "Three"}]}), encoding="utf-8"
    )
    out_csv = tmp_path / "out.csv"

    main([str(tmp_path), "-o", str(out_csv)])

    content = out_csv.read_text(encoding="utf-8")
    header = content.splitlines()[0]
    for col in ["file", "page", "textlines", "illustrations", "graphics", "strings", "path"]:
        assert col in header
    assert "doc1,1,0,0,0,3" in content


def test_rows_are_in_page_reading_order(tmp_path, monkeypatch):
    """(#31 Phase 5) The extractors join a document's pages in CSV row order into
    `content.text`; the rows used to follow the directory listing (and, for ALTO, thread
    completion), so page 2 could come before page 1 and 10 before 2."""
    import alto_stats_create
    import json_stats_create

    doc = tmp_path / "doc"
    doc.mkdir()
    for n in (10, 2, 1, 11, 3):
        (doc / f"doc-{n}.json").write_text('{"text": "x"}', encoding="utf-8")
        (doc / f"doc-{n}.alto.xml").write_text("<alto/>", encoding="utf-8")
    rows, _total, _skips = json_stats_create.process_json_files(str(doc))
    assert [r["page"] for r in rows] == ["1", "2", "3", "10", "11"]

    monkeypatch.setattr(alto_stats_create, "run_alto_tools_stats", lambda _path: {"textlines": 1})
    rows, _total, _skips = alto_stats_create.process_alto_files_with_alto_tools(str(doc))
    assert [r["page"] for r in rows] == ["1", "2", "3", "10", "11"]
