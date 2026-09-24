"""
tests/test_text_stats_create.py — stage 2 of the text-lines method (#31).
"""

import csv

import text_stats_create


def _write(path, text, encoding="utf-8"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode(encoding))
    return path


def test_rows_columns_counts_and_hyphenated_ids(tmp_path):
    root = tmp_path / "PAGE_TEXT"
    _write(root / "my-doc" / "my-doc-2.txt", "three words here\n\nand two\n")
    _write(root / "my-doc" / "my-doc-10.txt", "x\n")
    _write(root / "0001" / "0001-1.txt", "číslo\n", encoding="cp1250")  # decoded, not rejected
    _write(root / "loose-3.txt", "root level\n")

    rows, skipped = text_stats_create.process_text_files(str(root))
    assert skipped == []
    assert [(r["file"], r["page"], r["textlines"], r["strings"]) for r in rows] == [
        ("0001", 1, 1, 1),
        ("loose", 3, 1, 2),
        ("my-doc", 2, 2, 5),
        ("my-doc", 10, 1, 1),  # numeric page order, not "10" < "2"
    ]
    assert all(r["illustrations"] == 0 and r["graphics"] == 0 for r in rows)


def test_skips_unaddressable_names_and_hidden_staging_dirs(tmp_path):
    root = tmp_path / "PAGE_TEXT"
    _write(root / "doc" / "notes.txt", "no page suffix\n")
    _write(root / ".tmp-doc" / "doc-1.txt", "staging\n")
    _write(root / "doc" / "doc-1.txt", "ok\n")
    _write(root / "doc" / "doc-1.csv", "not text\n")
    rows, skipped = text_stats_create.process_text_files(str(root))
    assert [(r["file"], r["page"]) for r in rows] == [("doc", 1)]
    assert len(skipped) == 1 and skipped[0][0].endswith("notes.txt")


def test_illustrations_come_from_the_pages_report(tmp_path):
    root = tmp_path / "PAGE_TEXT"
    _write(root / "d" / "d-1.txt", "x\n")
    _write(
        root / "pages_report.csv",
        "file,page,page_label,text_layer,needs_ocr_reason,lines,images,flags\nd,1,i,ocr,,1,2,\n",
    )
    rows, _ = text_stats_create.process_text_files(str(root))
    assert rows[0]["illustrations"] == 2


def test_main_writes_the_seven_column_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "PAGE_TEXT"
    _write(root / "d" / "d-1.txt", "x\n")
    out = tmp_path / "out" / "stats.csv"
    assert text_stats_create.main([str(root), "-o", str(out)]) == 0
    with open(out, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == ["file", "page", "textlines", "illustrations", "graphics", "strings", "path"]
        assert [r["file"] for r in reader] == ["d"]
    assert text_stats_create.main([str(tmp_path / "missing"), "-o", str(out)]) == 1


# ── (#31 Phase 4) configured encodings, damaged reports, unreadable folders ───


def test_fallback_encodings_come_from_the_config(tmp_path, monkeypatch):
    root = tmp_path / "PAGE_TEXT"
    _write(root / "d" / "d-1.txt", "Grüße aus Köln\n", encoding="cp1252")
    config = tmp_path / "config.txt"
    config.write_text("[TEXT_INGEST]\nFALLBACK_ENCODINGS = cp1252\n", encoding="utf-8")
    monkeypatch.setattr(text_stats_create, "CONFIG_PATH", str(config))
    seen = []
    real = text_stats_create.decode_bytes

    def spy(data, fallbacks=()):
        seen.append(tuple(fallbacks))
        return real(data, fallbacks)

    monkeypatch.setattr(text_stats_create, "decode_bytes", spy)
    monkeypatch.chdir(tmp_path)
    assert text_stats_create.main([str(root), "-o", str(tmp_path / "s.csv")]) == 0
    assert seen == [("cp1252",)]


def test_a_damaged_pages_report_does_not_stop_the_stage(tmp_path, monkeypatch):
    root = tmp_path / "PAGE_TEXT"
    _write(root / "d" / "d-1.txt", "x\n")
    (root / "pages_report.csv").write_bytes(b"\xff\xfe\x00\x01broken")
    monkeypatch.chdir(tmp_path)
    assert text_stats_create.main([str(root), "-o", str(tmp_path / "s.csv")]) == 0
    with open(tmp_path / "s.csv", encoding="utf-8", newline="") as fh:
        assert [r["illustrations"] for r in csv.DictReader(fh)] == ["0"]


def test_an_unreadable_subdirectory_is_skipped_and_reported(tmp_path, monkeypatch):
    root = tmp_path / "PAGE_TEXT"
    _write(root / "good" / "good-1.txt", "x\n")
    _write(root / "bad" / "bad-1.txt", "y\n")
    real = text_stats_create.os.scandir

    def scandir(path):
        if str(path).endswith("bad"):
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(text_stats_create.os, "scandir", scandir)
    rows, skipped = text_stats_create.process_text_files(str(root))
    assert [r["file"] for r in rows] == ["good"]
    assert len(skipped) == 1 and "unreadable directory" in skipped[0][1]


def test_an_invalid_config_exits_2(tmp_path, monkeypatch):
    root = tmp_path / "PAGE_TEXT"
    root.mkdir()
    config = tmp_path / "config.txt"
    config.write_text("[TEXT_INGEST]\nFALLBACK_ENCODINGS = no-such-codec\n", encoding="utf-8")
    monkeypatch.setattr(text_stats_create, "CONFIG_PATH", str(config))
    monkeypatch.chdir(tmp_path)
    assert text_stats_create.main([str(root)]) == 2
