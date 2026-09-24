"""
tests/test_text_split.py — stage 1 of the text-lines method (#31), in-process.

Every test chdirs into tmp_path (ParadataLogger writes ./paradata) and points
text_split.CONFIG_PATH at a scratch config, so nothing touches the repo.
"""

import csv
import json
import os

import pytest

import text_split
from atrium_document import load_document
from tests.text_format_fixtures import compress_bytes, docx_bytes, make_zip, pdf_bytes, w_p, w_t, xlsx_bytes


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DOCUMENT_JSON_DIR", raising=False)
    monkeypatch.delenv("DOCUMENT_SOURCE_ORIGIN", raising=False)
    (tmp_path / "in").mkdir()
    return tmp_path


def _config(tmp_path, monkeypatch, *, json_dir="", origin="", strict=None, extra="", by_kind=""):
    lines = ["[DOCUMENT]", f"JSON_DIR = {json_dir}", f"SOURCE_ORIGIN = {origin}"]
    if by_kind:
        lines.append(f"SOURCE_ORIGIN_BY_KIND = {by_kind}")
    lines += ["", "[TEXT_INGEST]"]
    if strict is not None:
        lines.append(f"STRICT = {'true' if strict else 'false'}")
    lines.append(extra)
    path = tmp_path / "config.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    monkeypatch.setattr(text_split, "CONFIG_PATH", str(path))
    return path


def _report(out_dir):
    with open(os.path.join(out_dir, "ingest_report.csv"), encoding="utf-8", newline="") as fh:
        return {row["filename"]: row for row in csv.DictReader(fh)}


def _pages_report(out_dir):
    with open(os.path.join(out_dir, "pages_report.csv"), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_mixed_inputs_one_report_row_each_and_failures_isolated(workdir, monkeypatch):
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    (inp / "a.txt").write_text("Jedna\nDva\fTři\n", encoding="utf-8")
    (inp / "b.docx").write_bytes(docx_bytes(w_p(w_t("Odstavec"))))
    (inp / "c.pdf").write_bytes(b"%PDF-1.4 truncated")
    (inp / "d.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    (inp / "e.txt").write_text("\n\n   \n", encoding="utf-8")

    assert text_split.main([str(inp), str(out)]) == 0

    report = _report(out)
    assert {k: (v["status"], v["reason"]) for k, v in report.items()} == {
        "a.txt": ("ok", ""),
        "b.docx": ("ok", ""),
        "c.pdf": ("error", "corrupt"),
        "d.png": ("error", "image_needs_ocr"),
        "e.txt": ("error", "no_text"),
    }
    assert (out / "a" / "a-1.txt").read_text(encoding="utf-8") == "Jedna\nDva\n"
    assert (out / "a" / "a-2.txt").read_text(encoding="utf-8") == "Tři\n"
    assert (out / "b" / "b-1.txt").read_text(encoding="utf-8") == "Odstavec\n"
    assert not (out / "c").exists() and not (out / "e").exists()
    assert report["a.txt"]["pages"] == "2" and report["a.txt"]["lines"] == "3"
    assert report["a.txt"]["origin"] == "ocr:generic" and report["b.docx"]["origin"] == "digital-born-docx"


def test_pdf_pages_report_keeps_empty_pages_and_text_layers(workdir, monkeypatch):
    pytest.importorskip("pypdfium2")
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    (inp / "scan.pdf").write_bytes(pdf_bytes([["Text layer."], []], invisible=True))

    assert text_split.main([str(inp), str(out)]) == 0
    assert (out / "scan" / "scan-2.txt").read_text(encoding="utf-8") == ""  # page kept, numbering faithful
    rows = _pages_report(out)
    assert [(r["page"], r["text_layer"], r["needs_ocr_reason"]) for r in rows] == [
        ("1", "ocr", ""),
        ("2", "none", "no extractable text layer"),
    ]
    assert _report(out)["scan.pdf"]["origin"] == "ocr:pdf-text-layer"


def test_ignored_entries_are_reported_not_read(workdir, monkeypatch):
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    for name in (".hidden.txt", "~$lock.docx", ".~lock.x.odt#", "._res.txt", "Thumbs.db"):
        (inp / name).write_bytes(b"x")
    (inp / "sub").mkdir()
    (inp / "real.txt").write_text("text\n", encoding="utf-8")
    if hasattr(os, "symlink"):
        os.symlink(str(inp / "real.txt"), str(inp / "link.txt"))
    if hasattr(os, "mkfifo"):
        os.mkfifo(str(inp / "pipe"))

    assert text_split.main([str(inp), str(out)]) == 0
    report = _report(out)
    assert report["real.txt"]["status"] == "ok"
    ignored = {k for k, v in report.items() if v["status"] == "ignored"}
    assert {".hidden.txt", "~$lock.docx", ".~lock.x.odt#", "._res.txt", "Thumbs.db", "sub"} <= ignored
    if hasattr(os, "mkfifo"):
        assert report["pipe"]["reason"] == "not a regular file"
    if hasattr(os, "symlink"):
        assert report["link.txt"]["reason"] == "symbolic link (not followed)"


def test_doc_id_collisions_first_sorted_file_wins(workdir, monkeypatch):
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    (inp / "Report.txt").write_text("upper\n", encoding="utf-8")
    (inp / "report.md").write_text("lower\n", encoding="utf-8")
    (inp / "report.v2.pdf").write_bytes(b"%PDF-1.4")  # canonical_doc_id -> "report" too

    assert text_split.main([str(inp), str(out)]) == 0
    report = _report(out)
    assert report["Report.txt"]["status"] == "ok"
    assert report["report.md"]["reason"] == report["report.v2.pdf"]["reason"] == "doc_id_collision"
    assert (out / "Report" / "Report-1.txt").read_text(encoding="utf-8") == "upper\n"


def test_strict_exit_code_from_flag_and_config(workdir, monkeypatch):
    inp, out = workdir / "in", workdir / "out"
    (inp / "ok.txt").write_text("x\n", encoding="utf-8")
    (inp / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    _config(workdir, monkeypatch)
    assert text_split.main([str(inp), str(out)]) == 0
    assert text_split.main([str(inp), str(out), "--strict"]) == 1
    _config(workdir, monkeypatch, strict=True)
    assert text_split.main([str(inp), str(out)]) == 1


def test_rerun_replaces_a_document_by_renaming_it_aside(workdir, monkeypatch):
    """A re-run with a shorter version leaves no stale page 3, no staging and no set-aside dir."""
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    (inp / "d.txt").write_text("1\f2\f3\n", encoding="utf-8")
    text_split.main([str(inp), str(out)])
    assert sorted(os.listdir(out / "d")) == ["d-1.txt", "d-2.txt", "d-3.txt"]
    (inp / "d.txt").write_text("1\f2\n", encoding="utf-8")
    text_split.main([str(inp), str(out)])
    assert sorted(os.listdir(out / "d")) == ["d-1.txt", "d-2.txt"]
    assert not [n for n in os.listdir(out) if n.startswith((".tmp-", ".old-"))]


def test_invalid_config_and_missing_input_dir(workdir, monkeypatch):
    _config(workdir, monkeypatch, extra="MAX_PAGES = many")
    assert text_split.main([str(workdir / "in"), str(workdir / "out")]) == 2
    _config(workdir, monkeypatch)
    assert text_split.main([str(workdir / "nope"), str(workdir / "out")]) == 1


def test_bad_cli_arguments_exit_2(workdir):
    with pytest.raises(SystemExit) as info:
        text_split.main(["only-one-arg"])
    assert info.value.code == 2


@pytest.mark.parametrize(
    "doc_id,ok",
    [("CTX1", True), ("", False), ("..", False), ("a\x01b", False), ("x" * 201, False), ("č" * 100, True)],
)
def test_validate_doc_id(doc_id, ok):
    assert (text_split.validate_doc_id(doc_id) is None) is ok


# ── document record: truthful per-class origin ────────────────────────────────


def test_document_record_origin_and_page_count_per_class(workdir, monkeypatch):
    pytest.importorskip("jsonschema")
    docs = workdir / "docs"
    docs.mkdir()
    _config(workdir, monkeypatch, json_dir=str(docs))
    inp, out = workdir / "in", workdir / "out"
    (inp / "t.txt").write_text("text\n", encoding="utf-8")
    (inp / "w.docx").write_bytes(docx_bytes(w_p(w_t("a"), '<w:r><w:br w:type="page"/></w:r>') + w_p(w_t("b"))))
    (inp / "p.xml").write_text(
        '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"><Page>'
        '<TextRegion id="r"><TextLine id="l"><TextEquiv><Unicode>ocr</Unicode></TextEquiv></TextLine></TextRegion>'
        "</Page></PcGts>",
        encoding="utf-8",
    )

    assert text_split.main([str(inp), str(out)]) == 0
    t = load_document(str(docs / "t.document.json"))["source"]
    w = load_document(str(docs / "w.document.json"))["source"]
    p = load_document(str(docs / "p.document.json"))["source"]
    assert (t["origin"], t["media_type"]) == ("ocr:generic", "text/plain")
    assert "page_count" not in t
    # DOCX: born-digital, and no invented page_count (llm-enrich records DOCX as one page).
    assert w["origin"] == "digital-born-docx" and "page_count" not in w
    assert (p["origin"], p["page_count"]) == ("ocr:page-xml", 1)
    assert "born-digital origin" in _report(out)["w.docx"]["notes"]


def test_source_origin_override_precedence(workdir, monkeypatch):
    docs = workdir / "docs"
    docs.mkdir()
    _config(workdir, monkeypatch, json_dir=str(docs), origin="ocr:from-config")
    inp, out = workdir / "in", workdir / "out"
    (inp / "a.docx").write_bytes(docx_bytes(w_p(w_t("x"))))
    text_split.main([str(inp), str(out), "--source-origin", "ocr:tesseract"])
    assert load_document(str(docs / "a.document.json"))["source"]["origin"] == "ocr:tesseract"

    (docs / "a.document.json").unlink()
    text_split.main([str(inp), str(out)])
    assert load_document(str(docs / "a.document.json"))["source"]["origin"] == "ocr:from-config"


def test_paradata_records_the_pdf_component(workdir, monkeypatch):
    pytest.importorskip("pypdfium2")
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    (inp / "d.pdf").write_bytes(pdf_bytes([["x y z"]]))
    text_split.main([str(inp), str(out)])
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in (workdir / "paradata").glob("*.json")]
    assert any("pypdfium2" in json.dumps(log) for log in logs)


# ── (#31 Phase 4) output safety, partial reads, strictness, origins ───────────


def test_a_directory_text_split_did_not_write_is_never_replaced(workdir, monkeypatch):
    """Output dir `.` and an input `setup.txt` used to rmtree ./setup."""
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    (out / "setup").mkdir(parents=True)
    (out / "setup" / "config.txt").write_text("precious", encoding="utf-8")
    (inp / "setup.txt").write_text("text\n", encoding="utf-8")
    assert text_split.main([str(inp), str(out)]) == 0
    row = _report(out)["setup.txt"]
    assert (row["status"], row["reason"]) == ("error", "output_failed") and "did not write" in row["notes"]
    assert (out / "setup" / "config.txt").read_text(encoding="utf-8") == "precious"
    assert not [n for n in os.listdir(out) if n.startswith((".tmp-", ".old-"))]


def test_a_symlinked_page_dir_is_refused(workdir, monkeypatch):
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    elsewhere = workdir / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "d-1.txt").write_text("keep", encoding="utf-8")
    out.mkdir()
    os.symlink(elsewhere, out / "d")
    (inp / "d.txt").write_text("text\n", encoding="utf-8")
    text_split.main([str(inp), str(out)])
    assert _report(out)["d.txt"]["reason"] == "output_failed"
    assert (elsewhere / "d-1.txt").read_text(encoding="utf-8") == "keep"


def test_a_failed_rerun_removes_the_documents_stale_pages(workdir, monkeypatch):
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    (inp / "d.txt").write_text("1\f2\n", encoding="utf-8")
    text_split.main([str(inp), str(out)])
    assert (out / "d" / "d-2.txt").exists()
    (inp / "d.txt").write_bytes(bytes(range(256)) * 8)  # now unreadable
    text_split.main([str(inp), str(out)])
    row = _report(out)["d.txt"]
    assert (row["status"], row["reason"]) == ("error", "binary_content") and "stale_pages_removed" in row["notes"]
    assert not (out / "d").exists()


def test_the_record_is_written_before_the_pages_are_swapped_in(workdir, monkeypatch):
    _config(workdir, monkeypatch, json_dir=str(workdir / "docs"))
    inp, out = workdir / "in", workdir / "out"
    (inp / "d.txt").write_text("text\n", encoding="utf-8")

    def boom(*a, **k):
        raise RuntimeError("record write failed")

    monkeypatch.setattr(text_split.document_hook, "write_document_block", boom)
    assert text_split.main([str(inp), str(out)]) == 0
    row = _report(out)["d.txt"]
    assert (row["status"], row["reason"]) == ("error", "output_failed")
    assert not (out / "d").exists() and not [n for n in os.listdir(out) if n.startswith((".tmp-", ".old-"))]
    assert _pages_report(out) == []


def test_a_lossy_read_is_partial_and_counts_for_strict(workdir, monkeypatch):
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    (inp / "r.jsonl").write_text('{"text": "dobrý"}\n{broken\n{"text": "den"}\n', encoding="utf-8")
    assert text_split.main([str(inp), str(out)]) == 0
    row = _report(out)["r.jsonl"]
    assert (row["status"], row["reason"]) == ("partial", "jsonl_bad_records")
    assert (out / "r" / "r-1.txt").exists()  # partial documents are processed downstream
    assert text_split.main([str(inp), str(out), "--strict"]) == 1


def test_no_strict_overrides_the_config_and_strict_must_be_a_boolean(workdir, monkeypatch):
    inp, out = workdir / "in", workdir / "out"
    (inp / "bad.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    _config(workdir, monkeypatch, strict=True)
    assert text_split.main([str(inp), str(out)]) == 1
    assert text_split.main([str(inp), str(out), "--no-strict"]) == 0
    _config(workdir, monkeypatch, extra="STRICT = ture")
    assert text_split.main([str(inp), str(out)]) == 2


def test_source_origin_by_kind_and_its_precedence(workdir, monkeypatch):
    docs = workdir / "docs"
    docs.mkdir()
    _config(workdir, monkeypatch, json_dir=str(docs), origin="ocr:from-config", by_kind="xlsx = ocr:generic")
    inp, out = workdir / "in", workdir / "out"
    (inp / "s.xlsx").write_bytes(xlsx_bytes([("S", [["buňka"]])]))
    (inp / "w.docx").write_bytes(docx_bytes(w_p(w_t("x"))))

    def origin(name):
        return load_document(str(docs / f"{name}.document.json"))["source"]["origin"]

    text_split.main([str(inp), str(out)])
    assert origin("s") == "ocr:generic"  # the per-kind entry beats [DOCUMENT].SOURCE_ORIGIN
    assert origin("w") == "ocr:from-config"
    for f in docs.iterdir():
        f.unlink()
    monkeypatch.setenv("DOCUMENT_SOURCE_ORIGIN", "ocr:from-env")
    text_split.main([str(inp), str(out)])
    assert origin("s") == "ocr:from-env"  # a per-run statement beats every config key
    for f in docs.iterdir():
        f.unlink()
    text_split.main([str(inp), str(out), "--source-origin", "ocr:from-cli"])
    assert origin("s") == "ocr:from-cli"


@pytest.mark.parametrize("value", ["xls = ocr:generic", "xlsx ocr:generic", "xlsx = nonsense", "xlsx=a:b, xlsx=ocr:x"])
def test_a_bad_source_origin_by_kind_exits_2(workdir, monkeypatch, value):
    _config(workdir, monkeypatch, by_kind=value)
    assert text_split.main([str(workdir / "in"), str(workdir / "out")]) == 2


def test_the_born_digital_note_needs_a_record_and_names_the_override(workdir, monkeypatch):
    inp, out = workdir / "in", workdir / "out"
    (inp / "s.xlsx").write_bytes(xlsx_bytes([("S", [["buňka"]])]))
    (inp / "w.docx").write_bytes(docx_bytes(w_p(w_t("x"))))
    _config(workdir, monkeypatch)
    text_split.main([str(inp), str(out)])
    assert "born-digital" not in _report(out)["s.xlsx"]["notes"]  # no record, nothing held back
    _config(workdir, monkeypatch, json_dir=str(workdir / "docs"))
    text_split.main([str(inp), str(out)])
    report = _report(out)
    assert "SOURCE_ORIGIN_BY_KIND" in report["s.xlsx"]["notes"]
    assert (
        "born-digital origin" in report["w.docx"]["notes"] and "SOURCE_ORIGIN_BY_KIND" not in report["w.docx"]["notes"]
    )


def test_paradata_records_the_reader_options(workdir, monkeypatch):
    _config(workdir, monkeypatch, extra="PAGE_BREAKS = explicit")
    inp, out = workdir / "in", workdir / "out"
    (inp / "a.txt").write_text("x\n", encoding="utf-8")
    text_split.main([str(inp), str(out)])
    logs = [json.loads(p.read_text(encoding="utf-8")) for p in (workdir / "paradata").glob("*text*split*.json")]
    logs = logs or [json.loads(p.read_text(encoding="utf-8")) for p in (workdir / "paradata").glob("*.json")]
    assert any('"page_breaks": "explicit"' in json.dumps(log) for log in logs)


def test_compressed_files_and_zip_bundles_are_one_document_each(workdir, monkeypatch):
    _config(workdir, monkeypatch)
    inp, out = workdir / "in", workdir / "out"
    (inp / "g.txt.gz").write_bytes(compress_bytes("Komprimovaný\ntext\n"))
    (inp / "b.zip").write_bytes(make_zip([("p/1.txt", "strana jedna"), ("p/2.txt", "strana dvě")]))
    assert text_split.main([str(inp), str(out)]) == 0
    report = _report(out)
    assert (report["g.txt.gz"]["doc_id"], report["g.txt.gz"]["kind"]) == ("g", "txt")
    assert (report["b.zip"]["kind"], report["b.zip"]["pages"]) == ("zip-bundle", "2")
    assert (out / "b" / "b-2.txt").read_text(encoding="utf-8") == "strana dvě\n"
    assert [r["page_label"] for r in _pages_report(out) if r["file"] == "b"] == ["1", "2"]
