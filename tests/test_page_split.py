"""
tests/test_page_split.py
========================
In-process tests for page_split.py (Phase 2 / hub issue #10).

The former subprocess smoke tests only proved the script compiled; these call
main(argv) and split_alto_xml directly, covering the actual splitting
behaviour, the hardened parser, and the paradata accounting — no child
processes, no ML dependencies.

All tests chdir into tmp_path because ParadataLogger writes to the relative
``paradata/`` directory.
"""

import xml.etree.ElementTree as ET

import pytest

from page_split import main, split_alto_xml

_ALTO_NS = "http://www.loc.gov/standards/alto/ns-v3#"

_TWO_PAGE_DOC = f"""<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="{_ALTO_NS}">
  <Description><MeasurementUnit>pixel</MeasurementUnit></Description>
  <Styles/>
  <Layout>
    <Page ID="P1" PHYSICAL_IMG_NR="7"><PrintSpace/></Page>
    <Page ID="P2" PHYSICAL_IMG_NR="8"><PrintSpace/></Page>
  </Layout>
</alto>
"""

_ENTITY_BOMB_DOC = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE alto [<!ENTITY x "boom">]>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
  <Layout><Page ID="P1">&x;</Page></Layout>
</alto>
"""


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Isolated cwd with input/output dirs; paradata/ lands in tmp."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    return tmp_path


# ── CLI surface (former subprocess smoke tests, now in-process) ──────────────


def test_cli_help_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_missing_args():
    """argparse rejects a call without the two positional directories."""
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code != 0


def test_cli_nonexistent_input_dir(workdir, capsys):
    result = main([str(workdir / "nope"), str(workdir / "out")])
    assert result is None
    assert "Input directory not found" in capsys.readouterr().out


# ── functional splitting through main() ──────────────────────────────────────


def test_main_splits_multipage_document(workdir, capsys):
    (workdir / "in" / "doc.alto.xml").write_text(_TWO_PAGE_DOC, encoding="utf-8")

    main([str(workdir / "in"), str(workdir / "out")])

    out_dir = workdir / "out" / "doc"
    produced = sorted(p.name for p in out_dir.iterdir())
    assert produced == ["doc-7.alto.xml", "doc-8.alto.xml"]  # PHYSICAL_IMG_NR naming
    assert "Found 2 page(s)" in capsys.readouterr().out

    for name, page_id in (("doc-7.alto.xml", "P1"), ("doc-8.alto.xml", "P2")):
        root = ET.parse(out_dir / name).getroot()
        pages = root.findall(f".//{{{_ALTO_NS}}}Page")
        assert [p.get("ID") for p in pages] == [page_id]  # exactly one page each
        assert root.find(f"{{{_ALTO_NS}}}Description") is not None  # header kept
        assert root.find(f"{{{_ALTO_NS}}}Styles") is not None


def test_main_ignores_non_xml_files(workdir, capsys):
    (workdir / "in" / "notes.txt").write_text("not xml", encoding="utf-8")
    (workdir / "in" / "doc.alto.xml").write_text(_TWO_PAGE_DOC, encoding="utf-8")

    main([str(workdir / "in"), str(workdir / "out")])

    out = capsys.readouterr().out
    assert "doc.alto.xml" in out
    assert "notes.txt" not in out


def test_main_survives_malformed_document(workdir, capsys):
    """A broken file is logged as a skip; the run continues to the next doc."""
    (workdir / "in" / "aaa_broken.xml").write_text("<alto>", encoding="utf-8")
    (workdir / "in" / "bbb_good.alto.xml").write_text(_TWO_PAGE_DOC, encoding="utf-8")

    main([str(workdir / "in"), str(workdir / "out")])

    assert (workdir / "out" / "bbb_good").is_dir()  # later doc still processed
    assert not (workdir / "out" / "aaa_broken").exists()


def test_main_writes_paradata_accounting(workdir):
    """(#10) documents counted as inputs, pages as xml outputs."""
    import json

    (workdir / "in" / "doc.alto.xml").write_text(_TWO_PAGE_DOC, encoding="utf-8")
    main([str(workdir / "in"), str(workdir / "out")])

    paradata_files = list((workdir / "paradata").glob("*.json"))
    assert len(paradata_files) == 1
    data = json.loads(paradata_files[0].read_text(encoding="utf-8"))
    stats = data["statistics"]
    assert stats["input_files_total"] == 1  # documents in
    assert stats["successfully_processed"] == 1
    assert stats["output_counts_by_type"]["xml"] == 2  # pages out


# ── split_alto_xml unit behaviour ────────────────────────────────────────────


def test_split_returns_page_count_and_falls_back_to_index(workdir):
    doc = _TWO_PAGE_DOC.replace(' PHYSICAL_IMG_NR="7"', "").replace(' PHYSICAL_IMG_NR="8"', "")
    src = workdir / "in" / "doc.alto.xml"
    src.write_text(doc, encoding="utf-8")

    count = split_alto_xml(str(src), str(workdir / "out"))

    assert count == 2
    produced = sorted(p.name for p in (workdir / "out" / "doc").iterdir())
    assert produced == ["doc-1.alto.xml", "doc-2.alto.xml"]  # 1-based index fallback


def test_split_pageless_document_returns_zero(workdir, capsys):
    src = workdir / "in" / "empty.xml"
    src.write_text(f'<alto xmlns="{_ALTO_NS}"><Layout/></alto>', encoding="utf-8")
    assert split_alto_xml(str(src), str(workdir / "out")) == 0
    assert "No <Page> elements" in capsys.readouterr().out


def test_split_rejects_doctype_declarations(workdir):
    """(#5) Fail closed on DOCTYPE: entity declarations can never reach the
    parser, so entity-expansion inputs are rejected rather than expanded."""
    src = workdir / "in" / "bomb.xml"
    src.write_text(_ENTITY_BOMB_DOC, encoding="utf-8")
    with pytest.raises(ET.ParseError, match="DOCTYPE"):
        split_alto_xml(str(src), str(workdir / "out"))
    assert not (workdir / "out" / "bomb").exists()  # nothing written


def test_split_preserves_root_attributes(workdir):
    doc = _TWO_PAGE_DOC.replace(f'xmlns="{_ALTO_NS}"', f'xmlns="{_ALTO_NS}" SCHEMAVERSION="3.1"')
    src = workdir / "in" / "doc.alto.xml"
    src.write_text(doc, encoding="utf-8")

    split_alto_xml(str(src), str(workdir / "out"))

    root = ET.parse(workdir / "out" / "doc" / "doc-7.alto.xml").getroot()
    assert root.get("SCHEMAVERSION") == "3.1"
