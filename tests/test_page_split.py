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

import hashlib
import json
import xml.etree.ElementTree as ET

import pytest

import document_hook
from atrium_document import load_document, resolve_originator
from page_split import (
    _doc_id_from_filename,
    main,
    resolve_source_origin,
    split_alto_xml,
    split_json_document,
)

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


# ── (#31 Phase 5) every ALTO version; non-ALTO roots; stale pages ───────────


@pytest.mark.parametrize(
    "ns",
    [
        "http://www.loc.gov/standards/alto/ns-v2#",
        "http://www.loc.gov/standards/alto/ns-v4#",
        "http://bibnum.bnf.fr/ns/alto_prod",
    ],
)
def test_split_reads_the_namespace_from_the_root(workdir, ns):
    """page_split used to know only the v3 namespace: a v2/v4 file printed "No <Page>
    elements found" and got no pages. The page files keep the document's namespace, and
    their <Layout> is in it too (a bare Layout re-parsed as ALTO only for v3)."""
    src = workdir / "in" / "doc.alto.xml"
    src.write_text(_TWO_PAGE_DOC.replace(_ALTO_NS, ns), encoding="utf-8")

    assert split_alto_xml(str(src), str(workdir / "out")) == 2

    for name, page_id in (("doc-7.alto.xml", "P1"), ("doc-8.alto.xml", "P2")):
        root = ET.parse(workdir / "out" / "doc" / name).getroot()
        assert root.tag == f"{{{ns}}}alto"
        layout = root.find(f"{{{ns}}}Layout")
        assert layout is not None
        assert [p.get("ID") for p in layout.findall(f"{{{ns}}}Page")] == [page_id]
        assert root.find(f"{{{ns}}}Description") is not None


def test_split_v3_after_another_version_keeps_the_v3_default_namespace(workdir):
    """register_namespace() is process-global: a v4 document split first must not
    change how a later v3 document is written."""
    v4 = "http://www.loc.gov/standards/alto/ns-v4#"
    (workdir / "in" / "a.alto.xml").write_text(_TWO_PAGE_DOC.replace(_ALTO_NS, v4), encoding="utf-8")
    (workdir / "in" / "b.alto.xml").write_text(_TWO_PAGE_DOC, encoding="utf-8")
    split_alto_xml(str(workdir / "in" / "a.alto.xml"), str(workdir / "out"))
    split_alto_xml(str(workdir / "in" / "b.alto.xml"), str(workdir / "out"))

    text = (workdir / "out" / "b" / "b-7.alto.xml").read_text(encoding="utf-8")
    assert f'<alto xmlns="{_ALTO_NS}">' in text
    assert "<Layout>" in text and "ns0:" not in text


@pytest.mark.parametrize(
    "doc,root_name",
    [
        (
            '<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15">'
            '<Page imageFilename="a.jpg"/></PcGts>',
            "PcGts",
        ),
        ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><text/></TEI>', "TEI"),
        ('<alto><Layout><Page ID="P1"/></Layout></alto>', "alto"),
    ],
)
def test_split_skips_a_root_that_is_not_namespaced_alto(workdir, capsys, doc, root_name):
    """PAGE XML has <Page> elements too: with the namespace taken from the root it would
    otherwise be split as if it were ALTO. The text-lines method reads all of these."""
    src = workdir / "in" / "other.xml"
    src.write_text(doc, encoding="utf-8")
    assert split_alto_xml(str(src), str(workdir / "out")) == 0
    out = capsys.readouterr().out
    assert f"Not a namespaced ALTO document (root <{root_name}>)" in out
    assert "--method text-lines" in out
    assert not (workdir / "out" / "other").exists()


def test_resplit_removes_pages_the_input_no_longer_has(workdir):
    """A re-split with fewer pages used to leave the old extra page files behind, and the
    stats stage picked them up. Only `<doc>-<page>.alto.xml` files of this document go."""
    src = workdir / "in" / "doc.alto.xml"
    src.write_text(_TWO_PAGE_DOC, encoding="utf-8")
    split_alto_xml(str(src), str(workdir / "out"))
    doc_dir = workdir / "out" / "doc"
    (doc_dir / "notes.txt").write_text("keep", encoding="utf-8")
    (doc_dir / "other-1.alto.xml").write_text("keep", encoding="utf-8")
    (doc_dir / "doc-sub.alto.xml").mkdir()

    src.write_text(
        _TWO_PAGE_DOC.replace('<Page ID="P2" PHYSICAL_IMG_NR="8"><PrintSpace/></Page>', ""), encoding="utf-8"
    )
    assert split_alto_xml(str(src), str(workdir / "out")) == 1

    assert sorted(p.name for p in doc_dir.iterdir()) == [
        "doc-7.alto.xml",
        "doc-sub.alto.xml",
        "notes.txt",
        "other-1.alto.xml",
    ]


def test_main_removes_old_pages_of_a_document_that_now_fails(workdir, capsys):
    """An input that now fails (or yields no pages) must not leave the pages of an earlier
    run for the stats stage — text_split's `stale_pages_removed`, here too."""
    src = workdir / "in" / "doc.alto.xml"
    src.write_text(_TWO_PAGE_DOC, encoding="utf-8")
    main([str(workdir / "in"), str(workdir / "out")])
    assert len(list((workdir / "out" / "doc").glob("doc-*.alto.xml"))) == 2

    src.write_text("<alto>", encoding="utf-8")  # damaged
    main([str(workdir / "in"), str(workdir / "out")])

    assert list((workdir / "out" / "doc").glob("doc-*.alto.xml")) == []
    out = capsys.readouterr().out
    assert "Failed:" in out
    assert "Removed 2 page file(s) left by an earlier run of 'doc'" in out


def test_resplit_leaves_unchanged_page_files_untouched(workdir):
    """(#31 Phase 5) Extract and classify resume by file time: a re-split of an unchanged
    document must not make every page look new (it used to rewrite them all)."""
    import os

    src = workdir / "in" / "doc.alto.xml"
    src.write_text(_TWO_PAGE_DOC, encoding="utf-8")
    split_alto_xml(str(src), str(workdir / "out"))
    page = workdir / "out" / "doc" / "doc-7.alto.xml"
    st = os.stat(page)
    os.utime(page, ns=(st.st_atime_ns - 10**12, st.st_mtime_ns - 10**12))
    before = os.stat(page).st_mtime_ns

    split_alto_xml(str(src), str(workdir / "out"))
    assert os.stat(page).st_mtime_ns == before

    src.write_text(_TWO_PAGE_DOC.replace('<Page ID="P1" PHYSICAL_IMG_NR="7">', '<Page ID="P1b" PHYSICAL_IMG_NR="7">'))
    split_alto_xml(str(src), str(workdir / "out"))
    assert os.stat(page).st_mtime_ns > before


def test_resplit_json_removes_pages_the_input_no_longer_has(workdir):
    src = workdir / "in" / "doc.json"
    src.write_text(json.dumps({"pages": [{"text": "a"}, {"text": "b"}, {"text": "c"}]}), encoding="utf-8")
    assert split_json_document(str(src), str(workdir / "out")) == 3
    src.write_text(json.dumps({"text": "single page now"}), encoding="utf-8")
    assert split_json_document(str(src), str(workdir / "out")) == 1
    assert sorted(p.name for p in (workdir / "out" / "doc").iterdir()) == ["doc-1.json"]


# ── (#31) split_json_document unit behaviour ─────────────────────────────────


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_split_json_family_a_nested_pages_with_page_number_field(workdir):
    """Family A: top-level 'pages' list, each item carrying a page-number
    field (Azure/docTR shape). Sibling top-level keys are the JSON 'header'
    and are preserved on every output page."""
    doc = {
        "metadata": {"engine": "test"},
        "pages": [
            {"pageNumber": 1, "text": "hello"},
            {"pageNumber": 2, "text": "world"},
        ],
    }
    src = workdir / "in" / "doc.json"
    src.write_text(json.dumps(doc), encoding="utf-8")

    count = split_json_document(str(src), str(workdir / "out"))

    assert count == 2
    out_dir = workdir / "out" / "doc"
    produced = sorted(p.name for p in out_dir.iterdir())
    assert produced == ["doc-1.json", "doc-2.json"]

    page1 = _read_json(out_dir / "doc-1.json")
    assert page1["metadata"] == {"engine": "test"}  # header preserved
    assert page1["pages"] == {"pageNumber": 1, "text": "hello"}  # single page object, not a list

    page2 = _read_json(out_dir / "doc-2.json")
    assert page2["pages"] == {"pageNumber": 2, "text": "world"}


def test_split_json_family_a_falls_back_to_index_without_page_number_field(workdir):
    """Family A without any PAGE_NUMBER_FIELD_KEYS field falls back to a
    1-based index, mirroring ALTO's PHYSICAL_IMG_NR-or-index fallback."""
    doc = {"pages": [{"text": "hello"}, {"text": "world"}]}
    src = workdir / "in" / "doc.json"
    src.write_text(json.dumps(doc), encoding="utf-8")

    count = split_json_document(str(src), str(workdir / "out"))

    assert count == 2
    produced = sorted(p.name for p in (workdir / "out" / "doc").iterdir())
    assert produced == ["doc-1.json", "doc-2.json"]


def test_split_json_family_a_one_level_nested(workdir):
    """(D5) A page-list container one level under a dict-valued top-level key
    (e.g. Azure's analyzeResult.pages) is detected; sibling keys at both
    levels are preserved."""
    doc = {
        "status": "succeeded",
        "analyzeResult": {
            "modelId": "prebuilt-layout",
            "pages": [{"pageNumber": 1, "angle": 0.1}, {"pageNumber": 2, "angle": -0.2}],
        },
    }
    src = workdir / "in" / "doc.json"
    src.write_text(json.dumps(doc), encoding="utf-8")

    count = split_json_document(str(src), str(workdir / "out"))

    assert count == 2
    page1 = _read_json(workdir / "out" / "doc" / "doc-1.json")
    assert page1["status"] == "succeeded"  # top-level sibling preserved
    assert page1["analyzeResult"]["modelId"] == "prebuilt-layout"  # nested sibling preserved
    assert page1["analyzeResult"]["pages"] == {"pageNumber": 1, "angle": 0.1}


def test_split_json_family_b_tagged_flat_list(workdir):
    """Family B: a flat list tagged with a per-item page field and >1
    distinct value (AWS Textract shape) is grouped by that value."""
    doc = {
        "Blocks": [
            {"BlockType": "PAGE", "Page": 1},
            {"BlockType": "WORD", "Page": 1, "Text": "hello"},
            {"BlockType": "PAGE", "Page": 2},
            {"BlockType": "WORD", "Page": 2, "Text": "world"},
        ],
        "DocumentMetadata": {"Pages": 2},
    }
    src = workdir / "in" / "doc.json"
    src.write_text(json.dumps(doc), encoding="utf-8")

    count = split_json_document(str(src), str(workdir / "out"))

    assert count == 2
    out_dir = workdir / "out" / "doc"
    produced = sorted(p.name for p in out_dir.iterdir())
    assert produced == ["doc-1.json", "doc-2.json"]

    page1 = _read_json(out_dir / "doc-1.json")
    assert page1["DocumentMetadata"] == {"Pages": 2}  # sibling preserved
    assert [b["BlockType"] for b in page1["Blocks"]] == ["PAGE", "WORD"]
    assert all(b["Page"] == 1 for b in page1["Blocks"])


def test_split_json_family_b_single_value_falls_through_to_fallback(workdir):
    """(D6) A flat tagged list with only ONE distinct value is not evidence
    of a multi-page document — falls through to the Family C fallback."""
    doc = {"items": [{"page": 1, "text": "a"}, {"page": 1, "text": "b"}]}
    src = workdir / "in" / "doc.json"
    src.write_text(json.dumps(doc), encoding="utf-8")

    count = split_json_document(str(src), str(workdir / "out"))

    assert count == 1
    out_dir = workdir / "out" / "doc"
    produced = sorted(p.name for p in out_dir.iterdir())
    assert produced == ["doc-1.json"]
    assert _read_json(out_dir / "doc-1.json") == doc  # unchanged


def test_split_json_family_c_fallback_unchanged(workdir):
    """Family C (D4): today's exact single-page fixtures still produce
    exactly one unchanged <base>-1.json — the pipeline's prior behaviour."""
    doc = {"page": {"lines": [{"textline": "Hello World"}, {"textline": "Second line"}]}}
    src = workdir / "in" / "doc1.json"
    src.write_text(json.dumps(doc), encoding="utf-8")

    count = split_json_document(str(src), str(workdir / "out"))

    assert count == 1
    out_dir = workdir / "out" / "doc1"
    produced = sorted(p.name for p in out_dir.iterdir())
    assert produced == ["doc1-1.json"]
    assert _read_json(out_dir / "doc1-1.json") == doc


def test_split_json_empty_list_falls_through_to_fallback(workdir):
    """An empty 'pages' list matches neither Family A nor B (_is_dict_list
    requires a non-empty list) — falls through to the Family C fallback."""
    doc = {"pages": []}
    src = workdir / "in" / "doc.json"
    src.write_text(json.dumps(doc), encoding="utf-8")

    count = split_json_document(str(src), str(workdir / "out"))

    assert count == 1
    assert _read_json(workdir / "out" / "doc" / "doc-1.json") == doc


def test_main_dispatches_json_files_to_split_json_document(workdir):
    """main() discovers *.json alongside *.xml and dispatches each by
    extension."""
    doc = {"pages": [{"pageNumber": 1, "text": "a"}, {"pageNumber": 2, "text": "b"}]}
    (workdir / "in" / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

    main([str(workdir / "in"), str(workdir / "out")])

    out_dir = workdir / "out" / "doc"
    produced = sorted(p.name for p in out_dir.iterdir())
    assert produced == ["doc-1.json", "doc-2.json"]


def test_main_survives_malformed_json_document(workdir):
    """(#31) A broken JSON file is logged as a skip; the run continues to the
    next document — mirrors test_main_survives_malformed_document for ALTO."""
    (workdir / "in" / "aaa_broken.json").write_text("{not valid json", encoding="utf-8")
    (workdir / "in" / "bbb_good.json").write_text(json.dumps({"text": "OK"}), encoding="utf-8")

    main([str(workdir / "in"), str(workdir / "out")])

    assert (workdir / "out" / "bbb_good").is_dir()  # later doc still processed
    assert not (workdir / "out" / "aaa_broken").exists()


# ── (atrium-project#10) the document-json path, which had zero coverage here ──
#
# This file covered splitting and paradata only, so neither D5 (source.origin never
# written, leaving the §1a mixed-plane guard permanently deferred) nor D6 (the
# ParadataLogger OBJECT passed where a run_id string belongs) was visible from any
# test. tests/test_document_hook.py exercises only the already-correct string form,
# which is exactly why D6 went unnoticed.


class _FakeParadataLogger:
    """Only the three attributes document_hook.paradata_ref_for() reads."""

    def __init__(self, run_id, program="alto-postprocess", paradata_dir="paradata"):
        self.run_id = run_id
        self.program = program
        self.paradata_dir = paradata_dir


@pytest.fixture
def docdir(workdir, monkeypatch):
    """Enable the accretion hook for main() the way an operator does — through the
    DOCUMENT_JSON_DIR env override rather than by editing setup/config.txt."""
    target = workdir / "docjson"
    target.mkdir()
    monkeypatch.setenv("DOCUMENT_JSON_DIR", str(target))
    monkeypatch.delenv("DOCUMENT_SOURCE_ORIGIN", raising=False)
    return target


def test_doc_id_from_filename_delegates_to_canonical_doc_id():
    """(D3) One derivation, and it must not fork on a multi-dot name. The old
    `splitext()` + `.replace(".alto", "")` agreed with canonical_doc_id() on the
    conventional names only, which is what made the fork invisible."""
    assert _doc_id_from_filename("CTX000000001.alto.xml", "xml") == "CTX000000001"
    assert _doc_id_from_filename("CTX000000001.xml", "xml") == "CTX000000001"
    assert _doc_id_from_filename("CTX000000001.json", "json") == "CTX000000001"
    assert _doc_id_from_filename("CTX000000001.document.json", "json") == "CTX000000001"
    # A doc_id that legitimately contains dots keeps them — `split(".")[0]` did not.
    assert _doc_id_from_filename("sbn.2019-1.alto.xml", "xml") == "sbn.2019-1"
    # Format-agnostic: the same file resolves identically whatever the caller claims.
    assert _doc_id_from_filename("CTX1.alto.xml", "json") == _doc_id_from_filename("CTX1.alto.xml", "xml")


def test_resolve_source_origin_defaults_per_format_resolve_to_this_repo(monkeypatch):
    """(D5) The value is only useful if ORIGIN_ORIGINATORS maps it back to this repo:
    an unmatched prefix makes _assert_origin_consistent() abstain in silence, which is
    the state that had the guard switched off."""
    monkeypatch.delenv("DOCUMENT_SOURCE_ORIGIN", raising=False)
    assert resolve_originator(resolve_source_origin("xml")) == "alto-postprocess"
    assert resolve_originator(resolve_source_origin("json")) == "alto-postprocess"
    assert resolve_source_origin("xml") == "ABBYY-ALTO"
    assert resolve_source_origin("json").startswith("ocr:")


def test_resolve_source_origin_precedence(monkeypatch):
    """CLI flag > env var > config value > per-format default, as everywhere else."""
    monkeypatch.setenv("DOCUMENT_SOURCE_ORIGIN", "ocr:from-env")
    assert resolve_source_origin("xml", "ocr:from-config") == "ocr:from-env"
    assert resolve_source_origin("xml", "ocr:from-config", "ocr:from-cli") == "ocr:from-cli"
    monkeypatch.delenv("DOCUMENT_SOURCE_ORIGIN", raising=False)
    assert resolve_source_origin("xml", "ocr:from-config") == "ocr:from-config"


def test_resolve_source_origin_warns_when_the_override_disables_the_guard(monkeypatch, capsys):
    """A misspelled or foreign origin must not fail quietly: `docx` resolves to
    digital-convert, which would make §1a REFUSE every write this repo makes, and
    `nonsense` matches nothing at all, which makes it abstain."""
    monkeypatch.delenv("DOCUMENT_SOURCE_ORIGIN", raising=False)
    resolve_source_origin("xml", override="docx")
    resolve_source_origin("xml", override="nonsense")
    err = capsys.readouterr().err
    assert "digital-convert" in err
    assert "abstain" in err


def test_main_writes_source_origin_that_resolves_to_this_repo(workdir, docdir):
    """(D5) The real call site, end to end: page_split is the record's first writer of
    `source`, so if it omits `origin` nothing downstream can supply it — set_source() is
    first-writer-wins — and the §1a check defers forever."""
    src = workdir / "in" / "doc.alto.xml"
    src.write_text(_TWO_PAGE_DOC, encoding="utf-8")

    main([str(workdir / "in"), str(workdir / "out")])

    record = load_document(str(docdir / "doc.document.json"))
    source = record["source"]
    assert source["origin"] == "ABBYY-ALTO"
    assert resolve_originator(source["origin"]) == "alto-postprocess"
    assert source["page_count"] == 2
    assert source["filename"] == "doc.alto.xml"
    assert source["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert record["doc_id"] == "doc"


def test_main_json_input_records_an_ocr_origin(workdir, docdir):
    """The JSON path describes generic OCR/Doc-AI output (#31), so its origin carries
    the `ocr:` prefix — `source.origin` says how the ORIGINAL input was acquired, not
    that this stage happened to read JSON."""
    doc = {"pages": [{"pageNumber": 1, "text": "a"}, {"pageNumber": 2, "text": "b"}]}
    (workdir / "in" / "doc.json").write_text(json.dumps(doc), encoding="utf-8")

    main([str(workdir / "in"), str(workdir / "out")])

    source = load_document(str(docdir / "doc.document.json"))["source"]
    assert source["origin"].startswith("ocr:")
    assert resolve_originator(source["origin"]) == "alto-postprocess"
    assert source["media_type"] == "application/json"


def test_main_source_origin_cli_override_is_recorded(workdir, docdir):
    (workdir / "in" / "doc.alto.xml").write_text(_TWO_PAGE_DOC, encoding="utf-8")

    main([str(workdir / "in"), str(workdir / "out"), "--source-origin", "ocr:pero"])

    source = load_document(str(docdir / "doc.document.json"))["source"]
    assert source["origin"] == "ocr:pero"
    assert resolve_originator(source["origin"]) == "alto-postprocess"


def test_main_passes_a_run_id_string_and_a_paradata_ref_to_the_hook(workdir, docdir, monkeypatch):
    """(D6) Pins the SHAPE of the production call. This site used to pass the
    ParadataLogger object where every one of the seven sibling call sites passes
    `_logger.run_id`, and never passed a paradata_ref at all."""
    seen = {}

    def _spy(document_json_dir, doc_id, run_id, paradata_ref="", **kwargs):
        seen.update(
            document_json_dir=document_json_dir,
            doc_id=doc_id,
            run_id=run_id,
            paradata_ref=paradata_ref,
            kwargs=kwargs,
        )

    monkeypatch.setattr(document_hook, "write_document_block", _spy)
    (workdir / "in" / "doc.alto.xml").write_text(_TWO_PAGE_DOC, encoding="utf-8")

    main([str(workdir / "in"), str(workdir / "out")])

    assert isinstance(seen["run_id"], str) and seen["run_id"]
    assert seen["paradata_ref"].endswith("_alto-postprocess.json")
    assert seen["doc_id"] == "doc"
    assert seen["kwargs"]["source"]["origin"] == "ABBYY-ALTO"


def test_write_document_block_with_source_and_a_block_in_one_call(workdir, docdir):
    """(D6) The landmine itself: `source=` together with a block, which is the shape
    the sibling stages use and the shape this call site would take the day it starts
    contributing one. `set_source()` never stamps, so a non-serialisable run_id stayed
    invisible while only `source` was written; _stamp() embeds it, json.dump() raises
    TypeError, and DocumentRecord.__exit__ swallows that — leaving NO record at all,
    `source` included, plus a stray .tmp."""
    run_id = "260805-101112"
    document_hook.write_document_block(
        str(docdir),
        "CTXpair",
        run_id,
        document_hook.paradata_ref_for(_FakeParadataLogger(run_id)),
        source={"sha256": "c" * 64, "filename": "CTXpair.alto.xml", "origin": "ABBYY-ALTO"},
        merge_blocks={"pages": [{"page": "1", "quality_score": 0.5, "quality_band": "Noisy"}]},
    )

    record_path = docdir / "CTXpair.document.json"
    assert record_path.exists()  # a TypeError here would leave only CTXpair.document.json.tmp
    assert not (docdir / "CTXpair.document.json.tmp").exists()
    record = load_document(str(record_path))
    assert record["assembled"]["blocks"]["pages"]["run_id"] == run_id
    assert record["assembled"]["blocks"]["pages"]["paradata_ref"].endswith(f"{run_id}_alto-postprocess.json")
    # The §1a guard is live for this write rather than deferred, and it authorises it.
    assert resolve_originator(record["source"]["origin"]) == "alto-postprocess"
