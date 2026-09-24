"""
tests/test_doc_id_parity.py
===========================
One doc_id per document, whichever entry point derives it (atrium-project#10 D3).

This repo used to hand-roll the derivation in four places — `page_split.py`
(`_doc_id_from_filename` and the two `base_name`s that name the split output),
`run_pipeline.py` (`_single_input_doc_id`, whose docstring explicitly chose to
duplicate rather than import), `alto_stats_create.py` and `json_stats_create.py`
(`basename.split(".")[0]`) — plus `service/text_api.py`'s `Path(name.lower()).stem`.
All five agreed on the conventional single-dot names and forked on everything else,
which is precisely why nobody noticed: a fork does not fail, it silently produces a
second record under a key no other stage ever writes to, and the first one's blocks
are then lost (issue #13's original finding, reproduced inside one repo).

Every derivation now routes through `atrium_document.canonical_doc_id()`, and the tests
below pin THAT — not each implementation's private shape — so the parity survives a
future refactor of any single site. `service/text_api.py` calls `canonical_doc_id()`
directly and is pinned end-to-end by tests/test_service_api.py instead, because
importing it drags in the whole ML stack.
"""

import pytest

from atrium_document import canonical_doc_id
from page_split import _doc_id_from_filename
from run_pipeline import _single_input_doc_id

#: The multi-dot names the ecosystem actually passes around. Every one of them is a
#: real pipeline artefact name: alto-postprocess reads `.alto.xml`, hands `.document.json`
#: along the chain, and nlp-enrich/llm-enrich key the same document off `.teitok.xml` and
#: `.udpipe.conllu` — so a fork here orphans the record for the rest of the pipeline, not
#: just for this repo.
_MULTI_DOT_NAMES = [
    "CTX000000001.alto.xml",
    "CTX000000001.teitok.xml",
    "CTX000000001.udpipe.conllu",
    "CTX000000001.document.json",
    "CTX000000001.v2.csv",
    "CTX000000001.xml",
    "CTX000000001.json",
    # A doc_id that legitimately contains dots: `split(".")[0]` truncated it to "sbn",
    # inventing a document that shares its key with every other 2019 volume.
    "sbn.2019.alto.xml",
]


@pytest.mark.parametrize("name", _MULTI_DOT_NAMES)
def test_page_split_doc_id_matches_canonical(name):
    # The `fmt` argument is vestigial (canonical_doc_id is format-agnostic); passing
    # the wrong one must not change the answer either.
    assert _doc_id_from_filename(name, "xml") == canonical_doc_id(name)
    assert _doc_id_from_filename(name, "json") == canonical_doc_id(name)


@pytest.mark.parametrize("name", [n for n in _MULTI_DOT_NAMES if n.endswith((".xml", ".json"))])
def test_run_pipeline_single_input_doc_id_matches_canonical(name, tmp_path):
    """The orchestrator seeds the --document-json bridge directory under this doc_id and
    page_split then writes its record under its own: the two MUST agree or the bridge
    collects nothing and `--document-json-out` is silently never written."""
    (tmp_path / name).write_text("<alto/>", encoding="utf-8")
    input_format = "alto" if name.endswith(".xml") else "json"
    assert _single_input_doc_id(str(tmp_path), input_format) == canonical_doc_id(name)


@pytest.mark.parametrize(
    "name,doc_id,page",
    [
        ("CTX000000001-1.alto.xml", "CTX000000001", "1"),
        ("CTX000000001-17.alto.xml", "CTX000000001", "17"),
        ("CTX000000001.alto.xml", "CTX000000001", ""),
        ("sbn.2019-7.alto.xml", "sbn.2019", "7"),
    ],
)
def test_alto_stats_create_composes_canonical_with_the_page_split(name, doc_id, page, tmp_path, monkeypatch):
    """The stats stage is the COMPOSING case: page_split wrote `<doc_id>-<page>.alto.xml`,
    so this site strips a pipeline suffix AND splits a page label off, and canonical_doc_id()
    only does the first half. What must hold is that the two halves recombine into exactly
    the canonical id — `sbn.2019-7.alto.xml` used to yield file "sbn", page "2019".
    """
    pytest.importorskip("pandas")
    import alto_stats_create

    monkeypatch.setattr(alto_stats_create, "run_alto_tools_stats", lambda _path: {"textlines": "3"})
    record, skipped = alto_stats_create._process_single_xml(str(tmp_path / name), name)

    assert skipped is None
    assert record["file"] == doc_id
    assert record["page"] == page
    assert "-".join(p for p in (record["file"], record["page"]) if p) == canonical_doc_id(name)


@pytest.mark.parametrize(
    "name,doc_id,page",
    [
        ("CTX000000001-1.json", "CTX000000001", "1"),
        ("sbn.2019-7.json", "sbn.2019", "7"),
    ],
)
def test_json_stats_create_composes_canonical_with_the_page_split(name, doc_id, page, tmp_path):
    """json_stats_create.py mirrors alto_stats_create.py by design ("same convention"),
    so it has to mirror the fix too — its rows feed the identical downstream stages."""
    pytest.importorskip("pandas")
    import json_stats_create

    path = tmp_path / name
    path.write_text('{"text": "one line"}', encoding="utf-8")
    record, skipped = json_stats_create._process_single_json(str(path), name)

    assert skipped is None
    assert record["file"] == doc_id
    assert record["page"] == page
    assert "-".join(p for p in (record["file"], record["page"]) if p) == canonical_doc_id(name)


def test_split_output_directory_is_named_by_the_same_derivation(tmp_path):
    """page_split's per-document output directory IS the doc_id — every later stage
    re-derives `file` from it — so the splitter and the record key must not be two
    independent copies of one convention."""
    from page_split import split_alto_xml

    src = tmp_path / "sbn.2019.alto.xml"
    src.write_text(
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v3#"><Layout>'
        '<Page ID="P1" PHYSICAL_IMG_NR="1"/></Layout></alto>',
        encoding="utf-8",
    )
    out = tmp_path / "out"

    assert split_alto_xml(str(src), str(out)) == 1
    doc_id = canonical_doc_id(src.name)
    assert (out / doc_id / f"{doc_id}-1.alto.xml").exists()


# ── (#31) the text-lines path ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "parent,name,doc_id,page",
    [
        ("CTX000000001", "CTX000000001-1.txt", "CTX000000001", "1"),
        ("sbn.2019", "sbn.2019-7.txt", "sbn.2019", "7"),
        # A hyphenated doc_id: text_split names the directory after it, so the stats
        # stage keeps it whole (the ALTO/JSON scripts' split("-") would say "my", "doc").
        ("my-doc", "my-doc-3.txt", "my-doc", "3"),
        # Not in a doc_id directory: the LAST hyphen separates the page.
        ("loose", "a-b-12.txt", "a-b", "12"),
    ],
)
def test_text_stats_create_composes_canonical_with_text_split(parent, name, doc_id, page, tmp_path):
    import text_stats_create

    path = tmp_path / parent / name
    path.parent.mkdir()
    path.write_text("one line\n", encoding="utf-8")
    assert text_stats_create.file_page_from_path(str(path)) == (doc_id, page)
    assert f"{doc_id}-{page}" == canonical_doc_id(name)


@pytest.mark.parametrize("name", ["report.final.docx", "sbn.2019.txt", "CTX000000001.pdf", "scan.alto.xml"])
def test_run_pipeline_single_input_doc_id_text_format(name, tmp_path):
    """--document-json bridging for --method text-lines keys the record exactly as
    text_split.py does (page_split._doc_id_from_filename → canonical_doc_id)."""
    (tmp_path / name).write_bytes(b"x")
    (tmp_path / ".hidden").write_bytes(b"x")  # ignored by text_split, so ignored here
    (tmp_path / "~$lock.docx").write_bytes(b"x")
    assert _single_input_doc_id(str(tmp_path), "text") == canonical_doc_id(name) == _doc_id_from_filename(name)


def test_run_pipeline_single_input_doc_id_text_format_needs_exactly_one(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"x")
    (tmp_path / "b.pdf").write_bytes(b"x")
    assert _single_input_doc_id(str(tmp_path), "text") is None
