"""
tests/test_document_hook.py — tests for the LIVE document_hook.py surface
(write_document_block / resolve_document_json_dir / paradata_ref_for), which
every stage script (page_split, aggregate_STAT, classify_TEXT, the four
extract_*) actually calls.

Previously this file tested `write_document_record()`/`update_document_record()`
— two functions that bypassed DocumentRecord entirely, wrote keys from the
superseded aggregator draft (`assembled.source_run_ids`, `assembled.mode`), and
were never called from anywhere except these tests. They have been deleted
(issue #13 alignment audit, P2.1) since the real, wired-up path below already
does the job correctly. `test_no_baseline_creates_standalone`'s mocked logger
also assigned `logger.run_id = ...`, which no longer works now that `run_id` is
a read-only property on the real ParadataLogger.
"""

import json
import os

import jsonschema
import pytest

import document_hook
from atrium_document import load_document
from document_hook import (
    document_path,
    pages_and_content_from_text,
    paradata_ref_for,
    resolve_document_json_dir,
    write_document_block,
)


class _FakeLogger:
    def __init__(self, run_id, program, paradata_dir="paradata"):
        self.run_id = run_id
        self.program = program
        self.paradata_dir = paradata_dir


def test_resolve_document_json_dir_env_wins(monkeypatch):
    monkeypatch.setenv("DOCUMENT_JSON_DIR", "/from/env")
    assert resolve_document_json_dir("/from/config") == "/from/env"


def test_resolve_document_json_dir_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("DOCUMENT_JSON_DIR", raising=False)
    assert resolve_document_json_dir("/from/config") == "/from/config"


def test_resolve_document_json_dir_disabled_by_default(monkeypatch):
    monkeypatch.delenv("DOCUMENT_JSON_DIR", raising=False)
    assert resolve_document_json_dir(None) == ""
    assert resolve_document_json_dir("") == ""


def test_paradata_ref_for():
    logger = _FakeLogger(run_id="260731-101112", program="alto-postprocess", paradata_dir="paradata")
    assert paradata_ref_for(logger) == os.path.join("paradata", "260731-101112_alto-postprocess.json")


def test_write_document_block_noop_when_dir_falsy(monkeypatch, tmp_path):
    # Rule 3 / enablement: a falsy document_json_dir must return immediately,
    # before any filesystem access — verified by running from an empty cwd.
    monkeypatch.chdir(tmp_path)
    result = write_document_block("", "CTX01", run_id="r1", set_blocks={"content": {"text": "hi"}})
    assert result is None
    assert list(tmp_path.iterdir()) == []


def test_write_document_block_merges_without_erasing_co_owned_fields(tmp_path):
    doc_dir = str(tmp_path)

    # page-classification writes first: category + category_confidence on page 1.
    from atrium_document import DocumentRecord

    with DocumentRecord("CTX01", "page-classification", out_dir=doc_dir) as doc:
        doc.merge_block("pages", [{"page": "1", "category": "Text", "category_confidence": 0.97}])

    # alto-postprocess contributes via the live hook — quality fields on the SAME page row.
    write_document_block(
        doc_dir,
        "CTX01",
        run_id="r2",
        paradata_ref="paradata/r2_alto-postprocess.json",
        merge_blocks={"pages": [{"page": "1", "quality_score": 0.98, "quality_band": "Clear"}]},
    )

    record = load_document(document_path(doc_dir, "CTX01"))
    assert len(record["pages"]) == 1
    page = record["pages"][0]
    # Both contributions must survive on the same row.
    assert page["category"] == "Text"
    assert page["category_confidence"] == 0.97
    assert page["quality_score"] == 0.98
    assert page["quality_band"] == "Clear"


def test_write_document_block_set_blocks_uses_set_block(tmp_path):
    doc_dir = str(tmp_path)
    write_document_block(
        doc_dir,
        "CTX02",
        run_id="r1",
        set_blocks={"content": {"text": "full document text"}},
    )
    record = load_document(document_path(doc_dir, "CTX02"))
    assert record["content"] == {"text": "full document text"}
    assert record["assembled"]["blocks"]["content"]["program"] == "alto-postprocess"


def _write_pages(output_text_dir, file_id, page_texts):
    save_dir = output_text_dir / file_id
    save_dir.mkdir(parents=True, exist_ok=True)
    for page_id, text in page_texts.items():
        (save_dir / f"{file_id}-{page_id}.txt").write_text(text, encoding="utf-8")


def test_pages_and_content_default_one_row_per_page(tmp_path):
    _write_pages(tmp_path, "doc1", {1: "Page one text", 2: "Page two text"})

    pages, content = pages_and_content_from_text(str(tmp_path), "doc1", [1, 2], engine="json-keys")

    assert [p["page"] for p in pages] == ["1", "2"]
    assert all(p["ocr"] == {"engine": "json-keys"} for p in pages)
    assert content == {"text": "Page one text\n\nPage two text"}


def test_pages_and_content_force_single_page_collapses_rows(tmp_path):
    _write_pages(tmp_path, "doc2", {1: "Page one text", 2: "Page two text", 3: "Page three text"})

    pages, content = pages_and_content_from_text(
        str(tmp_path), "doc2", [1, 2, 3], engine="json-keys", force_single_page=True
    )

    # Exactly one schema-valid pages[] row, regardless of source page count.
    assert len(pages) == 1
    assert pages[0]["page"] == "1"
    assert pages[0]["ocr"]["engine"] == "json-keys"
    assert pages[0]["ocr"]["force_single_page"] is True
    # Original page labels traceable, in concatenation order — no data loss.
    assert pages[0]["ocr"]["source_pages"] == ["1", "2", "3"]
    # content.text is identical to the non-forced case: force_single_page is an
    # assembly-only policy switch, it does not change what gets extracted.
    assert content == {"text": "Page one text\n\nPage two text\n\nPage three text"}


def test_pages_and_content_force_single_page_skips_unreadable_pages_without_duplication(tmp_path):
    # Only page 1 is actually written to disk; page 2 is missing (e.g. its
    # extraction failed) and must be skipped, not guessed at or duplicated.
    _write_pages(tmp_path, "doc3", {1: "Only readable page"})

    pages, content = pages_and_content_from_text(
        str(tmp_path), "doc3", [1, 2], engine="json-keys", force_single_page=True
    )

    assert len(pages) == 1
    assert pages[0]["ocr"]["source_pages"] == ["1"]
    assert content == {"text": "Only readable page"}


def test_pages_and_content_force_single_page_no_readable_pages_yields_no_rows(tmp_path):
    pages, content = pages_and_content_from_text(
        str(tmp_path), "doc4", [1, 2], engine="json-keys", force_single_page=True
    )

    assert pages == []
    assert content == {"text": None}


# ── (atrium-project#10 D4) the Layer D validation gate at the chokepoint ─────
#
# These pin the POLICY, not merely that the call exists: hard-fail on this repo's own
# output, warn on an inherited baseline, and degrade loudly (never silently) when
# jsonschema is unavailable. Before this, validate_document() had zero production call
# sites in any of the five repos, so "no doc.json is emitted if validation fails"
# protected nothing at all.


def _write_raw_record(path, record):
    """A baseline written WITHOUT DocumentRecord, so it can be deliberately invalid."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh)


def test_invalid_inherited_baseline_warns_but_still_accretes(tmp_path, capsys):
    """An upstream tool's invalid record must not stall this stage: warn, name the
    schema error, keep going. Refusing here would turn one bad record into a stalled
    pipeline, and rule 6 already commits to passing unknown content through."""
    doc_dir = str(tmp_path)
    # quality_score has `maximum: 1` in the schema — 5 is invalid, and load_document
    # is happy to read it, which is exactly the situation this policy is about.
    _write_raw_record(
        document_path(doc_dir, "CTXbad"),
        {
            "schema_version": "1.0",
            "record_type": "atrium-document",
            "doc_id": "CTXbad",
            "pages": [{"page": "1", "quality_score": 5}],
        },
    )

    write_document_block(
        doc_dir,
        "CTXbad",
        run_id="r1",
        merge_blocks={"pages": [{"page": "1", "quality_band": "Clear"}]},
    )

    err = capsys.readouterr().err
    assert "inherited baseline" in err
    assert "does not validate" in err
    # The stage's own contribution landed, and the record was still written.
    record = load_document(document_path(doc_dir, "CTXbad"))
    assert record["pages"][0]["quality_band"] == "Clear"
    # Its own output gate was downgraded to a warning, since the defect is inherited.
    assert "downgraded" in err


def test_invalid_own_output_raises_and_emits_no_record(tmp_path):
    """The other half of the policy: this repo must never EMIT an invalid record. The
    raise happens inside DocumentRecord's context manager body, which is what keeps
    finalize() from running — so there is no record and no leftover .tmp either."""
    doc_dir = str(tmp_path)
    with pytest.raises(jsonschema.ValidationError):
        write_document_block(
            doc_dir,
            "CTXown",
            run_id="r1",
            # quality_score is ours to write, so nothing filters it — it is simply
            # out of the schema's [0, 1] range.
            merge_blocks={"pages": [{"page": "1", "quality_score": 42}]},
        )
    assert list(tmp_path.iterdir()) == []


def test_missing_jsonschema_warns_once_and_does_not_block_the_write(tmp_path, monkeypatch, capsys):
    """validate_document() raises RuntimeError when jsonschema is absent, deliberately,
    so a gate cannot quietly become a no-op. The hook must catch THAT case apart from a
    real validation failure: one loud warning naming the dependency, then continue —
    a standalone run cannot be made to depend on an optional package (rule 3)."""

    def _no_jsonschema(_record):
        raise RuntimeError("jsonschema is not installed, so the record cannot be validated.")

    monkeypatch.setattr(document_hook, "validate_document", _no_jsonschema)
    monkeypatch.setattr(document_hook, "_VALIDATION_UNAVAILABLE_WARNED", False)

    doc_dir = str(tmp_path)
    write_document_block(doc_dir, "CTXnojs", run_id="r1", set_blocks={"content": {"text": "hi"}})
    write_document_block(doc_dir, "CTXnojs", run_id="r2", set_blocks={"content": {"text": "hi again"}})

    err = capsys.readouterr().err
    assert "jsonschema" in err
    assert "DISABLED" in err
    # Loud once, not once per document — a batch run holds thousands of them.
    assert err.count("validation is DISABLED") == 1
    assert load_document(document_path(doc_dir, "CTXnojs"))["content"] == {"text": "hi again"}


# ── (atrium-project#10 D8) merge_block()'s silent filtering, made loud ───────


def test_merge_of_a_field_outside_this_repos_grant_raises(tmp_path):
    """`category` belongs to page-classification. merge_block() drops it silently and
    the result still validates (pages[] requires only `page`), which is how a wrong
    grant produced rows stripped down to their key. assert_fields_survived() turns that
    into a failure at the call site that got it wrong."""
    doc_dir = str(tmp_path)
    with pytest.raises(RuntimeError, match="dropped by merge_block"):
        write_document_block(
            doc_dir,
            "CTXgrant",
            run_id="r1",
            merge_blocks={"pages": [{"page": "1", "quality_score": 0.9, "category": "Text"}]},
        )
    assert list(tmp_path.iterdir()) == []


def test_declared_fields_survive_the_merge(tmp_path):
    """The same assertion must stay silent for every field this repo really owns, on
    both blocks it merges into — otherwise it would fire on production call sites."""
    doc_dir = str(tmp_path)
    write_document_block(
        doc_dir,
        "CTXok",
        run_id="r1",
        merge_blocks={
            "pages": [{"page": "1", "quality_score": 0.9, "quality_band": "Clear", "ocr": {"engine": "alto-tools"}}],
            "lines": [
                {"page": "1", "line": 1, "text": "a line", "lang": "ces", "quality_score": 0.8, "categ": "Clear"}
            ],
        },
    )
    record = load_document(document_path(doc_dir, "CTXok"))
    assert record["lines"][0]["text"] == "a line"
    assert record["pages"][0]["ocr"] == {"engine": "alto-tools"}


def test_write_document_block_records_source_once(tmp_path):
    doc_dir = str(tmp_path)
    write_document_block(
        doc_dir,
        "CTX03",
        run_id="r1",
        source={"sha256": "a" * 64, "filename": "CTX03.alto.xml", "origin": "ABBYY-ALTO"},
    )
    # A later write must not overwrite an existing source (set_source is first-writer-wins).
    write_document_block(
        doc_dir,
        "CTX03",
        run_id="r2",
        source={"sha256": "b" * 64, "filename": "other.xml", "origin": "digital-born-pdf"},
    )
    record = load_document(document_path(doc_dir, "CTX03"))
    assert record["source"]["sha256"] == "a" * 64


# ── (#31) the origin guard: no OCR-path blocks in a digital-born record ───────

_PAGES = [{"page": "1", "quality_score": 0.9, "quality_band": "Clear"}]
_LINES = [{"page": "1", "line": 1, "text": "a line", "categ": "Clear", "quality_score": 0.9}]


def _seed_source(doc_dir, doc_id, origin, pages=None):
    from atrium_document import DocumentRecord

    with DocumentRecord(doc_id, "digital-convert" if origin.startswith("digital") else "alto-postprocess",
                        out_dir=str(doc_dir)) as doc:  # fmt: skip
        doc.set_source(sha256="c" * 64, filename=f"{doc_id}.bin", origin=origin)
        if pages:
            doc.merge_block("pages", pages)


def test_guard_drops_positional_blocks_for_a_digital_born_record(tmp_path, capsys):
    doc_dir = str(tmp_path)
    _seed_source(doc_dir, "DB1", "digital-born-docx")
    write_document_block(doc_dir, "DB1", run_id="r1", merge_blocks={"pages": _PAGES, "lines": _LINES})
    record = load_document(document_path(doc_dir, "DB1"))
    assert "lines" not in record and not record.get("pages")
    assert record["source"]["origin"] == "digital-born-docx"
    assert "not writing lines, pages" in capsys.readouterr().err


@pytest.mark.parametrize("origin", ["ABBYY-ALTO", "ocr:pdf-text-layer", "unknown-method"])
def test_guard_keeps_blocks_for_own_or_unmatched_origins(tmp_path, origin):
    doc_dir = str(tmp_path)
    _seed_source(doc_dir, "OK1", origin)
    write_document_block(doc_dir, "OK1", run_id="r1", merge_blocks={"lines": _LINES})
    assert load_document(document_path(doc_dir, "OK1"))["lines"][0]["text"] == "a line"


def test_guard_honours_the_needs_ocr_hand_off(tmp_path):
    """digital-convert marked a page needs_ocr: this repo is ASKED to re-originate it."""
    doc_dir = str(tmp_path)
    _seed_source(
        doc_dir, "HO1", "digital-born-pdf",
        pages=[{"page": "1", "needs_ocr": True, "needs_ocr_reason": "garbled text layer"}],
    )  # fmt: skip
    write_document_block(doc_dir, "HO1", run_id="r1", merge_blocks={"lines": _LINES})
    assert load_document(document_path(doc_dir, "HO1"))["lines"][0]["categ"] == "Clear"


def test_guard_reads_the_origin_of_this_calls_source_when_no_baseline(tmp_path):
    doc_dir = str(tmp_path)
    write_document_block(
        doc_dir,
        "NEW1",
        run_id="r1",
        source={"sha256": "d" * 64, "filename": "NEW1.pdf", "origin": "digital-born-pdf"},
        merge_blocks={"pages": _PAGES},
    )
    record = load_document(document_path(doc_dir, "NEW1"))
    assert record["source"]["origin"] == "digital-born-pdf"
    assert not record.get("pages")


# ── (#31 Phase 4) SOURCE_ORIGIN_BY_KIND ───────────────────────────────────────


def test_parse_origin_by_kind_accepts_commas_newlines_and_any_case():
    parsed = document_hook.parse_origin_by_kind("XLSX = ocr:generic,\n pptx=ocr:tesseract ,", ["xlsx", "pptx", "odt"])
    assert parsed == {"xlsx": "ocr:generic", "pptx": "ocr:tesseract"}
    assert document_hook.parse_origin_by_kind("", ["xlsx"]) == {}


@pytest.mark.parametrize(
    "raw,why",
    [
        ("xlsx ocr:generic", "is not <kind> = <origin>"),
        ("xls = ocr:generic", "unknown kind"),
        ("xlsx = ocr:a, xlsx = ocr:b", "given twice"),
        ("xlsx =", "empty origin"),
        ("xlsx = scanned", "matches no known originator"),
    ],
)
def test_parse_origin_by_kind_rejects_typos_naming_the_key(raw, why):
    with pytest.raises(ValueError, match="SOURCE_ORIGIN_BY_KIND") as info:
        document_hook.parse_origin_by_kind(raw, ["xlsx", "pptx"])
    assert why in str(info.value)


def test_resolve_input_origin_precedence():
    kw = dict(override="ocr:cli", env="ocr:env", configured="ocr:cfg", by_kind={"xlsx": "ocr:kind"})
    resolve = document_hook.resolve_input_origin
    assert resolve("xlsx", "digital-born-xlsx", **kw) == "ocr:cli"
    assert resolve("xlsx", "digital-born-xlsx", **{**kw, "override": ""}) == "ocr:env"
    assert resolve("xlsx", "digital-born-xlsx", **{**kw, "override": "", "env": ""}) == "ocr:kind"
    assert resolve("docx", "digital-born-docx", **{**kw, "override": "", "env": ""}) == "ocr:cfg"
    assert resolve("docx", "digital-born-docx") == "digital-born-docx"


# ── (#31 Phase 5) resume helpers ─────────────────────────────────────────────


def test_write_if_changed_keeps_an_unchanged_file_untouched(tmp_path):
    import os

    path = tmp_path / "p.txt"
    assert document_hook.write_text_if_changed(path, "řádek\nřádek 2", newline="\n") is True
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns - 10**12, st.st_mtime_ns - 10**12))
    old = os.stat(path).st_mtime_ns
    assert document_hook.write_text_if_changed(path, "řádek\nřádek 2", newline="\n") is False
    assert os.stat(path).st_mtime_ns == old
    assert document_hook.write_text_if_changed(path, "jiný řádek", newline="\n") is True
    assert path.read_text(encoding="utf-8") == "jiný řádek"
    assert document_hook.write_bytes_if_changed(tmp_path / "b.bin", b"\x00") is True


def test_output_is_current_compares_file_times(tmp_path):
    import os

    src, out = tmp_path / "in.txt", tmp_path / "out.csv"
    assert document_hook.output_is_current(out, [src]) is False  # no output yet
    src.write_text("x", encoding="utf-8")
    out.write_text("y", encoding="utf-8")
    st = os.stat(src)
    os.utime(src, ns=(st.st_atime_ns - 10**12, st.st_mtime_ns - 10**12))
    assert document_hook.output_is_current(out, [src, tmp_path / "missing.txt"]) is True
    os.utime(src, ns=(st.st_atime_ns + 10**12, st.st_mtime_ns + 10**12))
    assert document_hook.output_is_current(out, [src]) is False


def test_read_page_index_keeps_doc_ids_as_text(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text("file,page,path\n0001,1,a\nNA,2,b\n12,3,c\n", encoding="utf-8")
    df = document_hook.read_page_index(str(path))
    assert list(df["file"]) == ["0001", "NA", "12"]
    assert list(df["page"]) == [1, 2, 3]
