"""
tests/test_quality_model_dataset.py
===================================
Fast, model-free tests for the issue #23 dataset factory
(``tools/quality_model/build_dataset.py`` + ``report_dataset.py``).

Uses the model-free offline scorer and a tiny committed fixture
(``tests/fixtures/quality_model_lines.csv``) — never reads ``data_samples/``.

Locked behaviours: Empty/Non-text exclusion, document-level split with no
leakage, gold-doc hold-out, dedup, deterministic/reproducible build, and a
sane monotonicity report.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_QM = _ROOT / "tools" / "quality_model"
if str(_QM) not in sys.path:
    sys.path.insert(0, str(_QM))

import build_dataset as B  # noqa: E402
import report_dataset as R  # noqa: E402

_FIXTURE = _ROOT / "tests" / "fixtures" / "quality_model_lines.csv"


def _rows():
    return B.read_rows([_FIXTURE])


def _build(**kwargs):
    rows = _rows()
    scorer = B.make_offline_scorer()
    defaults = dict(seed=23, variants_per_clear=3)
    defaults.update(kwargs)
    return B.build_dataset(rows, scorer, **defaults)


# ── Selection ──────────────────────────────────────────────────────────────


def test_excludes_empty_and_non_text_sources():
    items, _ = _build()
    # The Empty ("") and Non-text ("A123/2024") fixture rows must never appear.
    assert all(it["categ"] not in B.EXCLUDED_CATEGS for it in items)
    assert all(it["text"].strip() for it in items)
    texts = {it["text"] for it in items}
    assert "A123/2024" not in texts


def test_clear_lines_produce_corrupt_variants():
    items, manifest = _build(variants_per_clear=3)
    assert manifest["counts"]["by_provenance"].get("corrupt", 0) > 0
    # Every corrupt item descends from a Clear source line.
    assert all(it["source_categ"] == "Clear" for it in items if it["provenance"] == "corrupt")


# ── Splitting ──────────────────────────────────────────────────────────────


def test_no_document_leaks_across_splits():
    items, _ = _build()
    doc_to_splits = {}
    for it in items:
        doc_to_splits.setdefault(it["source_doc"], set()).add(it["split"])
    assert all(len(splits) == 1 for splits in doc_to_splits.values()), doc_to_splits


def test_gold_docs_forced_into_test_split():
    items, manifest = _build(gold_docs={"CTXB"})
    ctxb = [it for it in items if it["source_doc"] == "CTXB"]
    assert ctxb  # fixture has CTXB rows
    assert all(it["split"] == "test" for it in ctxb)
    assert manifest["doc_splits"]["CTXB"] == "test"


# ── Dedup ──────────────────────────────────────────────────────────────────


def test_duplicate_text_is_deduplicated():
    # "Sonda byla zaměřena geodeticky přesně." appears in both CTXA and CTXB.
    items, _ = _build(variants_per_clear=0)  # originals only, so the dup is unambiguous
    dup = [it for it in items if B.normalize_text(it["text"]) == "Sonda byla zaměřena geodeticky přesně."]
    assert len(dup) == 1


# ── Determinism ────────────────────────────────────────────────────────────


def test_build_is_reproducible():
    items1, manifest1 = _build()
    items2, manifest2 = _build()
    assert [it["text"] for it in items1] == [it["text"] for it in items2]
    assert [it["score_raw"] for it in items1] == [it["score_raw"] for it in items2]
    assert manifest1 == manifest2


def test_manifest_counts_match_items():
    items, manifest = _build()
    assert manifest["counts"]["total"] == len(items)
    assert sum(manifest["counts"]["by_split"].values()) == len(items)


# ── Balancing ──────────────────────────────────────────────────────────────


def test_bin_cap_reduces_train_rows():
    uncapped, _ = _build(bin_cap=None)
    capped, manifest = _build(bin_cap=1)
    assert len(capped) <= len(uncapped)
    assert manifest["balance_removed"] >= 0
    # val/test rows are never dropped by balancing.
    assert all(it["split"] == "train" for it in _dropped_are_train(uncapped, capped))


def _dropped_are_train(uncapped, capped):
    capped_keys = {(it["source_doc"], it["source_line"], it["text"]) for it in capped}
    return [it for it in uncapped if (it["source_doc"], it["source_line"], it["text"]) not in capped_keys]


# ── Report ─────────────────────────────────────────────────────────────────


def test_report_monotonicity_is_sane():
    items, _ = _build(variants_per_clear=3)
    report = R.build_report([_row_for_report(it) for it in items])
    acc = report["monotonicity"]["accuracy"]
    assert 0.0 <= acc <= 1.0
    # Corruption lowers the non-perplexity signals, so heavier variants should
    # mostly score no higher than lighter ones even under the offline scorer.
    assert acc >= 0.7


def test_report_has_distribution_and_realism():
    items, _ = _build()
    report = R.build_report([_row_for_report(it) for it in items])
    assert report["distribution"]["total"] == len(items)
    assert set(report["realism"]["features"]) == set(R.REALISM_FEATURES)


def _row_for_report(item: dict) -> dict:
    # build_report reads string CSV cells; mimic a written+reread row.
    return {k: ("" if v is None else str(v)) for k, v in item.items()}
