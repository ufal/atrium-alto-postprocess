"""
tests/test_quality_model_evaluate.py
====================================
Fast, ML-stack-free tests for the issue #23 evaluation harness
(``tools/quality_model/evaluate.py``).

Everything is driven from hand-written prediction rows, so the algorithm-vs-model
metrics, the calibration table, the predicted-score monotonicity check, and the
self-reference-free **gold gate** are all exercised without torch/transformers.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_QM = _ROOT / "tools" / "quality_model"
if str(_QM) not in sys.path:
    sys.path.insert(0, str(_QM))

import evaluate as EV  # noqa: E402

# ── Evaluation vs algorithm ──────────────────────────────────────────────────


def test_perfect_predictions_score_top_marks():
    rows = [
        {"score_raw": "0.90", "pred_score": "0.90", "categ": "Clear"},
        {"score_raw": "0.40", "pred_score": "0.40", "categ": "Trash"},
        {"score_raw": "0.65", "pred_score": "0.65", "categ": "Noisy"},
    ]
    rep = EV.evaluate_predictions(rows)
    assert rep["regression"]["mae"] == 0.0
    assert rep["banded_vs_algo"]["macro_f1"] == 1.0
    assert rep["banded_vs_algo"]["accuracy"] == 1.0


def test_banded_vs_algo_uses_stored_category():
    # pred_score bands to Clear but the algorithm called it Noisy → disagreement.
    rows = [{"score_raw": "0.85", "pred_score": "0.95", "categ": "Noisy"}]
    rep = EV.evaluate_predictions(rows)
    # predicted band = Clear, true categ = Noisy → not perfect
    assert rep["banded_vs_algo"]["accuracy"] == 0.0


def test_cathead_metrics_when_pred_categ_present():
    rows = [
        {"score_raw": "0.9", "pred_score": "0.9", "categ": "Clear", "pred_categ": "Clear"},
        {"score_raw": "0.4", "pred_score": "0.4", "categ": "Trash", "pred_categ": "Noisy"},
    ]
    rep = EV.evaluate_predictions(rows, pred_categ_col="pred_categ")
    assert "cathead_vs_algo" in rep
    assert rep["cathead_vs_algo"]["accuracy"] == 0.5


def test_calibration_bins_report_gap():
    rows = [
        {"score_raw": "0.10", "pred_score": "0.20", "categ": "Trash"},
        {"score_raw": "0.12", "pred_score": "0.25", "categ": "Trash"},
    ]
    rep = EV.evaluate_predictions(rows)
    filled = [c for c in rep["calibration"] if c.get("count")]
    assert filled and all("gap" in c for c in filled)


def test_monotonicity_on_predicted_score():
    # heavier corruption → lower predicted score = concordant
    rows = [
        {
            "source_doc": "D",
            "source_line": "1",
            "band": "none",
            "pred_score": "0.95",
            "score_raw": "0.95",
            "categ": "Clear",
        },
        {
            "source_doc": "D",
            "source_line": "1",
            "band": "light",
            "pred_score": "0.70",
            "score_raw": "0.70",
            "categ": "Noisy",
        },
        {
            "source_doc": "D",
            "source_line": "1",
            "band": "heavy",
            "pred_score": "0.30",
            "score_raw": "0.30",
            "categ": "Trash",
        },
    ]
    rep = EV.evaluate_predictions(rows)
    assert rep["monotonicity"]["accuracy"] == 1.0
    assert rep["monotonicity"]["pairs"] == 3


def test_stratification_produces_per_group_metrics():
    rows = [
        {"score_raw": "0.9", "pred_score": "0.9", "categ": "Clear", "lang": "ces_Latn"},
        {"score_raw": "0.4", "pred_score": "0.4", "categ": "Trash", "lang": "deu_Latn"},
    ]
    rep = EV.evaluate_predictions(rows, stratify_by="lang")
    assert "by_lang" in rep
    assert set(rep["by_lang"]) == {"ces_Latn", "deu_Latn"}
    assert rep["by_lang"]["ces_Latn"]["n"] == 1


# ── Gold gate (the objective check) ──────────────────────────────────────────


def test_gold_gate_passes_when_model_beats_algorithm():
    # model matches gold everywhere; algorithm is wrong on one line
    gold = [
        {"gold_categ": "Noisy", "algo_categ": "Trash", "pred_score": "0.65"},  # model→Noisy (right), algo→Trash (wrong)
        {"gold_categ": "Clear", "algo_categ": "Clear", "pred_score": "0.90"},
        {"gold_categ": "Trash", "algo_categ": "Trash", "pred_score": "0.30"},
    ]
    gate = EV.gold_gate(gold)
    assert gate["model_macro_f1"] >= gate["algo_macro_f1"]
    assert gate["passed"] is True


def test_gold_gate_fails_when_model_worse():
    # model always says Clear; algorithm matches gold → model far worse
    gold = [
        {"gold_categ": "Trash", "algo_categ": "Trash", "pred_score": "0.95"},
        {"gold_categ": "Noisy", "algo_categ": "Noisy", "pred_score": "0.95"},
        {"gold_categ": "Clear", "algo_categ": "Clear", "pred_score": "0.95"},
    ]
    gate = EV.gold_gate(gold, margin=0.01)
    assert gate["passed"] is False
    assert gate["algo_macro_f1"] > gate["model_macro_f1"]


def test_gold_gate_respects_pred_categ_col():
    # Model matches gold on both lines; the algorithm is wrong on the 2nd.
    # macro-F1 is over the fixed 3-class space, so a perfect 2-class subset scores
    # 2/3 (the absent Noisy class contributes 0) — the model still beats the algo.
    gold = [
        {"gold_categ": "Clear", "algo_categ": "Clear", "pred_categ": "Clear"},
        {"gold_categ": "Trash", "algo_categ": "Noisy", "pred_categ": "Trash"},
    ]
    gate = EV.gold_gate(gold, pred_categ_col="pred_categ")
    assert gate["model_macro_f1"] > gate["algo_macro_f1"]
    assert gate["passed"] is True


# ── Reporting ────────────────────────────────────────────────────────────────


def test_format_report_renders():
    rows = [{"score_raw": "0.9", "pred_score": "0.9", "categ": "Clear"}]
    text = EV.format_report(EV.evaluate_predictions(rows))
    assert "evaluation" in text.lower()
    assert "Spearman" in text
