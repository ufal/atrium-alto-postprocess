"""
tests/test_quality_model_train.py
=================================
Fast, ML-stack-free tests for the issue #23 training glue:
``common.py`` (config / metrics / banding / features), the pure parts of
``train.py`` (param wiring, compute_metrics, category ids), and
``train_baseline_gbm.py`` formatting.

The actual sklearn / torch training paths are covered by ``@pytest.mark.slow``
tests that self-skip when the libraries are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_QM = _ROOT / "tools" / "quality_model"
if str(_QM) not in sys.path:
    sys.path.insert(0, str(_QM))

import common as CM  # noqa: E402
import train as T  # noqa: E402
import train_baseline_gbm as GBM  # noqa: E402

_CONFIG = _ROOT / "setup" / "config_quality_model.txt"


# ── Config ───────────────────────────────────────────────────────────────────


def test_config_loads_with_expected_values():
    cfg = CM.load_config(_CONFIG)
    assert CM.cfg_get(cfg, "MODEL", "NAME", "") == "distilbert-base-multilingual-cased"
    assert CM.cfg_get(cfg, "MODEL", "MAX_LENGTH", 0, int) == 192
    assert CM.cfg_get(cfg, "MODEL", "CATEGORY_HEAD", False, bool) is True
    assert CM.cfg_get(cfg, "TRAIN", "HUBER_DELTA", 0.0, float) == 0.1
    assert CM.cfg_get(cfg, "TRAIN", "CE_WEIGHT", 0.0, float) == 0.3


def test_cfg_get_casts_and_defaults():
    cfg = CM.load_config(_CONFIG)
    assert CM.cfg_get(cfg, "TRAIN", "SEED", 0, int) == 42
    assert CM.cfg_get(cfg, "BASELINE", "INCLUDE_PERPLEXITY", False, bool) is True
    # missing key → default
    assert CM.cfg_get(cfg, "TRAIN", "NOPE", "fallback") == "fallback"


# ── Regression metrics ───────────────────────────────────────────────────────


def test_regression_metrics_perfect_fit():
    m = CM.regression_metrics([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    assert m["mae"] == 0.0 and m["rmse"] == 0.0
    assert m["pearson"] == 1.0 and m["spearman"] == 1.0


def test_regression_metrics_known_error():
    m = CM.regression_metrics([0.0, 0.0], [0.2, 0.4])
    assert m["mae"] == pytest.approx(0.3)
    assert m["rmse"] == pytest.approx(0.316228, abs=1e-5)


def test_spearman_is_rank_based():
    # monotonic but non-linear: spearman perfect, pearson not
    m = CM.regression_metrics([1.0, 2.0, 3.0], [1.0, 4.0, 9.0])
    assert m["spearman"] == 1.0
    assert m["pearson"] < 1.0


# ── Banding + category metrics ───────────────────────────────────────────────


def test_band_category_thresholds():
    assert CM.band_category(0.40) == "Trash"
    assert CM.band_category(0.55) == "Noisy"  # boundary is inclusive of Noisy
    assert CM.band_category(0.79) == "Noisy"
    assert CM.band_category(0.80) == "Clear"


def test_category_metrics_perfect_and_partial():
    perfect = CM.category_metrics(["Trash", "Noisy", "Clear"], ["Trash", "Noisy", "Clear"])
    assert perfect["macro_f1"] == 1.0 and perfect["accuracy"] == 1.0
    partial = CM.category_metrics(["Trash", "Noisy", "Clear"], ["Trash", "Trash", "Clear"])
    assert partial["accuracy"] == pytest.approx(2 / 3, abs=1e-4)


def test_banded_category_metrics_matches_scores():
    m = CM.banded_category_metrics([0.9, 0.3, 0.7], [0.85, 0.4, 0.6])
    assert m["macro_f1"] == 1.0  # same bands: Clear/Trash/Noisy


# ── Feature extraction ───────────────────────────────────────────────────────


def test_feature_columns_toggle_perplexity():
    assert CM.PERPLEXITY_COLUMN in CM.feature_columns(True)
    assert CM.PERPLEXITY_COLUMN not in CM.feature_columns(False)


def test_rows_to_xy_extracts_features_and_target():
    rows = [{"garbage_density": "0.1", "word_count": "5", "score_raw": "0.8"}]
    x, y = CM.rows_to_xy(rows, ["garbage_density", "word_count"], "score_raw")
    assert x == [[0.1, 5.0]] and y == [0.8]


# ── train.py pure glue ───────────────────────────────────────────────────────


def test_build_train_params_from_config():
    cfg = CM.load_config(_CONFIG)
    args = SimpleNamespace(model=None, batch_size=None, epochs=None, seed=None)
    params = T.build_train_params(cfg, args)
    assert params["model_name"] == "distilbert-base-multilingual-cased"
    assert params["category_head"] is True
    assert params["huber_delta"] == 0.1
    assert params["batch_size"] == 64
    assert params["seed"] == 42


def test_build_train_params_cli_overrides():
    cfg = CM.load_config(_CONFIG)
    args = SimpleNamespace(model="google/canine-s", batch_size=8, epochs=1, seed=7)
    params = T.build_train_params(cfg, args)
    assert params["model_name"] == "google/canine-s"
    assert params["batch_size"] == 8 and params["epochs"] == 1 and params["seed"] == 7


def test_category_ids_mapping():
    rows = [{"categ": "Trash"}, {"categ": "Noisy"}, {"categ": "Clear"}, {"categ": "???"}]
    assert T.category_ids(rows) == [0, 1, 2, 1]  # unknown → Noisy


def test_argmax():
    assert T._argmax([0.1, 0.9, 0.2]) == 1
    assert T._argmax([3.0, 1.0, 2.0]) == 0


def test_compute_metrics_with_category_head():
    compute = T.make_compute_metrics(category_head=True)
    preds = ([0.9, 0.4, 0.6], [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.2, 0.6, 0.2]])
    labels = ([0.95, 0.3, 0.7], [2, 0, 1])
    m = compute((preds, labels))
    assert "reg_mae" in m and "spearman" in m
    assert m["banded_macro_f1"] == 1.0  # pred bands match true bands
    assert m["cathead_macro_f1"] == 1.0  # argmax cats match label ids


def test_compute_metrics_without_category_head():
    compute = T.make_compute_metrics(category_head=False)
    m = compute(([0.9, 0.4], [0.9, 0.4]))
    assert m["reg_mae"] == 0.0
    assert "cathead_macro_f1" not in m


# ── GBM report formatting (pure) ─────────────────────────────────────────────


def test_gbm_format_report():
    report = {
        "dataset": "d.csv",
        "eval_split": "val",
        "variants": [
            {
                "variant": "with_perplexity",
                "regression": {"mae": 0.05, "rmse": 0.07, "spearman": 0.93},
                "banded_category": {"macro_f1": 0.9},
            }
        ],
    }
    text = GBM.format_report(report)
    assert "with_perplexity" in text and "Spearman=0.9300" in text


# ── Slow: real training smokes (skip if libs absent) ─────────────────────────


@pytest.mark.slow
def test_gbm_trains_on_tiny_dataset(tmp_path):
    pytest.importorskip("sklearn")
    import build_dataset as B

    rows = B.read_rows([_ROOT / "tests" / "fixtures" / "quality_model_lines.csv"])
    items, _ = B.build_dataset(rows, B.make_offline_scorer(), seed=1, variants_per_clear=3, ratios=(0.5, 0.25, 0.25))
    ds = tmp_path / "ds.csv"
    B.write_dataset(items, ds)
    report = GBM.run(ds, None, tmp_path / "out", eval_split="val", seed=1)
    assert len(report["variants"]) == 2


@pytest.mark.slow
def test_build_model_constructs_when_torch_present():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model = T.build_model(
        {
            "model_name": "distilbert-base-multilingual-cased",
            "category_head": True,
            "num_categories": 3,
            "huber_delta": 0.1,
            "ce_weight": 0.3,
        }
    )
    assert model is not None
