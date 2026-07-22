"""Shared pure-Python helpers for the quality-model training/eval scripts (issue #23).

Config parsing, dataset IO, feature extraction, band mapping and metrics live here
so ``train_baseline_gbm.py``, ``train.py`` and (later) ``evaluate.py`` share one
implementation. Everything in this module is dependency-light: no numpy / sklearn /
torch, so it imports and is unit-tested without the ML stack. The band thresholds
are read from the production ``text_util_langID`` so training/eval banding can never
drift from the categoriser.
"""

from __future__ import annotations

import configparser
import csv
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import text_util_langID as tu  # noqa: E402  (path bootstrap must run first)

# Band thresholds — reused from production so training/eval agree with categorize_line.
TRASH_MAX = tu.CATEG_TRASH_SCORE_MAX  # 0.55
NOISY_MAX = tu.CATEG_NOISY_SCORE_MAX  # 0.80
CATEGORIES = ["Trash", "Noisy", "Clear"]

# Default numeric feature columns present in a build_dataset.py CSV. "perplex" is
# split out so a baseline can be trained with and without the Qwen signal.
FEATURE_COLUMNS_NO_PPL = [
    "word_count",
    "char_count",
    "garbage_density",
    "word_weird",
    "vowel_ratio",
    "rot_ratio",
    "fused_words",
    "gibberish",
    "lang_score",
]
PERPLEXITY_COLUMN = "perplex"

DEFAULT_TARGET = "score_raw"
DEFAULT_TEXT = "text"
DEFAULT_CATEG = "categ"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> configparser.ConfigParser:
    """Load a quality-model config (same .ini/.txt style as config_langID.txt)."""
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    read = cfg.read(path, encoding="utf-8")
    if not read:
        raise FileNotFoundError(f"quality-model config not found: {path}")
    return cfg


def cfg_get(cfg, section: str, key: str, default, cast=str):
    if cfg is None or not cfg.has_option(section, key):
        return default
    raw = cfg.get(section, key)
    if cast is bool:
        return str(raw).strip().lower() in ("1", "true", "yes", "on")
    if cast is list:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return cast(raw)


# ---------------------------------------------------------------------------
# Dataset IO
# ---------------------------------------------------------------------------


def read_dataset(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def split_rows(rows: list[dict], split_col: str = "split") -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for r in rows:
        out.setdefault(r.get(split_col, "train"), []).append(r)
    return out


def feature_columns(include_perplexity: bool) -> list[str]:
    cols = list(FEATURE_COLUMNS_NO_PPL)
    if include_perplexity:
        cols.append(PERPLEXITY_COLUMN)
    return cols


def rows_to_xy(rows: list[dict], feature_cols: list[str], target_col: str = DEFAULT_TARGET):
    """Extract a numeric feature matrix and target vector from dataset rows."""
    x, y = [], []
    for r in rows:
        x.append([_to_float(r.get(c)) for c in feature_cols])
        y.append(_to_float(r.get(target_col)))
    return x, y


def rows_to_categories(rows: list[dict], categ_col: str = DEFAULT_CATEG) -> list[str]:
    return [r.get(categ_col, "") for r in rows]


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Banding
# ---------------------------------------------------------------------------


def band_category(score: float, trash_max: float = TRASH_MAX, noisy_max: float = NOISY_MAX) -> str:
    """Map a continuous score to the production category band (Trash/Noisy/Clear)."""
    if score < trash_max:
        return "Trash"
    if score < noisy_max:
        return "Noisy"
    return "Clear"


# ---------------------------------------------------------------------------
# Metrics (manual — no numpy/scipy so they are testable without the ML stack)
# ---------------------------------------------------------------------------


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mae": 0.0, "rmse": 0.0, "pearson": 0.0, "spearman": 0.0}
    abs_err = [abs(a - b) for a, b in zip(y_true, y_pred, strict=True)]
    sq_err = [(a - b) ** 2 for a, b in zip(y_true, y_pred, strict=True)]
    return {
        "n": n,
        "mae": round(sum(abs_err) / n, 6),
        "rmse": round(math.sqrt(sum(sq_err) / n), 6),
        "pearson": round(_pearson(y_true, y_pred), 6),
        "spearman": round(_pearson(_ranks(y_true), _ranks(y_pred)), 6),
    }


def _pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n == 0:
        return 0.0
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _ranks(values: list[float]) -> list[float]:
    """Fractional (average) ranks, so ties don't bias Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def category_metrics(y_true: list[str], y_pred: list[str], labels: list[str] | None = None) -> dict:
    labels = labels or CATEGORIES
    n = len(y_true)
    correct = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b)
    per_label = {}
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == lab and b != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_label[lab] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}
        f1s.append(f1)
    return {
        "n": n,
        "accuracy": round(correct / n, 4) if n else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
        "per_label": per_label,
    }


def banded_category_metrics(
    y_true_scores: list[float],
    y_pred_scores: list[float],
    trash_max: float = TRASH_MAX,
    noisy_max: float = NOISY_MAX,
) -> dict:
    """Band both score vectors, then score category agreement."""
    true_cats = [band_category(s, trash_max, noisy_max) for s in y_true_scores]
    pred_cats = [band_category(s, trash_max, noisy_max) for s in y_pred_scores]
    return category_metrics(true_cats, pred_cats)
