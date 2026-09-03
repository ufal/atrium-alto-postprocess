#!/usr/bin/env python3
"""
tools/recategorize_from_csv.py
==============================
Offline re-scorer + evaluator for config-constant calibration (#3 / #5).

It re-runs the categorisation logic over an already-produced ``DOC_LINE_CATEG``
CSV **without** FastText or the GPU perplexity model, by reusing the stored
signals:

    * ``perplex``                              — the GPU perplexity (frozen)
    * ``original_lang`` / ``orig_lang_score``  — the raw FastText output (frozen)
    * ``text`` / ``original_text``             — the cleaned and pre-repair lines

Everything downstream of those — ``remap_lang`` (the #3 A1 CAP), the structural
detectors, ``compute_quality_score``, the per-line ``categorize_line`` and the
document-level ``apply_document_postprocessing`` (#3 A3) — is recomputed with the
CURRENT production code. There is exactly ONE scoring engine: the real functions
imported from ``text_util`` / ``classify_TEXT``. Different constant
values are explored by temporarily overriding the module-level tunables with
``text_util.override_constants`` (see ``recategorize_dataframe``) — never a
parallel re-implementation — so the offline numbers match production by
construction.
"""

from __future__ import annotations

import argparse
import configparser
import math
import os
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Make the repo root importable when run as `python tools/recategorize_from_csv.py`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import classify_TEXT as _lc  # noqa: E402
import text_util as _tu  # noqa: E402
from classify_TEXT import (  # noqa: E402
    CSV_HEADER,
    apply_document_postprocessing,
    row_from_signals,
    score_line,
)
from text_util import (  # noqa: E402
    _lang_base,
    override_constants,
    probe_spaced_decimal_metre_candidate,
)

# Modules whose copies of the tunable constants must move in lock-step when a
# trial overrides them: text_util owns them; classify_TEXT imported its
# own bindings via `from text_util import *`.
_CONST_MODULES = (_tu, _lc)

OUTPUT_CATEGORY_ORDER = ("Empty", "Non-text", "Trash", "Noisy", "Clear")


# ---------------------------------------------------------------------------
# Faithful per-line / per-document re-scoring (the ONLY scoring engine)
# ---------------------------------------------------------------------------


def _load_lang_config(config_path: str):
    """Resolve EXPECTED_LANGS / TRUSTED_FOREIGN_LANGS exactly as classify_TEXT.main."""
    config = configparser.ConfigParser()
    config.read(config_path)
    expected = [
        s.strip() for s in config.get("CLASSIFY", "EXPECTED_LANGS", fallback="ces,deu,eng").split(",") if s.strip()
    ]
    trusted = [
        s.strip()
        for s in config.get("CLASSIFY", "TRUSTED_FOREIGN_LANGS", fallback="deu,eng,fra,pol,ita").split(",")
        if s.strip()
    ]
    known_bases = frozenset(_lang_base(code) for code in (trusted + expected))
    return expected, known_bases


def _is_fast_track(row) -> bool:
    """Empty / Non-text rows written by the pre-filter never carry scores."""
    try:
        wc = int(row.get("word_count", 0) or 0)
    except (ValueError, TypeError):
        wc = 0
    return row.get("categ") in ("Empty", "Non-text") and wc == 0


def _text_for_probe(row: Mapping[str, Any]) -> str:
    """Prefer the original OCR text when probing dormant heuristics."""
    text = str(row.get("original_text", "") or "")
    return text if text else str(row.get("text", "") or "")


def _count_spaced_decimal_metre_candidates(frame: pd.DataFrame) -> int:
    """Count rows that hit the dormant metre-spacing probe without scoring."""
    return sum(1 for _, row in frame.iterrows() if probe_spaced_decimal_metre_candidate(_text_for_probe(row)))


def _rescore_row(row: dict, expected_langs, known_bases) -> dict:
    """Recompute one previously-scored line from its frozen signals.

    Delegates to ``classify_TEXT.score_line`` / ``row_from_signals`` — the very
    same functions the live pipeline runs — so this path cannot drift from
    production by construction. Only the model-derived inputs come from the
    frozen CSV columns (``orig_lang_score``, ``perplex``); every other signal is
    recomputed from the stored text exactly as the pipeline computes it.

    Tunables are read inside those functions at call time, so a surrounding
    ``override_constants`` block is honoured.
    """
    text_content = str(row.get("text", "") or "")

    # Read 'original_text' so density/vowels match the live pipeline exactly.
    # Fall back to 'text' for older baseline CSVs missing this column.
    original_text = str(row.get("original_text", "") or "")
    if not original_text:
        original_text = text_content

    original_lang = str(row.get("original_lang", "") or "")
    try:
        original_lang_score = float(row.get("orig_lang_score", 0.0) or 0.0)
    except (ValueError, TypeError):
        original_lang_score = 0.0

    try:
        ppl_val = float(row.get("perplex", 0.0) or 0.0)
    except (ValueError, TypeError):
        ppl_val = 0.0

    sig = score_line(
        text_content=text_content,
        original_text=original_text,
        original_lang=original_lang,
        original_lang_score=original_lang_score,
        perplexity=ppl_val,
        known_lang_bases=known_bases,
        expected_langs=expected_langs,
    )

    out = dict(row)
    out.update(
        row_from_signals(
            sig,
            file_id=row.get("file"),
            page_id=row.get("page_num"),
            line_num=row.get("line_num"),
            text_content=text_content,
            original_text=original_text,
            split_ws=row.get("split_ws"),
            split_we=row.get("split_we"),
            original_lang=original_lang,
            original_lang_score=original_lang_score,
        )
    )
    return out


def _coerce_locators(df: pd.DataFrame) -> pd.DataFrame:
    """Force page_num / line_num to int so ordering is numeric, not lexical."""
    for col in ("page_num", "line_num"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def _recategorize_one_document(doc: pd.DataFrame, expected_langs, known_bases) -> pd.DataFrame:
    """Re-score one document's rows then apply the real page post-processing.

    Index is preserved so callers can realign with the input frame.
    """
    rows: list[dict] = []
    index: list = []
    for idx, r in doc.iterrows():
        rd = r.to_dict()
        index.append(idx)
        rows.append(rd if _is_fast_track(rd) else _rescore_row(rd, expected_langs, known_bases))

    new = pd.DataFrame(rows, index=index)
    if new.empty:
        return new
    new = _coerce_locators(new)
    # The real, byte-identical document smoothing (dedup / surrounded-trash /
    # page-majority + inverted-run sweep). Honours any active override_constants.
    return apply_document_postprocessing(new)


def recategorize_dataframe(
    df: pd.DataFrame,
    constants: Mapping[str, Any] | None = None,
    *,
    expected_langs: list[str] | None = None,
    known_bases: frozenset | None = None,
) -> pd.DataFrame:
    """Faithful, document-aware re-categorisation under an explicit constant set.

    ``constants=None`` uses the live module defaults. Rows are grouped by ``file``
    (one production document per group) and each group is re-scored and smoothed
    independently, exactly like production. The returned frame preserves the input
    row order/index.
    """
    if expected_langs is None or known_bases is None:
        expected_langs, known_bases = _load_lang_config(os.getenv("LANGID_CONFIG", str(_ROOT / "setup/config.txt")))

    work = _coerce_locators(df.copy())

    applied = coerce_constants(dict(constants)) if constants else {}
    if applied:
        validate_constants(applied)

    ctx = override_constants(applied, modules=_CONST_MODULES) if applied else nullcontext()
    frames: list[pd.DataFrame] = []
    with ctx:
        if "file" in work.columns:
            for _, doc in work.groupby("file", sort=False):
                frames.append(_recategorize_one_document(doc, expected_langs, known_bases))
        else:
            frames.append(_recategorize_one_document(work, expected_langs, known_bases))

    if not frames:
        return work
    result = pd.concat(frames)
    # Realign to the original row order; keep only rows we actually processed.
    return result.reindex(work.index)


def rescore_csv(in_path: Path, constants: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (old_df, new_df) for one per-document CSV (diff-report helper)."""
    old = pd.read_csv(in_path, dtype=str, keep_default_na=False)

    # --- Normalize legacy schema to current CSV_HEADER ---
    rename_map = {}
    if "page" in old.columns and "page_num" not in old.columns:
        rename_map["page"] = "page_num"
    if "line" in old.columns and "line_num" not in old.columns:
        rename_map["line"] = "line_num"
    elif "line_order" in old.columns and "line_num" not in old.columns:
        rename_map["line_order"] = "line_num"
    if rename_map:
        old = old.rename(columns=rename_map)

    old = _coerce_locators(old)

    new = recategorize_dataframe(old, constants)
    if not new.empty:
        cols = [c for c in CSV_HEADER if c in new.columns]
        cols += [c for c in new.columns if c not in cols]
        new = new[cols]

    if not old.empty:
        old = old.sort_values(by=["page_num", "line_num"], ascending=True)
    return old, new


# ---------------------------------------------------------------------------
# Tunable inventory + defaults (read from the live modules — never hardcoded,
# so the tool can never drift from config.txt)
# ---------------------------------------------------------------------------

# Production compute_quality_score sums these NINE weights (the legacy symbol
# weight was dropped in #3); validation/normalisation use the same set.
QS_WEIGHT_NAMES = (
    "QS_WEIGHT_VALID_WORD",
    "QS_WEIGHT_WEIRD",
    "QS_WEIGHT_PERPLEXITY",
    "QS_WEIGHT_LENGTH",
    "QS_WEIGHT_GARBAGE",
    "QS_WEIGHT_VOWEL",
    "QS_WEIGHT_LANG",
    "QS_WEIGHT_GIBBERISH",
    "QS_WEIGHT_FUSED",
)

# Everything below is read at call time inside compute_quality_score /
# categorize_line / determine_category / score_words_in_line /
# analyze_rotation_signals / apply_document_postprocessing, so overriding it
# actually moves categories. Pre-filter-only knobs are deliberately excluded.
_THRESHOLD_NAMES = (
    "CATEG_TRASH_SCORE_MAX",
    "CATEG_NOISY_SCORE_MAX",
    "CATEG_GARBAGE_DENSITY_HIGH",
    # (B2) separate QS normalisation scale, decoupled from the hard gate above
    "QS_GARBAGE_NORM_MAX",
    "ROT_RATIO_INVERTED_MIN",
    "WEIRD_RATIO_INVERTED_MIN",
    "PPL_INVERTED_MIN",
    "PERPLEXITY_THRESHOLD_MAX",
    "SHORT_PPL_CAP",
    # (#3) hard-sweep / extreme- and absolute-perplexity trash routes
    "HARD_SWEEP_LANG_MAX",
    "HARD_SWEEP_PPL_MIN",
    "PPL_EXTREME_MIN",
    "EXTREME_LANG_CONF",
    "PPL_GARBAGE_ABSOLUTE",
    # (#3) low-ppl Clear + LM-confident-Czech recovery + mostly-readable cap
    "LOWPPL_CLEAR_MAX",
    "LOWPPL_CZECH_CLEAR_MAX",
    "CZECH_CLEAR_GARBAGE_MAX",
    "MOSTLY_READABLE_VALID_MIN",
    "SHORT_NOISY_QS_PENALTY",
    "WORD_W_PENALTY",
    # (#3) rotation / inversion organic penalties + per-line route
    "GHOST_DOMINATED_MIN_RATIO",
    "SUSPICIOUS_ROT_RATIO",
    "SUSPICIOUS_WQX_RATIO",
    "INVERTED_WEIRD_PENALTY",
    "GHOST_HITS_INVERTED_MIN",
    "ROT_HIGH_LANG_CONF",
    "LANG_SCORE_ROUGH",
    # (#3 A3) page-level smoothing
    "INVERTED_RUN_MIN",
    "INVERTED_PAGE_MAJORITY",
    # (#5) page-context smoothing thresholds
    "SURROUNDED_TRASH_QS_MARGIN",
    "PAGE_GARBAGE_CLEAR_MAX",
    "PAGE_GARBAGE_LANG_MAX",
    "PAGE_GARBAGE_MEDIAN_QS_MAX",
    "PAGE_GARBAGE_NOISY_QS_MAX",
    "PAGE_CLEAN_CLEAR_MIN",
    "PAGE_CLEAN_MEDIAN_QS_MIN",
    "PAGE_CLEAN_RECOVER_QS_MIN",
)

TUNABLE_CONSTANTS = QS_WEIGHT_NAMES + _THRESHOLD_NAMES

# Constants that must stay integral.
INT_CONSTANTS = frozenset({"GHOST_HITS_INVERTED_MIN", "INVERTED_RUN_MIN"})


def _live_default(name: str) -> float | int:
    """Current value of a tunable, read from the live production modules."""
    for mod in _CONST_MODULES:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"Tunable constant {name!r} is not defined on the production modules")


# Live snapshot of the current config — the sweep's base point and the
# re-scorer's defaults. Reflects config.txt exactly (no hardcoded drift).
DEFAULT_CONSTANTS: dict[str, float | int] = {name: _live_default(name) for name in TUNABLE_CONSTANTS}


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def _parse_scalar(value: str) -> float | int | bool | str:
    raw = value.strip()
    lowered = raw.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        if re.fullmatch(r"[+-]?\d+", raw):
            return int(raw)
        return float(raw)
    except ValueError:
        return raw


def read_config_constants(config_path: Path | str | None) -> dict[str, Any]:
    """Read tunable constants from a config.txt-style INI file.

    Interpolation is disabled because the config may contain literal ``%``
    characters (punctuation/symbol strings). All sections are scanned
    case-sensitively and only known tunable constants are extracted; anything
    absent falls back to the live module default.
    """
    constants = dict(DEFAULT_CONSTANTS)
    if config_path is None:
        return constants
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#", ";"),
        strict=False,
    )
    parser.optionxform = str
    parser.read(config_path, encoding="utf-8")

    known = set(TUNABLE_CONSTANTS)
    for section in parser.sections():
        for key, value in parser.items(section, raw=True):
            if key in known:
                constants[key] = _parse_scalar(value)
    return constants


def parse_overrides(overrides: Iterable[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in TUNABLE_CONSTANTS:
            raise ValueError(f"Unknown tunable constant {key!r}. Known constants: {', '.join(TUNABLE_CONSTANTS)}")
        parsed[key] = _parse_scalar(value)
    return parsed


def coerce_constants(constants: Mapping[str, Any]) -> dict[str, Any]:
    """Coerce int constants to int and the remaining tunables to float."""
    out = dict(constants)
    for key in INT_CONSTANTS:
        if key in out:
            out[key] = int(out[key])
    for key in TUNABLE_CONSTANTS:
        if key in INT_CONSTANTS or key not in out:
            continue
        out[key] = float(out[key])
    return out


def validate_constants(constants: Mapping[str, Any]) -> None:
    """Fail fast for logically invalid configurations."""

    def _g(name):
        return float(constants[name]) if name in constants else float(_live_default(name))

    if _g("CATEG_TRASH_SCORE_MAX") >= _g("CATEG_NOISY_SCORE_MAX"):
        raise ValueError("Invalid constants: CATEG_TRASH_SCORE_MAX must be < CATEG_NOISY_SCORE_MAX")
    if _g("SHORT_PPL_CAP") >= _g("PERPLEXITY_THRESHOLD_MAX"):
        raise ValueError("Invalid constants: SHORT_PPL_CAP must be < PERPLEXITY_THRESHOLD_MAX")
    if sum(_g(name) for name in QS_WEIGHT_NAMES) <= 0:
        raise ValueError("Invalid constants: sum(QS_WEIGHT_*) must be positive")


# ---------------------------------------------------------------------------
# Data loading + helpers
# ---------------------------------------------------------------------------


def csv_paths(input_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def load_csvs(input_dir: Path, recursive: bool = False) -> pd.DataFrame:
    """Concatenate per-document CSVs, preserving a ``file`` column for grouping.

    Read as strings with NA disabled so the offline path sees the same raw cell
    values that ``rescore_csv`` does (consistent dtype/NA handling).
    """
    paths = csv_paths(input_dir, recursive=recursive)
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df["_source_file"] = str(path.relative_to(input_dir))
        if "file" not in df.columns:
            df["file"] = path.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def normalize_category(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    raw = str(value).strip()
    lowered = raw.lower().replace("_", "-")
    mapping = {
        "empty": "Empty",
        "non-text": "Non-text",
        "nontext": "Non-text",
        "trash": "Trash",
        "noisy": "Noisy",
        "clear": "Clear",
    }
    return mapping.get(lowered, raw)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def confusion_matrix_dict(original: Iterable[Any], predicted: Iterable[Any]) -> dict[str, dict[str, int]]:
    orig = [normalize_category(v) for v in original]
    pred = [normalize_category(v) for v in predicted]
    labels = sorted(set(orig) | set(pred) | set(OUTPUT_CATEGORY_ORDER))
    table = pd.crosstab(
        pd.Series(orig, name="original"),
        pd.Series(pred, name="predicted"),
        dropna=False,
    )
    result: dict[str, dict[str, int]] = {}
    for row_label in labels:
        result[row_label] = {}
        for col_label in labels:
            result[row_label][col_label] = int(
                table.loc[row_label, col_label] if row_label in table.index and col_label in table.columns else 0
            )
    return result


def f1_scores(confusion: dict[str, dict[str, int]]) -> dict[str, Any]:
    labels = list(confusion.keys())
    per_label: dict[str, float] = {}
    supports: dict[str, int] = {}
    for label in labels:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[row].get(label, 0) for row in labels) - tp
        fn = sum(confusion[label].values()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        per_label[label] = f1
        supports[label] = sum(confusion[label].values())
    total = sum(supports.values())
    macro_f1 = sum(per_label.values()) / len(per_label) if per_label else 0.0
    weighted_f1 = sum(per_label[label] * supports[label] for label in labels) / total if total else 0.0
    return {
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class_f1": per_label,
        "per_class_support": supports,
    }


def kl_divergence_from_counts(baseline_counts: dict[str, int], new_counts: dict[str, int]) -> float:
    labels = sorted(set(baseline_counts) | set(new_counts) | set(OUTPUT_CATEGORY_ORDER))
    p = np.array([baseline_counts.get(label, 0) for label in labels], dtype="float64")
    q = np.array([new_counts.get(label, 0) for label in labels], dtype="float64")
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    eps = 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum((p + eps) * np.log((p + eps) / (q + eps))))


def costed_flip_score(original: Iterable[Any], predicted: Iterable[Any]) -> float:
    """Operationally weighted per-line penalty; high cost to losing usable text."""
    cost = {
        ("Clear", "Trash"): 3.0,
        ("Clear", "Non-text"): 3.0,
        ("Clear", "Noisy"): 1.0,
        ("Noisy", "Trash"): 2.0,
        ("Noisy", "Non-text"): 2.0,
        ("Noisy", "Clear"): 0.5,
        ("Trash", "Clear"): 2.0,
        ("Trash", "Noisy"): 1.0,
        ("Non-text", "Clear"): 2.0,
        ("Empty", "Clear"): 2.0,
    }
    total = 0.0
    count = 0
    for old_raw, new_raw in zip(original, predicted, strict=False):
        old = normalize_category(old_raw)
        new = normalize_category(new_raw)
        if old != new:
            total += cost.get((old, new), 1.0)
        count += 1
    return float(total / count) if count else 0.0


def _metrics_from_labels(original: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    total = len(original)
    flip_count = int(np.sum(original != predicted))
    flip_rate = float(flip_count / total) if total else 0.0
    baseline_counts = Counter(original)
    predicted_counts = Counter(predicted)
    confusion = confusion_matrix_dict(original, predicted)
    f1 = f1_scores(confusion)
    kl = kl_divergence_from_counts(dict(baseline_counts), dict(predicted_counts))
    cost = costed_flip_score(original, predicted)
    category_rates = {
        label: float(predicted_counts.get(label, 0) / total) if total else 0.0 for label in OUTPUT_CATEGORY_ORDER
    }
    return {
        "line_count": int(total),
        "flip_count": flip_count,
        "flip_rate": flip_rate,
        "category_counts": dict(predicted_counts),
        "category_rates": category_rates,
        "trash_rate": category_rates.get("Trash", 0.0),
        "clear_rate": category_rates.get("Clear", 0.0),
        "baseline_category_counts": dict(baseline_counts),
        "confusion": confusion,
        "macro_f1": f1["macro_f1"],
        "weighted_f1": f1["weighted_f1"],
        "per_class_f1": f1["per_class_f1"],
        "per_class_support": f1["per_class_support"],
        "kl_divergence": kl,
        "costed_score": cost,
    }


def evaluate_dataframe(
    df: pd.DataFrame,
    constants: Mapping[str, Any] | None = None,
    *,
    original_category_column: str = "categ",
    expected_langs: list[str] | None = None,
    known_bases: frozenset | None = None,
) -> dict[str, Any]:
    """Faithfully re-categorise ``df`` under ``constants`` and score it against the
    stored categories. The evaluation runs the real production engine
    (document-aware, with page post-processing).
    """
    if original_category_column in df.columns:
        original = df[original_category_column].map(normalize_category).to_numpy()
    elif "orig_categ" in df.columns:
        original = df["orig_categ"].map(normalize_category).to_numpy()
    else:
        original = None

    predicted_df = recategorize_dataframe(df, constants, expected_langs=expected_langs, known_bases=known_bases)
    predicted = predicted_df["categ"].map(normalize_category).to_numpy()

    if original is None:
        original = predicted.copy()
    return _metrics_from_labels(original, predicted)


def evaluate_per_document(
    df: pd.DataFrame,
    constants: Mapping[str, Any] | None = None,
    *,
    original_category_column: str = "categ",
) -> dict[str, dict[str, Any]]:
    """Per-document metrics (importance can be collection-specific)."""
    group_col = "file" if "file" in df.columns else ("_source_file" if "_source_file" in df.columns else None)
    if group_col is None:
        return {"<all>": evaluate_dataframe(df, constants, original_category_column=original_category_column)}
    out: dict[str, dict[str, Any]] = {}
    for name, doc in df.groupby(group_col, sort=True):
        out[str(name)] = evaluate_dataframe(doc, constants, original_category_column=original_category_column)
    return out


def find_parity_mismatches(
    df: pd.DataFrame,
    constants: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Return line-level category mismatches between stored categories and the
    production-equivalent re-score.
    """
    predicted = recategorize_dataframe(df, constants)

    if len(predicted) != len(df):
        raise ValueError(
            f"recategorize_dataframe changed the number of rows: input={len(df)}, predicted={len(predicted)}"
        )

    result = df.copy()

    stored = result["categ"].map(normalize_category)
    predicted_categories = predicted["categ"].map(normalize_category)

    result["stored_category"] = stored.to_numpy()
    result["predicted_category"] = predicted_categories.to_numpy()

    result["category_changed"] = result["stored_category"] != result["predicted_category"]

    return result.loc[result["category_changed"]].copy()


# ---------------------------------------------------------------------------
# Diff report (CLI)
# ---------------------------------------------------------------------------


def _category_counts(df: pd.DataFrame):
    if "categ" not in df.columns:
        return {}
    return df["categ"].value_counts().to_dict()


def _report(in_path: Path, old: pd.DataFrame, new: pd.DataFrame) -> int:
    """Print a before/after report; return the number of changed lines."""
    oc, nc = _category_counts(old), _category_counts(new)
    cats = sorted(set(oc) | set(nc))
    print(f"\n=== {in_path.name} ({len(old)} lines) ===")
    print(f"  {'category':<10} {'before':>7} {'after':>7} {'delta':>7}")
    for c in cats:
        b, a = oc.get(c, 0), nc.get(c, 0)
        print(f"  {c:<10} {b:>7} {a:>7} {a - b:>+7}")

    changed = 0
    if "categ" in old.columns and "categ" in new.columns and len(old) == len(new):
        old_cat = old["categ"].reset_index(drop=True)
        new_cat = new["categ"].reset_index(drop=True)
        diff_mask = old_cat != new_cat
        changed = int(diff_mask.sum())
        if changed:
            print(f"  --- {changed} line(s) changed category ---")
            txt = new["text"].reset_index(drop=True) if "text" in new.columns else None
            shown = 0
            for i in diff_mask[diff_mask].index:
                snippet = str(txt.iloc[i])[:48] if txt is not None else ""
                print(f"    L{i:<4} {old_cat.iloc[i]:>8} -> {new_cat.iloc[i]:<8} | {snippet}")
                shown += 1
                if shown >= 25:
                    print(f"    … (+{changed - shown} more)")
                    break
    return changed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Faithfully re-categorise DOC_LINE_CATEG CSVs under a chosen constant set "
            "(no GPU/model inference). Inputs are read-only; revised CSVs go to --out."
        )
    )
    ap.add_argument("path", nargs="?", help="CSV file or directory of per-document CSVs")
    ap.add_argument("--input-dir", dest="input_dir", help="Alias for the positional path (a directory).")
    ap.add_argument("--out", "--output-dir", dest="out", help="Output file/dir (default: overwrite in place).")
    ap.add_argument("--config", help="setup/config.txt-style INI to source constants from.")
    ap.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Override individual constants, e.g. CATEG_TRASH_SCORE_MAX=0.45.",
    )
    ap.add_argument("--report-only", action="store_true", help="Print the diff report but do not write CSVs.")
    ap.add_argument(
        "--probe-metre-candidate",
        action="store_true",
        help="Also report hits for the dormant spaced-decimal metre probe.",
    )
    return ap


def _resolve_constants(args) -> dict[str, Any] | None:
    if not args.config and not args.override:
        return None
    constants = read_config_constants(Path(args.config)) if args.config else dict(DEFAULT_CONSTANTS)
    constants.update(parse_overrides(args.override))
    constants = coerce_constants(constants)
    validate_constants(constants)
    return constants


def main(argv=None):
    args = build_parser().parse_args(argv)

    raw_path = args.path or args.input_dir
    if not raw_path:
        print("error: provide a CSV path or --input-dir", file=sys.stderr)
        return 2
    in_path = Path(raw_path)

    if in_path.is_dir():
        csvs = sorted(in_path.glob("*.csv"))
    else:
        csvs = [in_path]
    if not csvs:
        print(f"No CSV files found at {in_path}", file=sys.stderr)
        return 1

    constants = _resolve_constants(args)
    if constants:
        print(
            f"Applying {sum(1 for k in constants if constants[k] != DEFAULT_CONSTANTS.get(k))} non-default constant(s)."
        )

    total_changed = 0
    grand_old: dict = {}
    grand_new: dict = {}
    for csv_path in csvs:
        old, new = rescore_csv(csv_path, constants)
        total_changed += _report(csv_path, old, new)

        if args.probe_metre_candidate:
            candidate_hits = _count_spaced_decimal_metre_candidates(old)
            print(f"  dormant metre-spacing probe hits: {candidate_hits}")

        for k, v in _category_counts(old).items():
            grand_old[k] = grand_old.get(k, 0) + v
        for k, v in _category_counts(new).items():
            grand_new[k] = grand_new.get(k, 0) + v

        if not args.report_only:
            if args.out:
                out_path = Path(args.out)
                if in_path.is_dir():
                    out_path.mkdir(parents=True, exist_ok=True)
                    out_path = out_path / csv_path.name
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = csv_path
            new.to_csv(out_path, index=False, encoding="utf-8")

    if len(csvs) > 1:
        print("\n=== GRAND TOTAL ===")
        for c in sorted(set(grand_old) | set(grand_new)):
            b, a = grand_old.get(c, 0), grand_new.get(c, 0)
            print(f"  {c:<10} {b:>7} {a:>7} {a - b:>+7}")
        print(f"  total lines changed category: {total_changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
