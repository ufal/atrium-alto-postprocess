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
    apply_page_perplexity_blend,
    row_from_signals,
    score_line,
)
from text_util import (  # noqa: E402
    DEFAULT_EXPECTED_LANGS,
    DEFAULT_TRUSTED_FOREIGN_LANGS,
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


def _load_lang_config(config_path: str | None):
    """Resolve EXPECTED_LANGS / TRUSTED_FOREIGN_LANGS exactly as classify_TEXT.main.

    Two things this deliberately does NOT do, both of which it used to.

    It no longer carries its own copy of the fallback strings. The offline copy
    had drifted from the shipped one -- it was missing ``slk`` -- so a Slovak
    line reached the guards at ``TRUST_TIER_UNKNOWN`` (0.50) here and
    ``TRUST_TIER_TRUSTED`` (0.85) in production. The strings now live in
    ``text_util`` and are imported by both callers.

    And it no longer degrades silently. ``configparser.read()`` ignores a path
    that does not exist, so a typo'd or stale ``--config`` used to fall through
    to the defaults and produce a confident, differently-scored run. Every
    documented command in this repository named ``config.txt``, which has not
    existed at the repo root since the file moved to ``setup/``. A named config
    that cannot be read, or that has no ``[CLASSIFY]`` section, is now an error.
    """
    config = configparser.ConfigParser()

    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Config file does not exist: {config_path}. "
                "Refusing to fall back to built-in language defaults -- that silently "
                "changes the trust tier of every non-expected language. "
                "Did you mean setup/config.txt?"
            )
        read_ok = config.read(path)
        if not read_ok:
            raise ValueError(f"Config file could not be parsed: {config_path}")
        if not config.has_section("CLASSIFY"):
            raise ValueError(
                f"Config file {config_path} has no [CLASSIFY] section, so EXPECTED_LANGS / "
                "TRUSTED_FOREIGN_LANGS would silently fall back to the built-in defaults."
            )

    expected = [
        s.strip()
        for s in config.get("CLASSIFY", "EXPECTED_LANGS", fallback=DEFAULT_EXPECTED_LANGS).split(",")
        if s.strip()
    ]
    trusted = [
        s.strip()
        for s in config.get("CLASSIFY", "TRUSTED_FOREIGN_LANGS", fallback=DEFAULT_TRUSTED_FOREIGN_LANGS).split(",")
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

    # Prefer the uncapped `perplex_raw`: `perplex` may already carry
    # SHORT_PPL_CAP (or a page-blended value), and re-scoring from that would
    # feed a derived number back through the cap on every pass. Older CSVs
    # written before the column existed fall back to `perplex`.
    ppl_source = row.get("perplex_raw", "")
    if ppl_source in (None, "", "nan"):
        ppl_source = row.get("perplex", 0.0)

    try:
        ppl_val = float(ppl_source or 0.0)
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
    # Page-relative perplexity blend runs before smoothing, in both paths, so
    # the offline re-scorer stays byte-identical to production. No-op with
    # PAGE_PPL_BLEND_ENABLE off.
    new = apply_page_perplexity_blend(new, known_lang_bases=known_bases, expected_langs=expected_langs)
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
        # `new` comes back in the INPUT frame's order, so sorting only `old`
        # leaves the two frames misaligned whenever a CSV is not already stored
        # in (page_num, line_num) order -- and `_report` compares them
        # positionally. Reversing the row order of a single document was enough
        # to make it report 4 phantom category changes out of 9 lines while the
        # category COUNTS stayed identical. Realign on the index, which both
        # frames preserve, so the diff is about categories and never about order.
        new = new.reindex(old.index)
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
    # PAGE_PPL_BLEND_WEIGHT / PAGE_PPL_LONG_MIN_WC / PAGE_PPL_MIN_LONG_LINES were
    # here. They are inert: apply_page_perplexity_blend() returns early while
    # PAGE_PPL_BLEND_ENABLE is false, which it is. See _DELIBERATELY_NOT_TUNABLE
    # in tests/test_recategorize_parity.py for the reason and the way back.
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
INT_CONSTANTS = frozenset(
    {
        "GHOST_HITS_INVERTED_MIN",
        "INVERTED_RUN_MIN",
        "PAGE_PPL_LONG_MIN_WC",
        "PAGE_PPL_MIN_LONG_LINES",
    }
)


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


def short_cap_arms_hard_sweep(constants: Mapping[str, Any] | None = None) -> bool:
    """True when perplexity can reach ``rule_hard_sweep`` on a 1-2 token line.

    At the shipped defaults it cannot: ``SHORT_PPL_CAP`` (850) sits below
    ``HARD_SWEEP_PPL_MIN`` (1000), so a short line's perplexity is flattened to
    the cap before any perplexity rule sees it. That is not an incidental
    ordering -- issue #30 measured **58,427 notation lines that are `Clear`
    only because of it**, 50,221 of them sitting exactly at the cap. They are
    `Clear` because the cap flattened their perplexity, not because the language
    gate cleared them.

    Raising the cap above the hard-sweep floor arms that route at ``wc <= 2``
    for the first time and puts those lines in reach of `Trash`. It is a
    legitimate configuration to explore -- ``SHORT_PPL_CAP`` sweeps [300, 950]
    and ``HARD_SWEEP_PPL_MIN`` sweeps [500, 3000], so the joint move is inside
    the search space even though no single-constant move reaches it -- but it
    should be an informed choice, not something a sweep stumbles into. Callers
    that report it are doing the informing.
    """

    def _g(name):
        if constants and name in constants:
            return float(constants[name])
        return float(_live_default(name))

    return _g("SHORT_PPL_CAP") > _g("HARD_SWEEP_PPL_MIN")


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


def confusion_matrix_dict(
    original: Iterable[Any],
    predicted: Iterable[Any],
    sample_weight: "np.ndarray | None" = None,
) -> dict[str, dict[str, float]]:
    """Confusion counts, or weighted mass when ``sample_weight`` is given.

    Cells stay ints in the unweighted case so every existing caller and every
    committed expectation is untouched; with weights they become floats, because
    a design weight is not a count and rounding it to one would quietly discard
    the reweighting.
    """
    orig = [normalize_category(v) for v in original]
    pred = [normalize_category(v) for v in predicted]
    labels = sorted(set(orig) | set(pred) | set(OUTPUT_CATEGORY_ORDER))
    table = pd.crosstab(
        pd.Series(orig, name="original"),
        pd.Series(pred, name="predicted"),
        values=None if sample_weight is None else pd.Series(sample_weight, dtype=float),
        aggfunc=None if sample_weight is None else "sum",
        dropna=False,
    )
    cast = int if sample_weight is None else float
    result: dict[str, dict[str, float]] = {}
    for row_label in labels:
        result[row_label] = {}
        for col_label in labels:
            raw = table.loc[row_label, col_label] if row_label in table.index and col_label in table.columns else 0
            if raw != raw:  # NaN, which crosstab emits for an empty weighted cell
                raw = 0
            result[row_label][col_label] = cast(raw)
    return result


def f1_scores(confusion: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Per-class and averaged F1 from a confusion table.

    Works on weighted mass as well as counts -- every term is a ratio, so the
    unit cancels. ``weighted_f1`` averages by CLASS SUPPORT and is not the
    design-weighted figure; sampling weights enter through the confusion table.
    """
    labels = list(confusion.keys())
    per_label: dict[str, float] = {}
    supports: dict[str, float] = {}
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


def kl_divergence_from_counts(baseline_counts: dict[str, float], new_counts: dict[str, float]) -> float:
    labels = sorted(set(baseline_counts) | set(new_counts) | set(OUTPUT_CATEGORY_ORDER))
    p = np.array([baseline_counts.get(label, 0) for label in labels], dtype="float64")
    q = np.array([new_counts.get(label, 0) for label in labels], dtype="float64")
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    eps = 1e-12
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum((p + eps) * np.log((p + eps) / (q + eps))))


def costed_flip_score(
    original: Iterable[Any],
    predicted: Iterable[Any],
    sample_weight: "np.ndarray | None" = None,
) -> float:
    """Operationally weighted per-line penalty; high cost to losing usable text.

    ``sample_weight`` is the SAMPLING weight and multiplies each line's penalty;
    the cost table below is the operational weight and is unrelated to it.
    """
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
    # Materialised once: `original` is an Iterable and may be a one-shot
    # generator, so it must not be walked twice to size the weights.
    weights = None if sample_weight is None else list(sample_weight)
    total = 0.0
    count = 0.0
    for i, (old_raw, new_raw) in enumerate(zip(original, predicted, strict=False)):
        w = 1.0 if weights is None else (weights[i] if i < len(weights) else 1.0)
        old = normalize_category(old_raw)
        new = normalize_category(new_raw)
        if old != new:
            total += cost.get((old, new), 1.0) * w
        count += w
    return float(total / count) if count else 0.0


def _metrics_from_labels(
    original: np.ndarray,
    predicted: np.ndarray,
    sample_weight: "np.ndarray | None" = None,
) -> dict[str, Any]:
    """Every metric for one (reference, prediction) pair.

    ``sample_weight`` is optional and opt-in; without it every number is exactly
    what it was before. With it, counts become weighted mass and every rate is a
    weighted rate, so a stratified gold set can be scored against the population
    it is meant to represent rather than against its own composition.

    ``line_count`` stays the unweighted row count, because it answers "how many
    lines is this measured on" -- a question the weights must not change.
    """
    total = len(original)
    is_flip = original != predicted
    if sample_weight is None:
        flip_count = int(np.sum(is_flip))
        flip_rate = float(flip_count / total) if total else 0.0
        baseline_counts = Counter(original)
        predicted_counts = Counter(predicted)
        rate_total = float(total)
    else:
        w = np.asarray(sample_weight, dtype=float)
        mass = float(w.sum())
        flip_count = int(np.sum(is_flip))
        flip_rate = float(w[is_flip].sum() / mass) if mass else 0.0
        baseline_counts = Counter()
        predicted_counts = Counter()
        for label, weight in zip(original, w, strict=False):
            baseline_counts[label] += float(weight)
        for label, weight in zip(predicted, w, strict=False):
            predicted_counts[label] += float(weight)
        rate_total = mass
    confusion = confusion_matrix_dict(original, predicted, sample_weight=sample_weight)
    f1 = f1_scores(confusion)
    kl = kl_divergence_from_counts(dict(baseline_counts), dict(predicted_counts))
    cost = costed_flip_score(original, predicted, sample_weight=sample_weight)
    category_rates = {
        label: float(predicted_counts.get(label, 0) / rate_total) if rate_total else 0.0
        for label in OUTPUT_CATEGORY_ORDER
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


def _stored_labels(df: pd.DataFrame, original_category_column: str) -> np.ndarray | None:
    """The pipeline's own labels, or None when the frame carries none."""
    if original_category_column in df.columns:
        return df[original_category_column].map(normalize_category).to_numpy()
    if "orig_categ" in df.columns:
        return df["orig_categ"].map(normalize_category).to_numpy()
    return None


GOLD_WEIGHT_COLUMN = "gold_weight"


def _gold_weights(df: pd.DataFrame, keep) -> "np.ndarray | None":
    """Per-row sampling weights for the annotated rows, or None when unweighted.

    Optional and opt-in: a sidecar that carries no ``gold_weight`` column scores
    unweighted, exactly as before. It exists because this gold set is stratified
    and the strata are inverted relative to the population it is used to tune --
    the 1920s are ~95x over-represented and the 2010s ~10x under-represented,
    which moves headline agreement by about 10 points on its own. An unweighted
    objective silently tunes for the decades that are cheap to annotate.

    Note this is NOT the same thing as ``weighted_f1``, which weights by CLASS
    support. Anyone reading that field as the design-weighted figure is reading
    the wrong number; `_print_gold_report` now says so.
    """
    if GOLD_WEIGHT_COLUMN not in df.columns:
        return None
    raw = pd.to_numeric(df.loc[keep, GOLD_WEIGHT_COLUMN], errors="coerce")
    if raw.isna().any():
        bad = int(raw.isna().sum())
        raise ValueError(
            f"{GOLD_WEIGHT_COLUMN!r} has {bad} non-numeric value(s) on annotated rows. "
            "A silently-dropped weight is a silently-reweighted objective."
        )
    if (raw < 0).any():
        raise ValueError(f"{GOLD_WEIGHT_COLUMN!r} contains negative weights")
    if float(raw.sum()) <= 0.0:
        raise ValueError(f"{GOLD_WEIGHT_COLUMN!r} sums to zero across the annotated rows")
    return raw.to_numpy(dtype=float)


def annotated_mask(df: pd.DataFrame, gold_column: str) -> pd.Series:
    """Rows carrying an actual human label, as a boolean Series aligned to ``df``.

    THE single definition of "annotated", because there used to be two and only
    one of them was right. ``_gold_report`` masked; ``evaluate_dataframe`` -- the
    path every sweep, ablation and A/B trial goes through -- did not.

    That mattered the moment gold arrived as a key-indexed sidecar. A join leaves
    every unmatched row blank, ``normalize_category`` maps blank to ``""``, and an
    unmasked ``""`` becomes a sixth category in the confusion matrix. On a 15-row
    frame with 2 rows annotated it reported ``line_count=15``, ``macro_f1=0.028``
    and a ``""`` class with support 13; on a real corpus, where the annotated
    share is well under 1%, ``macro_f1`` stops measuring agreement with humans
    and becomes a monotone function of how much of the corpus is unannotated.
    A sweep maximising it would have optimised the annotation rate for hours.
    """
    if gold_column not in df.columns:
        raise KeyError(
            f"gold column {gold_column!r} is not in the frame; "
            f"available columns: {sorted(df.columns)}. Refusing to fall back to the "
            "stored categories -- see tools/gold/GOLD.md for the expected schema."
        )
    return df[gold_column].fillna("").astype(str).str.strip() != ""


def evaluate_dataframe(
    df: pd.DataFrame,
    constants: Mapping[str, Any] | None = None,
    *,
    original_category_column: str = "categ",
    gold_category_column: str | None = None,
    expected_langs: list[str] | None = None,
    known_bases: frozenset | None = None,
) -> dict[str, Any]:
    """Faithfully re-categorise ``df`` under ``constants`` and score the result.

    The re-categorisation always runs the real production engine (document-aware,
    with page post-processing). What the result is scored *against* depends on
    ``gold_category_column``:

    * ``None`` (default) -- score against the frame's own stored ``categ``. This is
      the historical behaviour and it is **self-referential**: the shipped config
      is the optimum by construction, and a genuine accuracy improvement scores as
      pure damage. Use it to measure drift and parity, never to choose constants.
    * a column name -- score against human gold labels in that column. The metrics
      then mean agreement-with-gold, and ``baseline_vs_gold`` carries the same
      metrics for the pipeline's *stored* labels so a trial can be compared against
      the status quo rather than against itself.

    A missing gold column is an error, never a silent fallback: scoring predictions
    against themselves yields a perfect score, which is exactly the kind of quiet
    no-op this repository has been bitten by before.
    """
    stored = _stored_labels(df, original_category_column)

    predicted_df = recategorize_dataframe(df, constants, expected_langs=expected_langs, known_bases=known_bases)
    predicted = predicted_df["categ"].map(normalize_category).to_numpy()

    if gold_category_column is not None:
        # Score ONLY the annotated rows. `annotated_mask` explains why at length;
        # the short version is that an unmatched sidecar row is not a wrong
        # prediction, it is an absent opinion, and averaging it in as a sixth
        # category makes every metric a measure of coverage instead of accuracy.
        keep = annotated_mask(df, gold_category_column).to_numpy()
        reference = df.loc[keep, gold_category_column].map(normalize_category).to_numpy()
        predicted = predicted[keep]
        if stored is not None:
            stored = stored[keep]
        weights = _gold_weights(df, keep)
    else:
        reference = stored
        weights = None

    if reference is None:
        reference = predicted.copy()

    metrics = _metrics_from_labels(reference, predicted, sample_weight=weights)

    if gold_category_column is not None:
        metrics["gold_column"] = gold_category_column
        if stored is not None:
            baseline = _metrics_from_labels(reference, stored, sample_weight=weights)
            metrics["baseline_vs_gold"] = baseline
            metrics["gold_delta_macro_f1"] = float(metrics["macro_f1"] - baseline["macro_f1"])

    return metrics


def evaluate_per_document(
    df: pd.DataFrame,
    constants: Mapping[str, Any] | None = None,
    *,
    original_category_column: str = "categ",
    gold_category_column: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-document metrics (importance can be collection-specific)."""
    kwargs = {
        "original_category_column": original_category_column,
        "gold_category_column": gold_category_column,
    }
    group_col = "file" if "file" in df.columns else ("_source_file" if "_source_file" in df.columns else None)
    if group_col is None:
        return {"<all>": evaluate_dataframe(df, constants, **kwargs)}
    out: dict[str, dict[str, Any]] = {}
    for name, doc in df.groupby(group_col, sort=True):
        out[str(name)] = evaluate_dataframe(doc, constants, **kwargs)
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


_RE_DOC_YEAR = re.compile(r"^[A-Za-z]{2,4}(\d{4})")


def document_decade(file_id: Any) -> str:
    """Decade of a document, parsed from its identifier, or ``"unknown"``.

    The archive names documents ``CTX<year><seq>`` / ``MTX<year><seq>``, so the
    decade is free. Validated over the delivered population: 36,137 of 36,268
    documents parse (99.6%) and every parsed year falls in 1920-2024, with no
    outliers. The 131 that do not are a lowercase variant and a ``P009_*`` series.

    It matters because the issue-#30 gold set is stratified by decade and its
    strata are INVERTED relative to the population it is used to tune: the 1920s
    are ~95x over-represented, the 2010s ~10x under-represented, and the 2010+
    documents the issue is actually about are 55% of the changed population but
    9% of the sample. Unweighted agreement over that sample is 62.8%; reweighted
    by decade it is 72.7%. A single headline number hides a 10-point choice.
    """
    m = _RE_DOC_YEAR.match(str(file_id))
    if not m:
        return "unknown"
    year = int(m.group(1))
    return f"{(year // 10) * 10}s" if 1900 <= year <= 2030 else "unknown"


def _stratum_breakdown(
    frame: pd.DataFrame,
    labels: pd.Index,
    gold: np.ndarray,
    rescored: np.ndarray,
    column: str,
) -> dict[str, dict[str, Any]]:
    """Agreement per stratum, for one grouping column already present on ``frame``."""
    if column not in frame.columns:
        return {}
    keys = frame.loc[labels, column].astype(str).to_numpy()
    out: dict[str, dict[str, Any]] = {}
    for key in sorted(set(keys)):
        sel = keys == key
        n = int(sel.sum())
        if not n:
            continue
        agree = int((gold[sel] == rescored[sel]).sum())
        out[key] = {"n": n, "agreement": float(agree / n), "share": float(n / len(keys))}
    return out


def _gold_report(old: pd.DataFrame, new: pd.DataFrame, gold_column: str) -> dict[str, Any] | None:
    """Score stored and re-scored categories against a human-gold column.

    Returns None when the frame has no gold column at all, so a mixed directory of
    annotated and un-annotated documents still reports on the annotated ones. A
    frame that HAS the column but leaves it blank contributes no rows, which is
    reported as such rather than scored as perfect agreement.
    """
    if gold_column not in old.columns:
        return None

    gold_raw = old[gold_column].fillna("").astype(str)
    mask = gold_raw.str.strip() != ""
    if not mask.any():
        return {"n": 0}

    # Index-aligned on purpose: `new` may be ordered differently from `old`, and
    # comparing two positionally-taken slices would score line N's prediction
    # against line M's gold.
    labels = old.index[mask]
    gold = gold_raw.loc[labels].map(normalize_category).to_numpy()
    stored = old.loc[labels, "categ"].map(normalize_category).to_numpy()
    rescored = new["categ"].reindex(labels).map(normalize_category).to_numpy()

    weights = _gold_weights(old, mask.to_numpy())
    stored_metrics = _metrics_from_labels(gold, stored, sample_weight=weights)
    rescored_metrics = _metrics_from_labels(gold, rescored, sample_weight=weights)

    # The composition, always. A headline agreement figure over a stratified
    # sample is a weighted average whose weights nobody chose; printing the
    # strata beside it is what stops the number being read as the population's.
    strata = {"gold_source": _stratum_breakdown(old, labels, gold, rescored, "gold_source")}
    if "file" in old.columns:
        with_decade = old.loc[labels, ["file"]].copy()
        with_decade["_decade"] = with_decade["file"].map(document_decade)
        strata["decade"] = _stratum_breakdown(with_decade, labels, gold, rescored, "_decade")

    return {
        "n": int(mask.sum()),
        "weighted": weights is not None,
        "stored_vs_gold": stored_metrics,
        "rescored_vs_gold": rescored_metrics,
        "delta_macro_f1": float(rescored_metrics["macro_f1"] - stored_metrics["macro_f1"]),
        "strata": {k: v for k, v in strata.items() if v},
    }


def _print_gold_report(report: dict[str, Any] | None, gold_column: str) -> None:
    if report is None:
        return
    if not report.get("n"):
        print(f"  gold column {gold_column!r} present but empty - no rows scored")
        return
    stored = report["stored_vs_gold"]
    rescored = report["rescored_vs_gold"]
    delta = report["delta_macro_f1"]
    how = "design-weighted" if report.get("weighted") else "UNWEIGHTED"
    print(f"  vs gold ({report['n']} annotated line(s), column {gold_column!r}, {how}):")
    print(f"    stored   macro_f1={stored['macro_f1']:.4f}  agreement={1.0 - stored['flip_rate']:.4f}")
    print(f"    rescored macro_f1={rescored['macro_f1']:.4f}  agreement={1.0 - rescored['flip_rate']:.4f}")
    print(f"    delta    macro_f1={delta:+.4f}")
    print("    note: 'weighted_f1' above is weighted by CLASS SUPPORT, not by sampling design.")

    for name, groups in (report.get("strata") or {}).items():
        if len(groups) < 2:
            continue
        print(f"    by {name}:")
        for key, g in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            print(f"      {key:<18} n={g['n']:>6}  {g['share']:>6.1%} of gold  agreement={g['agreement']:.4f}")
        spread = max(g["agreement"] for g in groups.values()) - min(g["agreement"] for g in groups.values())
        if spread >= 0.10 and not report.get("weighted"):
            print(
                f"      ^ agreement varies by {spread:.0%} across {name}. This sample is stratified and "
                f"its strata do not match the population's, so the headline above is not a population "
                f"estimate. Supply a {GOLD_WEIGHT_COLUMN!r} column to weight it."
            )


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
        # Align on the shared index rather than on position: a positional
        # comparison silently reports phantom flips when the two frames are
        # ordered differently (see the note in `rescore_csv`).
        old_cat = old["categ"]
        new_cat = new["categ"].reindex(old.index)
        old_cat = old_cat.reset_index(drop=True)
        new_cat = new_cat.reset_index(drop=True)
        diff_mask = old_cat != new_cat
        changed = int(diff_mask.sum())
        if changed:
            print(f"  --- {changed} line(s) changed category ---")
            txt = new["text"].reindex(old.index).reset_index(drop=True) if "text" in new.columns else None
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


GOLD_COLUMN_DEFAULT = "gold_categ"

GOLD_COLUMN_HELP = (
    "Name of a human-gold category column in the input CSVs (conventionally "
    f"{GOLD_COLUMN_DEFAULT!r}). Without it, every metric is computed against the "
    "pipeline's OWN stored categories, which makes the shipped config optimal by "
    "construction and scores a genuine improvement as damage. With it, metrics mean "
    "agreement with human labels. A named column that is absent is an error, not a "
    "silent fallback. See tools/GOLD.md."
)


GOLD_SIDECAR_HELP = (
    "Path to a key-indexed gold sidecar CSV (file,page_num,line_num,gold_categ"
    "[,gold_source]) to join onto the loaded rows before scoring. Use it when the "
    "annotation arrives separately from the DOC_LINE_CATEG batch it labels, as the "
    "issue-#30 sets did -- see tools/gold/sidecars/issue30_gold_2067.csv. Supplies the "
    "column; --gold-column still selects it. See tools/gold/GOLD.md."
)

GOLD_SIDECAR_KEYS = ("file", "page_num", "line_num")


def attach_gold_sidecar(
    df: pd.DataFrame,
    sidecar_path: Path | str,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Join a key-indexed gold sidecar onto ``df`` by (file, page_num, line_num).

    A sidecar carries labels and nothing else, so it can only be scored once it is
    attached to real rows. That is the whole operation: a left join that adds
    ``gold_categ`` (and ``gold_source`` when present) and touches nothing else.

    Both sides go through ``_coerce_locators`` first. The sidecar is read as text
    and the batch may carry ``page_num`` as either, so without the coercion the
    join silently matches zero rows -- which then scores as "no gold at all"
    rather than as a failure.

    Partial matches are normal and are reported rather than treated as an error:
    only 484 of the issue-#30 508 keys are still in the changed population. A join
    that matches *nothing* is a different matter and raises, because the quiet
    version of that mistake is a run that looks like it scored against humans and
    did not.
    """
    sidecar = pd.read_csv(Path(sidecar_path), dtype=str, keep_default_na=False)

    missing = [k for k in GOLD_SIDECAR_KEYS if k not in sidecar.columns]
    if missing:
        raise ValueError(f"gold sidecar {sidecar_path} is missing key column(s): {', '.join(missing)}")
    if GOLD_COLUMN_DEFAULT not in sidecar.columns:
        raise ValueError(f"gold sidecar {sidecar_path} has no {GOLD_COLUMN_DEFAULT!r} column")

    frame_missing = [k for k in GOLD_SIDECAR_KEYS if k not in df.columns]
    if frame_missing:
        raise ValueError(f"cannot join gold sidecar: the loaded rows lack {', '.join(frame_missing)}")

    carried = [GOLD_COLUMN_DEFAULT] + [c for c in ("gold_source", "gold_note") if c in sidecar.columns]
    sidecar = _coerce_locators(sidecar[list(GOLD_SIDECAR_KEYS) + carried].copy())
    sidecar = sidecar.drop_duplicates(subset=list(GOLD_SIDECAR_KEYS), keep="first")

    out = _coerce_locators(df.copy())
    # Index is preserved on purpose: `_gold_report` realigns `old` and `new` by
    # label, so a join that reset it would score line N against line M's gold.
    original_index = out.index
    out = out.merge(sidecar, on=list(GOLD_SIDECAR_KEYS), how="left", suffixes=("", "_sidecar"))
    out.index = original_index

    matched = int((out[GOLD_COLUMN_DEFAULT].fillna("").astype(str).str.strip() != "").sum())
    if verbose:
        print(
            f"  gold sidecar {Path(sidecar_path).name}: {matched} of {len(sidecar)} labels "
            f"matched onto {len(out)} rows ({len(sidecar) - matched} unmatched)"
        )
    if matched == 0:
        raise ValueError(
            f"gold sidecar {sidecar_path} matched 0 of {len(out)} rows on "
            f"{GOLD_SIDECAR_KEYS}. Wrong batch, or the keys do not correspond."
        )
    return out


def attach_gold_sidecar_from_args(df: pd.DataFrame, args) -> pd.DataFrame:
    """No-op unless ``--gold-sidecar`` was passed. Call right after ``load_csvs``.

    A sidecar without ``--gold-column`` is refused rather than ignored. Attaching
    the labels and then not selecting them leaves ``gold_category_column=None``,
    which scores the re-categorisation against the pipeline's OWN stored labels --
    the circular objective this whole path exists to escape. It printed a
    reassuring "N labels matched" line on the way past, so a multi-hour cluster
    run looked exactly like a successful gold run and was worth nothing.
    """
    path = getattr(args, "gold_sidecar", None)
    if not path:
        return df
    if not getattr(args, "gold_column", None):
        raise SystemExit(
            f"error: --gold-sidecar {path} was given without --gold-column.\n"
            f"       The sidecar supplies the labels; --gold-column selects them. Without it "
            f"every metric is scored against the pipeline's own stored categories, which makes "
            f"the shipped config optimal by construction.\n"
            f"       Add: --gold-column {GOLD_COLUMN_DEFAULT}"
        )
    return attach_gold_sidecar(df, path)


def add_gold_column_argument(ap: argparse.ArgumentParser) -> None:
    """Register ``--gold-column`` and ``--gold-sidecar`` on a driver.

    Single-sourced on purpose: five tools score through ``evaluate_dataframe`` and
    the flag has to mean exactly the same thing in each, the same way there is one
    scoring engine rather than five. ``--gold-sidecar`` is registered here for that
    same reason -- it is where the gold column comes from when the annotation ships
    separately from the rows.
    """
    ap.add_argument(
        "--gold-column",
        dest="gold_column",
        default=None,
        metavar="COLUMN",
        help=GOLD_COLUMN_HELP,
    )
    ap.add_argument(
        "--gold-sidecar",
        dest="gold_sidecar",
        default=None,
        metavar="PATH",
        help=GOLD_SIDECAR_HELP,
    )


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
        "--recursive",
        action="store_true",
        help=(
            "Recurse into sub-directories. `csv_paths()` has always supported this; the CLI did not, "
            "so a two-archive layout (ARUP/ and ARUB/ under one parent) silently found nothing and "
            "reported success. Output mirrors the input tree, so same-named documents in different "
            "archives cannot overwrite each other."
        ),
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help=(
            "Overwrite the input CSVs. Required to write without --out: the previous default was to "
            "overwrite in place, which destroys the baseline a re-categorisation is measured against "
            "and leaves a half-converted collection after a crash."
        ),
    )
    add_gold_column_argument(ap)
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
        csvs = csv_paths(in_path, recursive=args.recursive)
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

    if not args.report_only and not args.out and not args.in_place:
        print(
            "error: writing without --out would overwrite the input CSVs. Pass --out DIR to write "
            "elsewhere, --in-place to overwrite deliberately, or --report-only to write nothing.",
            file=sys.stderr,
        )
        return 2

    total_changed = 0
    grand_old: dict = {}
    grand_new: dict = {}
    failures: list[tuple[Path, str]] = []
    for position, csv_path in enumerate(csvs, start=1):
        # Progress on every file, not just at the end: a 113k-document run that
        # dies must say where. Cheap next to a re-score.
        if len(csvs) > 1:
            print(f"[{position}/{len(csvs)}] {csv_path}", flush=True)

        try:
            old, new = rescore_csv(csv_path, constants)
        except Exception as exc:  # noqa: BLE001 - one bad CSV must not end the run
            # Previously a single malformed CSV aborted the whole loop, and
            # because writes were in place it left a half-converted collection
            # with nothing on disk saying where it stopped.
            print(f"  ! FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            failures.append((csv_path, f"{type(exc).__name__}: {exc}"))
            continue

        total_changed += _report(csv_path, old, new)

        if args.gold_column:
            # Attached to `old` only: `new` is the re-scored frame and gold is a
            # property of the line, not of the prediction. `_gold_report` reads
            # the column off `old` and realigns `new` by index.
            scored = attach_gold_sidecar_from_args(old, args)
            _print_gold_report(_gold_report(scored, new, args.gold_column), args.gold_column)

        if args.probe_metre_candidate:
            candidate_hits = _count_spaced_decimal_metre_candidates(old)
            print(f"  dormant metre-spacing probe hits: {candidate_hits}")

        for k, v in _category_counts(old).items():
            grand_old[k] = grand_old.get(k, 0) + v
        for k, v in _category_counts(new).items():
            grand_new[k] = grand_new.get(k, 0) + v

        if not args.report_only:
            if args.out:
                out_root = Path(args.out)
                if in_path.is_dir():
                    # Mirror the input tree. Flattening to `out / name` would let
                    # same-named documents in different archives (ARUP/CTX…csv and
                    # ARUB/CTX…csv) overwrite each other silently under --recursive.
                    out_path = out_root / csv_path.relative_to(in_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                else:
                    out_path = out_root
                    out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = csv_path
            # Write to a temp file and rename, so a kill mid-write cannot leave a
            # truncated CSV that a later resume would treat as complete.
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")
            new.to_csv(tmp_path, index=False, encoding="utf-8")
            tmp_path.replace(out_path)

    if len(csvs) > 1:
        print("\n=== GRAND TOTAL ===")
        for c in sorted(set(grand_old) | set(grand_new)):
            b, a = grand_old.get(c, 0), grand_new.get(c, 0)
            print(f"  {c:<10} {b:>7} {a:>7} {a - b:>+7}")
        print(f"  total lines changed category: {total_changed}")
        print(f"  files processed: {len(csvs) - len(failures)}/{len(csvs)}")

    if failures:
        print(f"\n=== {len(failures)} FILE(S) FAILED ===", file=sys.stderr)
        for path, reason in failures:
            print(f"  {path}: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
