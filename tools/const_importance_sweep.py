#!/usr/bin/env python3
"""
tools/const_importance_sweep.py
===============================
Surrogate-based config-constant importance sweep for ATRIUM ALTO post-processing
(issue #5).

It samples the tunable ``[TEXT_UTILS]`` constants, re-categorises the cached
per-line CSV features for each sample, and fits a lightweight surrogate to learn
which constants most control the chosen objective.

Faithful evaluator
------------------
Every trial is scored by ``recategorize_from_csv.evaluate_dataframe``, which runs
the REAL production engine (``compute_quality_score`` / ``categorize_line`` /
``apply_document_postprocessing``) under ``override_constants`` — there is no
parallel re-implementation. Consequently the baseline (current config) sits at
flip_rate == 0 by construction, and importances describe the production pipeline,
not a surrogate of a surrogate.

Methodology notes
-----------------
* Importance (fANOVA / MDI / permutation) is only meaningful over a (quasi-)
  uniform sample, so the default Optuna sampler here is **random**. TPE is for
  *optimisation* (finding a good config), not importance: fANOVA on a
  TPE-concentrated study is biased, so this tool warns when you combine them.
* The sklearn surrogate reports **out-of-bag R²** (held-out), not just train R².
* fANOVA / MDI are skipped for a single-parameter study (importance is trivially
  100% and uninformative).
* ``QS_WEIGHT_*`` are frozen by default: prior sweeps showed the linear weight
  composition has low practical influence vs. the category thresholds and the
  garbage/inversion/hard-sweep gates. Enable with ``--include-qs-weights``.
* Sub-sampling is by **document** (``--sample-docs``), never by line: the page
  post-processing needs whole pages, so splitting a document would corrupt it.

Examples
--------
    # sklearn surrogate, no extra deps, importance over the threshold/edge set
    python tools/const_importance_sweep.py \
        --input-dir data_samples/DOC_LINE_CATEG \
        --config config_langID.txt --output-dir sweep_out \
        --backend sklearn --metric macro_f1 --n-trials 400

    # Optuna + fANOVA (random sampling for unbiased importance)
    python tools/const_importance_sweep.py \
        --input-dir data_samples/DOC_LINE_CATEG \
        --config config_langID.txt --output-dir sweep_out \
        --backend optuna --sampler random --n-trials 400 --storage sqlite:///sweep.db
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Run as `python tools/const_importance_sweep.py` from the repo root.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from recategorize_from_csv import (  # noqa: E402
    QS_WEIGHT_NAMES,
    TUNABLE_CONSTANTS,
    _load_lang_config,
    coerce_constants,
    evaluate_dataframe,
    evaluate_per_document,
    load_csvs,
    read_config_constants,
    validate_constants,
)

# recategorize_from_csv pulls in pandas/numpy + the (import-light) production
# modules, but NO Torch/Transformers/FastText at module load, so this top-level
# import stays cheap.

OBJECTIVE_METRICS = ("flip_rate", "macro_f1", "weighted_f1", "trash_rate", "clear_rate", "costed_score")
QS_WEIGHT_PARAMS = set(QS_WEIGHT_NAMES)

# ---------------------------------------------------------------------------
# Search space (covers every tunable in recategorize_from_csv.TUNABLE_CONSTANTS,
# including the #3 hard-sweep / extreme-ppl / inversion routes the old sweep
# never varied). Ranges bracket the current config defaults.
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict[str, dict[str, Any]] = {
    # QS weights (frozen unless --include-qs-weights)
    **{name: {"type": "float", "low": 0.01, "high": 0.40} for name in QS_WEIGHT_NAMES},
    # Category boundaries
    "CATEG_TRASH_SCORE_MAX": {"type": "float", "low": 0.30, "high": 0.65},
    "CATEG_NOISY_SCORE_MAX": {"type": "float", "low": 0.70, "high": 0.97},
    "CATEG_GARBAGE_DENSITY_HIGH": {"type": "float", "low": 0.15, "high": 0.60},
    # Rotation / inversion gates
    "ROT_RATIO_INVERTED_MIN": {"type": "float", "low": 0.35, "high": 0.80},
    "WEIRD_RATIO_INVERTED_MIN": {"type": "float", "low": 0.15, "high": 0.60},
    "PPL_INVERTED_MIN": {"type": "float", "low": 50.0, "high": 500.0},
    # Perplexity normalisation
    "PERPLEXITY_THRESHOLD_MAX": {"type": "float", "low": 500.0, "high": 2000.0},
    "SHORT_PPL_CAP": {"type": "float", "low": 300.0, "high": 950.0},
    # Clean-prose gates (legacy; retained for continuity)
    "CLEAN_PROSE_MIN_SCORE": {"type": "float", "low": 0.50, "high": 0.84},
    "CLEAN_PROSE_WEIRD_MAX": {"type": "float", "low": 0.03, "high": 0.20},
    "CLEAN_PROSE_PPL_MAX": {"type": "float", "low": 150.0, "high": 700.0},
    "CLEAN_PROSE_WC_MIN": {"type": "int", "low": 2, "high": 6},
    # (#3) hard-sweep / extreme- and absolute-perplexity trash routes
    "HARD_SWEEP_LANG_MAX": {"type": "float", "low": 0.20, "high": 0.70},
    "HARD_SWEEP_PPL_MIN": {"type": "float", "low": 500.0, "high": 3000.0},
    "PPL_EXTREME_MIN": {"type": "float", "low": 1500.0, "high": 6000.0},
    "EXTREME_LANG_CONF": {"type": "float", "low": 0.60, "high": 0.95},
    "PPL_GARBAGE_ABSOLUTE": {"type": "float", "low": 10000.0, "high": 60000.0},
    # (#3) low-ppl Clear + LM-confident-Czech recovery + mostly-readable cap
    "LOWPPL_CLEAR_MAX": {"type": "float", "low": 20.0, "high": 120.0},
    "LOWPPL_CZECH_CLEAR_MAX": {"type": "float", "low": 80.0, "high": 300.0},
    "CZECH_CLEAR_GARBAGE_MAX": {"type": "float", "low": 0.05, "high": 0.30},
    "MOSTLY_READABLE_VALID_MIN": {"type": "float", "low": 0.70, "high": 0.95},
    "SHORT_NOISY_QS_PENALTY": {"type": "float", "low": 0.05, "high": 0.40},
    "WORD_W_PENALTY": {"type": "float", "low": 0.05, "high": 0.40},
    # (#3) rotation / inversion organic penalties + per-line route
    "GHOST_DOMINATED_MIN_RATIO": {"type": "float", "low": 0.30, "high": 0.80},
    "SUSPICIOUS_ROT_RATIO": {"type": "float", "low": 0.45, "high": 0.85},
    "SUSPICIOUS_WQX_RATIO": {"type": "float", "low": 0.05, "high": 0.40},
    "INVERTED_WEIRD_PENALTY": {"type": "float", "low": 0.20, "high": 0.70},
    "GHOST_HITS_INVERTED_MIN": {"type": "int", "low": 1, "high": 3},
    "ROT_HIGH_LANG_CONF": {"type": "float", "low": 0.80, "high": 0.98},
    "LANG_SCORE_ROUGH": {"type": "float", "low": 0.30, "high": 0.60},
    # (#3 A3) page-level smoothing
    "INVERTED_RUN_MIN": {"type": "int", "low": 2, "high": 8},
    "INVERTED_PAGE_MAJORITY": {"type": "float", "low": 0.40, "high": 0.80},
    "CLEAR_BAND_WC_MIN": {"type": "int", "low": 0, "high": 5},
}

_missing = [name for name in TUNABLE_CONSTANTS if name not in SEARCH_SPACE]
if _missing:  # pragma: no cover - guards future drift between the two files
    raise RuntimeError(f"SEARCH_SPACE is missing ranges for tunables: {_missing}")

EDGE_PARAMS = [
    "CATEG_GARBAGE_DENSITY_HIGH",
    "ROT_RATIO_INVERTED_MIN",
    "WEIRD_RATIO_INVERTED_MIN",
    "PPL_INVERTED_MIN",
    "HARD_SWEEP_LANG_MAX",
    "HARD_SWEEP_PPL_MIN",
    "PPL_EXTREME_MIN",
    "EXTREME_LANG_CONF",
    "PPL_GARBAGE_ABSOLUTE",
    "SUSPICIOUS_ROT_RATIO",
    "SUSPICIOUS_WQX_RATIO",
    "GHOST_DOMINATED_MIN_RATIO",
    "GHOST_HITS_INVERTED_MIN",
    "INVERTED_WEIRD_PENALTY",
    "INVERTED_RUN_MIN",
    "INVERTED_PAGE_MAJORITY",
    "ROT_HIGH_LANG_CONF",
    "LANG_SCORE_ROUGH",
]
THRESHOLD_PARAMS = [
    "CATEG_TRASH_SCORE_MAX",
    "CATEG_NOISY_SCORE_MAX",
    "CATEG_GARBAGE_DENSITY_HIGH",
    "CLEAN_PROSE_MIN_SCORE",
    "CLEAN_PROSE_WEIRD_MAX",
    "CLEAN_PROSE_PPL_MAX",
    "CLEAN_PROSE_WC_MIN",
    "LOWPPL_CLEAR_MAX",
    "LOWPPL_CZECH_CLEAR_MAX",
    "CZECH_CLEAR_GARBAGE_MAX",
    "MOSTLY_READABLE_VALID_MIN",
    "PERPLEXITY_THRESHOLD_MAX",
    "SHORT_PPL_CAP",
]
DEFAULT_SWEEP_PARAMS = [n for n in TUNABLE_CONSTANTS if n not in QS_WEIGHT_PARAMS]


def active_params(top_params, *, profile: str, include_qs_weights: bool) -> list[str]:
    """Return the parameter set to sample. QS_WEIGHT_* frozen by default (#5)."""
    if top_params:
        unknown = sorted(set(top_params) - set(TUNABLE_CONSTANTS))
        if unknown:
            raise ValueError(f"Unknown --top-param values: {unknown}")
        return list(top_params)
    if profile == "full":
        params = list(TUNABLE_CONSTANTS)
    elif profile == "default":
        params = list(DEFAULT_SWEEP_PARAMS)
    elif profile == "edge":
        params = list(EDGE_PARAMS)
    elif profile == "thresholds":
        params = list(THRESHOLD_PARAMS)
    else:
        raise ValueError(f"Unknown sweep profile: {profile}")
    if include_qs_weights:
        for name in QS_WEIGHT_NAMES:
            if name not in params:
                params.append(name)
    else:
        params = [name for name in params if name not in QS_WEIGHT_PARAMS]
    return params


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def is_valid_constants(constants: dict[str, Any]) -> bool:
    try:
        validate_constants(coerce_constants(constants))
    except ValueError:
        return False
    return True


def sample_random_constants(rng, base_constants, params) -> dict[str, Any]:
    constants = dict(base_constants)
    for name in params:
        spec = SEARCH_SPACE[name]
        if spec["type"] == "int":
            constants[name] = int(rng.integers(spec["low"], spec["high"] + 1))
        else:
            constants[name] = float(rng.uniform(spec["low"], spec["high"]))
    return coerce_constants(constants)


def sample_optuna_constants(trial, base_constants, params) -> dict[str, Any]:
    constants = dict(base_constants)
    for name in params:
        spec = SEARCH_SPACE[name]
        if spec["type"] == "int":
            constants[name] = trial.suggest_int(name, spec["low"], spec["high"])
        else:
            constants[name] = trial.suggest_float(name, spec["low"], spec["high"])
    return coerce_constants(constants)


def objective_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name not in OBJECTIVE_METRICS:
        raise ValueError(f"Unknown metric {metric_name!r}")
    value = metrics[metric_name]
    return float(int(value)) if isinstance(value, bool) else float(value)


def normalize_importances(importances: dict[str, float]) -> dict[str, float]:
    cleaned = {k: max(0.0, float(v)) for k, v in importances.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {k: 0.0 for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


def maybe_sample_documents(df: pd.DataFrame, *, sample_docs: int | None, seed: int) -> pd.DataFrame:
    """Sub-sample whole documents (never individual lines — that would corrupt the
    page-level post-processing)."""
    if not sample_docs or sample_docs <= 0 or "file" not in df.columns:
        return df
    files = df["file"].drop_duplicates().tolist()
    if len(files) <= sample_docs:
        return df
    rng = np.random.default_rng(seed)
    keep = set(rng.choice(files, size=sample_docs, replace=False).tolist())
    return df[df["file"].isin(keep)].reset_index(drop=True)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def write_trials_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False) if rows else path.write_text("", encoding="utf-8")


def save_importance_plot(output_dir: Path, importances: dict[str, float], title: str) -> None:
    """Best-effort bar chart; silently skips if matplotlib is unavailable."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return
    names = [k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(10, max(5.0, 0.32 * len(names))))
    ax.barh(names[::-1], values[::-1])
    ax.set_xlabel("Normalized importance")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_dir / "param_importance.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# sklearn backend
# ---------------------------------------------------------------------------


def run_sklearn_backend(
    *, data, base_constants, params, output_dir, n_trials, seed, metric, direction, eval_kwargs
) -> dict[str, Any]:
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
    except ImportError as exc:
        raise RuntimeError("The sklearn backend requires scikit-learn.") from exc

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    x_values: list[list[float]] = []
    y_values: list[float] = []

    attempts = 0
    max_attempts = max(n_trials * 20, n_trials + 100)
    while len(rows) < n_trials and attempts < max_attempts:
        attempts += 1
        constants = sample_random_constants(rng, base_constants, params)
        if not is_valid_constants(constants):
            continue
        metrics = evaluate_dataframe(data, constants, **eval_kwargs)
        y = objective_value(metrics, metric)
        rows.append(
            {
                "trial": len(rows),
                "objective": y,
                "metric": metric,
                **{name: constants[name] for name in params},
                "flip_rate": metrics["flip_rate"],
                "trash_rate": metrics["trash_rate"],
                "clear_rate": metrics["clear_rate"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "kl_divergence": metrics["kl_divergence"],
                "costed_score": metrics["costed_score"],
            }
        )
        x_values.append([float(constants[name]) for name in params])
        y_values.append(y)
        if len(rows) % 25 == 0:
            print(f"[sklearn] completed {len(rows)}/{n_trials} valid trials")

    if len(rows) < max(10, len(params)):
        raise RuntimeError(
            f"Only produced {len(rows)} valid trials after {attempts} attempts; check search-space constraints."
        )

    x = np.asarray(x_values, dtype="float64")
    y = np.asarray(y_values, dtype="float64")

    # bootstrap + oob_score so we can report a HELD-OUT R², not just train fit.
    rf = RandomForestRegressor(
        n_estimators=256,
        min_samples_leaf=2,
        random_state=seed,
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,
    )
    rf.fit(x, y)
    oob_r2 = float(getattr(rf, "oob_score_", float("nan")))

    result: dict[str, Any] = {
        "backend": "sklearn",
        "metric": metric,
        "direction": direction,
        "n_trials": len(rows),
        "attempts": attempts,
        "r2_train": float(rf.score(x, y)),
        "oob_r2": oob_r2,
        "n_params": len(params),
    }

    if len(params) < 2:
        print("[sklearn] WARNING: <2 swept params — importance is trivial; skipping MDI/permutation.")
        result["importance_skipped"] = "single-parameter study"
    else:
        mdi = normalize_importances({name: float(v) for name, v in zip(params, rf.feature_importances_, strict=True)})
        perm_raw = permutation_importance(rf, x, y, n_repeats=10, random_state=seed, n_jobs=-1)
        permutation = normalize_importances(
            {name: float(v) for name, v in zip(params, perm_raw.importances_mean, strict=True)}
        )
        save_json(output_dir / "param_importance.json", mdi)
        save_json(output_dir / "param_importance_permutation.json", permutation)
        save_importance_plot(output_dir, mdi, f"MDI importance ({metric})")
        result["mdi_importance"] = mdi
        result["permutation_importance"] = permutation

    best_idx = int(np.nanargmax(y) if direction == "maximize" else np.nanargmin(y))
    best_constants = dict(base_constants)
    best_constants.update({name: rows[best_idx][name] for name in params})
    write_trials_csv(output_dir / "trials.csv", rows)
    save_json(output_dir / "best_config.json", best_constants)
    result["best_trial"] = rows[best_idx]
    result["best_config"] = best_constants
    return result


# ---------------------------------------------------------------------------
# Optuna backend
# ---------------------------------------------------------------------------


def run_optuna_backend(
    *,
    data,
    base_constants,
    params,
    output_dir,
    n_trials,
    seed,
    metric,
    direction,
    sampler_name,
    storage,
    study_name,
    eval_kwargs,
) -> dict[str, Any]:
    try:
        import optuna
        from optuna.importance import FanovaImportanceEvaluator, get_param_importances
    except ImportError as exc:
        raise RuntimeError("The optuna backend requires Optuna.") from exc

    if sampler_name == "random":
        sampler = optuna.samplers.RandomSampler(seed=seed)
    elif sampler_name == "tpe":
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        print(
            "[optuna] WARNING: TPE concentrates trials near the optimum, which BIASES "
            "fANOVA importance. Use --sampler random for importance; reserve TPE for "
            "optimisation (finding a good config)."
        )
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )

    def objective(trial):
        constants = sample_optuna_constants(trial, base_constants, params)
        if not is_valid_constants(constants):
            raise optuna.TrialPruned("invalid constrained constants")
        metrics = evaluate_dataframe(data, constants, **eval_kwargs)
        value = objective_value(metrics, metric)
        if math.isnan(value) or math.isinf(value):
            raise optuna.TrialPruned("invalid objective value")
        for attr in (
            "flip_rate",
            "trash_rate",
            "clear_rate",
            "macro_f1",
            "weighted_f1",
            "kl_divergence",
            "costed_score",
        ):
            trial.set_user_attr(attr, metrics[attr])
        return value

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(complete) < max(10, len(params)):
        raise RuntimeError(f"Only {len(complete)} completed Optuna trials. Increase --n-trials.")

    result: dict[str, Any] = {
        "backend": "optuna",
        "metric": metric,
        "direction": direction,
        "sampler": sampler_name,
        "n_complete_trials": len(complete),
        "n_total_trials": len(study.trials),
        "n_params": len(params),
    }

    if len(params) < 2:
        print("[optuna] WARNING: <2 swept params — fANOVA is trivially 100%; skipping importance.")
        result["importance_skipped"] = "single-parameter study"
    else:
        fanova = normalize_importances(
            get_param_importances(study, evaluator=FanovaImportanceEvaluator(n_trees=64, seed=seed))
        )
        save_json(output_dir / "param_importance.json", fanova)
        save_importance_plot(output_dir, fanova, f"fANOVA importance ({metric}, {sampler_name})")
        result["fanova_importance"] = fanova
        if sampler_name == "tpe":
            result["importance_warning"] = "fANOVA computed on a TPE-biased study; treat as unreliable."

    rows = []
    for t in complete:
        row = {"trial": t.number, "objective": float(t.value), "metric": metric}
        row.update({name: t.params.get(name) for name in params})
        row.update(
            {
                a: t.user_attrs.get(a)
                for a in (
                    "flip_rate",
                    "trash_rate",
                    "clear_rate",
                    "macro_f1",
                    "weighted_f1",
                    "kl_divergence",
                    "costed_score",
                )
            }
        )
        rows.append(row)
    write_trials_csv(output_dir / "trials.csv", rows)

    best_constants = dict(base_constants)
    best_constants.update(study.best_trial.params)
    save_json(output_dir / "best_config.json", best_constants)
    result["best_trial_number"] = study.best_trial.number
    result["best_value"] = float(study.best_value)
    result["best_config"] = best_constants
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Config-constant importance sweep (faithful production engine).")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--config", type=str, default=None, help="config_langID.txt to source the base config from.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--backend", choices=("sklearn", "optuna"), default="sklearn")
    p.add_argument("--n-trials", type=int, default=400)
    p.add_argument("--top-param", type=str, nargs="*", help="Sweep only these constants.")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage URL (e.g. sqlite:///sweep.db).")
    p.add_argument("--profile", choices=("default", "full", "edge", "thresholds"), default="default")
    p.add_argument(
        "--include-qs-weights",
        action="store_true",
        help="Also sweep QS_WEIGHT_* (frozen by default; low practical importance).",
    )
    p.add_argument("--metric", choices=OBJECTIVE_METRICS, default="macro_f1")
    p.add_argument("--direction", choices=("maximize", "minimize"), default=None)
    p.add_argument("--recursive", action="store_true")
    p.add_argument(
        "--sample-docs", type=int, default=None, help="Sub-sample to ~this many whole documents (never splits a page)."
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sampler",
        choices=("random", "tpe"),
        default="random",
        help="Optuna sampler. 'random' for unbiased importance (default); 'tpe' for optimisation only.",
    )
    p.add_argument("--study-name", type=str, default="const_importance")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    params = active_params(args.top_param, profile=args.profile, include_qs_weights=args.include_qs_weights)
    direction = args.direction or (
        "maximize" if args.metric in {"macro_f1", "weighted_f1", "clear_rate"} else "minimize"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_constants = coerce_constants(read_config_constants(args.config))
    validate_constants(base_constants)

    print(f"Running sweep | profile={args.profile} | backend={args.backend} | metric={args.metric} ({direction})")
    print(f"Sweeping {len(params)} parameter(s): {params}")
    print(f"Loading CSVs from {args.input_dir} ...")
    data = load_csvs(args.input_dir, recursive=args.recursive)
    data = maybe_sample_documents(data, sample_docs=args.sample_docs, seed=args.seed)
    n_docs = data["file"].nunique() if "file" in data.columns else 1
    print(f"Loaded {len(data):,} lines across {n_docs} document(s)")

    # Resolve language config once; thread it through every trial (avoids a
    # configparser read per evaluation).
    expected_langs, known_bases = _load_lang_config(args.config or str(Path("config_langID.txt")))
    eval_kwargs = {"expected_langs": expected_langs, "known_bases": known_bases}

    baseline_metrics = evaluate_dataframe(data, base_constants, **eval_kwargs)
    save_json(args.output_dir / "baseline_metrics.json", baseline_metrics)
    save_json(args.output_dir / "baseline_per_document.json", evaluate_per_document(data, base_constants))
    save_json(args.output_dir / "base_config.json", base_constants)
    save_json(args.output_dir / "search_space.json", {name: SEARCH_SPACE[name] for name in params})
    print(
        f"Baseline (current config): flip_rate={baseline_metrics['flip_rate']:.4f} "
        f"macro_f1={baseline_metrics['macro_f1']:.4f}  (expect flip_rate≈0 — faithful engine)"
    )
    if baseline_metrics["flip_rate"] > 1e-9:
        print(
            "[WARNING] baseline flip_rate is not ~0 — the re-score is drifting from the stored "
            "labels; importances may be confounded. Investigate before trusting results."
        )

    common = dict(
        data=data,
        base_constants=base_constants,
        params=params,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        seed=args.seed,
        metric=args.metric,
        direction=direction,
        eval_kwargs=eval_kwargs,
    )
    if args.backend == "sklearn":
        result = run_sklearn_backend(**common)
    else:
        result = run_optuna_backend(
            **common, sampler_name=args.sampler, storage=args.storage, study_name=args.study_name
        )

    save_json(args.output_dir / "sweep_summary.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
