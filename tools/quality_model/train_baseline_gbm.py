"""Gradient-boosting baseline for the quality score (issue #23, Phase 3, D7).

A ``HistGradientBoostingRegressor`` over the pre-computed CSV features is the sanity
floor the neural encoder must beat. It is trained in two variants:

* **with** the perplexity feature — the strongest possible feature baseline, but it
  cannot replace Qwen (perplexity *is* a Qwen output), and
* **without** perplexity — the honest "can we drop Qwen without a neural net?"
  comparison; this is the number ``train.py`` must beat to justify the encoder.

Pure-Python data prep / metrics live in ``common.py`` and are unit-tested without
sklearn; sklearn is imported lazily so this module imports on a machine without it.

Run::

    python tools/quality_model/train_baseline_gbm.py --dataset dataset.csv --out runs/gbm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common as CM  # noqa: E402


def train_one(
    train_rows: list[dict],
    eval_rows: list[dict],
    include_perplexity: bool,
    params: dict,
) -> dict:
    """Fit one GBM variant and return its metrics dict. sklearn imported lazily."""
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: PLC0415

    feats = CM.feature_columns(include_perplexity)
    x_train, y_train = CM.rows_to_xy(train_rows, feats, params["target_col"])
    x_eval, y_eval = CM.rows_to_xy(eval_rows, feats, params["target_col"])

    model = HistGradientBoostingRegressor(
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        l2_regularization=params["l2"],
        early_stopping=params["early_stopping"],
        random_state=params["seed"],
    )
    model.fit(x_train, y_train)
    y_pred = [max(0.0, min(1.0, float(p))) for p in model.predict(x_eval)]

    reg = CM.regression_metrics(y_eval, y_pred)
    cat = CM.banded_category_metrics(y_eval, y_pred)
    return {
        "variant": "with_perplexity" if include_perplexity else "no_perplexity",
        "features": feats,
        "n_train": len(train_rows),
        "n_eval": len(eval_rows),
        "regression": reg,
        "banded_category": cat,
    }


def run(dataset_path: Path, cfg, out_dir: Path, eval_split: str, seed: int) -> dict:
    rows = CM.read_dataset(dataset_path)
    splits = CM.split_rows(rows, CM.cfg_get(cfg, "DATA", "SPLIT_COL", "split"))
    train_rows = splits.get("train", [])
    eval_rows = splits.get(eval_split, []) or splits.get("val", []) or splits.get("test", [])

    params = {
        "target_col": CM.cfg_get(cfg, "DATA", "TARGET_COL", CM.DEFAULT_TARGET),
        "max_iter": CM.cfg_get(cfg, "BASELINE", "MAX_ITER", 400, int),
        "learning_rate": CM.cfg_get(cfg, "BASELINE", "LEARNING_RATE", 0.05, float),
        "max_depth": CM.cfg_get(cfg, "BASELINE", "MAX_DEPTH", 6, int),
        "l2": CM.cfg_get(cfg, "BASELINE", "L2_REGULARIZATION", 0.0, float),
        "early_stopping": CM.cfg_get(cfg, "BASELINE", "EARLY_STOPPING", True, bool),
        "seed": seed,
    }

    results = [
        train_one(train_rows, eval_rows, include_perplexity=True, params=params),
        train_one(train_rows, eval_rows, include_perplexity=False, params=params),
    ]
    report = {"dataset": str(dataset_path), "eval_split": eval_split, "params": params, "variants": results}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gbm_metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def format_report(report: dict) -> str:
    lines = ["GBM baseline", "=" * 24, f"dataset={report['dataset']}  eval={report['eval_split']}", ""]
    for v in report["variants"]:
        reg, cat = v["regression"], v["banded_category"]
        lines.append(
            f"{v['variant']:<16} MAE={reg['mae']:.4f} RMSE={reg['rmse']:.4f} "
            f"Spearman={reg['spearman']:.4f} cat_macroF1={cat['macro_f1']:.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the GBM quality-score baseline (± perplexity).")
    p.add_argument("--dataset", type=Path, default=None, help="Dataset CSV (overrides config DATA.DATASET).")
    p.add_argument("--config", type=Path, default=_HERE.parents[1] / "setup" / "config_quality_model.txt")
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("runs/gbm"))
    args = p.parse_args(argv)

    cfg = CM.load_config(args.config) if Path(args.config).exists() else None
    dataset = args.dataset or Path(CM.cfg_get(cfg, "DATA", "DATASET", "dataset.csv"))
    report = run(dataset, cfg, args.out, args.eval_split, args.seed)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
