"""Evaluate a trained quality-score model (issue #23, Phase 4).

Three things, per strategy §3:

1. **Vs the algorithm (held-out docs)** — MAE / RMSE / Spearman on ``score_raw``,
   plus category agreement after banding the predicted score at 0.55 / 0.80 (and,
   if present, from the model's category head), a calibration table, and the
   corruption-severity monotonicity check on the *predicted* score.
2. **Vs expert gold (the only self-reference-free gate)** — compares
   **model-vs-gold** category agreement against **algorithm-vs-gold** on the same
   lines; the model passes if its macro-F1 is within a margin of (or beats) the
   algorithm's. This is the objective go/no-go.
3. **Stratified** — the held-out metrics can be broken down by any column
   (language, provenance) to catch minority-class regressions.

The metric math is pure Python (``common.py``) so this module and its fast tests
run without the ML stack. ``predict_dataset`` (torch/transformers, lazy) attaches
model predictions to a dataset; everything downstream works on a predictions CSV,
so evaluation is fully testable with hand-written rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import common as CM  # noqa: E402

_SEVERITY = {"none": 0, "light": 1, "medium": 2, "heavy": 3}


# ---------------------------------------------------------------------------
# Evaluation vs the algorithm (from a predictions CSV — model-free)
# ---------------------------------------------------------------------------


def calibration(true_scores: list[float], pred_scores: list[float], n_bins: int = 10) -> list[dict]:
    """Per-bin mean predicted vs mean actual score (reliability table)."""
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for t, p in zip(true_scores, pred_scores, strict=True):
        b = min(n_bins - 1, max(0, int(p * n_bins)))
        buckets[b].append((t, p))
    out = []
    for b in range(n_bins):
        pairs = buckets.get(b, [])
        if not pairs:
            out.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "count": 0})
            continue
        mean_true = sum(t for t, _ in pairs) / len(pairs)
        mean_pred = sum(p for _, p in pairs) / len(pairs)
        out.append(
            {
                "bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}",
                "count": len(pairs),
                "mean_pred": round(mean_pred, 4),
                "mean_true": round(mean_true, 4),
                "gap": round(mean_pred - mean_true, 4),
            }
        )
    return out


def monotonicity(rows: list[dict], pred_col: str) -> dict:
    """Does the model preserve the corruption ordering? Pairwise concordance:
    within a source line, a heavier variant should score <= a lighter one."""
    groups: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
    for r in rows:
        sev = _SEVERITY.get(r.get("band", "none"), 0)
        groups[(r.get("source_doc"), r.get("source_line"))].append((sev, CM._to_float(r.get(pred_col))))
    total = concordant = 0
    for members in groups.values():
        for sev_a, score_a in members:
            for sev_b, score_b in members:
                if sev_a < sev_b:
                    total += 1
                    if score_a >= score_b - 1e-9:
                        concordant += 1
    return {"pairs": total, "concordant": concordant, "accuracy": round(concordant / total, 4) if total else 1.0}


def evaluate_predictions(
    rows: list[dict],
    *,
    true_score_col: str = "score_raw",
    pred_score_col: str = "pred_score",
    true_categ_col: str = "categ",
    pred_categ_col: str | None = None,
    stratify_by: str | None = None,
) -> dict:
    """Full held-out evaluation of model predictions against the algorithm labels."""
    true_scores = [CM._to_float(r.get(true_score_col)) for r in rows]
    pred_scores = [CM._to_float(r.get(pred_score_col)) for r in rows]
    true_cats = [r.get(true_categ_col, "") for r in rows]
    pred_band_cats = [CM.band_category(s) for s in pred_scores]

    report: dict = {
        "n": len(rows),
        "regression": CM.regression_metrics(true_scores, pred_scores),
        "banded_vs_algo": CM.category_metrics(true_cats, pred_band_cats),
        "calibration": calibration(true_scores, pred_scores),
        "monotonicity": monotonicity(rows, pred_score_col),
    }
    if pred_categ_col:
        pred_cats = [r.get(pred_categ_col, "") for r in rows]
        report["cathead_vs_algo"] = CM.category_metrics(true_cats, pred_cats)

    if stratify_by:
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            groups[r.get(stratify_by, "")].append(r)
        report[f"by_{stratify_by}"] = {
            g: {
                "n": len(grp),
                "regression": CM.regression_metrics(
                    [CM._to_float(r.get(true_score_col)) for r in grp],
                    [CM._to_float(r.get(pred_score_col)) for r in grp],
                ),
                "banded_vs_algo": CM.category_metrics(
                    [r.get(true_categ_col, "") for r in grp],
                    [CM.band_category(CM._to_float(r.get(pred_score_col))) for r in grp],
                ),
            }
            for g, grp in sorted(groups.items())
        }
    return report


# ---------------------------------------------------------------------------
# The gold-set gate (the only self-reference-free check)
# ---------------------------------------------------------------------------


def gold_gate(
    gold_rows: list[dict],
    *,
    gold_categ_col: str = "gold_categ",
    algo_categ_col: str = "algo_categ",
    pred_score_col: str = "pred_score",
    pred_categ_col: str | None = None,
    margin: float = 0.01,
) -> dict:
    """Compare model-vs-gold against algorithm-vs-gold. Model passes if its macro-F1
    is >= the algorithm's minus ``margin`` (the parity floor from the plan)."""
    gold = [r.get(gold_categ_col, "") for r in gold_rows]
    algo = [r.get(algo_categ_col, "") for r in gold_rows]
    if pred_categ_col:
        model = [r.get(pred_categ_col, "") for r in gold_rows]
    else:
        model = [CM.band_category(CM._to_float(r.get(pred_score_col))) for r in gold_rows]

    model_m = CM.category_metrics(gold, model)
    algo_m = CM.category_metrics(gold, algo)
    passed = model_m["macro_f1"] >= algo_m["macro_f1"] - margin
    return {
        "n": len(gold_rows),
        "model_macro_f1": model_m["macro_f1"],
        "algo_macro_f1": algo_m["macro_f1"],
        "margin": margin,
        "passed": passed,
        "model_metrics": model_m,
        "algo_metrics": algo_m,
    }


# ---------------------------------------------------------------------------
# Model prediction (torch / transformers imported lazily)
# ---------------------------------------------------------------------------


def predict_dataset(
    model_dir: Path, dataset_path: Path, config_path: Path | None = None
) -> list[dict]:  # pragma: no cover - needs GPU/ML stack
    """Attach ``pred_score`` (and ``pred_categ`` if a category head) to each dataset
    row using a trained checkpoint. Reconstructs the model from the saved run_config."""
    import torch  # noqa: PLC0415
    import train as T  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    run_cfg = json.loads((model_dir / "run_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(run_cfg["model_name"])
    model = T.build_model(run_cfg)
    state = torch.load(model_dir / "pytorch_model.bin", map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    rows = CM.read_dataset(dataset_path)
    texts = [r.get(run_cfg["text_col"], "") for r in rows]
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            batch = texts[i : i + 64]
            enc = tokenizer(batch, truncation=True, max_length=run_cfg["max_length"], padding=True, return_tensors="pt")
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            preds = out["predictions"]
            scores = (preds[0] if isinstance(preds, tuple) else preds).tolist()
            cats = None
            if isinstance(preds, tuple) and preds[1] is not None:
                cats = [T.ID_TO_CATEGORY[int(row.index(max(row)))] for row in preds[1].tolist()]
            for j, s in enumerate(scores):
                rows[i + j]["pred_score"] = round(float(s), 4)
                if cats:
                    rows[i + j]["pred_categ"] = cats[j]
    return rows


# ---------------------------------------------------------------------------
# Reporting + CLI
# ---------------------------------------------------------------------------


def format_report(report: dict) -> str:
    reg, band = report["regression"], report["banded_vs_algo"]
    lines = [
        "Quality-model evaluation (vs algorithm)",
        "=" * 40,
        f"n={report['n']}",
        f"regression: MAE={reg['mae']:.4f} RMSE={reg['rmse']:.4f} Spearman={reg['spearman']:.4f}",
        f"banded vs algo categ: accuracy={band['accuracy']:.4f} macro_f1={band['macro_f1']:.4f}",
        f"monotonicity: accuracy={report['monotonicity']['accuracy']:.4f} ({report['monotonicity']['pairs']} pairs)",
    ]
    if "cathead_vs_algo" in report:
        ch = report["cathead_vs_algo"]
        lines.append(f"category head vs algo: accuracy={ch['accuracy']:.4f} macro_f1={ch['macro_f1']:.4f}")
    lines.append("")
    lines.append("calibration (pred vs true per bin):")
    for c in report["calibration"]:
        if c.get("count"):
            lines.append(
                f"    {c['bin']}  n={c['count']:>5}  pred={c['mean_pred']:.3f} true={c['mean_true']:.3f} gap={c['gap']:+.3f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a quality-score model.")
    p.add_argument("--predictions", type=Path, default=None, help="CSV with true + predicted columns.")
    p.add_argument("--pred-score-col", default="pred_score")
    p.add_argument("--pred-categ-col", default=None)
    p.add_argument("--true-score-col", default="score_raw")
    p.add_argument("--true-categ-col", default="categ")
    p.add_argument("--stratify-by", default=None, help="Column to break metrics down by (e.g. lang, provenance).")
    p.add_argument(
        "--gold", type=Path, default=None, help="Gold CSV (gold_categ + algo_categ + pred_score/pred_categ)."
    )
    p.add_argument("--margin", type=float, default=0.01)
    p.add_argument("--model-dir", type=Path, default=None, help="Trained checkpoint dir (predict then evaluate).")
    p.add_argument("--dataset", type=Path, default=None, help="Dataset CSV to predict on (with --model-dir).")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    if args.model_dir and args.dataset:  # pragma: no cover - needs ML stack
        rows = predict_dataset(args.model_dir, args.dataset)
    elif args.predictions:
        rows = CM.read_dataset(args.predictions)
    else:
        rows = []

    result: dict = {}
    if rows:
        result["vs_algorithm"] = evaluate_predictions(
            rows,
            true_score_col=args.true_score_col,
            pred_score_col=args.pred_score_col,
            true_categ_col=args.true_categ_col,
            pred_categ_col=args.pred_categ_col,
            stratify_by=args.stratify_by,
        )
        print(format_report(result["vs_algorithm"]))

    if args.gold:
        gold_rows = CM.read_dataset(args.gold)
        gate = gold_gate(
            gold_rows, pred_score_col=args.pred_score_col, pred_categ_col=args.pred_categ_col, margin=args.margin
        )
        result["gold_gate"] = gate
        verdict = "PASS" if gate["passed"] else "FAIL"
        print(f"\nGOLD GATE: model_f1={gate['model_macro_f1']:.4f} vs algo_f1={gate['algo_macro_f1']:.4f} -> {verdict}")

    if args.json and result:
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
