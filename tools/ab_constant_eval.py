#!/usr/bin/env python3
"""
tools/ab_constant_eval.py
=========================
A/B (or N-way) comparison of a single config constant against the frozen
ground-truth categories.

Unlike the importance sweep (which samples the whole space) this tool answers one
focused question: "if I move constant X to value V, how does agreement with the
stored categories change?" It runs the real production engine via
``evaluate_dataframe`` -- the same path as ``recategorize_from_csv`` -- so every
number is measured against the immutable ``categ`` ground truth.

Primary use: validate the importance sweep's one substantive high-impact
suggestion, ``CATEG_GARBAGE_DENSITY_HIGH`` 0.35 -> ~0.55, before touching the
production default.

CAVEAT: ``CATEG_GARBAGE_DENSITY_HIGH`` is reused in three places -- the hard
``rule_garbage_density`` gate, the quality-score garbage normalisation, and the
short-line penalty -- so moving it changes all three at once. Read the deltas
with that coupling in mind.

Example
-------
    python tools/ab_constant_eval.py \\
        --input-dir data_samples/DOC_LINE_CATEG --config config_langID.txt \\
        --const CATEG_GARBAGE_DENSITY_HIGH --values 0.35,0.55
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from recategorize_from_csv import (  # noqa: E402
    _load_lang_config,
    evaluate_dataframe,
    load_csvs,
    read_config_constants,
)


def _clear_loss(metrics: Dict[str, Any]) -> int:
    """True Clear -> Trash/Non-text count vs. ground truth (from the confusion matrix)."""
    clear_row = metrics.get("confusion", {}).get("Clear", {})
    return int(clear_row.get("Trash", 0)) + int(clear_row.get("Non-text", 0))


def _trash_recall(metrics: Dict[str, Any]) -> float:
    """Share of ground-truth Trash lines still predicted Trash (catches garbage leakage)."""
    conf = metrics.get("confusion", {})
    trash_row = conf.get("Trash", {})
    support = sum(int(v) for v in trash_row.values())
    return float(int(trash_row.get("Trash", 0)) / support) if support else float("nan")


def run_ab(
    df,
    const_name: str,
    values: List[float],
    base_constants: Dict[str, Any],
    eval_kwargs: Dict[str, Any],
) -> None:
    base_value = base_constants.get(const_name)
    n_lines = len(df)
    print(f"A/B on `{const_name}` over {n_lines:,} lines (current config value: {base_value})\n")

    rows: List[Dict[str, Any]] = []
    for value in values:
        trial = {**base_constants, const_name: value}
        metrics = evaluate_dataframe(df, trial, **eval_kwargs)
        rows.append(
            {
                "value": value,
                "macro_f1": float(metrics["macro_f1"]),
                "weighted_f1": float(metrics["weighted_f1"]),
                "costed_score": float(metrics["costed_score"]),
                "flip_rate": float(metrics["flip_rate"]),
                "trash_rate": float(metrics["trash_rate"]),
                "clear_rate": float(metrics["clear_rate"]),
                "kl": float(metrics["kl_divergence"]),
                "clear_loss": _clear_loss(metrics),
                "trash_recall": _trash_recall(metrics),
            }
        )

    header = (
        f"| {const_name} | macro_f1 | weighted_f1 | costed_score | flip_rate | "
        f"trash_rate | clear_rate | KL | Clear-loss | Trash-recall |"
    )
    print(header)
    print("| " + " | ".join(["---"] * 10) + " |")
    for r in rows:
        print(
            f"| {r['value']} | {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} | "
            f"{r['costed_score']:.4f} | {r['flip_rate']:.4f} | {r['trash_rate']:.4f} | "
            f"{r['clear_rate']:.4f} | {r['kl']:.5f} | {r['clear_loss']:,} | {r['trash_recall']:.4f} |"
        )

    # Headline delta against the first value (treated as the reference).
    if len(rows) >= 2:
        ref, alt = rows[0], rows[-1]
        print(
            f"\nΔ ({alt['value']} vs {ref['value']}): "
            f"macro_f1 {alt['macro_f1'] - ref['macro_f1']:+.4f}, "
            f"costed_score {alt['costed_score'] - ref['costed_score']:+.4f}, "
            f"trash_rate {alt['trash_rate'] - ref['trash_rate']:+.4f}, "
            f"Clear-loss {alt['clear_loss'] - ref['clear_loss']:+d}"
        )
    print(
        "\nNote: ground-truth flip_rate is ~0 at the current config by construction, so a "
        "non-zero flip_rate / macro_f1 < 1 here is deviation FROM the stored categories.\n"
        "Run on the full DOC_LINE_CATEG corpus -- the bundled sample is a smoke fixture."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B a single config constant vs. the stored categories.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of DOC_LINE_CATEG CSVs.")
    parser.add_argument("--config", type=str, default="config_langID.txt", help="config_langID.txt-style file.")
    parser.add_argument("--const", type=str, default="CATEG_GARBAGE_DENSITY_HIGH", help="Constant to vary.")
    parser.add_argument(
        "--values",
        type=str,
        default="0.35,0.55",
        help="Comma-separated values to test (first is the reference for deltas).",
    )
    args = parser.parse_args()

    values = [float(v.strip()) for v in args.values.split(",") if v.strip()]
    if not values:
        print("error: provide at least one --values entry", file=sys.stderr)
        sys.exit(1)

    df = load_csvs(args.input_dir, recursive=True)
    expected_langs, known_bases = _load_lang_config(args.config)
    base_constants = read_config_constants(args.config)
    eval_kwargs = {"expected_langs": expected_langs, "known_bases": known_bases}

    run_ab(df, args.const, values, base_constants, eval_kwargs)


if __name__ == "__main__":
    main()
