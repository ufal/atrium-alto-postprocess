#!/usr/bin/env python3
"""
tools/run_ablation_study.py
===========================
Executes a Leave-One-Out (LOO) ablation sweep over the ALTO post-processing rules.

It evaluates:
1. The Feature Variance (why some QS weights are useless).
2. Total Flips (Marginal Effect of the ablated rule).
3. Clear -> Trash Cost (Did removing this rule destructively destroy valid text?).

Output is a Markdown matrix summarizing keep/prune decisions for PR reviews.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Point Python to the root text_util_langID
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from recategorize_from_csv import _load_lang_config, evaluate_dataframe, load_csvs, read_config_constants  # noqa: E402

from text_util_langID import QS_WEIGHT_NAMES, override_constants  # noqa: E402

RULES_TO_ABLATE = [
    "rule_hard_sweep",
    "rule_extreme_ppl",
    "rule_absolute_ppl",
    "rule_inverted",
    "rule_allcaps",
    "rule_garbage_density",
    "rule_short_garbage",
    "rule_lowppl_clear",
    "rule_short_fragment_noisy",
    "rule_mostly_readable_noisy",
    "rule_trailing_fill_rescue",
    "penalty_wqx_rot",
    "penalty_vowelless",
    "penalty_ledger_fragmentation",
    "penalty_mid_uppercase",
]


def format_decision(flips: int, destructive_cost: int, std_dev: float = None) -> str:
    """Provides an automated recommendation based on strict asymmetric costing."""
    if std_dev is not None and std_dev < 0.005:
        return "**PRUNE** (Signal variance ≈ 0)"
    if flips == 0:
        return "**PRUNE** (Fully redundant)"
    if destructive_cost > 0:
        return "**KEEP** (Destructive to valid text if removed)"
    if flips > 100:
        return "**KEEP** (High Impact)"
    return "**REVIEW** (Low Impact)"


def run_ablation(df: pd.DataFrame, eval_kwargs: dict, base_constants: dict):
    print("Evaluating Baseline...")
    base_metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)
    base_lines = len(df)

    print("\nCalculating Feature Variance for Continuous Signals...")
    # Map QS_WEIGHT constants to their CSV dataframe columns
    feature_map = {
        "QS_WEIGHT_VOWEL": "vowel_ratio",
        "QS_WEIGHT_WEIRD": "weird_ratio",
        "QS_WEIGHT_GIBBERISH": "gibberish_ratio",
        "QS_WEIGHT_FUSED": "fused_ratio",
        "QS_WEIGHT_GARBAGE": "garbage_density",
    }

    report_rows = []

    # 1. Ablate QS Weights (Part A)
    for weight_name in QS_WEIGHT_NAMES:
        col = feature_map.get(weight_name)
        std_dev = df[col].std() if col in df.columns else None

        with override_constants({weight_name: 0.0}):
            metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

        flips = int(metrics["flip_rate"] * base_lines)

        # Calculate destructive cost (increase in trash rate while clear rate drops)
        clear_drop = max(0.0, base_metrics["clear_rate"] - metrics["clear_rate"])
        trash_rise = max(0.0, metrics["trash_rate"] - base_metrics["trash_rate"])
        destructive_cost = int(min(clear_drop, trash_rise) * base_lines)

        decision = format_decision(flips, destructive_cost, std_dev)
        report_rows.append(
            (
                weight_name,
                f"Continuous (σ={std_dev:.3f})" if std_dev else "Continuous",
                flips,
                destructive_cost,
                decision,
            )
        )

    # 2. Ablate Discrete Rules (Part B & C)
    for rule in RULES_TO_ABLATE:
        with override_constants({"DISABLED_RULES": frozenset([rule])}):
            metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

        flips = int(metrics["flip_rate"] * base_lines)

        clear_drop = max(0.0, base_metrics["clear_rate"] - metrics["clear_rate"])
        trash_rise = max(0.0, metrics["trash_rate"] - base_metrics["trash_rate"])
        destructive_cost = int(min(clear_drop, trash_rise) * base_lines)

        decision = format_decision(flips, destructive_cost)
        report_rows.append((rule, "Binary Gateway", flips, destructive_cost, decision))

    # 3. Render Matrix Output
    print("\n### Ablation Study Results")
    print("| Rule / Factor | Signal Type | Marginal Flips (LOO) | `Clear -> Trash` Cost | Decision |")
    print("| --- | --- | --- | --- | --- |")
    # Sort so High Impact/Keep items bubble to the top
    sorted_rows = sorted(report_rows, key=lambda x: (x[3] > 0, x[2]), reverse=True)
    for row in sorted_rows:
        print(f"| `{row[0]}` | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")


def main():
    parser = argparse.ArgumentParser(description="Run LOO Ablation on heuristics")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to DOC_LINE_CATEG cached CSVs")
    parser.add_argument("--config", type=str, default="config_langID.txt")
    args = parser.parse_args()

    print(f"Loading cached dataset from {args.input_dir}...")
    df = load_csvs(args.input_dir, recursive=True)

    expected_langs, known_bases = _load_lang_config(args.config)
    eval_kwargs = {"expected_langs": expected_langs, "known_bases": known_bases}
    base_constants = read_config_constants(args.config)

    run_ablation(df, eval_kwargs, base_constants)


if __name__ == "__main__":
    main()
