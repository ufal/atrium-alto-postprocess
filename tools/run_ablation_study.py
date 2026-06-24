#!/usr/bin/env python3
"""
tools/run_ablation_study.py
===========================
Executes a Leave-One-Out (LOO) ablation sweep over the ALTO post-processing rules.

This module provides a rigorous, automated feature selection pipeline for the
heuristic text classification engine. By systematically disabling individual
rules and evaluating the model against a cached dataset, it calculates:

1. Feature Variance: Identifies continuous quality score weights that have zero
   signal variance in the dataset.
2. Marginal Flips (LOO): The total number of document lines that change their
   final categorization when a specific rule is removed.
3. Destructive Cost: A strict asymmetric penalty applied when disabling a rule
   causes genuine text to be falsely categorized as garbage (`Clear -> Trash`).
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from recategorize_from_csv import _load_lang_config, evaluate_dataframe, load_csvs, read_config_constants  # noqa: E402

from text_util_langID import QS_WEIGHT_NAMES, override_constants  # noqa: E402

RULES_TO_ABLATE: List[str] = [
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


def format_decision(flips: int, destructive_cost: int, std_dev: Optional[float] = None) -> str:
    if std_dev is not None and std_dev < 0.005:
        return "**PRUNE** (Signal variance ≈ 0; feature absent from dataset)"
    if flips == 0:
        return "**PRUNE** (Fully redundant; zero marginal effect)"
    if destructive_cost > 0:
        return "**KEEP** (Critical safeguard: Destructive to valid text)"
    if flips > 100:
        return "**KEEP** (High Impact routing rule)"
    return "**REVIEW** (Low Impact; consider pruning if complexity is high)"


def run_ablation(df: pd.DataFrame, eval_kwargs: Dict[str, Any], base_constants: Dict[str, Any]) -> None:
    print(f"Evaluating Baseline across {len(df):,} document lines...")
    start_time = time.time()
    base_metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)
    base_lines = len(df)
    print(f"Baseline established in {time.time() - start_time:.2f} seconds.\n")

    feature_map: Dict[str, str] = {
        "QS_WEIGHT_VOWEL": "vowel_ratio",
        "QS_WEIGHT_WEIRD": "weird_ratio",
        "QS_WEIGHT_GIBBERISH": "gibberish_ratio",
        "QS_WEIGHT_FUSED": "fused_ratio",
        "QS_WEIGHT_GARBAGE": "garbage_density",
    }

    report_rows: List[Tuple[str, str, int, int, str]] = []

    # Phase 1: Continuous Signals
    for weight_name in QS_WEIGHT_NAMES:
        col = feature_map.get(weight_name)
        std_dev = df[col].std() if col and col in df.columns else None

        with override_constants({weight_name: 0.0}):
            metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

        flips = int(metrics["flip_rate"] * base_lines)
        clear_drop = max(0.0, base_metrics["clear_rate"] - metrics["clear_rate"])
        trash_rise = max(0.0, metrics["trash_rate"] - base_metrics["trash_rate"])
        destructive_cost = int(min(clear_drop, trash_rise) * base_lines)

        decision = format_decision(flips, destructive_cost, std_dev)
        raw_cov = "(Continuous)"
        report_rows.append((weight_name, raw_cov, flips, destructive_cost, decision))

    # Phase 2: Binary Gateways
    for rule in RULES_TO_ABLATE:
        with override_constants({"DISABLED_RULES": frozenset([rule])}):
            metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

        flips = int(metrics["flip_rate"] * base_lines)
        clear_drop = max(0.0, base_metrics["clear_rate"] - metrics["clear_rate"])
        trash_rise = max(0.0, metrics["trash_rate"] - base_metrics["trash_rate"])
        destructive_cost = int(min(clear_drop, trash_rise) * base_lines)

        decision = format_decision(flips, destructive_cost)
        raw_cov = "N/A (LOO)"
        report_rows.append((rule, raw_cov, flips, destructive_cost, decision))

    # Phase 3: Markdown Render
    print("### System Ablation Study Results")
    print(f"*Total Evaluation Corpus: {base_lines:,} lines*\n")
    print("| Rule / Factor | Raw Coverage | Marginal Flips (LOO) | `Clear -> Trash` Cost | Decision |")
    print("| --- | --- | --- | --- | --- |")

    sorted_rows = sorted(report_rows, key=lambda x: (x[3] > 0, x[2]), reverse=True)
    for row in sorted_rows:
        print(f"| `{row[0]}` | {row[1]} | {row[2]:,} | {row[3]:,} | {row[4]} |")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOO Ablation on heuristics.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--config", type=str, default="config_langID.txt")
    args = parser.parse_args()

    df = load_csvs(args.input_dir, recursive=True)
    expected_langs, known_bases = _load_lang_config(args.config)
    run_ablation(df, {"expected_langs": expected_langs, "known_bases": known_bases}, read_config_constants(args.config))


if __name__ == "__main__":
    main()
