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
   final categorization when a specific rule is removed. Because the baseline
   reproduces the stored ground-truth categories (`flip_rate == 0` by
   construction), every marginal flip is, by definition, a *newly introduced
   disagreement with the ground truth* -- i.e. real damage, not free movement.
3. Destructive Cost: The TRUE per-line count of `Clear -> Trash` / `Clear ->
   Non-text` transitions, read straight from the confusion matrix that
   ``evaluate_dataframe`` already computes against the frozen ground truth.
   (Earlier versions used an aggregate ``min(clear_drop, trash_rise)`` rate
   proxy that only fired when Clear shrank AND Trash grew at the same time; it
   was blind to Trash->Noisy/Clear leakage and Noisy->Clear promotion, which is
   why it scored every rule as "cost 0 / prune". We now use the real matrix.)
4. Macro-F1 delta: How much agreement with the stored categories is lost when
   the rule is disabled -- the accuracy cost the proxy ignored.
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

from recategorize_from_csv import (  # noqa: E402
    QS_WEIGHT_NAMES,
    _load_lang_config,
    evaluate_dataframe,
    load_csvs,
    read_config_constants,
)  # noqa: E402

from text_util_langID import override_constants  # noqa: E402

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


def format_decision(flips: int, clear_loss: int, macro_f1_drop: float, std_dev: Optional[float] = None) -> str:
    # A genuinely prunable feature must change NOTHING vs. the ground truth.
    if std_dev is not None and std_dev < 0.005:
        return "**PRUNE** (Signal variance ≈ 0; feature absent from dataset)"
    if clear_loss > 0:
        return f"**KEEP** (Critical safeguard: destroys valid text — {clear_loss:,} `Clear -> Trash/Non-text`)"
    if flips == 0 and macro_f1_drop <= 1e-9:
        return "**PRUNE** (Fully redundant; zero ground-truth change)"
    if flips > 100:
        return "**KEEP** (High Impact routing rule)"
    # flips > 0 means disabling the rule moved lines away from the ground truth,
    # even if no Clear was lost. That is real damage the old proxy hid as "cost 0".
    return "**REVIEW** (Introduces ground-truth disagreements; weigh vs. complexity)"


def true_clear_loss(metrics: Dict[str, Any]) -> int:
    """True per-line count of `Clear -> Trash`/`Clear -> Non-text` transitions vs.
    the frozen ground truth, read from the confusion matrix evaluate_dataframe
    already builds. Replaces the old aggregate ``min(clear_drop, trash_rise)`` proxy."""
    clear_row = metrics.get("confusion", {}).get("Clear", {})
    return int(clear_row.get("Trash", 0)) + int(clear_row.get("Non-text", 0))


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

    base_macro_f1 = float(base_metrics["macro_f1"])
    # (name, flips, clear_loss, macro_f1_drop, costed_score, decision)
    report_rows: List[Tuple[str, int, int, float, float, str]] = []

    # Phase 1: Continuous Signals
    for weight_name in QS_WEIGHT_NAMES:
        col = feature_map.get(weight_name)
        std_dev = pd.to_numeric(df[col], errors="coerce").std() if col and col in df.columns else None

        with override_constants({weight_name: 0.0}):
            metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

        flips = int(metrics["flip_rate"] * base_lines)
        clear_loss = true_clear_loss(metrics)
        macro_drop = base_macro_f1 - float(metrics["macro_f1"])
        decision = format_decision(flips, clear_loss, macro_drop, std_dev)
        report_rows.append((weight_name, flips, clear_loss, macro_drop, float(metrics["costed_score"]), decision))

    # Phase 2: Binary Gateways
    for rule in RULES_TO_ABLATE:
        with override_constants({"DISABLED_RULES": frozenset([rule])}):
            metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

        flips = int(metrics["flip_rate"] * base_lines)
        clear_loss = true_clear_loss(metrics)
        macro_drop = base_macro_f1 - float(metrics["macro_f1"])
        decision = format_decision(flips, clear_loss, macro_drop)
        report_rows.append((rule, flips, clear_loss, macro_drop, float(metrics["costed_score"]), decision))

    # Phase 3: Markdown Render
    print("### System Ablation Study Results")
    print(
        f"*Total Evaluation Corpus: {base_lines:,} lines (baseline macro-F1 vs. ground truth: {base_macro_f1:.4f})*\n"
    )
    print(
        "| Rule / Factor | Marginal Flips (LOO) | `Clear -> Trash/Non-text` (true) | Macro-F1 Δ | costed_score | Decision |"
    )
    print("| --- | --- | --- | --- | --- | --- |")

    # Most damaging first: real Clear-loss, then accuracy loss, then raw flips.
    sorted_rows = sorted(report_rows, key=lambda r: (r[2], r[3], r[1]), reverse=True)
    for name, flips, clear_loss, macro_drop, costed, decision in sorted_rows:
        print(f"| `{name}` | {flips:,} | {clear_loss:,} | {macro_drop:+.4f} | {costed:.4f} | {decision} |")


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
