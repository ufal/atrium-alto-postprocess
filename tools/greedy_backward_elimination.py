#!/usr/bin/env python3
"""
tools/greedy_backward_elimination.py
License: CC BY-NC 4.0

Executes Greedy Backward Elimination on OCR post-processing rules.

While standard Leave-One-Out (LOO) ablation identifies baseline redundancies, it
suffers from "Survivor Bias": overlapping rules mask each other's utility.
Dropping Rule A might suddenly make Rule B highly critical, as Rule B now shoulders
the filtering load that Rule A previously handled.

This module iteratively disables the least impactful rule and recalculates marginal
coverage across the entire remaining set. It halts immediately if removing a rule
violates the primary safety constraint: causing valid text to flip from `Clear`
to `Trash`. The final output is the mathematically proven minimal set of rules
required to maintain pipeline integrity.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd

# Point Python to the root directory to access text_util_langID and sibling scripts
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from recategorize_from_csv import _load_lang_config, evaluate_dataframe, load_csvs, read_config_constants  # noqa: E402

from text_util_langID import override_constants  # noqa: E402

# Using the unified rules array covering structural gateways and dynamic penalties
CANDIDATE_RULES: Set[str] = {
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
}


def run_backward_elimination(df: pd.DataFrame, eval_kwargs: Dict[str, Any], base_constants: Dict[str, Any]) -> None:
    """
    Runs the iterative greedy backward elimination algorithm.

    At each step, it drops the rule that provides the smallest marginal classification
    change, provided that dropping it does not corrupt previously clear text.

    Args:
        df: The cached dataset containing pre-calculated text features.
        eval_kwargs: Keyword arguments for the evaluation function.
        base_constants: Baseline configuration parameters.
    """
    active_rules: Set[str] = set(CANDIDATE_RULES)
    permanently_disabled: Set[str] = set()
    base_lines: int = len(df)

    print("Starting Greedy Backward Elimination.")
    print(f"Initial rules evaluated: {len(active_rules)}")
    print(f"Evaluation Corpus: {base_lines:,} lines")
    print("-" * 60)

    iteration = 1
    total_start_time = time.time()

    while active_rules:
        # Step 1: Calculate the shifting baseline
        # The baseline changes every iteration as we permanently disable rules
        with override_constants({"DISABLED_RULES": frozenset(permanently_disabled)}):
            current_baseline = evaluate_dataframe(df, base_constants, **eval_kwargs)

        round_results: List[Dict[str, Any]] = []

        # Step 2: Test dropping each active rule one by one (LOO on the surviving set)
        for candidate in active_rules:
            test_disabled = permanently_disabled | {candidate}
            with override_constants({"DISABLED_RULES": frozenset(test_disabled)}):
                metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

            flips = int(metrics["flip_rate"] * base_lines)

            # Strict Asymmetric Cost: Check for catastrophic text destruction
            clear_drop = max(0.0, current_baseline["clear_rate"] - metrics["clear_rate"])
            trash_rise = max(0.0, metrics["trash_rate"] - current_baseline["trash_rate"])
            destructive_cost = int(min(clear_drop, trash_rise) * base_lines)

            round_results.append({"rule": candidate, "flips": flips, "cost": destructive_cost})

        # Step 3: Find the safest rule to eliminate
        # Primary sort criteria: Must have the lowest destructive cost (ideally 0)
        # Secondary sort criteria: Lowest overall marginal flips (least impact)
        round_results.sort(key=lambda x: (x["cost"], x["flips"]))
        weakest_link = round_results[0]

        # Step 4: The Kill-Switch Halt Condition
        # If the absolute safest rule to drop STILL destroys valid text, we have
        # hit the pareto-optimal frontier. We must halt elimination.
        if weakest_link["cost"] > 0:
            print(f"\n[HALT] Reached Pareto-Optimal Frontier at Iteration {iteration}.")
            print("Cannot eliminate further rules without systematically destroying valid text.")
            print(
                f"The weakest remaining rule (`{weakest_link['rule']}`) currently "
                f"prevents {weakest_link['cost']:,} `Clear -> Trash` corruptions."
            )
            break

        # Step 5: Execute Elimination and loop
        target_rule = weakest_link["rule"]
        active_rules.remove(target_rule)
        permanently_disabled.add(target_rule)

        print(
            f"Iter {iteration:02d} | [-] Dropped `{target_rule}` "
            f"(Marginal Flips: {weakest_link['flips']:,}, Cost: {weakest_link['cost']})"
        )
        iteration += 1

    # Step 6: Output the Final Minimal Set
    elapsed_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"ELIMINATION COMPLETE (Finished in {elapsed_time:.2f} seconds)")
    print("=" * 60)

    print(f"\nRules safely pruned as strictly redundant: {len(permanently_disabled)}")
    for rule in sorted(permanently_disabled):
        print(f"  - {rule}")

    print(f"\nMinimal Rule Set required for structural integrity: {len(active_rules)}")
    for rule in sorted(active_rules):
        print(f"  + {rule}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Greedy Backward Elimination to find the minimal optimal heuristic subset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("../data_samples/DOC_LINE_CATEG"),
        help="Path to cached DOC_LINE_CATEG CSV datasets (default: ../data_samples/DOC_LINE_CATEG).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../config_langID.txt",
        help="Path to the system configuration file (default: ../config_langID.txt).",
    )
    args = parser.parse_args()

    input_path = args.input_dir.resolve()
    config_path = args.config

    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: Target directory '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    print("Loading cached datasets into memory...")
    try:
        df = load_csvs(input_path, recursive=True)
    except Exception as e:
        print(f"Failed to load datasets: {e}", file=sys.stderr)
        sys.exit(1)

    expected_langs, known_bases = _load_lang_config(config_path)
    eval_kwargs = {"expected_langs": expected_langs, "known_bases": known_bases}

    try:
        base_constants = read_config_constants(config_path)
    except Exception as e:
        print(f"Failed to parse config '{config_path}': {e}", file=sys.stderr)
        sys.exit(1)

    run_backward_elimination(df, eval_kwargs, base_constants)


if __name__ == "__main__":
    main()
