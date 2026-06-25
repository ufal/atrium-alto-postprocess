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
coverage across the entire remaining set. It halts as soon as the safest remaining
removal would degrade agreement with the frozen ground-truth categories -- measured
with the REAL signals ``evaluate_dataframe`` already returns: the per-line confusion
matrix (true `Clear -> Trash`/`Clear -> Non-text` loss) and the macro-F1 against the
stored categories. The final output is the minimal set of rules that can be dropped
without moving any line away from its ground-truth category.

NOTE on the previous metric: earlier versions gated elimination on an aggregate
``min(clear_drop, trash_rise)`` rate proxy. That proxy only triggered when the Clear
rate shrank AND the Trash rate grew at the same time, so it was blind to the dominant
effects of dropping these rules -- `Trash -> Noisy/Clear` leakage (garbage surviving)
and `Noisy -> Clear` promotion. With the proxy reading 0 for every rule, the loop
eliminated all 15 and reported a "Minimal Rule Set: 0". That conclusion was a metric
artifact, not a property of the engine; this version measures the real damage.
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
    "rule_mostly_readable_noisy",
    "rule_trailing_fill_rescue",
    "penalty_wqx_rot",
    "penalty_vowelless",
    "penalty_ledger_fragmentation",
    "penalty_mid_uppercase",
}


def _clear_loss(metrics: Dict[str, Any]) -> int:
    """True per-line `Clear -> Trash`/`Clear -> Non-text` count vs. the frozen ground
    truth, from the confusion matrix evaluate_dataframe already builds."""
    clear_row = metrics.get("confusion", {}).get("Clear", {})
    return int(clear_row.get("Trash", 0)) + int(clear_row.get("Non-text", 0))


def run_backward_elimination(
    df: pd.DataFrame,
    eval_kwargs: Dict[str, Any],
    base_constants: Dict[str, Any],
    macro_tol: float = 0.0,
) -> None:
    """
    Runs the iterative greedy backward elimination algorithm.

    At each step it drops the rule whose removal does the least damage to agreement
    with the stored ground-truth categories, and halts once even the safest removal
    would (a) push any `Clear` line into `Trash`/`Non-text`, or (b) drop macro-F1 (vs.
    the ground truth) by more than ``macro_tol``. With ``macro_tol == 0`` a rule is
    pruned only if removing it changes nothing the ground truth cares about.

    Args:
        df: The cached dataset containing pre-calculated text features.
        eval_kwargs: Keyword arguments for the evaluation function.
        base_constants: Baseline configuration parameters.
        macro_tol: Allowed marginal macro-F1 loss per elimination (default 0.0 =
            strict). Calibrate on the full corpus; tiny tolerances absorb float noise.
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
        base_clear_loss = _clear_loss(current_baseline)
        base_macro_f1 = float(current_baseline["macro_f1"])

        round_results: List[Dict[str, Any]] = []

        # Step 2: Test dropping each active rule one by one (LOO on the surviving set)
        for candidate in active_rules:
            test_disabled = permanently_disabled | {candidate}
            with override_constants({"DISABLED_RULES": frozenset(test_disabled)}):
                metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

            flips = int(metrics["flip_rate"] * base_lines)

            # REAL damage vs. ground truth (not the old min(clear_drop, trash_rise) proxy):
            #   - clear_loss: true Clear -> Trash/Non-text transitions newly introduced
            #   - macro_drop: marginal macro-F1 lost vs. the current baseline (>= 0,
            #     because the baseline already matches ground truth so disabling a rule
            #     can only add disagreements). This catches Trash-leakage the proxy missed.
            clear_loss = max(0, _clear_loss(metrics) - base_clear_loss)
            macro_drop = base_macro_f1 - float(metrics["macro_f1"])
            round_results.append(
                {
                    "rule": candidate,
                    "flips": flips,
                    "clear_loss": clear_loss,
                    "macro_drop": macro_drop,
                    "costed_score": float(metrics["costed_score"]),
                }
            )

        # Step 3: Find the safest rule to eliminate
        # Sort by least real damage: no Clear-loss first, then least macro-F1 loss, then flips.
        round_results.sort(key=lambda x: (x["clear_loss"], round(x["macro_drop"], 9), x["flips"]))
        weakest_link = round_results[0]

        # Step 4: The Kill-Switch Halt Condition
        # Halt once even the safest removal degrades agreement with the ground truth:
        # any Clear-loss, or a marginal macro-F1 drop beyond tolerance.
        if weakest_link["clear_loss"] > 0 or weakest_link["macro_drop"] > macro_tol + 1e-9:
            print(f"\n[HALT] Reached ground-truth-preserving frontier at Iteration {iteration}.")
            print("Cannot eliminate further rules without moving lines away from their stored category.")
            print(
                f"The safest remaining rule (`{weakest_link['rule']}`) still costs "
                f"{weakest_link['clear_loss']:,} `Clear -> Trash/Non-text` and "
                f"{weakest_link['macro_drop']:+.4f} macro-F1 if dropped."
            )
            break

        # Step 5: Execute Elimination and loop
        target_rule = weakest_link["rule"]
        active_rules.remove(target_rule)
        permanently_disabled.add(target_rule)

        print(
            f"Iter {iteration:02d} | [-] Dropped `{target_rule}` "
            f"(Flips: {weakest_link['flips']:,}, Clear-loss: {weakest_link['clear_loss']}, "
            f"Macro-F1 Δ: {weakest_link['macro_drop']:+.4f})"
        )
        iteration += 1

    # Step 6: Output the Final Minimal Set
    elapsed_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"ELIMINATION COMPLETE (Finished in {elapsed_time:.2f} seconds)")
    print("=" * 60)

    print(f"\nRules safely pruned (zero ground-truth loss within tolerance): {len(permanently_disabled)}")
    for rule in sorted(permanently_disabled):
        print(f"  - {rule}")

    print(f"\nMinimal Rule Set required to preserve the stored categories: {len(active_rules)}")
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
    parser.add_argument(
        "--macro-tol",
        type=float,
        default=0.0,
        help=(
            "Allowed marginal macro-F1 loss (vs. ground truth) per elimination. "
            "0.0 (default) = strict: prune only rules whose removal changes nothing. "
            "Raise it to permit small, deliberate accuracy trade-offs."
        ),
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

    run_backward_elimination(df, eval_kwargs, base_constants, macro_tol=args.macro_tol)


if __name__ == "__main__":
    main()
