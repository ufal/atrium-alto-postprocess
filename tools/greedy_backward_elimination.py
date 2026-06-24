#!/usr/bin/env python3
"""
tools/greedy_backward_elimination.py
License: CC BY-NC 4.0

Executes Greedy Backward Elimination on OCR post-processing rules.
Iteratively disables the least impactful rule and recalculates marginal
coverage, halting immediately if removing a rule causes `Clear -> Trash` flips.
"""

import sys
from pathlib import Path

import pandas as pd

# Point Python to the root text_util_langID
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from recategorize_from_csv import _load_lang_config, evaluate_dataframe, load_csvs, read_config_constants  # noqa: E402

from text_util_langID import override_constants  # noqa: E402

# Using the unified rules array
CANDIDATE_RULES = {
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


def run_backward_elimination(df: pd.DataFrame, eval_kwargs: dict, base_constants: dict):
    active_rules = set(CANDIDATE_RULES)
    permanently_disabled = set()
    base_lines = len(df)

    print(f"Starting Greedy Backward Elimination. Initial rules: {len(active_rules)}")

    while active_rules:
        # Step 1: Calculate the shifting baseline
        with override_constants({"DISABLED_RULES": frozenset(permanently_disabled)}):
            current_baseline = evaluate_dataframe(df, base_constants, **eval_kwargs)

        round_results = []

        # Step 2: Test dropping each active rule one by one
        for candidate in active_rules:
            test_disabled = permanently_disabled | {candidate}
            with override_constants({"DISABLED_RULES": frozenset(test_disabled)}):
                metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

            flips = int(metrics["flip_rate"] * base_lines)

            # Strict Asymmetric Cost: Did we destroy valid text?
            clear_drop = max(0.0, current_baseline["clear_rate"] - metrics["clear_rate"])
            trash_rise = max(0.0, metrics["trash_rate"] - current_baseline["trash_rate"])
            destructive_cost = int(min(clear_drop, trash_rise) * base_lines)

            round_results.append({"rule": candidate, "flips": flips, "cost": destructive_cost})

        # Step 3: Find the safest rule to eliminate
        # Sort by: Lowest destructive cost (must be 0), then lowest overall marginal flips
        round_results.sort(key=lambda x: (x["cost"], x["flips"]))
        weakest_link = round_results[0]

        # Step 4: The Kill-Switch Halt Condition
        if weakest_link["cost"] > 0:
            print("\n[HALT] Cannot eliminate further without destroying valid text.")
            print(
                f"The weakest remaining rule (`{weakest_link['rule']}`) costs {weakest_link['cost']} `Clear -> Trash` flips."
            )
            break

        # Eliminate and loop
        target_rule = weakest_link["rule"]
        active_rules.remove(target_rule)
        permanently_disabled.add(target_rule)

        print(f"[-] Dropped `{target_rule}` (Marginal Flips: {weakest_link['flips']}, Cost: {weakest_link['cost']})")

    # Step 5: Output the Minimal Set
    print("\n--- ELIMINATION COMPLETE ---")
    print(f"Rules safely pruned: {len(permanently_disabled)}")
    for rule in permanently_disabled:
        print(f"  - {rule}")

    print("\nMinimal Rule Set required for ufal/atrium-alto-postprocess integrity:")
    for rule in sorted(active_rules):
        print(f"  + {rule}")


def main():
    # Setup LLM Model Here (Ensure 4-bit quantization config is passed to the text_inference loader)
    # quant_config = BitsAndBytesConfig(load_in_4bit=True)
    # ...

    print("Loading cached datasets...")
    df = load_csvs(Path("../data_samples/DOC_LINE_CATEG"), recursive=True)

    expected_langs, known_bases = _load_lang_config("../config_langID.txt")
    eval_kwargs = {"expected_langs": expected_langs, "known_bases": known_bases}
    base_constants = read_config_constants("../config_langID.txt")

    run_backward_elimination(df, eval_kwargs, base_constants)


if __name__ == "__main__":
    main()
