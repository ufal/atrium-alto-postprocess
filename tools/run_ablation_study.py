#!/usr/bin/env python3
"""
tools/run_ablation_study.py
===========================
Executes a Leave-One-Out (LOO) ablation sweep over the ALTO post-processing rules.

This module provides a rigorous, automated feature selection pipeline for the
heuristic text classification engine. By systematically disabling individual
rules and evaluating the model against a cached dataset, it calculates:

1. Feature Variance: Identifies continuous quality score weights that have zero
   signal variance in the dataset (i.e., the artifact they detect never occurs).
2. Marginal Flips (LOO): The total number of document lines that change their
   final categorization when a specific rule is removed.
3. Destructive Cost: A strict asymmetric penalty applied when disabling a rule
   causes genuine text to be falsely categorized as garbage (`Clear -> Trash`).

Output is rendered as a standard Markdown matrix summarizing keep/prune decisions
for pull request reviews and architectural documentation.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Point Python to the root directory to access text_util_langID and sibling scripts
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

from recategorize_from_csv import _load_lang_config, evaluate_dataframe, load_csvs, read_config_constants  # noqa: E402

from text_util_langID import QS_WEIGHT_NAMES, override_constants  # noqa: E402

# The definitive list of binary gating rules and penalties to ablate
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
    """
    Provides an automated architectural recommendation based on strict asymmetric costing.

    Args:
        flips: Total number of categorization changes caused by the ablation.
        destructive_cost: Number of lines falsely flipped from Clear to Trash.
        std_dev: Standard deviation of the feature's signal across the dataset.

    Returns:
        A Markdown-formatted string containing the decision and its justification.
    """
    if std_dev is not None and std_dev < 0.005:
        return "**PRUNE** (Signal variance ≈ 0; feature absent from dataset)"
    if flips == 0:
        return "**PRUNE** (Fully redundant; zero marginal effect)"
    if destructive_cost > 0:
        return "**KEEP** (Critical safeguard: Destructive to valid text if removed)"
    if flips > 100:
        return "**KEEP** (High Impact routing rule)"
    return "**REVIEW** (Low Impact; consider pruning if complexity is high)"


def run_ablation(df: pd.DataFrame, eval_kwargs: Dict[str, Any], base_constants: Dict[str, Any]) -> None:
    """
    Executes the Leave-One-Out ablation study and prints the markdown report matrix.

    Args:
        df: The cached dataset containing pre-calculated text features.
        eval_kwargs: Keyword arguments for the evaluation function (e.g., known languages).
        base_constants: The baseline configuration parameters parsed from config_langID.txt.
    """
    print(f"Evaluating Baseline across {len(df):,} document lines...")
    start_time = time.time()
    base_metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)
    base_lines = len(df)
    print(f"Baseline established in {time.time() - start_time:.2f} seconds.")

    print("\nCalculating Feature Variance for Continuous Signals...")
    # Map QS_WEIGHT configuration constants to their corresponding dataframe columns
    feature_map: Dict[str, str] = {
        "QS_WEIGHT_VOWEL": "vowel_ratio",
        "QS_WEIGHT_WEIRD": "weird_ratio",
        "QS_WEIGHT_GIBBERISH": "gibberish_ratio",
        "QS_WEIGHT_FUSED": "fused_ratio",
        "QS_WEIGHT_GARBAGE": "garbage_density",
    }

    report_rows: List[Tuple[str, str, int, int, str]] = []

    # Phase 1: Ablate Continuous Quality Score (QS) Weights
    for weight_name in QS_WEIGHT_NAMES:
        col = feature_map.get(weight_name)
        std_dev = df[col].std() if col and col in df.columns else None

        # Override the specific weight to 0.0, allowing auto-renormalization to handle the rest
        with override_constants({weight_name: 0.0}):
            metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

        flips = int(metrics["flip_rate"] * base_lines)

        # Strict Asymmetric Cost: Calculate if valid text was destroyed (Clear rate drops, Trash rate rises)
        clear_drop = max(0.0, base_metrics["clear_rate"] - metrics["clear_rate"])
        trash_rise = max(0.0, metrics["trash_rate"] - base_metrics["trash_rate"])
        destructive_cost = int(min(clear_drop, trash_rise) * base_lines)

        decision = format_decision(flips, destructive_cost, std_dev)
        signal_type = f"Continuous (σ={std_dev:.3f})" if std_dev is not None else "Continuous"
        report_rows.append((weight_name, signal_type, flips, destructive_cost, decision))

    # Phase 2: Ablate Discrete Binary Gateway Rules
    for rule in RULES_TO_ABLATE:
        # Utilize the global DISABLED_RULES frozen set to surgically kill logic branches
        with override_constants({"DISABLED_RULES": frozenset([rule])}):
            metrics = evaluate_dataframe(df, base_constants, **eval_kwargs)

        flips = int(metrics["flip_rate"] * base_lines)

        clear_drop = max(0.0, base_metrics["clear_rate"] - metrics["clear_rate"])
        trash_rise = max(0.0, metrics["trash_rate"] - base_metrics["trash_rate"])
        destructive_cost = int(min(clear_drop, trash_rise) * base_lines)

        decision = format_decision(flips, destructive_cost)
        report_rows.append((rule, "Binary Gateway", flips, destructive_cost, decision))

    # Phase 3: Render Matrix Output for GitHub / PR Reviews
    print("\n### System Ablation Study Results")
    print(f"*Total Evaluation Corpus: {base_lines:,} lines*")
    print("\n| Rule / Factor | Signal Type | Marginal Flips (LOO) | `Clear -> Trash` Cost | Decision |")
    print("| --- | --- | --- | --- | --- |")

    # Sort the matrix strategically: Critical safeguards first, then highest impact, pruning candidates last.
    sorted_rows = sorted(report_rows, key=lambda x: (x[3] > 0, x[2]), reverse=True)
    for row in sorted_rows:
        print(f"| `{row[0]}` | {row[1]} | {row[2]:,} | {row[3]:,} | {row[4]} |")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LOO Ablation on heuristics to determine structural redundancy and rule overlap."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the directory containing cached DOC_LINE_CATEG CSV datasets.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_langID.txt",
        help="Path to the system configuration file (default: config_langID.txt).",
    )
    args = parser.parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        print(f"Error: Input directory {args.input_dir} does not exist or is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading cached dataset from {args.input_dir}...")
    try:
        df = load_csvs(args.input_dir, recursive=True)
    except Exception as e:
        print(f"Failed to load datasets: {e}", file=sys.stderr)
        sys.exit(1)

    expected_langs, known_bases = _load_lang_config(args.config)
    eval_kwargs = {"expected_langs": expected_langs, "known_bases": known_bases}

    try:
        base_constants = read_config_constants(args.config)
    except Exception as e:
        print(f"Failed to read configuration constants: {e}", file=sys.stderr)
        sys.exit(1)

    run_ablation(df, eval_kwargs, base_constants)


if __name__ == "__main__":
    main()
