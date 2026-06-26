#!/usr/bin/env python3
"""
tools/rule_coverage_report.py
==============================
Analyzes rule-fire coverage (Increment B5) to establish which structural rules
and per-line penalties in the categorisation engine are:

  DEAD            — fire_count == 0 across all supplied documents. The rule's
                    action never executes; it is unreachable dead code and can
                    be permanently deleted without a gold label set, because
                    deletion provably changes nothing.

  REDUNDANT-HERE  — fire_count > 0 but decisive_count == 0. The rule fires
                    but is currently masked by an overlapping rule that catches
                    the same line first (entanglement).  Keep it: the masking
                    order may change with corpus or config, so the rule is a
                    real guard that just appears redundant on this sample.

  LOAD-BEARING    — decisive_count > 0. Removing the rule changes at least one
                    line's category vs. the frozen ground truth. Always keep.

Coverage columns
----------------
  fire_count      Raw execution count within rule_fire_capture().
  fire_rate       fire_count / n_scored_lines (excludes Empty / Non-text
                  fast-track rows that never pass through the scorer).
  decisive_count  LOO: lines whose final category changes when the rule is
                  disabled via DISABLED_RULES, measured against the stored
                  categ (flip_rate × n_lines).
  clear_loss      LOO: lines that were Clear in the stored categ but become
                  Trash or Non-text when the rule is removed — the most
                  operationally expensive failure mode.
  class           Derived classification: DEAD / REDUNDANT-HERE / LOAD-BEARING.

Usage
-----
  # Directory of per-document CSVs
  python tools/rule_coverage_report.py --input-dir data_samples/DOC_LINE_CATEG

  # Single CSV file
  python tools/rule_coverage_report.py data_samples/DOC_LINE_CATEG/some_doc.csv

  # With custom config and JSON output
  python tools/rule_coverage_report.py \\
      --input-dir data_samples/DOC_LINE_CATEG \\
      --config config_langID.txt \\
      --output rule_coverage.json

Exit codes
----------
  0  No dead rules found (or run completed normally).
  1  One or more dead rules detected; list printed to stdout.
  2  Bad arguments / missing path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from text_util_langID import override_constants, rule_fire_capture  # noqa: E402
from tools.recategorize_from_csv import (  # noqa: E402
    _load_lang_config,
    evaluate_dataframe,
    load_csvs,
    recategorize_dataframe,
)

# ---------------------------------------------------------------------------
# Canonical rule / penalty registry
# Keep in sync with _fire() call-sites in text_util_langID.py.
# ---------------------------------------------------------------------------
RULES: list[str] = sorted(
    [
        # determine_category rules
        "rule_hard_sweep",
        "rule_extreme_ppl",
        "rule_absolute_ppl",
        "rule_inverted",
        "rule_allcaps",
        "rule_garbage_density",
        "rule_trailing_fill_rescue",
        "rule_short_garbage",
        "rule_lowppl_clear",
        "rule_mostly_readable_noisy",
        # categorize_line penalties
        "penalty_wqx_rot",
        "penalty_vowelless",
        "penalty_ledger_fragmentation",
        "penalty_mid_uppercase",
    ]
)

_DETERMINE_RULES = [r for r in RULES if r.startswith("rule_")]
_PENALTY_RULES = [r for r in RULES if r.startswith("penalty_")]

# Columns widths for terminal output
_W_NAME = 34
_W_COUNT = 10
_W_RATE = 10
_W_DEC = 10
_W_LOSS = 10
_W_CLASS = 17


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_dataframe(raw_path: str) -> tuple[pd.DataFrame, Path]:
    """Load a single CSV or a directory of CSVs into one DataFrame."""
    in_path = Path(raw_path)
    if in_path.is_dir():
        df = load_csvs(in_path)
    elif in_path.is_file():
        df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
        df["_source_file"] = str(in_path.name)
        df["file"] = in_path.stem
    else:
        raise FileNotFoundError(f"Path not found: {in_path}")
    return df, in_path


def _n_scored(df: pd.DataFrame) -> int:
    """Number of lines that pass through the scorer (excludes fast-track rows)."""
    if "categ" not in df.columns:
        return len(df)
    fast_track = df["categ"].isin(("Empty", "Non-text"))
    try:
        wc = pd.to_numeric(df.get("word_count", pd.Series(dtype=float)), errors="coerce").fillna(1)
        fast_track = fast_track & (wc == 0)
    except Exception:
        pass
    return int((~fast_track).sum())


# ---------------------------------------------------------------------------
# LOO decisive count
# ---------------------------------------------------------------------------


def _loo_metrics(
    df: pd.DataFrame,
    rule: str,
    expected_langs: list[str],
    known_bases: frozenset,
) -> tuple[int, int]:
    """Return (decisive_count, clear_loss) for a single LOO disable of *rule*.

    decisive_count — lines whose category changes vs. the stored categ when
                     this rule is removed (flip_count from evaluate_dataframe).
    clear_loss     — among those flips, how many go Clear → Trash / Non-text.
    """
    with override_constants({"DISABLED_RULES": frozenset([rule])}):
        metrics = evaluate_dataframe(
            df,
            constants=None,
            expected_langs=expected_langs,
            known_bases=known_bases,
        )

    decisive_count = int(metrics["flip_count"])
    clear_row = metrics.get("confusion", {}).get("Clear", {})
    clear_loss = int(clear_row.get("Trash", 0)) + int(clear_row.get("Non-text", 0))
    return decisive_count, clear_loss


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify(fire_count: int, decisive_count: int) -> str:
    if fire_count == 0:
        return "DEAD"
    if decisive_count == 0:
        return "REDUNDANT-HERE"
    return "LOAD-BEARING"


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------


def run_coverage(
    raw_path: str,
    config_path: str | None = None,
    output_path: str | None = None,
    quiet: bool = False,
    skip_loo: bool = False,
) -> dict[str, dict]:
    """Run coverage instrumentation + optional LOO analysis over *raw_path*.

    Parameters
    ----------
    raw_path:    Path to a CSV file or a directory of per-document CSVs.
    config_path: Optional path to config_langID.txt INI.
    output_path: If given, write ``rule_coverage.json`` to this path.
    quiet:       Suppress the per-rule table.
    skip_loo:    Skip the LOO decisive-count pass (faster; coverage only).

    Returns
    -------
    dict mapping rule name → {fire_count, fire_rate, decisive_count,
                               clear_loss, class}.
    """
    df, in_path = _load_dataframe(raw_path)
    resolved_config = config_path or str(_ROOT / "config_langID.txt")
    expected_langs, known_bases = _load_lang_config(resolved_config)

    n_total = len(df)
    n_scored = _n_scored(df)
    print(f"Loaded {n_total:,} lines ({n_scored:,} scored) from {in_path}")

    # ------------------------------------------------------------------
    # Phase 1: fire-count capture
    # ------------------------------------------------------------------
    print("Phase 1 — fire-count pass …")
    with rule_fire_capture() as raw_counts:
        recategorize_dataframe(df, expected_langs=expected_langs, known_bases=known_bases)

    # ------------------------------------------------------------------
    # Phase 2: LOO decisive count (one recategorize pass per rule)
    # ------------------------------------------------------------------
    loo: dict[str, tuple[int, int]] = {}
    if skip_loo:
        print("Phase 2 — LOO skipped (--skip-loo).")
        for rule in RULES:
            loo[rule] = (0, 0)
    else:
        print(f"Phase 2 — LOO pass ({len(RULES)} rules × 1 recategorize each) …")
        for i, rule in enumerate(RULES, 1):
            decisive, closs = _loo_metrics(df, rule, expected_langs, known_bases)
            loo[rule] = (decisive, closs)
            print(f"  [{i:>2}/{len(RULES)}] {rule:<34} decisive={decisive}  clear_loss={closs}")

    # ------------------------------------------------------------------
    # Assemble result dict
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}
    for rule in RULES:
        fc = raw_counts.get(rule, 0)
        fr = fc / n_scored if n_scored > 0 else 0.0
        decisive, closs = loo[rule]
        cls = _classify(fc, decisive)
        results[rule] = {
            "fire_count": fc,
            "fire_rate": round(fr, 6),
            "decisive_count": decisive,
            "clear_loss": closs,
            "class": cls,
        }

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    if not quiet:
        _print_table(results, n_scored)

    _print_summary(results)

    if output_path:
        payload = {
            "input": str(in_path),
            "n_lines": n_total,
            "n_scored": n_scored,
            "rules": results,
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON written → {out}")

    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _print_table(results: dict[str, dict], n_scored: int) -> None:
    sep = "-" * (_W_NAME + _W_COUNT + _W_RATE + _W_DEC + _W_LOSS + _W_CLASS + 15)
    hdr = (
        f"  {'Rule / Penalty':<{_W_NAME}}"
        f" | {'fire_count':>{_W_COUNT}}"
        f" | {'fire_rate':>{_W_RATE}}"
        f" | {'decisive':>{_W_DEC}}"
        f" | {'clr_loss':>{_W_LOSS}}"
        f" | {'class':<{_W_CLASS}}"
    )
    print(f"\n=== Rule Coverage Report (n_scored={n_scored:,}) ===")
    print(hdr)
    print(sep)

    for section_label, section_rules in [
        ("— determine_category —", _DETERMINE_RULES),
        ("— categorize_line penalties —", _PENALTY_RULES),
    ]:
        print(f"\n  {section_label}")
        for rule in section_rules:
            r = results[rule]
            dead_flag = "  ← DEAD" if r["class"] == "DEAD" else ""
            print(
                f"  {rule:<{_W_NAME}}"
                f" | {r['fire_count']:>{_W_COUNT}}"
                f" | {r['fire_rate']:>{_W_RATE}.4f}"
                f" | {r['decisive_count']:>{_W_DEC}}"
                f" | {r['clear_loss']:>{_W_LOSS}}"
                f" | {r['class']:<{_W_CLASS}}{dead_flag}"
            )
    print()


def _print_summary(results: dict[str, dict]) -> None:
    dead = [r for r, v in results.items() if v["class"] == "DEAD"]
    redund = [r for r, v in results.items() if v["class"] == "REDUNDANT-HERE"]
    bearing = [r for r, v in results.items() if v["class"] == "LOAD-BEARING"]

    print(f"Summary: {len(bearing)} LOAD-BEARING  |  {len(redund)} REDUNDANT-HERE  |  {len(dead)} DEAD")

    if dead:
        print("\nDEAD rules (fire_count == 0 — safe to retire after full-corpus confirmation):")
        for r in dead:
            print(f"  - {r}")
        print(
            "\n  ⚠  A rule dead on the smoke fixture may still fire on unseen documents.\n"
            "     Run on the full corpus on the cluster before deleting. See\n"
            "     tools/RULE_COVERAGE.md for the retirement criterion."
        )
    else:
        print("\nAll rules fired at least once — no dead code detected on this dataset.")

    if redund:
        print("\nREDUNDANT-HERE rules (fire_count > 0, decisive_count == 0 — keep; entanglement suspected):")
        for r in redund:
            print(f"  - {r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="rule_coverage_report.py",
        description=(
            "Rule-fire coverage + LOO decisive-count report (B5). "
            "Classifies each rule as DEAD / REDUNDANT-HERE / LOAD-BEARING."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "path",
        nargs="?",
        metavar="PATH",
        help="CSV file or directory of per-document CSVs.",
    )
    ap.add_argument(
        "--input-dir", dest="input_dir", metavar="DIR", help="Alias for the positional PATH (directory form)."
    )
    ap.add_argument("--config", metavar="FILE", help="config_langID.txt-style INI.  Default: <repo>/config_langID.txt.")
    ap.add_argument(
        "--output", metavar="JSON_FILE", help="Write full results to this JSON file (e.g. rule_coverage.json)."
    )
    ap.add_argument(
        "--skip-loo", action="store_true", help="Skip the LOO decisive-count pass; report fire counts only."
    )
    ap.add_argument("--quiet", "-q", action="store_true", help="Suppress the per-rule table; only print the summary.")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raw_path = args.path or args.input_dir
    if not raw_path:
        print(
            "error: provide a path to a CSV file or a directory via the positional argument or --input-dir.",
            file=sys.stderr,
        )
        return 2

    try:
        results = run_coverage(
            raw_path=raw_path,
            config_path=args.config,
            output_path=args.output,
            quiet=args.quiet,
            skip_loo=args.skip_loo,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    dead_rules = [r for r, v in results.items() if v["class"] == "DEAD"]
    return 1 if dead_rules else 0


if __name__ == "__main__":
    sys.exit(main())
