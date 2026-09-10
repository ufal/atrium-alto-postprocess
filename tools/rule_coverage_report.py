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
      --config setup/config.txt \\
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

from text_util import override_constants, rule_fire_capture  # noqa: E402
from tools.recategorize_from_csv import (  # noqa: E402
    _load_lang_config,
    add_gold_column_argument,
    attach_gold_sidecar_from_args,
    coerce_constants,
    evaluate_dataframe,
    load_csvs,
    read_config_constants,
    recategorize_dataframe,
    validate_constants,
)

# ---------------------------------------------------------------------------
# Canonical rule / penalty registry
#
# Must match the _fire() call-sites in text_util.py exactly. That used to be a
# comment asking for manual upkeep, and it drifted: the five rules added by the
# issue #30 work (rule_short_line, rule_damaged_token, rule_reference_floor,
# rule_bigram_run, rule_fragment_tokens) were missing, so every coverage report
# produced after PR #32 silently omitted them. It is now enforced by
# tests/test_rule_coverage.py::test_rules_registry_matches_fire_call_sites.
# ---------------------------------------------------------------------------
RULES: list[str] = sorted(
    [
        "rule_hard_sweep",
        "rule_extreme_ppl",
        "rule_absolute_ppl",
        "rule_inverted",
        "rule_allcaps",
        "rule_garbage_density",
        "rule_trailing_fill_rescue",
        "rule_short_garbage",
        "rule_short_garbage_witness",
        "rule_domain_notation",
        "rule_short_line",
        "rule_zero_alpha",
        "rule_lowppl_clear",
        "rule_mostly_readable_noisy",
        "rule_damaged_token",
        "rule_reference_floor",
        "rule_wqx_rot",
        "rule_vowelless",
        "rule_ledger_fragmentation",
        "rule_mid_uppercase",
        "rule_bigram_run",
        "rule_fragment_tokens",
        "rule_forgiven_headline",
    ]
)

# There used to be a _DETERMINE_RULES / _PENALTY_RULES split here, partitioning
# RULES on a "penalty_" prefix. Both were dead code and one was a lie: the
# penalties were renamed to rule_* long ago, so _PENALTY_RULES had been the
# empty list ever since, and _DETERMINE_RULES was just RULES again under another
# name. The same stale prefix survived in the two ablation drivers, where it did
# real damage -- see the comment above run_ablation_study.RULES_TO_ABLATE.

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


def _load_dataframe(raw_path: str, gold_args=None) -> tuple[pd.DataFrame, Path]:
    """Load a single CSV or a directory of CSVs into one DataFrame.

    ``gold_args`` carries the gold flags (an argparse namespace, or None). This is step 1 of ``run_optim_pipeline.sh``
    and it was the one loader in the repository with no sidecar join, so a gold
    run would have reported coverage over an unannotated frame while every later
    stage scored against gold.
    """
    in_path = Path(raw_path)
    if in_path.is_dir():
        df = load_csvs(in_path)
    elif in_path.is_file():
        df = pd.read_csv(in_path, dtype=str, keep_default_na=False)
        df["_source_file"] = str(in_path.name)
        df["file"] = in_path.stem
    else:
        raise FileNotFoundError(f"Path not found: {in_path}")
    if gold_args is not None:
        df = attach_gold_sidecar_from_args(df, gold_args)
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
    constants: dict | None = None,
) -> tuple[int, int]:
    """Return (decisive_count, clear_loss) for a single LOO disable of *rule*.

    decisive_count — lines whose category changes vs. the stored categ when
                     this rule is removed (flip_count from evaluate_dataframe).
    clear_loss     — among those flips, how many go Clear → Trash / Non-text.

    ``constants`` is the resolved config for the run. Passing it matters: a rule
    is only DEAD or LOAD-BEARING *relative to a configuration*, and measuring
    that under the import-time defaults while the caller asked for another
    config answers a question nobody posed.
    """
    with override_constants({"DISABLED_RULES": frozenset([rule])}):
        metrics = evaluate_dataframe(
            df,
            constants=constants,
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
# Word-count breakdown (issue #30)
# ---------------------------------------------------------------------------

WC_BUCKETS = ("1", "2", "3", "4", "5+")


def _wc_bucket(wc: int) -> str:
    if wc <= 0:
        return "1"
    return str(wc) if wc <= 4 else "5+"


def run_wc_breakdown(
    raw_path: str,
    config_path: str | None = None,
    quiet: bool = False,
    gold_args=None,
) -> dict[str, dict[str, int]]:
    """Attribute every rule fire to the word count of the line that produced it.

    Why this exists
    ---------------
    @david-spacil observed in issue #30 that "59.4% of hard-sweep-family firings
    land exactly on `wc == 3`" and noted it was "not measured further". The
    observation matters because the short-line regimes are structurally
    different, not merely shorter: ``SHORT_PPL_CAP`` covers ``wc <= 2`` only, so
    at three tokens raw perplexity reaches the sweep for the first time. That,
    rather than "shorter is riskier", is what explains the error gradient the
    thread argued about. Nothing in the repository could reproduce the figure,
    so it stayed an anecdote.

    Method, and its one caveat
    --------------------------
    Each line is scored individually inside its own ``rule_fire_capture()``
    block, so a fire can be attributed to the line that caused it. This is the
    PER-LINE decision only: ``apply_document_postprocessing`` is not run, because
    page-level smoothing has no single owning line. Document post-processing
    changes the category of a substantial share of lines, so the categories here
    are not the pipeline's final answer -- the *fires* are, and those are what
    this reports.
    """
    df, in_path = _load_dataframe(raw_path, gold_args)
    resolved_config = config_path or str(_ROOT / "setup" / "config.txt")
    expected_langs, known_bases = _load_lang_config(resolved_config)
    constants = coerce_constants(read_config_constants(resolved_config))
    validate_constants(constants)

    from tools.recategorize_from_csv import _is_fast_track, _rescore_row  # noqa: PLC0415

    counts: dict[str, dict[str, int]] = {r: dict.fromkeys(WC_BUCKETS, 0) for r in RULES}
    lines_by_bucket: dict[str, int] = dict.fromkeys(WC_BUCKETS, 0)
    scored = 0

    with override_constants(constants):
        for _idx, row in df.iterrows():
            rd = row.to_dict()
            if _is_fast_track(rd):
                continue
            text = str(rd.get("text", "") or "")
            bucket = _wc_bucket(len(text.split()))
            lines_by_bucket[bucket] += 1
            scored += 1
            with rule_fire_capture() as fired:
                _rescore_row(rd, expected_langs, known_bases)
            for name in fired:
                if name in counts:
                    counts[name][bucket] += 1

    if not quiet:
        print(f"\n=== rule fires by word count (per-line; n_scored={scored:,}) ===")
        print(f"  {'rule':<34} " + " ".join(f"{b:>7}" for b in WC_BUCKETS) + f" {'total':>8}  {'peak':>6}")
        print("  " + "-" * 34 + "-" * (8 * len(WC_BUCKETS) + 18))
        for rule in RULES:
            row_counts = counts[rule]
            total = sum(row_counts.values())
            if not total:
                continue
            peak_bucket = max(WC_BUCKETS, key=lambda b: row_counts[b])
            peak_share = row_counts[peak_bucket] / total
            print(
                f"  {rule:<34} "
                + " ".join(f"{row_counts[b]:>7,}" for b in WC_BUCKETS)
                + f" {total:>8,}  {peak_bucket:>3} {peak_share:>5.0%}"
            )
        print("  " + "-" * 34 + "-" * (8 * len(WC_BUCKETS) + 18))
        print(f"  {'lines in bucket':<34} " + " ".join(f"{lines_by_bucket[b]:>7,}" for b in WC_BUCKETS))
        print(
            "\n  NOTE: per-line fires only -- document post-processing is not applied here,\n"
            "  so the categories these fires lead to are not the pipeline's final answer."
        )

    return counts


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------


def run_coverage(
    raw_path: str,
    config_path: str | None = None,
    output_path: str | None = None,
    quiet: bool = False,
    skip_loo: bool = False,
    gold_args=None,
) -> dict[str, dict]:
    """Run coverage instrumentation + optional LOO analysis over *raw_path*.

    Parameters
    ----------
    raw_path:    Path to a CSV file or a directory of per-document CSVs.
    config_path: Optional path to config.txt INI.
    output_path: If given, write ``rule_coverage.json`` to this path.
    quiet:       Suppress the per-rule table.
    skip_loo:    Skip the LOO decisive-count pass (faster; coverage only).

    Returns
    -------
    dict mapping rule name → {fire_count, fire_rate, decisive_count,
                               clear_loss, class}.
    """
    df, in_path = _load_dataframe(raw_path, gold_args)
    resolved_config = config_path or str(_ROOT / "setup" / "config.txt")
    expected_langs, known_bases = _load_lang_config(resolved_config)

    # `--config` used to feed ONLY the language lists: this function never read
    # the file's constants, so every threshold came from whatever text_util
    # imported at start-up. Two runs with different `--config` files produced
    # byte-identical reports, and an INVALID config (one that makes
    # recategorize_from_csv raise) produced a clean table. Since this report is
    # what classifies rules DEAD / LOAD-BEARING and gates retirement decisions
    # in RULE_COVERAGE.md, it has to measure the configuration it was handed.
    constants = coerce_constants(read_config_constants(resolved_config))
    validate_constants(constants)

    n_total = len(df)
    n_scored = _n_scored(df)
    print(f"Loaded {n_total:,} lines ({n_scored:,} scored) from {in_path}")
    print(f"Config: {resolved_config}")

    # ------------------------------------------------------------------
    # Phase 1: fire-count capture
    # ------------------------------------------------------------------
    print("Phase 1 — fire-count pass …")
    with rule_fire_capture() as raw_counts:
        recategorize_dataframe(df, constants, expected_langs=expected_langs, known_bases=known_bases)

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
            decisive, closs = _loo_metrics(df, rule, expected_langs, known_bases, constants)
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
        ("— determine_category rules —", RULES),
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
    ap.add_argument("--config", metavar="FILE", help="config.txt-style INI.  Default: <repo>/config.txt.")
    ap.add_argument(
        "--output", metavar="JSON_FILE", help="Write full results to this JSON file (e.g. rule_coverage.json)."
    )
    ap.add_argument(
        "--skip-loo", action="store_true", help="Skip the LOO decisive-count pass; report fire counts only."
    )
    ap.add_argument("--quiet", "-q", action="store_true", help="Suppress the per-rule table; only print the summary.")
    ap.add_argument(
        "--by-wc",
        action="store_true",
        help=(
            "Instead of the coverage table, attribute every rule fire to the word count of the "
            "line that caused it (issue #30). Per-line only: document post-processing is not applied."
        ),
    )
    add_gold_column_argument(ap)
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

    if args.by_wc:
        try:
            counts = run_wc_breakdown(raw_path=raw_path, config_path=args.config, quiet=args.quiet, gold_args=args)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.output:
            Path(args.output).write_text(json.dumps(counts, indent=2), encoding="utf-8")
            print(f"\nJSON written → {args.output}")
        return 0

    try:
        results = run_coverage(
            raw_path=raw_path,
            config_path=args.config,
            output_path=args.output,
            quiet=args.quiet,
            skip_loo=args.skip_loo,
            gold_args=args,
        )
    except (FileNotFoundError, ValueError) as exc:
        # ValueError is the gold-sidecar join refusing a zero-match or a
        # malformed sidecar. An operator running this as stage 1 of
        # run_optim_pipeline.sh needs the reason, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    dead_rules = [r for r, v in results.items() if v["class"] == "DEAD"]
    return 1 if dead_rules else 0


if __name__ == "__main__":
    sys.exit(main())
