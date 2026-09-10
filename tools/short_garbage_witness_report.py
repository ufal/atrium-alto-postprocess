#!/usr/bin/env python3
"""
tools/short_garbage_witness_report.py — measure `_has_shape_garbage_evidence()`
against real delivered `DOC_LINE_CATEG` CSVs (issue #30).

WHY THIS EXISTS
---------------
`_has_shape_garbage_evidence()` ships behind `SHORT_GARBAGE_WITNESS_ENABLE`
(default false). Since PR #48 merged it IS wired -- read by the short-line
garbage gate as a second disjunct -- but the flag is still off, so it cannot
change any outcome until someone turns it on. Two consequences, both verified:

  * turning the flag on changes **no category** — a full
    `tools/recategorize_from_csv.py` run over `data_samples/DOC_LINE_CATEG`
    reports `total lines changed category: 0` with the flag off *and* on;
  * the four `SHORT_GARBAGE_WITNESS_*` constants are deliberately absent from
    `TUNABLE_CONSTANTS` (see `_DELIBERATELY_NOT_TUNABLE` in
    `tests/test_recategorize_parity.py`), so `--override` rejects them.

So the predicate cannot be exercised end-to-end yet, and the in-tree sample
corpus could not validate it anyway: of its 15 lines, exactly 2 reach the
route's text-only entry condition, both already `Trash`, with no rare-vocabulary
line among them to expose a false positive.

This tool closes that gap for the part that does not need a call site. It
answers, over any collection you already have on disk: **which lines would the
witness reach, and what does the pipeline currently call them?** That is the
measurement the flag is gated on ("do not turn this on before measuring it
against annotated lines"), and it needs no GPU, no FastText and no re-scoring.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does **not** re-score, and it does **not** reconstruct any signal. Every
column it computes is a pure function of the line's text: the three vetoes
(`has_cz_diacs`, `is_structured_line`, `is_domain_notation`), the witness, and
which of the witness's four clauses fired. The signal-dependent terms of the
route's condition — `lang_score`, `rot_ratio`, `gibberish_present`,
`weird_ratio` — are **not** re-derived here, because the stored CSV columns are
not the values the rules see (`lang_score` in the CSV is the `remap_lang` cap;
the rules read the two-tier trust score) and reconstructing them is exactly the
harness bug already fixed twice in this repo, in
`tests/test_rotation_regression.py` and `tests/test_calibration.py::_categ`.
Re-scoring is `tools/recategorize_from_csv.py`'s job and it reuses
`classify_TEXT.score_line`; this tool stays on the side of the line where text
is all it needs.

Because of that, the population it reports is an **upper bound** on the lines
the route acts on: every line it counts satisfies the text-only half of the
entry condition (`word_count <= ISOLATED_CHAR_MIN_TOKENS`, no Czech diacritics,
not structured, not notation), but some of them fail the signal half and never
reach the rule. Read the counts as exposure, not as an effect size.

The `categ` column is the **current pipeline's** answer, not ground truth. A
witnessed line sitting at `Clear` is a false-positive *candidate*; whether it is
actually wrong is an annotation question, which is what `--out` is for.

USAGE
-----
    # exposure over a delivered collection
    python3 tools/short_garbage_witness_report.py --input-dir /path/to/DOC_LINE_CATEG

    # a single document, with examples of every clause
    python3 tools/short_garbage_witness_report.py DOC_LINE_CATEG/CTX200205348.csv --examples 8

    # write the candidate lines out for blind annotation
    python3 tools/short_garbage_witness_report.py --input-dir DOC_LINE_CATEG \
        --out /tmp/witness_candidates.csv

    # ad-hoc: one line per row of a plain text file (no CSV schema needed)
    python3 tools/short_garbage_witness_report.py --lines /tmp/probe.txt

Stdlib only (`csv`, not pandas), so it runs anywhere `text_util` imports.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import text_util as tu  # noqa: E402

# The witness's four clauses, re-expressed for DIAGNOSIS ONLY so the report can
# say which one fired. `_clauses_for_token` mirrors the structure of
# `_has_shape_garbage_evidence`; `classify_line` then asserts the two agree, so
# this cannot silently drift from the predicate it describes. If that assertion
# ever fires, the predicate changed and this list is stale — fix this, not it.
_CLAUSE_ORDER = ("vowel_run", "triple", "initial_geminate", "low_variety")


def _clauses_for_token(core: str) -> list[str]:
    """Which witness clauses a single sub-token satisfies (diagnosis only)."""
    letters = [c for c in core if c.isalpha()]
    if len(letters) < tu.SHORT_GARBAGE_WITNESS_MIN_ALPHA:
        return []

    lowered = core.lower()
    if lowered in tu._NEUTRAL_LEXICON or lowered in tu.SHORT_EXCEPTION_TOKENS or lowered in tu.SHORT_VALID_WORDS:
        return []

    hits: list[str] = []
    if tu._RE_FUSED_VOWEL_RUN.search(core):
        hits.append("vowel_run")
    if len(letters) <= tu.SHORT_GARBAGE_WITNESS_TRIPLE_MAX_ALPHA and tu._RE_TRIPLE_ALPHA_RUN.search(core):
        hits.append("triple")
    if tu._RE_INITIAL_CONSONANT_GEMINATE.match(core):
        hits.append("initial_geminate")
    if len(letters) >= tu.SHORT_GARBAGE_WITNESS_VARIETY_MIN_ALPHA and (
        len({c.lower() for c in letters}) / len(letters) <= tu.SHORT_GARBAGE_WITNESS_VARIETY_MAX
    ):
        hits.append("low_variety")
    return hits


def clauses_for_line(text: str) -> list[str]:
    """Union of the clauses fired across a line's sub-tokens, in fixed order."""
    if tu.has_cz_diacs(text) or tu.is_structured_line(text) or tu.is_domain_notation(text):
        return []
    found: set[str] = set()
    for word in text.split():
        for sub in tu._split_subtokens(word):
            found.update(_clauses_for_token(sub.strip(tu._STRIP_CHARS)))
    return [c for c in _CLAUSE_ORDER if c in found]


def classify_line(text: str, word_count: int | None = None) -> dict:
    """Text-only verdicts for one line. No scoring, no signal reconstruction."""
    wc = len(text.split()) if word_count is None else word_count
    diacs = tu.has_cz_diacs(text)
    structured = tu.is_structured_line(text)
    notation = tu.is_domain_notation(text)
    witness = tu._has_shape_garbage_evidence(text)
    clauses = clauses_for_line(text)

    # Guard against this module's diagnosis drifting from the predicate.
    assert bool(clauses) == witness, (
        f"clause diagnosis disagrees with _has_shape_garbage_evidence() on {text!r}: "
        f"clauses={clauses} witness={witness}. The predicate changed; update _clauses_for_token."
    )

    return {
        "text": text,
        "word_count": wc,
        "has_cz_diacs": diacs,
        "structured": structured,
        "notation": notation,
        # The text-only half of rule_short_garbage's entry condition. An upper
        # bound on the population: the signal half is not evaluated here.
        "route_eligible": bool(wc) and wc <= tu.ISOLATED_CHAR_MIN_TOKENS and not (diacs or structured or notation),
        "witness": witness,
        "clauses": ",".join(clauses),
    }


def _iter_csv_rows(paths: list[Path]):
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                raw_wc = (row.get("word_count") or "").strip()
                try:
                    wc = int(raw_wc) if raw_wc else None
                except ValueError:
                    wc = None
                yield path.name, text, wc, (row.get("categ") or "").strip() or "?"


def _iter_plain_lines(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            yield path.name, text, None, "?"


def _collect_csvs(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.glob("*.csv") if p.is_file())
    return [path]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report which lines _has_shape_garbage_evidence() would reach, against the category "
            "the pipeline currently assigns. Does not re-score and does not reconstruct signals."
        )
    )
    parser.add_argument("path", nargs="?", help="DOC_LINE_CATEG CSV file or directory of them")
    parser.add_argument("--input-dir", help="Alias for the positional path (a directory).")
    parser.add_argument("--lines", help="Plain text file, one candidate line per row (no CSV schema).")
    parser.add_argument("--out", help="Write the witnessed candidate lines to this CSV, for annotation.")
    parser.add_argument("--examples", type=int, default=0, metavar="N", help="Print up to N examples per clause.")
    parser.add_argument(
        "--all-lengths",
        action="store_true",
        help="Report every line, not only those meeting the route's text-only entry condition.",
    )
    args = parser.parse_args(argv)

    if args.lines:
        source = _iter_plain_lines(Path(args.lines))
    else:
        target = args.input_dir or args.path
        if not target:
            parser.error("give a CSV path, --input-dir, or --lines")
        csvs = _collect_csvs(Path(target))
        if not csvs:
            parser.error(f"no CSV files found under {target}")
        source = _iter_csv_rows(csvs)

    total = 0
    eligible = 0
    witnessed_rows: list[tuple[str, dict, str]] = []
    by_categ: Counter = Counter()
    witnessed_by_categ: Counter = Counter()
    clause_counts: Counter = Counter()
    examples: dict[str, list[str]] = {c: [] for c in _CLAUSE_ORDER}

    for doc, text, wc, categ in source:
        total += 1
        verdict = classify_line(text, wc)
        in_scope = args.all_lengths or verdict["route_eligible"]
        if not in_scope:
            continue
        eligible += 1
        by_categ[categ] += 1
        if verdict["witness"]:
            witnessed_by_categ[categ] += 1
            witnessed_rows.append((doc, verdict, categ))
            for clause in verdict["clauses"].split(","):
                clause_counts[clause] += 1
                if args.examples and len(examples[clause]) < args.examples:
                    examples[clause].append(text)

    print(f"\nSHORT_GARBAGE_WITNESS_ENABLE = {tu.SHORT_GARBAGE_WITNESS_ENABLE}  (does not affect this report)")
    print(
        f"witness constants: MIN_ALPHA={tu.SHORT_GARBAGE_WITNESS_MIN_ALPHA} "
        f"VARIETY_MIN_ALPHA={tu.SHORT_GARBAGE_WITNESS_VARIETY_MIN_ALPHA} "
        f"VARIETY_MAX={tu.SHORT_GARBAGE_WITNESS_VARIETY_MAX} "
        f"TRIPLE_MAX_ALPHA={tu.SHORT_GARBAGE_WITNESS_TRIPLE_MAX_ALPHA}"
    )
    scope = (
        "all lines"
        if args.all_lengths
        else f"word_count <= {tu.ISOLATED_CHAR_MIN_TOKENS}, no diacritics/structure/notation"
    )
    print(f"\n{total} lines read; {eligible} in scope ({scope})")

    print("\n=== stored category x witnessed ===")
    print(f"  {'category':10} {'in scope':>9} {'witnessed':>10} {'share':>7}")
    for categ in sorted(by_categ):
        n = by_categ[categ]
        w = witnessed_by_categ[categ]
        print(f"  {categ:10} {n:9d} {w:10d} {(w / n if n else 0):6.1%}")
    n_tot = sum(by_categ.values())
    w_tot = sum(witnessed_by_categ.values())
    print(f"  {'TOTAL':10} {n_tot:9d} {w_tot:10d} {(w_tot / n_tot if n_tot else 0):6.1%}")

    if clause_counts:
        print("\n=== which clause fired (a line may fire several) ===")
        for clause in _CLAUSE_ORDER:
            if clause_counts[clause]:
                print(f"  {clause:18} {clause_counts[clause]:8d}")

    # The two numbers the flag decision rests on. Neither is an error rate:
    # `categ` is the pipeline's own answer, so these are exposure counts that
    # tell you how many lines an annotator has to look at, and where.
    retained = witnessed_by_categ.get("Trash", 0)
    exposure = witnessed_by_categ.get("Clear", 0) + witnessed_by_categ.get("Noisy", 0)
    print("\n=== exposure (NOT an error rate — `categ` is the pipeline's answer, not gold) ===")
    print(f"  witnessed and currently Trash        : {retained:8d}   would stay Trash with the witness armed")
    print(f"  witnessed and currently Clear/Noisy  : {exposure:8d}   false-positive candidates — annotate these first")

    if args.examples:
        print("\n=== examples ===")
        for clause in _CLAUSE_ORDER:
            if examples[clause]:
                print(f"  [{clause}]")
                for text in examples[clause]:
                    print(f"    {text!r}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["document", "text", "word_count", "categ_current", "clauses", "gold_categ"])
            for doc, verdict, categ in witnessed_rows:
                writer.writerow([doc, verdict["text"], verdict["word_count"], categ, verdict["clauses"], ""])
        print(f"\nwrote {len(witnessed_rows)} candidate lines to {out_path}")
        print("  `gold_categ` is left blank on purpose: fill it blind, then the file is a gold set")
        print("  `tools/quality_model/evaluate.py::gold_gate()` can consume.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
