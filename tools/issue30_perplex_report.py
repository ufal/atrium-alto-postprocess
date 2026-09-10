#!/usr/bin/env python3
"""
tools/issue30_perplex_report.py
===============================
Aggregate report over the issue-#30 raw-perplexity delivery.

**The input file is deliberately not in this repository and must not be added.**
It is one row per line that PR #48 changes -- 367,208 rows across both
collections -- carrying `file` and the line `text`, which makes it the whole
scored corpus with its document names attached. The maintainer's instruction is
that it is never published. `.gitignore` blocks the filename; this tool is the
supported way to get numbers out of it without moving the rows anywhere.

Everything printed here is a count, a share or a threshold volume. No document
name, no line text, and no per-row output -- by construction, so the report can
be pasted into a public issue thread without a redaction pass.

Input columns (from the delivery's own README):

    collection, file, page_num, line_num, text, word_count,
    categ_delivered, categ_unpatched, categ_patched, perplex, perplex_raw

`perplex` is the stored value, capped at ``SHORT_PPL_CAP`` for ``word_count <= 2``;
`perplex_raw` is the uncapped recompute.

**Fidelity caveat, and it applies to every number below.** `perplex_raw` was
recomputed at batch size 32 rather than the configured 128. Against the stored
values on the 67,633 lines where the stored figure is uncapped, the median
relative difference is 2.9% and the 99th percentile 13.5%, arriving in whole
bf16 steps of the per-line mean token loss (one step is about 3.2% at typical
values). **Treat any line within a few percent of a threshold as undecided** --
which is why the cap section reports a margin band rather than a single count.

Usage:

    python tools/issue30_perplex_report.py /path/to/issue30_perplex_raw_367208.csv
    python tools/issue30_perplex_report.py <path> --cap 950
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

# The delivery is wider than the csv module's default field limit in the worst
# case (a long OCR line), and dying halfway through a 26 MB scan with an opaque
# error is not a useful failure.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

REQUIRED_COLUMNS = (
    "collection",
    "word_count",
    "categ_unpatched",
    "categ_patched",
    "perplex",
    "perplex_raw",
)

# Production default (`SHORT_PPL_CAP`), applied to word_count <= 2.
DEFAULT_CAP = 850.0
DEFAULT_SHORT_MAX_WC = 2

# Lines whose raw perplexity lands within this fraction of the cap are counted
# separately: the bf16 reproduction noise is larger than the distance, so their
# side of the threshold is not established by this data.
UNDECIDED_MARGIN = 0.05


def _fmt_share(part: int, whole: int) -> str:
    return f"{part:>9,}  ({100.0 * part / whole:5.1f}%)" if whole else f"{part:>9,}  (   -- )"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Aggregate-only report over the issue-#30 raw-perplexity delivery.",
        epilog="Prints counts and shares only. No document names, no line text.",
    )
    ap.add_argument("path", help="Path to issue30_perplex_raw_367208.csv (kept outside the repo).")
    ap.add_argument(
        "--cap",
        type=float,
        default=DEFAULT_CAP,
        help=f"Short-line perplexity cap to report against (default: {DEFAULT_CAP:g}, = SHORT_PPL_CAP).",
    )
    ap.add_argument(
        "--short-max-wc",
        type=int,
        default=DEFAULT_SHORT_MAX_WC,
        help=f"Highest word_count the cap applies to (default: {DEFAULT_SHORT_MAX_WC}).",
    )
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.path)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(f"error: {path.name} is missing column(s): {', '.join(missing)}", file=sys.stderr)
            return 2

        total = 0
        by_collection: Counter = Counter()
        by_wc: Counter = Counter()
        transitions: Counter = Counter()
        delivered_gap = 0
        single_token = 0

        short_total = 0
        short_above_cap = 0
        short_undecided = 0
        malformed = 0

        for row in reader:
            total += 1
            by_collection[row["collection"]] += 1

            try:
                wc = int(row["word_count"])
                raw = float(row["perplex_raw"])
            except (TypeError, ValueError):
                malformed += 1
                continue

            by_wc[wc if wc <= 3 else 4] += 1
            transitions[(row["categ_unpatched"], row["categ_patched"])] += 1

            if "categ_delivered" in row and row["categ_delivered"] != row["categ_unpatched"]:
                delivered_gap += 1
            if raw == 1.0:
                single_token += 1

            if wc <= args.short_max_wc:
                short_total += 1
                if raw > args.cap:
                    short_above_cap += 1
                if abs(raw - args.cap) <= args.cap * UNDECIDED_MARGIN:
                    short_undecided += 1

    if not total:
        print("error: no rows read", file=sys.stderr)
        return 1

    print(f"\n=== issue #30 raw-perplexity delivery: {path.name} ===")
    print(f"rows: {total:,}")
    for name, n in sorted(by_collection.items()):
        print(f"  {name:<6} {_fmt_share(n, total)}")
    if malformed:
        print(f"  ! {malformed:,} row(s) had an unreadable word_count/perplex_raw and were skipped")

    print("\n--- word_count ---")
    for wc in sorted(by_wc):
        label = f"{wc}" if wc <= 3 else "4+"
        print(f"  wc {label:<3} {_fmt_share(by_wc[wc], total)}")

    print("\n--- category movement (offline recompute, unpatched -> patched) ---")
    for (before, after), n in transitions.most_common():
        print(f"  {before:<9} -> {after:<9} {_fmt_share(n, total)}")
    print(f"\n  delivered vs unpatched disagreement: {delivered_gap:,}")
    print("    (the known gap between the live run and the offline re-scorer)")
    print(f"  single-token lines (perplex_raw == 1.0): {single_token:,}")
    print("    (nothing to predict; production stores 1.0 for these too)")

    print(f"\n--- the {args.cap:g} cap on wc <= {args.short_max_wc} ---")
    print(f"  lines the cap applies to: {short_total:,}")
    print(f"  raw perplexity above the cap: {_fmt_share(short_above_cap, short_total)}")
    print("    These are the lines the cap is actually holding up: uncapped, they")
    print("    carry a perplexity the short-line route would read differently.")
    print(f"  within +/-{UNDECIDED_MARGIN:.0%} of the cap: {_fmt_share(short_undecided, short_total)}")
    print("    Undecided by this data -- the bf16 reproduction noise is wider than")
    print("    the margin, so do not quote these on either side of the threshold.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
