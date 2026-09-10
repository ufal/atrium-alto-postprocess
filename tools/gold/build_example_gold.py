#!/usr/bin/env python3
"""
tools/gold/build_example_gold.py
================================
Regenerate ``tools/gold/example_gold.csv`` -- the worked example of the gold-CSV
contract documented in ``tools/GOLD.md``.

Why this exists
---------------
Every driver in ``tools/`` scores a trial against the pipeline's own stored
``categ``, which makes the shipped configuration optimal by construction. The
``--gold-column`` flag breaks that circularity, but a flag with no data behind it
is untested plumbing. This script builds a small, real, production-schema gold
CSV from labels the repository already owns, so the gold path is exercised
end to end without waiting on an external annotation file.

What goes in, and what deliberately does not
--------------------------------------------
Only rows whose category the suite asserts with **exact equality** are used, so
every gold label here is a committed, unambiguous claim about the right answer:

* ``CLEAR``                -- ``assert _categ(...) == "Clear"``
* ``TRASH_INVERTED``       -- ``assert categ == "Trash"`` in test_rotation_regression.py
* the hard-sweep subset of ``TRASH_GARBAGE`` -- ``assert _categ(...) == "Trash"``
  (the ``_HARD_SWEEP`` filter: ppl > 1000 and orig_lang_score < 0.45)

Excluded, on purpose:

* ``NOISY``, the rest of ``TRASH_GARBAGE``, ``ROT_FALSE_POSITIVE_GUARDS``,
  ``HEADLINE_NUMBERED`` and ``SHORT_EXCEPTIONS`` -- the suite asserts only a
  FLOOR on these (``!= "Trash"`` / ``!= "Clear"``), deliberately, because "the
  0.80 boundary may legitimately lift some of these further to Clear, and the
  invariant under test is the floor, not the exact band". Their fourth element
  is therefore an intent, not a pinned band, and promoting it to gold would
  invent precision the repository has explicitly declined to claim.
* ``VOCABULARY_SHORT`` and ``NOTATION_SHORT`` -- these record CURRENT BEHAVIOUR,
  not truth. Their own comment block says so: the `Trash` -> `Clear` flip on
  those rows "IS the change under discussion, recorded here as a diff rather
  than left as a claim". Importing them as gold would smuggle the pipeline's own
  answer back into the objective, which is the exact circularity this file
  exists to break.
* ``NON_TEXT`` and ``ALLCAPS_HEADLINE`` -- pre-filter cases with no frozen
  perplexity, and ``ALLCAPS_HEADLINE`` expects ``Process``, a routing sentinel
  rather than a category.

This keeps the example small. That is the right trade for a worked example of a
schema: it demonstrates the contract and exercises the plumbing, and it is not a
substitute for the real annotation sets (see tools/GOLD.md).

One row disagrees with production on purpose: ``oueussd`` carries gold
``Trash``. That is the documented correct answer (``tests/calibration_fixtures.py``
records it as accepted technical debt, not a re-baselined expectation), and the
merged #30 gate lets it reach ``Clear``. A gold set that agreed with the pipeline
on every line would demonstrate nothing.

``original_lang`` follows ``tests/test_calibration.py::_categ``: ``ces_Latn`` for
the five-tuple Czech fixtures, and each row's own language where the list carries
one. Hardcoding it for non-Czech rows is a known harness bug in this repository
and is not repeated here.

Usage
-----
    python tools/gold/build_example_gold.py            # rewrite the CSV
    python tools/gold/build_example_gold.py --check    # verify it is up to date
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from recategorize_from_csv import (  # noqa: E402
    GOLD_COLUMN_DEFAULT,
    _load_lang_config,
    _rescore_row,
)

from tests import calibration_fixtures as CF  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
OUT_PREFIX = "GOLD_"

# (attribute name, arity) -- arity 5 rows take the ces_Latn default, arity 6 rows
# carry their own language in position 3. Mirrors _FIXTURE_ARITIES in
# tests/test_calibration.py.
GROUND_TRUTH_LISTS = [
    ("CLEAR", 5),
    ("TRASH_INVERTED", 5),
    ("TRASH_GARBAGE_HARD_SWEEP", 5),
]

DEFAULT_LANG = "ces_Latn"


def _hard_sweep_rows():
    """The ``TRASH_GARBAGE`` rows the suite pins to ``Trash`` exactly.

    Mirrors ``_HARD_SWEEP`` in tests/test_calibration.py. The remaining
    ``TRASH_GARBAGE`` rows carry only a ``!= "Clear"`` floor, so their exact band
    is not a committed claim and they are not gold.
    """
    return [
        f
        for f in CF.TRASH_GARBAGE
        if _row_values(f)[1] is not None
        and _row_values(f)[2] is not None
        and _row_values(f)[2] < 0.45
        and _row_values(f)[1] > 1000.0
    ]


def _row_values(row):
    """Unwrap a ``pytest.param(...)`` ParameterSet; plain tuples pass through."""
    return tuple(getattr(row, "values", row))


def build_rows() -> list[dict]:
    expected_langs, known_bases = _load_lang_config(str(_ROOT / "setup" / "config.txt"))
    rows: list[dict] = []

    for list_name, arity in GROUND_TRUTH_LISTS:
        fixtures = _hard_sweep_rows() if list_name == "TRASH_GARBAGE_HARD_SWEEP" else getattr(CF, list_name)
        for line_num, raw in enumerate(fixtures, start=1):
            values = _row_values(raw)
            if arity == 5:
                text, ppl, lang_score, gold, note = values
                original_lang = DEFAULT_LANG
            else:
                text, ppl, lang_score, original_lang, gold, note = values

            if ppl is None or lang_score is None:
                # Pre-filter fixtures carry no frozen model signals; they are not
                # scoring cases and cannot be re-scored faithfully.
                continue

            seed = {
                "text": text,
                "original_text": text,
                "original_lang": original_lang,
                "orig_lang_score": f"{lang_score}",
                "perplex": f"{ppl}",
                "perplex_raw": f"{ppl}",
                "categ": "Noisy",
                "word_count": str(len(text.split())),
                "file": f"GOLD_{list_name}",
                "page_num": 1,
                "line_num": line_num,
                "split_ws": "",
                "split_we": "",
            }
            scored = _rescore_row(seed, expected_langs, known_bases)
            scored[GOLD_COLUMN_DEFAULT] = gold
            scored["gold_source"] = f"tests/calibration_fixtures.py::{list_name}"
            scored["gold_note"] = note
            rows.append(scored)

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="Verify the committed CSVs match a fresh build.")
    args = ap.parse_args()

    df = pd.DataFrame(build_rows())

    # gold columns last, so each file still reads as a DOC_LINE_CATEG CSV.
    tail = [GOLD_COLUMN_DEFAULT, "gold_source", "gold_note"]
    df = df[[c for c in df.columns if c not in tail] + tail]

    # ONE CSV PER DOCUMENT, exactly like data_samples/DOC_LINE_CATEG. The whole
    # offline stack groups by the `file` column and smooths each document
    # independently, so packing several documents into one CSV would put lines
    # from different documents on a shared page and let page-level
    # post-processing act across a boundary that does not exist in production.
    written: dict[str, pd.DataFrame] = {}
    for file_id, doc in df.groupby("file", sort=True):
        written[str(file_id)] = doc.reset_index(drop=True)

    if args.check:
        stale = []
        for file_id, doc in written.items():
            path = OUT_DIR / f"{file_id}.csv"
            if not path.exists():
                stale.append(f"missing: {path}")
                continue
            current = pd.read_csv(path, dtype=str, keep_default_na=False)
            fresh = pd.read_csv(__import__("io").StringIO(doc.to_csv(index=False)), dtype=str, keep_default_na=False)
            if not current.equals(fresh):
                stale.append(f"stale: {path}")
        if stale:
            print("\n".join(stale), file=sys.stderr)
            return 1
        print(f"up to date: {len(written)} document(s), {len(df)} rows")
        return 0

    for file_id, doc in written.items():
        path = OUT_DIR / f"{file_id}.csv"
        doc.to_csv(path, index=False, encoding="utf-8")
        print(f"wrote {path.name} -- {len(doc)} rows, {len(doc.columns)} columns")

    agree = int((df["categ"] == df[GOLD_COLUMN_DEFAULT]).sum())
    print(f"\n{len(df)} gold rows across {len(written)} document(s)")
    print(f"per-line re-score agrees with gold on {agree}/{len(df)} rows")
    for _, r in df[df["categ"] != df[GOLD_COLUMN_DEFAULT]].iterrows():
        print(f"  DISAGREES: {r['text']!r}  pipeline={r['categ']}  gold={r[GOLD_COLUMN_DEFAULT]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
