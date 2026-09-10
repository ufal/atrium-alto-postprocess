# Gold labels: making the tuning objective non-circular

## The problem this solves

Every driver in `tools/` scores a trial through
`recategorize_from_csv.evaluate_dataframe()`. By default that function grades the
re-score against **the CSV's own stored `categ` column** — labels the pipeline
itself wrote.

That makes the objective self-referential, with two consequences that are easy to
miss because nothing errors:

1. **The shipped configuration is optimal by construction.** `SWEEP_NOTES.md`
   records the baseline as `flip_rate=0.0000`, i.e. a perfect score. No sweep can
   improve on a perfect score; it can only measure how far a trial drifts.
2. **A genuine accuracy improvement scores as pure damage.** A constant change
   that corrects 1,000 wrong labels registers as 1,000 flips away from "truth".

`agent_dev_logs/plans/30.plan.md` states it plainly:

> Every offline tool in `tools/` grades against the pipeline's own stored `categ`,
> so a label-changing improvement scores as pure damage on all of them. Without a
> gold set wired in, Phase 3 cannot tell success from regression.

`--gold-column` is the fix. It does not change how anything is scored; it changes
**what the score is measured against**.

## The contract

A gold CSV is an ordinary `DOC_LINE_CATEG` CSV with extra columns appended:

| column        | required | meaning                                                                                                                                       |
|---------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `gold_categ`  | yes      | The human label. One of `Clear`, `Noisy`, `Trash`, `Non-text`, `Empty`. Blank means "not annotated" and the row is skipped by the CLI report. |
| `gold_source` | no       | Where the label came from (annotator, round, file). Provenance has mattered in this repository.                                               |
| `gold_note`   | no       | Free text — why this line is interesting, or why the call was hard.                                                                           |

Rules that the tooling enforces rather than trusts:

* **One CSV per document**, named for the document, exactly like
  `data_samples/DOC_LINE_CATEG`. The whole offline stack groups by the `file`
  column and smooths each document independently; packing several documents into
  one CSV puts lines from different documents on a shared page and lets
  page-level post-processing act across a boundary that does not exist in
  production.
* **A named gold column that is absent is an error, not a fallback.** Scoring
  predictions against themselves yields a perfect score, and a silent perfect
  score is worse than a crash. `evaluate_dataframe` raises `KeyError`.
* **Values are normalised** through `normalize_category()`, the same function the
  rest of the re-scorer uses, so `clear` / `Clear` / `CLEAR` agree.

## Using it

```bash
# Re-score and report agreement with gold, per document and in total.
python tools/recategorize_from_csv.py --input-dir tools/gold \
    --config setup/config.txt --report-only --gold-column gold_categ

# A/B one constant against gold. Prints an adopt/parity/regression verdict per
# value, measured against the SHIPPED labels' own gold score.
python tools/ab_constant_eval.py --input-dir <gold dir> --config setup/config.txt \
    --gold-column gold_categ --const SHORT_PPL_CAP --values 850,950

# A full importance sweep whose objective is agreement with humans.
python tools/const_importance_sweep.py --input-dir <gold dir> \
    --config setup/config.txt --output-dir sweep_gold \
    --backend sklearn --metric macro_f1 --n-trials 400 --gold-column gold_categ

# Ablation and greedy elimination take the same flag.
python tools/run_ablation_study.py --input-dir <gold dir> --gold-column gold_categ
python tools/greedy_backward_elimination.py --input-dir <gold dir> --gold-column gold_categ
```

### Reading the numbers

With `--gold-column` set, the familiar metric names change meaning:

* `flip_rate` is **disagreement with gold**, not drift from the stored labels. A
  non-zero baseline is expected and is the thing being minimised — it is no
  longer a warning sign.
* `macro_f1` is agreement with humans.
* `baseline_vs_gold` carries the same metrics for the pipeline's **stored**
  labels, and `gold_delta_macro_f1` is the difference. **This is the number that
  matters.** Beating the incumbent on the incumbent's own labels means nothing;
  beating it on human labels is the whole point.

`tools/ab_constant_eval.py` applies the same rule
`tools/quality_model/evaluate.py::gold_gate()` already used: a candidate is worth
adopting only when it beats the shipped labels against gold **and** does not
raise `Clear-loss` (true-`Clear` lines pushed to `Trash`/`Non-text`).

One caveat on small sets: `macro_f1` averages over all five categories, so a
document containing only two of them scores low in absolute terms no matter how
well it does. Compare rows against each other, never against 1.0.

## The worked example in `tools/gold/`

`tools/gold/` holds three small CSVs generated by
`tools/gold/build_example_gold.py` from labels this repository already owns. It
exists so the gold path is exercised end to end without waiting on an external
annotation file — **it is a schema demonstration, not a tuning corpus.** Fifteen
lines cannot calibrate anything.

Only rows whose category the suite asserts with **exact equality** are used, so
every label is a committed claim about the right answer:

| source                             | rows | assertion that makes it gold                            |
|------------------------------------|------|---------------------------------------------------------|
| `CLEAR`                            | 9    | `assert _categ(...) == "Clear"`                         |
| `TRASH_INVERTED`                   | 3    | `assert categ == "Trash"` (test_rotation_regression.py) |
| `TRASH_GARBAGE`, hard-sweep subset | 3    | `assert _categ(...) == "Trash"`                         |

Deliberately excluded: `NOISY`, the rest of `TRASH_GARBAGE`,
`ROT_FALSE_POSITIVE_GUARDS`, `HEADLINE_NUMBERED` and `SHORT_EXCEPTIONS` — the
suite asserts only a **floor** on these (`!= "Trash"` / `!= "Clear"`) because
"the invariant under test is the floor, not the exact band", so their fourth
element is an intent rather than a pinned category. Also excluded:
`VOCABULARY_SHORT` and `NOTATION_SHORT`, which record *current behaviour* rather
than truth — importing them as gold would smuggle the pipeline's own answer back
into the objective.

**One row disagrees with production on purpose.** `oueussd` carries gold `Trash`;
the merged issue-#30 gate lets it reach `Clear`. That disagreement is the
accepted technical debt recorded in `tests/calibration_fixtures.py`, and it is
the point: a gold set that agreed with the pipeline everywhere would demonstrate
nothing. Regenerate with:

```bash
python tools/gold/build_example_gold.py          # rewrite
python tools/gold/build_example_gold.py --check  # verify committed == fresh
```

## The real annotation sets — delivered

Both sets arrived on 2026-09-10 (issue #30, `#issuecomment-5620689006`) and are
in the tree as **`tools/gold/sidecars/issue30_gold_2067.csv`**:

| `gold_source`      | rows      | what it is                                                                     |
|--------------------|-----------|--------------------------------------------------------------------------------|
| `issue30_508`      | **508**   | graded from the issue-#30 changed population, stratified by decade             |
| `calibration_1567` | **1,559** | annotated during earlier calibration rounds                                    |
|                    | **2,067** | 816 distinct documents, no duplicate keys                                      |

Of the 1,567 originally annotated calibration lines, **eight are absent**: seven
keys do not exist in the delivered `2907` rows and one has different text
(`3.zm` annotated against `3.2m` delivered).

**Read the provenance before quoting a figure from this file.** The
`issue30_508` labels were originally *derived* from a right/borderline/wrong
grade on the direction of each move, not annotated: *right* was mapped to the
promoted category and *wrong* to the original one. That mapping is invalid —
*right* means the move went the right way, not that it landed on the right
category. The rows were re-labelled by hand afterwards and **55 of the 508
labels changed**. Every agreement figure computed before 2026-09-10 used the
derived labels; the patch's agreement drops from 76.7% to 68.8% on the corrected
set. The quality figures (76.5 / 9.6 / 13.9, 5.5 : 1) are computed from the
grades directly and are unaffected.

### It is a sidecar, not a per-document CSV

This file breaks the one-CSV-per-document rule above, deliberately and as the
only exception. It is a **key-indexed sidecar**: `file`, `page_num`, `line_num`,
`gold_categ`, `gold_source` and nothing else.

It cannot take the documented form. The 40 feature columns live in the delivered
`2907` batch, which is not in this repository and cannot be — it is the whole
collection, with document names and line text. And 2,067 lines spread over 816
documents would mean 816 CSVs averaging 2.5 rows each, which is not a corpus
layout, it is a filesystem full of fragments.

The rule it exists to serve is still enforced, just at a different moment:
because the sidecar carries no `categ` of its own, it can only be scored after
being joined onto real `DOC_LINE_CATEG` rows, and that join is per document. The
page-boundary hazard the rule guards against is therefore structurally absent —
a sidecar row cannot put itself on someone else's page.

**Sidecars live in `tools/gold/sidecars/`, never in `tools/gold/` itself.** That
is not tidiness. `tools/gold/` is a valid `--input-dir`, and the drivers glob it
for `*.csv` and treat every hit as a scoreable per-document gold set. A 2,067-row,
816-document, `categ`-less file sitting next to `GOLD_CLEAR.csv` breaks the
per-document invariant and silently corrupts anything loaded from that directory —
it was tried, and it took three tests down with it. The subdirectory keeps the
non-recursive glob clean; `tests/test_gold_objective.py` pins both halves.

One hazard remains and is worth stating: `ab_constant_eval`, `run_ablation_study`
and `greedy_backward_elimination` call `load_csvs(..., recursive=True)`. **Do not
point their `--input-dir` at `tools/gold/`** — pass the delivered batch directory
and name the sidecar with `--gold-sidecar`, which is what the flag is for.

### Joining it

`--gold-sidecar` attaches the labels onto whatever frame is being scored, so the
`--gold-column` path above works unchanged:

```bash
# The sidecar supplies gold_categ; --gold-column then scores against it.
python tools/recategorize_from_csv.py --input-dir <delivered DOC_LINE_CATEG dir> \
    --config setup/config.txt --report-only \
    --gold-sidecar tools/gold/sidecars/issue30_gold_2067.csv --gold-column gold_categ
```

Every driver that already takes `--gold-column` takes `--gold-sidecar` too; both
are registered in one place (`recategorize_from_csv.add_gold_column_argument`)
for the reason given in that function's docstring.

The join is on `(file, page_num, line_num)` with the locator coercion the
re-scorer already applies, and **unmatched rows on either side are reported
rather than dropped silently**. A sidecar that matches nothing produces a frame
with an empty gold column, which `_gold_report` already refuses to score as a
perfect result.

Expect a partial match, and expect it to be correct: only **484 of the 508**
`issue30_508` keys are still in the changed population. The 24 absent ones are
the lines `is_domain_notation()` now lifts on its own, so the patch no longer
moves them — the same 378,735 → 367,208 shrink recorded in the digest.

### What is still missing

The sidecar makes the objective non-circular. It does not make it complete:

* **Per-line only.** `apply_document_postprocessing()` changes the category of
  roughly 27% of these lines in production and is absent from the per-line
  grading that produced the quality figures. `evaluate_dataframe` *does* apply
  the smoothing, so a document-aware re-grade is now possible — it needs the
  `2907` delivery, not more annotation.
* **The 850 cap.** Raw uncapped perplexity for the changed population exists as
  an external file and is deliberately not in this repository; see
  `tools/issue30_perplex_report.py`.

`SWEEP_NOTES.md`'s standing warning — *do not adopt `best_config.json` blindly* —
still applies. A 2,067-line sidecar over 816 documents is enough to tell an
improvement from a regression; it is not enough to fit constants against.
