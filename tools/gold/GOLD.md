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

## The real annotation sets

The sets that can actually calibrate this pipeline are **not in the tree**:

* **508 lines** graded from the issue-#30 changed population, stratified by
  decade — the basis for every quality figure quoted in `agent_dev_logs/`.
* **1,567 lines** annotated during earlier calibration rounds.

Both exist only as @david-spacil's own files. They are the blocker for enabling
`SHORT_GARBAGE_WITNESS_ENABLE` and for any real parameter refinement. When they
arrive, converting them is a matter of joining the annotation onto the delivered
`DOC_LINE_CATEG` rows by `(file, page_num, line_num)` and writing `gold_categ`.

Until then, treat every "best config" produced by these tools as unvalidated:
`SWEEP_NOTES.md`'s standing warning — *do not adopt `best_config.json` blindly* —
applies with full force.
