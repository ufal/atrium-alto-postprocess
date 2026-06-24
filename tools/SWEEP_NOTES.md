# Config-constant importance tooling (issue #5)

Two offline tools live here. Both read the per-line `DOC_LINE_CATEG` CSVs as
**immutable ground truth** and write any revised CSVs to a separate directory.

| Tool                        | Purpose                                                                                                                                                       |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `recategorize_from_csv.py`  | Faithful offline re-scorer + evaluator. Re-runs categorisation from the frozen `perplex` / `orig_lang_score` / `text` signals under a chosen constant set.    |
| `const_importance_sweep.py` | Samples the tunable constants, scores each with the re-scorer, and reports which constants drive a chosen objective (RandomForest / Optuna+fANOVA surrogate). |

## One engine, not two

There is a single scoring engine: the **real** production functions
(`compute_quality_score`, `categorize_line`, `apply_document_postprocessing`).
A trial's constant set is applied by temporarily overriding the module-level
tunables with `text_util_langID.override_constants(...)` — there is no parallel
NumPy re-implementation to drift out of sync.

**Parity guarantee:** at the current config the re-score reproduces the stored
`categ` exactly, so the sweep baseline sits at `flip_rate == 0`. This is locked
by `tests/test_recategorize_parity.py`; if a future change breaks it, the sweep
is no longer measuring production and the test fails.

## Running a sweep

```bash
pip install -r tools/requirements-sweep.txt   # scikit-learn / optuna / matplotlib

# sklearn surrogate (no Optuna needed), importance for macro_f1 agreement
python tools/const_importance_sweep.py \
    --input-dir data_samples/DOC_LINE_CATEG --config config_langID.txt \
    --output-dir sweep_out --backend sklearn --metric macro_f1 --n-trials 400

# Optuna + fANOVA — RANDOM sampling for unbiased importance
python tools/const_importance_sweep.py \
    --input-dir data_samples/DOC_LINE_CATEG --config config_langID.txt \
    --output-dir sweep_out --backend optuna --sampler random --n-trials 400
```

Outputs: `baseline_metrics.json`, `baseline_per_document.json`, `param_importance.json`
(+ `_permutation` for sklearn), `best_config.json`, `trials.csv`, `param_importance.png`,
`sweep_summary.json`.

## Methodology guardrails (what changed for #5)

- **fANOVA needs ~uniform sampling**, so the Optuna default is `--sampler random`.
  TPE concentrates near the optimum and biases fANOVA; the tool warns if you ask
  for TPE. Use TPE only to *optimise* a config, not to rank importance.
- The sklearn surrogate reports **out-of-bag R²** (held-out), not just train R².
- Importance is **skipped for a single-parameter study** (it is trivially 100%).
- **`QS_WEIGHT_*` are frozen by default** (`--include-qs-weights` to sweep them):
  prior runs showed the linear weight composition has low practical influence vs.
  the category thresholds and the garbage / inversion / hard-sweep gates.
- The search space now includes the **#3 routing thresholds** the previous sweep
  never varied (`HARD_SWEEP_*`, `PPL_EXTREME_MIN`, `PPL_GARBAGE_ABSOLUTE`,
  `LOWPPL_*`, `MOSTLY_READABLE_VALID_MIN`, `INVERTED_*`, `SUSPICIOUS_*`, …) — these
  now do most of the real Trash routing.
- Sub-sampling is **by document** (`--sample-docs`), never by line: page-level
  post-processing needs whole pages.

## Interpreting results

Importance is corpus-specific and **relative to the current config** (the
`flip_rate == 0` baseline). The CSVs under `data_samples/DOC_LINE_CATEG` are a
tiny smoke fixture (a few documents); the numbers there exercise the machinery
but are **not** a basis for tuning. Run on the full `DOC_LINE_CATEG` corpus for
meaningful importances, report per-document (`baseline_per_document.json`), and
treat the ranking as directional until it is stable across both backends with
random sampling.
