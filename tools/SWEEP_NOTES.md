# Config-constant importance tooling (issue #5)

Two offline tools live here. Both read the per-line `DOC_LINE_CATEG` CSVs as
**immutable ground truth** and write any revised CSVs to a separate directory.

| Tool                        | Purpose                                                                                                                                                                                                                                                                                                                          |
|-----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `recategorize_from_csv.py`  | Faithful offline re-scorer + evaluator. Re-runs categorisation from the frozen `perplex` / `orig_lang_score` / `text` signals under a chosen constant set.                                                                                                                                                                       |
| `const_importance_sweep.py` | Samples the tunable constants, scores each with the re-scorer, and reports which constants drive a chosen objective (RF / Optuna+fANOVA / Morris / Sobol).                                                                                                                                                                       |
| `importance_consensus.py`   | Cross-backend consensus tool. Loads importance JSONs from different backends and identifies robust parameters that consistently rank in the top-K.                                                                                                                                                                               |
| `rule_coverage_report.py`   | Rule-fire coverage instrumentation. Runs the production categorisation engine over a dataset and counts how many times each structural rule and per-line penalty actually executes. A rule with a fire count of 0 across all documents is provably dead code and can be permanently retired without requiring human gold labels. |

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
pip install -r tools/requirements-sweep.txt   # scikit-learn / optuna / matplotlib / SALib

# sklearn surrogate (no Optuna needed), importance for macro_f1 agreement
python tools/const_importance_sweep.py \
    --input-dir data_samples/DOC_LINE_CATEG --config config_langID.txt \
    --output-dir sweep_out --backend sklearn --metric macro_f1 --n-trials 400

# Optuna + fANOVA — RANDOM sampling for unbiased importance
python tools/const_importance_sweep.py \
    --input-dir data_samples/DOC_LINE_CATEG --config config_langID.txt \
    --output-dir sweep_out --backend optuna --sampler random --n-trials 400

# Morris screening (fast, constraint-repaired model-free sensitivity)
python tools/const_importance_sweep.py \
    --input-dir data_samples/DOC_LINE_CATEG --config config_langID.txt \
    --output-dir sweep_out --backend morris --morris-r 10

# Sobol global sensitivity (expensive, run on cluster, constraint-free params only)
python tools/const_importance_sweep.py \
    --input-dir data_samples/DOC_LINE_CATEG --config config_langID.txt \
    --output-dir sweep_out --backend sobol --sobol-n 256

# Cross-backend consensus
python tools/importance_consensus.py sweep_out_rf sweep_out_optuna sweep_out_sobol
```

## Methodology guardrails (what changed for #5)

* **fANOVA needs ~uniform sampling**, so the Optuna default is `--sampler random`.
TPE concentrates near the optimum and biases fANOVA; the tool warns if you ask
for TPE. Use TPE only to *optimise* a config, not to rank importance.
* The sklearn surrogate reports **out-of-bag R²** (held-out), not just train R².
* SALib backends (Morris/Sobol) evaluate global sensitivity and interaction coupling.
Morris uses constraint repair to keep samples valid; Sobol holds constrained variables
at baseline to ensure pure main/total effect indices.
* Importance is **skipped for a single-parameter study** (it is trivially 100%).
* **`QS_WEIGHT_*` are frozen by default** (`--include-qs-weights` to sweep them):
prior runs showed the linear weight composition has low practical influence vs.
the category thresholds and the garbage / inversion / hard-sweep gates.
* Sub-sampling is **by document** (`--sample-docs`), never by line: page-level
post-processing needs whole pages.

## Findings & Close-out (Issue #5)

After running the full suite of importance sweeps and rule coverage on the full corpus, the following conclusions govern the configuration space:

1. **The "Big Two" dominate**: The readability gate (`MOSTLY_READABLE_VALID_MIN`) and low-perplexity rescue (`LOWPPL_CLEAR_MAX`) control the vast majority of categorization movement.
2. **`QS_WEIGHT_*` are non-identifiable**: The 9 quality score weights sum to 1.0; tuning them offline merely shifts the distribution arbitrarily without moving the actual accuracy frontier. They are frozen by default and should not be deleted, just left at their current values.
3. **No Dead Rules**: The `rule_coverage_report.py` proved that **0 rules are dead code**. While the greedy backward elimination tool with loose tolerances (`--macro-tol 0.02`) suggested pruning 12 rules, coverage instrumentation shows they *do* fire and act as critical safeguards (e.g. inverted run detection). Do not act on the greedy output as a deletion mandate. All 14 rules stay.
4. **Garbage Density De-confounded**: `CATEG_GARBAGE_DENSITY_HIGH` was decoupled from the QS scaling factor (`QS_GARBAGE_NORM_MAX`). Sweep results (via Sobol S1/ST separation) show the hard gate is what matters (~9% importance), while the QS scale drops to the noise floor.
5. **Near-Optimum caveat**: The best configurations found by the surrogate only achieve a ~2.4% flip rate deviation with minimal KL divergence (KL ≈ 0.0015). The current production configuration is already near-optimal against its own labels. *Do not adopt `best_config.json` blindly*, as values for low-importance parameters in that file are simply statistical noise.

## Interpreting results

**Outputs:** `param_importance.json` (or `param_importance_permutation.json` / `S1.json` / `ST.json` depending on backend), `importance_consensus.json` (from the consensus tool), `best_config.json`, `trials.csv`, `baseline_metrics.json`.

Importance is corpus-specific and **relative to the current config** (the `flip_rate == 0` baseline). The CSVs under `data_samples/DOC_LINE_CATEG` are a tiny smoke fixture (a few documents); the numbers there exercise the machinery but are **not** a basis for tuning. Run on the full `DOC_LINE_CATEG` corpus for meaningful importances, report per-document (`baseline_per_document.json`), and treat the ranking as directional until it is stable across multiple backends (RF, Optuna, Morris, Sobol) using `importance_consensus.py`. Morris/Sobol backends are purely for importance and coupling analysis against self-generated labels; final objective recalibration remains blocked until human-annotated gold labels exist.
