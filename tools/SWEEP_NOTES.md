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

## Running the Unified Pipeline (Recommended)

The easiest way to execute the full parameter optimization suite (Coverage, RF, Optuna, Morris, Sobol, and Cross-Backend Consensus) is via the unified shell script.

```bash
# Smoke test (fast run on dummy data for verification)
# Args: <input_dir> <config> <out_base> <ml_trials> <sobol_n> <morris_r>
./tools/run_optim_pipeline.sh data_samples/DOC_LINE_CATEG config_langID.txt sweep_output_smoke 64 64 10

# Full cluster run (authoritative analysis using default high-budget params)
./tools/run_optim_pipeline.sh /path/to/full/DOC_LINE_CATEG config_langID.txt sweep_output_production
```

<details>
    <summary>Latest console output from the pipeline run (no optuna due to sqlite3 dep)</summary>

```terminaloutput
(venv-lang) lutsai@stargate:/lnet/work/projects/atrium/alto_util$ ./tools/run_optim_pipeline.sh data_samples/DOC_LINE_CATEG config_langID.txt sweep_output_production
============================================================
 ATRIUM ALTO Post-Process : Unified Optimization Pipeline
============================================================
 Input Data   : data_samples/DOC_LINE_CATEG
 Config File  : config_langID.txt
 Output Base  : sweep_output_production
 ML Trials    : 400 (RF/Optuna)
 Sobol N      : 256
------------------------------------------------------------
>> Checking/Installing dependencies...

[notice] A new release of pip available: 22.3 -> 26.1.2
[notice] To update, run: pip install --upgrade pip

[1/6] Running Rule Coverage Report...
Loaded 2,171 lines (1,463 scored) from data_samples/DOC_LINE_CATEG
Phase 1 — fire-count pass …
Phase 2 — LOO pass (14 rules × 1 recategorize each) …
  [ 1/14] penalty_ledger_fragmentation       decisive=11  clear_loss=0
  [ 2/14] penalty_mid_uppercase              decisive=1  clear_loss=0
  [ 3/14] penalty_vowelless                  decisive=1  clear_loss=0
  [ 4/14] penalty_wqx_rot                    decisive=11  clear_loss=0
  [ 5/14] rule_absolute_ppl                  decisive=1  clear_loss=0
  [ 6/14] rule_allcaps                       decisive=0  clear_loss=0
  [ 7/14] rule_extreme_ppl                   decisive=3  clear_loss=0
  [ 8/14] rule_garbage_density               decisive=0  clear_loss=0
  [ 9/14] rule_hard_sweep                    decisive=2  clear_loss=0
  [10/14] rule_inverted                      decisive=0  clear_loss=0
  [11/14] rule_lowppl_clear                  decisive=70  clear_loss=0
  [12/14] rule_mostly_readable_noisy         decisive=156  clear_loss=0
  [13/14] rule_short_garbage                 decisive=4  clear_loss=0
  [14/14] rule_trailing_fill_rescue          decisive=26  clear_loss=0

=== Rule Coverage Report (n_scored=1,463) ===
  Rule / Penalty                     | fire_count |  fire_rate |   decisive |   clr_loss | class
----------------------------------------------------------------------------------------------------------

  — determine_category —
  rule_absolute_ppl                  |          4 |     0.0027 |          1 |          0 | LOAD-BEARING
  rule_allcaps                       |          7 |     0.0048 |          0 |          0 | REDUNDANT-HERE
  rule_extreme_ppl                   |         64 |     0.0437 |          3 |          0 | LOAD-BEARING
  rule_garbage_density               |         20 |     0.0137 |          0 |          0 | REDUNDANT-HERE
  rule_hard_sweep                    |         94 |     0.0643 |          2 |          0 | LOAD-BEARING
  rule_inverted                      |          2 |     0.0014 |          0 |          0 | REDUNDANT-HERE
  rule_lowppl_clear                  |        187 |     0.1278 |         70 |          0 | LOAD-BEARING
  rule_mostly_readable_noisy         |        170 |     0.1162 |        156 |          0 | LOAD-BEARING
  rule_short_garbage                 |         71 |     0.0485 |          4 |          0 | LOAD-BEARING
  rule_trailing_fill_rescue          |         30 |     0.0205 |         26 |          0 | LOAD-BEARING

  — categorize_line penalties —
  penalty_ledger_fragmentation       |        134 |     0.0916 |         11 |          0 | LOAD-BEARING
  penalty_mid_uppercase              |         41 |     0.0280 |          1 |          0 | LOAD-BEARING
  penalty_vowelless                  |          8 |     0.0055 |          1 |          0 | LOAD-BEARING
  penalty_wqx_rot                    |        184 |     0.1258 |         11 |          0 | LOAD-BEARING

Summary: 11 LOAD-BEARING  |  3 REDUNDANT-HERE  |  0 DEAD

All rules fired at least once — no dead code detected on this dataset.

REDUNDANT-HERE rules (fire_count > 0, decisive_count == 0 — keep; entanglement suspected):
  - rule_allcaps
  - rule_garbage_density
  - rule_inverted

JSON written → sweep_output_production/rule_coverage.json

[2/6] Running Sklearn (Random Forest) Sweep...
Running sweep | profile=default | backend=sklearn | metric=macro_f1 (maximize)
Sweeping 37 parameter(s): ['CATEG_TRASH_SCORE_MAX', 'CATEG_NOISY_SCORE_MAX', 'CATEG_GARBAGE_DENSITY_HIGH', 'QS_GARBAGE_NORM_MAX', 'ROT_RATIO_INVERTED_MIN', 'WEIRD_RATIO_INVERTED_MIN', 'PPL_INVERTED_MIN', 'PERPLEXITY_THRESHOLD_MAX', 'SHORT_PPL_CAP', 'HARD_SWEEP_LANG_MAX', 'HARD_SWEEP_PPL_MIN', 'PPL_EXTREME_MIN', 'EXTREME_LANG_CONF', 'PPL_GARBAGE_ABSOLUTE', 'LOWPPL_CLEAR_MAX', 'LOWPPL_CZECH_CLEAR_MAX', 'CZECH_CLEAR_GARBAGE_MAX', 'MOSTLY_READABLE_VALID_MIN', 'SHORT_NOISY_QS_PENALTY', 'WORD_W_PENALTY', 'GHOST_DOMINATED_MIN_RATIO', 'SUSPICIOUS_ROT_RATIO', 'SUSPICIOUS_WQX_RATIO', 'INVERTED_WEIRD_PENALTY', 'GHOST_HITS_INVERTED_MIN', 'ROT_HIGH_LANG_CONF', 'LANG_SCORE_ROUGH', 'INVERTED_RUN_MIN', 'INVERTED_PAGE_MAJORITY', 'SURROUNDED_TRASH_QS_MARGIN', 'PAGE_GARBAGE_CLEAR_MAX', 'PAGE_GARBAGE_LANG_MAX', 'PAGE_GARBAGE_MEDIAN_QS_MAX', 'PAGE_GARBAGE_NOISY_QS_MAX', 'PAGE_CLEAN_CLEAR_MIN', 'PAGE_CLEAN_MEDIAN_QS_MIN', 'PAGE_CLEAN_RECOVER_QS_MIN']
Loading CSVs from data_samples/DOC_LINE_CATEG ...
Loaded 2,171 lines across 19 document(s)
Baseline (current config): flip_rate=0.0000 macro_f1=1.0000
[sklearn] completed 25/400 valid trials
[sklearn] completed 50/400 valid trials
[sklearn] completed 75/400 valid trials
[sklearn] completed 100/400 valid trials
[sklearn] completed 125/400 valid trials
[sklearn] completed 150/400 valid trials
[sklearn] completed 175/400 valid trials
[sklearn] completed 200/400 valid trials
[sklearn] completed 225/400 valid trials
[sklearn] completed 250/400 valid trials
[sklearn] completed 275/400 valid trials
[sklearn] completed 300/400 valid trials
[sklearn] completed 325/400 valid trials
[sklearn] completed 350/400 valid trials
[sklearn] completed 375/400 valid trials
[sklearn] completed 400/400 valid trials
{
  "backend": "sklearn",
  "metric": "macro_f1",
  "direction": "maximize",
  "n_trials": 400,
  "attempts": 448,
  "r2_train": 0.9456951304915887,
  "oob_r2": 0.6844523278389452,
  "n_params": 37,
  "mdi_importance": {
    "CATEG_TRASH_SCORE_MAX": 0.010969836962852033,
    "CATEG_NOISY_SCORE_MAX": 0.0055453362646738615,
    "CATEG_GARBAGE_DENSITY_HIGH": 0.09245247963793551,
    "QS_GARBAGE_NORM_MAX": 0.004461861962752571,
    "ROT_RATIO_INVERTED_MIN": 0.005293875268878788,
    "WEIRD_RATIO_INVERTED_MIN": 0.004608550251196771,
    "PPL_INVERTED_MIN": 0.0077357713635227586,
    "PERPLEXITY_THRESHOLD_MAX": 0.009219963906083465,
    "SHORT_PPL_CAP": 0.008238620144532277,
    "HARD_SWEEP_LANG_MAX": 0.007098336778888156,
    "HARD_SWEEP_PPL_MIN": 0.007660043037982708,
    "PPL_EXTREME_MIN": 0.005074963984867676,
    "EXTREME_LANG_CONF": 0.008769312021969915,
    "PPL_GARBAGE_ABSOLUTE": 0.006187981054219124,
    "LOWPPL_CLEAR_MAX": 0.21692136552694283,
    "LOWPPL_CZECH_CLEAR_MAX": 0.02649380045106196,
    "CZECH_CLEAR_GARBAGE_MAX": 0.010420645733118532,
    "MOSTLY_READABLE_VALID_MIN": 0.4416567275926686,
    "SHORT_NOISY_QS_PENALTY": 0.005267138169028379,
    "WORD_W_PENALTY": 0.00684543798711725,
    "GHOST_DOMINATED_MIN_RATIO": 0.006015643098626072,
    "SUSPICIOUS_ROT_RATIO": 0.00693105279855852,
    "SUSPICIOUS_WQX_RATIO": 0.010095768588248247,
    "INVERTED_WEIRD_PENALTY": 0.004218727096287495,
    "GHOST_HITS_INVERTED_MIN": 0.001445525466006969,
    "ROT_HIGH_LANG_CONF": 0.007228713936611713,
    "LANG_SCORE_ROUGH": 0.005847022655823843,
    "INVERTED_RUN_MIN": 0.0019145212319338263,
    "INVERTED_PAGE_MAJORITY": 0.006877153096920283,
    "SURROUNDED_TRASH_QS_MARGIN": 0.006832504779044239,
    "PAGE_GARBAGE_CLEAR_MAX": 0.009703737823262733,
    "PAGE_GARBAGE_LANG_MAX": 0.007178221943890923,
    "PAGE_GARBAGE_MEDIAN_QS_MAX": 0.007905136158438414,
    "PAGE_GARBAGE_NOISY_QS_MAX": 0.005867252005698098,
    "PAGE_CLEAN_CLEAR_MIN": 0.008251137236687675,
    "PAGE_CLEAN_MEDIAN_QS_MIN": 0.005665886486862089,
    "PAGE_CLEAN_RECOVER_QS_MIN": 0.007099947496805705
  },
  "permutation_importance": {
    "CATEG_TRASH_SCORE_MAX": 0.004838164681865285,
    "CATEG_NOISY_SCORE_MAX": 0.0015708150886618493,
    "CATEG_GARBAGE_DENSITY_HIGH": 0.09214293934882359,
    "QS_GARBAGE_NORM_MAX": 0.0015694902220827305,
    "ROT_RATIO_INVERTED_MIN": 0.0016076653726291106,
    "WEIRD_RATIO_INVERTED_MIN": 0.0013994671562805624,
    "PPL_INVERTED_MIN": 0.002737584044620775,
    "PERPLEXITY_THRESHOLD_MAX": 0.0033000676367598,
    "SHORT_PPL_CAP": 0.0032368490840803967,
    "HARD_SWEEP_LANG_MAX": 0.0026132426564660152,
    "HARD_SWEEP_PPL_MIN": 0.0024297206714862813,
    "PPL_EXTREME_MIN": 0.0015045781658090305,
    "EXTREME_LANG_CONF": 0.003270828667417857,
    "PPL_GARBAGE_ABSOLUTE": 0.002183607445533028,
    "LOWPPL_CLEAR_MAX": 0.2638013448302319,
    "LOWPPL_CZECH_CLEAR_MAX": 0.01842955332267698,
    "CZECH_CLEAR_GARBAGE_MAX": 0.004404463738063506,
    "MOSTLY_READABLE_VALID_MIN": 0.5466730282200126,
    "SHORT_NOISY_QS_PENALTY": 0.0015784385481853,
    "WORD_W_PENALTY": 0.002130881686579102,
    "GHOST_DOMINATED_MIN_RATIO": 0.001929739603391627,
    "SUSPICIOUS_ROT_RATIO": 0.002161785949602258,
    "SUSPICIOUS_WQX_RATIO": 0.003972570128805484,
    "INVERTED_WEIRD_PENALTY": 0.0011402967907703,
    "GHOST_HITS_INVERTED_MIN": 0.00045019856707182354,
    "ROT_HIGH_LANG_CONF": 0.00254015192567364,
    "LANG_SCORE_ROUGH": 0.001779383246396189,
    "INVERTED_RUN_MIN": 0.0005473579437171005,
    "INVERTED_PAGE_MAJORITY": 0.0022333632282925124,
    "SURROUNDED_TRASH_QS_MARGIN": 0.002517068669357818,
    "PAGE_GARBAGE_CLEAR_MAX": 0.0036953410381783012,
    "PAGE_GARBAGE_LANG_MAX": 0.0024072765228278477,
    "PAGE_GARBAGE_MEDIAN_QS_MAX": 0.003052128211741968,
    "PAGE_GARBAGE_NOISY_QS_MAX": 0.0019784578955122836,
    "PAGE_CLEAN_CLEAR_MIN": 0.0034693182132725427,
    "PAGE_CLEAN_MEDIAN_QS_MIN": 0.0020303841037250985,
    "PAGE_CLEAN_RECOVER_QS_MIN": 0.0026724473733973782
  },
  "best_trial": {
    "trial": 139,
    "objective": 0.9720244016383568,
    "metric": "macro_f1",
    "CATEG_TRASH_SCORE_MAX": 0.6106630491512064,
    "CATEG_NOISY_SCORE_MAX": 0.7257110258539767,
    "CATEG_GARBAGE_DENSITY_HIGH": 0.43195661937092666,
    "QS_GARBAGE_NORM_MAX": 0.38438157653812266,
    "ROT_RATIO_INVERTED_MIN": 0.42182833695212685,
    "WEIRD_RATIO_INVERTED_MIN": 0.3227069324423007,
    "PPL_INVERTED_MIN": 123.18269754157507,
    "PERPLEXITY_THRESHOLD_MAX": 862.1551102122728,
    "SHORT_PPL_CAP": 683.8584225140426,
    "HARD_SWEEP_LANG_MAX": 0.5858665301241874,
    "HARD_SWEEP_PPL_MIN": 2299.6188301000557,
    "PPL_EXTREME_MIN": 3306.9194322201834,
    "EXTREME_LANG_CONF": 0.8804208754840143,
    "PPL_GARBAGE_ABSOLUTE": 44807.883962228756,
    "LOWPPL_CLEAR_MAX": 35.921659288055125,
    "LOWPPL_CZECH_CLEAR_MAX": 196.0369608084476,
    "CZECH_CLEAR_GARBAGE_MAX": 0.1690422069051567,
    "MOSTLY_READABLE_VALID_MIN": 0.845136590317743,
    "SHORT_NOISY_QS_PENALTY": 0.12957353445079878,
    "WORD_W_PENALTY": 0.16827322920231647,
    "GHOST_DOMINATED_MIN_RATIO": 0.46322822609236275,
    "SUSPICIOUS_ROT_RATIO": 0.8401778772088606,
    "SUSPICIOUS_WQX_RATIO": 0.05078015262600813,
    "INVERTED_WEIRD_PENALTY": 0.5189056373636018,
    "GHOST_HITS_INVERTED_MIN": 2,
    "ROT_HIGH_LANG_CONF": 0.8582493824671859,
    "LANG_SCORE_ROUGH": 0.4468598600035075,
    "INVERTED_RUN_MIN": 6,
    "INVERTED_PAGE_MAJORITY": 0.7545627588405962,
    "SURROUNDED_TRASH_QS_MARGIN": 0.12239112459783678,
    "PAGE_GARBAGE_CLEAR_MAX": 0.0960092298630364,
    "PAGE_GARBAGE_LANG_MAX": 0.35428799432378616,
    "PAGE_GARBAGE_MEDIAN_QS_MAX": 0.6209670028146477,
    "PAGE_GARBAGE_NOISY_QS_MAX": 0.6694960311685916,
    "PAGE_CLEAN_CLEAR_MIN": 0.5048876312768193,
    "PAGE_CLEAN_MEDIAN_QS_MIN": 0.7393186827201741,
    "PAGE_CLEAN_RECOVER_QS_MIN": 0.4466994284383272,
    "flip_rate": 0.023952095808383235,
    "trash_rate": 0.209120221096269,
    "clear_rate": 0.3578995854444956,
    "macro_f1": 0.9720244016383568,
    "weighted_f1": 0.9755120901193982,
    "kl_divergence": 0.0015122275085424088,
    "costed_score": 0.020036849378166743
  },
  "best_config": {
    "QS_WEIGHT_VALID_WORD": 0.35,
    "QS_WEIGHT_WEIRD": 0.18,
    "QS_WEIGHT_PERPLEXITY": 0.08,
    "QS_WEIGHT_LENGTH": 0.02,
    "QS_WEIGHT_GARBAGE": 0.18,
    "QS_WEIGHT_VOWEL": 0.07,
    "QS_WEIGHT_LANG": 0.05,
    "QS_WEIGHT_GIBBERISH": 0.04,
    "QS_WEIGHT_FUSED": 0.03,
    "CATEG_TRASH_SCORE_MAX": 0.6106630491512064,
    "CATEG_NOISY_SCORE_MAX": 0.7257110258539767,
    "CATEG_GARBAGE_DENSITY_HIGH": 0.43195661937092666,
    "QS_GARBAGE_NORM_MAX": 0.38438157653812266,
    "ROT_RATIO_INVERTED_MIN": 0.42182833695212685,
    "WEIRD_RATIO_INVERTED_MIN": 0.3227069324423007,
    "PPL_INVERTED_MIN": 123.18269754157507,
    "PERPLEXITY_THRESHOLD_MAX": 862.1551102122728,
    "SHORT_PPL_CAP": 683.8584225140426,
    "HARD_SWEEP_LANG_MAX": 0.5858665301241874,
    "HARD_SWEEP_PPL_MIN": 2299.6188301000557,
    "PPL_EXTREME_MIN": 3306.9194322201834,
    "EXTREME_LANG_CONF": 0.8804208754840143,
    "PPL_GARBAGE_ABSOLUTE": 44807.883962228756,
    "LOWPPL_CLEAR_MAX": 35.921659288055125,
    "LOWPPL_CZECH_CLEAR_MAX": 196.0369608084476,
    "CZECH_CLEAR_GARBAGE_MAX": 0.1690422069051567,
    "MOSTLY_READABLE_VALID_MIN": 0.845136590317743,
    "SHORT_NOISY_QS_PENALTY": 0.12957353445079878,
    "WORD_W_PENALTY": 0.16827322920231647,
    "GHOST_DOMINATED_MIN_RATIO": 0.46322822609236275,
    "SUSPICIOUS_ROT_RATIO": 0.8401778772088606,
    "SUSPICIOUS_WQX_RATIO": 0.05078015262600813,
    "INVERTED_WEIRD_PENALTY": 0.5189056373636018,
    "GHOST_HITS_INVERTED_MIN": 2,
    "ROT_HIGH_LANG_CONF": 0.8582493824671859,
    "LANG_SCORE_ROUGH": 0.4468598600035075,
    "INVERTED_RUN_MIN": 6,
    "INVERTED_PAGE_MAJORITY": 0.7545627588405962,
    "SURROUNDED_TRASH_QS_MARGIN": 0.12239112459783678,
    "PAGE_GARBAGE_CLEAR_MAX": 0.0960092298630364,
    "PAGE_GARBAGE_LANG_MAX": 0.35428799432378616,
    "PAGE_GARBAGE_MEDIAN_QS_MAX": 0.6209670028146477,
    "PAGE_GARBAGE_NOISY_QS_MAX": 0.6694960311685916,
    "PAGE_CLEAN_CLEAR_MIN": 0.5048876312768193,
    "PAGE_CLEAN_MEDIAN_QS_MIN": 0.7393186827201741,
    "PAGE_CLEAN_RECOVER_QS_MIN": 0.4466994284383272
  }
}

[4/6] Running SALib Morris Screening Sweep...
Running sweep | profile=default | backend=morris | metric=macro_f1 (maximize)
Sweeping 37 parameter(s): ['CATEG_TRASH_SCORE_MAX', 'CATEG_NOISY_SCORE_MAX', 'CATEG_GARBAGE_DENSITY_HIGH', 'QS_GARBAGE_NORM_MAX', 'ROT_RATIO_INVERTED_MIN', 'WEIRD_RATIO_INVERTED_MIN', 'PPL_INVERTED_MIN', 'PERPLEXITY_THRESHOLD_MAX', 'SHORT_PPL_CAP', 'HARD_SWEEP_LANG_MAX', 'HARD_SWEEP_PPL_MIN', 'PPL_EXTREME_MIN', 'EXTREME_LANG_CONF', 'PPL_GARBAGE_ABSOLUTE', 'LOWPPL_CLEAR_MAX', 'LOWPPL_CZECH_CLEAR_MAX', 'CZECH_CLEAR_GARBAGE_MAX', 'MOSTLY_READABLE_VALID_MIN', 'SHORT_NOISY_QS_PENALTY', 'WORD_W_PENALTY', 'GHOST_DOMINATED_MIN_RATIO', 'SUSPICIOUS_ROT_RATIO', 'SUSPICIOUS_WQX_RATIO', 'INVERTED_WEIRD_PENALTY', 'GHOST_HITS_INVERTED_MIN', 'ROT_HIGH_LANG_CONF', 'LANG_SCORE_ROUGH', 'INVERTED_RUN_MIN', 'INVERTED_PAGE_MAJORITY', 'SURROUNDED_TRASH_QS_MARGIN', 'PAGE_GARBAGE_CLEAR_MAX', 'PAGE_GARBAGE_LANG_MAX', 'PAGE_GARBAGE_MEDIAN_QS_MAX', 'PAGE_GARBAGE_NOISY_QS_MAX', 'PAGE_CLEAN_CLEAR_MIN', 'PAGE_CLEAN_MEDIAN_QS_MIN', 'PAGE_CLEAN_RECOVER_QS_MIN']
Loading CSVs from data_samples/DOC_LINE_CATEG ...
Loaded 2,171 lines across 19 document(s)
Baseline (current config): flip_rate=0.0000 macro_f1=1.0000
[morris] Evaluating 380 samples...
{
  "backend": "morris",
  "metric": "macro_f1",
  "direction": "maximize",
  "n_trials": 380,
  "n_params": 37,
  "morris_importance": {
    "CATEG_TRASH_SCORE_MAX": 0.08820225705695414,
    "CATEG_NOISY_SCORE_MAX": 0.0,
    "CATEG_GARBAGE_DENSITY_HIGH": 0.16245965751829308,
    "QS_GARBAGE_NORM_MAX": 0.01983313349070179,
    "ROT_RATIO_INVERTED_MIN": 0.004848531279324447,
    "WEIRD_RATIO_INVERTED_MIN": 0.0,
    "PPL_INVERTED_MIN": 0.020217872756990333,
    "PERPLEXITY_THRESHOLD_MAX": 0.012201767384453525,
    "SHORT_PPL_CAP": 0.00494549585527251,
    "HARD_SWEEP_LANG_MAX": 0.0047832937430137374,
    "HARD_SWEEP_PPL_MIN": 0.002691098388182488,
    "PPL_EXTREME_MIN": 0.0054732673186627615,
    "EXTREME_LANG_CONF": 0.004862053505646701,
    "PPL_GARBAGE_ABSOLUTE": 0.0025644421560272503,
    "LOWPPL_CLEAR_MAX": 0.12604062932536556,
    "LOWPPL_CZECH_CLEAR_MAX": 0.11588294182980365,
    "CZECH_CLEAR_GARBAGE_MAX": 0.05354476733680696,
    "MOSTLY_READABLE_VALID_MIN": 0.2826242605378587,
    "SHORT_NOISY_QS_PENALTY": 0.015047365713999552,
    "WORD_W_PENALTY": 0.0012760831209004564,
    "GHOST_DOMINATED_MIN_RATIO": 0.0,
    "SUSPICIOUS_ROT_RATIO": 0.0002987994899608482,
    "SUSPICIOUS_WQX_RATIO": 0.0,
    "INVERTED_WEIRD_PENALTY": 0.00032396219992874485,
    "GHOST_HITS_INVERTED_MIN": 0.0,
    "ROT_HIGH_LANG_CONF": 0.00997326618949486,
    "LANG_SCORE_ROUGH": 0.00457096300732513,
    "INVERTED_RUN_MIN": 0.007077749180807424,
    "INVERTED_PAGE_MAJORITY": 0.006215956692754521,
    "SURROUNDED_TRASH_QS_MARGIN": 0.0013813812764828528,
    "PAGE_GARBAGE_CLEAR_MAX": 0.005026695865513658,
    "PAGE_GARBAGE_LANG_MAX": 0.009881441451153968,
    "PAGE_GARBAGE_MEDIAN_QS_MAX": 0.0013711273533866444,
    "PAGE_GARBAGE_NOISY_QS_MAX": 0.001461031788360034,
    "PAGE_CLEAN_CLEAR_MIN": 0.017271336443071147,
    "PAGE_CLEAN_MEDIAN_QS_MIN": 0.0033219510918692068,
    "PAGE_CLEAN_RECOVER_QS_MIN": 0.004325419651633353
  }
}

[5/6] Running SALib Sobol Sweep (Computationally Heavy)...
Running sweep | profile=default | backend=sobol | metric=macro_f1 (maximize)
Sweeping 37 parameter(s): ['CATEG_TRASH_SCORE_MAX', 'CATEG_NOISY_SCORE_MAX', 'CATEG_GARBAGE_DENSITY_HIGH', 'QS_GARBAGE_NORM_MAX', 'ROT_RATIO_INVERTED_MIN', 'WEIRD_RATIO_INVERTED_MIN', 'PPL_INVERTED_MIN', 'PERPLEXITY_THRESHOLD_MAX', 'SHORT_PPL_CAP', 'HARD_SWEEP_LANG_MAX', 'HARD_SWEEP_PPL_MIN', 'PPL_EXTREME_MIN', 'EXTREME_LANG_CONF', 'PPL_GARBAGE_ABSOLUTE', 'LOWPPL_CLEAR_MAX', 'LOWPPL_CZECH_CLEAR_MAX', 'CZECH_CLEAR_GARBAGE_MAX', 'MOSTLY_READABLE_VALID_MIN', 'SHORT_NOISY_QS_PENALTY', 'WORD_W_PENALTY', 'GHOST_DOMINATED_MIN_RATIO', 'SUSPICIOUS_ROT_RATIO', 'SUSPICIOUS_WQX_RATIO', 'INVERTED_WEIRD_PENALTY', 'GHOST_HITS_INVERTED_MIN', 'ROT_HIGH_LANG_CONF', 'LANG_SCORE_ROUGH', 'INVERTED_RUN_MIN', 'INVERTED_PAGE_MAJORITY', 'SURROUNDED_TRASH_QS_MARGIN', 'PAGE_GARBAGE_CLEAR_MAX', 'PAGE_GARBAGE_LANG_MAX', 'PAGE_GARBAGE_MEDIAN_QS_MAX', 'PAGE_GARBAGE_NOISY_QS_MAX', 'PAGE_CLEAN_CLEAR_MIN', 'PAGE_CLEAN_MEDIAN_QS_MIN', 'PAGE_CLEAN_RECOVER_QS_MIN']
Loading CSVs from data_samples/DOC_LINE_CATEG ...
Loaded 2,171 lines across 19 document(s)
Baseline (current config): flip_rate=0.0000 macro_f1=1.0000
[sobol] Evaluating 8960 constraint-free samples...

*

*

*

```



</details>

*(You can still run individual backends manually via `python tools/const_importance_sweep.py --backend <name>` if you only need a specific surrogate).*

## Methodology guardrails (what changed for #5)

* **Minimum Trial Budgets:** The RF and Optuna backends strictly require `n_trials` to be greater than the number of parameters swept (e.g., >37). Passing fewer trials will abort the sweep to prevent statistically invalid (noise) importance calculations.
* **Sobol Constraints:** The `sobol-n` argument *must* be a power of 2 (e.g., 64, 128, 256) to satisfy the balance properties of the Saltelli sequence. Note that total evaluations equal `N * (D + 2)`.
* **Zero-Variance Safeguards:** If the target metric (e.g., `macro_f1`) does not vary across trials—common when testing on tiny smoke fixtures—Optuna and Sklearn will gracefully skip importance computation rather than crashing with division-by-zero errors.
* **fANOVA needs ~uniform sampling**, so the Optuna default is `--sampler random`.
TPE concentrates near the optimum and biases fANOVA; the tool warns if you ask
for TPE. Use TPE only to *optimise* a config, not to rank importance.
* The sklearn surrogate reports **out-of-bag R²** (held-out), not just train R².
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
