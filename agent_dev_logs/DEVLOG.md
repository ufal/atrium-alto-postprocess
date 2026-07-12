# 📓 atrium-alto-postprocess — agent_dev_logs/DEVLOG.md (timeline index)
> _OCR/ALTO post-processing + line categorization. 5 open issues (#2–#6). `test` HEAD `6882857` (2026-07-12) · **v0.20.2**._
> _Per-issue detail: `digests/{id}.digest.md` · `plans/{id}.plan.md` · `issues/` exports (source of truth). Cross-repo/hub history lives in `ufal/atrium-project/agent_dev_logs/DEVLOG.md` (deduplicated out of this file)._

## 2026-03-13
- **#2 Update text-category definitions & logic** — Opened by K4TEL: add regex for digits-fused-to-letters and symbols-inside-words; consider ignoring poorly-scored language during category assignment.

## 2026-03-14
- **#2** — Commit `884316e` first attempt; updated result files included.

## 2026-03-19
- **#2** — motyc shared Dana & Tomáš's expert-reviewed CSV plus a ChatGPT analysis: five categories (Clear/Noisy/Trash/Non-text/Empty), recoverability as the core axis, key features (symbol ratio, valid-word ratio, perplexity), starter thresholds, a weighted scoring formula, and an optional decision-tree model.

## 2026-03-20
- **#2** — Commit `391d574` refines the algorithm; documentation updated; full ARUP & ARUB categorization launched.

## 2026-03-22
- **#2** — Commit `5b198e2` adds average-quality-score and ww-ratio columns to the summary files.

## 2026-04-01
- **#2** — Main work moved to the `test` branch; fixing `problems_260327.csv`.

## 2026-04-02
- **#2** — Commit `5bb8b2b` (and prior) attempt a fix; calibration flagged as needed.

## 2026-04-17
- **#2** — Posted the result-file column reference.
- **#3 Calibration of categorization logic** — Opened by K4TEL: define safe European languages (else "Noisy" if the page is mostly clean-language), use the 0–1 quality score as a real per-category decision factor (e.g. Trash 0.0–0.3 / Noisy 0.3–0.9 / Clear 0.9–1.0); current v0.13.0 logic copied from the README.

## 2026-04-20
- **#3** — DanaKriv: language-ID failures — Czech `sonda 9` detected "tur" → wrongly Noisy; trash `VX5P3SosAX` detected "vie" → wrongly Clear; suggested treating any language outside eng/deu/fra/pol as suspect.

## 2026-04-21
- **#3** — Commit `28650e2` adds `TRUSTED_FOREIGN_LANGS` remapping of untrusted FastText results to Czech (so clean short Czech phrases hit the perplexity-forgiveness path → Clear); commit `2969295` updates samples; full ARUP/ARUB collections to be shared.

## 2026-04-28
- **#3** — DanaKriv feedback: drop the "not in EXPECTED_LANGS and confidence < 0.60" penalty (fix lang confidence to ~0.5?); make the quality score the actual category decision; carefully verify columns on `CTX193001369`.

## 2026-04-29
- **#3** — Quality score is now an explicit weighted sum (valid-word 30%, symbol density 20%, weirdness 20%, perplexity 20%, length 10%); strict thresholds <0.40 → Trash, <0.70 → Noisy, else Clear.

## 2026-05-01
- **#3** — Commit `77b5c75` (v0.14.0) result samples; CPU pre-filter assigns Empty / Non-text before any ML (≥4 chars, ≥30% letters, not digit/symbol-dominated).

## 2026-05-02
- **#3** — Configurable perplexity model via `MODEL_NAME` (Qwen/Qwen2.5-0.5B); commit `a29f420` posts both Qwen- and GPT-based results (v0.15.0).

## 2026-05-03
- **#3** — Clarified that basing the category **solely** on the quality score is impossible; instead the QS ranges are matched to the manually-assigned Trash/Noisy/Clear categories.

## 2026-05-04
- **#3** — Error analysis: upside-down/mirror-scanned text wrongly Clear/Noisy (should be Trash), clean short Czech wrongly downgraded to Noisy; commit `0f90477` swaps distilgpt2 → Qwen2.5-0.5B and re-tunes perplexity thresholds (`PERPLEXITY_THRESHOLD_MAX` 5000→1000, etc.).

## 2026-05-05
- **#3** — v0.15.1 samples; removed `_` from `ALLOWED_INTERNAL` (it was letting garbled tokens like `b/eralowýřt_` score as valid and reach Clear); kept `/` for `km/h`-style abbreviations.

## 2026-05-07
- **#3** — Commit `646fc5e`: language trust is now encoded **inside** the score — `lang_score` becomes a weighted component of `quality_score` rather than a standalone external penalty.

## 2026-05-09
- **#3** — Commit `53a6faa`: further `config_langID.txt` + `text_util_langID.py` changes.

## 2026-05-11
- **#3** — v0.15.2 demo samples; collection-level results shared via Filesender.

## 2026-05-14
- **#3** — DanaKriv: regression — missing spaces between words when reading ALTO (e.g. `obilostkslužběskonečnou`), affecting some categories; but the Clear category improved (less trash leaking in).

## 2026-05-15
- **#3** — Commit `fddbafd` fixes the space-collapsing bug (an aggressive OCR word-split regex in `pre_filter_line` was merging single-letter Czech prepositions).

## 2026-05-18
- **#3** — Posted the v0.15.3 categorization-logic diagram.

## 2026-05-27
- **#3** — Released v0.15.4 with unit tests on main.

## 2026-05-28
- **#3** — Requested review of shared data + relating computed factors to the QS to define/edit/remove rules.
- **#4 Documentation of categorization logic** — Opened by K4TEL (structured README, edge-case unit tests, document overrides). Commit `eeb4e7a` extends the README; motyc: don't close issues until follow-ups are solved; keep #4 open until the README is confirmed fine.

## 2026-06-02
- **#3** — DanaKriv TODO: fix the Lang / Lang_score columns first (`deu` wrongly changed to `ces` on `CTX192900489`; non-trusted languages should score 0.75) — no further analysis until the data is trustworthy.

## 2026-06-03
- **#3** — `quality_score_exceptions.txt` posted; `CTX192900489` + `CTX192100040` reserved as next-version test cases; multi-character symbols now ignored; `deu`→`ces` mis-mapping addressed.

## 2026-06-15
- **#3** — Opus Max review: Task 2 (QS drives the category) done & released in v0.15.4 — pure threshold routing on `quality_score` + 3 structural overrides replaced the old ~150-line penalty cascade.

## 2026-06-16
- **#3** — Sample files updated; v0.18.0 (changes A & B); DanaKriv: the updated samples look weird; K4TEL: a test let bad output through, fix coming tomorrow.

## 2026-06-17
- **#3** — Corrected `CTX192100040` / `CTX192900489` CSVs for the meeting; the Qwen 2.5 pipeline had to run on the cluster GPU.

## 2026-06-18
- **#3** — DanaKriv posted meeting notes; post-meeting column changes (`categ` + `quality_score` first; add `original_lang`/`orig_lang_score`); failing-test report + options; updated samples (commits `5672d0e`, `f0a8a3d`).

## 2026-06-19
- **#3** — Commits `4dfc084` / `85fd4b9`: the rotation/inversion trap is fixed (decoupled `rot_ratio` from weirdness); Gemini 3.1 Pro CSV review; **v0.19.0** major update — rotation moved out of `compute_quality_score()` into a lexicon-based per-line override + an expanded page-level sweep, two immediate-Trash overrides, one diagnostic column.
- **#4** — README updated to match (cross-linked from #3).

## 2026-06-20
- **#3** — Post-v0.19.0: four new boundary thresholds (`PPL_EXTREME_MIN`, `EXTREME_LANG_CONF`, `LOWPPL_CZECH_CLEAR_MAX`, `CZECH_CLEAR_GARBAGE_MAX`); a few questionable `CTX192601143` cases remain.

## 2026-06-21
- **#4** — Commit `5868b0f` refines the current-state logic description across all markdown files.

## 2026-06-22
- **#3** — Released **v0.19.2** (technical fixes, logic unchanged); DanaKriv: don't send the whole collection until samples are approved, and the lang_score 0.5/0.75 variants are missing — "keep agreements!"; K4TEL provided the 0.5 versions; the `tools/recategorize_from_csv.py` helper (with `--help`) runs after config edits.

## 2026-06-24
- **#3** — Cross-referenced #5 for the parameter-set analysis.
- **#5 Small model for config-constant importance** — Opened by K4TEL: a surrogate over the `[CLASSIFY]`/`[TEXT_UTILS]` bool/int/float constants, with immutable per-line CSVs as ground truth and `recategorize_from_csv.py` as the entry point. Many comments same day: tooling design (`recategorize_from_csv.py` + `const_importance_sweep.py`), sklearn/Optuna sweeps (400→1000→2000 trials), an ablation study + `greedy_backward_elimination.py`, the coverage-vs-marginal-effect insight, the "delete all 15 rules" result identified as a **metric artifact** (refined to 9 droppable rules / 6 load-bearing), plus GPT-5 and Opus 4.8 cross-reviews; commit `f9b5e35`.
- **#6 Starting points in the pipeline run script** — Opened by K4TEL (skip flags, mainly skip-text-extraction); commit `e09fa9b` auto-tested implementation.

## 2026-06-25
- **#5** — Commit `b4bd545`: parameterized eight page-context smoothing thresholds (config-driven), removed the dead `rule_short_fragment_noisy` + `CLEAR_BAND_WC_MIN`, and the legacy `CLEAN_PROSE_*` near-boundary constants; new sweep output posted; further reviews from GPT-5.5, GPT-5 and Gemini DR Pro 3.1 (the "Survivor Bias" framing, refactoring direction). Sweep search space now 36 parameters.

## 2026-06-28
- **#3 / #5** — Parameter-study coverage report relayed into the calibration loop (`n_scored=1,463`): **11 LOAD-BEARING · 3 REDUNDANT-HERE · 0 DEAD** — every rule fired at least once. Redundant-here (fire but never decisive; kept, entanglement suspected): `rule_allcaps`, `rule_garbage_density`, `rule_inverted`. Cross-backend-robust parameters, ranked: `MOSTLY_READABLE_VALID_MIN`, `LOWPPL_CLEAR_MAX`, `CATEG_GARBAGE_DENSITY_HIGH`, `LOWPPL_CZECH_CLEAR_MAX`, `CZECH_CLEAR_GARBAGE_MAX`.

## 2026-07-02
- **#3** — DanaKriv calibration meeting (verified against `CTX199603106`, a Charles-Bridge archaeological report), five findings: ① short domain abbreviations (`mm`, `Tb.`, `č.neg.`) Trash/Non-text → **Noisy**; ② short numbered headlines/captions (`4. Literatura 5`, `Plánek č. 1`) Trash → **Noisy**; ③ all-caps single-word headlines (`LITERATURA`) scored normally instead of Non-text; ④ Noisy↔Clear boundary **0.85 → 0.80**; ⑤ remapped lang score **always** 0.75/0.5 ("the original lang score should not matter"). All five **committed**: `ffcfa48` (post-meeting logic edits, incl. the `is_forgiven_headline` rescue) + `440a066` (`LANG_REMAP_ALWAYS` config switch).

## 2026-07-03
- **#3** — Calibration pass **shipped**: merged `test` → `main`, released **v0.20.1**; `data_samples/` result CSVs re-baselined to the new logic (`6acbd94`); collection-wise **ARUP + ARUB result archives** (`307` suffix) shared via FileSender for the **final review round**.
- **#4** — README documents the post-pass logic (0.80 boundary, forgiven-headline rescue, `LANG_REMAP_ALWAYS` switch); a full annotator feedback cycle (07-02 review → v0.20.1) has now completed. Issue stays open pending team confirmation + the final review round's outcome.

## 2026-07-12
- Repo at **v0.20.2** — dependency bumps, shared `tests/test_para_licenses.py` per the hub template, fixed automatic version reading (`_read_tool_version()`); license-parity drift enforcement now default via the renamed hub `para-drift.reusable.yml`. No categorization-logic change.
- Digests/plans refreshed against the issue exports: **#2** `CATEG_NOISY_SCORE_MAX` corrected to **0.80**; **#3/#4** updated from the stale "pending push" premise to the shipped v0.20.1 reality. Open #3 cases queued for a Czech-speaker check: vowelless illusion (`WVL A` as Clear), ledger/table loophole (fragmented number lines ≈0.85), symbol cluster (`At . O/wvi` at Noisy). NEXT: fold the final ARUP/ARUB verdicts into the next pass (or close); **#6** stays open solely for practical full-collection verification of the skip/`--start-from` flags.

---
_Timeline index refreshed 2026-07-12 against `test` HEAD and the refreshed digests/plans. Nothing removed from the issues themselves (per hub #29); this file is a derived reading aid in `agent_dev_logs/`._
