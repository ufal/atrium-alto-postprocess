# 📓 atrium-alto-postprocess — agent_dev_logs/DEVLOG.md (timeline index)
> _OCR/ALTO post-processing + line categorization. 7 open issues (#2, #3, #4, #23, #30, #31, #37); #5/#6 closed. **v1.5.1-beta** released 2026-09-22 at `09c9640`. `test` = `master` = `2e2794d` (2026-09-24, CI green): the post-tag #30 work (`5b27900`, `0c30517`, `3b02959`, `858be91`), #31's `--method text-lines` (`103e30a`), its TEITOK follow-ups (`fb72526`) and #31 Phase 4 (`267e334`, `3bd10f9`, `2e2794d`), all unreleased. Next tag needs the version bump in `CITATION.cff` + `setup/para_config.txt`. #37 is ready to close._
> _Per-issue detail: `digests/{id}.digest.md` · `plans/{id}.plan.md` · `issues/` exports (source of truth). Cross-repo/hub history lives in `ufal/atrium-project/agent_dev_logs/DEVLOG.md` (deduplicated out of this file)._

## 2026-03-13
- **#2 Update text-category definitions & logic** — Opened by K4TEL: add regex for digits-fused-to-letters and
symbols-inside-words; consider ignoring poorly-scored language during category assignment.

## 2026-03-14
- **#2** — Commit `884316e` first attempt; updated result files included.

## 2026-03-19
- **#2** — motyc shared Dana & Tomáš's expert-reviewed CSV plus a ChatGPT analysis: five categories
(Clear/Noisy/Trash/Non-text/Empty), recoverability as the core axis, key features (symbol ratio, valid-word ratio, perplexity),
starter thresholds, a weighted scoring formula, and an optional decision-tree model.

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
- **#3 Calibration of categorization logic** — Opened by K4TEL: define safe European languages (else "Noisy" if
the page is mostly clean-language), use the 0–1 quality score as a real per-category decision factor (e.g.
Trash 0.0–0.3 / Noisy 0.3–0.9 / Clear 0.9–1.0); current v0.13.0 logic copied from the README.

## 2026-04-20
- **#3** — DanaKriv: language-ID failures — Czech `sonda 9` detected "tur" → wrongly Noisy; trash `VX5P3SosAX`
detected "vie" → wrongly Clear; suggested treating any language outside eng/deu/fra/pol as suspect.

## 2026-04-21
- **#3** — Commit `28650e2` adds `TRUSTED_FOREIGN_LANGS` remapping of untrusted FastText results to Czech (so clean
short Czech phrases hit the perplexity-forgiveness path → Clear); commit `2969295` updates samples; full ARUP/ARUB collections to be shared.

## 2026-04-28
- **#3** — DanaKriv feedback: drop the "not in EXPECTED_LANGS and confidence < 0.60" penalty (fix lang confidence to
~0.5?); make the quality score the actual category decision; carefully verify columns on `CTX193001369`.

## 2026-04-29
- **#3** — Quality score is now an explicit weighted sum (valid-word 30%, symbol density 20%, weirdness 20%, perplexity
20%, length 10%); strict thresholds <0.40 → Trash, <0.70 → Noisy, else Clear.

## 2026-05-01
- **#3** — Commit `77b5c75` (v0.14.0) result samples; CPU pre-filter assigns Empty / Non-text before any ML (≥4 chars,
≥30% letters, not digit/symbol-dominated).

## 2026-05-02
- **#3** — Configurable perplexity model via `MODEL_NAME` (Qwen/Qwen2.5-0.5B); commit `a29f420` posts both Qwen-
and GPT-based results (v0.15.0).

## 2026-05-03
- **#3** — Clarified that basing the category **solely** on the quality score is impossible; instead the QS ranges
are matched to the manually-assigned Trash/Noisy/Clear categories.

## 2026-05-04
- **#3** — Error analysis: upside-down/mirror-scanned text wrongly Clear/Noisy (should be Trash), clean short
Czech wrongly downgraded to Noisy; commit `0f90477` swaps distilgpt2 → Qwen2.5-0.5B and re-tunes perplexity
thresholds (`PERPLEXITY_THRESHOLD_MAX` 5000→1000, etc.).

## 2026-05-05
- **#3** — v0.15.1 samples; removed `_` from `ALLOWED_INTERNAL` (it was letting garbled tokens like `b/eralowýřt_`
score as valid and reach Clear); kept `/` for `km/h`-style abbreviations.

## 2026-05-07
- **#3** — Commit `646fc5e`: language trust is now encoded **inside** the score — `lang_score` becomes a
weighted component of `quality_score` rather than a standalone external penalty.

## 2026-05-09
- **#3** — Commit `53a6faa`: further `config_langID.txt` + `text_util_langID.py` changes.

## 2026-05-11
- **#3** — v0.15.2 demo samples; collection-level results shared via Filesender.

## 2026-05-14
- **#3** — DanaKriv: regression — missing spaces between words when reading ALTO (e.g. `obilostkslužběskonečnou`),
affecting some categories; but the Clear category improved (less trash leaking in).

## 2026-05-15
- **#3** — Commit `fddbafd` fixes the space-collapsing bug (an aggressive OCR word-split regex in `pre_filter_line`
was merging single-letter Czech prepositions).

## 2026-05-18
- **#3** — Posted the v0.15.3 categorization-logic diagram.

## 2026-05-27
- **#3** — Released v0.15.4 with unit tests on main.

## 2026-05-28
- **#3** — Requested review of shared data + relating computed factors to the QS to define/edit/remove rules.
- **#4 Documentation of categorization logic** — Opened by K4TEL (structured README, edge-case unit tests,
document overrides). Commit `eeb4e7a` extends the README; motyc: don't close issues until follow-ups are solved;
keep #4 open until the README is confirmed fine.

## 2026-06-02
- **#3** — DanaKriv TODO: fix the Lang / Lang_score columns first (`deu` wrongly changed to `ces` on `CTX192900489`;
non-trusted languages should score 0.75) — no further analysis until the data is trustworthy.

## 2026-06-03
- **#3** — `quality_score_exceptions.txt` posted; `CTX192900489` + `CTX192100040` reserved as next-version test
cases; multi-character symbols now ignored; `deu`→`ces` mis-mapping addressed.

## 2026-06-15
- **#3** — Opus Max review: Task 2 (QS drives the category) done & released in v0.15.4 — pure threshold routing
on `quality_score` + 3 structural overrides replaced the old ~150-line penalty cascade.

## 2026-06-16
- **#3** — Sample files updated; v0.18.0 (changes A & B); DanaKriv: the updated samples look weird; K4TEL: a test let
bad output through, fix coming tomorrow.

## 2026-06-17
- **#3** — Corrected `CTX192100040` / `CTX192900489` CSVs for the meeting; the Qwen 2.5 pipeline had to run on the cluster GPU.

## 2026-06-18
- **#3** — DanaKriv posted meeting notes; post-meeting column changes (`categ` + `quality_score` first; add
`original_lang`/`orig_lang_score`); failing-test report + options; updated samples (commits `5672d0e`, `f0a8a3d`).

## 2026-06-19
- **#3** — Commits `4dfc084` / `85fd4b9`: the rotation/inversion trap is fixed (decoupled `rot_ratio` from weirdness);
Gemini 3.1 Pro CSV review; **v0.19.0** major update — rotation moved out of `compute_quality_score()` into a
lexicon-based per-line override + an expanded page-level sweep, two immediate-Trash overrides, one diagnostic column.
- **#4** — README updated to match (cross-linked from #3).

## 2026-06-20
- **#3** — Post-v0.19.0: four new boundary thresholds (`PPL_EXTREME_MIN`, `EXTREME_LANG_CONF`, `LOWPPL_CZECH_CLEAR_MAX`,
`CZECH_CLEAR_GARBAGE_MAX`); a few questionable `CTX192601143` cases remain.

## 2026-06-21
- **#4** — Commit `5868b0f` refines the current-state logic description across all markdown files.

## 2026-06-22
- **#3** — Released **v0.19.2** (technical fixes, logic unchanged); DanaKriv: don't send the whole collection until
samples are approved, and the lang_score 0.5/0.75 variants are missing — "keep agreements!"; K4TEL provided the 0.5
versions; the `tools/recategorize_from_csv.py` helper (with `--help`) runs after config edits.

## 2026-06-24
- **#3** — Cross-referenced #5 for the parameter-set analysis.
- **#5 Small model for config-constant importance** — Opened by K4TEL: a surrogate over the `[CLASSIFY]`/`[TEXT_UTILS]`
bool/int/float constants, with immutable per-line CSVs as ground truth and `recategorize_from_csv.py` as the entry point.
Many comments same day: tooling design (`recategorize_from_csv.py` + `const_importance_sweep.py`), sklearn/Optuna sweeps
(400→1000→2000 trials), an ablation study + `greedy_backward_elimination.py`, the coverage-vs-marginal-effect insight,
the "delete all 15 rules" result identified as a **metric artifact** (refined to 9 droppable rules / 6 load-bearing),
plus GPT-5 and Opus 4.8 cross-reviews; commit `f9b5e35`.
- **#6 Starting points in the pipeline run script** — Opened by K4TEL (skip flags, mainly skip-text-extraction);
commit `e09fa9b` auto-tested implementation.

## 2026-06-25
- **#5** — Commit `b4bd545`: parameterized eight page-context smoothing thresholds (config-driven), removed the dead
`rule_short_fragment_noisy` + `CLEAR_BAND_WC_MIN`, and the legacy `CLEAN_PROSE_*` near-boundary constants; new sweep
output posted; further reviews from GPT-5.5, GPT-5 and Gemini DR Pro 3.1 (the "Survivor Bias" framing, refactoring
direction). Sweep search space now 36 parameters.

## 2026-06-28
- **#3 / #5** — Parameter-study coverage report relayed into the calibration loop (`n_scored=1,463`): **11 LOAD-BEARING
· 3 REDUNDANT-HERE · 0 DEAD** — every rule fired at least once. Redundant-here (fire but never decisive; kept,
entanglement suspected): `rule_allcaps`, `rule_garbage_density`, `rule_inverted`. Cross-backend-robust parameters,
ranked: `MOSTLY_READABLE_VALID_MIN`, `LOWPPL_CLEAR_MAX`, `CATEG_GARBAGE_DENSITY_HIGH`, `LOWPPL_CZECH_CLEAR_MAX`, `CZECH_CLEAR_GARBAGE_MAX`.

## 2026-07-02
- **#3** — DanaKriv calibration meeting (verified against `CTX199603106`, a Charles-Bridge archaeological report),
five findings: ① short domain abbreviations (`mm`, `Tb.`, `č.neg.`) Trash/Non-text → **Noisy**; ② short numbered
headlines/captions (`4. Literatura 5`, `Plánek č. 1`) Trash → **Noisy**; ③ all-caps single-word headlines (`LITERATURA`)
scored normally instead of Non-text; ④ Noisy↔Clear boundary **0.85 → 0.80**; ⑤ remapped lang score **always** 0.75/0.5
("the original lang score should not matter"). All five **committed**: `ffcfa48` (post-meeting logic edits, incl. the
`is_forgiven_headline` rescue) + `440a066` (`LANG_REMAP_ALWAYS` config switch).

## 2026-07-03
- **#3** — Calibration pass **shipped**: merged `test` → `main`, released **v0.20.1**; `data_samples/` result CSVs
re-baselined to the new logic (`6acbd94`); collection-wise **ARUP + ARUB result archives** (`307` suffix) shared
via FileSender for the **final review round**.
- **#4** — README documents the post-pass logic (0.80 boundary, forgiven-headline rescue, `LANG_REMAP_ALWAYS` switch);
a full annotator feedback cycle (07-02 review → v0.20.1) has now completed. Issue stays open pending team confirmation
+ the final review round's outcome.

## 2026-07-12
- Repo at **v0.20.2** — dependency bumps, shared `tests/test_para_licenses.py` per the hub template, fixed automatic
version reading (`_read_tool_version()`); license-parity drift enforcement now default via the renamed hub
`para-drift.reusable.yml`. No categorization-logic change.
- Digests/plans refreshed against the issue exports: **#2** `CATEG_NOISY_SCORE_MAX` corrected to **0.80**; **#3/#4**
updated from the stale "pending push" premise to the shipped v0.20.1 reality. Open #3 cases queued for a Czech-speaker
check: vowelless illusion (`WVL A` as Clear), ledger/table loophole (fragmented number lines ≈0.85), symbol cluster
(`At . O/wvi` at Noisy). NEXT: fold the final ARUP/ARUB verdicts into the next pass (or close); **#6** stays open
solely for practical full-collection verification of the skip/`--start-from` flags.

## 2026-07-17

* **#23 Consider finetuning NLP model for quality score assignement** — Opened by K4TEL: Proposes training a model
on the regression task (score 0.0-1.0) using FastText and Qwen 0.5B gold data. Existing data can be utilized by
manipulating `Clear` lines, auto-correcting `Noisy` lines, and using `Trash` and `Not-Text` lines as is. Empty
lines will be ignored. The goal is to produce a fine-tuned model smaller in size than the currently used Qwen 0.5B.

## 2026-07-22

* **#23** — The full offline tooling for the pipeline is drafted on the `test` branch under `tools/quality_model/`
with 75 model-free fast tests. It includes an end-to-end pipeline covering Phase 1 through 4: data corruption
(`corrupt.py`), dataset building (`build_dataset.py`), LLM/korektor auto-correction (`correct.py`), baseline
training (`train_baseline_gbm.py`), primary training on `distilbert-base-multilingual-cased` (`train.py`), and
evaluation (`evaluate.py`). The target model regresses the raw pre-clamp score along with an auxiliary category head.

## 2026-07-24

* **#31 Adaptation to other text inputs** — Opened by K4TEL: Seeks to adapt the code to process other text-containing
formats like JSON or XML, aiming to provide a `<format>-2-txt` extraction utility for different OCR engine outputs.

## 2026-07-25

* **#3** — K4TEL shared visualized results of the parameter optimization models run (from Issue #5) to showcase the
rating of importance in categorization.
* **#31** — Released `v1.2.1-beta`, which adds a draft of the JSON-2-TXT extractor. Implementation is complete
on the `test` branch: added `split_json_document()` to `page_split.py` to handle multiple JSON structures
(Family A, B, C); updated `json_stats_create.py` to derive file/page metadata from split filenames; integrated JSON
splitting into `run_pipeline.py`; and updated `README.md` and `setup/config.txt`. Tests achieved 41/41 passing coverage.

## 2026-07-26

* **#31** — Analysis flagged critical integration gaps and cross-repo architectural disconnects with `atrium-llm-enrich`.
While Issue #31 generalized the plain text producing path, the downstream contract `atrium_document.py` expects output
to be structured inside a shared `doc.json`. Outstanding issues include misstated code status in planning documents,
the absence of real OCR-engine sample data (only synthetic JSON exists), a missing `--force-single-page` function, and
a lack of end-to-end subprocess tests for the `--method json-keys` approach.

## 2026-07-28 – 2026-07-31 (partly reconstructed)

* Releases **v1.4.0-beta** (07-28, folding in @david-spacil's PR #32 categorization-logic contribution — the repo's
first outside-core-team PR) through **v1.4.1-beta** (07-31, GHA/`@v1` pin work) are tagged, but the commits under
those tags (`b25a175`…`9804c58`, including `e658586` "update GHA with ref to v1" and the Opus-authored
`docker`/`check_version.py` GHA overhaul also landing hub-side this same window) are **no longer reachable from
`test`/`master`** — a dangling-tag artifact of the same class #30's digest later documents explicitly ("lost with
that session's container"). The substance survived even though the commit chain didn't: `test` picks back up cleanly
at `86a35a0` (07-31) already carrying the completed GHA state (`docker-tool.reusable.yml@v1` present from this
commit on), and `atrium_document.py`/`atrium_document.schema.json` gain `doc_id`-aware handling as the shared JSON
contract keeps expanding ahead of any per-repo consumption logic.

## 2026-08-01

* **#37** — Pipeline invocation and dependencies updated for the upcoming JSON-2-TXT/`atrium_document` integration
(`4b70916`, `a6964b5`); issue logs refreshed. Released **v1.4.2-beta**.

## 2026-08-02

* **#37 JSON-2-TXT breaks downstream architectural contract** — Opened by K4TEL: the draft JSON-2-TXT extractor
from v1.2.1-beta (#31) predates the `atrium_document.schema.json` accretion model the ecosystem converged on around
07-31; must be refactored to ingest/mutate a shared `doc.json` rather than emit isolated plain text, and must add the
still-missing `--force-single-page` flag.
* **#37** — Implemented same day. Audited `test` HEAD before writing any code and found most of the "contract
migration" had already happened via #13's earlier `atrium_document.py`/`document_hook.py` work (Extraction/Accretion
split, field-ownership boundaries, `main()` already calling `write_document_block()`); the real gap was narrower.
Shipped `--force-single-page` as a tri-state CLI flag falling back to a `[EXTRACT].FORCE_SINGLE_PAGE_JSON` config key
(needed because `run_pipeline.py` invokes the script as a bare subprocess with no extra args); added
`tests/test_json_subprocess.py`, shelling out via `subprocess.run` for success, the flag, the config fallback,
malformed args, a missing input CSV, and partial/total corrupted-JSON failure; documented the previously-uncovered
`json-keys` extraction method in `README.md` (the only one of four methods without its own subsection). Pushed to
`test`; awaiting review and independent confirmation that `atrium-llm-enrich` can consume the output without an
adapter (no checkout of that repo was available in-session to verify directly).

## 2026-08-03 – 2026-08-06

* Hub template (`atrium_document.py` / `atrium_document.schema.json`) iterated three more times on `test`
(`01decdc`, `617a3fa`, `ea6f0b3` — the last alone adds 487 lines) as the shared JSON contract kept growing underneath
#37's already-landed implementation.
* **#37** — A further LLM-review round ("edits for LLM review+fix round by Opus", `f83a27a`) touched `page_split.py`,
`document_hook.py`, `json_stats_create.py`, `run_pipeline.py`, `alto_stats_create.py` and `README.md` on top of the
08-02 implementation; `ruff.toml` hardened (`07dcc9f`); two more `atrium_document.py` fix passes (`032fb43`) and a
version bump (`9b24fd0`) shipped as **v1.4.3-beta**.

## 2026-08-19

* GHA hardening: `release.yml`/`scheduled-smoke.yml` timeout/guard fixes (`aa8bc35`); a further Opus-reviewed round
adding `.coveragerc` exclusions, a repo `dependabot.yml`, and CodeQL/security workflow permission fixes (`6b23af0`);
one more `atrium_document.py` alignment pass with a new `tests/test_document_originators.py` (`2c81f9c`). Released
**v1.4.4-beta**.

## 2026-08-24

* **#30 (reopened)** — @david-spacil reopens after consulting @DanaKriv over the 07-29 batch: a class of lines she
flagged traces back to the same root cause the issue was originally closed on, concentrated in the 2000+-year
documents #30 is about.

## 2026-09-03

* **#30** — K4TEL and @david-spacil resume the calibration thread. Repo-side, `061bc70` unifies three previously-
diverged scoring code paths — the API service (`service/text_inference.py`), the standalone
`tools/recategorize_from_csv.py`, and the first-run pipeline (`classify_TEXT.py`) — into one algorithm; `24acaab`
aligns `docs/categorization_logic.md`/README/CONTRIBUTING; `25bc186` adds `tests/test_scoring_single_source.py`
(348 lines) to lock the three paths together going forward. Version bumped to **v1.4.5-beta**.

## 2026-09-04

* **#30** — Further back-and-forth with @david-spacil. `8fadeb9` ships short-line refinements ahead of the fuller
patch: `rule_domain_notation`, a narrow shape-based predicate for grid references, counts and abbreviation chains
(deliberately scoped to notation, not vocabulary), plus `apply_page_perplexity_blend()` groundwork.

## 2026-09-05

* **#30** — K4TEL responds in detail (09:00): `rule_domain_notation` now also exempts `rule_extreme_ppl`/
`rule_absolute_ppl` (perplexity-only convictions, agreed inverted on this population) but deliberately **not**
`rule_hard_sweep` — it needs `orig_lang_score < 0.45` as an independent second witness, since shape alone can't
separate capitalised dot-chains from real abbreviations (>50% false-accept either way on measurement).
`Bokalisace: B-XII-c` fixed via a closed label lexicon. `apply_page_perplexity_blend()`'s inert re-capping bug is
fixed (`apply_short_cap=False`) — it was re-scoring through `score_line()`, which re-applied `SHORT_PPL_CAP` and
pinned the blended value back to 850 before any rule ever saw it, so the blend could never actually move a category.
`fec8537`/`d823522` land the fix plus `tests/test_page_perplexity_blend.py`; `19433cb` cleans up configs/service/
requirements. New digest + plan committed for #30 (`cdcdf57`, `f698b7e`).

## 2026-09-06

* GHA: scheduled-smoke workflow timing fix (`94cf2f1`).

## 2026-09-07

* Confirmed live: all five GHA workflows now reference the hub's tagged **`@v1`** release rather than the mutable
`@test` ref (`docker-tool.reusable.yml@v1` on the current `Docker Build & Publish` run) — this closes the standing
cross-repo N8 finding from the hub's `project_state_3007.md` (07-30), which had found 45 live `@test` references
across the ecosystem including this repo's.
* `f4195c6` pins `service/requirements.txt` to the same version constraints as `setup/requirements.txt` — torch,
transformers, fasttext, numpy, lxml, fastapi, uvicorn and python-multipart were previously **unpinned** in the
service file, so an API-only install could resolve a different version than the batch pipeline for the same
package. A live instance of the "12-factor II" pinning gap the hub's cross-repo audit flagged elsewhere.
* Released **v1.4.6-beta**; dependabot batch-bumped scikit-learn/scipy/lxml/httpx2/bitsandbytes/fastapi/
python-multipart/matplotlib (#47); CI green on both `test` and `master`.
* **State**: 7 open issues (#2, #3, #4, #23, #30, #31, #37); #5 and #6 closed 2026-07-21/07-25. `test` and `master`
are identical at `cb235b5`. The two active threads are #30 (reopened, blocked on three questions to @david-spacil
before his PR can open — see `digests/30.digest.md`) and #37 (implementation shipped, blocked on independent
cross-repo confirmation from `atrium-llm-enrich`).

## 2026-09-08

* `1bd64d8` gives the service a deployable surface: a new **`api` Dockerfile stage**
(`EXPOSE 8000`, explicit `STOPSIGNAL SIGTERM`, `HEALTHCHECK` → `service/healthcheck.py`,
`ENTRYPOINT ["python", "service/text_api.py"]`), plus `ServiceState`, an in-flight middleware and
`serve_lifecycle` with a `/ready` endpoint in `service/atrium_service.py` (+246), and five `/ready`/drain
tests. Before this, the FastAPI app was reachable only via a docker-compose `entrypoint:` override on the
batch image, so no runnable API image was ever published for ARÚP/ARÚB to deploy.
* That stage shipped on a **push**, and `docker-build-smoke` is gated `if: github.event_name ==
'pull_request'` in the hub reusable — so nothing ever started the image. The first fork PR did, and it died
at import: `ModuleNotFoundError: No module named 'atrium_document'`. Launching a *script* puts
`/app/service` on `sys.path[0]` and leaves `/app` absent. `4139e93` adds the bootstrap above the
first-party imports and pins it two ways in `tests/test_service_entrypoint.py` — a `slow` subprocess test
reproducing the container's `sys.path`, and a cheap source-order guard, because ruff's import sorter is
what would undo it. **The probe still does not run on pushes**, so the fix is verified by unit test rather
than by a container start.
* `fdf35b1` + `d774ba3` land the SKOS controlled-label registry (`atrium_vocab.py`, +1,152, hub issue #51)
and keep `tests/test_atrium_vocab.py` out of ruff's reach — it is one of the eleven hub-canonical files
`para-drift` compares with `diff -u`, so a local reformat would break drift.
* Dependabot: `cb235b5` (setup-deps ×8, #47) and `2b7a400` (service-deps ×4, #49).

## 2026-09-09

* **Issue #30 — the blocking claim was falsified, in part.** Three rounds had concluded that separating
`oueussd` from `malakofauna` needs a lexicon. It does not: `detect_fused_words()` already returns 1 for
`oueussd` and 0 for `malakofauna`, and **gate 7 already reads it** — inside its `damage` term, behind
`and not structurally_clean`, where `structurally_clean = valid_word_ratio >= 1.0`. Since
`compute_valid_ratio` is shape-only (length ≥ 3, ≥ 70% alphabetic, no strange char, no mid-word
uppercase), it is 1.0 for every line in this population. The one witness that discriminates was being
suppressed by the one signal that cannot. What genuinely needs word knowledge is the narrower residue —
`malakofauna` against `edelite` — where nothing about either spelling is wrong.
* `693e1b4` acts on that: **`_has_shape_garbage_evidence()`**, four `SHORT_GARBAGE_WITNESS_*` constants,
and `SHORT_GARBAGE_WITNESS_ENABLE` — **off by default, and with no call site**, because the conditional it
would join is the one-hunk change still under review in PR #48.
*(Superseded: PR #48 merged as `070620f`, and the witness was wired into gate 6 as a second disjunct in the
D15 follow-up. It is still **off by default** — wiring and enabling are separate, and the flag waits on a
gold set. The "no call site" statement above describes `693e1b4` and is kept as the record of that commit.)* Measured 8/12 thread-reported garbage
reached at 0/21 false positives on real vocabulary and notation, and 0 across all 33 committed positive
fixtures. Two clauses were **deliberately not** reused from `detect_fused_words`: its `len > 14` test
(flags `Skelettmaterial`) and `_RE_FUSED_CONSONANT_RUN` (flags `vrstva`, `vrstvy`, `ctvrtek` — and
`vrstva` is the commonest noun in archaeological field documentation). Each refusal carries a test naming
its counterexample.
* Same commit closes four test-suite blind spots found alongside. `tests/test_smoke.py` was a **third**
hand-rolled scoring harness — feeding `categorize_line` the `LANG_SCORE_REMAP` cap instead of
`trust_lang_score`, and omitting `orig_lang_score` and `garbage_density` entirely, which left them at
`1.0`/`0.0` and silently disabled `rule_hard_sweep`, `rule_extreme_ppl` and the density branch of
`_has_strong_garbage_evidence`; it now goes through `_rescore_row`. Three **discriminating** golden edge
cases replace a pin that could not see the change under review (the old `short_garbage` case sets
`gibberish_present=True` *and* `valid_word_ratio=0.0`, either of which short-circuits the evidence
predicate to `True`). A swept **Trash-side** counterpart joins the existing clean-Czech sweep, which only
ever guarded one direction. And the ablation rule lists in `run_ablation_study.py` /
`greedy_backward_elimination.py` were stale from before the `penalty_* → rule_*` rename, so
`DISABLED_RULES` silently did not match and some rules were missing outright — neither list had any test.
* `5e669bb` fixes the tuner lane. `setup/requirements-sweep.txt` floors scikit-learn at 1.9 and matplotlib
at 3.11.1, and **both are published for Python ≥ 3.11 only** — on 3.10 the resolver finds nothing and
`run_optim_pipeline.sh` aborts on a wall of candidate versions that never mentions the interpreter. That
is now recorded as the measured reason the file had asked for, mirrored in `requirements-finetune.txt`, and
the script fails fast with the cause, the fix and a `SKIP_DEP_INSTALL=1` escape hatch. Separately, the
**Sobol backend was broken under numpy 2**: on a zero-variance objective SALib returns `np.array([0.0])`
from its estimators, and numpy ≥ 2 refuses to assign that into a scalar slot — a degenerate objective
surfacing as a dtype error two frames inside SALib. Reachable by default, since the sample corpus cannot
move most constants. Guarded with the `importance_skipped` convention `run_optuna_backend` already used.
* New `tools/short_garbage_witness_report.py` makes the witness measurable **before** it has a call site:
it reports, over any delivered `DOC_LINE_CATEG` collection, which lines the predicate reaches against the
category the pipeline currently assigns, and writes candidates with a blank `gold_categ` column for blind
annotation. It deliberately computes **text-only** predicates and never re-scores — a test source-inspects
it for `score_line`/`trust_lang_score` and friends, because reconstructing signals from stored columns is
the harness bug this repo has now fixed three times.
* Rule coverage regenerated on 2,171 lines: **16 LOAD-BEARING · 3 REDUNDANT-HERE · 3 DEAD**. Full table and
the diff against the stale 14-rule log in `tools/SWEEP_NOTES.md`. `rule_hard_sweep` holds at 94 fires;
`rule_allcaps` and `rule_garbage_density` move `REDUNDANT-HERE → LOAD-BEARING`; `rule_short_garbage`'s
decisives go 4 → 16. `rule_short_line` posts the first non-zero `clear_loss` (2) on record.
`rule_bigram_run` and `rule_vowelless` fired zero times — retirement *candidates* pending the full-corpus
run `RULE_COVERAGE.md` designates as authoritative, not retirements.
* **State**: 7 open issues. `test` `4017a76`; `master` `ebaec0a`, **4 commits behind** — every previous tag
is an ancestor of both branches, so a release syncs them first. `CITATION.cff` and
`setup/para_config.txt` both still read `1.4.6-beta`, and `check_version.py --require-tag` rejects a
mismatched tag, so the bump precedes the tag. #30's PR [#48](https://github.com/ufal/atrium-alto-postprocess/pull/48)
is **open as a draft** and unmerged: the shape witness therefore ships **inert**. Merging it flips the three
new golden pins to `Clear`, which is why they were written to be able to see it.

## 2026-09-20
- **#30** — **Stage 7 re-read against its own delivery**, rather than against the summaries written when it ran.
The thirteen CSVs and two logs attached to the thread on 2026-09-19 were pulled down and recomputed. Five findings
(S1–S5 in `digests/30.digest.md`); three overturn a decision cell in `plans/30.plan.md`, and all three were
refuted by evidence that was already inside the delivered logs.
* **S1 — D25 fails at full scale, and 07a is the run that refutes it.** The de-gemination cap was fitted to a
4.4× gap on the 822-document table (`ppole` df 35, every confirmed artefact ≤ 8). Over 113,100 documents the gap
is **1.17×** (229 against `ssuti`'s 195), and by *ratio* `ppole` is **fifth of seven** with four artefacts below
it. The constant is an ABSOLUTE document count, so it does not rescale with the lexicon: at the shipped `10` on
the full table the guard is right on **2 of 7** tokens instead of 7 of 7. Value deliberately unchanged — eight
tokens is not a population to fit a production threshold to. New `geminate_cap_scale_warning()` says so on
stderr; both tables pinned by tests; the portable signal (per-collection concentration) is stage 8c.
* **S2 — 07b is void, and it is the 07c bug one indirection in.** 07b and 07c produced *byte-identical* tables
for two unrelated constants, `KL` to five decimals included. `quality_word_set()` is a zero-argument
`functools.lru_cache`; `ab_constant_eval.py` runs both arms in one process; the `True` arm read the `False`
arm's cached `None`. `_DERIVED_FROM_FLAG` had fixed one *form* of flag-freezing (a module constant built at
import) and this was the other form, already in the tree. Fixed by `_CACHES_FROM_FLAG` in `override_constants()`
plus a **source-level guard test** that fails when a new zero-argument cache appears unregistered. D26 therefore
has no measurement at all — not a null one.
* **S3 — the 07f annotation pack is a re-cut of the witness queue, not a second population.** All four tabs join
back at 100%: 2,424 of the same 5,005 strings, 17,635 of the same 20,078 lines. H1/H2/H3 had been asking for the
same work three times. New `tools/build_annotation_sample.py` sizes the real ask at **293 decisions** — 93
census rows reaching 94.8% of at-risk exposure exactly, 200 sampled tail rows at ±6.9 points — with the sampling
frame written beside them, because a sample without its frame is 200 anecdotes.
* **S4 — the modal-dedup blast radius is 7 groups / 22 lines.** Of the 5,222 (document, string) votes behind the
queue, 5,200 are unanimous and 4,903 are a single line; 22 are contested and only **7** rest on a bare plurality.
And the cascade is *protective* here: cascade on, the witness breaks **1** line; cascade off, **3**, with the
baseline `Clear`-loss 55 instead of 40. H8 option 2 is now a production-wide change motivated by 22 local lines.
* **S5 — 07d already reports the adoption gate passing.** Both arms print `-> ADOPT-CANDIDATE` at 503 errors /
`Clear`-loss **40** / cost 0.2829 against shipped 513 / 40 / 0.2917. Every other account still reads the flag as
rejected on `Clear`-loss +1. It is also the third distinct figure for the same nominal config (42 → 41 → 40),
never root-caused. **Stage 8a** — one A/B over `SHORT_GARBAGE_WITNESS_ENABLE` on the stage-7 tree from one
output directory — settles both in one run, and needs no annotation, no GPU and neither collaborator.
* **Collaborator-facing docs**, written at B2+ English for readers who have not followed the thread:
`docs/issue30_annotation_guide.md` (@DanaKriv — the 293 decisions, how to read the evidence columns, and the two
questions that are not in the files) and `docs/issue30_review_request.md` (@david-spacil — the 503/40
sanity-check, the dedup decision costed on both sides, the 508 re-score, and the eight doubled-initial tokens at
full scale, which only he can label).
* **Stage 8 written and dry-run green**, as `issue30_stage8_job.sh` (cluster-side, like the stage-6 and stage-7
jobs — not a repo file). Eight stages: 08a the gold join plus the geminates' per-collection split (D30's
experiment, seconds, and it reads columns already in the table); **08b the decisive witness-flag A/B** that S5
says may already pass the adoption gate; 08c the same with the cascade off; 08d/08e the two void re-runs;
**08f the only stage that genuinely needs every document** — witness exposure, the annotation queue and the
dedup groups over BOTH archives; 08g the 293-decision ask built from that real queue; 08h opt-in, the
pre-cascade frame. The gold-scored A/Bs deliberately run on the 822-document gold corpus: `--gold-column`
scores only the annotated rows and `recategorize_dataframe` is per-document, so the collections would give the
same numbers 5.7x slower. Validated against a synthetic two-archive layout: seven stages OK, every stage SKIPs
on re-submit, `FORCE=<stage>` re-runs exactly one, a wrong archive path exits 2 before anything runs, and
deleting `_CACHES_FROM_FLAG` from the tree makes it refuse to start.
* **Three CLI gaps closed so stage 8 could exist at all (D31).** `--input-dir` on the witness report now
REPEATS, because "all of the collection documents" was not sayable: there is no common parent holding only
ARUP and ARUB, and a staging directory of symlinks **silently reads zero** —
`pathlib.Path.glob("**/*.csv")` does not follow directory symlinks before Python 3.13 and does not fail.
Verified on the venv's 3.11 and pinned by a test that will fail if a future Python fixes it. Also `--recursive`,
`--by-group` (the dedup's blast radius in the unit it votes in), and `--no-postprocessing` on the re-scorer so
the pre-cascade frame can be written rather than only scored.
* **S4 corrected by reading the implementation rather than the mechanism.** The modal dedup resolves with
`x.mode()[0]`, and `mode()` returns its tied values **sorted** — so a tie takes the alphabetically first
category and `Clear` < `Noisy` < `Trash`. **A tie can never demote to `Trash`.** Recomputed: of 5,222
(document, string) votes, 5,200 unanimous, 15 strict majority, 7 tie, and **0 bare plurality**. The vote
destroys **17 `Clear`** lines and rescues **31 `Trash`** ones, net **+14** in the cascade's favour. So H6/H8
option 2 — "stop a bare plurality demoting `Clear` → `Trash`" — is a **no-op**: the case it removes does not
occur. The earlier S4 figure ("7 groups / 22 lines") conflated ties with bare pluralities and did not check
where a tie lands.
* **State**: no categorisation change. Every flag involved still ships `false`, and the flag-off re-score of the
sample CSVs moves **0 categories**. Suite **1252 → 1288** passing, 0 failed, `ruff` clean. Stage 6 still
running; stage 8 written, validated and not yet submitted.

---
_Timeline index refreshed 2026-09-09 against live `test`/`master` HEAD, the current release list, open-issue state
via the GitHub API, and the refreshed `30.digest.md`/`37.digest.md`; 2026-09-20 entry appended from the stage-7
re-read (v1.5.0-beta is now the current tag). Nothing removed from the issues themselves
(per hub #29); this file is a derived reading aid in `agent_dev_logs/`._

## 2026-09-21
- **#30** — **Stage 8 delivered and read against its own delivery.** Eleven CSVs and `logs8.log` pulled
down from the thread and recomputed rather than read from the summaries written when they ran. Fifteen
findings (T1–T15 in `digests/30.digest.md`); four questions settled, one defect found in the delivery, and
two findings made earlier in the same read corrected by re-measuring them properly.
* **The A/Bs settle four things.** **08b** — the run stage 8a was written for — is the paired A/B over
`SHORT_GARBAGE_WITNESS_ENABLE` itself: errors 513 → 503, `Clear`-loss **40 → 40**, cost 0.2917 → 0.2829,
fixes 12 / breaks 2, exact McNemar **p = 0.01294**. **The adoption gate passes**, and the 42 → 41 → 40
drift that has never been root-caused resolves to 40 in *both* arms of one run — it was between trees, not
within a measurement. **08c** (cascade off) rejects on `Clear`-loss 55 → 56, so `apply_document_postprocessing()`
is what makes the witness adoptable. **08d** kills D26: `QUALITY_VOCABULARY_ENABLE` is 212 fixes against
**540 breaks**, `Clear`-loss 40 → **180**, p ≈ 7.8e-34 — it buys 78.3% `Trash`-recall by destroying 13.8% of
all gold `Clear`. **08e** closes D27 as a genuine null (5 fixes / 5 breaks, p = 1) rather than void.
* **08a inverts D30 rather than merely failing it.** The per-collection concentration experiment — an
artefact belongs to the scanner that made it, an abbreviation is a convention and should appear in both —
returns `ppole` at **3 ARUP / 226 ARUB**, the *most* collection-concentrated of the eight tokens, with every
confirmed artefact less concentrated. The abbreviation looks more like an artefact than the artefacts do,
because it is a convention of one institution's 2010s forms. Neither ratio, nor absolute document frequency,
nor per-collection concentration separates the classes; the de-gemination guard has no portable signal left
and the remaining choice (keep it fitted to one table, or set it to `0`) is @david-spacil's.
* **08f retires H6 option 2 on a corpus-wide denominator.** 61,682 (document, string) groups over both
archives: 61,359 unanimous, 143 strict majority, 180 tie. The cascade rescues **305** `Trash` lines and
destroys **24** `Clear` — net **+281 in its favour**, a wider margin than the 31-against-17 S4 measured on
the small queue. Groups option 2 would change: **0**. The `mode()[0]` tie-break stays alphabetical and
load-bearing — `Clear` < `Noisy` < `Trash`, so none of the 180 ties can land on `Trash`.
* **T1 — the defect, and the reason stage 9 exists.** **08f and 08g were run with the lexicon off.**
`setup/config.txt:199` says the witness and the table are one decision and the witness never ships shape-only.
The proof is in the queue rather than a missing log line: `ppole` is document frequency 229 at full scale,
above the geminate cap of 10, so `_has_vocabulary_support()` already exempts it and a lexicon-on queue
**cannot contain it** — yet it is the largest row in the delivered one at 15,466 lines. Estimated from 08g's
own `token_status`, **73.1% of the at-risk exposure (26,868 of 36,744 lines) is exempt** once a table is
configured, leaving 9,876 lines / 7,433 strings. The 592-row census then settles **30.7%** of the real
population rather than the advertised 82.8%, because the survivors run 1.33 lines per string and are 93.7%
singletons — there is no head left to take a census of. **The delivered ask should not go to @DanaKriv until
it is re-cut.** 08b–08e are unaffected: they are gold-scored against the 2,064-line sidecar and their logs
carry the geminate-scale warning, so the table was configured for them.
* **T4 is the most direct measurement of this issue's actual title anyone has made in it.** 65.9% of at-risk
exposure sits in 2010-or-later documents and the 2010s carry the highest at-risk rate of any decade (49.1%),
while the clause mix inverts across that boundary: pre-2010 is `vowel_run` 76%, 2010+ is **`initial_geminate`
66%** — and **96.1% of those 2010+ geminate lines are `ppole`**. The modern half of the archive is a different
document genre (structured excavation forms with short field values), and the rule that fires hardest on it
fires almost entirely on one legitimate abbreviation. MTX is ARUB and CTX is ARUP (`ppole` 15,662 MTX / 11 CTX),
and `ppole` has a date: 2 lines in the 1990s, 45 in the 2000s, 13,144 in the 2010s.
* **T9 — `Trash` is answering two questions.** 76.9% of witnessed `Trash` lines carry no repeated-character
run and no unusual glyph; `http://www.arub.cz` is stored `Trash` on 5,309 lines while being correctly
transcribed and perfectly legible. Under the definitions in use (`Trash` = nobody can read this) that is a
mislabel — legibility and usefulness are different axes, `Non-text` already exists for the second, and it is
not being used for it. Stated before annotation starts, because an annotator applying the definitions as
written will call that URL `Clear` and nobody will know whether the metric moved because of the annotator or
the definition.
* **D32/D33/D34 — three predicate questions, one defect, and two of them were this read's own mistake.**
`Lepus europaeus`, `Mammalia indet.` and `Triticum monococcum` all convict on the shipped predicate — **with
no lexicon configured**, which is T1's mistake applied to my own findings. Re-measured with a table armed at
the real full-collection frequencies, every one is already exempt through `_has_vocabulary_support()` (D14),
so **D32 is not a predicate gap and nothing was changed**; a generic "capitalised genus + lowercase epithet"
exemption was drafted, measured against the corpus, and **rejected** because it also matches `Chenopoaium
hycnaum`, `Loua (oxkuku` and `Laaid. bazic 64999/`. **D33 is real and is fixed**: boilerplate URLs are attested
by repetition, but a citation quoted once can never be attested by a document-frequency table — 144 such lines
survive a configured lexicon, and `is_domain_notation()` now recognises the shape with **0 misses on the 144 and
0 false positives against the other 7,289 at-risk survivors**. **D34 is measured and left alone**: `Kaukasus`,
`Hallstatthaus` and `Schuhleistenkeilbruchstueck` are real, unattested and convicted by `low_variety`, and no
length, ratio or case-shape cut separates them from garbage of the same shape (`vodovod`/`PSSPPOP` are both 7
letters at ratio 0.43; `VODOVOD`/`PSSPPOP` are both ALLCAPS). Retuning from two named words is what D25/D30 did
with the geminate cap and what 07a then refuted at scale, so it is named, pinned debt instead.
* **Stage 9 specified** in `plans/30.plan.md`: **9a** re-runs 08b/08c on the post-D33 tree with the base
configuration *printed* (T1 cost a delivery because a configuration had to be inferred from which warnings
appeared); **9b** is the exposure pass 08f should have been, with `SHORT_GARBAGE_LEXICON_PATH` set; **9c**
re-cuts the ask from the 9b queue at `--strong-df 113` rather than 10 (90% of the delivered recoverability
evidence is `edit1`, at a threshold 08g's own log calls too permissive); **9d** re-measures the dedup groups
on that queue. None needs a collaborator, a GPU or a new label.
* **Collaborator-facing docs updated**, plus a new shared one: `docs/issue30/issue30_corpus_profile.md` —
what the full-collection pass says about the archive itself (the telegraphic register the vocabulary veto
destroys, the two collections, the 2010s form genre, Latin taxonomy, the excavator's own name, footer URLs)
and what follows for quality categorisation. `issue30_annotation_guide.md` and `issue30_review_request.md`
rewritten around the stage-8 answers; `annotation_ask_README.md` and the delivered `census.csv` / `sample.csv` /
`frame.json` replaced with the stage-8 set, carrying the T1/T2 caveat on their own sizing.
* Suite 1,288 → 1,290 passing, 0 failed, `ruff` clean.

## 2026-09-22
- **#30** — **Stage 6 delivered (87h) and read against its own delivery. R3 is closed**, and the
delivery contains three instrument defects, one of which nominated this issue's own feature for deletion.
Eight findings (U0–U8 in `digests/30.digest.md`).
* **R3 closed.** Every `DEAD` / `LOAD-BEARING` / `clear_loss` verdict this issue ever quoted was scored
against the pipeline's own output; all 23 rules are now scored against the 2,064-row gold sidecar.
**Baseline gold `Clear`-loss reads 40 — the same number 08b reports, from a separate job with a different
tool.** First cross-run consistency check this issue has had, and it passes.
* **U1 — the sweep nominated `rule_short_garbage_witness` for retirement.** `fire_count == 0`, so
`_classify()` returned `DEAD`, whose docstring reads "unreachable dead code [that] can be permanently
deleted ... because deletion provably changes nothing". It is not dead, it is **switched off**:
`_fire()` sits behind `SHORT_GARBAGE_WITNESS_ENABLE`, which ships false, so the count is zero on any
corpus. 08f had measured the same predicate at **100,824 lines** and 08b had flipped the flag and passed
the adoption gate (p = 0.01294). `RULE_COVERAGE.md` called `fire_count == 0` the "config-**independent**"
retirement criterion — the false word the whole criterion rested on — and the tool **exits 1** on any
`DEAD` rule, so a flag that ships off makes the instrument fail a pipeline driver.
  **It was known twice, in places the artefact does not carry.**
  `tests/test_pipeline_parity.py::UNREACHABLE_RULES` had the taxonomy ("unreachable BY CONFIGURATION"
  vs gate shadowing) in a test no tool can read; and `issue30_stage6_job.sh`'s own epilogue says it in
  plain words — but prints to the SLURM `.out`, not into `06_coverage.log`, which is what `tee` captures
  and what gets attached to the issue. The delivered artefact carries "safe to retire" unqualified.
  **Fixed as D35**: `text_util.CONFIG_GATED_RULES` maps rule → gating flag beside the flag itself,
  `_classify()` gains a fourth class `INERT` consulted *before* the zero branch, `INERT` is excluded
  from the exit-1 set, and the criterion in `RULE_COVERAGE.md` gains the matching clause. The gate is
  read only when the count is zero, so a gated rule that somehow fires is still reported — a flag must
  not hide a real finding. Reproduced live on the smoke fixture both ways.
* **U2 — neither `DEAD` rule is retirable.** `rule_mid_uppercase` is unreachable by gate shadowing
(9d behind 7), already in `UNREACHABLE_RULES`. And the class is sample-sensitive in fact, not just in
principle: the 2026-09-09 sweep (1,471 scored lines) called `rule_bigram_run` and `rule_vowelless` DEAD
too; at 4,886,492 scored lines they fire **52** and **1,086**. Two of three verdicts were sample
artefacts, exactly as `SWEEP_NOTES.md` warned. Recorded in the retirement criterion as a measurement.
* **U3 — seven rules score better on gold when removed, and five of them are noise.** The report carries
**no significance testing at all** — no McNemar, no effective-n; `evaluate_dataframe`'s
`return_correctness` mask is left `False` by both `_loo_metrics` calls, and the repo's exact-McNemar in
`ab_constant_eval.py` is never reached from here. Calibrating on the same 2,064 rows (08b: +0.0171 over
10 net-corrected rows ≈ **0.0017 macro-F1 per gold row**), five of the seven move **less than one gold
row**. Only `rule_short_line` (≈4) and `rule_trailing_fill_rescue` (≈1.3) clear the floor, and both
*cost* `Clear` lines when removed. Nothing in that column is actionable as a retirement.
* **U4 — `rule_short_garbage` destroys 7 of the 40 gold `Clear`-losses.** `rule_short_line` protects
**31**, `rule_reference_floor` protects 10, `rule_hard_sweep` destroys 2. The 40 the flag decision has
turned on for two months is partly the product of the rule the witness was built to narrow — a number
nobody has had before.
* **U5/D36 — `gold_clear_loss` is an absolute count printed beside a delta.** Twelve rules report
exactly 40; that is the LOO arm's absolute figure, not twelve rules each destroying 40. The baseline was
computed by the same pass (`baseline_vs_gold`) and discarded. Now emitted as `gold_clear_loss_baseline`
in the JSON and the table, at zero extra cost.
* **U6/D37 — `decisive_cascade` is not the cascade a rule's removal sets off.** There is no
all-rules-on / smoothing-off baseline pass anywhere in the tool, so `decisive_line` carries the whole
smoothing footprint. The two zero-firing rules expose the floor: both read `decisive_line = 165,482`,
`decisive_cascade = −165,482`, netting zero — **165,482 is the smoothing footprint itself**, sitting
under every other row. Corrected in the docstring, the table label (now `smoothing residual`) and
`RULE_COVERAGE.md`; the missing fourth pass is recorded as an option, not proposed.
* **U7 — one row is already stale, and the first attempt to size it was circular.** Stage 6 ran before
`d4b9973` landed D33, so its `rule_domain_notation` row is pre-D33. Comparing the predicate before and
after D33 over the stage-8 witness queue gives "0 before, 8,082 after" — **a tautology**: every line in
that queue is there *because* pre-D33 `is_domain_notation()` returned False for it. Read the other way
it does say something real — D33 removes 8,082 lines (808 distinct strings) from the witness-reachable
population — but the corpus-wide effect is unmeasured. Caught before it was written down as a finding.
* **T1 upgraded from inference to confirmed.** `issue30_stage8_job.sh` settles the stage-8 finding
directly: `08b_witness_flag_ab` and `08c_witness_flag_perline` both set
`ATRIUM_TEXT_UTILS_SHORT_GARBAGE_LEXICON_PATH="$LEXICON"`; `08f_exposure_full` sets no `ATRIUM_*` at all.
**The gold-scored A/Bs arm the lexicon; the one stage that measures the population does not.** 08g then
passes `--lexicon` for its evidence columns but inherits 08f's queue, so the ask carries the lexicon-off
population regardless.
* **Stage 9 updated** with **9e**, and 9b sharpened to a one-line change: add
`ATRIUM_TEXT_UTILS_SHORT_GARBAGE_LEXICON_PATH="$LEXICON"` to the 08f stage command, delete
`$OUT/08f_exposure_full.ok` and `$OUT/08g_annotation_ask.ok`, re-submit — the `stage()` marker contract
re-runs exactly those two (~1h). 9e is **not** recommended as a full 87h re-run: it would refresh one
stale row and a classification D35 already corrects, and `issue30_stage6_job.sh` is explicit that arming
the witness flag must be a separate job (it refuses to start otherwise).
* Suite 1,290 → 1,297 passing, 0 failed, `ruff` clean. An existing test caught the new class correctly
(`test_run_coverage_smoke` enumerates the valid class set) and was updated rather than worked around.

## 2026-09-22 (second entry)
- **#30** — **Stages 9b/9c ran: the exposure pass and the annotation ask were re-cut with the vocabulary
lexicon armed.** The configuration `setup/config.txt` says the witness only ever ships in. Same corpus,
same 56,599,631 lines read. Twelve findings (V0–V11 in `digests/30.digest.md`); the ask landed on `test`
as `d9c41a5` and **the hold on @DanaKriv's request is lifted**.
* **V1 — T1's estimate was low, in the direction it keeps being low.** T1 put the exempt share at
**73.1%**; measured it is **82.1%**. At-risk falls from 8,529 strings / 37,555 lines to
**5,563 / 6,714**, not the projected 7,433 / 9,876. Nothing that followed from T1/T2 changes sign — the
ask still had to be re-cut and still came out smaller — but every estimate this issue has made of how far
`_has_vocabulary_support()` reaches has been too conservative, and that is worth carrying forward.
* **V2 — D33's corpus-wide effect is measured, for free, and it is a floor.** Both runs read the same
lines; in-scope moved **7,493,429 → 7,477,924**, so **D33 removes 15,505 lines from the witness's scope**.
Sound only because D33 is the *only* change to the three scope vetoes between the runs. The production
figure is larger, because the witness's scope also demands `word_count <= 3` and `rule_domain_notation`
does not. **This falsifies the `unreleased` release row's "no categorisation change"** — rewritten, and
the release is a minor bump.
* **V3 — the witness is one clause with three small companions.** `vowel_run` is the **sole** reason for
**4,305 of 6,668 at-risk lines (64.6%)** and is the least discriminating of the four (79.8% agreement with
the pipeline's existing answer, against `triple`'s 97.9%). 93.7% of at-risk strings satisfy exactly one
clause, so the attribution is clean rather than a slice.
* **V4 — the at-risk head is no longer the archive's own vocabulary.** `ppole`, `ARCHAIA` and the Latin
taxonomy are all exempt now. What is left is **`Dauerleihe`** — German for *permanent loan*, **286 `Clear`
lines**, scanned perfectly — plus `J. Vysoean`, `Dated=Dated (relatively)` and `FEUILLETON.`. The
mechanism generalises: the lexicon is built from the archive's own text, so it protects what the archive
says *often* and offers nothing to a word that is correct but rare *here*. Third appearance of the same
failure, through a third door, and the first where it lands on a language rather than a register.
* **V5 — there is a lever, and it is not adopted.** `SHORT_GARBAGE_WITNESS_VOWEL_RUN_MIN` 3 → 4 is
estimated to cut at-risk exposure **56.6%**, sparing every correctly-read item in the head while keeping
the `OUUITN`/`OUOISP` page-stamp family and `eaual to:`. **Estimate, not measurement** — it applies the
clause's regex to the delivered text rather than running `shape_garbage_clauses()` — and it is not
gold-scored, so 08b's adoption gate does not cover it. Fitting a threshold to four nameable words is
precisely the D25/D30 mistake; it becomes stage 10a/10b, both local and cheap.
* **V6 — H6 option 2 is closed by measurement.** Blocking a bare plurality from demoting `Clear` →
`Trash` changes **0 groups and saves 0 lines** corpus-wide. The cascade is a net rescuer by **+41**
(51 `Trash` lines lifted, 10 `Clear` lines pulled down) — the sign flipped relative to the 822-document
queue, where it was 31 against 17. The alphabetical tie-break is still load-bearing and still undefended
by a test that names it, but decides 17 groups now rather than T14's 180.
* **V7 — none of the eight doubled-letter tokens is in the queue.** Zero rows of 42,853. Three carry
Czech diacritics and could never be in scope; the rest are exempt by document frequency. § 4 of the
review request stops blocking the ask and becomes a question about the de-gemination guard alone (D40).
* **V8 — two reconciliation gaps, recorded rather than chased.** 08f writes 42,853 distinct strings and
08g's queue reports 42,248; 08g's at-risk is 5,563 / 6,714 against 5,695 / 6,668 recomputed from its own
CSV. 2.3% of strings, 0.7% of lines, no conclusion affected — **but 18.1% is quoted to collaborators
against a denominator that cannot be reproduced from the delivered files**, so the ask README says so.
* **V9 — 18.1% is 15.9% once the controls come out.** 39 of the 157 census rows (149 lines) are
`confirms_trash` checks whose label cannot move a decision. The arithmetic closes: 5,563 − 118 = 5,445,
exactly the frame's string total. And the instrument has changed character because the population did:
**95.7% of at-risk strings occur exactly once**, the top 500 settle 12.5% of witnessed lines, so the
census keeps only 60 frequency rows and the 200-row sample carries the estimate. 357 decisions cover
**516 distinct spellings** — `Dauerleihe` alone folds 24.
* **Release prepared as `v1.6.0-beta`**, not a patch, because of V2. `CITATION.cff` and
`setup/para_config.txt` bumped together; `check_version.py` agrees with and without `--tag`.
* **Collaborator documents refreshed for the round with @DanaKriv and @david-spacil**: the ⏸️ notices
removed from `annotation_ask_README.md`, `issue30_annotation_guide.md` and `docs/issue30/README.md`;
§ 0 / § 3 of `issue30_corpus_profile.md` rebuilt on the lexicon-on figures with the old table kept and
labelled; a new § 7 in `issue30_review_request.md` putting the vowel-run trade to @david-spacil, whose
judgement about this material decides it. Re-measured after editing: median sentence 11.5–16 words,
sentences over 35 words 0.0–3.7%, no unglossed jargon.
* **One correction inside the collaborator documents.** The previous revision told @DanaKriv that
`Dauerleihe` was protected by the dictionary. It is not — it is the single largest item at risk. Fixed
where it appeared.
* Suite 1,333 passing, 0 failed, `ruff` clean. Documentation-only change plus the version bump.

## 2026-09-22 (third entry)
- **#30** — **@david-spacil answered the review request in full**, section by section, without
waiting for the tag. Four items close. **Two of them close by correcting us**, and both corrections
are of things this repository had already measured and then read past — not of things it could not
see. Six findings (W1–W6 in `digests/30.digest.md`).
* **W2 — `ssuti` / `ssutí` / `ssutě` are an old spelling of *suť*, not scanning errors.** The
doubled-letter split is **four legitimate against four damaged**, not one against seven. Corrected in
six places. The negative result survives and gets a better reason: the per-collection table was read
as an inversion (*the* legitimate token being the most one-sided of the eight), and it is not one —
**both kinds of real language are house conventions, one per institution** (`ppole` is ARUB's at
3/226, `ssut*` is ARÚP's at 152/43), while the four real errors are small and split across both.
Concentration separates *institutions*, and each institution has its own vocabulary. It also sharpens
the ratio table: the "genuinely empty gap from 2.60× to 4.66×" that
`SHORT_GARBAGE_LEXICON_GEMINATE_MAX_DF` is fitted to has real language on **both** sides of it.
* **W3 — D40 resolves: the de-gemination guard comes out.** *"if the dictionary already covers all
eight, it looks redundant – no objection to dropping it."* V7 is the measurement that says it does —
zero of 42,853 queue rows carry any of the eight. Stage **11c**; not changed in this pass, because
this pass was the reading.
* **W4 — T9 resolves in favour of legibility.** `Trash` = illegible; legible is `Clear`; easily
decipherable is `Noisy`; **regardless of usefulness**. Pending @DanaKriv. So `http://www.arub.cz` is
`Clear` and D33 left it one step short at `Noisy`. **And it explains the gold set for the first
time**: he could not tell `Non-text` from `Trash` without the page image, so he labelled only three
categories — which is why the sidecar carries 45 `Non-text` rows against 1,302 `Clear`, an asymmetry
visible in every class-support table this issue has printed and never accounted for. One thing it
resolves in the code's favour: the root `README.md`'s `Trash` row ("re-processed by another OCR
tool") matches his answer exactly, so only the legibility half of the two descriptions needed
reconciling. **D43.**
* **W5 — the vowel-run clause is the right rule at the wrong scope, and V5 mislabelled two of its
own examples.** `J. Vysoean` and `B/ POSTKRANIAINY SKELET:` are scanning errors, not correctly-read
text — **V4's own table said so and V5 contradicted it two sections later**. Corrected in place: the
honest trade at min=4 is **299 confirmed-good lines spared against at least 226 confirmed-bad ones
released**, not the clean rescue V5 described. His direction is better than the threshold: a
three-vowel run is a fact about Czech phonotactics, so gate the clause on the detected language.
**Cost checked rather than assumed** — `lang`/`original_lang` exist per line, but
`determine_category()` takes only `lang_score`, `orig_lang_score` and `is_upright_czech`, and that
last reduces to a Czech word-list hit on this population, which both `Dauerleihe` and the damaged
`J. Vysoean` fail. Signature change plus its own A/B. **D44**, conditional on sizing it first.
* **W6 — the consequence nobody asked about.** He annotated the gold sidecar *before* discovering
`ssut*` is real language. If any of its 2,064 rows is an `ssut*` line labelled `Trash`, that row is
now wrong — and it is the sidecar `Clear`-loss **40** is measured against, the number the adoption
gate turns on. Exposure is small (1,667 `ssuti` lines in 56.6M, over an 822-document gold corpus),
which is exactly why it should be *seen* to be zero rather than argued to be. **Stage 11a**, and
until it runs, 40 carries an unquantified asterisk.
* **Stage 11 added** (11a gold check · 11b `lang` distribution of the queue · 11c guard removal ·
11d language gate · 11e reconcile the five category descriptions), and stage 10's *reading* updated
without touching the running job: 10b's spared set is a mixture rather than a rescue list, and 10c's
overlap with the delivered ask is how the mixture gets quantified.
* Also fixed: § 8 of the review request still numbered its sub-items 7.1/7.2/7.3 after the section
was renamed, which is why he had to write "§ 8 (item 7.1)".
* Documentation only. Suite 1,335 passing, `ruff` clean.

## 2026-09-22 (fourth entry)
- **#30** — **Stages 10d and 10e landed, and the post-tag work went in after `v1.5.1-beta`** — first as ten commits on `claude/great-pasteur-65xjgw`, then onto `test`/`master` as `5b27900` and `0c30517`.
Everything new ships OFF; re-scoring `data_samples/DOC_LINE_CATEG` changes 0 categories at every
step. Suite 1,335 → 1,418 passing, `ruff` clean.

**Measurement (digest X1–X6, Y1–Y4).**
* **10e — the adoption gate passes on the post-D33 tree**, which is the run stage 9a was owed:
errors 513 → 503, cost 0.2897 → 0.2810, `Clear`-loss 38 → 38, `Trash`-recall 12.2% → 18.9%,
fixes 12 / breaks 2, exact McNemar **p = 0.01294**. Stage 9a closes.
* **10d — a global `VOWEL_RUN_MIN` of 4 is dead.** The gate is indifferent (errors +0, cost +0,
`Clear`-loss +0, fixes 2 / breaks 2, p = 1) and both secondary measures move the wrong way
(macro_f1 −0.0020, `Trash`-recall 34/180 → 32/180). Written into the plan as "a regression guard
rather than the decision"; it was the decision.
* **`Clear`-loss is 38, not 40**, and 10e shows it is 38 in BOTH arms where 08b had 40 in both of
its own — a flag-independent shift, which is the shape D33 predicts and not one the witness can
produce. Corroboration, not proof; the cheap confirming check still stands.
* **The witness's only two errors on gold are 10d's two fixes** — the same two lines from the two
sides of the `vowel_run` clause. So everything it gets wrong is the one clause @david-spacil
identified, on the one kind of text he identified, and nothing else it does is in dispute.
* **The language split's effect is computed, not guessed (Y4).** Both breaks are `vowel_run`-only
and so are exactly two of the twelve fixes; the other ten fire `triple`, `initial_geminate` or
`low_variety`. Likely outcome **11 fixes / 1 break** — errors 503 → 504, `Clear`-loss 38 → 37. It
trades a gold-`Clear` error for a gold-`Trash` one, improving the composition and not the count,
**and the adoption gate as written rejects any +1 on errors.** The gate needs the argument, not
the split.

**Code, all shipping off.**
* **D40 — the de-gemination guard removed**, owner-approved. Probed before and after on a
full-collection table: it withdrew attestation from exactly two of the eight tokens, and neither
can be convicted either way (`jjámy` carries a diacritic, `oobjekt` is a doubled vowel the
consonant clause never matches). Live reach was nil, not small.
* **D43 — `DOMAIN_NOTATION_CATEG`**, a category NAME in config rather than a switch, because this
line has had three answers already. Every address the pattern catches gets it, and the rule runs
first in the cascade. Ships empty.
* **D44 — the vowel-run language split**: 3 vowels outside `deu,fra`, 4 anywhere. `lang` plumbed
through `determine_category()` / `categorize_line()`; `classify_TEXT` passes the RAW
`original_lang`, never the remapped `lang`, which `remap_lang()` rewrites to Czech.
* **The dedup tie-break pinned on `apply_document_postprocessing()` itself.** It was only ever
tested against the offline tool's copy of the vote.
* **`setup/word_lists.txt` — the hand-maintained word lists.** Eight lists migrated in (three from
`text_util.py`, five from `setup/config.txt`), plus **`[allowed]`**, the open-class layer that
never existed: every list the code carried was closed-class, the open-class vocabulary lived only
in `tests/`, and `SHORT_GARBAGE_LEXICON_PATH` ships empty with no frequency table anywhere in the
repository. A listed token is not debuffed in the quality score; reach beyond that is deferred to
@DanaKriv and @david-spacil. Ships empty, candidates commented.

**Four things found by doing the work, each contradicting something written first.**
1. **A test can fail by passing.** `test_the_ratio_can_be_disabled` stayed green after D40 because
`override_constants()` gates on `hasattr` and silently ignores a deleted constant. Deleted, not
kept — and it means re-running the `GEMINATE_RATIO 4.0-vs-0.0` A/B would "confirm" the removal
with a vacuous null. D28's third instance.
2. **The first URL route claimed a separation the code cannot make.** Its own test showed
`detect_fused_words` fires on the address SHAPE, correct and damaged alike, and that
`e-mail: officeauappmost.cz` already reads `Clear` without any new route.
3. **Three of eleven word lists could not migrate** — `ACADEMIC_TITLES` is matched
case-sensitively, `METADATA_MARKERS` carries load-bearing trailing spaces, `LDL_ALLOWED_FOLLOW` is
punctuation. Documented in `config.txt` where an editor will look.
4. **The migration nearly removed the override path**, caught by
`test_tier1_key_roundtrip_from_alternate_config`. Precedence is now layered: config/env, then
file, then in-code default.

## 2026-09-22 (fifth entry)
- **#30** — **Stages 10f and 11 finished, and with them the measurement programme this issue has
run since July.** What remains is decisions, not runs. Stage 11 was three joins over the gold
corpus — minutes, no re-score — and it answered every question stage 10 left open. Digest
§ "Stage 10f + stage 11" (Z1–Z6).
* **10f — per line, the witness FAILS its own gate.** `Clear`-loss 53 → 54 with the page cascade
disabled, and the tool prints REJECT; with the cascade (10e) it is 38 → 38. That reproduces 08c,
so it is confirmed twice: **the document-level dedup is a PRECONDITION for the witness**, not a
refinement. @david-spacil's § 2 answer — keep the step — is load-bearing for the flag decision.
* **11a — W6 closes.** Zero gold rows carry `ssuti`, `ssutí` or `ssutě`. `Clear`-loss 38 needs no
correction.
* **11c — X1 confirmed, and the 42 / 41 / 40 / 38 drift is accounted for at last.** Exactly two
gold-`Clear` rows are stored `Trash` and URL-shaped — both `http://www.arub.cz` — and D33 moves
exactly those two `Trash` → `Noisy`. **The stage printed the wrong verdict while confirming it**:
its pass condition counted every URL-shaped row (3) instead of those stored `Trash` (2). The rows it
printed were right; the summary written into it was not.
* **11b — Y4 was wrong, in our favour.** The four labels are `deu` / **`afr`** / `fin` / `eng`, so
only the German break is spared and every fix survives: **12 fixes / 1 break, errors 503 → 502,
`Clear`-loss 38 → 37 — both down.** The adoption gate passes, and the "decide the gate first"
recommendation is withdrawn. Y4's arithmetic was sound and its premise was not: a line that reads
German was detected Afrikaans.
* **And that is why it is fragile (Z5).** The fix survives only because of that mis-detection;
`Frauenzimmerbad` is exempted on a `deu` label at 0.359; `http://www.arub.cz` is detected
Cantonese. The label on a short damaged line is largely noise. **Open:** a confidence floor on
`_vowel_run_min_for()`, which would put the German sentence back at risk.
* **44.5% non-Czech is the wrong number to read; 5.8% is the right one (Z6).** The tail is
Vietnamese, Estonian, Xhosa and Uzbek — the detector failing, which `remap_lang()` absorbs. The
split acts only on `deu` + `fra`.
* `issue30_stage11_job.sh` written for the cluster in the stage 6–10 contract; not tracked, like
its predecessors. Documentation only in-tree. Suite 1,418 passing, `ruff` clean.

## 2026-09-22 (sixth entry)
- **#30** — **Documentation sync after comment 61; no code, flag or word-list entry changed.** The
tree was read against the thread's settled answers and against the collaborator documents, looking
for decisions already made that the repository did not yet say. Digest § "After the tag" (AT1–AT3).
* **AT1 — comment 61 overstated what `[allowed]` does.** It told @DanaKriv and @david-spacil that
listing a word stops the program discarding it. `shape_garbage_clauses()` never consults the list:
probed through `score_line()`, with `ssuti` listed and the witness on, `ssuti` alone stays `Clear`
but `ssuti vfetennl` is `Trash` on `ssuti`'s `initial_geminate` alone. The file header and
`tests/test_word_lists.py` both record the reach as "an open question for @DanaKriv and
@david-spacil" — it had never been asked. Corrected in `docs/issue30/README.md` and
`categorization_logic.md`; asked as Q5b.
* **AT2 — the open questions, asked.** Q1–Q3 for @DanaKriv (agree with W4; dissent on the dedup
trade; the 357 decisions), Q4–Q6 for both (what every recognised address should be under
`DOMAIN_NOTATION_CATEG`; which `[allowed]` candidates to arm and how far they reach; what
`rule_short_garbage` should answer on a short line that might be decipherable), Q7 for
@david-spacil (tag or `master` for the 508 re-check, since D40 and D44 landed after
`v1.5.1-beta`). Mirrored as plan rows H9–H13 and in `docs/issue30/README.md` § "Still open".
* **AT3 — drift fixed, each correction marked in place.** `categorization_logic.md` named a
`rule_domain_notation_clear` / `SHORT_LINE_URL_CLEAR_ENABLE` that does not exist — rewritten to the
real `rule_domain_notation_categ`, which gains gate row 0b; row 5c gains D44, its exemptions and the
gold-measured status; row 5b gains D33's URL shape; a note on what `[allowed]` does to the score.
Review request: "1950s" → "1940s" plus the Krofta clause (the sidecar agrees: 24 of 45 `Non-text`
rows in 1940s ARÚB documents), Y4's superseded prediction marked, § 3 warned that tag ≠ `master`,
new § 9. Annotation guide: the worked example row was not in the 157-row census (now `eaual to:`),
"94% `edit1` (9,649 of 10,246)" had no source (the two files give 102 of 113), one count fixed,
three § 6 questions re-framed from usefulness to legibility to match its own § 4.
`annotation_ask_README.md`: `Non-text` → leave blank, and "one notch too tight" replaced by the
split. Root `README.md`: the gold set exists and the gate result is stated; a note on
`setup/word_lists.txt` and `DOMAIN_NOTATION_CATEG`. `setup/config.txt`: the witness block's "15
lines cannot validate it" comment replaced by the measured status (comment only).
* Plan: H1–H3 / H5 carry real status, H6 names the right dropped option, H7's checklist item is
ticked, 11e's "README weight table" item was already done (`9121c4c`), two broken links fixed.
Issue log refreshed with comment 61.
* Suite unchanged — 1,422 passed, 13 skipped, 2 xfailed — and `ruff` clean.

## 2026-09-23
- **#30** — **@david-spacil answered Q7 and ran the re-check; it exposed a guard that had been gone
since 2026-09-18 (D45).** Digest § "D45", plan § "2026-09-23".
* **H5 closes.** `master`, default config (no lexicon), witness on: 326 → 335/508; 16 lines moved,
11 fixed / 2 broken / 3 wrong either way. Tag and `master` differ on `Frauenzimmerbad` only.
* **D45 — `cc4990e` deleted both round-2 witness guards.** `_RE_FUSED_GRID_REF` (`S-VIIIb`) and
`has_expected_lang_diacs()` were added in `9bc218b` and measured in round 2 (14/1, p = 0.00098),
then deleted that evening by a lexicon-parsing commit; `031fa58` restored the pre-round-2 config
block the same hour. A stale working copy, not a decision, and no test held either guard. Every run
from 5f on scored the guard-less witness; S4's 05a-vs-05f comparison crossed the deletion and is
marked confounded.
* **Grid guard restored** in `shape_garbage_clauses()`'s line-level veto (witness-local, inert while
the flag is off), with its history in the comment, and **pinned**: the seven-string series and a
narrowness test in `tests/test_shape_witness_vocabulary.py`, plus an `S-VIIIb` row in the wiring
test's keep list. Checked both ways — 9 failures with the guard removed, all green with it.
Expected on his run: 336/508, `Lokolieace: •VIII,` the only break (Q6's case).
* **Not restored, open for the maintainer:** the German-diacritic veto (overlaps D44, moves ~239
corpus lines, needs its cache registered and its own A/B). **Flagged:** the config lexicon block
`031fa58` put back is wrong in both its versions since W2.
* Docs: review request § 3 (✅ block with his table, the `S-VIIIb` sentence corrected, Q7 closed in
§ 9 and the banner), `docs/issue30/README.md` "Still open", `categorization_logic.md` row 5c.
Issue log refreshed with his comment.
* Suite on `origin/test` `4ed309a`: 1,441 → **1,449 passed** (the 8 new tests), 13 skipped, 2 xfailed;
`ruff` clean; `recategorize_from_csv --report-only` over `data_samples/DOC_LINE_CATEG` changes 0
categories — the flag is off, so nothing moves.

## 2026-09-23 (second entry)
- **#30** — **@david-spacil confirmed the repair: 336/508 on `3b02959`** (11:05 UTC). `S-VIIIb` is
`Clear` again and `Lokolieace: •VIII,` is the only break (15 moved: 11 fixed, 1 broken, 3 wrong
either way); with the flag off, 0/508 change against `4ed309a`, so the restored guard is inert until
the flag flips. H5 and Q7 are closed for good. He is taking Q1–Q6 through with @DanaKriv by e-mail.
* **Then a sync of the tree to the settled answers — text, plus one report fix; no behaviour
change.** No flag, threshold, `[allowed]` entry or config value moved: every `KEY = value` line of
`setup/config.txt` and every entry of `setup/word_lists.txt` is byte-identical, and
`recategorize_from_csv --report-only` over `data_samples/DOC_LINE_CATEG` still moves 0 categories.
* **The coupling advisory, re-stated.** `uncoupled_witness_warning()` (docstring and stderr text) and
the `setup/config.txt` witness block keep the rule — never arm the witness without a table, advisory
only — but its reason is now corpus exposure: 37,555 kept lines at risk with no table against 6,714
with the 113,100-document one (08f → 9b; 9b also carries D33), and what the table spares is led by
`ppole`, `ARCHAIA` and the Latin binomials. The retracted 1:3 / 1.7:1 figures (scored against
`categ`, 90.5% of the gap `ppole`) survive as one marked history line.
* **Config comments** brought up to 08d (vocabulary signal: reject), 08e (glyph stripping: null), 10d
(global vowel run 4: rejected), 05e (`MIN_DF` stays 3) and 5c (convict clause: rejected), with the
gold runs noted as predating the grid-guard restore and `[allowed]` noted as commented out and not
reaching the shape tests (Q5b).
* **Parity fix — the one code change.** `tools/short_garbage_witness_report.py` now passes each row's
raw `original_lang` to the witness, as `classify_TEXT.score_line` has since D44 (`"?"` → the strict
threshold, production's default), and its banner prints `VOWEL_RUN_EXEMPT_LANGS=` /
`VOWEL_RUN_MIN_EXEMPT=` plus a no-language note for `--lines`. Before, every row got the Czech
threshold, so German rows the gate spares (`Dauerleihe`) counted as exposure. New tests: a CSV
judged per row language, the banner guard now also scanning `_vowel_run_min_for()`, and the
`--lines` note — all fail on the old tool.
* **`ppole` is an abbreviation everywhere now** — the per-collection note `build_token_lexicon.py`
writes into every table header, its docstring and CLI note, `ocr_neighbours.py`'s CLI note (which
contradicted its own docstring), and `tests/test_ocr_neighbours.py` (a test renamed, no assertion
weakened). Dangling `30.runbook.md` pointers repointed in code, config and `tools/gold/GOLD.md`.
* **Docs.** `issue30_gold_ab_findings.md`: a D45 top section and 🛑 markers where D45, W2, D33 and
11a/11b overturned it, and its § 3 blockquote no longer swallows the paragraph it corrects. Review
request: the `3b02959` row, "should read" → confirmed, the question counts fixed, § 10 (Q8).
`docs/issue30/README.md`: Q7 at 336, a Q8 row. `RULE_COVERAGE.md` / `SWEEP_NOTES.md`: the INERT
class. `service/README.md`: the `Clear` threshold is 0.80, not 0.85. Root `README.md`: the #30
limitation dates from `v1.5.0-beta` (PR #48 merged 15:37 UTC, after `v1.4.7-beta` at 09:34).
`CONTRIBUTING.md`: an `unreleased` row for the post-tag work.
* **New, optional question — Q8 (plan H14):** are `ä`/`ö`/`ü`/`ß` reliable signs of German in this
archive, or does damaged Czech produce them? It decides whether the German-diacritic veto D45 left
out (~239 witnessed lines on the gold corpus, 156 kept today) is worth its own measurement.
* Suite 1,449 → **1,452 passed**, 13 skipped, 2 xfailed; `ruff check` and `ruff format --check`
clean; `check_version` agrees. Committed locally, not pushed.

## 2026-09-23 (third entry)
- **#31** — **`--method text-lines`: any text-bearing input → ordered pages × lines → CSV rows.**
K4TEL re-scoped the issue:
  * DOCX and PDF must be accepted as ordered pages;
  * every sequentially readable textual format should be;
  * lines become CSV rows;
  * pages and lines encode reading order and blocks for non-paged formats;
  * input handling must be strict;
  * readers should be lightweight;
  * ALTO is untouched and categorization is not run.
  Delivered as full files in chat for review — **not committed or pushed**. Plan § "Phase 3"
  (D9–D20); digest refreshed.
* **Status reconciled first.** `31.plan.md`/`31.digest.md` had said "design only" since 07-25 while
  the JSON design (D1–D8) was on `test`; #37 had since added `--force-single-page` and the E2E suite.
* **Sibling survey.** nlp-enrich has no PDF/DOCX reader (its flexiconv→TEITOK adapter is
  non-functional, nlp-enrich #10). llm-enrich's `digital_to_json.py` (pdfplumber + python-docx) is
  `digital-convert`'s: extension-only, no corrupt/encrypted handling, DOCX as one page. Reused: its
  PDF text-layer thresholds, its deterministic fixture builders, and its no-AGPL licence policy.
* **New:**
  * `text_formats.py`: content-first detection; stdlib + lxml readers for DOCX, XLSX, PPTX,
    ODT/ODS/ODP, EPUB, RTF, HTML/hOCR, PAGE XML, TEI, XML, JSON/JSONL, CSV/TSV, Markdown and TXT;
    pypdfium2 for PDF (a per-page text-layer class: none/garbled/ocr/digital);
    charset-normalizer restricted to cp1250/iso8859-2/cp1252.
  * the stage scripts `text_split.py` (pages + `ingest_report.csv`/`pages_report.csv`),
    `text_stats_create.py` (hyphen-safe ids) and `extract_TEXT_2_TXT.py` (classify-ready text +
    `DOC_LINES_TEXT/<doc>.csv` line tables whose numbering equals `DOC_LINE_CATEG`'s).
* **Robustness:** a reason code per file; zip-bomb caps checked on declared sizes; no-entity XML
  parsing; PDFs in a subprocess with a timeout; atomic page directories; doc_id collision and
  invalid-id refusal; symlink/FIFO/lock-file skipping; `--strict`.
* **Contract.** `source.origin` is truthful per class (`ocr:*` / `digital-born-<kind>`). New guard
  in `document_hook`: §1a only warns when not strict, so without it classify/aggregate would have
  written OCR categories into digital-convert's records. Positional blocks are held back for them,
  unless a page carries the `needs_ocr` hand-off.
* **Other code changes:**
  * `classify_TEXT.load_page_index` keeps `0001`/`NA` doc ids;
  * `run_pipeline` is table-driven per format, with a warning when `--input-csv` does not reach
    extract/classify (a trap that predates this work);
  * `/process` accepts all of the above as `task_type=document`;
  * `release.yml` bundles `text_formats.py` + `page_split.py`;
  * config `[TEXT_INGEST]`, two conditional `para_config` components, requirements.
* **Samples and docs:** `data_samples/TEXT/` (CTX000000004–14) and `data_samples/JSON/CTX000000015.json`
  (closes the "no JSON sample" gap). New `docs/text_inputs.md`; README Steps 1–3, API and paradata
  sections (plus three stale spots fixed); `service/README.md`; `CONTRIBUTING.md`.
* **Unchanged, verified:** `page_split.py`, the ALTO/JSON stats scripts and extractors,
  `aggregate_STAT.py` and every hub-canonical file. `run_pipeline --dry-run` for all four ALTO/JSON
  methods is byte-identical before and after.
* **Suite 1448 → 1581 passed**, 11 skipped, 2 xfailed, 0 failed. `ruff check` and
  `ruff format --check` are clean; coverage is 72.7%. `tests/test_alto_tools.py` now selects ALTO
  samples by root element (the same 10 files), so the new PAGE XML sample is not treated as ALTO.
* **Cross-repo follow-ups:**
  * hub `KNOWN_PIPELINE_SUFFIXES` (office/PDF suffixes);
  * hub `test_document_required.py` (a fifth extractor twin);
  * hub docs that call alto's input ALTO-only;
  * llm-enrich V-1;
  * the stale llm-enrich "program name" TODO.

## 2026-09-24

* **#31 on `test`.** `103e30a` "edits issue #31 - non-alto input formats added" landed the `--method text-lines` work
  delivered on 09-23 (the "not committed or pushed" note above is historical). Not in a release yet.
* **TEITOK round 4 (atrium-nlp-enrich umbrella plan, Stage 7) — audit, dev logs.** Checked this repo's TEI/TEITOK
  reader against nlp-enrich's TEITOK format 2 (v0.21.0): `read_tei` renumbers pages (ignores `<pb n>`, drops empty
  pages), ends a line at every `</s>`, and has no `</n>` repair or TEITOK fixture. nlp-enrich now depends on this repo's
  `page_num` (its UDPipe chunks were being counted as pages), and with `FLEXICONV_ANNOTATE` a `DOC_LINE_CATEG` table
  wins at its stage 1 while the layout comes from flexiconv. Seven code comments cite `atrium-project#13` where they
  mean atrium-llm-enrich #13. #31 digest + plan refreshed ("not on test" and "nlp-enrich flexiconv path
  non-functional" corrected; a TEITOK handoff section and follow-ups added).
* **#31** — K4TEL posted the `text-lines` status on the issue (06:48).
* **TEITOK follow-ups — implemented the same day (delivered as files, not yet on `test`).** `read_tei` keeps every
  `<pb/>` as a page (blank ones too) labelled with `pb@n`, and in tokenized TEITOK takes lines from `<lb/>` rather
  than `</s>`; `parse_xml_bytes` repairs the legacy `</n>` exactly (`name_close_repaired`); three new tests;
  `docs/text_inputs.md`; the seven `atrium-project#13` comments now say `atrium-llm-enrich#13`. Suite **1588 passed**
  (1585 before), 0 failed; `ruff` clean. *(Superseded: on `test` since `fb72526`.)*

## 2026-09-24 (second entry)
- **#31 Phase 4 — hardening and more dialects** (plan § "Phase 4", D21–D35; digest refreshed). K4TEL asked for the
  subtask to be re-inspected against every issue's dev logs and the sibling repos, and for its gaps in docs, code and
  config to be closed:
  * handling of users' inputs strict enough;
  * every sequentially readable textual format accepted;
  * ordered lines as CSV rows;
  * the ALTO workflow unchanged;
  * categorization never run.

  Audited at `fb72526` with all four repos fetched at their `test` HEADs. Delivered as full files in chat; the
  code, tests, config and samples landed on `test` as `267e334` (unreleased).
* **K4TEL's decisions:**
  * a per-kind origin override, `[DOCUMENT].SOURCE_ORIGIN_BY_KIND`;
  * the doc-id fix in classify/aggregate, ALTO-neutral and never run;
  * DOCX/ODT notes on their page, `[TEXT_INGEST].NOTES = page|end|skip`;
  * all four new input groups (OCR exports, ZIP bundles, compression wrappers, subtitles + e-mail).
* **New kinds:**
  * `tesseract-tsv`, `abbyy-xml` and `djvu-xml`, with native pages and `ocr:*` origins. Word and character rows are
    joined into lines; they used to come out as one word or one character per line.
  * `srt`, `vtt`, `eml` and `mbox`: a page per message, Subject then text/plain.
  * gzip/bzip2/xz wrappers, under the existing size and ratio caps.
  * `zip-bundle`: a ZIP of per-page OCR files read as one document, in natural order. Metadata and images are
    ignored, one kind is kept per stem, and PDFs and nested containers are skipped.
  * Sniffing is rewritten over bytes. `%PDF-` must lead the file, so a ZIP storing a PDF is no longer "pdf". A new
    reason code `unreadable`.
* **Reader fixes:**
  * PAGE-XML table cells and nested regions;
  * CSV/TSV stray-quote re-parse (`csv_unbalanced_quote`), and a numeric column is never the text;
  * JSON line granularity (Azure Read v3 and Textract no longer duplicate each line); JSON no longer imports pandas;
  * sheets over `MAX_LINES_PER_PAGE` continue on further pages;
  * ODS comments and numbers dropped, hidden sheets and slides counted;
  * RTF per-font code pages and surrogate pairs;
  * lone surrogates dropped (they crashed the page write);
  * TEI P4, corpora and `<choice>`;
  * EPUB `%`-hrefs;
  * lxml elements tracked by identity, not `id()`: a real flake that could skip a PAGE-XML line.
* **Strictness:**
  * ingest status `partial` (read with a lossy note), which `--strict` counts;
  * page directories: only a directory of nothing but `<doc>-<n>.txt` is ever replaced, and the previous pages of a
    failing document are removed (`stale_pages_removed`). This closes the `rmtree` hazard of output dir `.` with an
    input `setup.txt`;
  * per-document isolation in the extract and stats stages;
  * `--strict/--no-strict` on both stages and through `run_pipeline`;
  * `STRICT` validated.
* **Other code:**
  * `document_hook`: the `SOURCE_ORIGIN_BY_KIND` parser and resolver, `DIGITAL_CONVERT_KINDS`, and a docstring
    rewrite (stale program names, dead link).
  * `run_pipeline`: `[PIPELINE].INPUT_DIR_TEXT` (a bare `--method text-lines` used to ingest the ALTO samples);
    `--input-csv`, `--strict` and `--source-origin` pass-through.
  * classify/aggregate read `file` as text, and aggregate sorts naturally. `0001` stayed `1` before, and mixed ids
    crashed the sort.
  * The service honours `[TEXT_INGEST]` and the origin keys and decodes `.txt` like the batch path. Unsupported →
    400, everything else → 422, including a new `no_text`.
  * Both frontends' `accept=` lists are pinned to `supported_extensions()`.
  * Report-only page flags: `mojibake_cp1252` (llm-enrich's CP1250↔CP1252 table, made conservative), and
    `mirrored_text` / `rotated_text` from PDF matrices.
* **Config, samples, docs:**
  * `setup/config.txt` (`SOURCE_ORIGIN_BY_KIND`, `NOTES`, `INPUT_DIR_TEXT`), `.gitignore`, `.env.example`;
  * `data_samples/TEXT/CTX000000016–24`: Tesseract, ABBYY, DjVu, SRT, VTT, EML, MBOX, `.txt.gz` and a bundle;
  * `docs/text_inputs.md` rewritten (detection order, matrix, bundles, notes, blank pages, page ids, reports and
    statuses, reason codes with HTTP statuses, flags, the origin override);
  * `README.md`, `CONTRIBUTING.md`, `service/README.md`, `data_samples/README.md`;
  * plan header repaired; digest title typo "othet" fixed.
* **Unchanged, verified:**
  * the ALTO/JSON scripts (`page_split.py`, both stats scripts, the three ALTO extractors, `extract_JSON_2_TXT.py`)
    and every hub-canonical file;
  * `run_pipeline --dry-run` byte-identical to `fb72526` for eight ALTO/JSON variants;
  * classify's re-read and aggregate's sort replayed byte-identical on `data_samples/DOC_LINE_CATEG(_gpt)`;
  * strict E2E on the 20 text samples: all `ok`, 94/94 line-table rows match, every `doc.json` validates;
  * a hostile-input directory gave a reason for every file and wrote nothing outside `out/`.
* **`267e334` turned Paradata Canonical Drift red.** The hub-canonical `tests/test_env_contract.py` held the contents
  of `tests/env_contract_data.py` (a file mapped to the wrong path on apply), and the data file kept its old text.
  Both were re-sent: the test byte-identical to hub `v1` (`eec0682`), the data file in its Phase 4 version. A local
  replay of the drift loop passes all 17 manifest files. The same follow-up carries the docs and dev logs and an Apple
  `.pages` fix: a package with a preview image is `archive_unsupported`, not `image_needs_ocr`.
* **Suite 1584 → 1707 passed**, 11 skipped, 2 xfailed, 0 failed (baseline at `fb72526` in the same environment).
  `ruff check` and `ruff format --check` are clean. Categorization was not run: torch and fasttext are not installed.
* **Cross-repo follow-ups:**
  * hub `KNOWN_PIPELINE_SUFFIXES` (office/PDF/new suffixes; `.md` is already there);
  * hub `test_document_required.py` (the text writers; drifted line references);
  * ALTO-only hub docs;
  * the hub `ORIGIN_ORIGINATORS` `digital-born` prefix, which covers kinds digital-convert cannot read;
  * llm-enrich V-1.
* _Status 2026-09-24 (round 5):_ the ALTO-only hub docs are fixed (see the next entry).

## 2026-09-24 (third entry): pushed; #30 answers; round 5 — input formats reference

* **Pushed and green.** Phase 4 landed as `267e334`, then `3bd10f9` (the env-contract test restored — Paradata
Canonical Drift green again) and `2e2794d` (docs); `test` = `master` = `2e2794d`, CI green on both. Unreleased.
* **#30** — two answers arrived on 2026-09-23 after the last revision: @david-spacil's Q8 (5795411354: ~73 % of short
umlaut lines are German, ~13 % misread Czech, ~10 % rubbish — a German-letter exemption would keep a mild existing
error, so the left-out veto is worth measuring) and @DanaKriv's Q3 (5798302434: the 357 decisions by the end of next
week, ≈ 2026-10-02). Recorded in the #30 digest and plan; nothing switched on.
* **Round 5 (docs only in this repo).**
  * `docs/text_inputs.md` opens with **Formats and their standards**: for every adopted input (ALTO, PAGE XML, hOCR,
    ABBYY FineReader XML, DjVuXML, Tesseract TSV, OCR JSON, PDF text layers, TEI/TEITOK; office, EPUB, RTF, HTML,
    Markdown, CSV, JSON, text, subtitles, e-mail, bundles) the standard and its steward, typical producers, what the
    format records (pages, lines, coordinates, confidences, typography), what this repo keeps and the `source.origin`;
    and where the coordinates go (atrium-nlp-enrich's TEITOK takes boxes from ALTO or flexiconv conversions only).
  * **Found:** `page_split.py` splits ALTO in the **v3 namespace only** — a v2/v4 file gets `No <Page> elements found`
    and no pages (verified on synthetic v2/v3/v4 files). Documented in §8 and a README note; the fix (derive the
    namespace from the root, as `service/utils.py` does) is proposed, not made.
  * §2 *Page ids* now says that nlp-enrich numbers TEITOK pages by their order in the ALTO file, not by
    `PHYSICAL_IMG_NR`, and that `DOC_LINE_CATEG` has no `page_label`.
  * README: intro sentence on the other inputs, the Step 1 pointer, the ALTO v3 note, one broken anchor
    (`#-paradata-logging`) fixed; CONTRIBUTING's unreleased row notes the docs.
  * Dev logs: #2/#3/#4 (milestones; the rules and docs moved on with #30), #30 (answers), #31 (merge state, round 5),
    **#37 ready to close** (llm-enrich's `json_to_md.py` reads a json-keys record without an adapter, checked).

## 2026-09-25: #31 Phase 5 — the remaining gaps, and a first run on real engine output

* **Asked:** cover all remaining gaps of #31; deliverables are full files in chat (no patch files, no remote actions),
  so nothing below is pushed. Decisions with K4TEL: this repo + hub + llm-enrich (page-classification joined for two
  tests), json-keys fixed by default, `ORIGIN_ORIGINATORS` kept, only our own Tesseract output committed, and the hub
  suffix list extended although `CTX01.scan.pdf` becomes `CTX01.scan`.
* **ALTO/JSON traps closed (D36–D44):** `page_split.py` splits every ALTO version and skips non-ALTO roots by name,
  and a re-split removes stale page files; hyphenated doc ids survive the stats stage; json-keys reads only the split
  page, each text once (`page_text_lines`, also the service's JSON path); **found** — the stats rows followed the
  directory listing, so `content.text` could start with page 2 — rows are in page order now; `0001` ids reach every
  extractor; `--input-csv` reaches extract and classify, and text-lines has `[EXTRACT].INPUT_CSV_TEXT`; resume only
  while the output is newer than its input (classify re-classifies a changed document, lists foreign outputs), and
  split/extract rewrite a page file only when it changed, so an unchanged full re-run still skips; the
  dry-run tags page_split `[paradata]`; the dead commented-out `split_json_document` is gone.
* **Real engine output (D45–D47):** Tesseract 5.3.4 installed in-session; two rendered Czech pages → TSV/hOCR/ALTO read
  exactly; **found** — PDFium's line-end hyphen marker (`\x02`, U+FFFE) was stripped and merged two lines; restored;
  **found** — a Tesseract ALTO was `ABBYY-ALTO`: text-lines now records the engine an ALTO/hOCR file names. New
  samples `CTX000000025.tsv`/`CTX000000026.hocr` + `tools/make_tesseract_samples.py`. Four AWS Textract responses and
  Azure's example result (in-session only): both JSON methods give exactly the engine's LINEs; json-keys used to write
  3–6× as many.
* **Cross-repo (D48–D49), as files:** hub `KNOWN_PIPELINE_SUFFIXES` (+ skeleton mirror, tests), `test_document_required.py`
  (function anchors, the text-lines writers), V-1 docs, the origin decision; llm-enrich V-1 (`DROP_CATEGORIES` =
  `UNTRUSTWORTHY_LINE_CATEGORIES` — the hub had called it done on 09-16, the code had not changed); page-classification's
  two doc-id tests.
* **Verified:** suite 1707 → 1750 passed, 0 failed; `ruff` clean; coverage 75.4 %; dry-run diffs only as intended; v3
  ALTO/JSON splits byte-identical; strict text-lines E2E 22/22 `ok`, 128/128 line-table rows, 22 valid records; json-keys
  and alto-tools E2E (v2, v4, `0001`, hyphenated, Tesseract ALTO) correct and in page order; the new hub files swapped
  into all five tool repos move only page-classification's two tests. Categorization not run.
* **Still open:** the re-vendor sequence (hub merge → retag `v1` → three files into five repos, page-classification's
  tests with them); nlp-enrich's TEITOK page numbering (Pitfall 14, certain for Tesseract ALTO); ABBYY/DjVu XML
  synthetic-only; a classify output left by a crash counts as current.
