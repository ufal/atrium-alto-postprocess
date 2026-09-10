# 📓 atrium-alto-postprocess — agent_dev_logs/DEVLOG.md (timeline index)
> _OCR/ALTO post-processing + line categorization. 7 open issues (#2, #3, #4, #23, #30, #31, #37); #5/#6 closed. `test` HEAD `4017a76` (2026-09-09), `master` `ebaec0a` (4 behind) · **v1.4.6-beta** released; next tag needs the version bump in `CITATION.cff` + `setup/para_config.txt`._
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

---
_Timeline index refreshed 2026-09-09 against live `test`/`master` HEAD, the current release list, open-issue state
via the GitHub API, and the refreshed `30.digest.md`/`37.digest.md`. Nothing removed from the issues themselves
(per hub #29); this file is a derived reading aid in `agent_dev_logs/`._
