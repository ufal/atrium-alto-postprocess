**# 📓 atrium-alto-postprocess — agent_dev_logs/DEVLOG.md (history seed)
> _OCR/ALTO post-processing + line categorization. 5 open issues. `test` HEAD `b4bd545` (2026-06-25)._

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
- **#5** — Commit `b4bd545`: parameterized eight page-context smoothing thresholds (config-driven), removed the dead `rule_short_fragment_noisy` + `CLEAR_BAND_WC_MIN`, and the legacy `CLEAN_PROSE_*` near-boundary constants; new sweep output posted; further reviews from GPT-5.5, GPT-5 and Gemini DR Pro 3.1 (the "Survivor Bias" framing, refactoring direction).

---

# 📓 atrium-project — agent_dev_logs/DEVLOG.md (history seed)
> _Hub/planning repo. Reconstructed from 15 open issues. `test` HEAD `11ba0ff` (2026-06-24)._

## 2026-03-13
- **#4 SSH Open Marketplace records** — Opened by stranak: create SSHOMP records for every tool in our workflows (UDPipe ✅, NameTag ✅, rest TBD).
- **#6 Review & summarise licenses** — Opened by stranak (review tool+model licenses, check where CC-BY-NC-SA is required). K4TEL posted the first license inventory: FastText/AMCR-vocab CC BY-NC, layoutreader CC BY-NC-SA, distilgpt2/alto-tools/GLM/Qwen2.5 Apache-2.0, ViT/EffNet/RegNet/CLIP MIT, NameTag3/CUBBITT CC BY-NC-SA, UDP2 MPL-2.0, AISCR Teater GPL-3.0.
- **#9 Paradata of outputs** — Opened by K4TEL: unified run-logging (incl. output license) across all four tool repos.

## 2026-03-15
- **#4** — Page classifier added to SSHOMP as a Suggested Tool under the `ATRIUM catalogue` keyword.
- **#9** — Translator, textline & page classifiers tested with paradata output; basic `.json` paradata in all repos via a shared `atrium_paradata.py`.
- **#10 LLM validation of source code** — Opened by K4TEL (validate every repo's source with an LLM).

## 2026-03-22
- **#10** — All projects checked with Sonnet 4.6 Extended, then re-checked with Gemini 3.

## 2026-03-25
- **#13 CAA Proceedings paper to PCJ** — Opened by K4TEL: submit a paper to the CAA2026 proceedings / PCI Archaeology; text draft posted (5000-word limit, no figures yet).

## 2026-03-26
- **#13** — Added the full project diagram, an updated report PDF with the diagram inserted, and a Zenodo submission draft.

## 2026-04-04
- **#13** — Overleaf editor invites sent to David and Dana; CAA-proceedings project + Springer extended-version project to be reformatted into CAA styles.

## 2026-04-11
- **#4** — The remaining three repositories suggested as SSHOMP tools.

## 2026-04-16
- **#4** — ALTO post-processor, NLP enrichment, translator and page classifier all uploaded as tool-or-service under the **ATRIUM catalogue** tag.

## 2026-05-13
- **#13** — motyc: proceedings deadline is **31 October 2026**.

## 2026-05-27
- **#15 Submission to IJDL** — Opened by motyc (review ASAP, link in the minutes).
- **#16 List ARUP/B data storage locations** — Opened by motyc (so ARUP/B can later remove all copies). K4TEL listed the `data_samples` dirs across repos, the LINDAT annotated dataset, thesis/presentation page samples, and the UFAL filesystem.
- **#17 Review SSHOMP workflow descriptions** — Opened by motyc.
- **#18 Docker compose + GH action wrapper for CU forks** — Opened by motyc (links the four ARUP-CAS forks).

## 2026-05-28
- **#9** — Mass→single-file paradata records merged per repo; open questions on license source, missing tool-version tag, dynamic runner reference, and a Docker-image placeholder.
- **#10** — Slated for re-examination by Opus 4.7 and Sonnet 4.6 across all four repos.
- **#17** — K4TEL posted the four marketplace tool links; motyc thanked; noted relation to #4.

## 2026-05-29
- **#9** — Detailed per-repo license breakdown: the tool-vs-model split (NameTag3/UDPipe engines MPL-2.0 but their models CC BY-NC-SA), Teater app GPL vs data CC BY-NC, and the internal-academic-use vs external-commercial-use distinction.
- **#10** — motyc: "Opus 4.8 is just out :)".
- **#16** — Posted per-repo licensed-asset tables (alto 39, nlp 34, translator 14, page-classification 84 documents) mapped to licenses from the global metadata collection.

## 2026-06-02
- **#9** — The two easy repos (translator, page-classification) updated with paradata licenses; the two multi-step repos (alto, nlp) remain (sequential-log aggregation); alto full-pipeline commit landed.

## 2026-06-03
- **#9** — nlp-enrich commit adds licensed paradata for API scripts + keyword extraction (LLM samples to follow).

## 2026-06-08
- **#16** — Full current-state inventory of every `data_samples/` dir; alto & nlp **resolved to contain only synthetic data**; translator still holds 16 real ARUP/B source documents; page-classification has ~245 PNGs across 11 category folders.

## 2026-06-10
- **#6** — License summary (from #9) implemented for all four repos; TODO to attach the list to the SSHOMP workflows.
- **#9** — Only nlp-enrich remains (LLM samples); all-stage merging done.
- **#18** — Opus strategy: repos are already pre-wired — `atrium_paradata.py` reads `ATRIUM_RUNNER_IMAGE/REPO/REF`, so GHCR-published self-identifying containers are the plan.

## 2026-06-12
- **#9** — Merged paradata for nlp stages 1–4 + one keyword method; all seven checklist items marked done.
- **#10** — Plan to review each repo with Fable by 22 June.
- **#18** — Per-repo Docker drafts summarised (shared template, per-repo knobs); motyc: discuss orchestration with rharasim, no overall wrapper needed (containers reachable via API).
- **#21 LINDAT annotated dataset release** — Opened by K4TEL: two ways to fix the licensing problem (modify old handle vs publish new + redirect); per-file metadata fields; sample JSON/CSV; motyc OK with option 1, notes some files can't be openly published (metadata-only).

## 2026-06-14
- **#21** — Posted the 82 GB ready-to-publish `licensed_archives/` listing: `CITATION.cff`, CC BY-NC `LICENSE`, per-document licensed CSV/JSON, cross-val folds, category ZIPs, and a `not_included` CSV for disallowed-license files.

## 2026-06-15
- **#10** — Released alto v0.17.0, page-classification v1.4.0-beta, translator v0.6.0, nlp v0.12.0 with LLM-review edits applied (Fable was unavailable 😮‍💨).
- **#18** — translator & page-classification passed GH Actions; posted the "Align & Expand Docker + GHA" strategy (one reusable workflow template + thin per-repo callers).

## 2026-06-16
- **#9** — Old paradata files to be replaced and `para_config` versions bumped across all four repos.
- **#10** — Defined the next review round's aspects: Docker+GHA, merged pipeline & API, per-function test coverage, architecture, file tree, CONTRIBUTING release history, + a per-repo review plan.
- **#18** — Commit `676a1fe` lands the centralized DRY CI/CD (`ci-cd-strategy.md`, `docker-tool.reusable.yml`, caller example, shared `.coveragerc`/`ruff.toml`, dependabot appendix); all four repos pass GHA; docs updated for rharasim to test.

## 2026-06-17
- **#10** — Combined per-repo review plan committed (`aba539e`); posted the post-review validation matrix (Tier-1 compileall/ruff + Tier-2 pytest/coverage, run pc→alto→nlp→translator).
- **#22 Document Understanding eval** — Opened by K4TEL (benchmark for document understanding — OmniDocBench?). Gemini "Deep Research" report posted; Opus 4.8 follow-up corrected its fabrications/mis-attributions, separated parsing-fidelity from semantic understanding, flagged **CHURRO/CHURRO-DS** as the real historical-doc match, and recommended an OOTB-VLM-vs-legacy-pipeline comparison first.

## 2026-06-19
- **#4** — SSHOMP tool records updated with license tables.
- **#6** — TODO to attach license lists to the marketplace workflows; new versions to be set by admins on the default tool views.
- **#21** — Major licensing discussion: 318 unpublishable files (<0.01%) removed; CC BY-NC vs BY-NC-SA debated (stranak/motyc lean to dropping SA → plain NC, citing EOSC/Open-Access policy); tombstone + "incomplete dataset" metadata text drafted; link replacements queued for arXiv/README/Zenodo. stranak already published the record; license to be swapped.
- **#22** — stranak: when running a big VLM (e.g. MiniMax-M3), contact Viktor about vLLM on the reserved Grace Hopper machine.
- **#24 LLM applications to data** — Opened by K4TEL (various local/remote LLM tasks).

## 2026-06-20
- **#21** — motyc proposed Description-field text (318 files, accessible under conditions at digiarchiv, GitHub repo for full pipeline).
- **#26 Run models larger than GPU memory via CPU** — Opened by K4TEL (explore unified-memory mechanism).
- **#27 H100 multi-GPU runs** — Opened by K4TEL (MiniMax-M3 FP8 ~440 GB on a single multi-GPU node).

## 2026-06-21
- **#6** — Admins to update default tool versions; license tables added to each SSHOMP description.
- **#10** — Opus 4.8 review round: new findings — `/info` version drift, `para_licenses.py` diverged + zero tests, nlp ruff blocking, secret-scanning unverified; posted a phased strategy.
- **#18** — Further GHA-integration strategy (Codecov gate bug, `@main` vs `@test` pin drift, action version floor, per-repo P0/P1/P2); all four repos released as vX.Y.Z+1 passing ruff/pre-commit.
- **#26** — Opus recommendation: vLLM `cpu_offload_gb` (UVA zero-copy) over Ollama layer-split or raw CUDA UVM — memory-only offload keeps CPU cores free for the existing queue.
- **#27** — Opus recommendation: 8×80 GB H100 SXM5, vLLM/SGLang tensor-parallel-size 8 + expert-parallel + fp8 KV cache, capped `max-model-len`; support is brand-new (use nightly/Docker).

## 2026-06-22
- **#21** — kosarko, stranak and motyc debate whether the corrected dataset record even needs the 318-files warning (agreed it belongs on the *models*/tombstone, while keeping it discoverable via the `not_included` CSV).

## 2026-06-23
- **#10** — `docs/plan_repo_review.md` declared the canonical plan to execute across the whole ecosystem.
- **#13** — Handle/DOI to be replaced in the Overleaf bibliography (marked DONE).
- **#15** — Dataset reference to be replaced in the post-review IJDL edit; arXiv preprint to be updated.
- **#16** — #21 designated the canonical "where licensed samples are shared" reference; motyc: keep open until end of project.
- **#21** — Links updated in both arXiv papers, the README, and the Zenodo DOI (one un-editable spot remains: the official CU MFF thesis record).

## 2026-06-24
- **#21** — kosarko refined the Description wording (bolded "318 files" claim to fact-check); K4TEL confirmed the 318 count via `wc -l` on the CSVs; stranak proposed keeping the original dataset (restricted, incl. the 318 files) **plus** a CC-only derived subset, linked together.

## 2026-06-25
- **#15** — arXiv `2606.07558` updated with the new dataset link (references only).
- **#16** — Both arXiv versions (`2507.21114`, `2606.07558`) updated with the new dataset licensing link.
- **#29 Add `agent_dev_logs` directory per repo** — Opened by K4TEL (this initiative): per-repo markdown dev logs on `test`, seeded from issue history, replacing agent work-documentation in issue comments.
