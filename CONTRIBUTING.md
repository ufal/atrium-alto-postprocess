# 🤝 Contributing to the ALTO XML Postprocessing Pipeline of the ATRIUM project


Welcome! Thank you for your interest in contributing. This repository [^8] provides a
robust workflow for transforming raw OCR outputs (ALTO XML) into clean and classified
textual data. It addresses common challenges in digital archives, such as multi-column
layout reconstruction, word-split recovery, and automated quality filtering.

The next step in the pipeline: [atrium-nlp-enrich](https://github.com/ufal/atrium-nlp-enrich)

This document describes the project's capabilities, development workflow, code conventions,
and rules for contributors.

## 📦 Release History

> [!NOTE] This repository is currently in the `v0.x` pre-release phase. The path to a stable `v1.0.0` will be triggered
> once the `run_pipeline.py` batch orchestrator and the `service/text_api.py` FastAPI wrapper are fully load-tested in production and the `atrium-project` monorepo reaches its stable milestone.*

| Version     | Highlights                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Status      |
|:------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **v0.20.0** | Added ablation study of the parameters and rules in the categorization. Updated paradata-related scripts with atrium-project template. Added agent_dev_logs directory with issue logs + their digests and plans.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Pre-release |
| **v0.19.2** | GHA ruff and pre-commit checks + fixed possible GPU bugs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Pre-release |
| **v0.19.1** | Docker GH Actions alignment of workflows + ruff and pre-commit checks                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Pre-release |
| **v0.19.0** | Categorization logic improvements on trash-Noisy-Non-text differences and routing - updated config contents                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Pre-release |
| **v0.18.1** | Next round of LLM review edits + Docker GH Actions alignment                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Pre-release |
| **v0.18.0** | Diagnostic rule flags and exact symbol occurrence counting (Added 9 boolean diagnostic columns to track which categorization rule or post-processing override decided the final label, modified `detect_strange_symbols` to return exact occurrence counts of disallowed internal characters, extracted `determine_category` for reason-tag tracking, updated unit test coverage and documentation). And Docker wrapper update to fix GH actions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Pre-release |
| **v0.17.0** | Fixes and improvements applied according to the LLm review of this repository                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Pre-release |
| **v0.16.0** | Language remapping changed in the categorization logic, preservation tests added for the intermediate steps of text lines processing, merged pipeline of the whole functionality is added                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Pre-release |
| **v0.15.5** | Synthetic data samples, license-aware paradata, and end-to-end pipeline (Replaced all real ARUP/ARUB records in `data_samples/` with 3 fully synthetic demo docs — `CTX000000001`/`2`/`3`, covering clear, mixed-quality, and short/poor OCR — and regenerated every derived directory `ALTO`/`PAGE_ALTO`/`PAGE_TXT`/`_LR`/`_LLM`/`DOC_LINE_CATEG`/`_gpt`/`DOC_LINE_STATS`/`_gpt` plus the moved summary CSVs, resolving the restricted-record licensing issue; Reworked paradata into a per-run **effective output license** resolver — the output license is computed as the most-restrictive among components actually used, driven by `para_config.txt` (component→license) and `para_licenses.py` (ranking + share-alike / non-commercial rules), with the custom GLM-4 model license registered and a `tool_version` tag; Made the text-extraction scripts fully config-driven via the new `[EXTRACT]` section of `config_langID.txt` instead of hardcoded constants (and fixed undefined-variable crashes in the LayoutReader/GLM paradata logging); Added `run_pipeline.py`, a config-driven orchestrator (`[PIPELINE]` section, LayoutReader default) that runs split→stats→extract→classify→aggregate sequentially and merges all per-stage paradata into one run summary describing the stages, intermediate formats, and end-to-end license. | Pre-release |
| **v0.15.4** | Developer docs and test coverage (Added pytest unit tests, Expanded `CONTRIBUTING.md` with testing guidelines and release history section)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Pre-release |
| **v0.15.3** | Categorization based on Qwen2.5-0.5B, bugs fixed, files cleaned (Space-collapsing bug fix in `pre_filter_line`, Short Czech labels mis-Trashed fix, Repeat detector missed garble fix, Score bands tightened: Trash ≤0.50, Noisy [0.50–0.90), Clear ≥0.90, Config cleanup: removed unused keys, added `QS_WEIGHT_VOWEL`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Pre-release |
| **v0.15.2** | Categorization based on quality score computed from all factors (Switched from distilgpt2 to Qwen2.5-0.5B — 4–5× larger multilingual model, Category Clear/Noisy/Trash derived from quality score thresholds, Extended config controls with weighted QS parameters)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Pre-release |
| **v0.15.0** | Perplexity by Qwen2.5 and distilgpt2 + categorization logic update (Added Qwen2.5-0.5B for perplexity measures alongside existing distilgpt2, Changed categorization logic to depend on quality score, Quality score computed from a weighted sum of factors)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Pre-release |
| **v0.13.0** | Extended gibberish detection and logging (Service functionality updated to match main repo, Added many additional values per text line to log in result files, Characters now play a role in gibberish detection, GPU worker for perplexity via distilgpt2, CPU workers for fasttext/regex, Draft of semi-final categorization logic)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Pre-release |
| **v0.11.1** | Summary includes quality scores + LLM-enhanced (Weird words ratio and quality score averages per page recorded in output summary files, LLM-enhanced code)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               | Pre-release |
| **v0.11.0** | Quality score and per-word measures (Categorization logic change, Added quality score column in output, Added weird words ratio in output, Paradata logging included)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Pre-release |
| **v0.9.1**  | ALTO via (LayoutReader + language ident. + perplexity) = textlines categorized (Options to get text from ALTO XML, Split words fix and record to output, Text quality category assignment per text line)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | Pre-release |
| **v0.8.0**  | API service draft (Added LINDAT API service and interface setup, Change of files structure logic)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Pre-release |
| **v0.7.0**  | GLM added and LR results fixed for TextLine level (LayoutReader is processing on TextLine level now -> fix of broken line splits, Added GLM text extraction option (requires images), Added data samples of text)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Pre-release |
| **v0.6.0**  | LayoutReader added and KER removed (alto-tool switched to LayoutReader per String XML element, KER removed, Extracted text post-processing implemented)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Pre-release |
| **v0.5.0**  | alto-tool extraction and result samples (KER scores explanation is included in documentation, Results samples for classified textlines are included, Only alto-tool extraction)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | Pre-release |
| **v0.4.0**  | Removal of API calls + result recording per document (Merged NER and UDP -> moved API calls to separate repository, Per-document result files saving)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Pre-release |
| **v0.3.0**  | CPU-GPU-based division into substeps + separate TXT files extraction (CPU/GPU division of script steps, Removed CSV expansion with text, Added config for textlines classification, alto-tools as an extractor of TXT from ALTO XML)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Pre-release |
| **v0.2.0**  | Per-line categorization + LINDAT API calls (Text classification update, Switch to per-textline model calls, Option of a summary stats CSV with raw texts per cell is present)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Pre-release |
| **v0.1.0**  | Per-page categorization + KER fix + LINDAT API calls (Language identification and other text processing are called per-page, KER raw suffixes fixed, NER + UDP calls included)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Pre-release |

## 🏗️ Project Contributions & Capabilities

This pipeline contributes 4 major stages to the data processing lifecycle, as detailed in the
section of the main [README 🛤️ Workflow Stages](README.md#-workflow-stages)

### 1. Granular Data Management
The pipeline allows archives to move from document-level files to page-level management.
* **Splitting:** Automatically breaks down document-level ALTO XMLs into individual page files.
* **Page Inventory:** Generates a foundational CSV statistics table capturing for every page in an archive:
  * text line counts
  * number of illustrations and graphics

### 2. Multi-Method Text Extraction
Archive managers can choose extraction methods based on their specific hardware and accuracy requirements:

| Method                    | Best For...               | Resource Type   | Key Feature                                                        |
|---------------------------|---------------------------|-----------------|--------------------------------------------------------------------|
| **LayoutReader** [^9]     | Multi-column layouts      | **GPU**         | Reconstructs natural reading order and handles hyphenated words.   |
| **alto-tools** [^1]       | Fast, large-scale batches | **CPU**         | High speed, low memory usage for raw content overview.             |
| **GLM (LLM-based)** [^10] | Degraded/Complex scans    | **GPU** (48GB+) | Generative OCR directly from images; patches transcription errors. |

### 3. Automated Quality & Language Assessment
A core contribution of this project is the ability to filter "noisy" OCR data without manual review.
Every text line is categorized using **FastText** [^2] for language identification
and **Qwen2.5-0.5B** [^6] for perplexity scoring (or `distilgpt2` for English-only collections).

**Data Quality Categories:**
* ✅ **Clear:** High-quality, fluent text ready for research.
* ⚠️ **Noisy (Rough):** Contains minor OCR errors; usable but imperfect.
* 🗑️ **Trash:** Unintelligible OCR or non-prose content.
* 🔣 **Non-text:** Mostly digits or symbols.
* 🫙 **Empty:** Whitespace only.

### 4. Lightweight API & Testing Interface
The project includes a **FastAPI** service that allows for easy integration into existing
archival systems. It provides:
* **RESTful Endpoints:** `/process` and `/info` for remote file processing.
* **Visual Testing:** A built-in JS frontend for immediate manual verification of model performance.

---

## 🌿 Branches & Environments

| Branch   | Environment          | Rule                                                                            |
|----------|----------------------|---------------------------------------------------------------------------------|
| `test`   | Staging              | Base for all development. Always branch from `test`.                            |
| `master` | Stable / Integration | Merged exclusively by a human reviewer. Do not open PRs directly into `master`. |

```text
test    ←  feature-<name>
test    ←  bugfix-<name>
master  ←  (humans only, after test stabilises)

```

### 🏷️ Branch Naming

| Type             | Pattern          | Example                |
|------------------|------------------|------------------------|
| New feature      | `feature-<name>` | `feature-regex-factor` |
| Bug fix          | `bugfix-<name>`  | `bugfix-chunking`      |
| Hotfix on master | `hotfix-<name>`  | `hotfix-reqs-modules`  |

---

## 🔁 Contributor Workflow

1. **Create an issue** (or find an existing one) describing the problem or feature.
2. **Branch from `test`:**
```bash
git checkout test
git pull origin test
git checkout -b feature-<name>
```
3. **Implement your changes** observing the project's code conventions.
4. **Run the minimum tests** (see the Testing section).
5. **Open a Pull Request** targeting the `test` branch.

---

## 📋 Pull Request Format

Every PR must include:

* **Issue link:** `Closes #<number>` or `Refs #<number>`
* **Motivation:** why the change is needed
* **Description of change:** what was changed and how
* **Testing:** what was run, what passed, what could not be executed

Use a **Draft PR** if the work is not ready for review.

**Do not open PRs into `master` — merging into `master` is exclusively the
maintainers' responsibility.

> **Note on issue tracking:** Issues reference the commits and PRs that resolved
> them — not the other way around. Commit messages describe *what changed*; the issue
> is the place to record *why* and link the resulting commits together.

---

## ✏️ Commit Messages

Format:

```text
[type] concise description of what changed
```

Allowed types:

| Type       | When to use                           |
|------------|---------------------------------------|
| `add`      | Added content (general)               |
| `edit`     | Edited existing content (general)     |
| `remove`   | Removed existing content (general)    |
| `fix`      | Bug fix                               |
| `refactor` | Refactoring without behaviour change  |
| `test`     | Adding or updating tests              |
| `docs`     | Documentation only                    |
| `chore`    | Build, dependencies, CI configuration |
| `style`    | Formatting, no logic change           |
| `perf`     | Performance optimisation              |

---


## 🧪 Code Conventions & Testing

### Code Conventions & Linting

This repository has standardized on **Ruff** for all linting and formatting,
replacing legacy tools (`flake8`, `black`, `isort`).

Before committing, ensure your code complies by running the pre-commit hooks.
This prevents CI failures in the `test` branch:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

If you need to run the linter manually, use:

```bash
ruff check . --fix
ruff format .
```

* **Comments:** informative but short, may be LLM-generated, added when function name does
not explain its functionality in detail
* **Argument types:** set default type (e.g., `int`, `list`) for function arguments
* **Console flags:** when a new one added, provide help message for it
* **Config files:** when set of variables changes it should be reflected in repository documentation
* **Generated code:** always should be manually launched and checked for mistakes before pushing

### Minimum checks before every commit

Always run basic validation locally before pushing:

```bash
# 1. Python compilation check
python -m compileall -q .

# 2. Pre-commit hooks (runs Ruff, etc.)
pre-commit run --all-files

```

> [!NOTE]
>  If specific scripts or extraction modules are updated, please run a smoke-test
> against the `data_samples/` directory to verify extraction integrity.

---


### Running the test suite

The repository ships a lightweight `pytest` harness that requires **no ML models or GPU**
for standard unit tests. Heavy tests that do require models or network access are marked
`slow` and are excluded from the default run.

```bash
pip install -r requirements-test.txt  # pytest>=8.0 and pytest-cov only
```

```bash
pytest -m "not slow" --tb=short                              # fast — use before every commit
pytest --tb=short                                            # full suite (requires model setup)
pytest -m "not slow" --cov=. --cov-report=term-missing      # with coverage
```

`tests/test_paradata.py` (`ParadataLogger`, `_sanitise`) is shared across all repos.
Repo-specific modules and GPU-heavy tests are marked `@pytest.mark.slow` and skipped by default.

<details>
<summary>Test layout, per-repo targets, and fixture conventions</summary>

```text
tests/
├── __init__.py              # empty
├── conftest.py              # shared fixtures (tmp_path wrappers, sample data loaders)
├── fixtures/                # small static test-data files committed to the repo
└── test_<module>.py         # repo-specific unit tests
```

**Per-repo targets:**

| Repository                | Test file           | Primary targets                                                                                                                                                                                |
|---------------------------|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `atrium-nlp-enrich`       | `test_keywords.py`  | `_extract_surface_text`, `_extract_lemmas`, `_extract_legacy`, `extract_keywords`, `_sort_csv_file`                                                                                            |
| `atrium-alto-postprocess` | `test_text_util.py` | Density/ratio helpers, detectors, `pre_filter_line`, `parse_line_splits`, `determine_category` (reason-tag coverage), `categorize_line` (ppl passed directly, no GPU), `compute_quality_score` |
| `atrium-alto-postprocess` | `test_utils.py`     | `directory_scraper`, `dataframe_results` (Top-1 and Top-N), `collect_images`                                                                                                                   |
| `atrium-translator`       | `test_utils.py`     | `_resolve_namespaces`, `validate_xml_with_xsd`, `process_alto_xml`, `process_amcr_xml` (mock translator injected)                                                                              |

**Slow tests** — any test loading a model checkpoint, calling an external API, or requiring a GPU must be decorated with `@pytest.mark.slow`. Document in the PR description which resource it requires and how to enable it locally.

**Fixtures** — small, self-contained files committed under `tests/fixtures/`. Tests must not read from `data_samples/` directly. Add a minimal fixture file in the same commit as any test that needs new sample data.

</details>

---


## 📁 Repository Documentation Management

Each documentation file has one target audience and one responsibility. Rules are not repeated — cross-references are used instead.

| File              | Audience        | Responsibility                                 |
|-------------------|-----------------|------------------------------------------------|
| `README.md`       | GitHub visitors | Project overview, workflow stages, quick start |
| `CONTRIBUTING.md` | Developers      | Code conventions, branches, PRs, testing       |

* **Do not duplicate rules:** if a rule is defined in `CONTRIBUTING.md`, other files
reference it rather than copying it.
* **When changing a rule:** update the canonical source and verify that referencing files
still point correctly.

---

## 📞 Contacts & Acknowledgements

For support or specific archival integration questions, contact **lutsai.k@gmail.com**.

**Issues:** https://github.com/ufal/atrium-alto-postprocess/issues

* **Developed by:** UFAL [^7]
* **Funded by:** ATRIUM [^4]
* **Models:**
  * FastText [^2]
  * Qwen2.5-0.5B [^6] (default; `distilgpt2` for English-only collections)
  * GLM-4v-9b [^10]
  * LayoutLMv3 [^9]

**©️ 2026 UFAL & ATRIUM**


[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^4]: https://atrium-research.eu/
[^5]: https://github.com/ufal/atrium-nlp-enrich
[^6]: https://huggingface.co/Qwen/Qwen2.5-0.5B
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^7]: https://ufal.mff.cuni.cz/home-page
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b
