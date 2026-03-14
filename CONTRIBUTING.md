# 🤝 Contributing to the ALTO XML Postprocessing Pipeline of the ATRIUM project


Welcome! Thank you for your interest in contributing. This repository [^8] provides a 
robust workflow for transforming raw OCR outputs (ALTO XML) into clean and classified 
textual data. It addresses common challenges in digital archives, such as multi-column 
layout reconstruction, word-split recovery, and automated quality filtering.

The next step in the pipeline: [atrium-nlp-enrich](https://github.com/ufal/atrium-nlp-enrich)

This document describes the project's capabilities, development workflow, code conventions, 
and rules for contributors.


## 📦 Release History

| Version    | Highlights                                                                                                                                                                                                                           | Status      |
|:-----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------|
| **v0.9.1** | ALTO via (LayoutReader + language ident. + perplexity) = textlines categorized (Options to get text from ALTO XML, Split words fix and record to output, Text quality category assignment per text line)                             | Pre-release |
| **v0.8.0** | API service draft (Added LINDAT API service and interface setup, Change of files structure logic)                                                                                                                                    | Pre-release |
| **v0.7.0** | GLM added and LR results fixed for TextLine level (LayoutReader is processing on TextLine level now -> fix of broken line splits, Added GLM text extraction option (requires images), Added data samples of text)                    | Pre-release |
| **v0.6.0** | LayoutReader added and KER removed (alto-tool switched to LayoutReader per String XML element, KER removed, Extracted text post-processing implemented)                                                                              | Pre-release |
| **v0.5.0** | alto-tool extraction and result samples (KER scores explanation is included in documentation, Results samples for classified textlines are included, Only alto-tool extraction)                                                      | Pre-release |
| **v0.4.0** | Removal of API calls + result recording per document (Merged NER and UDP -> moved API calls to separate repository, Per-document result files saving)                                                                                | Pre-release |
| **v0.3.0** | CPU-GPU-based division into substeps + separate TXT files extraction (CPU/GPU division of script steps, Removed CSV expansion with text, Added config for textlines classification, alto-tools as an extractor of TXT from ALTO XML) | Pre-release |
| **v0.2.0** | Per-line categorization + LINDAT API calls (Text classification update, Switch to per-textline model calls, Option of a summary stats CSV with raw texts per cell is present)                                                        | Pre-release |
| **v0.1.0** | Per-page categorization + KER fix + LINDAT API calls (Language identification and other text processing are called per-page, KER raw suffixes fixed, NER + UDP calls included)                                                       | Pre-release |

---

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
and **DistilGPT2** [^6] for perplexity scoring.

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

### Code Conventions

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

# 2. Pre-commit hooks (runs black, isort, flake8, etc.)
pre-commit run --all-files

```

> [!NOTE]
>  If specific scripts or extraction modules are updated, please run a smoke-test 
> against the `data_samples/` directory to verify extraction integrity.

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
  * DistilGPT2 [^6]
  * GLM-4v-9b [^10]
  * LayoutLMv3 [^9]

**©️ 2026 UFAL & ATRIUM**


[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^3]: https://github.com/ufal/ker
[^4]: https://atrium-research.eu/
[^5]: https://github.com/ufal/atrium-nlp-enrich
[^6]: https://huggingface.co/distilbert/distilgpt2
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^7]: https://ufal.mff.cuni.cz/home-page
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b
