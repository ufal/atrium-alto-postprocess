# 🤝 Contributing to the ALTO XML Postprocessing Pipeline of the ATRIUM project


Welcome! Thank you for your interest in contributing. This repository [^8] provides a 
robust workflow for transforming raw OCR outputs (ALTO XML) into clean and classified 
textual data. It addresses common challenges in digital archives, such as multi-column 
layout reconstruction, word-split recovery, and automated quality filtering.

The next step in the pipeline: [atrium-nlp-enrich](https://github.com/ufal/atrium-nlp-enrich)

This document describes the project's capabilities, development workflow, code conventions, 
and rules for contributors.

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
test  ←  feature/<issue>
test  ←  bugfix/<issue>
master   ←  (humans only, after test stabilises)

```

### 🏷️ Branch Naming

| Type             | Pattern           | Example                        |
|------------------|-------------------|--------------------------------|
| New feature      | `feature/<issue>` | `feature/42-api-integration`   |
| Bug fix          | `bugfix/<issue>`  | `bugfix/17-layout-split-error` |
| Hotfix on master | `hotfix/<issue>`  | `hotfix/99-api-timeout`        |

---

## 🔁 Contributor Workflow

1. **Create an issue** (or find an existing one) describing the problem or feature.
2. **Branch from `test`:**
```bash
git checkout test
git pull origin test
git checkout -b feature/<issue-number>

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

*Note: Do not open PRs into `master` — merging into `master` is exclusively the maintainers' responsibility.*

---

## ✏️ Commit Messages

Format:

```
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

* **Formatting:** Python code should be formatted using `black` (line length 120) and `isort`.
* **Linting:** Ensure compliance with `flake8`.
* **Docstrings:** Use descriptive, concrete language avoiding generic templates (e.g., 
avoid simply writing "Return value of the function").

### Minimum checks before every commit

Always run basic validation locally before pushing:

```bash
# 1. Python compilation check
python -m compileall -q .

# 2. Pre-commit hooks (runs black, isort, flake8, etc.)
pre-commit run --all-files

```


*Note: If specific scripts or extraction modules are updated, please run a smoke-test 
against the `data_samples/` directory to verify extraction integrity.*

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
