<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" title="Python Version"></a>
  <a href="https://huggingface.co/facebook/fasttext-language-identification"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-fasttext--langID-yellow.svg" title="FastText Language Identification"></a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-0.5B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Qwen2.5--0.5B-yellow.svg" title="Qwen2.5-0.5B Perplexity"></a>
  <a href="https://github.com/cneud/alto-tools"><img src="https://img.shields.io/badge/dep-alto--tools-lightgrey.svg" title="alto-tools"></a>
  <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/github/license/ufal/atrium-alto-postprocess" title="MIT License"></a>
  <a href="https://atrium-research.eu/"><img src="https://img.shields.io/badge/funded%20by-ATRIUM-8A2BE2.svg" title="ATRIUM Project"></a>
</p>

---

# 📦 ALTO XML Files Postprocessing Pipeline

This project provides a complete workflow for processing **ALTO XML** 📄 files. It takes raw ALTO
XMLs and transforms them into structured **statistics tables** 📊, performs text classification,
and filters low-quality **OCR** 🔍 results.

The core of the quality filtering relies on **language identification** 🌐 and a composite **quality
score** 📈 — combining structural detectors, **perplexity** 📉, and character-level metrics — to identify
and categorize noisy or unreliable **OCR** 🔍 output.

---

## 📖 Table of Contents

- [⚙️ Setup](#-setup)
- [🛤️ Workflow Stages](#-workflow-stages)
  - [🚀 Run the whole pipeline at once](#-run-the-whole-pipeline-at-once)
  - [Step 1: Split Document-Specific Inputs into Pages ✂️](#-step-1-split-document-specific-inputs-into-pages-)
  - [Step 2: Create Page Statistics Table 📈](#-step-2-create-page-statistics-table-)
  - [Step 3: Extract text from ALTO XML ⛏️](#-step-3-extract-text-from-alto-xml-)
    - [LayoutReader method 📐](#1st-choice-layoutreader-method-)
    - [alto-tools method 🧰](#2nd-option-alto-tools-method-)
    - [GLM method 🤖](#3rd-alternative-glm-method-llm-based-)
    - [json-keys method 🔑](#4th-alternative-json-keys-method-generic-json-input-31--37-)
  - [Step 4: Classify Page Text Quality & Language 🗂️](#-step-4-classify-page-text-quality--language-)
    - [4.1 Classify Lines (GPU Bound) 🚀](#41-classify-lines-gpu-bound-)
      - [Categorisation logic reference 📚](docs/categorization_logic.md)
    - [4.2 Aggregate Statistics (Memory Bound) 🧠](#42-aggregate-statistics-memory-bound-)
  - [Paradata logging 🗒️](#paradata-logging)
    - [Output licensing ⚖️](#output-licensing-)
- [Acknowledgements 🙏](#acknowledgements-)

---

## ⚙️ Setup

Before you begin, set up your environment.

1.  Create and activate a new **virtual environment** 🖥️ in the project directory.
2.  Install the required **Python** 🐍 packages:
    ```bash
    pip install -r setup/requirements.txt
    ```
3. Download the **FastText** 🌐 model for language identification:
    ```bash
    wget "[https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin](https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin)" -O lid.176.bin
    ```
4. Clone and install `alto-tools` 🔧, which is used for statistics and text extraction in low memory environments:
    ```bash
    git clone [https://github.com/cneud/alto-tools.git](https://github.com/cneud/alto-tools.git)
    cd alto-tools
    pip install .
    cd ..
    ```
5. Copy the `v3` folder from the 📐`layoutreader` 🔧 repository [^9] to the project directory for the LR-based text extraction method:
    ```bash
    git clone [https://github.com/ppaanngggg/layoutreader.git](https://github.com/ppaanngggg/layoutreader.git)
    cp -r layoutreader/v3/ ./
    rm -rf layoutreader/
    ```

You are now ready to start the workflow.

---


## 🛤️ Workflow Stages

The process is divided into sequential steps, starting from raw **ALTO** 📄 files and ending
with extracted linguistic and statistic data 📊.

You can run the **entire pipeline end-to-end** with a single command (see below), or run each
stage individually as described in Steps 1–4.

---

### 🚀 Run the whole pipeline at once

The [run_pipeline.py](run_pipeline.py) 🐍 orchestrator runs every stage sequentially
(**split → statistics → text extraction → classification → aggregation**) and, at the end,
**merges all per-stage paradata** 🗒️ logs into a single run summary describing every stage, the
intermediate file formats produced, and the **effective end-to-end output license** ⚖️ (see
[Paradata logging](#paradata-logging)).

```bash
python3 run_pipeline.py                      # all settings from config.txt
python3 run_pipeline.py --method glm         # override just the extraction backend
python3 run_pipeline.py --skip-split         # PAGE_ALTO already populated
python3 run_pipeline.py --dry-run            # print the resolved plan, run nothing
```

* **Configuration ⚙️:** every setting is read from [config.txt](setup/config.txt) 📎
(section `[PIPELINE]`, with `INPUT_CSV` taken from `[EXTRACT]`). Precedence is
**CLI flag > config value > built-in default**. Point at a different config with `--config`
or the `LANGID_CONFIG` environment variable.
* **Extraction method 🔀:** `[PIPELINE] METHOD` selects the **Step 3** backend —
`alto-tools`, `layoutreader` (**default**), or `glm`. The choice flows through to the merged
license: a **LayoutReader** 📐 run resolves to **CC BY-NC-SA 4.0**, an **alto-tools** 🧰 run to
**CC BY-NC 4.0**.
* **Output 📤:** a merged `<YYMMDD-HHmmss>_pipeline-run.json` in the [paradata](paradata) 📁
directory, alongside the individual per-stage logs.

> [!NOTE]
> `page_split.py` (Step 1) does not emit paradata of its own, so a full run typically merges
> **four** logged stages (Steps 2–4 plus aggregation). The merged license is re-derived from the
> **union** of components used across all stages, so the end-to-end most-restrictive rule holds.

> [!TIP]
> Prefer to inspect or re-run a single stage? The individual scripts below remain fully usable on
> their own — the orchestrator simply calls them in order.

---

### ▶️ Step 1: Split Document-Specific Inputs into Pages ✂️

First, ensure you have a directory 📁 containing your document-level input files. This script
will split them into individual page-specific files — it supports **both** of the pipeline's
input formats, dispatched automatically by file extension.

```
python3 page_split.py <input_dir> <output_dir>
```

Each page-specific file retains the header from its original source document 📌.

#### ALTO XML input

* **Input 📥:** `../ALTO/` (input directory with **ALTO XML** 📄 documents)
* **Output 📤:** `../PAGE_ALTO/` (output directory with **ALTO XML** 📄 files split into pages)

Example of the output directory with divided per-page XML files: [PAGE_ALTO](data_samples/PAGE_ALTO) 📁.

```
PAGE_ALTO/
├── <file1>
│   ├── <file1>-<page>.alto.xml
│   └── ...
├── <file2>
│   ├── <file2>-<page>.alto.xml
│   └── ...
└── ...
```

#### Generic JSON input (#31)

Real OCR/Doc-AI engines don't agree on how they represent multi-page documents, so the JSON
path is detected heuristically rather than assuming one vendor's schema: a nested page-list
container (e.g. Azure Document Intelligence's/docTR's `pages` array), a flat element list
tagged with a per-item page field (e.g. AWS Textract), or — when neither pattern is found —
today's single-page-per-file default (e.g. pero-ocr, OCR.space).

* **Input 📥:** `../JSON/` (input directory with generic JSON OCR-engine 📄 documents)
* **Output 📤:** `../PAGE_JSON/` (output directory with **JSON** 📄 files split into pages)

```
PAGE_JSON/
├── <file1>
│   ├── <file1>-<page>.json
│   └── ...
├── <file2>
│   ├── <file2>-<page>.json
│   └── ...
└── ...
```

#### `source.origin` — how the original input was acquired

This is the pipeline's **first** stage to see the original file, so when
`[DOCUMENT].JSON_DIR` is configured it is also the first writer of the record's `source`
block (`sha256`, `filename`, `media_type`, `page_count`, `origin`). `source` is
immutable — first writer wins — and since the `atrium_document` §1a hardening its
`origin` is what **authorises** this repo to write the positional blocks
(`pages`/`content`/`lines`/`tables`), so that no document can end up with half an
OCR-derived plane and half a digital-born one. A value that matches no known originator
prefix silently switches that check off, which is why the resolution is verified and a
mismatch warns 📣.

| Input     | Default `origin` | Meaning                                                                    |
|-----------|------------------|----------------------------------------------------------------------------|
| ALTO XML  | `ABBYY-ALTO`     | ALTO from the ABBYY toolchain the extractors already assume (see `extract_LytRdr_ALTO_2_TXT.py`) |
| JSON      | `ocr:generic`    | Generic OCR/Doc-AI export whose specific engine the file does not name      |

Override it when the engine **is** known — the prefix must stay one this repo owns
(`ABBYY-ALTO`, `ocr:<engine>`, `vlm:<engine>`):

```
python3 page_split.py <input_dir> <output_dir> --source-origin ocr:pero
```

Also settable as `[DOCUMENT].SOURCE_ORIGIN` in [`setup/config.txt`](setup/config.txt) 📎
or via the `DOCUMENT_SOURCE_ORIGIN` env var, so orchestrated `run_pipeline.py` runs
(which invoke this script with no extra CLI flags) can set it too. Precedence: CLI flag >
env var > config value > per-format default.

---

### ▶️ Step 2: Create Page Statistics Table 📈

Next, use the output directory from Step 1 as the input for this script to generate a
foundational **CSV** 📊 statistics file.

```
python3 alto_stats_create.py <input_dir> -o output.csv
```

This script writes a **CSV** 📊 file line-by-line, capturing metadata for each page:

```
file, page, textlines, illustrations, graphics, strings, path
CTX200205348, 1, 33, 1, 10, 163, /lnet/.../A-PAGE/CTX200205348/CTX200205348-1.alto.xml
CTX200205348, 2, 0, 1, 12, 0, /lnet/.../A-PAGE/CTX200205348/CTX200205348-2.alto.xml
...

```

The extraction is powered by the **alto-tools** 🔧 framework [^1](https://github.com/cneud/alto-tools).

* **Input 📥:** `../PAGE_ALTO/` (input directory with **ALTO XML** 📄 files split into pages from Step 1)
* **Output 📤:** `output.csv` (table with page-level statistics and paths to ALTO files)

> [!IMPORTANT]
> This statistics table is the basis for subsequent processing steps.
> Example: [test_alto_stats.csv](data_samples/test_alto_stats.csv) 📎.

---

### ▶️ Step 3: Extract text from ALTO XML ⛏️

This script runs in parallel ⚡ (using multiple **CPU** 💻 cores) to extract text from **ALTO XMLs** 📄 into `.txt` 📝 files.
It reads the **CSV** 📊 from Step 2.

* **Input 1 📥:** `output.csv` (from Step 2)
* **Input 2 📥:** `../PAGE_ALTO/` (input directory with **ALTO XML** 📄 files split into pages from Step 1)
* **Output 📤:** `../PAGE_TXT/` or `../PAGE_TXT_LR/` (directory containing raw **text** 📝 files)

#### 1st choice: LayoutReader method 📐

> [!CAUTION]
> The model responsible for spatial layout 📐 analysis requires a **GPU** 🚀 to run efficiently.

```
python3 extract_LytRdr_ALTO_2_TXT.py
```

Uses the **LayoutReader** 📐 framework [^9](https://github.com/ppaanngggg/layoutreader) to extract text and bounding boxes of **XML** 📄 elements
(specifically, `<TextLine>` elements containing `String`s with `CONTENT` attribute),
process them to reconstruct the reading order of lines (columns-friendly), handle words split
between two lines (adding the full form of the word), and group page contents into paragraphs
based on the vertical spread of text lines.

Example of per-page text files: [PAGE_TXT_LR](data_samples/PAGE_TXT_LR) 📁.

```
PAGE_TXT_LR/
├── <file1>
│   ├── <file1>-<page>.txt
│   └── ...
├── <file2>
│   ├── <file2>-<page>.txt
│   └── ...
└── ...
```

---

#### 2nd option: alto-tools method 🧰

> [!NOTE]
> The method is **CPU** 💻-bound and faster than the LayoutReader method, but the text lines may not be in the correct
> reading order, and full forms of hyphenated split words are not reconstructed.

```
python3 extract_ALTO_2_TXT.py
```

Uses the `alto-tools` 🔧 framework [^1](https://github.com/cneud/alto-tools) to extract text lines from **XML** 📄 elements directly,
with no post-processing. Suitable for a quick overview of raw text content.

Example of per-page text files: [PAGE_TXT](data_samples/PAGE_TXT) 📁.

```
PAGE_TXT/
├── <file1>
├── <file2>
│   ├── <file2>-<page>.txt
│   └── ...
└── ...
```

---

#### 3rd alternative: GLM method (LLM-based) 🤖

> [!WARNING]
> The method is **GPU** 🚀-bound, slower than the LayoutReader method, and requires a `gpuram48G` card.

```
python3 extract_LLM_ALTO_2_TXT.py
```

Uses the **GLM-4v-9b** 🤖 multimodal large language model [^10](https://huggingface.co/THUDM/glm-4v-9b) to perform generative **OCR** 🔍 directly from
page images, prompted as `Transcribe all text on this page exactly as it appears`. The script
trims whitespace and resizes high-resolution images to fit model constraints.

> [!NOTE]
> This method is significantly slower than parsing **XML** 📄 but often yields higher quality text for complex
> layouts 📐 or degraded scans. It patches the transformers configuration to run the GLM-4v architecture.

Example of per-page text files: [PAGE_TXT_LLM](data_samples/PAGE_TXT_LLM) 📁.

```
PAGE_TXT_LLM/
├── <file1>
├── <file2>
│   ├── <file2>-<page>.txt
│   └── ...
└── ...
```

---

#### 4th alternative: json-keys method (generic JSON input, #31 / #37) 🔑

> [!NOTE]
> Use this method with the [Generic JSON input](#generic-json-input-31) split from Step 1 —
> the other three methods above all consume **ALTO XML** 📄 instead.

```
python3 extract_JSON_2_TXT.py
```

Reads each page's generic OCR/Doc-AI **JSON** 📄 and walks a whitelist of informative keys
(`content`, `text`, `line`, `word`, ... — no assumption about a particular vendor's schema
beyond "text lives under a key named roughly `text`/`line`/`word`"), yielding every string
leaf in document order. This is the **Extraction Layer**: it makes no other change to the
JSON and does not know about `doc.json` at all.

Example of per-page text files: [PAGE_TXT_JSON](data_samples/PAGE_TXT_JSON) 📁.

```
PAGE_TXT_JSON/
├── <file1>
│   ├── <file1>-<page>.txt
│   └── ...
├── <file2>
│   ├── <file2>-<page>.txt
│   └── ...
└── ...
```

##### `doc.json` accretion (issue #37)

When [`[DOCUMENT].JSON_DIR`](#-paradata-logging) (or the `DOCUMENT_JSON_DIR` env var) is
configured, a second, separate **Accretion Layer** runs after extraction: for every document,
it reads back that document's already-written `.txt` pages and merges them into
`<doc_id>.document.json` as this repo's owned fields — `pages[].ocr`
(`alto-postprocess`'s share of the field-split `pages` block) and the whole `content` block —
leaving every other block (`page_categories`, `entities`, `source`, ...) untouched, per the
`atrium_document.schema.json` paired-hook contract. With no baseline `doc.json` yet on disk,
the record is created holding just this contribution (accretion rule 3). This mirrors what
the other three extraction methods above already do.

`--force-single-page` is a **document-assembly policy**, not an extraction-rule change — it
only decides how many `pages[]` rows describe the pages already extracted above:

```
python3 extract_JSON_2_TXT.py --force-single-page
```

|                       | `pages[]` shape                                                                                                                             | `content.text`                                                           |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| default               | one row per source page (`page: "1"`, `"2"`, ...)                                                                                           | always the full document: every page's text concatenated in source order |
| `--force-single-page` | every source page collapses into **one** row (`page: "1"`), with `ocr.source_pages` listing the original page labels in concatenation order | unchanged — same joined text either way                                  |

`--force-single-page` can also be set as a config-file default —
`[EXTRACT].FORCE_SINGLE_PAGE_JSON = true` in [`setup/config.txt`](setup/config.txt) — so
`run_pipeline.py` orchestrated runs (which invoke this script with no extra CLI flags) can
still opt in. The CLI flag takes precedence over the config value when both are given.

---

### ▶️ Step 4: Classify Page Text Quality & Language 🗂️

This is a key ⌛ time-consuming step that analyzes the **text quality** 📈 of each page line-by-line,
assigning each line a quality category to filter out **OCR** 🔍 noise.

It uses the [FastText language identification model](https://huggingface.co/facebook/fasttext-language-identification) 🌐
and **perplexity** 📉 scores from [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 🤖 to detect noise [^2](https://huggingface.co/facebook/fasttext-language-identification) [^6](https://huggingface.co/Qwen/Qwen2.5-0.5B).

More post-processing of **TXT** 📝 files can be found in the [GitHub repository](https://github.com/ufal/atrium-nlp-enrich)
of the ATRIUM project, which covers NLP enrichment using Nametag for NER and UDPipe for CONLL-U files with lemmas & POS tags [^5](https://github.com/ufal/atrium-nlp-enrich).

As the script processes, it assigns each line one of five categories 🪧:

| Category        | Action                                             | Description                                                                                                                                                                    |
|-----------------|----------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ✅ **Clear**     | Ready to be processed by further NLP               | Passes all structural checks; high composite **quality score** 📈.                                                                                                             |
| ⚠️ **Noisy**    | Corrections of generally readable words are needed | Partially degraded: moderate **quality score** 📈 indicating isolated symbol issues, fused tokens, mid-word uppercase, or elevated **perplexity** 📉.                          |
| 🗑️ **Trash**   | Should be re-processed by another **OCR** 🔍 tool  | Severely corrupted: composite **quality score** 📈 below the Trash threshold, or routed here by an override (unreadable all-caps line, inverted-scan page block).              |
| 🔣 **Non-text** | May be checked for identifiers of finds/sites      | Filtered by the CPU 💻 pre-filter: line is too short, has too few unique symbols, contains fewer than 30% alphabetic characters, or consists mostly of digits and punctuation. |
| 🫙 **Empty**    | Can be ignored                                     | Line contains only whitespace (paragraphs separator)                                                                                                                           |

> [!NOTE]
> This script generates two primary output directories:
> `DOC_LINE_LANG_CLASS/` and `DOC_LINE_STATS/`, while the
> raw **text** 📝 files (primary input) are stored in `../PAGE_TXT/` generated from `../PAGE_ALTO/`.

All input/output paths and tunable parameters are configured ⚙️ in [config.txt](setup/config.txt) 📎.
Parameters are organized into **three sections**: `[CLASSIFY]`, `[AGGREGATE]`, and `[TEXT_UTILS]`.

```ini
[CLASSIFY]
BATCH_SIZE = 128        # Batch size for processing lines
WORKERS_MAX = 32        # Max CPU workers for parallel tasks
EXPECTED_LANGS = ces,deu,eng    # Expected languages (ISO codes); first is default
TRUSTED_FOREIGN_LANGS = deu,eng,fra,pol,ita     # Allowed foreign languages (ISO codes)
MODEL_NAME = Qwen/Qwen2.5-0.5B  # Language model for perplexity scoring; English-only collections: distilgpt2

[TEXT_UTILS]

QS_WEIGHT_VALID_WORD  = 0.35    # Weight for valid word ratio in QS
QS_WEIGHT_WEIRD       = 0.18    # Weight for inverted word weirdness in QS
QS_WEIGHT_PERPLEXITY  = 0.08    # Weight for inverted normalized perplexity in QS
QS_WEIGHT_LENGTH      = 0.02    # Weight for length reward in QS
QS_WEIGHT_GARBAGE     = 0.18    # Weight for inverted garbage density in QS
QS_WEIGHT_VOWEL       = 0.07    # Weight for vowel quality in QS
QS_WEIGHT_LANG        = 0.05    # Weight for language confidence in QS
QS_WEIGHT_GIBBERISH   = 0.04    # Weight for inverted gibberish ratio in QS
QS_WEIGHT_FUSED       = 0.03    # Weight for inverted fused word ratio in QS
QS_LENGTH_MAX         = 100.0   # Max length for normalization

CATEG_TRASH_SCORE_MAX       = 0.55      # Max QS for Trash category
CATEG_NOISY_SCORE_MAX       = 0.80      # Max QS for Noisy category (#3 2026-07-02: lowered 0.85 -> 0.80)
REPEATED_DOUBLE_MIN         = 2         # Minimum occurrence count for doubled-char penalty
SHORT_NOISY_QS_PENALTY      = 0.20      # Opt-in QS penalty for short strings exhibiting OCR oddities

# --- New since last revision: Phase-2 categoriser overrides ---
LOWPPL_CLEAR_MAX            = 50.0      # ppl ceiling for Override 3 (was hardcoded)
HARD_SWEEP_LANG_MAX         = 0.45      # orig_lang_score ceiling for the hard-sweep route
HARD_SWEEP_PPL_MIN          = 1000.0    # ppl floor for the hard-sweep route
GHOST_DOMINATED_MIN_RATIO   = 0.5       # min ghost-token share to flag ghost_dominated
WORD_W_PENALTY              = 0.20      # per-word weirdness penalty for tokens containing 'w'
ROT_HIGH_LANG_CONF          = 0.90      # lang_score ceiling for the page-level rotation arm

```

Parameters that scale with the **perplexity** 📉 model:

These parameters must be re-tuned whenever you switch between multilingual `Qwen2.5-0.5B`🤖 and English-adapted `distilgpt2`🤖,
because the two models produce **perplexity** 📉 on very different numerical scales — `Qwen2.5-0.5B`🤖 assign scores roughly 3× lower
than `distilgpt2`🤖 on the same Czech 🇨🇿 text:

| Parameter                  | Qwen2.5-0.5B | distilgpt2 | What it controls                                                                                                                                                                                                                                                                      |
|----------------------------|--------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PERPLEXITY_THRESHOLD_MAX` | 1000.0       | 3000.0     | The ceiling used to normalise raw **perplexity** 📉 into [0, 1] for the **quality score** 📈. A value at or above this ceiling contributes 0 to the score (worst); a value of 0 contributes 1 (best).                                                                                 |
| `SHORT_PPL_CAP`            | 850.0        | 2500.0     | Maximum **perplexity** 📉 applied to 1–2 word lines before quality scoring. Short text fragments receive extreme **perplexity** 📉 scores from any LM because there is no context to condition on; this cap prevents legitimate short labels and codes from being unfairly penalised. |
| `PPL_INVERTED_MIN`         | 200.0        | 500.0      | **Perplexity** 📉 floor for the inverted-scan detection arm. A line is considered a candidate for the inverted-scan penalty only if the LM is also uncertain about it (**perplexity** 📉 above this value).                                                                           |
| `CLEAN_PROSE_PPL_MAX`      | 400.0        | 1000.0     | Maximum **perplexity** 📉 a line may have to qualify for the near-boundary `Clear` promotion (Override 4). Lines with **perplexity** 📉 above this value are not promoted even if all other conditions are met.                                                                       |

Parameters that are model-independent 🤖 and stable across different choices of **perplexity** 📉 model 🤖:

These parameters are expressed as ratios or quality-score fractions, not as **perplexity** 📉 values, so their meaning
does not change between models and their defaults are stable across either choice:

| Parameter                   | Default | What it controls                                                                                                                                                                                                                                                            |
|-----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ROT_RATIO_INVERTED_MIN`    | 0.55    | Minimum fraction of structurally rotatable characters (`pbqdnuwmoxszeyv`) among alphabetic characters that must be present before a rotation penalty is even considered. A value of 0.55 means more than half of all letters in the line must belong to this ambiguous set. |
| `WEIRD_RATIO_INVERTED_MIN`  | 0.35    | Minimum mean per-word weirdness score required to *confirm* an inverted scan when `rot_ratio` is already above the threshold. This second condition prevents Czech 🇨🇿 sentences that happen to contain many `p`, `d`, `b`, `q` letters from being falsely penalised.      |
| `CLEAN_PROSE_MIN_SCORE`     | 0.65    | Lower bound of the quality-score range within which the near-boundary promotion (Override 4) can fire. A line must score at least this well before it is a candidate for promotion from `Noisy` to `Clear`.                                                                 |
| `CLEAN_PROSE_WEIRD_MAX`     | 0.08    | Maximum mean per-word weirdness a line may have to qualify for the near-boundary promotion. Even a single notably corrupted token disqualifies the line from being promoted.                                                                                                |
| `CLEAN_PROSE_WC_MIN`        | 4       | Minimum word count a line must have to qualify for near-boundary promotion. Very short lines (1–3 words) have unreliable **perplexity** 📉 scores and are therefore never promoted regardless of their **quality score** 📈.                                                |
| `MOSTLY_READABLE_VALID_MIN` | 0.85    | Minimum ratio of structurally valid words required. Semi-readable lines dipping below this ratio are capped at `Noisy` and prevented from achieving `Clear`.                                                                                                                |

> [!NOTE]
> The `CLEAN_PROSE_*` rows above (`CLEAN_PROSE_PPL_MAX`, `CLEAN_PROSE_MIN_SCORE`, `CLEAN_PROSE_WEIRD_MAX`,
> `CLEAN_PROSE_WC_MIN`) parameterise the near-boundary **"Override 4"** clean-prose promotion, which has been
> **removed** from `determine_category()` (see the callout in
> [Categorisation Logic](docs/categorization_logic.md#categorisation-logic)). These
> keys — together with the never-implemented `CLEAR_BAND_WC_MIN` guard — have now been **removed from
> `config.txt`** as well (#7 Phase 0 of the config-coverage audit); they are **not read** by any current
> scoring or categorisation path. The rows are kept here only as historical documentation of the removed override.

Language- and collection-specific data 🇨🇿 moved from hardcoded Python literals into the config (#7 Tier 1). Defaults
are bit-identical to the previous in-code values, so the shipped config produces exactly the same categorisation:

| Parameter                   | Section        | Default                                                               | What it controls                                                                                                                                                                                                                   |
|-----------------------------|----------------|-----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DEU_DIACS`                 | `[TEXT_UTILS]` | `äöüßÄÖÜ`                                                             | German diacritic glyphs 🇩🇪; together with `CZ_DIACS` rebuilds the per-language diacritic map used by `infer_lang_from_diacritics()`.                                                                                             |
| `DIACRITIC_INFER_THRESHOLD` | `[TEXT_UTILS]` | 0.07                                                                  | Minimum diacritic share among alphabetic characters for diacritic-based language inference.                                                                                                                                        |
| `WQX_CHARS`                 | `[TEXT_UTILS]` | `wqxWQX`                                                              | Letters rare in Czech 🇨🇿 — wqx-heavy tokens signal OCR noise in `score_word`, `score_words_in_line` and `determine_category`.                                                                                                    |
| `ROT_WHITELIST`             | `[TEXT_UTILS]` | `po,pod,do,od,on,ony,by,bez,ne,nebo,ven,den,zde,se,ve,mez,pouze,bude` | Czech 🇨🇿 function words recognisable upright; their mirror/rotation ghost images (`ROT_GHOSTLIST`) are **derived at import time** — changing this key requires re-import (`override_constants()` does not rebuild it).           |
| `GHOST_WORD_COLLISIONS`     | `[TEXT_UTILS]` | `no,bo`                                                               | Ghost images that collide with real words and must never count as ghost hits.                                                                                                                                                      |
| `TRAILING_FILL_CHARS`       | `[TEXT_UTILS]` | `\x20._:-<\u2013\u2014`                                               | Trailing filler characters stripped before headline/short-line checks. Unicode-escape decoded — the leading space is written as `\x20` because configparser strips leading whitespace from values.                                 |
| `NONTEXT_MARKERS`           | `[TEXT_UTILS]` | `IVerc`                                                               | Collection-specific literal markers (ARUP/B stamp) forcing the `Non-text` route in `pre_filter_line()`.                                                                                                                            |
| `FASTTEXT_MODEL`            | `[CLASSIFY]`   | `lid.176.bin`                                                         | Path to the **FastText** 🌐 language-ID weights loaded by each CPU worker.                                                                                                                                                         |
| `TRUST_TIER_TRUSTED`        | `[CLASSIFY]`   | 0.85                                                                  | Trust multiplier on the **FastText** 🌐 confidence for a *known but unexpected* language. The product (`trust_lang_score`) is what feeds **both** the **quality score** 📈 and the structural gates — not the stored `lang_score`. |
| `TRUST_TIER_UNKNOWN`        | `[CLASSIFY]`   | 0.50                                                                  | Trust multiplier for an *unknown* language. Because it caps `trust_lang_score` at 0.50 — below `LANG_SCORE_REMAP` (0.75) — gates of the form `lang_score <= LANG_SCORE_REMAP` are always true for unknown-language lines.          |
| `REMAP_KEEP_SCORE_LANGS`    | `[CLASSIFY]`   | `slk`                                                                 | Languages that keep their original **FastText** 🌐 confidence when remapped to the default language (Slovak ≈ Czech 🇨🇿, so the confidence stays meaningful after the label swap).                                                |

---

#### 4.1 Classify Lines (GPU Bound) 🚀

This script reads the extracted **text** 📝 files, batches lines together 📦, and runs the **FastText** 🌐 and
**Qwen2.5-0.5B** 🤖 models. It uses a **CPU** 💻/**GPU** 🚀 split architecture:

* A single dedicated **GPU** 🚀 worker holds the only **Qwen2.5-0.5B** 🤖 instance and processes **perplexity** 📉 batches to
prevent VRAM OOM errors.
* Multiple **CPU** 💻 workers (up to `WORKERS_MAX`, default 32) read files, run **FastText** 🌐 and structural detectors, and
submit text batches to the **GPU** 🚀 worker via a shared queue. **CPU** 💻 workers poll the result dictionary while the GPU
processes, running **language identification** 🌐 concurrently.

> [!WARNING]
> The **first** item of `EXPECTED_LANGS` list of languages 🌐 should be the most expected language in the processed
> collection to work as a default replacement of ambiguous language recognition predictions.

```bash
python3 classify_TEXT.py

```

* **Input 1 📥:** `../PAGE_TXT/` from Step 3
* **Input 2 📥:** `output.csv` from Step 2
* **Output 📤:** `DOC_LINE_LANG_CLASS/` containing per-document **CSVs** 📊 (e.g., [DOC_LINE_CATEG](data_samples/DOC_LINE_CATEG) 📁)

> [!TIP]
> This script is resume-capable. If interrupted, run it again and already-present output files will be skipped.

`<doc_name>.csv` 📊: Detailed classification results for every single line within a document, **columns**:

* `file` — document identifier 🆔
* `page_num` — page number 📄
* `line_num` — line number, starts from 1 for each page 🔢
* `text` — cleaned text of the line 📝
* `original_text` — original pre-repair text of the line 📝
* `split_ws` — hyphenated word prefix at the end of the line (split word start)
* `split_we` — hyphenated word suffix at the start of the line (split word end)
* `word_count` — **count** of whitespace-delimited tokens in the line (**count** of **words**)
* `char_count` — **count** of total character in the cleaned line
* `garbage_density` — ratio of non-alphanumeric characters to total line length (calculated on `original_text`)
* `upper` — **count** of **words** with unexpected mid-word uppercase letters
* `repeated` — **count** of **words** where a non-standard character makes up ≥ 30% of the word, or containing consecutive doubled garble characters
* `ldl_fuses` — **count** of **words** with a letter–digit–letter sandwich (e.g., `vyt1ačená`), excluding valid measurements.
* `fused_words` — **count** of tokens that appear to be fused **words** (abnormal consonant/vowel runs or extreme length)
* `gibberish` — **count** of **words** flagged as gibberish (high vowel ratio)
* `weird_wx` — **count** of words with an abnormal density of 'w' or 'x' glyphs
* `word_weird` — mean per-word weirdness score in [0, 1]; combines strange-symbol (0.40), repeated-char (0.35), LDL-fusion (0.15), mid-uppercase (0.10), and a `WORD_W_PENALTY`-weighted (default 0.20) signal for tokens containing the letter `w` — rare in Czech and a strong inverted/mirror-OCR fingerprint — plus a separate caps-prefix penalty (0.20). The combined score is clamped to [0, 1]. Isolated single letters score 0.85 (OCR noise) or 0.25 (digit/measurement).
* `vowel_ratio` — ratio of vowel characters to total alphabetic and symbol characters in the `original_text`
* `rot_ratio` — the ratio of structurally ambiguous/rotatable characters (`pbqdnuwmoxszeyv`) to the total number of alphabetic characters in the line.

##### `<doc_name>.csv`'s key resulting output **columns** that depict the final classification and quality assessment:

* `quality_score` — composite **quality score** 📈 in [0, 1] based on **9** combined signals; higher = cleaner
* `categ` — assigned category: **Clear** ✅, **Noisy** ⚠️, **Trash** 🗑️, **Non-text** 🔣, or **Empty** 🫙

##### `<doc_name>.csv`'s **columns** useful for archive managers information apart from the **quality score** 📈 and category:

* `lang` — predicted ISO **language code** from the **FastText** 🌐 model (remapped if unknown)
* `lang_score` — **FastText** 🌐 confidence score for the predicted language (capped if remapped, #3)
* `original_lang` — predicted language **before** remapping logic
* `orig_lang_score` — original **FastText** confidence **before** remapping
* `perplex` — Qwen2.5-0.5B 🤖 **perplexity** 📉 score of the line 📉
* `caps_header` — **boolean** flag indicating whether all alphabetic words in the line are uppercase (typical of section headers)

**Diagnostic flags (#3):**
Ten boolean audit columns follow `caps_header`. Six name the categoriser rule that decided the line —
`allcaps_novowel`, `lowppl_clear`, `cleanprose_clear`, `trash_threshold`, `noisy_threshold`, `clear_threshold`
(exactly one `True`, or none for `Empty`). Two further internal reason codes — `trash_hard_sweep` (route 1a)
and `trash_inverted` (route 1b) — also exist but are folded into the `trash_threshold` column rather than
getting their own column, so the per-line reason granularity is coarser in the CSV than inside the categoriser.
Four name the document-level post-pass that later changed it — `pp_dedup` (header/footer mode-harmonisation),
`pp_surrounded_trash` (rolling-window smoothing), `pp_inverted_run` (page-level inverted-scan sweep), and
`pp_page_context` (page-context Trash/Noisy adjustment, see below).
A categoriser flag and a `pp_` flag may both be `True` on one line: that is the intended trail from the
original decision to the override.

---

##### Categorisation logic reference 📚

The full decision logic — CPU pre-filter, language handling, structural detectors, the composite
quality score, the categorisation gates/rescues, and the four document-level post-processing
passes — lives in **[docs/categorization_logic.md](docs/categorization_logic.md)**.

| Section                                                                             | What it covers                                                                                 |
|-------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [CPU 💻 Pre-filter](docs/categorization_logic.md#cpu--pre-filter)                   | `pre_filter_line()`: `Empty`/`Non-text` routing and the two OCR repairs, before any model runs |
| [Language 🌐 Handling](docs/categorization_logic.md#language--handling)             | FastText trust tiers, `remap_lang()`, and the `LANG_REMAP_ALWAYS` switch                       |
| [Structural Detectors](docs/categorization_logic.md#structural-detectors)           | the per-line signal detectors (rotation, gibberish, fused/vowel-less words, damage)            |
| [Composite Quality Score](docs/categorization_logic.md#composite-quality-score)     | the weighted `compute_quality_score()` formula and its dynamic adjustments                     |
| [Categorisation Logic](docs/categorization_logic.md#categorisation-logic)           | `determine_category()`: the ordered gates, `check_rescues()`, and threshold routing            |
| [Post-Processing Smoothing](docs/categorization_logic.md#post-processing-smoothing) | `apply_document_postprocessing()`: dedup, rolling window, page context, inverted-scan sweep    |

To replay a config change over already-scored CSVs without a GPU, use
[`tools/recategorize_from_csv.py`](tools/recategorize_from_csv.py) — it reads the same constants and
calls the very same scoring function as the pipeline (`classify_TEXT.score_line()`), so a re-scored
corpus differs from the shipped batch only by the change under test. The FastAPI `/process` endpoint
uses that same function too; see
[Line Categorisation Logic](docs/categorization_logic.md), which opens with the three callers and
what each of them supplies.

---

Example of per-document **CSV** 📊 files: [DOC_LINE_CATEG](data_samples/DOC_LINE_CATEG) 📁 by **Qwen2.5-0.5B** 🤖
and [DOC_LINE_CATEG_gpt](data_samples/DOC_LINE_CATEG_gpt) 📁 by **distilgpt2** 🤖.

```
DOC_LINE_LANG_CLASS/
├── <docname1>.csv
├── <docname2>.csv
└── ...
```

---

#### 4.2 Aggregate Statistics (Memory Bound) 🧠

This script processes the `DOC_LINE_LANG_CLASS/` directory with **CSV** 📊 files in chunks 🧩 to produce
final page-level statistics. It is **CPU** 💻-bound and parallelized with `ProcessPoolExecutor`.

```
python3 aggregate_STAT.py
```

* **Input 📥:** `DOC_LINE_LANG_CLASS/` (directory with **CSV** 📊 files from the previous step)
* **Output 1 📤:** `final_page_stats.csv` 📊 (configurable via `OUTPUT_STATS`) — global page-level summary across all documents
* **Output 2 📤:** `DOC_LINE_STAT/` (configurable via `OUTPUT_DOC_DIR`) — per-document **CSVs** 📊 with the same schema

For each page, the aggregation computes features outputted in the following strict schema order:

**Totals & Counts:**

* `num_lines` — the total number of valid lines processed on the page
* `Clear`, `Noisy`, `Trash`, `Non-text`, `Empty` — integer count of lines in each category
* `total_word_count` — total number of words across scoreable lines
* `total_char_count` — total number of characters across scoreable lines

**Averages** (mean over the same `Clear` ✅ and `Noisy` ⚠️ lines):

* `avg_quality_score` — mean composite **quality score** 📈 in [0, 1]; higher = cleaner **OCR** 🔍 output
* `avg_word_weird` — mean per-word weirdness ratio in [0, 1]; 0 = fully clean, lower is better 📉
* `avg_lang_score` — mean **FastText** 🌐 confidence score
* `avg_perplex` — mean **Qwen2.5-0.5B** 🤖 **perplexity** 📉 score
* `avg_vowel_ratio` — mean vowel-to-alphabetic-character ratio per line
* `avg_rot_ratio` — mean rotatable character ratio per line
* `ch_ratio` — mean fraction of lines flagged as all-caps headers (`caps_header = True`)

**Language profile:**

* `main_lang` — the statistical mode (most frequent) language 🌐 predicted for the page

> [!NOTE]
> `avg_*` columns and `main_lang` will be `NaN` / `None` for pages whose only lines are
> `Empty` or `Non-text` (i.e., pages with no scoreable text content).
> Additional per-line diagnostic variables (e.g. `weird_wx`, `original_lang`, `original_text`) and flags added
> and ignored for this page-level aggregation to ensure stability.

All numeric averages are rounded to 4 decimal places; totals are stored as integers.

* *Examples*: [arup_page_stats_SHORT.csv](data_samples/arup_page_stats_SHORT.csv) 📊, [arub_page_stats_SHORT.csv](data_samples/arub_page_stats_SHORT.csv) 📊

Example of per-document aggregate **CSV** 📊 files: [DOC_LINE_STATS](data_samples/DOC_LINE_STATS) 📁 by **Qwen2.5-0.5B** 🤖
and [DOC_LINE_STATS_gpt](data_samples/DOC_LINE_STATS_gpt) 📁 by **distilgpt2** 🤖:

```
DOC_LINE_STAT/
├── stats_<docname1>.csv
├── stats_<docname2>.csv
└── ...
```

This is the end of the text quality classification and filtering step. You can now use [arup_page_stats_SHORT.csv](data_samples/arup_page_stats_SHORT.csv) 📎 to
identify files that need another round of **OCR** 🔍 or manual correction based on the line type counts. Pages with the
majority of **Clear** ✅ lines can be marked for further processing. The absence of clear lines combined with a high proportion
of **Trash** 🗑️ lines may also indicate handwritten content, which can be excluded before Handwritten Text Recognition (HTR) is applied.

## API Service Integration

In addition to the batch pipeline, this repository ships with a FastAPI wrapper (`service/text_api.py`) that exposes
the core `text_util_langID` quality classification engine over HTTP. The `/process` endpoint accepts ALTO XML,
plain-text, and generic JSON uploads (`task_type` `alto` / `text` / `json`, or `auto`-detected from the file
extension), returning the same per-line classification fields as the batch pipeline for all three formats.

The batch pipeline and the API service share the same `text_util_langID` categorization engine and `config.txt`
settings — including the default **Qwen2.5-0.5B** 🤖 perplexity model — to ensure zero drift between local processing
and web uploads.

For deployment instructions, endpoint specifications (`/process`, `/info`), and frontend integration details,
please see the dedicated **[Service Documentation](service/README.md)**.

## Paradata logging

This project incorporates a unified provenance and **paradata** 🗒️ logging system to seamlessly track the execution
details of every pipeline stage. The logger automatically captures run-time metadata and saves it in a
structured **JSON** 📄 format.

**What gets logged?**

* **Provenance 🏛️:** Captures the tool name, a tool **version** 🏷️ tag, the repository/runner reference, the running
container image (when set), the **Python** 🐍 version, and assigns a unique `run_id` to each execution. The repository
reference is resolved **dynamically** — environment overrides (`ATRIUM_RUNNER_REPO`, `ATRIUM_RUNNER_REF`,
`ATRIUM_RUNNER_IMAGE`) take precedence over the static fallback in [para_config.txt](setup/para_config.txt) 📎 — so the log
points at the image actually executing rather than a fixed fork.
* **Output license ⚖️:** Computes the **effective output license** 📜 of the run from the licensed components it actually
exercised, and records it as `license` / `license_url` plus a detailed `license_detail` block (per-component licenses,
which component(s) `determined_by` the result, `is_non_commercial` / `is_share_alike` flags, and any unknown licenses).
See [Output licensing](#output-licensing-) below.
* **Configuration ⚙️:** Stores run-time configuration ⚙️, including script
names, input/output paths, and specific model choices.
* **Timing ⏱️:** Records precise UTC start times, end times, and the total duration of the run in seconds.
* **Statistics 📊:** Tracks the total number of input files, successfully processed documents, and computes performance
throughput (e.g., output files generated per minute).
* **Error Tracking 🐛:** Maintains a `skipped_files_detail` list that logs the exact filename and specific error reason
if a file fails to process.

**Log Location**

By default, **JSON** 📄 logs are written to the [paradata](paradata) 📁 directory following the naming convention
`<YYMMDD-HHmmss>_<program>.json`. Paradata is intended to live alongside the **outputs** 📤 (not committed to the
repository); the **paradata** 🗒️ JSON files themselves are distributed under the **CC BY-NC 4.0** license.

---

### Output licensing ⚖️

> [!IMPORTANT]
> The license of the files a run **produces** is **not fixed** — it is computed per run as the **most restrictive**
> license among the components (models, data, APIs) that the run actually used. The mechanism is data-driven via
> [para_config.txt](setup/para_config.txt) 📎 (component → license) and [para_licenses.py](para_licenses.py) 📎
> (restrictiveness ranking + share-alike / non-commercial rules), so the licensing owner can adjust it without touching
> the logger.

Each repository ships a [para_config.txt](setup/para_config.txt) 📎 listing its components. Components flagged `always` count
toward every run (the worst-case baseline); components flagged `conditional` are only counted when the script that uses
them records it. For this repository the components and their effect on the **effective output license** 📜 are:

| Component                                                                              | License         | Counted     | Used by                                                        |
|----------------------------------------------------------------------------------------|-----------------|-------------|----------------------------------------------------------------|
| **alto-tools** 🔧 [^1](https://github.com/cneud/alto-tools)                            | Apache-2.0      | always      | page split, statistics, alto-tools text extraction             |
| **FastText** 🌐 [^2](https://huggingface.co/facebook/fasttext-language-identification) | CC BY-NC 4.0    | always      | language identification (`classify_TEXT.py`)                   |
| **Qwen2.5-0.5B** 🤖 [^6](https://huggingface.co/Qwen/Qwen2.5-0.5B)                     | Apache-2.0      | conditional | **perplexity** 📉 scoring (default, `classify_TEXT.py`)        |
| **distilgpt2** 🤖                                                                      | Apache-2.0      | conditional | **perplexity** 📉 scoring (English-only alternative)           |
| **LayoutLMv3** 📐 [^9](https://github.com/ppaanngggg/layoutreader)                     | CC BY-NC-SA 4.0 | conditional | LayoutReader text extraction (`extract_LytRdr_ALTO_2_TXT.py`)  |
| **GLM-4v-9b** 🤖 [^10](https://huggingface.co/THUDM/glm-4v-9b)                         | glm-4           | conditional | generative **OCR** 🔍 extraction (`extract_LLM_ALTO_2_TXT.py`) |

Because the always-on **FastText** 🌐 weights are **CC BY-NC 4.0**, the baseline effective output license for this
repository is **CC BY-NC 4.0** (non-commercial). Runs that additionally use the **LayoutReader** 📐 method escalate to
**CC BY-NC-SA 4.0** (non-commercial **and** share-alike), the most restrictive option here. A run that exercised only
permissive components would resolve to **Apache-2.0**.

> [!NOTE]
> The restrictiveness ordering encoded in [para_licenses.py](para_licenses.py) 📎 is a mechanical engineering
> approximation, **not legal advice**; unrecognised licenses are treated conservatively as maximally restrictive so a
> missing entry can never silently relax the recorded output license.

---

## Acknowledgements 🙏

**For support write to:** lutsai.k@gmail.com — responsible for this GitHub repository [^8](https://github.com/ufal/atrium-alto-postprocess) 🔗

* **Developed by** UFAL [^7](https://ufal.mff.cuni.cz/home-page) 👥
* **Funded by** ATRIUM [^4](https://atrium-research.eu/) 💰
* **Shared by** ATRIUM [^4](https://atrium-research.eu/) & UFAL [^7](https://ufal.mff.cuni.cz/home-page) 🔗
* **Models used**:
  * **FastText** 🌐 [^2](https://huggingface.co/facebook/fasttext-language-identification) for language identification
  * **Qwen2.5-0.5B** 🤖 [^6](https://huggingface.co/Qwen/Qwen2.5-0.5B) for **perplexity** 📉 scoring
  * **GLM-4v-9b** 🤖 [^10](https://huggingface.co/THUDM/glm-4v-9b) for generative **OCR** 🔍 (LLM-based method)
  * **LayoutLMv3** 📐 [^9](https://github.com/ppaanngggg/layoutreader) for layout-aware text extraction

**©️ 2026 UFAL & ATRIUM**

[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^4]: https://atrium-research.eu/
[^5]: https://github.com/ufal/atrium-nlp-enrich
[^6]: https://huggingface.co/Qwen/Qwen2.5-0.5B
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b
