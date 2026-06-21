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
  - [Step 1: Split Document-Specific ALTOs into Pages ✂️](#-step-1-split-document-specific-altos-into-pages-)
  - [Step 2: Create Page Statistics Table 📈](#-step-2-create-page-statistics-table-)
  - [Step 3: Extract text from ALTO XML ⛏️](#-step-3-extract-text-from-alto-xml-)
    - [LayoutReader method 📐](#1st-choice-layoutreader-method-)
    - [alto-tools method 🧰](#2nd-option-alto-tools-method-)
    - [GLM method 🤖](#3rd-alternative-glm-method-llm-based-)
  - [Step 4: Classify Page Text Quality & Language 🗂️](#-step-4-classify-page-text-quality--language-)
    - [4.1 Classify Lines (GPU Bound) 🚀](#41-classify-lines-gpu-bound-)
      - [CPU 💻 Pre-filter](#cpu--pre-filter)
      - [Language 🌐 Handling](#language-handling)
      - [Structural Detectors](#structural-detectors)
      - [Quality Score 📈 Computation](#composite-quality-score)
      - [Categorization Logic](#categorisation-logic)
      - [Post-Processing Smoothing](#post-processing-smoothing)
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
    pip install -r requirements.txt
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
python3 run_pipeline.py                      # all settings from config_langID.txt
python3 run_pipeline.py --method glm         # override just the extraction backend
python3 run_pipeline.py --skip-split         # PAGE_ALTO already populated
python3 run_pipeline.py --dry-run            # print the resolved plan, run nothing
```

* **Configuration ⚙️:** every setting is read from [config_langID.txt](config_langID.txt) 📎
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

### ▶️ Step 1: Split Document-Specific ALTOs into Pages ✂️

First, ensure you have a directory 📁 containing your document-level `<file>.alto.xml` files.
This script will split them into individual page-specific **XML** 📄 files.

```
python3 page_split.py <input_dir> <output_dir>
```

Each page-specific file retains the header from its original source document 📌.

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

All input/output paths and tunable parameters are configured ⚙️ in [config_langID.txt](config_langID.txt) 📎.
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
# NOTE: QS_WEIGHT_SYMBOL no longer exists — there is no symbol_ratio term in compute_quality_score().

CATEG_TRASH_SCORE_MAX       = 0.55      # Max QS for Trash category
CATEG_NOISY_SCORE_MAX       = 0.85      # Max QS for Noisy category
REPEATED_DOUBLE_MIN         = 2         # Minimum occurrence count for doubled-char penalty
SHORT_NOISY_QS_PENALTY      = 0.20      # Opt-in QS penalty for short strings exhibiting OCR oddities

# --- New since last revision: Phase-2 categoriser overrides ---
LOWPPL_CLEAR_MAX            = 50.0      # ppl ceiling for Override 3 (was hardcoded)
HARD_SWEEP_LANG_MAX         = 0.45      # orig_lang_score ceiling for the hard-sweep route
HARD_SWEEP_PPL_MIN          = 1000.0    # ppl floor for the hard-sweep route
GHOST_DOMINATED_MIN_RATIO   = 0.5       # min ghost-token share to flag ghost_dominated
WORD_W_PENALTY              = 0.20      # per-word weirdness penalty for tokens containing 'w'
ROT_HIGH_LANG_CONF          = 0.90      # lang_score ceiling for the page-level rotation arm
CLEAR_BAND_WC_MIN           = 0         # short-fragment Clear-band guard (0 = disabled)

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
python3 langID_classify.py

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

##### CPU 💻 Pre-filter

Before any **GPU** 🚀 or model inference, `pre_filter_line()` applies a fast **CPU** 💻-side check and assigns `Empty` or `Non-text`
directly, bypassing the ML pipeline entirely. It also applies two lightweight **OCR** 🔍 text repairs to every line before
the rules are evaluated.

Firstly, two fixes correct the most common systematic **OCR** 🔍 substitution errors before any rule is checked. They modify
the text that is passed forward but do not on their own affect what category a line receives.

* **Digit-for-letter substitution:** A `1` surrounded by alphabetic characters on both sides is replaced with `l`
(e.g., `poh1ed` → `pohled`); a `2` at the start of a token followed immediately by a lowercase letter is replaced
with `z`. These substitutions reflect common **OCR** 🔍 confusions between visually similar characters.
* **Spaced-letter collapse:** A sequence of individually spaced single uppercase letters (`P R A H A`) is recognised
as a prostrkávání/spaced-text typographic style and collapsed back into a normally-cased word (`Praha`). Without this
repair, spaced words fail the letter-ratio check and would be discarded as `Non-text`.

1. Line is blank or contains only whitespace → `Empty`
2. Line consists entirely of digits, arithmetic/date separators, and punctuation with no letters → `Non-text` (e.g. `1998`, `5.3.`, `- 14 -`)
3. Line is a Roman numeral, optionally followed by a period → `Non-text` (e.g. `XIV.`, `iii`)
4. Line is a standalone alphanumeric archive or inventory code — a short letter prefix of up to 3 characters
followed by 3 or more digits, with an optional slash-separated suffix → `Non-text` (e.g. `A1739`, `CTX200205348`, `A679/2015`)
5. Line matches a stamp-like ratio pattern — a short alphanumeric string, optional non-alphanumeric characters,
two 2-to-4 digit numbers separated by a `/`, and optional trailing non-alphanumeric characters → `Non-text`
(e.g., `123/456`, `1998/01`, `NZ1998/01`)
6. Fewer than 4 total characters, or fewer than 3 unique non-whitespace symbols → `Non-text`
(lines this short cannot carry meaningful archaeological text)
7. Alphabetic characters make up less than 30% of total characters → `Non-text`
(the line is dominated by digits, punctuation, or special characters)
8. **Isolated Chars & Fusions**: A line dominated by isolated alphanumeric tokens, or a single token fusing letters, digits, and symbols → `Non-text`.
9. **Otherwise** → forwarded for ML classification as `Process`

Finally, two categories of exception send a line directly to `Process` even if it would otherwise be caught by a `Non-text` rule:

* **Metadata marker bypass** — If the line contains any of the following patterns (checked case-insensitively),
it is forwarded as **Process** regardless of how short it is or how few letters it contains. These strings are
structural metadata markers specific to Czech 🇨🇿 archaeological report forms. Without this bypass they would be
discarded as `Non-text` because they are typically very short, heavily abbreviated, or contain mostly punctuation —
but their presence is meaningful for downstream NLP and archival cataloguing:

| Marker                                                    | Typical context in Czech 🇨🇿 archaeological records |
|-----------------------------------------------------------|------------------------------------------------------|
| `Tb.`                                                     | Table reference abbreviation (Czech 🇨🇿: *tabulka*) |
| `č.neg`, `č. neg`, `č neg`, `č.neg.`, `č. neg.`, `č neg.` | Negative number reference (*číslo negativu*)         |
| `neg.`, `neg `                                            | Negative reference shorthand                         |
| `obr.`, `obr `                                            | Figure reference (*obrázek*)                         |
| `č.`                                                      | General Czech 🇨🇿 number abbreviation (*číslo*)     |
| `str.`                                                    | Page abbreviation (*strana*)                         |
| `Datum`                                                   | Date field label on standard report forms            |

* **High digit-ratio bypass** — If digits make up more than 40% of the line's total characters, the line is
forwarded as **Process** regardless of its letter ratio. This preserves content-bearing strings that are intentionally
numeric-heavy: measurement records (e.g., `váha 90,9g`, `30–50 cm`), date strings (e.g., `5.XI.1946`), grid
coordinates, and catalogue references that combine letters and numbers. Without this bypass, most measurement lines
would be discarded by rule 7 above.

---

##### Language Handling

**FastText** 🌐 [^2](https://huggingface.co/facebook/fasttext-language-identification) is run on the **lowercased** line text and returns a predicted ISO 639-3 language code
(e.g. `ces` for Czech 🇨🇿, `deu` for German 🇩🇪) and a confidence score between 0 and 1. The pipeline then applies
a series of remapping rules before the `lang` and `lang_score` fields are finalised for storage and before
the score is used in quality computation.

**Configuration keys (in `[CLASSIFY]`):**

* `EXPECTED_LANGS` — comma-separated list of language 🌐 codes the collection is expected to contain (e.g., `ces,deu,eng`).
The **first** entry is the **default fallback language** used when **FastText** 🌐 predicts a language that is not in
either `EXPECTED_LANGS` or `TRUSTED_FOREIGN_LANGS`. If your collection is primarily Czech 🇨🇿, `ces` should be
first. If your collection is primarily German 🇩🇪 archival material, put `deu` first and adjust the **perplexity** 📉
thresholds accordingly.
* `TRUSTED_FOREIGN_LANGS` — comma-separated list of foreign languages 🌐 whose presence in the collection is considered
genuine and should be kept as-is. A language belongs in this list if you expect real documents or passages
in that language (e.g., German-language summaries in a Czech 🇨🇿 report, Latin citations, English 🇬🇧 abstracts).
Languages on this list are **not remapped** to the default, regardless of confidence.

**Language score thresholds (in `[TEXT_UTILS]`):**

* `LANG_SCORE_REMAP = 0.75` — the minimum confidence floor assigned to unknown lines force-remapped to the collection default.
* `LANG_SCORE_ROUGH = 0.45` — a **FastText** 🌐 confidence below this is considered too unreliable to trust. This threshold
is used by the page-level inverted-scan sweep (see Post-Processing Smoothing) to identify pages where **FastText** 🌐 cannot
confidently assign any language to any line — a strong signal that the page content is not readable text.

**Remapping logic (applied per line, in order):**

1. If the predicted language 🌐 code appears in `EXPECTED_LANGS` or `TRUSTED_FOREIGN_LANGS` → the **FastText** 🌐 prediction
and confidence score are **kept unchanged**. No remapping occurs.
2. If the predicted language is `slk` (Slovak), it is considered a near-twin of Czech and is remapped to the collection default, but its **original score is preserved**.
3. If the predicted language 🌐 is **not** in either set and not Slovak (e.g., **FastText** 🌐 guesses Danish `dan` on a
Czech 🇨🇿 line) → the language 🌐 code is **force-remapped** to the **first entry of `EXPECTED_LANGS**` (the collection
default). The stored `lang_score` is set to `min(original_score, cap)` — **(#3)** *capped*, not floored: a Latin-script
guess is capped at `LANG_SCORE_REMAP` (**0.75**) and a non-Latin-script guess (Hangul, Cyrillic, CJK, …) at
`LANG_SCORE_REMAP_FAR` (**0.50**). A *confident* foreign guess on Czech archival data is evidence of inverted or garbled
**OCR** 🔍, not of trustworthy language ID, so the old `max`-floor (which inflated weak foreign guesses up to 0.75) was
backwards. Capping leaves the stored score honestly low, which is exactly what the page-level inverted-scan
`diacritic-absence` arm keys off.

**What gets stored vs. what drives the **quality score** 📈:**

These two values are intentionally different:

* The **stored `lang_score` column** in the output **CSV** 📊 reflects the post-remapping value (i.e., capped to
`LANG_SCORE_REMAP` if remapping occurred). This is what you see in the file.
* The **quality score 📈 computation** (`compute_quality_score()`) receives the **original pre-remapping **FastText** 🌐
confidence** (`original_lang_score`). This means the `QS_WEIGHT_LANG` component of the **quality score** 📈 honestly
reflects how confident **FastText** 🌐 was about the line's language — not the artificially raised value. A line where
**FastText** 🌐 was genuinely uncertain gets a lower language-confidence contribution to its **quality score** 📈 regardless of
what language 🌐 code is stored.

---

##### Structural Detectors

Lines that pass the pre-filter are analysed by structural detectors defined in [text_util_langID.py](text_util_langID.py)📎:

| Detector                     | What it counts                                                                                                                                                                             |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `detect_strange_symbols`     | **Occurrences** of any character that is not alphanumeric and not in the **allowed** set `{ . - , + ( ) " ' — – : % ; ? ! / }`. Edge punctuation is stripped before inspection.            |
| `detect_letter_digit_letter` | Words with a **letter–digit–letter sandwich** — the fingerprint of **OCR** 🔍 digit insertions mid-word (e.g., `vyt1ačená`). **Legitimate** measurements (`30cm`, `90,9g`) do not trigger. |
| `detect_mid_uppercase`       | Words with unexpected uppercase mid-word (`dalSÍ`). Academic titles (`PhDr`, `MUDr`) are **excluded**.                                                                                     |
| `detect_repeated_chars`      | Words with triple character runs, or double runs occurring ≥3 times. Exempts vowels `o, u` and digits to protect legitimate Czech 🇨🇿 doubles (e.g., *denní*).                            |
| `detect_gibberish_words`     | Words of length ≥ 4 with a vowel ratio above `VOWEL_RATIO_HIGH` (0.70). All-caps and mostly numeric words are **excluded**. Sub-tokens are split on internal punctuation first.            |
| `compute_rotatable_ratio`    | Measures the concentration of structurally ambiguous/rotatable letters (`pbqdnuwmoxszeyv`) to catch severe visual noise interpreting graphical textures as characters.                     |
| `detect_fused_words`         | Counts tokens that are likely multiple words merged without a space (token length > 14, consonant run of 5+, or vowel run of 3+). Sub-tokens are split on internal punctuation first.      |
| `detect_wx_words`            | Tokens with an abnormal density of 'w'/'x' glyphs (≥ 2 per sub-token). By default, this is folded into the gibberish ratio to punish severe mirror scans.                                  |

##### Composite Quality Score

After structural detection, each line receives a single floating-point `quality_score` 📈 in [0, 1] computed by
`compute_quality_score()` in [text_util_langID.py](text_util_langID.py)📎. The score is a weighted sum of **nine**
normalised signals, **dynamically divided by the total sum of weights** to strictly bound the maximum
score to 1.0 (preventing score inflation):

```text
base_score =
    QS_WEIGHT_VALID_WORD  (def: 0.35) × valid_word_ratio
  + QS_WEIGHT_WEIRD       (def: 0.18) × (1 − min(word_weird_ratio, 1.0))
  + QS_WEIGHT_PERPLEXITY  (def: 0.08) × (1 − min(perplexity / PERPLEXITY_THRESHOLD_MAX, 1.0))
  + QS_WEIGHT_LENGTH      (def: 0.02) × min(char_count / QS_LENGTH_MAX, 1.0)
  + active_garbage_wt     (def: 0.18) × (1 − min(garbage_density / CATEG_GARBAGE_DENSITY_HIGH, 1.0))
  + QS_WEIGHT_VOWEL       (def: 0.07) × vowel_quality_score
  + QS_WEIGHT_LANG        (def: 0.05) × lang_score
  + QS_WEIGHT_GIBBERISH   (def: 0.04) × (1 − min(gibberish_ratio, 1.0))
  + QS_WEIGHT_FUSED       (def: 0.03) × (1 − min(fused_ratio, 1.0))

quality_score = (base_score / total_weight) − short_penalty

```

> [!NOTE]
> There is no `symbol_ratio` term and no `rot_penalty` subtraction in the current implementation — both
> appeared in earlier revisions. Symbol density is no longer fed into the quality score (the `symbol`
> per-line column has also been removed, see below); rotation/inversion is now handled entirely by the
> per-line lexicon override and the page-level sweep described elsewhere in this document.

| Signal             | Source                                                                   | Normalisation                                                                           | Notes                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|--------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `valid_word_ratio` | fraction of structurally valid word tokens                               | used directly [0, 1]                                                                    | A token is valid if ≥ 70% alphabetic, no disallowed internal symbols, and not a garbled all-caps **OCR** 🔍 prefix followed by lowercase (e.g., `AAMMNAbSSOAO`, `XAterenta`) — this guard prevents spurious uppercase runs from inflating the signal.                                                                                                                                                                                        |
| `word_weird_ratio` | mean per-word weirdness across tokens                                    | inverted: `1 − ratio`                                                                   | The per-word score combines strange-symbol (0.40), repeated-char (0.35), LDL-fusion (0.15), mid-uppercase (0.10), and a `WORD_W_PENALTY`-weighted (default 0.20) signal for tokens containing the letter `w` — rare in Czech and a strong inverted/mirror-OCR fingerprint — plus a separate caps-prefix penalty (0.20). The combined score is clamped to [0, 1]. Isolated single letters score 0.85 (OCR noise) or 0.25 (digit/measurement). |
| `perplexity` 📉    | `Qwen2.5-0.5B` 🤖 NLL per token                                          | inverted & capped: `1 − min(ppl / PERPLEXITY_THRESHOLD_MAX, 1.0)`                       | A value of 0 assigned when ppl ≥ threshold (worst), 1 when ppl = 0 (best). Calibrated for `Qwen2.5-0.5B` 🤖 at default `1000.0`; use `3000.0` for `distilgpt2` 🤖.                                                                                                                                                                                                                                                                           |
| `char_count`       | total character count                                                    | `min(count / QS_LENGTH_MAX, 1.0)`                                                       | Full reward for lines ≥ `QS_LENGTH_MAX` (default 100) characters.                                                                                                                                                                                                                                                                                                                                                                            |
| `garbage_density`  | fraction of unusual non-alnum characters in `original_text`              | inverted & capped: `1 − min(density / CATEG_GARBAGE_DENSITY_HIGH, 1.0)`                 | Ellipsis (`...`) runs are **not** stripped. Density evaluates strictly on the original uncleaned line length.                                                                                                                                                                                                                                                                                                                                |
| `vowel_ratio`      | vowel fraction among alphabetic and symbol characters in `original_text` | linear ramp: full reward in `[VOWEL_RATIO_LOW, VOWEL_RATIO_HIGH]`, ramps to 0.0 outside | Evaluated on the `original_text` to prevent hiding symbol noise.                                                                                                                                                                                                                                                                                                                                                                             |
| `lang_score`       | **FastText** 🌐 language confidence (original, pre-remapping)            | used directly; default 0.5 when unavailable                                             |                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `gibberish_ratio`  | fraction of extreme high-vowel words and w/x noise (word length ≥ 4)     | inverted: `1 − ratio`                                                                   | Words ≥ 60% digits/separators excluded. Folded together with `detect_wx_words` density.                                                                                                                                                                                                                                                                                                                                                      |
| `fused_ratio`      | fraction of suspected merged tokens                                      | inverted: `1 − ratio`                                                                   |                                                                                                                                                                                                                                                                                                                                                                                                                                              |

**Dynamic adjustments inside `compute_quality_score()` formula:**

These conditional modifications are applied during scoring. They adjust intermediate numerical values inside the
score formula before the final weighted sum is computed.

**1. Garbage Penalty Guard (short clean strings)**

*Trigger:* `char_count ≤ 12`, `word_weird == 0.0`, **and** low garbage density

*What happens:* `active_garbage_wt` is **halved** from `QS_WEIGHT_GARBAGE` (default 0.18) to 0.09. A compensating
constant of the same amount is added back to `base_score` so the total effective weight sum is unchanged and the
maximum possible score remains 1.0.

*Why:* Short archival label strings — `Lokalita:`, `Osada:`, `Okres:`, `Datum:` — contain a colon or other
structural punctuation that is counted as "garbage". Since the line is short, completely structurally clean (no weirdness),
and mostly legible, the reduced weight prevents the label from being unfairly penalised.

**2. Short Noisy Strings Penalty (`SHORT_NOISY_QS_PENALTY`)**

*Trigger:* `char_count ≤ 12` **and** (`word_weird > 0.0` or high garbage density).

*Applied:* Applies the configurable `SHORT_NOISY_QS_PENALTY` (default 0.20) as a direct subtraction to the final score.

*Why:* An opt-in penalty to sink very short noisy strings that might otherwise artificially float into acceptable score ranges due to their minimal features.

**3. Short **Perplexity** 📉 Cap (`SHORT_PPL_CAP`)**

*Trigger:* `word_count ≤ 2` **and** raw LM **perplexity** 📉 `> SHORT_PPL_CAP` (default **850.0** for Qwen2.5-0.5B 🤖,
**2500.0** for distilgpt2 🤖)

*Applied:* in `langID_classify.py`, **before** `compute_quality_score()` is called. The **perplexity** 📉 value
*passed to scoring* is clamped to `SHORT_PPL_CAP`. **The stored `perplex` column in the output **CSV** 📊 is not
changed**.

*Why:* Language models assign **perplexity** 📉 by predicting tokens. With only 1–2
words, there is almost no context available, so the model assigns extremely high **perplexity** 📉 even to perfectly valid words.
Without this cap, every single-word or two-word line would receive a near-zero **perplexity** 📉 component in its **quality score** 📈.

---

##### Categorisation Logic

`categorize_line()` classifies each line using **immediate overrides** checked in priority order, followed by
**quality score 📈 threshold routing**, with one additional **promotion override** and a **mostly-readable cap** inside the `Noisy` band.
The function also aligns the stored `quality_score` value to be consistent with the assigned category band, so that
downstream analytics can rely on the score as a monotone proxy for category rank without re-running thresholds.

Checked in order - the first match wins and skips all remaining checks including thresholds:

| #  | Condition                                                                                                                                                                                          | Result  | Rationale                                                                                                                                                                                                                                                                                                                                                              |
|----|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1  | `word_count == 0` or line contains only whitespace                                                                                                                                                 | `Empty` | Structural blank — no content to evaluate. Assigned before any scoring.                                                                                                                                                                                                                                                                                                |
| 1a | `orig_lang_score < HARD_SWEEP_LANG_MAX` (def: 0.45) **and** `ppl > HARD_SWEEP_PPL_MIN` (def: 1000.0)                                                                                               | `Trash` | **Hard sweep.** Catches confident garbage that the language-remap CAP would otherwise mask — FastText could not place the line at all *and* the LM also found it surprising. Recorded as `trash_hard_sweep`, folded into the `trash_threshold` CSV flag.                                                                                                               |
| 1b | `ghost_dominated` **and not** `is_upright_czech`                                                                                                                                                   | `Trash` | **Inverted/mirrored scan (per-line).** `ghost_dominated`: a majority of word tokens are flip-images of common Czech function words (`analyze_rotation_signals`, see below). `is_upright_czech` (a Czech diacritic, or a real upright function word) is a hard protective signal that bypasses this route. Recorded as `trash_inverted`, folded into `trash_threshold`. |
| 2  | All alphabetic words are uppercase **and** `vowel_ratio < 0.10`                                                                                                                                    | `Trash` | Definitively unreadable: an all-caps block with almost no vowels is a visual scramble. (Recorded as `allcaps_novowel`)                                                                                                                                                                                                                                                 |
| 2a | `garbage_density >= CATEG_GARBAGE_DENSITY_HIGH` (def: 0.35)                                                                                                                                        | `Trash` | **Garbage-density hard override.** A line whose raw non-alphanumeric density alone exceeds the QS normalisation ceiling is routed to Trash directly, bypassing the weighted score. Folded into `trash_threshold`.                                                                                                                                                      |
| 2b | `word_count ≤ ISOLATED_CHAR_MIN_TOKENS` (def: 3) **and** no Czech 🇨🇿 diacritics **and** post-cap `lang_score ≤ LANG_SCORE_REMAP` (def: 0.75) **and** (gibberish present **or** `word_weird > 0`) | `Trash` | **(#3)** Structural short-garbage route (e.g. `olie`). Folded into the `trash_threshold` reason.                                                                                                                                                                                                                                                                       |
| 3  | `perplexity < LOWPPL_CLEAR_MAX` (def: 50.0) 📉 **and** `word_count ≥ 3`                                                                                                                            | `Clear` | The language model is near-certain about the text. Recorded as `lowppl_clear`.*                                                                                                                                                                                                                                                                                        |

*(Mostly-Readable Cap Applies: If override 3 fires but `valid_word_ratio < MOSTLY_READABLE_VALID_MIN`, the line is capped at `Noisy` instead, recorded as `noisy_threshold`.)*

> [!NOTE]
> Inverted/180°-rotated scans are now detected at **two** levels:
> * A **per-line** lexicon check (override 1b above) catches lines dominated by flip-images of common Czech function words.
> * A **page-level** pass (see [Post-Processing Smoothing](#post-processing-smoothing) below) independently catches contiguous runs — or a page-wide majority — of suspicious lines, using rotatable-character density, perplexity, and language-confidence signals.
>
>
> `compute_quality_score()` no longer subtracts a `rot_penalty` from the quality score. The `is_upright_czech` flag it still accepts as a parameter has no effect on the computed score in the current implementation.

**Quality score 📈 threshold routing** (applied to all lines not caught by an override above):

```text
quality_score < CATEG_TRASH_SCORE_MAX  (def: 0.55)  →  Trash  (Recorded as trash_threshold)
quality_score < CATEG_NOISY_SCORE_MAX  (def: 0.85)  →  Noisy  (unless Override 4 fires, Recorded as noisy_threshold)
otherwise                                           →  Clear  (unless capped by mostly-readable min)

```

**Mostly Readable Cap:** If a line qualifies for `Clear` (either by pure quality score or Override 3), but its `valid_word_ratio < MOSTLY_READABLE_VALID_MIN` (default 0.85), it is capped at `Noisy`.

> [!NOTE]
> **(#3) Short Czech function words count as valid.** `compute_valid_ratio` previously rejected every token shorter than
> three characters, so genuine Czech 🇨🇿 prose dense in one/two-letter prepositions, conjunctions and clitics
> (`v první řadě po stříbrných penězích`) scored an artificially low `valid_word_ratio` and was denied the clean-prose
> promotion. The validator now also accepts members of the configurable `SHORT_VALID_WORDS` list and the
> `SINGLE_CHAR_ALLOWED` set before the 3-character structural gate. Short *non-words* are still rejected, so the
> Mostly-Readable Cap continues to protect against garbage.

**Override 4 — Near-Boundary Clean Prose Promotion** (fires inside the `Noisy` band, before `Noisy` is returned):

If all four of the following conditions hold simultaneously, the line is promoted from `Noisy` → `Clear`
(Recorded as `cleanprose_clear`):

| Condition                                      | Parameter                   | Default | What it ensures                                                                                                                                                                                                   |
|------------------------------------------------|-----------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `quality_score ≥ CLEAN_PROSE_MIN_SCORE`        | `CLEAN_PROSE_MIN_SCORE`     | 0.65    | The line is in the upper part of the `Noisy` band — it scored reasonably well but was held back by one or two minor signals.                                                                                      |
| `word_count ≥ CLEAN_PROSE_WC_MIN`              | `CLEAN_PROSE_WC_MIN`        | 4       | The line has enough tokens for **perplexity** 📉 to be meaningful. With 1–3 words, the LM has almost no context and its **perplexity** 📉 score is unreliable; the promotion is only trusted for longer text.     |
| `word_weird < CLEAN_PROSE_WEIRD_MAX`           | `CLEAN_PROSE_WEIRD_MAX`     | 0.08    | No token in the line shows significant structural corruption. Even a single noticeably strange token (e.g., a letter–digit–letter fusion or a mid-word uppercase) disqualifies the line from promotion.           |
| `perplexity < CLEAN_PROSE_PPL_MAX`             | `CLEAN_PROSE_PPL_MAX`       | 400.0   | The language model considers the line reasonably likely. A very high **perplexity** 📉 even on a structurally clean line can indicate foreign or domain-specific vocabulary that is genuinely `Noisy` to process. |
| `valid_word_ratio ≥ MOSTLY_READABLE_VALID_MIN` | `MOSTLY_READABLE_VALID_MIN` | 0.85    | Semi-readable lines dipping below this ratio are excluded from promotion.                                                                                                                                         |

**Short-fragment Clear-band guard (`CLEAR_BAND_WC_MIN`, default 0 = disabled):** when set above 0, a line
shorter than `CLEAR_BAND_WC_MIN` words that reaches the `Clear` quality-score band but still carries some
noise (`word_weird > 0` or `garbage_density > 0`) is held at `Noisy` instead — recorded as `noisy_threshold`.
This guard is skipped entirely by the Override-3 LM fast-track (`lowppl_clear`). It exists to stop very short
noisy OCR fragments from floating into `Clear` on minimal-feature lines, without affecting clean short prose.

**Rationale for Override 4:** Readable Czech 🇨🇿 archaeological prose — field measurements, dig site descriptions,
formal letter phrases — can score just below `CATEG_NOISY_SCORE_MAX` (0.85) for two systematic reasons: (a) the
**perplexity** 📉 of short isolated sentence fragments is inherently elevated even when the words are perfectly correct, and
(b) minor but common **OCR** 🔍 artefacts such as period-abbreviations (`Obr.`, `Viz`) and occasionally merged words slightly
depress the valid-word-ratio component. When the line is long enough, structurally clean, and the **LM** 🤖 is reasonably
confident, these small penalties should not prevent the line from reaching `Clear`.

After the category is determined, the stored `quality_score` 📈 is clamped to the range corresponding to the assigned
band. This ensures that the **CSV** 📊 value is always internally consistent with the `categ` label:

* `Trash` → score clamped to `min(qs, CATEG_TRASH_SCORE_MAX − ε)` — always below 0.55
* `Noisy` → score clamped to `[CATEG_TRASH_SCORE_MAX, CATEG_NOISY_SCORE_MAX − ε]` — always in [0.55, 0.85)
* `Clear` → score clamped to `max(qs, CATEG_NOISY_SCORE_MAX)` — always ≥ 0.85

Lines promoted by Override 4 have their raw score (which was somewhere in [0.65, 0.85)) raised to 0.85 in the **CSV** 📊.
Lines whose Override 2 or Override 3 fired receive a score consistent with the override result regardless of what
the formula computed.

---

##### Post-Processing Smoothing

After all lines in a document are classified and written to **CSV** 📊, a final data-smoothing pass is applied before the file
is finalized. This pass corrects categorisation anomalies that only become visible at the document or page
level — patterns that per-line scoring cannot detect because it evaluates each line in isolation.

*What it does:* Before the dedup and rolling-window passes, two page-context rules adjust borderline
categories using page-level aggregates — `median_qs`, `clear_ratio`, and the fraction of lines in a trusted
language (`decent_lang_ratio`, over `ces, deu, eng, fra, pol, ita, slk`):

* **Heavily-garbage pages:** if a page's `Clear` ratio is ≤ 0.05, its trusted-language ratio is < 0.50, and
its median quality score is < 0.55, every `Noisy` line on that page scoring below 0.80 is downgraded to
`Trash`.
* **Predominantly-clean pages:** if a page's `Clear` ratio is > 0.60 and its median quality score is > 0.80,
every `Trash` line on that page scoring ≥ 0.45 *and* in a trusted language is promoted to `Noisy`.

**Recorded as** `pp_page_context`

*Why:* A page that is almost entirely garbage rarely contains a genuinely-recoverable `Noisy` line; a page
that is almost entirely clean rarely contains a genuinely-unrecoverable `Trash` line. These rules use the
page as additional context the per-line categoriser cannot see.

*What it does:* All occurrences of the exact same text string across a document are identified. If the same string
has been assigned to different categories on different pages (e.g., `Obr. 1. SKUHROV NAD BĚLOU` is `Clear` on page 3
but `Noisy` on page 4 due to slightly different surrounding context affecting the LM), all occurrences are
harmonised to the **statistical mode** — the category assigned most frequently to that string across the document.

**Recorded as** `pp_dedup`

*Why:* Repeating strings are boilerplate — page headers, footers, running titles, standard form labels. The
same physical text should receive the same label throughout a document, and the majority vote across its
occurrences is the most reliable estimate of the correct category.

*What it does:* Scans the document line-by-line. If a `Noisy` ⚠️ line is surrounded by `Trash` 🗑 on both sides in a
5-line window (positions −2 and −1 are `Trash` 🗑 **and** positions +1 and +2 are `Trash` 🗑), **and** the `Noisy` ⚠️ line's
quality score is below `CATEG_TRASH_SCORE_MAX + 0.15` (default: **0.70**), it is downgraded to `Trash` 🗑.

**Recorded as** `pp_surrounded_trash`

*Why:* A single `Noisy` ⚠️ island embedded in four consecutive `Trash` 🗑️ lines is almost certainly corrupted text that
narrowly escaped the `Trash` 🗑 threshold. The rolling window catches these borderline cases. The score guard of 0.70
ensures that only near-boundary `Noisy` ⚠️ lines are affected — a `Noisy` ⚠️ line with a **quality score** 📈 of 0.80
is left alone even in a `Trash` 🗑 neighbourhood, because its quality is genuinely different from the surrounding lines.

*What it does:* After the rolling-window pass, each page is scanned independently. A line is **suspicious** when it
meets **either** of the two detection arms below. Suspicious lines are downgraded to `Trash`🗑 when they form a
contiguous run of 4 or more (`INVERTED_RUN_MIN`) non-`Empty`🫙/non-`Non-text`🔣 lines, **or (#3)** when they make up at
least `INVERTED_PAGE_MAJORITY` (default **0.60**) of the page's scoreable lines — the **page-majority arm**:

* **Diacritic-absence arm:** the line lacks Czech 🇨🇿 diacritics (`á č ď é ě í ň ó ř š ť ů ú ý ž` and
uppercase equivalents) **and** has a **FastText** 🌐 confidence below `LANG_SCORE_ROUGH` (default **0.45**).
Together these indicate that the **OCR** 🔍 produced no Czech-identifiable content and the language model cannot
confidently assign any language 🌐 — a strong signal of a page-level scan fault.
* **Rotation arm:** the line has `perplexity ≥ PPL_INVERTED_MIN` 📉 (default **200.0**), `word_weird > 0.0`,
**and** a stored `lang_score < ROT_HIGH_LANG_CONF` (default **0.90**).

> [!WARNING]
> Although `rot_ratio` is computed for every candidate line in this pass, it is **not** part of the boolean
> condition above in the current implementation — only perplexity, word-weirdness, and language confidence
> gate this arm. If a `rot_ratio` requirement is intended, this is a code/doc discrepancy worth confirming
> with the maintainers rather than something this document should silently "fix" by asserting a behaviour
> the code doesn't have.

**Recorded as** `pp_inverted_run`

*Why a page-majority arm? (#3)* Some inverted/garbage scans break up into many short, isolated fragments separated by
`Empty`🫙 lines, `Non-text`🔣 stamps, or single-token noise, so the suspicious lines never form a single run of four and
escape the run-based rule. When the **majority** of a page's scoreable lines are individually suspicious, the page as a
whole is treated as an inverted/garbage scan and every suspicious line on it is downgraded, regardless of run length.
Lines carrying Czech 🇨🇿 diacritics or a confident **FastText** 🌐 score are never suspicious, so genuine content
interleaved on the page is preserved.

*Why two arms?* Inverted-scan pages sometimes produce partial Czech 🇨🇿 diacritics: the **OCR** 🔍 engine recognises some
upside-down glyphs as plausible Latin characters and occasionally matches diacritical forms. The diacritic-absence
arm alone would miss these pages. The rotation arm catches them independently by using the character-shape signal
and LM uncertainty together, without requiring the absence of diacritics.

---

The table below consolidates every factor that influences `quality_score` 📈 or the final category assignment, including
where each factor is controlled and any known edge cases.

| Factor                                        | Where applied                                                    | Config key(s)                                                                                                  | Edge cases / exceptions                                                                                                                                                                                                                                                                                                                                            |
|-----------------------------------------------|------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Valid word ratio                              | `compute_quality_score` (35% weight)                             | `QS_WEIGHT_VALID_WORD`                                                                                         | All-caps **OCR** 🔍 prefix guard: tokens like `AAMMNAbSSOAO` are excluded from valid-word count even though they are alphabetically dominant.                                                                                                                                                                                                                      |
| Word weirdness ratio                          | `compute_quality_score` (18% weight)                             | `QS_WEIGHT_WEIRD`                                                                                              | Isolated single letters score 0.85 (**OCR** 🔍 spaced-out noise); isolated digits/measurements score 0.25 (tolerable). All-caps words and specific capitalized sequences excluded from mid-uppercase detection.                                                                                                                                                    |
| Perplexity 📉 (LM)                            | `compute_quality_score` (8% weight)                              | `QS_WEIGHT_PERPLEXITY`, `PERPLEXITY_THRESHOLD_MAX`                                                             | Short-text **perplexity** 📉 is capped at `SHORT_PPL_CAP` before scoring to prevent penalising legitimate 1–2 word inputs. Override 3 (`ppl < 50`) bypasses thresholds entirely for highly confident predictions.                                                                                                                                                  |
| Text length                                   | `compute_quality_score` (2% weight)                              | `QS_WEIGHT_LENGTH`, `QS_LENGTH_MAX`                                                                            | Full reward for lines ≥ 100 chars; no minimum penalty for short lines.                                                                                                                                                                                                                                                                                             |
| Garbage density                               | `compute_quality_score` (18% weight)                             | `QS_WEIGHT_GARBAGE`, `CATEG_GARBAGE_DENSITY_HIGH`                                                              | **Halved** to 9% for lines ≤ 12 characters with zero weirdness (short-string guard) to protect clean labels with colons. Evaluated on the original text string.                                                                                                                                                                                                    |
| Vowel quality                                 | `compute_quality_score` (7% weight)                              | `QS_WEIGHT_VOWEL`, `VOWEL_RATIO_LOW`, `VOWEL_RATIO_HIGH`                                                       | Linear ramp: full score in [0.20, 0.70] vowel ratio, ramps to 0.0 outside that range.                                                                                                                                                                                                                                                                              |
| Language 🌐 confidence                        | `compute_quality_score` (5% weight)                              | `QS_WEIGHT_LANG`                                                                                               | Uses the **original** (pre-remapping) **FastText** 🌐 score; defaults to 0.5 when unavailable.                                                                                                                                                                                                                                                                     |
| Gibberish ratio                               | `compute_quality_score` (4% weight)                              | `QS_WEIGHT_GIBBERISH`                                                                                          | Words ≥ 60% digits/separators excluded. Detection only on words ≥ 4 characters. Folds in the w/x count.                                                                                                                                                                                                                                                            |
| Fused word ratio                              | `compute_quality_score` (3% weight)                              | `QS_WEIGHT_FUSED`, `FUSED_VOWEL_RUN_MIN`                                                                       | Triggers on tokens > 14 chars, consonant runs of 5+, or vowel runs of 3+.                                                                                                                                                                                                                                                                                          |
| Hard sweep (Trash override 1a)                | `categorize_line`                                                | `HARD_SWEEP_LANG_MAX`, `HARD_SWEEP_PPL_MIN`                                                                    | Fires before quality-score thresholds; folds to `trash_threshold`.                                                                                                                                                                                                                                                                                                 |
| Inverted/mirrored lexicon (Trash override 1b) | `categorize_line`, `analyze_rotation_signals`                    | `GHOST_DOMINATED_MIN_RATIO`                                                                                    | Bypassed by any Czech diacritic or upright whitelist word; folds to `trash_threshold`.                                                                                                                                                                                                                                                                             |
| Garbage-density hard override (2a)            | `categorize_line`                                                | `CATEG_GARBAGE_DENSITY_HIGH`                                                                                   | Fires before the short-garbage route and QS thresholds; folds to `trash_threshold`.                                                                                                                                                                                                                                                                                |
| Short noisy penalty                           | `compute_quality_score` (dynamic, subtracted post-normalisation) | `SHORT_NOISY_QS_PENALTY`                                                                                       | Applies a flat penalty to short strings (≤12 chars) that exhibit minor OCR oddities like elevated word weirdness or high garbage density.                                                                                                                                                                                                                          |
| All-caps + low vowel override (Override 2)    | `categorize_line`                                                | none (hardcoded)                                                                                               | Fires only if **all** alphabetic words are uppercase **and** `vowel_ratio < 0.10`. A legitimate all-caps header with normal vowel density is **not** affected. **Recorded as** `allcaps_novowel`                                                                                                                                                                   |
| High LM confidence override (Override 3)      | `categorize_line`                                                | `LOWPPL_CLEAR_MAX` (NOT `PERPLEXITY_THRESHOLD_MAX`)                                                            | Requires both `ppl < LOWPPL_CLEAR_MAX` (def. 50.0) **and** `word_count ≥ 3`; single-word lines are never fast-tracked to `Clear` by this override. Capped at Noisy if mostly-readable threshold triggers. **Recorded as** `lowppl_clear`                                                                                                                           |
| Near-boundary promotion (Override 4)          | `categorize_line`                                                | `CLEAN_PROSE_MIN_SCORE`, `CLEAN_PROSE_WC_MIN`, `CLEAN_PROSE_WEIRD_MAX`, `CLEAN_PROSE_PPL_MAX`                  | Only fires within the `Noisy` band (score ∈ [0.65, 0.85)). Requires all four conditions simultaneously. **Recorded as** `cleanprose_clear`                                                                                                                                                                                                                         |
| Mostly readable valid cap                     | `categorize_line`                                                | `MOSTLY_READABLE_VALID_MIN`                                                                                    | Capps semi-readable strings at `Noisy` that might otherwise slip into `Clear` via override 3 or 4.                                                                                                                                                                                                                                                                 |
| Short **perplexity** 📉 cap                   | `langID_classify.py` (before scoring)                            | `SHORT_PPL_CAP`                                                                                                | Applied only to lines with ≤ 2 words. Does not change the stored `perplex` column; affects only the value passed to quality scoring.                                                                                                                                                                                                                               |
| Language 🌐 remapping                         | `langID_classify.py` (before scoring)                            | `EXPECTED_LANGS`, `TRUSTED_FOREIGN_LANGS`, `LANG_SCORE_REMAP`                                                  | Unknown languages remapped to first entry of `EXPECTED_LANGS`; score **capped (#3)** at `LANG_SCORE_REMAP` (Latin) or `LANG_SCORE_REMAP_FAR` (non-Latin), except `slk` which retains its original score. The original score is still used as the `QS_WEIGHT_LANG` input.                                                                                           |
| Context smoothing (rolling window)            | Post-processing in `langID_classify.py`                          | `CATEG_TRASH_SCORE_MAX`                                                                                        | `Noisy` line must be surrounded by 2 `Trash` lines on **each** side (4 total); score must be < `Trash` threshold + 0.15. **Recorded as** `pp_surrounded_trash`                                                                                                                                                                                                     |
| Page-level inverted-scan sweep                | Post-processing in `langID_classify.py`                          | `ROT_RATIO_INVERTED_MIN`, `PPL_INVERTED_MIN`, `LANG_SCORE_ROUGH`, `INVERTED_RUN_MIN`, `INVERTED_PAGE_MAJORITY` | A line is suspicious via one of two arms (diacritic-absence + low confidence, or rotation ratio + high **perplexity** 📉). Suspicious lines are Trashed when they form a run of **at least 4** consecutive non-`Empty`/`Non-text` lines, **or (#3)** when they are **≥ 60 %** of the page's scoreable lines (page-majority arm). **Recorded as** `pp_inverted_run` |
| Header/footer deduplication                   | Post-processing in `langID_classify.py`                          | none                                                                                                           | Based on **exact text match** across the whole document; harmonises to modal category. **Recorded as** `pp_dedup`                                                                                                                                                                                                                                                  |

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
python3 langID_aggregate_STAT.py
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
the core `text_util_langID` quality classification engine over HTTP.

The batch pipeline and the API service share the same `text_util_langID` categorization engine and `config_langID.txt`
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
`ATRIUM_RUNNER_IMAGE`) take precedence over the static fallback in [para_config.txt](para_config.txt) 📎 — so the log
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
> [para_config.txt](para_config.txt) 📎 (component → license) and [para_licenses.py](para_licenses.py) 📎
> (restrictiveness ranking + share-alike / non-commercial rules), so the licensing owner can adjust it without touching
> the logger.

Each repository ships a [para_config.txt](para_config.txt) 📎 listing its components. Components flagged `always` count
toward every run (the worst-case baseline); components flagged `conditional` are only counted when the script that uses
them records it. For this repository the components and their effect on the **effective output license** 📜 are:

| Component                                                                              | License         | Counted     | Used by                                                        |
|----------------------------------------------------------------------------------------|-----------------|-------------|----------------------------------------------------------------|
| **alto-tools** 🔧 [^1](https://github.com/cneud/alto-tools)                            | Apache-2.0      | always      | page split, statistics, alto-tools text extraction             |
| **FastText** 🌐 [^2](https://huggingface.co/facebook/fasttext-language-identification) | CC BY-NC 4.0    | always      | language identification (`langID_classify.py`)                 |
| **Qwen2.5-0.5B** 🤖 [^6](https://huggingface.co/Qwen/Qwen2.5-0.5B)                     | Apache-2.0      | conditional | **perplexity** 📉 scoring (default, `langID_classify.py`)      |
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

```

[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^4]: https://atrium-research.eu/
[^5]: https://github.com/ufal/atrium-nlp-enrich
[^6]: https://huggingface.co/Qwen/Qwen2.5-0.5B
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b
