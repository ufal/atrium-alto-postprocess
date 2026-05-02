<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" title="Python Version"></a>
  <a href="https://huggingface.co/facebook/fasttext-language-identification"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-fasttext--langID-yellow.svg" title="FastText Language Identification"></a>
  <a href="https://huggingface.co/Qwen/Qwen2.5-0.5B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-Qwen2.5--0.5B-yellow.svg" title="Qwen2.5-0.5B Perplexity"></a>
  <a href="https://github.com/cneud/alto-tools"><img src="https://img.shields.io/badge/dep-alto--tools-lightgrey.svg" title="alto-tools"></a>
  <a href="https://opensource.org/license/mit/"><img src="https://img.shields.io/github/license/ufal/atrium-alto-postprocess" title="MIT License"></a>
  <a href="https://atrium-research.eu/"><img src="https://img.shields.io/badge/funded%20by-ATRIUM-8A2BE2.svg" title="ATRIUM Project"></a>
</p>

---

# 📦 ALTO XML Files Postprocessing Pipeline

This project provides a complete workflow for processing ALTO XML files. It takes raw ALTO
XMLs and transforms them into structured statistics tables, performs text classification,
and filters low-quality OCR results.

The core of the quality filtering relies on language identification and a composite quality
score — combining structural detectors, perplexity, and character-level metrics — to identify
and categorize noisy or unreliable OCR output.

---

## 📖 Table of Contents

- [⚙️ Setup](#-setup)
- [🛤️ Workflow Stages](#-workflow-stages)
  - [Step 1: Split Document-Specific ALTOs into Pages ✂️](#-step-1-split-document-specific-altos-into-pages-)
  - [Step 2: Create Page Statistics Table 📈](#-step-2-create-page-statistics-table-)
  - [Step 3: Extract text from ALTO XML ⛏️](#-step-3-extract-text-from-alto-xml-)
    - [LayoutReader method 📐](#1st-choice-layoutreader-method-)
    - [alto-tools method 🧰](#2nd-option-alto-tools-method-)
    - [GLM method 🤖](#3rd-alternative-glm-method-llm-based-)
  - [Step 4: Classify Page Text Quality & Language 🗂️](#-step-4-classify-page-text-quality--language-)
    - [4.1 Classify Lines (GPU Bound) 🚀](#41-classify-lines-gpu-bound-)
    - [4.2 Aggregate Statistics (Memory Bound) 🧠](#42-aggregate-statistics-memory-bound-)
  - [Paradata logging 🗒️](#paradata-logging)
- [Acknowledgements 🙏](#acknowledgements-)

---

## ⚙️ Setup

Before you begin, set up your environment.

1.  Create and activate a new virtual environment in the project directory 🖥.
2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the FastText model 😊 for language identification:
    ```bash
    wget "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin" -O lid.176.bin
    ```
4. Clone and install `alto-tools` 🔧, which is used for statistics and text extraction in low memory environments:
    ```bash
    git clone https://github.com/cneud/alto-tools.git
    cd alto-tools
    pip install .
    cd ..
    ```
5. Copy the `v3` folder from the `layoutreader` 🔧 repository [^9] to the project directory for the LR-based text extraction method:
    ```bash
    git clone https://github.com/ppaanngggg/layoutreader.git
    cp -r layoutreader/v3/ ./
    rm -rf layoutreader/
    ```

You are now ready to start the workflow.

---

## 🛤️ Workflow Stages

The process is divided into sequential steps, starting from raw ALTO files 📄 and ending
with extracted linguistic and statistic data 📊.

---

### ▶️ Step 1: Split Document-Specific ALTOs into Pages ✂️

First, ensure you have a directory 📁 containing your document-level `<file>.alto.xml` files.
This script will split them into individual page-specific XML files 📄.

    python3 page_split.py <input_dir> <output_dir>

Each page-specific file retains the header from its original source document 📌.

* **Input 📥:** `../ALTO/` (input directory with ALTO XML documents)
* **Output 📤:** `../PAGE_ALTO/` (output directory with ALTO XML files split into pages)

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
foundational CSV statistics file 📑.

    python3 alto_stats_create.py <input_dir> -o output.csv

This script writes a CSV file line-by-line, capturing metadata for each page:

    file, page, textlines, illustrations, graphics, strings, path
    CTX200205348, 1, 33, 1, 10, 163, /lnet/.../A-PAGE/CTX200205348/CTX200205348-1.alto.xml
    CTX200205348, 2, 0, 1, 12, 0, /lnet/.../A-PAGE/CTX200205348/CTX200205348-2.alto.xml
    ...

The extraction is powered by the **alto-tools** framework [^1].

* **Input 📥:** `../PAGE_ALTO/` (input directory with ALTO XML files split into pages from Step 1)
* **Output 📤:** `output.csv` (table with page-level statistics and paths to ALTO files)

> [!IMPORTANT]
> This statistics table is the basis for subsequent processing steps.
> Example: [test_alto_stats.csv](test_alto_stats.csv) 📎.

---

### ▶️ Step 3: Extract text from ALTO XML ⛏️

This script runs in parallel ⚡ (using multiple **CPU** cores 💻) to extract text from ALTO XMLs into `.txt` files.
It reads the CSV from Step 2.

* **Input 1 📥:** `output.csv` (from Step 2)
* **Input 2 📥:** `../PAGE_ALTO/` (input directory with ALTO XML files split into pages from Step 1)
* **Output 📤:** `../PAGE_TXT/` or `../PAGE_TXT_LR/` (directory containing raw text files)

#### 1st choice: LayoutReader method 📐

> [!CAUTION]
> The model responsible for spatial layout analysis requires a **GPU** to run efficiently.

    python3 extract_LytRdr_ALTO_2_TXT.py

Uses the LayoutReader framework [^9] to extract text and bounding boxes of XML elements
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
> The method is **CPU**-bound and faster than the LayoutReader method, but the text lines may not be in the correct
> reading order, and full forms of hyphenated split words are not reconstructed.

    python3 extract_ALTO_2_TXT.py

Uses the `alto-tools` framework [^1] to extract text lines from XML elements directly,
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
> The method is **GPU**-bound, slower than the LayoutReader method, and requires a `gpuram48G` card.

    python3 extract_LLM_ALTO_2_TXT.py

Uses the GLM-4v-9b multimodal large language model [^10] to perform generative OCR directly from
page images, prompted as `Transcribe all text on this page exactly as it appears`. The script
trims whitespace and resizes high-resolution images to fit model constraints.

> [!NOTE]
> This method is significantly slower than parsing XML but often yields higher quality text for complex
> layouts or degraded scans. It patches the transformers configuration to run the GLM-4v architecture.

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

This is a key ⌛ time-consuming step that analyzes the text quality of each page line-by-line,
assigning each line a quality category to filter out OCR noise 🔇.

It uses the [FastText language identification model](https://huggingface.co/facebook/fasttext-language-identification) 😊
and perplexity scores from [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 😊 to detect noise [^2] [^6].

More post-processing of TXT files can be found in the [GitHub repository](https://github.com/ufal/atrium-nlp-enrich)
of the ATRIUM project, which covers NLP enrichment using Nametag for NER and UDPipe for CONLL-U files with lemmas & POS tags [^5].

As the script processes, it assigns each line one of five categories 🪧:

|    Category     |                       Action                       | Description                                                                                                                                                                 |
|:---------------:|:--------------------------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   ✅ **Clear**   |        Ready to be processed by further NLP        | Passes all structural checks; high composite quality score.                                                                                                                 |
|  ⚠️ **Noisy**   | Corrections of generally readable words are needed | Partially degraded: moderate quality score indicating isolated symbol issues, fused tokens, mid-word uppercase, or elevated perplexity.                                     |
|  🗑️ **Trash**  |     Should be re-processed by another OCR tool     | Severely corrupted: high garbage density or a composite quality score below the Trash threshold.                                                                            |
| 🔣 **Non-text** |   May be checked for identifiers of finds/sites    | Filtered by the CPU pre-filter: line is too short, has too few unique symbols, contains fewer than 30% alphabetic characters, or consists mostly of digits and punctuation. |
|  🫙 **Empty**   |                   Can be ignored                   | Line contains only whitespace (paragraphs separator)                                                                                                                        |

> [!NOTE]
> This script generates two primary output directories:
> `DOC_LINE_LANG_CLASS/` and `DOC_LINE_STATS/`, while the
> raw text files (primary input) are stored in `../PAGE_TXT/` generated from `../PAGE_ALTO/`.

All input/output paths and tunable parameters are configured in [config_langID.txt](config_langID.txt) 📎.
Parameters are organized into **three sections**: `[CLASSIFY]`, `[AGGREGATE]`, and `[TEXT_UTILS]`.

<details>

<summary>Default config parameters briefly commented 👀</summary>

 
```ini
[CLASSIFY]
BATCH_SIZE = 32        # Batch size for processing lines
WORKERS_MAX = 32        # Max CPU workers for parallel tasks
EXPECTED_LANGS = ces,deu,eng    # Expected languages (ISO codes); first is default
TRUSTED_FOREIGN_LANGS = deu,eng,fra,pol,ita     # Allowed foreign languages (ISO codes)
MODEL_NAME = Qwen/Qwen2.5-0.5B  # Language model for perplexity scoring; English-only collections: distilgpt2 

[TEXT_UTILS]

PERPLEXITY_THRESHOLD_MAX = 1000.0       # Normalization ceiling for quality score (Qwen2.5-0.5B range)
LANG_SCORE_ROUGH = 0.45     # Threshold for rough language confidence
LANG_SCORE_CLEAR = 0.75     # Threshold for clear language confidence
ALLOWED_INTERNAL = .-,+()"'_—–:%;?!/        # Allowed punctuation inside words
STRIP_CHARS = .,;:!?()[]"'\/\       # Characters to strip from word edges

QS_WEIGHT_VALID_WORD  = 0.3     # Weight for valid word ratio in QS
QS_WEIGHT_SYMBOL      = 0.2     # Weight for symbol ratio in QS
QS_WEIGHT_WEIRD       = 0.2     # Weight for weirdness ratio in QS
QS_WEIGHT_PERPLEXITY  = 0.2     # Weight for perplexity in QS
QS_WEIGHT_LENGTH      = 0.1     # Weight for length in QS
QS_LENGTH_MAX         = 100     # Max length for normalization

CATEG_GARBAGE_DENSITY_HIGH  = 0.35      # High garbage density for Trash
CATEG_GARBAGE_DENSITY_SHORT = 0.20      # Garbage density for short lines
CATEG_GARBAGE_SHORT_WC      = 3     # Word count for short line checks
CATEG_TRASH_SCORE_MAX       = 0.40      # Max QS for Trash category
CATEG_NOISY_SCORE_MAX       = 0.70      # Max QS for Noisy category
CATEG_PPL_SHORT_MAX         = 700.0     # Perplexity ceiling for short-line trap (Qwen2.5-0.5B; was 2000.0 for distilgpt2)
CATEG_PPL_WEIRD_MAX         = 400.0     # Perplexity ceiling for weird+high-ppl Trash catch (Qwen2.5-0.5B; was 1000.0)
```

</details>

Parameters that depend on the perplexity model choice are tabulated below:

| Parameter                  | Qwen2.5-0.5B | distilgpt2 |
|----------------------------|--------------|------------|
| `PERPLEXITY_THRESHOLD_MAX` | 1000.0       | 2500.0     |
| `CATEG_PPL_SHORT_MAX`      | 700.0        | 2000.0     |
| `CATEG_PPL_WEIRD_MAX`      | 400.0        | 1000.0     |
---

#### 4.1 Classify Lines (GPU Bound) 🚀

This script reads the extracted text files, batches lines together 📦, and runs the FastText [^2]
and Qwen2.5-0.5B [^6] models. It uses a **CPU/GPU split architecture**:

- A single dedicated **GPU worker** holds the only Qwen2.5-0.5B instance and processes perplexity batches 
to prevent VRAM OOM errors.
- Multiple **CPU workers** (up to `WORKERS_MAX`, default 32) read files, run FastText and structural detectors, 
and submit text batches to the GPU worker via a shared queue. CPU workers poll the result dictionary while 
the GPU processes, running language identification concurrently.

> [!WARNING]
> The first of `EXPECTED_LANGS` list of languages should be the most expected language in the processed 
> collection to work as a default replacement of ambiguous language recognition predictions.


    python3 langID_classify.py

* **Input 1 📥:** `../PAGE_TXT/` from Step 3
* **Input 2 📥:** `output.csv` from Step 2
* **Output 📤:** `DOC_LINE_LANG_CLASS/` containing per-document CSVs (e.g., [DOC_LINE_QWEN_CATEG](data_samples/DOC_LINE_QWEN_CATEG) 📁)

> [!TIP]
> This script is resume-capable. If interrupted, run it again and already-present output files will be skipped.

`<doc_name>.csv`: Detailed classification results for every single line within a document, with columns:

* `file` — document identifier 🆔
* `page_num` — page number 📄
* `line_num` — line number, starts from 1 for each page 🔢
* `text` — original text of the line 📝
* `split_ws` — hyphenated word prefix at the end of the line (split word start)
* `split_we` — hyphenated word suffix at the start of the line (split word end)

Predicted or computed features for each line:

* `lang` — predicted ISO language code from the FastText model ([full list](https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200)) 🌐
* `lang_score` — FastText confidence score for the predicted language 🎯
* `perplex` — Qwen2.5-0.5B (or any other model of your choice, like `distilgpt2` for English) perplexity score of the line 📉
* `word_count` — number of whitespace-delimited tokens in the line
* `char_count` — total character count of the line
* `garbage_density` — ratio of non-alphanumeric, non-standard-punctuation characters to total line length
* `symbol` — count of words containing disallowed internal symbols (see detectors below)
* `upper` — count of words with unexpected mid-word uppercase letters
* `repeated` — count of words where a non-standard character makes up ≥ 40% of the word
* `ldl_fuses` — count of words with a letter–digit–letter sandwich (e.g., `w0rd`)
* `gibberish` — count of words flagged as gibberish (all-caps, no vowels, or extreme vowel ratio)
* `word_weird` — mean per-word weirdness score in [0, 1]; combines strange-symbol, repeated-symbol, LDL-fusion, 
and mid-uppercase signals weighted per token (0 = fully clean). *Note: Random isolated letters receive a severe weirdness 
penalty (0.85) to catch spaced-out OCR noise, while isolated numbers/measurements receive a lower, tolerable penalty (0.25).*
* `vowel_ratio` — ratio of vowel characters to total alphabetic characters in the line
* `quality_score` — composite quality score in [0, 1] based on valid-word ratio, symbol ratio, perplexity, 
text length, and word weirdness; higher = cleaner 📈
* `categ` — assigned category: **Clear** ✅, **Noisy** ⚠️, **Trash** 🗑️, **Non-text** 🔣, or **Empty** 🫙
* `caps_header` — boolean flag indicating whether all alphabetic words in the line are uppercase (typical of section headers)

##### CPU Pre-filter

Before any GPU or model inference, `pre_filter_line()` applies a fast CPU-side check and assigns `Empty` or `Non-text` 
directly, bypassing the ML pipeline entirely:

* Line is blank → **Empty**
* Fewer than 4 characters, or fewer than 3 unique non-whitespace symbols → **Non-text**
* Letter ratio below 30% of total characters → **Non-text**
* Matches the all-digits/symbols regex pattern → **Non-text**
* Otherwise → forwarded for ML classification as **Process**

##### Language Handling

FastText is run on the lowercased line text. If the predicted language is not in either `EXPECTED_LANGS` or 
`TRUSTED_FOREIGN_LANGS`, the language is force-remapped to the first entry of `EXPECTED_LANGS` (default `ces`), 
**preserving the FastText script suffix (e.g., `_Latn`)**, with a minimum confidence of `LANG_SCORE_CLEAR` (default 0.75). 
This prevents foreign-language false positives from polluting the quality assessment for predominantly Czech collections, 
while keeping the output format stable for downstream consumers.

##### Structural Detectors

Lines that pass the pre-filter are analysed by five structural detectors defined in `text_util_langID.py`:

| Detector                     | What it counts                                                                                                                                                                                                 |
|------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `detect_strange_symbols`     | Words containing any character that is not alphanumeric and not in the **allowed** set `{ . - , + ( ) " ' / _ — – : % }`. Edge punctuation is stripped before inspection.                                      |
| `detect_letter_digit_letter` | Words with a **letter–digit–letter sandwich** — the fingerprint of OCR digit insertions mid-word (e.g., `vyt1ačená`, `nalez2í`). **Legitimate** patterns like `90,9g`, `80-90cm`, `26.IX.1957` do not trigger. |
| `detect_mid_uppercase`       | Words with unexpected uppercase mid-word (`dalSÍ`, `obkLADem`) or an uppercase run at the start followed by lowercase (`XXWžkumu`). All-caps words and **titles** (`PhDr`, `MUDr`) are **excluded**.           |
| `detect_repeated_chars`      | Words where a single non-standard character makes up ≥ 40% of the word and appears at least **3 times** (e.g., OCR stutter like `bxxxoxx`).                                                                    |
| `detect_gibberish_words`     | Words of length ≥ 4 that contain no vowels, or have a vowel ratio below 15% or above 80%. Words that are **predominantly numeric** (≥ 60% digits and separators) are **excluded**.                             |

##### Composite Quality Score

After structural detection, each line receives a single floating-point `quality_score` in [0, 1] computed by 
`compute_quality_score()` in `text_util_langID.py`. The score is a weighted sum of five normalised signals:
```text
quality_score =
    QS_WEIGHT_VALID_WORD (def: 0.3)  × valid_word_ratio                    # share of structurally clean words
  + QS_WEIGHT_SYMBOL (def: 0.2)      × (1 − min(symbol_ratio, 1.0))       # inverted non-alphanumeric density
  + QS_WEIGHT_WEIRD (def: 0.2)       × (1 − min(word_weird_ratio, 1.0))   # inverted mean per-word weirdness
  + QS_WEIGHT_PERPLEXITY: (def 0.2)  × (1 − min(perplexity / PERPLEXITY_THRESHOLD_MAX, 1.0)) # inverted normalised perplexity
  + QS_WEIGHT_LENGTH (def: 0.1)      × min(char_count / QS_LENGTH_MAX, 1.0)  # reward for longer lines
```

The two scale parameters that normalize unbounded signals before weighting in the quality score formula are
`PERPLEXITY_THRESHOLD_MAX` (default **1000.0**), which caps raw perplexity to map it into [0, 1] assigning 0 to values at or above the threshold 
(worst) and 1 to 0 (best), calibrated for **Qwen2.5-0.5B** on corrupted OCR text to penalize noisy lines more aggressively 
when lowered or widen the scoring range when raised; and `QS_LENGTH_MAX` (default **100**), which sets the character-count 
ceiling for rewarding longer lines, granting the full `QS_WEIGHT_LENGTH` bonus to **lines at or above this length**.

Default weights and scale parameters (all tunable in `[TEXT_UTILS]`):

> [!NOTE]
> Perplexity contributes only one weighted component of the quality score. Although `Qwen2.5-0.5B` is
> multilingual and handles **Czech**, **German**, and **English** natively (unlike the **English-only** `distilgpt2` it
> replaced), it is still intentionally diluted by the four other signals rather than used as a standalone
> threshold. This keeps the score robust against edge cases where even a strong model assigns unexpectedly
> high perplexity to valid but atypical text (e.g., highly abbreviated archival labels or form-field lines).

##### Categorisation Logic

`categorize_line()` in `text_util_langID.py` classifies each line in two stages:

**Immediate Trash overrides** (checked first, before the quality score):

* Garbage density > `CATEG_GARBAGE_DENSITY_HIGH` (default 0.35) → **Trash**
* Line has ≤ `CATEG_GARBAGE_SHORT_WC` words (default 3) **and** garbage density > `CATEG_GARBAGE_DENSITY_SHORT` (default 0.20) → **Trash**
* **Severe fragmentation**: Line has ≥ 5 words, average stripped-word length < 2.0 characters, **and**
`weird_ratio` > 0.1 → **Trash** *(prevents valid measurement lines from being trashed)*.
* **High perplexity on short lines**: Perplexity > `CATEG_PPL_SHORT_MAX` (default 700.0) and < 5 words → **Trash**, with two exceptions: 
lines composed entirely of Roman numerals and standard separators are bypassed; lines where garbage density < 0.1 **and** `weird_ratio` 
< 0.20 fall back to **Noisy** instead.
* **Single-character fragmentation**: Line has ≥ 3 words, ≥ 50% of words are isolated characters, **and**
`weird_ratio` > 0.15 → **Trash** *(catches spaced-out gibberish like `"C A s 8."`)*.
* **Extreme vowel ratio**: Line > 5 characters with vowel ratio < 10% or > 90% → **Trash** *(catches random consonants/vowels like `"FAXAPOOXAXXXX"`)*.
* **High overall weirdness**: `weird_ratio` ≥ 0.25 → **Trash**.
* **Moderate weirdness + high perplexity**: `weird_ratio` > 0.15 and perplexity > `CATEG_PPL_WEIRD_MAX` (default 400.0) → **Trash**.

**Quality score thresholds** (applied to lines that pass all overrides):
```text
quality_score < CATEG_TRASH_SCORE_MAX  (def: 0.40)  →  Trash
quality_score < CATEG_NOISY_SCORE_MAX  (def: 0.70)  →  Noisy
otherwise                                              →  Clear
```

All threshold values are configurable in the `[TEXT_UTILS]` section of `config_langID.txt`.

##### Post-Processing Smoothing

After all lines in a document are classified and written to CSV, a final data-smoothing pass is applied before the file
is finalized to prevent unnatural categorization anomalies:

1. **Header/Footer Deduplication** — Resolves edge-case flip-flopping. If the exact same text string appears multiple 
times across a document, all instances are harmonized to share the statistical mode (most frequent) category assigned to that string.
2. **Context Smoothing (Rolling Window)** — Applies a 3-line rolling window. If a **Noisy** line is sandwiched between 
two consecutive **Trash** lines (one immediately before, one immediately after), it is automatically downgraded 
to **Trash** to prevent isolated "noisy" categorizations in otherwise heavily corrupted regions.

Example of per-document CSV files: [DOC_LINE_QWEN_CATEG](data_samples/DOC_LINE_QWEN_CATEG) 📁 by Qwen2.5-0.5B 
and [DOC_LINE_GPT_CATEG](data_samples/DOC_LINE_GPT_CATEG) 📁 by distilgpt2.
```
DOC_LINE_LANG_CLASS/
├── <docname1>.csv
├── <docname2>.csv
└── ...
```

---

#### 4.2 Aggregate Statistics (Memory Bound) 🧠

This script processes the `DOC_LINE_LANG_CLASS/` directory with CSV files in chunks 🧩 to produce
final page-level statistics. It is **CPU-bound** and parallelized with `ProcessPoolExecutor`.

```
python3 langID_aggregate_STAT.py
```

* **Input 📥:** `DOC_LINE_LANG_CLASS/` (directory with CSV files from the previous step)
* **Output 1 📤:** `arup_page_stats_SHORT.csv` — global page-level summary across all documents
* **Output 2 📤:** `DOC_LINE_STAT/` — per-document CSVs with the same schema

For each page, the aggregation computes:

**Category counts** (from all lines regardless of category):

* `Clear`, `Noisy`, `Trash`, `Non-text`, `Empty` — integer count of lines in each category

**Totals** (summed over lines classified as **Clear** or **Noisy** only — Trash, Empty, and Non-text excluded):

* `total_word_count` — total number of words across scoreable lines
* `total_char_count` — total number of characters across scoreable lines

**Averages** (mean over the same **Clear** and **Noisy** lines):

* `avg_quality_score` — mean composite quality score in [0, 1]; higher = cleaner OCR output 📈
* `avg_word_weird` — mean per-word weirdness ratio in [0, 1]; 0 = fully clean, lower is better 📉
* `avg_lang_score` — mean FastText confidence score
* `avg_perplex` — mean Qwen2.5-0.5B perplexity score
* `avg_symbol` — mean strange-symbol word count per line
* `avg_vowel_ratio` — mean vowel-to-alphabetic-character ratio per line
* `ch_ratio` — mean fraction of lines flagged as all-caps headers (`caps_header = True`)

**Language profile:**

* `main_lang` — the statistical mode (most frequent) language predicted for the page

> [!NOTE]
> `avg_*` columns and `main_lang` will be `NaN` / `None` for pages whose only lines are
> Empty or Non-text (i.e., pages with no scoreable text content).

All numeric averages are rounded to 4 decimal places; totals are stored as integers.

- *Examples*: [arub_page_stats_SHORT.csv](arub_page_stats_SHORT.csv) [arup_page_stats_SHORT.csv](arup_page_stats_SHORT.csv) 📎

Example of per-document aggregate CSV files: [DOC_LINE_QWEN_STATS](data_samples/DOC_LINE_QWEN_STATS) 📁 by Qwen2.5-0.5B 
and [DOC_LINE_GPT_STATS](data_samples/DOC_LINE_GPT_STATS) 📁 by distilgpt2:
```
DOC_LINE_STAT/
├── stats_<docname1>.csv
├── stats_<docname2>.csv
└── ...
```

This is the end of the text quality classification and filtering step. You can now use `ARUP_short_page_stats.csv` to
identify files that need another round of OCR or manual correction based on the line type counts. Pages with the
majority of clear lines can be marked for further processing. The absence of clear lines combined with a high proportion
of trash lines may also indicate handwritten content, which can be excluded before Handwritten Text Recognition (HTR) is applied.

---

## Paradata logging

This project incorporates a unified provenance and paradata logging system to seamlessly track the execution
details of every pipeline stage. The logger automatically captures run-time metadata and saves it in a
structured JSON format.

**What gets logged?**

* **Provenance 🏛️:** Captures the tool name, repository URL, Python version, and assigns a unique `run_id` to each execution.
* **Configuration ⚙️:** Stores a complete snapshot of the runtime configuration, including script names, input/output 
paths, and specific model choices.
* **Timing ⏱️:** Records precise UTC start times, end times, and the total duration of the run in seconds.
* **Statistics 📊:** Tracks the total number of input files, successfully processed documents, and computes performance 
throughput (e.g., output files generated per minute).
* **Error Tracking 🐛:** Maintains a `skipped_files_detail` list that logs the exact filename and specific error reason 
if a file fails to process.

**Log Location & Licensing**

By default, JSON logs are written to the [paradata](paradata) 📁 directory following the naming convention
`<YYMMDD-HHmmss>_<program>.json`. All generated paradata log files are distributed under the **CC BY-NC 4.0** license.

---

## Acknowledgements 🙏

**For support write to:** lutsai.k@gmail.com — responsible for this GitHub repository [^8] 🔗

- **Developed by** UFAL [^7] 👥
- **Funded by** ATRIUM [^4] 💰
- **Shared by** ATRIUM [^4] & UFAL [^7] 🔗
- **Models used**:
  - FastText [^2] for language identification
  - Qwen2.5-0.5B [^6] for perplexity scoring
  - GLM-4v-9b [^10] for generative OCR (LLM-based method)
  - LayoutLMv3 [^9] for layout-aware text extraction

**©️ 2026 UFAL & ATRIUM**

[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^3]: https://github.com/ufal/ker
[^4]: https://atrium-research.eu/
[^5]: https://github.com/ufal/atrium-nlp-enrich
[^6]: https://huggingface.co/Qwen/Qwen2.5-0.5B
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b