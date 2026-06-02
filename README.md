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
    wget "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin" -O lid.176.bin
    ```
4. Clone and install `alto-tools` 🔧, which is used for statistics and text extraction in low memory environments:
    ```bash
    git clone https://github.com/cneud/alto-tools.git
    cd alto-tools
    pip install .
    cd ..
    ```
5. Copy the `v3` folder from the 📐`layoutreader` 🔧 repository [^9] to the project directory for the LR-based text extraction method:
    ```bash
    git clone https://github.com/ppaanngggg/layoutreader.git
    cp -r layoutreader/v3/ ./
    rm -rf layoutreader/
    ```

You are now ready to start the workflow.

---

## 🛤️ Workflow Stages

The process is divided into sequential steps, starting from raw **ALTO** 📄 files and ending
with extracted linguistic and statistic data 📊.

---

### ▶️ Step 1: Split Document-Specific ALTOs into Pages ✂️

First, ensure you have a directory 📁 containing your document-level `<file>.alto.xml` files.
This script will split them into individual page-specific **XML** 📄 files.

    python3 page_split.py <input_dir> <output_dir>

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

    python3 alto_stats_create.py <input_dir> -o output.csv

This script writes a **CSV** 📊 file line-by-line, capturing metadata for each page:

    file, page, textlines, illustrations, graphics, strings, path
    CTX200205348, 1, 33, 1, 10, 163, /lnet/.../A-PAGE/CTX200205348/CTX200205348-1.alto.xml
    CTX200205348, 2, 0, 1, 12, 0, /lnet/.../A-PAGE/CTX200205348/CTX200205348-2.alto.xml
    ...

The extraction is powered by the **alto-tools** 🔧 framework [^1].

* **Input 📥:** `../PAGE_ALTO/` (input directory with **ALTO XML** 📄 files split into pages from Step 1)
* **Output 📤:** `output.csv` (table with page-level statistics and paths to ALTO files)

> [!IMPORTANT]
> This statistics table is the basis for subsequent processing steps.
> Example: [test_alto_stats.csv](test_alto_stats.csv) 📎.

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

    python3 extract_LytRdr_ALTO_2_TXT.py

Uses the **LayoutReader** 📐 framework [^9] to extract text and bounding boxes of **XML** 📄 elements
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

    python3 extract_ALTO_2_TXT.py

Uses the `alto-tools` 🔧 framework [^1] to extract text lines from **XML** 📄 elements directly,
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

    python3 extract_LLM_ALTO_2_TXT.py

Uses the **GLM-4v-9b** 🤖 multimodal large language model [^10] to perform generative **OCR** 🔍 directly from
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
and **perplexity** 📉 scores from [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) 🤖 to detect noise [^2] [^6].

More post-processing of **TXT** 📝 files can be found in the [GitHub repository](https://github.com/ufal/atrium-nlp-enrich)
of the ATRIUM project, which covers NLP enrichment using Nametag for NER and UDPipe for CONLL-U files with lemmas & POS tags [^5].

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

<details>
    <summary><strong>CONFIG.TXT 📝 variables explained (click to expand 👀)</strong></summary>

```ini
[CLASSIFY]
BATCH_SIZE = 128        # Batch size for processing lines
WORKERS_MAX = 32        # Max CPU workers for parallel tasks
EXPECTED_LANGS = ces,deu,eng    # Expected languages (ISO codes); first is default
TRUSTED_FOREIGN_LANGS = deu,eng,fra,pol,ita     # Allowed foreign languages (ISO codes)
MODEL_NAME = Qwen/Qwen2.5-0.5B  # Language model for perplexity scoring; English-only collections: distilgpt2 

[TEXT_UTILS]

PERPLEXITY_THRESHOLD_MAX = 1000.0       # or 3000 Normalization ceiling for quality score (Qwen2.5-0.5B range)
SHORT_PPL_CAP = 850.0                   # or 2500 Effective perplexity cap for 1-2 word lines

LANG_SCORE_ROUGH = 0.45     # Threshold for rough language confidence
LANG_SCORE_CLEAR = 0.75     # Threshold for clear language confidence
ALLOWED_INTERNAL = .-,+()"'—–:%;?!/        # Allowed punctuation inside words
STRIP_CHARS = .,;:!?()[]"'\/\       # Characters to strip from word edges

QS_WEIGHT_VALID_WORD  = 0.25    # Weight for valid word ratio in QS
QS_WEIGHT_SYMBOL      = 0.13    # Weight for inverted non-alphanumeric density in QS
QS_WEIGHT_WEIRD       = 0.13    # Weight for inverted word weirdness in QS
QS_WEIGHT_PERPLEXITY  = 0.15    # Weight for inverted normalized perplexity in QS
QS_WEIGHT_LENGTH      = 0.05    # Weight for length reward in QS
QS_WEIGHT_GARBAGE     = 0.20    # Weight for inverted garbage density in QS
QS_WEIGHT_VOWEL       = 0.07    # Weight for vowel quality in QS
QS_WEIGHT_LANG        = 0.05    # Weight for language confidence in QS
QS_WEIGHT_GIBBERISH   = 0.04    # Weight for inverted gibberish ratio in QS
QS_WEIGHT_FUSED       = 0.03    # Weight for inverted fused word ratio in QS
QS_LENGTH_MAX         = 100.0   # Max length for normalization

CATEG_GARBAGE_DENSITY_HIGH  = 0.35      # High garbage density used to normalize the garbage-density signal in QS
CATEG_TRASH_SCORE_MAX       = 0.50      # Max QS for Trash category
CATEG_NOISY_SCORE_MAX       = 0.90      # Max QS for Noisy category

# --- Inverted / 180°-rotated scan detection (rot_penalty inside QS + page-level post-processing sweep) ---
ROT_RATIO_INVERTED_MIN      = 0.55      # Min rotatable-char ratio to suspect inverted scan
WEIRD_RATIO_INVERTED_MIN    = 0.35      # Min word-weirdness ratio to confirm inverted scan
PPL_INVERTED_MIN            = 200.0     # or 500 Min perplexity (LM must also be uncertain)

# --- Near-boundary clean prose promotion (Override 4) ---
CLEAN_PROSE_MIN_SCORE       = 0.65      # Lower bound of the promotable Noisy band
CLEAN_PROSE_WEIRD_MAX       = 0.08      # Max word-weirdness for promotion to Clear
CLEAN_PROSE_PPL_MAX         = 400.0     # or 1000 Max perplexity for promotion to Clear
CLEAN_PROSE_WC_MIN          = 4         # Min word count for promotion to Clear
```

</details>
    
Parameters that scale with the **perplexity** 📉 model:

<details>
    <summary><strong>Perplexity-related variables 👀</strong></summary>

These parameters must be re-tuned whenever you switch between multilingual `Qwen2.5-0.5B`🤖 and English-adapted `distilgpt2`🤖, 
because the two models produce **perplexity** 📉 on very different numerical scales — `Qwen2.5-0.5B`🤖 assign scores roughly 3× lower 
than `distilgpt2`🤖 on the same Czech 🇨🇿 text:

| Parameter                  | Qwen2.5-0.5B | distilgpt2 | What it controls                                                                                                                                                                                                                                                                      |
|----------------------------|--------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PERPLEXITY_THRESHOLD_MAX` | 1000.0       | 3000.0     | The ceiling used to normalise raw **perplexity** 📉 into [0, 1] for the **quality score** 📈. A value at or above this ceiling contributes 0 to the score (worst); a value of 0 contributes 1 (best).                                                                                 |
| `SHORT_PPL_CAP`            | 850.0        | 2500.0     | Maximum **perplexity** 📉 applied to 1–2 word lines before quality scoring. Short text fragments receive extreme **perplexity** 📉 scores from any LM because there is no context to condition on; this cap prevents legitimate short labels and codes from being unfairly penalised. |
| `PPL_INVERTED_MIN`         | 200.0        | 500.0      | **Perplexity** 📉 floor for the inverted-scan detection arm. A line is considered a candidate for the inverted-scan penalty only if the LM is also uncertain about it (**perplexity** 📉 above this value).                                                                           |
| `CLEAN_PROSE_PPL_MAX`      | 400.0        | 1000.0     | Maximum **perplexity** 📉 a line may have to qualify for the near-boundary `Clear` promotion (Override 4). Lines with **perplexity** 📉 above this value are not promoted even if all other conditions are met.                                                                       |
</details>

Parameters that are model-independent 🤖 and stable across different choices of **perplexity** 📉 model 🤖:

<details>
    <summary><strong> Perplexity-independent variables 👀</strong></summary>

These 4 parameters below are expressed as ratios or quality-score fractions, not as **perplexity** 📉 values, so their meaning 
does not change between models and their defaults are stable across either choice:

| Parameter                  | Default | What it controls                                                                                                                                                                                                                                                            |
|----------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ROT_RATIO_INVERTED_MIN`   | 0.55    | Minimum fraction of structurally rotatable characters (`pbqdnuwmoxszeyv`) among alphabetic characters that must be present before a rotation penalty is even considered. A value of 0.55 means more than half of all letters in the line must belong to this ambiguous set. |
| `WEIRD_RATIO_INVERTED_MIN` | 0.35    | Minimum mean per-word weirdness score required to *confirm* an inverted scan when `rot_ratio` is already above the threshold. This second condition prevents Czech 🇨🇿 sentences that happen to contain many `p`, `d`, `b`, `q` letters from being falsely penalised.      |
| `CLEAN_PROSE_MIN_SCORE`    | 0.65    | Lower bound of the quality-score range within which the near-boundary promotion (Override 4) can fire. A line must score at least this well before it is a candidate for promotion from `Noisy` to `Clear`.                                                                 |
| `CLEAN_PROSE_WEIRD_MAX`    | 0.08    | Maximum mean per-word weirdness a line may have to qualify for the near-boundary promotion. Even a single notably corrupted token disqualifies the line from being promoted.                                                                                                |
| `CLEAN_PROSE_WC_MIN`       | 4       | Minimum word count a line must have to qualify for near-boundary promotion. Very short lines (1–3 words) have unreliable **perplexity** 📉 scores and are therefore never promoted regardless of their **quality score** 📈.                                                |

</details> 

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
* `text` — original text of the line 📝
* `split_ws` — hyphenated word prefix at the end of the line (split word start)
* `split_we` — hyphenated word suffix at the start of the line (split word end)

<details>
    <summary><strong>Predicted or computed factors (columns) for each line: 👀</strong></summary>

* `word_count` — **count** of whitespace-delimited tokens in the line (**count** of **words**)
* `char_count` — **count** of total character in the line
* `garbage_density` — ratio of non-alphanumeric, non-standard-punctuation characters to total line length
* `symbol` — **count** of **words** containing disallowed internal symbols (see detectors below)
* `upper` — **count** of **words** with unexpected mid-word uppercase letters
* `repeated` — **count** of **words** where a non-standard character makes up ≥ 30% of the word, or containing consecutive doubled garble characters
* `ldl_fuses` — **count** of **words** with a letter–digit–letter sandwich (e.g., `w0rd`)
* `fused_words` — **count** of tokens that appear to be fused **words** (abnormal consonant/vowel runs or extreme length)
* `gibberish` — **count** of **words** flagged as gibberish (all-caps, no vowels, or extreme vowel ratio)
* `word_weird` — mean per-word weirdness score in [0, 1]; combines strange-symbol, repeated-symbol, LDL-fusion,
and mid-uppercase signals weighted per token (0 = fully clean). *Note: Random isolated letters receive a severe weirdness
penalty (0.85) to catch spaced-out **OCR** 🔍 noise, while isolated numbers/measurements receive a lower, tolerable penalty (0.25).*
* `vowel_ratio` — ratio of vowel characters to total alphabetic characters in the line
* `rot_ratio` — the ratio of structurally ambiguous/rotatable characters (`pbqdnuwmoxszeyv`) to the total number of alphabetic characters in the line.

</details>

##### `<doc_name>.csv`'s key resulting output **columns** that depict the final classification and quality assessment:

* `quality_score` — composite **quality score** 📈 in [0, 1] based on 10 combined signals; higher = cleaner
* `categ` — assigned category: **Clear** ✅, **Noisy** ⚠️, **Trash** 🗑️, **Non-text** 🔣, or **Empty** 🫙

##### `<doc_name>.csv`'s **columns** useful for archive managers information apart from the **quality score** 📈 and category:

* `lang` — predicted ISO **language code** from the **FastText** 🌐 model ([full list](https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200)) 🌐
* `lang_score` — **FastText** 🌐 [^2] confidence score for the predicted language 
* `perplex` — Qwen2.5-0.5B 🤖 [^6] (or any other model of your choice, like `distilgpt2` 🤖 for English 🇬🇧) **perplexity** 📉 score of the line 📉
* `caps_header` — **boolean** flag indicating whether all alphabetic words in the line are uppercase (typical of section headers)

---

##### CPU 💻 Pre-filter

Before any **GPU** 🚀 or model inference, `pre_filter_line()` applies a fast **CPU** 💻-side check and assigns `Empty` or `Non-text`
directly, bypassing the ML pipeline entirely. It also applies two lightweight **OCR** 🔍 text repairs to every line before
the rules are evaluated.

Firstly, two fixes correct the most common systematic **OCR** 🔍 substitution errors before any rule is checked. They modify
the text that is passed forward but do not on their own affect what category a line receives.

<details>
    <summary><strong>Step 1 -  Minor OCR 🔍 repairs (applied first, to every line): 👀</strong></summary>

* **Digit-for-letter substitution:** A `1` surrounded by alphabetic characters on both sides is replaced with `l`
(e.g., `poh1ed` → `pohled`); a `2` at the start of a token followed immediately by a lowercase letter is replaced
with `z`. These substitutions reflect common **OCR** 🔍 confusions between visually similar characters.
* **Spaced-letter collapse:** A sequence of individually spaced single uppercase letters (`P R A H A`) is recognised
as a prostrkávání/spaced-text typographic style and collapsed back into a normally-cased word (`Praha`). Without this
repair, spaced words fail the letter-ratio check and would be discarded as `Non-text`.

</details>

<details>
    <summary><strong>Step 2 — Standard `Non-text` / `Empty` rules (checked in order from 1 to 8; first match wins): 👀</strong></summary>

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
8. **Otherwise** → forwarded for ML classification as `Process`

</details>  

Finally, two categories of exception send a line directly to `Process` even if it would otherwise be caught by a `Non-text` rule:

<details>
    <summary><strong>Step 3 — Bypass exceptions (override the rules above, checked before rules 2–7): 👀</strong></summary>

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

</details>

---

##### Language Handling

**FastText** 🌐 [^2] is run on the **lowercased** line text and returns a predicted ISO 639-3 language code
(e.g. `ces` for Czech 🇨🇿, `deu` for German 🇩🇪) and a confidence score between 0 and 1. The pipeline then applies
a series of remapping rules before the `lang` and `lang_score` fields are finalised for storage and before
the score is used in quality computation.

<details>
    <summary><strong>Language-related config ⚙️ parameters (click to expand 👀)</strong></summary>
        
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

* `LANG_SCORE_ROUGH = 0.45` — a **FastText** 🌐 confidence below this is considered too unreliable to trust. This threshold
is used by the page-level inverted-scan sweep (see Post-Processing Smoothing) to identify pages where **FastText** 🌐 cannot
confidently assign any language to any line — a strong signal that the page content is not readable text.
* `LANG_SCORE_CLEAR = 0.75` — the minimum confidence floor assigned to lines whose language has been force-remapped
to the collection default. See remapping rule 2 below.

</details>

**Remapping logic (applied per line, in order):**

1. If the predicted language 🌐 code appears in `EXPECTED_LANGS` or `TRUSTED_FOREIGN_LANGS` → the **FastText** 🌐 prediction
and confidence score are **kept unchanged**. No remapping occurs.

2. If the predicted language 🌐 is **not** in either set (e.g., **FastText** 🌐 guesses Slovenian `slv` or Slovak `slk` on a
Czech 🇨🇿 line) → the language 🌐 code is **force-remapped** to the **first entry of `EXPECTED_LANGS`** (the collection
default). Any script suffix from the original **FastText** 🌐 output (e.g., `_Latn`, `_Cyrl`) is preserved in the remapped
code. The stored `lang_score` is set to `max(original_score, LANG_SCORE_CLEAR)`, meaning it is floored upwards to at
least **0.75** (the `LANG_SCORE_CLEAR` default). This prevents nearby-language false positives (e.g., Slovak or
Polish being predicted on standard Czech 🇨🇿 text) from artificially lowering the language-confidence component of the
**quality score** 📈 for otherwise clean lines.

**What gets stored vs. what drives the **quality score** 📈:**

These two values are intentionally different:

* The **stored `lang_score` column** in the output **CSV** 📊 reflects the post-remapping value (i.e., floored to
`LANG_SCORE_CLEAR` if remapping occurred). This is what you see in the file.
* The **quality score 📈 computation** (`compute_quality_score()`) receives the **original pre-remapping **FastText** 🌐
confidence** (`original_lang_score`). This means the `QS_WEIGHT_LANG` component of the **quality score** 📈 honestly
reflects how confident **FastText** 🌐 was about the line's language — not the artificially raised value. A line where
**FastText** 🌐 was genuinely uncertain gets a lower language-confidence contribution to its **quality score** 📈 regardless of
what language 🌐 code is stored.

<details>
    <summary><strong>Diacritic-based language inference of edge cases [NOT USED in categorization] 👀</strong></summary>

The codebase contains a helper function `infer_lang_from_diacritics()` that can attempt to assign a language 🌐 purely
from the density of language-specific diacritic characters in a line (e.g., a high density of `á č ď é ě í ň ó ř š ť ů ú ý ž`
suggests Czech 🇨🇿; a high density of `ä ö ü ß` suggests German 🇩🇪). **This function is not called in the main
classification pipeline.** It exists as a utility for downstream analytics, debug tooling, or future pipeline
extensions that may need a fast, model-free language 🌐 signal. It does not affect `lang`, `lang_score`, or
`quality_score` in any output file.
</details>

---

##### Structural Detectors

<details>
    <summary><strong>Structural detectors analysis of the PER-WORD inputs (click to expand 👀)</strong></summary>

Lines that pass the pre-filter are analysed by structural detectors defined in [text_util_langID.py](text_util_langID.py)📎:

| Detector                     | What it counts                                                                                                                                                                                    |
|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `detect_strange_symbols`     | Words containing any character that is not alphanumeric and not in the **allowed** set `{ . - , + ( ) " ' — – : % ; ? ! / }`. Edge punctuation is stripped before inspection.                     |
| `detect_letter_digit_letter` | Words with a **letter–digit–letter sandwich** — the fingerprint of **OCR** 🔍 digit insertions mid-word (e.g., `vyt1ačená`). **Legitimate** patterns like `90,9g` do not trigger.                 |
| `detect_mid_uppercase`       | Words with unexpected uppercase mid-word (`dalSÍ`). All-caps words and specific capitalized sequences are **excluded**.                                                                           |
| `detect_repeated_chars`      | Words where a single character makes up ≥ 30% of the word and appears at least 3 times, or contains unnatural consecutive doubles. Explicitly ignores common Czech 🇨🇿 vowels (`a, e, i, o, u`). |
| `detect_gibberish_words`     | Words of length ≥ 4 that contain no vowels, or have a vowel ratio below 15% or above 80%. Words that are **predominantly numeric** (≥ 60% digits and separators) are **excluded**.                |
| `compute_rotatable_ratio`    | Measures the concentration of structurally ambiguous/rotatable letters (`pbqdnuwmoxszeyv`) to catch severe visual noise interpreting graphical textures as characters.                            |
| `detect_fused_words`         | Counts tokens that are likely multiple words merged without a space (e.g. token length > 14, unnatural consonant run of 5+, or vowel run of 4+).                                                  |

</details>

##### Composite Quality Score

After structural detection, each line receives a single floating-point `quality_score` 📈 in [0, 1] computed by
`compute_quality_score()` in [text_util_langID.py](text_util_langID.py)📎. The score is a weighted sum of ten 
normalised signals, **dynamically divided by the total sum of weights** to strictly bound the maximum 
score to 1.0 (preventing score inflation):

```text
base_score =
    QS_WEIGHT_VALID_WORD (def: 0.25) × valid_word_ratio
  + QS_WEIGHT_SYMBOL     (def: 0.13) × (1 − min(symbol_ratio, 1.0))
  + QS_WEIGHT_WEIRD      (def: 0.13) × (1 − min(word_weird_ratio, 1.0))
  + QS_WEIGHT_PERPLEXITY (def: 0.15) × (1 − min(perplexity / PERPLEXITY_THRESHOLD_MAX, 1.0))
  + QS_WEIGHT_LENGTH     (def: 0.05) × min(char_count / QS_LENGTH_MAX, 1.0)
  + active_garbage_wt    (def: 0.20) × (1 − min(garbage_density / CATEG_GARBAGE_DENSITY_HIGH, 1.0))
  + QS_WEIGHT_VOWEL      (def: 0.07) × vowel_quality_score
  + QS_WEIGHT_LANG       (def: 0.05) × lang_score
  + QS_WEIGHT_GIBBERISH  (def: 0.04) × (1 − min(gibberish_ratio, 1.0))
  + QS_WEIGHT_FUSED      (def: 0.03) × (1 − min(fused_ratio, 1.0))

quality_score = (base_score / total_weight) - rot_penalty
```

<details>
    <summary><strong>Signal definitions and normalisations (click to expand 👀)</strong></summary>

| Signal             | Source                                                              | Normalisation                                                           | Notes                                                                                                                                                                                                                                                              |
|--------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `valid_word_ratio` | fraction of structurally valid word tokens                          | used directly [0, 1]                                                    | A token is valid if ≥ 70% alphabetic, no disallowed internal symbols, and not a garbled all-caps **OCR** 🔍 prefix followed by lowercase (e.g., `AAMMNAbSSOAO`, `XAterenta`) — this guard prevents spurious uppercase runs from inflating the signal.              |
| `symbol_ratio`     | fraction of non-alphanumeric, non-space characters                  | inverted: `1 − ratio`                                                   |                                                                                                                                                                                                                                                                    |
| `word_weird_ratio` | mean per-word weirdness across tokens                               | inverted: `1 − ratio`                                                   | The per-word score combines strange-symbol (0.40), repeated-char (0.35), LDL-fusion (0.15), and mid-uppercase (0.10) sub-signals, plus a separate caps-prefix penalty (0.20). Isolated single letters score 0.85 ( **OCR** 🔍  noise) or 0.25 (digit/measurement). |
| `perplexity` 📉    | `Qwen2.5-0.5B` 🤖 NLL per token                                     | inverted & capped: `1 − min(ppl / PERPLEXITY_THRESHOLD_MAX, 1.0)`       | A value of 0 assigned when ppl ≥ threshold (worst), 1 when ppl = 0 (best). Calibrated for `Qwen2.5-0.5B` 🤖 at default `1000.0`; use `3000.0` for `distilgpt2` 🤖.                                                                                                 |
| `char_count`       | total character count                                               | `min(count / QS_LENGTH_MAX, 1.0)`                                       | Full reward for lines ≥ `QS_LENGTH_MAX` (default 100) characters.                                                                                                                                                                                                  |
| `garbage_density`  | fraction of unusual non-alnum characters after stripping `...` runs | inverted & capped: `1 − min(density / CATEG_GARBAGE_DENSITY_HIGH, 1.0)` |                                                                                                                                                                                                                                                                    |
| `vowel_ratio`      | vowel fraction among alphabetic characters                          | linear ramp: full reward in [0.20, 0.75], ramps down to 0.0 at extremes |                                                                                                                                                                                                                                                                    |
| `lang_score`       | **FastText** 🌐 language confidence (original, pre-remapping)       | used directly; default 0.5 when unavailable                             |                                                                                                                                                                                                                                                                    |
| `gibberish_ratio`  | fraction of vowel-less or vowel-extreme words (word length ≥ 4)     | inverted: `1 − ratio`                                                   | Words ≥ 60% digits/separators are excluded from gibberish detection.                                                                                                                                                                                               |
| `fused_ratio`      | fraction of suspected merged tokens                                 | inverted: `1 − ratio`                                                   |                                                                                                                                                                                                                                                                    |

</details>

**Dynamic adjustments inside `compute_quality_score()` formula:**

These three conditional modifications are applied during scoring. Unlike the categorisation overrides described in the
next section, they do not change a line's category directly — they adjust intermediate numerical values inside the
score formula before the final weighted sum is computed.

**1. Garbage Penalty Guard (short clean strings)**

*Trigger:* `char_count ≤ 12` **and** `word_weird == 0.0`

*What happens:* `active_garbage_wt` is **halved** from `QS_WEIGHT_GARBAGE` (default 0.20) to 0.10. A compensating
constant of the same amount is added back to `base_score` so the total effective weight sum is unchanged and the
maximum possible score remains 1.0.

*Why:* Short archival label strings — `Lokalita:`, `Osada:`, `Okres:`, `Datum:` — contain a colon or other
structural punctuation that is counted as "garbage" by the garbage-density measure. Without this guard, a
5-character label with one colon would have a garbage density of ~0.20, which under the full 0.20 weight would
already push the line down considerably. Since the line is short and completely structurally clean (no weirdness),
the reduced weight prevents the label from being unfairly penalised.

**2. Conditional Rotation Penalty Gate**

*Trigger:* `rot_ratio ≥ ROT_RATIO_INVERTED_MIN` (default 0.55) **must be true first**, then one of:
  - Arm A: `word_weird ≥ WEIRD_RATIO_INVERTED_MIN` (default 0.35) → strong penalty
  - Arm B: `perplexity ≥ PPL_INVERTED_MIN` 📉 (default 200.0) **and** `word_weird > 0.0` → moderate penalty

*What happens:*
  - Arm A: `rot_penalty = (rot_ratio × word_weird) × 2.0`
  - Arm B: `rot_penalty = 0.40 × min(word_weird / WEIRD_RATIO_INVERTED_MIN, 1.0)`
  - In both arms: if `lang_score ≥ 0.90`, `rot_penalty` is **halved** regardless of its magnitude
  - If neither arm fires: `rot_penalty = 0.0`, regardless of how high `rot_ratio` is

*Why:* Inverted (upside-down) scans produce text where many glyphs are recognised as structurally rotatable
characters (`p` ↔ `d`, `b` ↔ `q`, `n` ↔ `u`, etc.), producing a high `rot_ratio`. However, perfectly normal Czech 🇨🇿
sentences also contain many of these letters naturally — the word *podrobný* alone contains `p`, `d`, `b`. A high
`rot_ratio` alone therefore cannot be used as a penalty signal. The gate requires an independent second signal
(high weirdness or high **perplexity** 📉) to confirm the content is genuinely corrupted, not merely Czech 🇨🇿. The
`lang_score ≥ 0.90` halving adds a third confirmation: if **FastText** 🌐 is highly confident the line is a known
language 🌐, the penalty is softened further, protecting readable Czech 🇨🇿 prose from aggressive downgrading.

**3. Short **Perplexity** 📉 Cap (`SHORT_PPL_CAP`)**

*Trigger:* `word_count ≤ 2` **and** raw LM **perplexity** 📉 `> SHORT_PPL_CAP` (default **850.0** for Qwen2.5-0.5B 🤖,
**2500.0** for distilgpt2 🤖)

*Applied:* in `langID_classify.py`, **before** `compute_quality_score()` is called. The **perplexity** 📉 value
*passed to scoring* is clamped to `SHORT_PPL_CAP`. **The stored `perplex` column in the output **CSV** 📊 is not
changed** — it always reflects the raw model output.

*Why:* Language models assign **perplexity** 📉 by predicting each token given all preceding tokens. With only 1–2
words, there is almost no context available, so the model makes a near-random guess and assigns an extremely high 
**perplexity** 📉 even to perfectly valid words. For example, the single-word line `hrad` (Czech 🇨🇿 for "castle") scores
850 **perplexity** 📉 from Qwen2.5-0.5B 🤖 because the model has seen no preceding text — yet it is a completely valid
Czech 🇨🇿 word. Without this cap, every single-word or two-word line (form-field labels, inventory tags, section
headings) would receive a near-zero **perplexity** 📉 component in its **quality score** 📈 and risk being routed to `Trash`.

**Normalisation ceiling parameters:**

* `PERPLEXITY_THRESHOLD_MAX` (default **1000.0** for Qwen2.5-0.5B): maps raw **perplexity** 📉 into [0, 1] — assigning a
**perplexity** 📉 component of 0 to values at or above the threshold (worst) and 1 to 0 (best). Lowering this value
penalises `Noisy` lines more aggressively; raising it widens the scoring range.
* `QS_LENGTH_MAX` (default **100.0**): sets the character-count ceiling for the length reward, granting the full
`QS_WEIGHT_LENGTH` bonus to lines at or above this length.

> [!NOTE]
> **Perplexity** 📉 contributes only one weighted component (15%) of the **quality score** 📈. Although `Qwen2.5-0.5B` is
> multilingual and handles **Czech 🇨🇿**, **German 🇩🇪**, and **English 🇬🇧** natively (unlike the **English-only 🇬🇧** `distilgpt2` it
> replaced), it is still intentionally diluted by the nine other signals rather than used as a standalone
> threshold. This keeps the score robust against edge cases where even a strong model assigns unexpectedly
> high **perplexity** 📉 to valid but atypical text (e.g., highly abbreviated archival labels or form-field lines).

---

* **Garbage Penalty Guard:** If a text line is short (`≤ 12` characters) and completely clean structurally
(`word_weird == 0.0`), the garbage weight (`active_garbage_wt`) is reduced by `50%`. This prevents over-penalising short
but perfectly legible archival tags (like `Lokalita:`).
* **Conditional Rotation Penalty Gate:** A rotation penalty (`rot_penalty`) is calculated for strings with a high
`rot_ratio` (`>0.55`), *but only if the line also exhibits internal **OCR** 🔍 weirdness* (`word_weird > 0.0`). This safely
prevents perfectly readable short Czech sentences with naturally high occurrences of ambiguous characters (like
`p`, `d`, `b`, `q`) from being unfairly penalized. Furthermore, this penalty is reduced by `50%` if the **LM** 🤖 is highly
confident the text is legible (`lang_score >= 0.90`).

`valid_word_ratio` counts tokens that are alphabetically dominant (≥ 70% alpha chars), contain no disallowed
internal symbols, and are **not** a leading all-caps **OCR** 🔍 prefix followed by lowercase letters. This last guard
prevents garbled tokens such as `AAMMNAbSSOAO`, `XAterenta`, or `SeverW` — where the **OCR** 🔍 engine has prepended
a run of spurious uppercase characters to a recognisable word fragment — from inflating the valid-word signal
and pushing the overall **quality score** 📈 into the Clear band.

The parameters that normalize unbounded signals before weighting in the **quality score** 📈 formula are
`PERPLEXITY_THRESHOLD_MAX` (default **1000.0**), which caps raw **perplexity** 📉 to map it into [0, 1] assigning 0 to values at or above the threshold
(worst) and 1 to 0 (best), calibrated for **Qwen2.5-0.5B** 🤖 on corrupted **OCR** 🔍 text to penalize noisy lines more aggressively
when lowered or widen the scoring range when raised; and `QS_LENGTH_MAX` (default **100.0**), which sets the character-count
ceiling for rewarding longer lines, granting the full `QS_WEIGHT_LENGTH` bonus to **lines at or above this length**.

> [!NOTE]
> **Perplexity** 📉 contributes only one weighted component of the **quality score** 📈. Although **Qwen2.5-0.5B** 🤖 is
> multilingual and handles **Czech** 🇨🇿, **German** 🇩🇪, and **English** 🇬🇧 natively (unlike the **English-only** 🇬🇧 `distilgpt2` it
> replaced), it is still intentionally diluted by the nine other signals rather than used as a standalone
> threshold. This keeps the score robust against edge cases where even a strong model assigns unexpectedly
> high **perplexity** 📉 to valid but atypical text (e.g., highly abbreviated archival labels or form-field lines).

##### Categorisation Logic

`categorize_line()` classifies each line using **immediate overrides** checked in priority order, followed by
**quality score 📈 threshold routing**, with one additional **promotion override** inside the `Noisy` band.
The function also aligns the stored `quality_score` value to be consistent with the assigned category band, so that
downstream analytics can rely on the score as a monotone proxy for category rank without re-running thresholds.

<details>
    <summary><strong>1, 2, and 3 Immediate Overrides 👀</strong></summary>

Checked in order - the first match wins and skips all remaining checks including thresholds:

| # | Condition                                                       | Result  | Rationale                                                                                                                                                                                                                                                                                                                                                                          |
|---|-----------------------------------------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | `word_count == 0` or line contains only whitespace              | `Empty` | Structural blank — no content to evaluate. Assigned before any scoring.                                                                                                                                                                                                                                                                                                            |
| 2 | All alphabetic words are uppercase **and** `vowel_ratio < 0.10` | `Trash` | Definitively unreadable: an all-caps block with almost no vowels is a visual scramble (e.g., a block of symbols the **OCR** 🔍 engine interpreted as capital letters). This cannot be a legitimate section header — real all-caps Czech 🇨🇿 section headers contain vowels (`SEZNAM NÁLEZŮ`).                                                                                     |
| 3 | `perplexity < 50.0` 📉 **and** `word_count ≥ 3`                 | `Clear` | The language model is near-certain about the text (near-zero NLL, below any reasonable noise threshold). Lines with **perplexity** 📉 this low are almost exclusively fluent natural-language sentences. The `word_count ≥ 3` guard prevents single tokens like proper nouns from being fast-tracked to `Clear` purely because they happen to follow a low-**perplexity** 📉 path. |

> [!NOTE]
> Inverted/180°-rotated scans are handled **outside** the per-line categoriser: their fingerprint (high
> rotatable-character density combined with internal weirdness or LM uncertainty) is encoded as a `rot_penalty`
> subtracted directly inside `compute_quality_score`, depressing the QS so the standard thresholds route the line to
> `Trash`. A second, page-level pass (see [Post-Processing Smoothing](#post-processing-smoothing) below) catches 
> contiguous runs of rotated lines that may have individually escaped the per-line penalty.

</details>

**Quality score 📈 threshold routing** (applied to all lines not caught by an override above):

```text
quality_score < CATEG_TRASH_SCORE_MAX  (def: 0.50)  →  Trash
quality_score < CATEG_NOISY_SCORE_MAX  (def: 0.90)  →  Noisy  (unless Override 4 fires)
otherwise                                            →  Clear
```

**Override 4 — Near-Boundary Clean Prose Promotion** (fires inside the `Noisy` band, before `Noisy` is returned):

If all four of the following conditions hold simultaneously, the line is promoted from `Noisy` → `Clear`:

<details>
    <summary><strong>Override 4 conditions explained 👀</strong></summary>

| Condition                               | Parameter               | Default | What it ensures                                                                                                                                                                                                   |
|-----------------------------------------|-------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `quality_score ≥ CLEAN_PROSE_MIN_SCORE` | `CLEAN_PROSE_MIN_SCORE` | 0.65    | The line is in the upper part of the `Noisy` band — it scored reasonably well but was held back by one or two minor signals.                                                                                      |
| `word_count ≥ CLEAN_PROSE_WC_MIN`       | `CLEAN_PROSE_WC_MIN`    | 4       | The line has enough tokens for **perplexity** 📉 to be meaningful. With 1–3 words, the LM has almost no context and its **perplexity** 📉 score is unreliable; the promotion is only trusted for longer text.     |
| `word_weird < CLEAN_PROSE_WEIRD_MAX`    | `CLEAN_PROSE_WEIRD_MAX` | 0.08    | No token in the line shows significant structural corruption. Even a single noticeably strange token (e.g., a letter–digit–letter fusion or a mid-word uppercase) disqualifies the line from promotion.           |
| `perplexity < CLEAN_PROSE_PPL_MAX`      | `CLEAN_PROSE_PPL_MAX`   | 400.0   | The language model considers the line reasonably likely. A very high **perplexity** 📉 even on a structurally clean line can indicate foreign or domain-specific vocabulary that is genuinely `Noisy` to process. |

</details>

**Rationale for Override 4:** Readable Czech 🇨🇿 archaeological prose — field measurements, dig site descriptions,
formal letter phrases — can score just below `CATEG_NOISY_SCORE_MAX` (0.90) for two systematic reasons: (a) the
 **perplexity** 📉 of short isolated sentence fragments is inherently elevated even when the words are perfectly correct, and
(b) minor but common **OCR** 🔍 artefacts such as period-abbreviations (`Obr.`, `Viz`) and occasionally merged words slightly
depress the valid-word-ratio component. When the line is long enough, structurally clean, and the **LM** 🤖 is reasonably
confident, these small penalties should not prevent the line from reaching `Clear`.

<details>
    <summary><strong>Quality score 📈 alignment after categorisation 👀</strong></summary>

After the category is determined, the stored `quality_score` 📈 is clamped to the range corresponding to the assigned
band. This ensures that the **CSV** 📊 value is always internally consistent with the `categ` label:

* `Trash` → score clamped to `min(qs, CATEG_TRASH_SCORE_MAX − ε)` — always below 0.50
* `Noisy` → score clamped to `[CATEG_TRASH_SCORE_MAX, CATEG_NOISY_SCORE_MAX − ε]` — always in [0.50, 0.90)
* `Clear` → score clamped to `max(qs, CATEG_NOISY_SCORE_MAX)` — always ≥ 0.90

Lines promoted by Override 4 have their raw score (which was somewhere in [0.65, 0.90)) raised to 0.90 in the **CSV** 📊.
Lines whose Override 2 or Override 3 fired receive a score consistent with the override result regardless of what
the formula computed.

</details>

---

##### Post-Processing Smoothing

After all lines in a document are classified and written to **CSV** 📊, a final data-smoothing pass is applied before the file
is finalized. This pass corrects categorisation anomalies that only become visible at the document or page
level — patterns that per-line scoring cannot detect because it evaluates each line in isolation.


<details>
    <summary><strong>1. Header/Footer Deduplication 👀</strong></summary>

*What it does:* All occurrences of the exact same text string across a document are identified. If the same string
has been assigned to different categories on different pages (e.g., `Obr. 1. SKUHROV NAD BĚLOU` is `Clear` on page 3
but `Noisy` on page 4 due to slightly different surrounding context affecting the LM), all occurrences are
harmonised to the **statistical mode** — the category assigned most frequently to that string across the document.

*Why:* Repeating strings are boilerplate — page headers, footers, running titles, standard form labels. The
same physical text should receive the same label throughout a document, and the majority vote across its
occurrences is the most reliable estimate of the correct category.

</details>

<details>
    <summary><strong>2. Context Smoothing (Rolling 5-line Window) 👀</strong></summary>

*What it does:* Scans the document line-by-line. If a `Noisy` ⚠️ line is surrounded by `Trash` 🗑 on both sides in a
5-line window (positions −2 and −1 are `Trash` 🗑 **and** positions +1 and +2 are `Trash` 🗑), **and** the `Noisy` ⚠️ line's
quality score is below `CATEG_TRASH_SCORE_MAX + 0.15` (default: **0.65**), it is downgraded to `Trash` 🗑.

*Why:* A single `Noisy` ⚠️ island embedded in four consecutive `Trash` 🗑️ lines is almost certainly corrupted text that
narrowly escaped the `Trash` 🗑 threshold. The rolling window catches these borderline cases. The score guard of 0.65
ensures that only near-boundary `Noisy` ⚠️ lines are affected — a `Noisy` ⚠️ line with a **quality score** 📈 of 0.80 
is left alone even in a `Trash` 🗑 neighbourhood, because its quality is genuinely different from the surrounding lines.

</details>

<details>
    <summary><strong>3. Page-level Inverted-Scan Sweep 👀</strong></summary>

*What it does:* After the rolling-window pass, each page is scanned independently for contiguous runs of 4 or more
non-`Empty`🫙/non-`Non-text`🔣 lines. If a run meets **either** of the two detection arms below, the entire run is
downgraded to `Trash`🗑:

* **Diacritic-absence arm:** all lines in the run lack Czech 🇨🇿 diacritics (`á č ď é ě í ň ó ř š ť ů ú ý ž` and
  uppercase equivalents) **and** all lines have a **FastText** 🌐 confidence below `LANG_SCORE_ROUGH` (default **0.45**).
  Together these indicate that the **OCR** 🔍 produced no Czech-identifiable content and the language model cannot
  confidently assign any language 🌐 — a strong signal of a page-level scan fault.

* **Rotation arm:** all lines in the run have `rot_ratio ≥ ROT_RATIO_INVERTED_MIN` (default **0.55**) **and**
  `perplexity ≥ PPL_INVERTED_MIN` 📉 (default **200.0**). A high concentration of rotationally ambiguous characters
  *combined* with high LM **perplexity** 📉 confirms the **OCR** 🔍 was processing visually flipped content.

*Why two arms?* Inverted-scan pages sometimes produce partial Czech 🇨🇿 diacritics: the **OCR** 🔍 engine recognises some
upside-down glyphs as plausible Latin characters and occasionally matches diacritical forms. The diacritic-absence
arm alone would miss these pages. The rotation arm catches them independently by using the character-shape signal
and LM uncertainty together, without requiring the absence of diacritics.

</details>

---

<details>
    <summary><strong>SUMMARY of All Factors Affecting Quality Score (click to expand 👀)</strong></summary>


The table below consolidates every factor that influences `quality_score` 📈 or the final category assignment, including
where each factor is controlled and any known edge cases.

| Factor                                     | Where applied                                                    | Config key(s)                                                                                 | Edge cases / exceptions                                                                                                                                                                                               |
|--------------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Valid word ratio                           | `compute_quality_score` (25% weight)                             | `QS_WEIGHT_VALID_WORD`                                                                        | All-caps **OCR** 🔍 prefix guard: tokens like `AAMMNAbSSOAO` are excluded from valid-word count even though they are alphabetically dominant.                                                                         |
| Symbol ratio                               | `compute_quality_score` (13% weight)                             | `QS_WEIGHT_SYMBOL`                                                                            | Characters in `ALLOWED_INTERNAL` are not counted as symbols; edge punctuation stripped before inspection.                                                                                                             |
| Word weirdness ratio                       | `compute_quality_score` (13% weight)                             | `QS_WEIGHT_WEIRD`                                                                             | Isolated single letters score 0.85 (**OCR** 🔍 spaced-out noise); isolated digits/measurements score 0.25 (tolerable). All-caps words and specific capitalized sequences excluded from mid-uppercase detection.       |
| Perplexity 📉 (LM)                         | `compute_quality_score` (15% weight)                             | `QS_WEIGHT_PERPLEXITY`, `PERPLEXITY_THRESHOLD_MAX`                                            | Short-text **perplexity** 📉 is capped at `SHORT_PPL_CAP` before scoring to prevent penalising legitimate 1–2 word inputs. Override 3 (`ppl < 50`) bypasses thresholds entirely for highly confident predictions.     |
| Text length                                | `compute_quality_score` (5% weight)                              | `QS_WEIGHT_LENGTH`, `QS_LENGTH_MAX`                                                           | Full reward for lines ≥ 100 chars; no minimum penalty for short lines.                                                                                                                                                |
| Garbage density                            | `compute_quality_score` (20% weight)                             | `QS_WEIGHT_GARBAGE`, `CATEG_GARBAGE_DENSITY_HIGH`                                             | **Halved** to 10% for lines ≤ 12 characters with zero weirdness (short-string guard) to protect clean labels with colons.                                                                                             |
| Vowel quality                              | `compute_quality_score` (7% weight)                              | `QS_WEIGHT_VOWEL`                                                                             | Linear ramp: full score in [0.20, 0.75] vowel ratio, ramps to 0.0 outside that range.                                                                                                                                 |
| Language 🌐 confidence                     | `compute_quality_score` (5% weight)                              | `QS_WEIGHT_LANG`                                                                              | Uses the **original** (pre-remapping) **FastText** 🌐 score; defaults to 0.5 when unavailable.                                                                                                                        |
| Gibberish ratio                            | `compute_quality_score` (4% weight)                              | `QS_WEIGHT_GIBBERISH`                                                                         | Words ≥ 60% digits/separators excluded. Detection only on words ≥ 4 characters.                                                                                                                                       |
| Fused word ratio                           | `compute_quality_score` (3% weight)                              | `QS_WEIGHT_FUSED`                                                                             | Triggers on tokens > 14 chars, consonant runs of 5+, or vowel runs of 4+.                                                                                                                                             |
| Rotation penalty                           | `compute_quality_score` (dynamic, subtracted post-normalisation) | `ROT_RATIO_INVERTED_MIN`, `WEIRD_RATIO_INVERTED_MIN`, `PPL_INVERTED_MIN`                      | **Not applied** unless `rot_ratio ≥ 0.55` **and** (`word_weird ≥ 0.35` or `ppl ≥ 200`). Penalty halved when `lang_score ≥ 0.90`.                                                                                      |
| All-caps + low vowel override (Override 2) | `categorize_line`                                                | none (hardcoded)                                                                              | Fires only if **all** alphabetic words are uppercase **and** `vowel_ratio < 0.10`. A legitimate all-caps header with normal vowel density is **not** affected.                                                        |
| High LM confidence override (Override 3)   | `categorize_line`                                                | none (configurable only via `PERPLEXITY_THRESHOLD_MAX`)                                       | Requires both `ppl < 50` **and** `word_count ≥ 3`; single-word lines are never fast-tracked to `Clear` by this override.                                                                                              |
| Near-boundary promotion (Override 4)       | `categorize_line`                                                | `CLEAN_PROSE_MIN_SCORE`, `CLEAN_PROSE_WC_MIN`, `CLEAN_PROSE_WEIRD_MAX`, `CLEAN_PROSE_PPL_MAX` | Only fires within the `Noisy` band (score ∈ [0.65, 0.90)). Requires all four conditions simultaneously.                                                                                                               |
| Short **perplexity** 📉 cap                | `langID_classify.py` (before scoring)                            | `SHORT_PPL_CAP`                                                                               | Applied only to lines with ≤ 2 words. Does not change the stored `perplex` column; affects only the value passed to quality scoring.                                                                                  |
| Language 🌐 remapping                      | `langID_classify.py` (before scoring)                            | `EXPECTED_LANGS`, `TRUSTED_FOREIGN_LANGS`, `LANG_SCORE_CLEAR`                                 | If the predicted language 🌐 is not in either list, code is remapped to first entry of `EXPECTED_LANGS` and score floored to `LANG_SCORE_CLEAR`. The original score is still used as the `QS_WEIGHT_LANG` input.      |
| Context smoothing (rolling window)         | Post-processing in `langID_classify.py`                          | `CATEG_TRASH_SCORE_MAX`                                                                       | `Noisy` line must be surrounded by 2 `Trash` lines on **each** side (4 total); score must be < `Trash` threshold + 0.15.                                                                                              |
| Page-level inverted-scan sweep             | Post-processing in `langID_classify.py`                          | `ROT_RATIO_INVERTED_MIN`, `PPL_INVERTED_MIN`, `LANG_SCORE_ROUGH`                              | Requires a run of **at least 4** consecutive non-`Empty`/`Non-text` lines all meeting the condition. Two independent detection arms (diacritic-absence + low confidence, or rotation ratio + high **perplexity** 📉). |
| Header/footer deduplication                | Post-processing in `langID_classify.py`                          | none                                                                                          | Based on **exact text match** across the whole document; harmonises to modal category.                                                                                                                                |

</details>

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
* `avg_symbol` — mean strange-symbol word count per line
* `avg_vowel_ratio` — mean vowel-to-alphabetic-character ratio per line
* `avg_rot_ratio` — mean rotatable character ratio per line
* `ch_ratio` — mean fraction of lines flagged as all-caps headers (`caps_header = True`)

**Language profile:**

* `main_lang` — the statistical mode (most frequent) language 🌐 predicted for the page 

> [!NOTE]
> `avg_*` columns and `main_lang` will be `NaN` / `None` for pages whose only lines are
> `Empty` or `Non-text` (i.e., pages with no scoreable text content).

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
* **Configuration ⚙️:** Stores a complete snapshot of the runtime configuration ⚙️, including script names, input/output
paths, and specific model choices.
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

| Component                | License         | Counted     | Used by                                                        |
|--------------------------|-----------------|-------------|----------------------------------------------------------------|
| **alto-tools** 🔧 [^1]   | Apache-2.0      | always      | page split, statistics, alto-tools text extraction             |
| **FastText** 🌐 [^2]     | CC BY-NC 4.0    | always      | language identification (`langID_classify.py`)                 |
| **Qwen2.5-0.5B** 🤖 [^6] | Apache-2.0      | conditional | **perplexity** 📉 scoring (default, `langID_classify.py`)      |
| **distilgpt2** 🤖        | Apache-2.0      | conditional | **perplexity** 📉 scoring (English-only alternative)           |
| **LayoutLMv3** 📐 [^9]   | CC BY-NC-SA 4.0 | conditional | LayoutReader text extraction (`extract_LytRdr_ALTO_2_TXT.py`)  |
| **GLM-4v-9b** 🤖 [^10]   | glm-4           | conditional | generative **OCR** 🔍 extraction (`extract_LLM_ALTO_2_TXT.py`) |

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

**For support write to:** lutsai.k@gmail.com — responsible for this GitHub repository [^8] 🔗

- **Developed by** UFAL [^7] 👥
- **Funded by** ATRIUM [^4] 💰
- **Shared by** ATRIUM [^4] & UFAL [^7] 🔗
- **Models used**:
  - **FastText** 🌐 [^2] for language identification
  - **Qwen2.5-0.5B** 🤖 [^6] for **perplexity** 📉 scoring
  - **GLM-4v-9b** 🤖 [^10] for generative **OCR** 🔍 (LLM-based method)
  - **LayoutLMv3** 📐 [^9] for layout-aware text extraction

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