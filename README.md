# 📦 ALTO XML Files Postprocessing Pipeline

This project provides a complete workflow for processing ALTO XML files. It takes raw ALTO
XMLs and transforms them into structured statistics tables, performs text classification,
and filters low-quality OCR results.

The core of the quality filtering relies on language identification and perplexity measures
to identify and categorize noisy or unreliable OCR output.

---

## 📖 Table of Contents

- [⚙️ Setup](#️-setup)
- [🛤️ Workflow Stages](#️-workflow-stages)
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
and perplexity scores from [distilGPT2](https://huggingface.co/distilbert/distilgpt2) 😊 to detect noise [^2] [^6].

More post-processing of TXT files can be found in the [GitHub repository](https://github.com/ufal/atrium-nlp-enrich)
of the ATRIUM project, which covers NLP enrichment using Nametag for NER and UDPipe for CONLL-U files with lemmas & POS tags [^5].

As the script processes, it assigns each line one of five categories 🪧:

* ✅ **Clear** — Passes all structural checks; low cumulative penalty score.
* ⚠️ **Noisy** — Partially degraded: moderate cumulative penalty from isolated symbol issues, fused tokens, mid-word uppercase, or elevated perplexity on longer lines.
* 🗑️ **Trash** — Severely corrupted: high garbage density, extreme perplexity combined with weirdness, or a cumulative penalty score above the Trash threshold.
* 🔣 **Non-text** — Filtered by the CPU pre-filter: line is too short, has too few unique symbols, contains fewer than 30% alphabetic characters, or consists mostly of digits and punctuation.
* 🫙 **Empty** — Line contains only whitespace.

> [!NOTE]
> This script generates two primary output directories:
> `DOC_LINE_LANG_CLASS/` and `DOC_LINE_STATS/`, while the
> raw text files (primary input) are stored in `../PAGE_TXT/` generated from `../PAGE_ALTO/`.

All input/output paths and tunable parameters are configured in [config_langID.txt](config_langID.txt) 📎.
Parameters are organized into **three sections**: `[CLASSIFY]`, `[AGGREGATE]`, and `[TEXT_UTILS]`.

---

#### 4.1 Classify Lines (GPU Bound) 🚀

This script reads the extracted text files, batches lines together 📦, and runs the FastText [^2]
and DistilGPT2 [^6] models. It uses a **CPU/GPU split architecture**:

- A single dedicated **GPU worker** holds the only DistilGPT2 instance and processes perplexity batches to prevent VRAM OOM errors.
- Multiple **CPU workers** (up to `WORKERS_MAX`, default 32) read files, run FastText and structural detectors, and submit text batches to the GPU worker via a shared queue. CPU workers poll the result dictionary while the GPU processes, running language identification concurrently.

    python3 langID_classify.py

* **Input 1 📥:** `../PAGE_TXT/` from Step 3
* **Input 2 📥:** `output.csv` from Step 2
* **Output 📤:** `DOC_LINE_LANG_CLASS/` containing per-document CSVs (e.g., [DOC_LINE_LANG_CLASS](data_samples/DOC_LINE_LANG_CLASS) 📁)

> [!TIP]
> This script is resume-capable. If interrupted, run it again and already-present output files will be skipped.

`<doc_name>.csv`: Detailed classification results for every single line within a document, with columns:

* `file` — document identifier 🆔
* `page_num` — page number 📄
* `line_num` — line number, starts from 1 for each page 🔢
* `text` — original text of the line 📝
* `split_ws` — hyphenated word prefix at the end of the line (split word start)
* `split_we` — hyphenated word suffix at the start of the line (split word end)
* `lang` — predicted ISO language code from the FastText model ([full list](https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200)) 🌐
* `lang_score` — FastText confidence score for the predicted language 🎯
* `perplex` — DistilGPT2 perplexity score of the line 📉
* `word_count` — number of whitespace-delimited tokens in the line
* `char_count` — total character count of the line
* `garbage_density` — ratio of non-alphanumeric, non-standard-punctuation characters to total line length
* `symbol` — count of words containing disallowed internal symbols (see detectors below)
* `upper` — count of words with unexpected mid-word uppercase letters
* `repeated` — count of words where a non-standard character makes up ≥ 40% of the word
* `ldl_fuses` — count of words with a letter–digit–letter sandwich (e.g., `w0rd`)
* `gibberish` — count of words flagged as gibberish (all-caps, no vowels, or extreme vowel ratio)
* `word_weird` — mean per-word weirdness score in [0, 1]; combines strange-symbol, repeated-symbol, LDL-fusion, and mid-uppercase signals weighted per token (0 = fully clean)
* `quality_score` — composite quality score in [0, 1] based on valid-word ratio, symbol ratio, perplexity, and text length; higher = cleaner
* `categ` — assigned category: **Clear** ✅, **Noisy** ⚠️, **Trash** 🗑️, **Non-text** 🔣, or **Empty** 🫙

##### CPU Pre-filter

Before any GPU or model inference, `pre_filter_line()` applies a fast CPU-side check and assigns `Empty` or `Non-text` directly, bypassing the ML pipeline entirely:

* Line is blank → **Empty**
* Fewer than 4 characters, or fewer than 3 unique non-whitespace symbols → **Non-text**
* Letter ratio below 30% of total characters → **Non-text**
* Matches the all-digits/symbols regex pattern → **Non-text**
* Otherwise → forwarded for ML classification as **Process**

##### Structural Detectors

Lines that pass the pre-filter are analysed by four structural detectors defined in `text_util_langID.py`:

| Detector | What it counts |
|---|---|
| `detect_strange_symbols` | Words containing any character that is not alphanumeric and not in the allowed set `{ . - , + ( ) " ' / _ — – : % }`. Edge punctuation is stripped before inspection. |
| `detect_letter_digit_letter` | Words with a **letter–digit–letter sandwich** — the fingerprint of OCR digit insertions mid-word (e.g., `vyt1ačená`, `nalez2í`). Legitimate patterns like `90,9g`, `80-90cm`, `26.IX.1957` do not trigger. |
| `detect_mid_uppercase` | Words with unexpected uppercase mid-word (`dalSÍ`, `obkLADem`) or an uppercase run at the start followed by lowercase (`XXWžkumu`). All-caps words and titles (`PhDr`, `MUDr`) are excluded. |
| `detect_repeated_chars` | Words where a single non-standard character makes up ≥ 40% of the word (e.g., OCR stutter like `bxxxoxx`). |
| `detect_gibberish_words` | Words of length ≥ 7 that are all-uppercase, contain no vowels, or have a vowel ratio below 15% or above 80%. |

##### Categorisation Logic (Cumulative Penalty System)

`categorize_line()` in `text_util_langID.py` uses a cumulative floating-point penalty score rather than a fixed decision tree. The full logic is evaluated as follows:

**Immediate Trash overrides** (checked first, before penalty accumulation):

* Garbage density > 0.35, or garbage density > 0.20 on lines of ≤ 3 words → **Trash**
* Perplexity > 500 **and** a structural weirdness ratio (`word_weird`) > 0.4 simultaneously → **Trash** *(hard override to catch severe, high-confidence garbage)*

**Penalty accumulation** (for lines that pass the overrides):

| Signal | Penalty added |
|---|---|
| Each word with a strange symbol (`sym_count`) | `sym_count × 0.4` |
| Two or more strange-symbol words | additional `+0.5` |
| Each LDL-fused token | `× 0.3` |
| Each mid-word uppercase word | `× 0.2` |
| Each word with repeated non-standard char | `× 0.4` |
| Each gibberish word | `× 0.5` |

**Perplexity penalty** (skipped for short phrases that are structurally clean):

A line with fewer than 5 words, whose language is in the `EXPECTED_LANGS` allowlist and whose structural penalty is zero, is treated as a "forgiven short phrase" and perplexity thresholds are not applied. For all other lines:

* Perplexity > `PERPLEXITY_THRESHOLD_MIN` (default 1500, scaled to `× 1.5` for lines < 5 words) → `+0.5`
* Perplexity > `PERPLEXITY_THRESHOLD_MAX` (default 5000) → additional `+1.0`

> [!NOTE]
> Perplexity is intentionally **not used** as a Trash signal in isolation. `distilgpt2` is an English model and
> assigns very high perplexity to legitimate short Czech strings (place names, postal codes, form-field labels),
> making it unreliable as a Trash indicator. It is applied only as an additive penalty with structural context.

**Language confidence penalty:**

* Predicted language is **not** in `EXPECTED_LANGS` and confidence < 0.60 → `+0.8`
* Predicted language **is** in `EXPECTED_LANGS` but confidence < 0.30 → `+0.5`

**Final classification** via normalized penalty:

```
normalized_penalty = total_penalties / max(1.0, word_count / 5.0)

normalized_penalty ≥ 1.2  →  Trash
normalized_penalty ≥ 0.3  →  Noisy
otherwise                 →  Clear
```

##### Post-Processing Smoothing

After all lines in a document are classified and written to CSV, a final data-smoothing pass is applied before the file is finalized to prevent unnatural categorization anomalies:

1. **Header/Footer Deduplication** — Resolves edge-case flip-flopping. If the exact same text string appears multiple times across a document, all instances are harmonized to share the statistical mode (most frequent) category assigned to that string.
2. **Context Smoothing (Rolling Window)** — Applies a 3-line rolling window. If a **Noisy** line is sandwiched between two consecutive **Trash** lines, it is automatically downgraded to **Trash** to prevent isolated "noisy" categorizations in otherwise heavily corrupted regions.

Example of per-document CSV files: [DOC_LINE_LANG_CLASS](data_samples/DOC_LINE_LANG_CLASS) 📁.
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
* **Output 1 📤:** `ARUP_short_page_stats.csv` — global page-level summary across all documents
* **Output 2 📤:** `../DOC_LINE_STAT/` — per-document CSVs with the same schema

For each page, the aggregation computes:

**Category counts** (from all lines regardless of category):

* `Clear`, `Noisy`, `Trash`, `Non-text`, `Empty` — integer count of lines in each category

**Totals** (summed over lines classified as Clear, Noisy, or Trash only — Empty and Non-text excluded):

* `total_word_count` — total number of words across scoreable lines
* `total_char_count` — total number of characters across scoreable lines

**Averages** (mean over the same Clear/Noisy/Trash lines):

* `avg_garbage_density` — mean garbage density ratio
* `avg_lang_score` — mean FastText confidence score
* `avg_perplex` — mean DistilGPT2 perplexity score
* `avg_symbol` — mean strange-symbol word count
* `avg_upper` — mean mid-uppercase word count
* `avg_repeated` — mean repeated-char word count
* `avg_ldl_fuses` — mean LDL-fusion word count
* `avg_gibberish` — mean gibberish word count
* `avg_word_weird` — mean per-word weirdness ratio in [0, 1]; 0 = fully clean, lower is better 📉
* `avg_quality_score` — mean composite quality score in [0, 1]; higher = cleaner OCR output 📈

**Language profile:**

* `main_lang` — the statistical mode (most frequent) language predicted for the page, excluding lines where FastText returned `N/A` or `unknown`

> [!NOTE]
> `avg_*` columns and `main_lang` will be `NaN` / `unknown` for pages whose only lines are
> Empty or Non-text (i.e., pages with no scoreable text content).

All numeric averages are rounded to 4 decimal places; totals are stored as integers.

- *Example*: [ARUP_short_page_stats.csv](ARUP_short_page_stats.csv) 📎

Example of per-document aggregate CSV files: [DOC_LINE_STAT](data_samples/DOC_LINE_STAT) 📁.
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
* **Configuration ⚙️:** Stores a complete snapshot of the runtime configuration, including script names, input/output paths, and specific model choices.
* **Timing ⏱️:** Records precise UTC start times, end times, and the total duration of the run in seconds.
* **Statistics 📊:** Tracks the total number of input files, successfully processed documents, and computes performance throughput (e.g., output files generated per minute).
* **Error Tracking 🐛:** Maintains a `skipped_files_detail` list that logs the exact filename and specific error reason if a file fails to process.

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
  - DistilGPT2 [^6] for perplexity scoring
  - GLM-4v-9b [^10] for generative OCR (LLM-based method)
  - LayoutLMv3 [^9] for layout-aware text extraction

**©️ 2026 UFAL & ATRIUM**

[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^3]: https://github.com/ufal/ker
[^4]: https://atrium-research.eu/
[^5]: https://github.com/ufal/atrium-nlp-enrich
[^6]: https://huggingface.co/distilbert/distilgpt2
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b