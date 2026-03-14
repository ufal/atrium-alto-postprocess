# 📦 ALTO XML Files Postprocessing Pipeline

This project provides a complete workflow for processing ALTO XML files. It takes raw ALTO 
XMLs and transforms them into structured statistics tables, performs text classification, 
filters low-quality OCR results.

The core of the quality filtering relies on language identification and perplexity measures 
to identify and categorize noisy or unreliable OCR output.

---

## 📖 Table of Contents

- [ ⚙️ Setup](#-setup)
- [🛤️ Workflow Stages](#-workflow-stages)
  - [Step 1: Split Document-Specific ALTOs into Pages ✂️](#-step-1-split-document-specific-altos-into-pages-)
  - [Step 2: Create Page Statistics Table 📈](#-step-2-create-page-statistics-table-)
  - [Step 3: Extract text from ALTO XML ⛏️](#-step-3-extract-text-from-alto-xml-)
    - [LayoutReader method 📐](#1st-choice-layoutreader--method-)
    - [alto-tools method 🧰](#2nd-option-alto-tools--method)
    - [GLM method 🤖](#3rd-alternative-glm--method-llm-based)
  - [Step 4: Classify Page Text Quality \& Language 🗂️](#-step-4-classify-page-text-quality--language-)
    - [4.1 Classify Lines (GPU Bound) 🚀](#41-classify-lines-gpu-bound-)
    - [4.2 Aggregate Statistics (Memory Bound) 🧠](#42-aggregate-statistics-memory-bound-)
- [Acknowledgements 🙏](#acknowledgements-)

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
5. Copy `v3` folder from the `layoutreader` 🔧 repository [^9] to the project directory for the LR-based text extraction method:
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

#### 1st choice: LayoutReader 🔧 method 

> [!CAUTION]
> The model responsible for spatial layout analysis requires a **GPU** to run efficiently.

    python3 extract_LytRdr_ALTO_2_TXT.py

that uses the LayoutReader framework [^9] to extract text and bounding boxes of XML elements (
specifically, `<TextLine>` elements containing `String`s with `CONTENT` attribute), 
process them to reconstruct the reading order of lines (columns-friendly), then handle words split
between two lines (added whole word nearby), and based on the vertical spread of text lines groups
page contents into paragraphs and lines of the output `.txt` file.

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
#### 2nd option: alto-tools 🔧 method

> [!NOTE]
> The method is **CPU**-bound and faster than the LayoutReader method, but the text lines may not be in the correct 
> reading order, as well as the full forms of split words are not included.

    python3 extract_ALTO_2_TXT.py

that uses the `alto-tools` framework [^1] to extract text lines from contents of XML elements.
There is no post-processing of the extracted text, but this method is faster and can be used
to get a quick overview of the raw text content.

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
#### 3rd alternative: GLM 🔧 method (LLM-based)

> [!WARNING]
> The method is **GPU**-bound, slower than the LayoutReader method, and requires `gpuram48G` card.

    python3 extract_LLM_ALTO_2_TXT.py

that uses uses the GLM-4v-9b multimodal large language model [^10] to perform generative OCR directly from page images.

Unlike the previous methods that parse existing ALTO XML text, this script basically uses source page images to generate 
text prompted as `Transcribe all text on this page exactly as it appears`, Trims whitespace and resizes high-resolution 
images to fit model constraints

> [!NOTE]
> This method is significantly slower than parsing XML but often yields higher quality text for complex 
layouts or degraded scans. It specifically patches the transformers configuration to run the GLM-4v architecture.

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

This is a key ⌛ time-consuming step that analyzes the text quality of each page,
line-by-line, counting lines of defined types, to filter out OCR noise 🔇.

It uses the [FastText language identification model](https://huggingface.co/facebook/fasttext-language-identification) 😊 
and perplexity scores from [distilGPT2](https://huggingface.co/distilbert/distilgpt2) 😊 to detect noise [^2] [^6].

More post-processing of TXT files can be found in the [GitHub repository](https://github.com/ufal/atrium-nlp-enrich) 
of ATRIUM project dedicated to based on NLP enrichment of the textual data using Nametag for 
NER and UDPipe for CONLL-U files with lemmas & POS tags [^5].

As the script processes, it aggregates line counts for each page into categories 🪧:

* ✅ **Clear** - High-confidence, low-perplexity, common language.
* ⚠️ **Noisy (Rough)** - Medium or Low-confidence, high-perplexity, or other OCR issues.
* 🗑️ **Trash** - Hard to guess language, very high perplexity, or non-prose.
* 🔣 **Non-text** - Failed heuristic checks (e.g., mostly digits/symbols).
* 🫙 **Empty** - Line contains only whitespace.

> [!NOTE]
> This script generates two primary output directories: 
> `DOC_LINE_LANG_CLASS/` and `DOC_LINE_STATS/`, while the
> raw text files (primary input) are stored in `../PAGE-TXT/`generated from `../PAGE_ALTO`.

All of the input-output files and changeable parameters are available in [config_langID.txt](config_langID.txt) 📎 where
variables are divided into two sections according to the processing stage of Step 4 (classification or aggregation).

#### 4.1 Classify Lines (GPU Bound) 🚀

This script reads the extracted text files, batches lines together 📦, and runs the FastText [^2]
and DistilGPT2 [^6] models on the **GPU**. It logs results immediately to a raw CSV to save memory 💾.

    python3 langID_classify.py

* **Input 1 📥:** `../PAGE_TXT/` from Step 3
* **Input 2 📥:** `output.csv` from Step 2
* **Output 📤:** `DOC_LINE_LANG_CLASS/` containing per-document CSVs (e.g., [DOC_LINE_LANG_CLASS](data_samples/DOC_LINE_LANG_CLASS) 📁) 

> [!TIP]
> This script is resume-capable. If interrupted, run it again, and already present in the output directory files will be skipped.

`<doc_name>.csv`: Detailed classification results for *every single line* within a document, with columns:
* `file` - document identifier 🆔
* `page_num` - page number 📄
* `line_num` - line number, starts from 1 for each line on the ALTO page 🔢
* `text` - original text of the line from ALTO page 📝
* `split_we` - hyphen end (split word ending - first word in line)
* `split_ws` - hyphen start (split word beginning - last word in line)
* `lang` - predicted ISO language code of the line ([list of all possible language labels predicted by FastText model)](https://github.com/facebookresearch/flores/tree/main/flores200#languages-in-flores-200) 🌐
* `lang_score` - confidence score of the predicted language code 🎯
* `perplex` - perplexity score of the original line text 📉
* `symbol` - count of tokens with strange symbols (see below)
* `upper` - count of words with unexpected mid-word uppercase (see below)
* `categ` - assigned category of the line (**Clear** ✅, **Noisy** ⚠️, **Trash** 🗑️, **Non-text** 🔣, or **Empty** 🫙)

##### Categorisation logic

`Empty` and `Non-text` are assigned by a fast CPU pre-filter (letter ratio, length, 
unique-symbol count). The remaining three categories are assigned by `categorize_line()` in 
`text_util_langID.py` after GPU perplexity scoring, using three structural detectors:

| Detector                     | What it counts                                                                                                                                                                                               |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `detect_strange_symbols`     | Tokens containing any character that is not alphanumeric and not in the allowed set `{ . - , + }`. Edge punctuation is stripped before inspection so trailing colons or parentheses don't inflate the count. |
| `detect_letter_digit_letter` | Tokens with a **letter–digit–letter sandwich** — the fingerprint of OCR digit insertions mid-word (e.g. `vyt1ačená`, `nalez2í`). Legitimate patterns like `90,9g`, `80-90cm`, `26.IX.1957` do not trigger.   |
| `detect_mid_uppercase`       | Words with unexpected uppercase mid-word (`dalSÍ`, `obkLADem`) or an uppercase-run at the start followed by lowercase (`XXWžkumu`). All-caps words and titles (`PhDr`, `MUDr`) are excluded.                 |

Decision tree (evaluated top to bottom, first match wins):

```
sym >= 2                         → Trash
sym == 1  AND  rep > 0           → Trash   (repeated strange symbol in token)
fuse >= 2                        → Trash   (multiple fused tokens in line)
sym >= 1  AND  fuse >= 1         → Trash   (symbol + fusion combined)

sym == 1                         → Noisy
fuse >= 1                        → Noisy
upper > 0                        → Noisy
ppl >= 1500  (only if wc >= 7)   → Noisy   (PPL gate disabled for short lines)

otherwise                        → Clear
```

> [!NOTE]
> Perplexity is **not** used to determine Trash. `distilgpt2` is an English model and 
> assigns very high PPL to legitimate short Czech strings (place names, postal codes, 
> form-field labels), making it unreliable as a Trash signal. It is retained only as a 
> weak Noisy signal on lines with ≥ 7 words.

Example of per-document CSV file with per-line statistics: [DOC_LINE_LANG_CLASS](data_samples/DOC_LINE_LANG_CLASS) 📁.
```
DOC_LINE_LANG_CLASS/
├── <docname1>.csv 
├── <docname2>.csv
└── ...
```

#### 4.2 Aggregate Statistics (Memory Bound) 🧠

This script processes the directory `DOC_LINE_LANG_CLASS/` with CSV files in chunks 🧩 to produce the
final page-level statistics and per-document splits (**CPU** can handle this 💻).

```
python3 langID_aggregate_STAT.py
```

* **Input 📥:** `DOC_LINE_LANG_CLASS/` (directory with CSV files from previous step)
* **Output 1 📤:** `final_page_stats.csv` (The input CSV augmented with line counts: `clear_lines`, `noisy_lines`, etc. ➕)
* **Output 2 📤:** `../DOC_LINE_STAT/` (Folder containing per-document CSVs 📁)

`final_page_stats.csv`: Page-level summary of line counts per text category 📋


- *Example*: [final_page_stats.csv](final_page_stats.csv) 📎
- *Columns*:
  * `file` - document identifier 🆔
  * `page` - page number 📄
  * `Clear` - clear lines **count**, clean and ready to be processed ✅
  * `Non-text` - non-text lines **count**, contain mostly digits/symbols 🔣
  * `Trash` - trash lines **count**, unintelligible or very high perplexity (due to OCR errors) 🗑️
  * `Noisy` - noisy lines **count**, some errors but partially understandable ⚠️
  * `Empty` - empty lines **count**, contain only whitespace 🫙
   

Example of per-document CSV file with per-page statistics of line type counts: [DOC_LINE_STAT](data_samples/DOC_LINE_STAT) 📁.
```
DOC_LINE_STAT/
├── stats_<docname1>.csv 
├── stats_<docname2>.csv
└── ...
```
This is the end of the text quality classification and filtering step. You can now use the `final_page_stats.csv` to
find files that need another round of OCR or manual correction based on the line type counts. The files with the 
majority of clean lines can be marked for further processing based on text. It is also possible to guess handwritten 
files by the absence of clear text lines or majority of trash lines, these files can be excluded from further processing
before the Handwritten Text Recognition (HTR) processing is applied.

---

## Acknowledgements 🙏

**For support write to:** lutsai.k@gmail.com responsible for this GitHub repository [^8] 🔗

- **Developed by** UFAL [^7] 👥
- **Funded by** ATRIUM [^4]  💰
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
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^7]: https://ufal.mff.cuni.cz/home-page
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b