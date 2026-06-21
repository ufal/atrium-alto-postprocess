# ATRIUM Text Processor API Service 🚀

### Goal: Serve historical text cleaning and quality estimation models via a lightweight REST API

**Scope:** This service provides a **FastAPI** interface for the ATRIUM Text Processing pipeline.
It allows users to upload ALTO XML or raw text files to perform intelligent layout analysis,
split-word reconstruction, and line-level quality classification (e.g., `Clear`, `Noisy`, `Trash`,
`Non-text`, `Empty`) using **LayoutLMv3**, **FastText**, and **Qwen2.5-0.5B** [^9] [^2] [^6].
Two frontend variants are included: a **standalone** interface (`frontend/`) and a
**LINDAT-integrated** interface (`frontend-lindat/`).

### Table of contents 📑

* [Service Description 📇](#service-description-)
* [Directory Structure 📂](#directory-structure-)
* [Supported Models 🧠](#supported-models-)
* [Quality Categories 🪧](#quality-categories-)
* [API Usage 📡](#api-usage-)
* [Installation & Setup 🛠](#installation--setup-)
  * [Prerequisites](#1-prerequisites)
  * [Install Dependencies](#2-install-dependencies)
  * [Model Weights](#3-model-weights)
* [Quick API Test Launch 🚀](#quick-api-test-launch-)
* [Launch Instructions](#launch-instructions)
  * [Running the Server 🚀](#running-the-server-)
  * [Standalone Frontend 🖥️](#standalone-frontend-)
  * [LINDAT-integrated Frontend 🎨](#lindat-integrated-frontend-)
* [Contacts 📧](#contacts-)
* [Acknowledgements 🙏](#acknowledgements-)

---
## Service Description 📇

The API is built using **FastAPI** and is designed to turn raw OCR output into clean, classified text data.
It acts as a bridge between complex NLP models and downstream applications or web interfaces.

Key features:

* **Layout Analysis:** Uses **LayoutLMv3** to correctly reorder tokens from ALTO XML files based on 2D spatial layout, handling multi-column pages [^9].
* **Text Cleaning:** Automatically detects and merges hyphenated words split across lines using ALTO `SUBS_TYPE` / `SUBS_CONTENT` attributes and regex-based reconstruction.
* **Quality Classification:** Classifies every line with a composite **quality score** built from structural detectors (strange symbols, mid-word uppercase, letter–digit–letter fusions, gibberish, fused/rotated tokens) and **Qwen2.5-0.5B** perplexity, implemented in `text_util_langID.py` [^6]. The category is then assigned from quality-score thresholds plus named overrides.
* **GPU Support:** Automatically detects and utilises CUDA devices for inference if available [^3].
* **Two Frontend Variants:** A self-contained standalone interface for direct use, and a LINDAT-integrated interface for deployment within the LINDAT Common framework.
* **CORS Support:** Cross-Origin Resource Sharing is configurable via the `ALLOWED_ORIGINS` environment variable (defaults to `http://localhost:8080,http://localhost:5500`).

## Directory Structure 📂

The service logic resides in the `service/` directory, while models are expected in a `models/` directory at the project root.

```text
atrium-alto-postprocess/
├── v3/                          # 📦 LayoutReader helper scripts
├── models/                      # 📦 Model weights (downloaded externally)
│   └── lid.176.bin              # FastText language identification binary
├── service/                     # 🚀 API source code
│   ├── text_api.py              # FastAPI application entry point
│   ├── text_inference.py        # Model manager (LayoutLMv3, FastText, Qwen2.5-0.5B)
│   ├── utils.py                 # XML parsing, box normalisation, cleaning logic
│   ├── frontend/                # 🖥️  Standalone frontend (no external dependencies)
│   │   ├── index.html           # Self-contained web interface
│   │   └── script.js            # Vanilla JS — no jQuery, no build step required
│   ├── frontend-lindat/         # 🎨 LINDAT-integrated frontend
│   │   ├── index.html           # Interface styled for lindat-common
│   │   └── script.js            # JS adapted to the lindat-common webpack bundle
│   ├── requirements.txt         # Python dependencies
│   └── README.md                # API service documentation
├── text_util_langID.py          # Structural quality detectors and categorisation logic
├── setup_api_server.sh          # Sets up virtual environment and installs dependencies
├── README.md                    # Project overview and documentation (this file)
├── LICENSE
└── ...                          # Other project files (scripts, data samples, paradata)
```


## Supported Models 🧠

The pipeline applies three models in sequence, balancing structural layout understanding with semantic quality estimation.

| Model            | Purpose                                                                                                 | Source             |
|------------------|---------------------------------------------------------------------------------------------------------|--------------------|
| **LayoutLMv3**   | **Reading Order:** Reorders tokens in ALTO XML files based on 2D bounding-box layout.                   | by `hantian` [^9]  |
| **FastText**     | **Language ID:** Identifies the language of each line as a pre-filter signal.                           | by `facebook` [^2] |
| **Qwen2.5-0.5B** | **Perplexity:** Measures how linguistically "surprising" a line is — elevated scores suggest OCR noise. | by `Qwen` [^6]     |

> [!NOTE]
> The category is decided by the composite **quality score** — a weighted sum of nine structural, language and
> perplexity signals routed through thresholds, with named overrides — not by a fixed detector decision-tree.
> Perplexity is one weighted signal; on short 1–2 word lines it is capped before scoring because the LM has too
> little context. `distilgpt2` remains available as an English-only alternative via the `GPT2_MODEL_NAME`
> environment variable (re-tune `PERPLEXITY_THRESHOLD_MAX`, see [Troubleshooting](#hardware--configuration-troubleshooting)).
> Full logic: main [README → Composite Quality Score](../README.md#composite-quality-score) and
> [Categorisation Logic](../README.md#categorisation-logic).

## Quality Categories 🪧

The service classifies every text line into one of five categories. The first two (`Empty`, `Non-text`)
are assigned by a fast CPU pre-filter before any model inference. The remaining three are assigned by
`text_util_langID.categorize_line()` from the composite **quality score**, after immediate overrides.

| Label         | Description                                                           | Primary Signal                                                                                    |
|---------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| `Clear` 🟢    | **High quality.** Ready for downstream NLP.                           | `quality_score ≥ CATEG_NOISY_SCORE_MAX` (0.85), or a low-perplexity / clean-prose override.        |
| `Noisy` 🟡    | **Usable but degraded.** Minor OCR artefacts, recoverable downstream. | `CATEG_TRASH_SCORE_MAX` (0.55) ≤ `quality_score` < `CATEG_NOISY_SCORE_MAX` (0.85).                |
| `Trash` 🔴    | **Structurally corrupt.** Not worth downstream processing.            | `quality_score < CATEG_TRASH_SCORE_MAX` (0.55), or a hard override (all-caps/no-vowel, inverted).  |
| `Non-text` 🔵 | **No meaningful text.** Purely numeric / separator content.           | CPU pre-filter: dates, page numbers, archive/stamp codes, or digit ratio > 40 % on short lines.   |
| `Empty` ⚪     | **Blank line.** Whitespace only.                                      | `word_count == 0` / whitespace only.                                                              |

> [!NOTE]
> The thresholds and the full set of overrides (hard-sweep, inverted-scan, low-perplexity-clear, clean-prose
> promotion, mostly-readable cap, and the document/page post-passes) are documented once in the main
> [README → Categorisation Logic](../README.md#categorisation-logic) and are not duplicated here.


## API Usage 📡

### Endpoints 🔗

| Method | Path       | Description                                                                                   |
|--------|------------|-----------------------------------------------------------------------------------------------|
| `GET`  | `/`        | Serves the standalone `index.html` interface for manual testing.                              |
| `GET`  | `/info`    | Returns service status, active device (`cpu` or `cuda`), line fields, and quality categories. |
| `POST` | `/process` | Uploads a file for layout analysis, cleaning, and line-level classification.                  |

### Request Example 💻

**Endpoint:** `/process`

**Parameters (Form Data):**

* `file`: The document file (`.xml` ALTO or `.txt`).
* `task_type`: `alto`, `text`, or `auto` (default — detected from file extension).

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/page_01.xml" \
  -F "task_type=auto"
```

### Response Schema

Each item in `cleaned_lines` carries the fields used by the classification pipeline.

```json
{
  "type": "alto_xml",
  "filename": "page_01.xml",
  "cleaned_lines": [
    {
      "line_num": 1,
      "text": "The quick brown fox jumps over the lazy dog.",
      "lang": "eng",
      "lang_score": 0.9821,
      "perplexity": 12.5,
      "sym_count": 0,
      "upper_count": 0,
      "word_weird": 0.0,
      "quality_score": 0.9501,
      "category": "Clear"
    },
    {
      "line_num": 2,
      "text": "TYRSOVA5===aras T>r«l",
      "lang": "ces",
      "lang_score": 0.4201,
      "perplexity": 4800.0,
      "sym_count": 2,
      "upper_count": 0,
      "word_weird": 0.85,
      "quality_score": 0.1205,
      "category": "Trash"
    },
    {
      "line_num": 3,
      "text": "1956–1959",
      "lang": "N/A",
      "lang_score": 0.0,
      "perplexity": 0.0,
      "sym_count": 0,
      "upper_count": 0,
      "word_weird": 0.0,
      "quality_score": 0.0,
      "category": "Non-text"
    }
  ]
}
```

**Response fields:**

| Field           | Type   | Description                                                                                                                                |
|-----------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `line_num`      | int    | 1-based line position after layout reordering.                                                                                             |
| `text`          | string | Cleaned line text with split-word merges applied.                                                                                          |
| `lang`          | string | ISO language code predicted by FastText (e.g., `eng`, `ces`).                                                                              |
| `lang_score`    | float  | FastText confidence score `[0, 1]`.                                                                                                        |
| `perplexity`    | float  | Qwen2.5-0.5B perplexity. `0` means the line was pre-filtered and inference was skipped.                                                    |
| `sym_count`     | int    | Tokens containing characters outside the allowed internal set (`detect_strange_symbols`).                                                  |
| `upper_count`   | int    | Tokens with mid-word uppercase artefacts — Patterns 1–3 (`detect_mid_uppercase`).                                                          |
| `word_weird`    | float  | Mean per-word weirdness score `[0, 1]`; combines strange-symbol, repeated-char, LDL-fusion, mid-uppercase and mirror-OCR (`w` / caps-prefix) signals; `0` = fully clean. |
| `quality_score` | float  | Composite quality score `[0, 1]`; weighted sum of nine signals (valid-word ratio, word-weirdness, perplexity, length, garbage density, vowel quality, language confidence, gibberish, fused-word ratio); higher = cleaner. |
| `category`      | string | One of: `Clear`, `Noisy`, `Trash`, `Non-text`, `Empty`.                                                                                    |


## Installation & Setup 🛠

### 1. Prerequisites

* **Python 3.10+** virtual environment [^5].
* **Standard CPU** (sufficient for inference; GPU recommended for batch processing).
* **CUDA-capable GPU** (optional — auto-detected at startup for faster inference) [^3].
* **NodeJS** (only required for the **LINDAT-integrated** frontend — `export NODE_OPTIONS=--openssl-legacy-provider` is a common fix for Webpack 4 compatibility with NodeJS 17+).

### 2. Install Dependencies

Clone the repository and run the setup script from the project root. It creates a virtual
environment, installs all Python dependencies, fetches the `v3/` LayoutReader scripts via
sparse checkout, and downloads the FastText binary:

```bash
git clone [https://github.com/ufal/atrium-alto-postprocess.git](https://github.com/ufal/atrium-alto-postprocess.git)
cd atrium-alto-postprocess
chmod +x setup_api_server.sh
./setup_api_server.sh
```

Key libraries: `fastapi`, `uvicorn`, `python-multipart`, `torch`, `transformers`, `fasttext`, `lxml`, `numpy`.
Full list in `service/requirements.txt` for manual installation if needed.

> [!NOTE]
> The virtual environment name is set in `setup_api_server.sh` and can be changed to match an existing environment.

### 3. Model Weights

The setup script downloads the FastText binary automatically. If you prefer to download it manually:

```bash
mkdir -p models
wget "[https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin](https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin)" \
     -O models/lid.176.bin
```

> [!NOTE]
> LayoutLMv3 and Qwen2.5-0.5B are downloaded and cached automatically by Hugging Face Transformers on the first run [^9] [^6].

## Quick API Test Launch 🚀

```bash
source venv/bin/activate
python service/text_api.py
```

The server starts at `http://0.0.0.0:8000`. The standalone frontend is served at `/`.
Send a test request in a second terminal:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@data_samples/ALTO/CTX195603828.alto.xml" \
  -F "task_type=alto"
```

## Launch Instructions

### Running the Server 🚀

Activate your virtual environment and start the API with hot-reloading (useful during development):

```bash
cd atrium-alto-postprocess
source venv/bin/activate          # or: source venv-api/bin/activate
uvicorn service.text_api:app --reload
```

The server will be available at `http://0.0.0.0:8000`.

---

### Standalone Frontend 🖥️

`service/frontend/` is a self-contained interface with no build step or external framework required.
It is served directly by the FastAPI server at `http://localhost:8000` and works out of the box.

**To use it**, simply start the server (see above) and open `http://localhost:8000` in your browser.

Features:
- Drag-and-drop or click-to-upload for `.xml` and `.txt` files.
- Processing mode selector (`auto` / `alto` / `text`).
- Results table with `Sym`, `Upper`, and `PPL` columns aligned to `text_util_langID.py`.
- Category breakdown bar showing counts for all five labels.
- Raw extracted text toggle.

> [!NOTE]
> If you are running the frontend from a local dev server (e.g. Live Server on port `5500`),
> `script.js` automatically redirects API calls to `http://localhost:8000`.

---

### LINDAT-integrated Frontend 🎨

`service/frontend-lindat/` is the frontend variant styled and bundled for deployment within the
[LINDAT Common](https://github.com/ufal/lindat-common) framework. It requires NodeJS and the
`lindat-common` webpack build.

Open a **second terminal window** alongside your running server and follow these steps:

**1. Place the project inside `lindat-common`:**

```bash
git clone [https://github.com/ufal/lindat-common.git](https://github.com/ufal/lindat-common.git)
cd lindat-common
cp -r /path/to/atrium-alto-postprocess .
```

**2. Install NodeJS and dependencies:**

```bash
curl -o- [https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh](https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh) | bash
nvm install stable
nvm use stable
export NODE_OPTIONS=--openssl-legacy-provider
npm install
```

**3. Start the webpack dev server:**

```bash
make run
```

Expected output:

```
> lindat-common@3.5.0 start
> webpack-dev-server -p --debug --quiet

> Project is running at http://localhost:8080/
> webpack output is served from /
> Content not from webpack is served from /home/.../lindat-common
```

Open `http://localhost:8080` and navigate to the
`atrium-alto-postprocess/service/frontend-lindat` directory in the file tree.

For further details on the LINDAT development workflow see the
[LINDAT Common Development Guide](https://github.com/ufal/lindat-common/?tab=readme-ov-file#development).


---

## Hardware & Configuration Troubleshooting

* **GLM-4v VRAM Requirements:** The GLM-4v Vision-Language Model requires massive GPU memory. You **must have a GPU
with at least 48 GB of VRAM** (e.g., an NVIDIA RTX A6000 or a multi-GPU setup) to run the extraction pipeline
successfully. Running this on consumer GPUs (like a 3090/4090) will likely result in Out-Of-Memory (OOM) crashes.
* **Perplexity Threshold Coupling:** The service uses **Qwen2.5-0.5B** by default, matched to `PERPLEXITY_THRESHOLD_MAX
= 1000.0` in `config_langID.txt`. If you switch the perplexity model via the `GPT2_MODEL_NAME` environment variable
(e.g., to the English-only `distilgpt2`), you **must** recalibrate `PERPLEXITY_THRESHOLD_MAX` — perplexity scales differ
wildly between architectures (≈ `3000.0` suits `distilgpt2`), so a value tuned for one model is mis-calibrated for the other.

---

## Contacts 📧

**For support write to:** lutsai.k@gmail.com — responsible for this GitHub repository [^8] 🔗

## Acknowledgements 🙏

* **Developed by** UFAL [^7] 👥
* **Funded by** ATRIUM [^4] 💰
* **Shared by** ATRIUM [^4] & UFAL [^7] 🔗
* **Models used:**
  - **LayoutLMv3** for reading-order layout analysis [^9]
  - **FastText** for language identification [^2]
  - **Qwen2.5-0.5B** for perplexity estimation [^6]

**©️ 2026 UFAL & ATRIUM**

---

[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^3]: https://developer.nvidia.com/cuda-python
[^4]: https://atrium-research.eu/
[^5]: https://docs.python.org/3/library/venv.html
[^6]: https://huggingface.co/Qwen/Qwen2.5-0.5B
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^9]: https://github.com/ppaanngggg/layoutreader
