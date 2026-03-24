# ATRIUM Text Processor API Service 🚀

### Goal: Serve historical text cleaning and quality estimation models via a lightweight REST API

**Scope:** This service provides a **FastAPI** interface for the ATRIUM Text Processing pipeline.
It allows users to upload ALTO XML or raw text files to perform intelligent layout analysis,
split-word reconstruction, and line-level quality classification (e.g., `Clear`, `Noisy`, `Trash`,
`Non-text`, `Empty`) using **LayoutLMv3**, **FastText**, and **DistilGPT2** [^9] [^2] [^6].
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
* **Quality Classification:** Classifies every line using structural regex detectors (strange symbols, mid-word uppercase, letter–digit–letter fusions) and DistilGPT2 perplexity, implemented in `text_util_langID.py` [^6].
* **GPU Support:** Automatically detects and utilises CUDA devices for inference if available [^3].
* **Two Frontend Variants:** A self-contained standalone interface for direct use, and a LINDAT-integrated interface for deployment within the LINDAT Common framework.

## Directory Structure 📂

The service logic resides in the `service/` directory, while models are expected in a `models/` directory at the project root.

```text
atrium-alto-postprocess/
├── v3/                          # 📦 LayoutReader helper scripts
├── models/                      # 📦 Model weights (downloaded externally)
│   └── lid.176.bin              # FastText language identification binary
├── service/                     # 🚀 API source code
│   ├── text_api.py              # FastAPI application entry point
│   ├── text_inference.py        # Model manager (LayoutLMv3, FastText, DistilGPT2)
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

| Model          | Purpose                                                                                                 | Source               |
|----------------|---------------------------------------------------------------------------------------------------------|----------------------|
| **LayoutLMv3** | **Reading Order:** Reorders tokens in ALTO XML files based on 2D bounding-box layout.                   | by `hantian` [^9]    |
| **FastText**   | **Language ID:** Identifies the language of each line as a pre-filter signal.                           | by `facebook` [^2]   |
| **DistilGPT2** | **Perplexity:** Measures how linguistically "surprising" a line is — elevated scores suggest OCR noise. | by `distilbert` [^6] |

> [!NOTE]
> Classification is performed primarily by the structural detectors in `text_util_langID.py`.
> DistilGPT2 perplexity is a **supporting signal** used only on longer lines (word count ≥ 7),
> because it is unreliable on short or non-English text.
> FastText language scores are **not** used as a primary Trash/Clear indicator in the current pipeline.

## Quality Categories 🪧

The service classifies every text line into one of five categories. The first two (`Empty`, `Non-text`)
are assigned by a fast CPU pre-filter before any model inference. The remaining three are assigned by
the structural detectors and perplexity gate in `text_util_langID.categorize_line()`.

| Label         | Description                                                           | Primary Signal                                                                             |
|---------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| `Clear` 🟢    | **High quality.** Passes all structural checks and perplexity gate.   | 0 strange-symbol tokens, 0 uppercase artefacts, PPL < threshold (on long lines).           |
| `Noisy` 🟡    | **Usable but degraded.** Minor OCR artefacts, recoverable downstream. | 1 strange-symbol token, OR mid-word uppercase artefacts, OR elevated PPL on long lines.    |
| `Trash` 🔴    | **Structurally corrupt.** Not worth downstream processing.            | ≥ 2 strange-symbol tokens, or co-occurring symbol + uppercase / fusion artefacts.          |
| `Non-text` 🔵 | **No meaningful text.** Purely numeric / separator content.           | RE_NON_TEXT match (dates, page numbers, measurements) or digit ratio > 40 % on short line. |
| `Empty` ⚪     | **Blank line.** Whitespace only.                                      | `len(stripped) == 0`                                                                       |

## API Usage 📡

### Endpoints 🔗

| Method | Path       | Description                                                                       |
|--------|------------|-----------------------------------------------------------------------------------|
| `GET`  | `/`        | Serves the standalone `index.html` interface for manual testing.                  |
| `GET`  | `/info`    | Returns service status, active device (`cpu` or `cuda`), and quality categories.  |
| `POST` | `/process` | Uploads a file for layout analysis, cleaning, and line-level classification.      |

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

Each item in `cleaned_lines` carries the fields used by `text_util_langID.categorize_line()`.
`lang` and `lang_conf` are **not** returned — FastText language scores are not part of the
structural classification decision.

```json
{
  "type": "alto_xml",
  "filename": "page_01.xml",
  "cleaned_lines": [
    {
      "line_num": 1,
      "text": "The quick brown fox jumps over the lazy dog.",
      "perplexity": 12.5,
      "sym_count": 0,
      "upper_count": 0,
      "category": "Clear"
    },
    {
      "line_num": 2,
      "text": "TYRSOVA5===aras T>r«l",
      "perplexity": 4800.0,
      "sym_count": 2,
      "upper_count": 0,
      "category": "Trash"
    },
    {
      "line_num": 3,
      "text": "1956–1959",
      "perplexity": 0,
      "sym_count": 0,
      "upper_count": 0,
      "category": "Non-text"
    }
  ]
}
```

**Response fields:**

| Field         | Type   | Description                                                                               |
|---------------|--------|-------------------------------------------------------------------------------------------|
| `line_num`    | int    | 1-based line position after layout reordering.                                            |
| `text`        | string | Cleaned line text with split-word merges applied.                                         |
| `perplexity`  | float  | DistilGPT2 perplexity. `0` means the line was pre-filtered and inference was skipped.     |
| `sym_count`   | int    | Tokens containing characters outside the allowed internal set (`detect_strange_symbols`). |
| `upper_count` | int    | Tokens with mid-word uppercase artefacts — Patterns 1–3 (`detect_mid_uppercase`).         |
| `category`    | string | One of: `Clear`, `Noisy`, `Trash`, `Non-text`, `Empty`.                                   |

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
git clone https://github.com/ufal/atrium-alto-postprocess.git
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
wget "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin" \
     -O models/lid.176.bin
```

> [!NOTE]
> LayoutLMv3 and DistilGPT2 are downloaded and cached automatically by Hugging Face Transformers on the first run [^9] [^6].

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
git clone https://github.com/ufal/lindat-common.git
cd lindat-common
cp -r /path/to/atrium-alto-postprocess .
```

**2. Install NodeJS and dependencies:**

```bash
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh | bash
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

## Contacts 📧

**For support write to:** lutsai.k@gmail.com — responsible for this GitHub repository [^8] 🔗

## Acknowledgements 🙏

* **Developed by** UFAL [^7] 👥
* **Funded by** ATRIUM [^4] 💰
* **Shared by** ATRIUM [^4] & UFAL [^7] 🔗
* **Models used:**
  - **LayoutLMv3** for reading-order layout analysis [^9]
  - **FastText** for language identification [^2]
  - **DistilGPT2** for perplexity estimation [^6]

**©️ 2026 UFAL & ATRIUM**

---

[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^3]: https://developer.nvidia.com/cuda-python
[^4]: https://atrium-research.eu/
[^5]: https://docs.python.org/3/library/venv.html
[^6]: https://huggingface.co/distilbert/distilgpt2
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-alto-postprocess
[^9]: https://github.com/ppaanngggg/layoutreader