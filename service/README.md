
# ATRIUM Text Processor API Service 🚀

### Goal: Serve historical text cleaning and quality estimation models via a lightweight REST API

**Scope:** This service provides a **FastAPI** interface for the Atrium Text Processing pipeline. 
It allows users to upload ALTO XML or raw text files to perform intelligent layout analysis, 
split-word reconstruction, and line-level quality classification (e.g., Clear, Noisy, Trash) using 
**LayoutLMv3**, **FastText**, and **DistilGPT2** [^9] [^2] [^6]. It includes a static HTML frontend for immediate testing.


### Table of contents 📑

* [Service Description 📇](#service-description-)
* [Directory Structure 📂](#directory-structure-)
* [Supported Models 🧠](#supported-models-)
* [Quality Categories 🪧](#quality-categories-)
* [API Usage 📡](#api-usage-)
* [Installation & Setup 🛠](#installation--setup-)
* [Quick API Test Launch 🚀](#quick-api-test-launch-)
* [Contacts 📧](#contacts-)
* [Acknowledgements 🙏](#acknowledgements-)

---

## Service Description 📇

The API is built using **FastAPI** and is designed to turn raw OCR outputs into clean, analyzed text data. It acts as 
a bridge between complex NLP models and downstream applications or web interfaces.

Key features:

* **Layout Analysis:** Uses **LayoutLMv3** to correctly reorder text from ALTO XML files based on 2D spatial layout (handling multi-column layouts) [^9].
* **Text Cleaning:** Automatically detects and merges hyphenated words split across lines using regex-based reconstruction.
* **Quality Filtering:** Classifies every line based on language confidence and perplexity to filter out OCR noise [^6].
* **GPU Support:** Automatically detects and utilizes CUDA devices for inference if available [^3].
* **Lightweight Frontend:** Includes a simple HTML/JS interface for manual file uploads and visualization.

## Directory Structure 📂

The service logic resides in the `service/` directory, while models are expected in a `models/` directory at the project root.

```text
atrium-alto-postprocess/
├── v3/                      # 📦 LayoutReader files (helper scripts)
├── models/                  # 📦 Model weights (downloaded externally)
│   └── lid.176.bin          # FastText binary file
├── service/                 # 🚀 API Source Code
│   ├── text_api.py          # FastAPI application entry point
│   ├── text_inference.py    # Model manager (LayoutLMv3, FastText, GPT2)
│   ├── utils.py             # XML parsing, Math helpers, and cleaning logic
│   ├── frontend/            # 🎨 Frontend assets
│   │   ├── index.html       # Web interface
│   │   └── script.js        # Logic for the web interface
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # This file - API service documentation
├── README.md                # Project overview and documentation
├── LICENSE                  # Open source license
└── ...                      # other project files

```

## Supported Models 🧠

The pipeline utilizes a cascade of three distinct models to process text, balancing structural understanding with semantic quality checks.

| Model          | Purpose                                                                                       | Source                       |
|----------------|-----------------------------------------------------------------------------------------------|------------------------------|
| **LayoutLMv3** | **Reading Order:** Reorders words in ALTO XML files based on 2D spatial layout.               | `hantian/layoutreader` [^9]  |
| **FastText**   | **Language ID:** Identifies the language of a text line to ensure it matches expectations.    | `facebook/fasttext` [^2]     |
| **DistilGPT2** | **Perplexity:** Calculates how "surprising" the text is. High perplexity indicates OCR noise. | `distilbert/distilgpt2` [^6] |

## Quality Categories 🪧

The service classifies every text line into one of three structural categories based on Perplexity (PPL) and Language Confidence scores:

| Label   | Description                                                             | Criteria (Approximate)                      |
|---------|-------------------------------------------------------------------------|---------------------------------------------|
| `Clear` | 🟢 **High Quality.** Fluent text in a known language.                   | High Lang Confidence + Low Perplexity.      |
| `Noisy` | 🟡 **Usable but Rough.** Text with minor OCR errors or mixed fragments. | Moderate Perplexity or Language Confidence. |
| `Trash` | 🔴 **Unusable.** Headers, page numbers, or severe OCR garbage.          | Very High Perplexity or Unknown Language.   |

## API Usage 📡

### Endpoints 🔗

| Method | Path       | Description                                                                     |
|--------|------------|---------------------------------------------------------------------------------|
| `GET`  | `/`        | Serves the static `index.html` interface for manual testing.                    |
| `GET`  | `/info`    | Returns service status, active device (`cpu` or `cuda`), and supported formats. |
| `POST` | `/process` | Uploads a file for cleaning, extraction, and classification.                    |

### Request Example 💻

**Endpoint:** `/process`

**Parameters (Form Data):**

* `file`: The document file (`.xml` ALTO or `.txt`).
* `task_type`: `alto`, `text`, or `auto` (default).

Request example using `curl`:

```bash
curl -X POST "http://localhost:8000/process" \
  -F "file=@/path/to/page_01.xml" \
  -F "task_type=auto"

```

Example JSON response:

```json
{
  "type": "alto_xml",
  "filename": "page_01.xml",
  "cleaned_lines": [
    {
      "line_num": 1,
      "text": "The quick brown fox jumps over the lazy dog.",
      "lang": "eng_Latn",
      "lang_conf": 0.98,
      "perplexity": 12.5,
      "category": "Clear"
    },
    {
      "line_num": 2,
      "text": "x8& s9d 1!!",
      "lang": "unknown",
      "lang_conf": 0.2,
      "perplexity": 5200.0,
      "category": "Trash"
    }
  ]
}

```

## Installation & Setup 🛠

### 1. Prerequisites

* **Python 3.10+** 
* **CUDA-capable GPU** (Recommended for LayoutLMv3 inference speed, though CPU is supported) [^3].

### 2. Install Dependencies

Navigate to the project root and install the required packages using a virtual environment:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (ensure pytorch is installed according to your CUDA version)
pip install -r service/requirements.txt

```

### 3. Model Weights

The service attempts to load models automatically. However, **FastText** requires a specific binary file.
Create a `models` directory in the project root and download the binary [^2] :

```bash
mkdir models
wget "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin" -O models/lid.176.bin

```

> [!NOTE]
> LayoutLMv3 and DistilGPT2 will be downloaded by Hugging Face Transformers on the first run and cached locally [^9] [^6]).

## Quick API Test Launch 🚀

Use this guide to verify the processing service is running correctly.

### Launch Instructions

Open a terminal window and run the following command from the project root:

```bash
# Activate environment if not already active
source venv/bin/activate

# Run the API service
python service/text_api.py

```

You should see startup logs indicating the server is running on `http://0.0.0.0:8000`.

## Client Side Test 🎨

The service comes with a built-in testing tool accessible via a web browser.

1. Open your browser to `http://localhost:8000`.
2. Click the upload box to select an `.xml` (ALTO) or `.txt` file.
3. Click **Start Processing**.
4. View the extracted text and quality tables in the browser.






## Installation & Setup 🛠

### 1. Prerequisites

* **Python 3.10+**
* **CUDA-capable GPU** (Recommended for LayoutLMv3 speed, though CPU is supported).


### 3. Model Weights

The service attempts to load models automatically. However, **FastText** requires a specific binary file.
Create a `models` directory in the project root and download the binary:

```bash
mkdir models
wget "https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin" -O models/lid.176.bin

```

*Note: LayoutLMv3 and DistilGPT2 will be downloaded by Hugging Face Transformers on the first run.*

## Quick API Test Launch 🚀

Use this guide to verify the processing service is running.

### Launch Instructions

Run the server from the root of the project:

```bash
# From project root
python service/text_api.py

```
You should see startup logs indicating the server is running on `http://0.0.0.0:8000`.


## Contacts 📧

**For support write to:** lutsai.k@gmail.com responsible for this GitHub repository [^8] 🔗

## Acknowledgements 🙏

* **Developed by** UFAL [^7] 👥
* **Funded by** ATRIUM [^4]  💰
* **Shared by** ATRIUM [^4] & UFAL [^7] 🔗

**©️ 2026 UFAL & ATRIUM**

---

[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^3]: https://developer.nvidia.com/cuda-python
[^4]: https://atrium-research.eu/
[^5]: https://github.com/K4TEL/atrium-nlp-enrich
[^6]: https://huggingface.co/distilbert/distilgpt2
[^8]: https://github.com/K4TEL/atrium-alto-postprocess
[^7]: https://ufal.mff.cuni.cz/home-page
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b
