
# ATRIUM Text Processor API Service 🚀

### Goal: Serve historical text cleaning and quality estimation models via a lightweight REST API

**Scope:** This service provides a **FastAPI** interface for the ATRIUM Text Processing pipeline. 
It allows users to upload ALTO XML or raw text files to perform intelligent layout analysis, 
split-word reconstruction, and line-level quality classification (e.g., `Clear`, `Noisy`, `Trash`) using 
**LayoutLMv3**, **FastText**, and **DistilGPT2** [^9] [^2] [^6]. It includes a static HTML 
frontend for immediate testing.

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
  * [Client Side Test 🎨](#client-side-test-)
  * [Running the Server 🚀](#running-the-server-)
  * [Using the client-side test interface](#using-the-client-side-test-interface)
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
├── setup_api_server.sh      # Shell script to set up the Python environment and install dependencies
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

| Label      | Description                                                          | Criteria (Approximate)                      |
|------------|----------------------------------------------------------------------|---------------------------------------------|
| `Clear` 🟢 | **High Quality.** Fluent text in a known language.                   | High Lang Confidence + Low Perplexity.      |
| `Noisy` 🟡 | **Usable but Rough.** Text with minor OCR errors or mixed fragments. | Moderate Perplexity or Language Confidence. |
| `Trash` 🔴 | **Unusable.** Strange language, or uncommon text formatting.         | Very High Perplexity or Unknown Language.   |

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
* **NodeJS** (For client-side development - `export NODE_OPTIONS=--openssl-legacy-provider` common fix for Webpack 4 compatibility with NodeJS 17+`).
* **Standard CPU** (Sufficient for **Client-side** development).
* **CUDA-capable GPU** (Recommended for **Server-side** inference speed, though CPU is supported). [^3]


### 2. Install Dependencies

Navigate to the root `atrium-alto-postprocess` directory, then run a setup script to 
create a virtual environment [^5], and install all of the required packages:

```bash
# Create and activate virtual environment
git clone https://github.com/ufal/atrium-alto-prostprocess.git
cd atrium-alto-postprocess
chmod + x setup_api_server.sh
./setup_api_server.sh
```

Key libraries include: fastapi, uvicorn, python-multipart, pillow, torch, timm, transformers. These
libraries can be found in `service/requirements.txt` available for manual installation if needed.

> [!NOTE] The virtual environment name is stated in the setup script and can be changed to the already existing
> one if needed.

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

Use this guide to verify the processing service is running correctly. Open a terminal 
window and run the following command from the project root:

```bash
# Activate environment if not already active
source venv/bin/activate

# Run the API service
python service/text_api.py

```

## Launch Instructions

Open two terminal windows (or tabs) and run the following commands:

```bash
source venv-api/bin/activate
cd atrium-alto-postprocess/service/
```

Then, in each window, execute the respective commands:

| **Server Console (Window 1)**                                                                                                                                                                         | **Client Console (Window 2)**                                                                                                                                                                                        |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1. Start the API:**<br><br>Run the FastAPI server from the service directory.<br><br>`python3 api.py`<br><br>You should see startup logs indicating the server is running on `http://0.0.0.0:8000`. | **2. Send a Request:**<br><br> Top-3 Classification of `image.png`:<br><br>`python3 test_api.py -<br/> -f .../image.png -v v5.3 --top 3`<br><br> where `-f` and `-v` stand for **input file** and **model version**. |


### Client Side Test 🎨

This API service includes a lightweight vanilla JS frontend (`service/frontend/script.js`) for immediate testing. 
However, the full LINDAT client integration is developed separately. [^5]

For client-side development, open a **second console window** and follow these steps:

1.  **Clone the repository** and place `atrium-alto-postprocess` project files to `lindat-common` directory
    ```bash
    git clone [https://github.com/ufal/lindat-common.git](https://github.com/ufal/lindat-common.git)
    cd lindat-common
    cp -r atrium-alto-postprocess .
    # or
    mv atrium-alto-postprocess .
    ```

2.  **Install NodeJS environment** (unless you already have one) and **Install dependencies for development:**
    ```bash
    curl -o- [https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh](https://raw.githubusercontent.com/creationix/nvm/v0.25.4/install.sh) | bash
    nvm install stable
    nvm use stable
    export NODE_OPTIONS=--openssl-legacy-provider
    npm install
    ```

3. **Run development server:**
    ```bash
    make run
    ```

For further details, please refer to the **LINDAT Common Development Guide**:
[https://github.com/ufal/lindat-common/?tab=readme-ov-file#development](https://github.com/ufal/lindat-common/?tab=readme-ov-file#development).

### Running the Server 🚀

To start the API server with hot-reloading enabled (useful for development), ensure your virtual 
environment is activated in your **first console window**: [^3]

```bash
cd atrium-alto-postprocess
source venv-api/bin/activate
uvicorn service.api:app --reload
```

The server will start at http://0.0.0.0:8000 (access to use the built-in visual testing tool).

### Using the client-side test interface

Assuming your **second console** output ends like this:

```commandline
> lindat-common@3.5.0 start
> webpack-dev-server -p --debug --quiet

(node:2985155) Warning: `--localstorage-file` was provided without a valid path
(Use `node --trace-warnings ...` to show where the warning was created)
> Project is running at http://localhost:8080/
> webpack output is served from /
> Content not from webpack is served from /home.../lindat-common
```

Open the URL `http://localhost:8080` in your web browser to access the LINDAT client interface. 

Follow the file tree to the `atrium-alto-postprocess/service/frontend` directory. The frontend interface
will open and allow you to upload images and test the API.

## Contacts 📧

**For support write to:** lutsai.k@gmail.com responsible for this GitHub repository [^8] 🔗

## Acknowledgements 🙏

* **Developed by** UFAL [^7] 👥
* **Funded by** ATRIUM [^4]  💰
* **Shared by** ATRIUM [^4] & UFAL [^7] 🔗
* **Model type:** 
  - **LayoutLMv3** for layout analysis [^9]
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
[^8]: https://github.com/K4TEL/atrium-alto-postprocess
[^7]: https://ufal.mff.cuni.cz/home-page
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b
