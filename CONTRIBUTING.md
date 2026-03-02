# 🤝 Contributing to the ALTO XML Postprocessing Pipeline of the ATRIUM project

Welcome! This repository [^8] provides a robust workflow for transforming raw OCR outputs 
(ALTO XML) into clean and classified textual data. It addresses common challenges 
in digital archives, such as multi-column layout reconstruction, word-split recovery, 
and automated quality filtering.

## 🏗️ Project Contributions & Capabilities

This pipeline contributes 4 major stages to the data processing lifecycle, 
as detailed in the section of the main [README 🛤️ Workflow Stages](README.md#-workflow-stages)

### 1. Granular Data Management

The pipeline allows archives to move from document-level files to page-level management.

* **Splitting:** Automatically breaks down document-level ALTO XMLs into individual page files.
* **Page Inventory:** Generates a foundational CSV statistics table (Step 2) capturing for every page in an archive: 
  * text line counts
  * number of illustrations and graphics 

### 2. Multi-Method Text Extraction

Archive managers can choose extraction methods based on their specific hardware and accuracy requirements:

| Method                    | Best For...               | Resource Type   | Key Feature                                                        |
|---------------------------|---------------------------|-----------------|--------------------------------------------------------------------|
| **LayoutReader** [^9]     | Multi-column layouts      | **GPU**         | Reconstructs natural reading order and handles hyphenated words.   |
| **alto-tools** [^1]       | Fast, large-scale batches | **CPU**         | High speed, low memory usage for raw content overview.             |
| **GLM (LLM-based)** [^10] | Degraded/Complex scans    | **GPU** (48GB+) | Generative OCR directly from images; patches transcription errors. |

### 3. Automated Quality & Language Assessment

A core contribution of this project is the ability to filter "noisy" OCR data without manual 
review. Every text line is categorized using **FastText** [^2] for language identification and 
**DistilGPT2** [^6] for perplexity scoring.

**Data Quality Categories:**

* ✅ **Clear:** High-quality, fluent text ready for research.
* ⚠️ **Noisy (Rough):** Contains minor OCR errors; usable but imperfect.
* 🗑️ **Trash:** Unintelligible OCR or non-prose content.
* 🔣 **Non-text:** Mostly digits or symbols.
* 🫙 **Empty:** Whitespace only.

### 4. Lightweight API & Testing Interface

The project includes a **FastAPI** service that allows for easy integration into existing 
archival systems. It provides:

* **RESTful Endpoints:** `/process` and `/info` for remote file processing.
* **Visual Testing:** A built-in JS frontend for immediate manual verification of model performance.

---

## 📞 Contacts & Acknowledgements

For support or specific archival integration questions, contact **lutsai.k@gmail.com**.

* **Developed by:** UFAL [^7]
* **Funded by:** ATRIUM [^4]
* **Models:** 
  * FastText [^2]
  * DistilGPT2 [^6]
  * GLM-4v-9b [^10]
  * LayoutLMv3 [^9]

**©️ 2026 UFAL & ATRIUM**


[^1]: https://github.com/cneud/alto-tools
[^2]: https://huggingface.co/facebook/fasttext-language-identification
[^3]: https://github.com/ufal/ker
[^4]: https://atrium-research.eu/
[^5]: https://github.com/K4TEL/atrium-nlp-enrich
[^6]: https://huggingface.co/distilbert/distilgpt2
[^8]: https://github.com/K4TEL/atrium-alto-postprocess
[^7]: https://ufal.mff.cuni.cz/home-page
[^9]: https://github.com/ppaanngggg/layoutreader
[^10]: https://huggingface.co/THUDM/glm-4v-9b
