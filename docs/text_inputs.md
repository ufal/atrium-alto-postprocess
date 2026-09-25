# Input formats: standards, producers and readers (#31)

`--method text-lines` accepts any text-bearing file and turns it into an ordered list of **pages**,
each an ordered list of **lines**. Those lines become the rows of the categorized CSV output: first
`DOC_LINES_TEXT/<doc>.csv`, then, after Step 4, `DOC_LINE_CATEG/<doc>.csv`. This page is the reference for
the input formats. [Formats and their standards](#formats-and-their-standards) says which standard
defines each format, which tools write it and what this repo keeps of it; §1–§8 say what the readers
in [`text_formats.py`](../text_formats.py) do. The pipeline mechanics are in the
[README](../README.md#any-other-text-bearing-input-pdf-docx-txt--31-).

```
TEXT/ (any mix) ──text_split.py──▶ PAGE_TEXT/<doc>/<doc>-<n>.txt + ingest_report.csv + pages_report.csv
                ──text_stats_create.py──▶ stats CSV (file,page,textlines,illustrations,graphics,strings,path)
                ──extract_TEXT_2_TXT.py──▶ PAGE_TXT_TEXT/<doc>/<doc>-<n>.txt + DOC_LINES_TEXT/<doc>.csv
                ──classify_TEXT.py / aggregate_STAT.py (unchanged)──▶ DOC_LINE_CATEG/, DOC_LINE_STATS/
```

ALTO XML keeps its own methods (`layoutreader`, `alto-tools`, `glm`), which reconstruct reading order
and dehyphenate. Generic OCR JSON keeps `json-keys`. text-lines reads both as well, in document
order, when they arrive mixed with other files; see §2 for how its page numbers differ.

Without `--input-dir`, `run_pipeline.py --method text-lines` reads `[PIPELINE].INPUT_DIR_TEXT`
(default `data_samples/TEXT`; it falls back to `INPUT_DIR` when that key is absent).

## Formats and their standards

The inputs come in two families. **OCR and layout formats** are the output of text recognition on
a page image. They have real pages and record where each line sits on the image, often each word or
character too, how sure the engine was, and sometimes the typography. **Text-bearing documents**
carry text and a logical structure (paragraphs, cells, slides, messages), but no page image and no
coordinates. Many of them (DOCX, ODT, EPUB, HTML) do not fix pages at all.

This repository keeps the **text** of every format, cut into pages and lines, and each page's
original label (§2). Its outputs keep no coordinates, engine confidences or typography (fonts,
sizes, bold, italics). The quality score of Step 4 is computed from the text alone. The ALTO
methods are the exception: LayoutReader orders the lines by their word boxes, and the page statistics
count ALTO's lines, strings, illustrations and graphics.

### OCR and layout formats

| Format                     | Standard (steward)                                                                                                                                                                          | Typical producers                                                                                           | Read by                                                                            | `source.origin`                                                                                           |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| ALTO XML                   | ALTO (Analyzed Layout and Text Object), an XML schema of the ALTO Editorial Board with the Library of Congress as maintenance agency; v2–v4 in `http://www.loc.gov/standards/alto/ns-v<N>#` | ABBYY FineReader (the ATRIUM collections), Tesseract (`alto`), Kraken / eScriptorium, Transkribus, PERO-OCR | the ALTO methods (`page_split.py`, Step 3; every version), text-lines, the service | `ABBYY-ALTO`; text-lines: the engine the header names (`ocr:tesseract`, `ocr:pero`, …), else `ABBYY-ALTO` |
| PAGE XML                   | PAGE (Page Analysis and Ground-truth Elements), PRImA Research Lab, University of Salford; dated schema versions, e.g. `http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15`    | Transkribus, eScriptorium / Kraken, OCR-D, PERO-OCR, Aletheia (ground truth)                                | text-lines, the service                                                            | `ocr:page-xml`                                                                                            |
| hOCR                       | hOCR, a microformat that puts OCR results into HTML class and `title` attributes (Thomas Breuel; community specification 1.2)                                                               | Tesseract (`hocr`), Kraken, OCRopus                                                                         | text-lines, the service                                                            | the engine `ocr-system` names (`ocr:tesseract`, …), else `ocr:hocr`                                       |
| ABBYY FineReader XML       | ABBYY's own export schema, versioned in its namespace (the sample: `http://www.abbyy.com/FineReader_xml/FineReader10-schema-v1.xml`)                                                        | ABBYY FineReader Engine and Server                                                                          | text-lines, the service                                                            | `ocr:abbyy-finereader`                                                                                    |
| DjVuXML                    | the XML form of a DjVu file's hidden text layer, defined by DjVuLibre's DTD                                                                                                                 | DjVuLibre `djvutoxml`; the Internet Archive's `_djvu.xml` files                                             | text-lines, the service                                                            | `ocr:djvu`                                                                                                |
| Tesseract TSV              | Tesseract's tab-separated output (`tesseract … tsv`, `pytesseract.image_to_data`): 12 fixed columns                                                                                         | Tesseract                                                                                                   | text-lines, the service                                                            | `ocr:tesseract`                                                                                           |
| OCR / Doc-AI JSON          | no common standard: every service has its own schema                                                                                                                                        | Azure AI Document Intelligence, Google Cloud Vision / Document AI, AWS Textract, docTR, pero-ocr, OCR.space | json-keys (`page_split.py`, Step 3), text-lines, the service                       | `ocr:generic`                                                                                             |
| PDF with an OCR text layer | PDF (ISO 32000); the recognised text is drawn invisibly (text render mode 3) over the scanned page image                                                                                    | ABBYY FineReader, OCRmyPDF (Tesseract), Adobe Acrobat, scanner software                                     | text-lines, the service                                                            | `ocr:pdf-text-layer`, decided per document (§5)                                                           |
| TEI / TEITOK               | TEI P5 Guidelines (TEI Consortium, `http://www.tei-c.org/ns/1.0`); TEITOK is the TEI-based format of the TEITOK corpus platform, usually written without the namespace                      | atrium-nlp-enrich (TEITOK format 2), flexiconv, the TEITOK platform, TEI editors, Transkribus TEI export    | text-lines, the service                                                            | `ocr:generic` (TEI does not say how the text was made)                                                    |

What each of them records, and what this repo keeps:

* **ALTO.** `Layout/Page` (`PHYSICAL_IMG_NR`, `WIDTH`, `HEIGHT`) → `PrintSpace` → `TextBlock` /
  `ComposedBlock` → `TextLine` → `String` (`CONTENT`), with `SP` for spaces and `HYP` for a line-end
  hyphen; `Illustration` and `GraphicalElement` mark regions without text. Every element has
  `HPOS`/`VPOS`/`WIDTH`/`HEIGHT`, in the unit of `Description/MeasurementUnit` (`pixel`, `mm10` =
  1/10 mm, `inch1200` = 1/1200 inch). Typography sits in `Styles/TextStyle` (`FONTFAMILY`,
  `FONTSIZE`, `FONTSTYLE`), referenced by `STYLEREFS`. Confidence is in `WC` (word) and `CC`
  (characters), and the full form of a hyphenated word in `SUBS_CONTENT`. The ALTO methods split the
  file into one ALTO per page with its header (Step 1), extract the text with LayoutReader (reading
  order from the boxes, the full form for split words), alto-tools (ALTO order) or GLM (which reads
  the page image again), and count lines, strings, illustrations and graphics per page (Step 2).
  text-lines keeps each `TextLine`'s strings in document order. `page_split.py` splits every ALTO
  version: it takes the namespace from the root (it used to know only `ns-v3#`, so a v2/v4 file got
  no pages, until #31 Phase 5), and skips a root that is not a namespaced `<alto>` with its name.
  The engine is named in `Description/OCRProcessing/…/processingSoftware/softwareName`; text-lines
  records a known one as the origin (§6). Tesseract numbers its ALTO pages from 0
  (`PHYSICAL_IMG_NR="0"`), so the ALTO methods' page ids start at 0 for its files.
* **PAGE XML.** `PcGts/Page` (`imageFilename`, `imageWidth`, `imageHeight`) → `ReadingOrder`
  (`OrderedGroup` / `UnorderedGroup` of `RegionRefIndexed`) and the regions (`TextRegion`,
  `TableRegion` with its cells, `ImageRegion`, `GraphicRegion`, …) → `TextLine` (a `Coords@points`
  polygon and a `Baseline`) → `Word` → `Glyph`. Text is in `TextEquiv/Unicode` on every level, with
  alternatives ranked by `@index` and a `@conf`. Typography is in `TextStyle`. Kept: each
  `TextLine`'s first `TextEquiv` (the lowest `@index`), else its `Word`s, in reading order (§2); the
  page label is the image file's stem.
* **hOCR.** Ordinary HTML whose elements carry OCR classes: `ocr_page` → `ocr_carea` → `ocr_par` →
  `ocr_line` (also `ocr_caption`, `ocr_header`, `ocr_textfloat`) → `ocrx_word`. Geometry and metadata
  are properties in the `title` attribute: `bbox x0 y0 x1 y1` in image pixels, `image`, `ppageno`
  (0-based), `baseline`, `x_wconf` (word confidence), `x_font` / `x_fsize`; the engine is in
  `<meta name="ocr-system">`. Kept: the text of each line element; the page label is `ppageno` + 1;
  a known engine is the origin (§6).
* **ABBYY FineReader XML.** `document` → `page` (`width`, `height`, `resolution`) → `block`
  (`blockType` Text, Table, Picture, …, with an `l`/`t`/`r`/`b` box) → `text` → `par` → `line`
  (`baseline` and a box) → `formatting` (language, font and style) → `charParams`, one per
  character, with its box, `wordStart` and `charConfidence`. Kept: each line's characters
  concatenated, a missing space restored at `wordStart`, a line-final `¬` read as a hyphen; tables
  row by row.
* **DjVuXML.** `DjVuXML/BODY/OBJECT`, one per page (`width`, `height`, `PARAM name="PAGE"` naming the
  page file) → `HIDDENTEXT` → `PAGECOLUMN` → `REGION` → `PARAGRAPH` → `LINE` → `WORD` with `coords`
  in page pixels. There is no typography or confidence. Kept: each `LINE`'s words joined by spaces
  (a page without lines: its paragraphs); the label comes from the `PAGE` parameter or `usemap`.
* **Tesseract TSV.** One row per element: `level` (1 page, 2 block, 3 paragraph, 4 line, 5 word),
  `page_num`, `block_num`, `par_num`, `line_num`, `word_num`, the box `left`/`top`/`width`/`height` in
  image pixels, `conf` (0–100 for words, −1 otherwise) and `text`. Kept: the level-5 words of each
  (block, paragraph, line), joined by spaces; every `page_num` is a page, a blank one too.
  `data_samples/TEXT/CTX000000025.tsv` and `CTX000000026.hocr` are real Tesseract 5.3.4 output of two
  rendered Czech pages (`tools/make_tesseract_samples.py`); its TSV, hOCR, ALTO and searchable PDF of
  those pages all read, line for line, as Tesseract's own text output (#31 Phase 5).
* **OCR / Doc-AI JSON.** Each service nests pages, blocks, lines and words its own way and gives
  coordinates in its own shape. Azure: `pages[].lines[].content` with a `polygon` in the page's
  `unit`. Textract: a flat `Blocks` list typed `PAGE`/`LINE`/`WORD` with a relative
  `Geometry.BoundingBox`. docTR: `pages[].blocks[].lines[].words[]` with a relative `geometry`. Google
  Vision: `fullTextAnnotation.pages[].blocks[].paragraphs[].words[].symbols[]`. Kept: json-keys
  splits pages by the page-list and page-tag heuristics of Step 1; both methods then take each text
  of the page once, at the line (JSON granularity, §2) — json-keys read the whole page file, header
  and words included, until #31 Phase 5. Checked on four real AWS Textract responses (1–2 pages,
  28–74 lines) and Microsoft's Azure `GetAnalyzeDocumentResult` example: both methods give exactly
  the engine's `LINE`s, page by page.
* **PDF text layer.** The page's content stream places each character at a position in some font.
  Its Unicode value comes from the font's encoding, usually a `/ToUnicode` map; a subset font
  without one gives the `garbled` class. The printed page numbers are in `/PageLabels`. Kept: the
  text-layer lines in PDFium's order, the page label, and a class per page (`none` / `garbled` /
  `ocr` / `digital`, §5). Positions and fonts are not kept. PDFium reports a hyphen that ends a line
  as `\x02` (or U+FFFE) and drops the line break after it; the reader restores both (`-` and the
  break), so a hyphenated line stays two lines.
* **TEI / TEITOK.** TEI marks the structure (`<text>`, `<div>`, `<p>`, `<head>`, `<l>`), page and
  line breaks (`<pb/>`, `<lb/>`, often with `@facs`), editorial choices (`<choice>` with `sic`/`corr`,
  `orig`/`reg`, `abbr`/`expan`) and typography (`<hi rend>`). TEITOK adds tokens (`<tok>` with their
  annotations as attributes, `<dtok>` for the parts of a multi-word token), sentences (`<s>`), named
  entities (`<name>`) and page geometry (`bbox` on `pb`/`lb`/`tok`, `<facsimile>` surfaces). ATRIUM's
  TEITOK is atrium-nlp-enrich's format 2 (`teitok-2`), described in that repository's README (section
  "TEITOK XML — Unified Output Format") and checked against its `schemas/teitok`. Kept: the text
  between `<pb/>`s as pages (label `pb@n`), lines at `<lb/>` and block ends. In tokenized TEITOK a
  sentence is not a line. `<choice>` gives its edited reading. The header, `facsimile`, `standOff` and
  `sourceDoc` are skipped, and token annotations and boxes are not kept. A TEITOK file that
  nlp-enrich wrote is read back into the pages nlp-enrich's own readers count.

**Where the coordinates go.** This repo writes text only. The page geometry that a TEITOK file shows
over the page image comes from atrium-nlp-enrich. Its writer reads the ALTO file itself
(`INPUT_ALTO_DIR`) or a flexiconv conversion of ALTO, PAGE XML or hOCR (`api_flexiconv.sh`). The
boxes of ABBYY FineReader XML, DjVuXML, Tesseract TSV and OCR JSON reach no ATRIUM output. When a
TEITOK view over the page image is wanted, export ALTO, PAGE XML or hOCR from the engine as well.

### Text-bearing documents

| Format            | Standard (steward)                                                                        | Pages in the format                                                                                                         | Typical producers                                                        | `source.origin`                         |
|-------------------|-------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|-----------------------------------------|
| DOCX, XLSX, PPTX  | Office Open XML: ECMA-376, also ISO/IEC 29500 (Ecma TC45)                                 | DOCX stores none: explicit breaks, plus the breaks Word rendered when it last saved the file (§2); XLSX sheets; PPTX slides | Microsoft Office, LibreOffice, Google Docs exports, OCR engines' exports | `digital-born-docx` / `-xlsx` / `-pptx` |
| ODT, ODS, ODP     | OpenDocument Format (OASIS; ODF 1.2 is ISO/IEC 26300)                                     | ODT stores none: page-break styles, plus `text:soft-page-break` where LibreOffice last laid it out; sheets; slides          | LibreOffice, Apache OpenOffice, Google Docs exports                      | `digital-born-odt` / `-ods` / `-odp`    |
| EPUB              | EPUB 3 (W3C, formerly IDPF): a ZIP whose package document lists XHTML chapters in a spine | none; each spine chapter is a page                                                                                          | publishing tools, Calibre, Sigil                                         | `digital-born-epub`                     |
| RTF               | Rich Text Format Specification 1.9.1 (Microsoft)                                          | `\page`                                                                                                                     | Word, WordPad, TextEdit                                                  | `digital-born-rtf`                      |
| HTML / XHTML      | HTML Living Standard (WHATWG); XHTML                                                      | only CSS page breaks                                                                                                        | "save as web page", CMS exports                                          | `digital-born-html`                     |
| Markdown          | CommonMark; tables as in GitHub Flavored Markdown                                         | form feeds                                                                                                                  | editors, pandoc                                                          | `ocr:generic`                           |
| CSV / TSV         | RFC 4180 (CSV); the IANA `text/tab-separated-values` registration                         | a `page` / `page_num` column                                                                                                | spreadsheets, transcription and annotation tools                         | `ocr:generic`                           |
| JSON / JSON Lines | RFC 8259; JSON Lines (one value per line)                                                 | a page list or page tag, else top-level children / records                                                                  | APIs, annotation tools                                                   | `ocr:generic`                           |
| Other XML         | any vocabulary without a reader of its own                                                | root children                                                                                                               | —                                                                        | `ocr:generic`                           |
| Plain text        | Unicode (UTF-8/16/32) or a legacy code page (`FALLBACK_ENCODINGS`, §3)                    | form feeds, as `pdftotext` and Tesseract write them between pages                                                           | any                                                                      | `ocr:generic`                           |
| SRT / WebVTT      | SubRip (de facto); WebVTT (W3C)                                                           | none                                                                                                                        | subtitle editors, speech recognition                                     | `ocr:generic`                           |
| EML / MBOX        | Internet Message Format (RFC 5322) with MIME (RFC 2045–2049); mbox (RFC 4155)             | one message per page                                                                                                        | mail clients and archives                                                | `digital-born-eml` / `-mbox`            |
| ZIP bundle        | ZIP (PKWARE APPNOTE) of per-page files                                                    | the member files                                                                                                            | Transkribus (`page/*.xml` with `mets.xml`), eScriptorium (ALTO or PAGE)  | its members'                            |
| gzip, bzip2, xz   | gzip (RFC 1952), bzip2, xz: one compressed file                                           | the inner file's                                                                                                            | any                                                                      | the inner file's                        |

`ocr:generic` stands for text whose making the file does not record: it may be OCR output, a
transcription or typed text. Set a truthful origin per kind when it is known (§6). Images, legacy
binary Office files (`.doc`, `.xls`, `.ppt`) and other archives (tar, 7z, RAR, zstd) are refused with
a reason (§1, §4). An image needs OCR first; feed its ALTO, PAGE XML, hOCR, TSV or text output
instead.

## 1. How a file is recognised

The **content** decides. The extension only chooses between plain-text dialects, and it wins over
an unknown XML root and over the JSON probe, so a `README.md` that starts with `<p align="center">`
is still Markdown. The tests run in this order:

| Order | Test                                                                                                                                                                                                                                          | Result                                                                                                                                                                                     |
|-------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | empty file                                                                                                                                                                                                                                    | refused `empty_file`                                                                                                                                                                       |
| 2     | gzip / bzip2 / xz signature                                                                                                                                                                                                                   | decompressed (bounded, §4) and detected again under the inner name (`a.txt.gz` → `a.txt`); inside it a ZIP, archive, PDF or another compressed stream is refused `archive_unsupported`     |
| 3     | `%PDF-` at the start (after a BOM or whitespace)                                                                                                                                                                                              | **pdf**                                                                                                                                                                                    |
| 4     | OLE2 signature `D0 CF 11 E0`                                                                                                                                                                                                                  | refused `legacy_office_unsupported` (.doc/.xls/.ppt)                                                                                                                                       |
| 5     | ZIP signature → `mimetype` member (ODF/EPUB) or `[Content_Types].xml` (OOXML)                                                                                                                                                                 | **odt/ods/odp/epub** or **docx/xlsx/pptx**                                                                                                                                                 |
| 6     | any other ZIP with page files (§2, ZIP bundle)                                                                                                                                                                                                | **zip-bundle**: one document; a ZIP of images is `image_needs_ocr`, of PDFs or containers only `archive_unsupported`, an office/EPUB package without its type marker `archive_unsupported` |
| 7     | `%PDF-` elsewhere in the first 1 KiB, unless a text extension holds text bytes (a note that quotes a PDF header)                                                                                                                              | **pdf**                                                                                                                                                                                    |
| 8     | PNG/JPEG/GIF/TIFF/WebP/JPEG 2000 signature                                                                                                                                                                                                    | refused `image_needs_ocr`                                                                                                                                                                  |
| 9     | tar (`ustar`, or a `.tar` inner name), 7z, RAR, zstd                                                                                                                                                                                          | refused `archive_unsupported`                                                                                                                                                              |
| 10    | `{\rtf` (after a BOM or whitespace)                                                                                                                                                                                                           | **rtf**                                                                                                                                                                                    |
| 11    | NUL/control-heavy bytes that are not UTF-16                                                                                                                                                                                                   | refused `binary_content`                                                                                                                                                                   |
| 12    | starts with `<`: root `alto` / `PcGts` / `TEI`, `teiCorpus`, `TEI.2` / `html` (hOCR if `ocr_page`/`ocr_line` classes) / ABBYY `document` / `DjVuXML`                                                                                          | **alto / page-xml / tei / hocr / html / abbyy-xml / djvu-xml**; any other root: `.html`/`.htm`/`.xhtml`/`.hocr` → **html**, a dialect extension (row 16) → that dialect, else **xml**      |
| 13    | `WEBVTT`                                                                                                                                                                                                                                      | **vtt**                                                                                                                                                                                    |
| 14    | a first line equal to Tesseract's 12 TSV columns (`level … conf text`)                                                                                                                                                                        | **tesseract-tsv**                                                                                                                                                                          |
| 15    | `.jsonl`/`.ndjson` → **jsonl**; `.json` → **json**                                                                                                                                                                                            | by extension                                                                                                                                                                               |
| 16    | `.md`/`.markdown`/`.mdown` → **md**; `.csv` → **csv**; `.tsv`/`.tab` → **tsv**; `.srt` → **srt**; `.vtt` → **vtt**; `.eml` → **eml**; `.mbox`/`.mbx` → **mbox**                                                                               | by extension                                                                                                                                                                               |
| 17    | plain-text or unknown extension: a `From ` separator line → **mbox**; a strict RFC 822 header block (≥ 3 fields, an origin field and a technical one such as `Message-ID`/`MIME-Version`) → **eml**; a cue number and a timing line → **srt** | a typed memo ("From: … To: … Subject: …") stays plain text                                                                                                                                 |
| 18    | other `{`/`[` files                                                                                                                                                                                                                           | **json** if the whole file parses, **jsonl** if its first 50 non-blank lines each parse to an object, array or string, else **txt**                                                        |
| 19    | anything else                                                                                                                                                                                                                                 | **txt**                                                                                                                                                                                    |

A mismatch between extension and content is recorded in the ingest report's `notes`, for example
`extension .txt but content is pdf`, and the file is still read by its content. A `.txt` that merely
starts with `<` and is not well-formed XML is read as plain text (noted
`xml_parse_failed_read_as_text`), not recovered as XML.

## 2. What a page and a line are, per format

| Kind                 | Page (block)                                                                                                                                                                                                | Line                                                                                                                                                                                                                                                                          | Library        |
|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| TXT                  | form-feed (`\f`) section; otherwise the whole file                                                                                                                                                          | physical line                                                                                                                                                                                                                                                                 | stdlib         |
| Markdown             | form-feed section                                                                                                                                                                                           | physical line, with front matter, code fences, comments, rules and link definitions dropped; heading/list/quote markers and inline markup stripped; table rows become tab-joined cells                                                                                        | stdlib         |
| CSV / TSV            | the file; or groups of a `page`/`page_num`/`page_number`/`pagenumber`/`page_no` column, in first-seen order                                                                                                 | the first non-numeric column of `text`/`line`/`content`/`transcription`/`sentence`/`string`/`word`/`token`; rows sharing a numeric (page, block, paragraph, line) address are joined by spaces (one word per row); without a text column, all non-empty cells joined by tab   | stdlib `csv`   |
| Tesseract TSV        | `page_num` (every page, a blank one too)                                                                                                                                                                    | the level-5 words of one (block, par, line), joined by spaces                                                                                                                                                                                                                 | stdlib `csv`   |
| JSON                 | page list (`pages`, …) or page-tagged list (`Page`, `pageNumber`, …), the same detection as json-keys; otherwise each top-level child object/list that holds text; a top-level array of strings is one page | string leaf under a text key (`content`, `text`, `line`, …), at line granularity (below); when none exist, every string containing a letter (noted `json_all_strings`)                                                                                                        | stdlib         |
| JSON Lines           | record                                                                                                                                                                                                      | as JSON                                                                                                                                                                                                                                                                       | stdlib         |
| ALTO v2/v3/v4        | `Page` (label `PHYSICAL_IMG_NR`)                                                                                                                                                                            | `TextLine`: `String@CONTENT` joined by spaces, `HYP` → `-`                                                                                                                                                                                                                    | lxml           |
| PAGE XML             | `Page` (label = image file stem)                                                                                                                                                                            | `TextLine` in `ReadingOrder` region order, every region kind: `TableRegion` cells row by row (Transkribus `TableCell@row/col`, PAGE 2019 `TableCellRole`), a nested region the ReadingOrder does not name read inside its parent; first `TextEquiv/Unicode`, else its `Word`s | lxml           |
| hOCR                 | `ocr_page` (label from `ppageno`, 1-based)                                                                                                                                                                  | `ocr_line`, `ocrx_line`, `ocr_caption`, `ocr_header`, `ocr_textfloat`                                                                                                                                                                                                         | lxml.html      |
| ABBYY FineReader XML | `page` (every page)                                                                                                                                                                                         | `line`: its `charParams` concatenated (a missing space is restored at `wordStart`, a final `¬` is a hyphen); table blocks row by row                                                                                                                                          | lxml           |
| DjVuXML              | `OBJECT` (label from its `PAGE` param or `usemap`)                                                                                                                                                          | `LINE`: its `WORD`s joined by spaces; `PARAGRAPH`s when a page has no `LINE`                                                                                                                                                                                                  | lxml           |
| HTML / XHTML         | CSS `page-break-before/after` (`break-before: page`); otherwise one page                                                                                                                                    | block element (`p`, `h1`–`h6`, `li`, `td`, …) or `<br>`; `script`/`style`/`head` dropped                                                                                                                                                                                      | lxml.html      |
| TEI / TEITOK         | `<pb/>`, every one, a blank page too (label `pb@n`, else its number); a `teiCorpus` text after text                                                                                                         | `<lb/>` and block ends (`p`, `head`, `l`, `item`, `cell`, `ab`, …); `<s>` too, except in tokenized TEITOK (`<tok>` + `<lb/>`), where a sentence is not a line; in `<choice>` the `corr`/`reg`/`expan` reading; `teiHeader` skipped                                            | lxml           |
| other XML            | root child elements, when two or more hold text (not when they are all line elements)                                                                                                                       | an element with its own (mixed) text; an element named `line`/`textline` is one line of its word or character children; pure containers are descended into                                                                                                                    | lxml           |
| PDF                  | PDF page (label = the PDF's page label, else its number)                                                                                                                                                    | text-layer line, in PDFium's order                                                                                                                                                                                                                                            | **pypdfium2**  |
| DOCX                 | explicit page break, `pageBreakBefore`, non-continuous section break; in `auto` mode also Word's rendered breaks                                                                                            | paragraph (`w:br`/`w:cr` split it); table cells row by row; text boxes after their anchor paragraph; footnotes and endnotes per `NOTES` (below)                                                                                                                               | zipfile + lxml |
| ODT                  | `fo:break-before/after="page"` paragraph styles; in `auto` mode also `text:soft-page-break`                                                                                                                 | paragraph / heading (`text:line-break` splits it); table cells; `text:note` per `NOTES`                                                                                                                                                                                       | zipfile + lxml |
| XLSX / ODS           | sheet, in workbook order (label = sheet name); hidden sheets too (counted)                                                                                                                                  | row: its text cells joined by tab; numbers, dates, booleans (and ODS comments) dropped                                                                                                                                                                                        | zipfile + lxml |
| PPTX / ODP           | slide, in presentation order; hidden slides too (counted)                                                                                                                                                   | paragraph                                                                                                                                                                                                                                                                     | zipfile + lxml |
| EPUB                 | spine chapter (label = file stem)                                                                                                                                                                           | as HTML                                                                                                                                                                                                                                                                       | zipfile + lxml |
| RTF                  | `\page`                                                                                                                                                                                                     | `\par`, `\line`, `\row`; `\cell` → tab; header/footer/footnote/picture/field-instruction groups skipped; `\'xx` decoded with the current font's `\fcharsetN`/`\cpgN`, else `\ansicpgN`; `\u` surrogate pairs combined                                                         | stdlib         |
| SRT / WebVTT         | the file                                                                                                                                                                                                    | cue text line: cue numbers, identifiers, timings, the `WEBVTT` header, `NOTE`/`STYLE`/`REGION` blocks and inline markup (`<i>`, `<c.x>`, `<v Speaker>`, `{\an8}`) dropped                                                                                                     | stdlib         |
| EML / MBOX           | message (MBOX: one page per message)                                                                                                                                                                        | the decoded Subject, then the `text/plain` body (`text/html` through the HTML reader when there is no plain part); other headers and attachments are not read (attachments counted)                                                                                           | stdlib `email` |
| ZIP bundle           | the member files in natural path order (`1`, `2`, …, `10`), each read by content with the readers above; their pages concatenated                                                                           | the member format's line                                                                                                                                                                                                                                                      | zipfile        |

**JSON granularity.** An engine often gives the same text at several levels, and text-lines reads it
once, at the line: a dict with a line container (`lines`) skips its word containers and its own coarse
text (a page's or paragraph's full `text`/`content`, also when the lines sit deeper below it); a line
object with its own text skips the words below it (Azure Read); a line object without its own text
becomes one line of its words (docTR); a flat list typed LINE and WORD (AWS Textract `Blocks`) keeps
the LINE items. Base64, hex digests and other embedded binary strings are skipped. Each is counted in
the notes (`json_word_leaves_skipped`, `json_coarse_leaves_skipped`, `json_blobs_skipped`,
`json_non_text_strings`).

**ZIP bundles** are how a folder of per-page files (a Transkribus or eScriptorium export, one PAGE XML
or TXT per scan) becomes one multi-page document. Export metadata (`mets.xml`, `doc.xml`,
`metadata.xml`, METS/OPC/manifest roots), `__MACOSX/`, hidden files, images, READMEs and unknown types
are ignored. When one page is exported in several formats (`page/0001.xml` and `alto/0001.xml`),
the first of PAGE XML, ALTO, ABBYY, hOCR, DjVu, Tesseract TSV, TEI, JSON, … TXT is read and the others
are counted. PDFs and nested containers are not read inside a bundle (a PDF is only read in its
isolated process); a member that fails is skipped and named in the notes, and the document is
`partial`. Page labels are the member paths relative to their common folder, without the extension
(`0001`, or `0001/2` for the second page of a multi-page member). Member names are never used as
paths, so `../` entries cannot escape.

Pages are written as `<doc>-1 … <doc>-N` in reading order. The original label (sheet name, PDF
page label, JSON page number, bundle member …) is kept in `pages_report.csv` and in the line table's
`page_label` column. Blank pages:
* every page of a format with real pages is written, blank ones included: PDF, ALTO, PAGE XML, hOCR,
  ABBYY, DjVu, Tesseract TSV, every TEI `<pb/>`, every slide. Page numbers stay faithful to the source.
* a blank form-feed section, or a blank DOCX/ODT page between two breaks, is kept; a trailing one is
  not, and there is never a blank first page before the first break.
* a block without text (a sheet with only numbers, a JSON child or JSONL record with only ids, an
  EPUB chapter or e-mail without text) is not a page.

**Page ids.** text-lines numbers pages 1..N in reading order, for every format. The ALTO and JSON
methods (`page_split.py`) name ALTO pages by `PHYSICAL_IMG_NR` and JSON pages by their page number.
So the same ALTO/JSON file read through text-lines can get other `page_num`s than through its own
method, while the source's label is kept in `page_label` (a Tesseract ALTO: `page_num` 1, 2, … in
text-lines, page ids 0, 1, … in the ALTO methods). nlp-enrich takes its pages from the layout
when it has one, numbered by their order in the ALTO file, and else from these tables' `page_num`,
so read one document through one method. When an ALTO file's `PHYSICAL_IMG_NR`s are not 1, 2, 3 … in
order, the ALTO methods' `page_num`s, which also key the record's `pages[]`, differ from the TEITOK
page numbers (atrium-nlp-enrich README, Pitfalls 14). `DOC_LINE_CATEG` has no `page_label` column.

**DOCX/ODT pagination** is set by `[TEXT_INGEST].PAGE_BREAKS`:
* `auto` (default): explicit breaks plus the breaks Word or LibreOffice rendered when the file was
  last saved. A rendered break right after an explicit one opens no extra page. A rendered break
  in the middle of a paragraph splits it where the page really ended.
* `explicit`: only the breaks the author inserted.
* `none`: one page.

Page breaks inherited through the style hierarchy are ignored, for DOCX and ODT alike (only direct
formatting counts). Deleted revisions and the VML fallback copies of text boxes are skipped.

**Footnotes and endnotes** follow `[TEXT_INGEST].NOTES` (DOCX `footnotes.xml`/`endnotes.xml`, ODT
`text:note`):
* `page` (default): a footnote's lines end the page that references it, as in print; endnotes end the
  document.
* `end`: every note at the end of the document, in reference order.
* `skip`: not read (counted, `notes_not_read=N`).

Word's separator notes are recognised by their `w:type`, not by their conventional ids. Headers and
footers are not read; the parts that hold text are counted (`headers_footers_not_read=N`).

## 3. Normalisation (the line invariant)

`text_split.py` writes each line after `normalize_line()`:
* every line separator except the page break becomes `\n` (`\r\n`, `\r`, `\v`, `\x1c–\x1e`, U+0085,
  U+2028/2029). `classify_TEXT` splits with `readlines()`, so any separator left inside a line would
  shift line numbers.
* NFC; the ligatures U+FB00–FB06 are expanded.
* zero-width, BOM and bidi controls are removed (ZWJ/ZWNJ kept); NBSP-like spaces become spaces.
* C0/C1 control characters except tab, and lone UTF-16 surrogates, are removed (a surrogate used to
  fail the whole document's page write; now it is counted, `lone_surrogates_dropped=N`).
* a soft hyphen, or PDFium's `\x02` hyphen marker, at the end of a line becomes `-`, and is dropped
  anywhere else.
* the line is stripped.

`extract_TEXT_2_TXT.py`, and the service, then apply `shape_lines()`:
* blank lines are dropped (`KEEP_BLANK_LINES = false`), because `classify_TEXT` would score each one
  as an `Empty` row.
* lines over `MAX_LINE_CHARS` (1000) are wrapped at a word boundary, and hard-split when there is
  none. The perplexity batch pads every line to the longest one in the batch, so one 30,000-character
  paragraph would cost the whole batch.

The line table is built from the same list that is written to the page file, so for every row
`readlines()[line_num - 1] == text`, exactly the numbering `DOC_LINE_CATEG` will use.

Plain text is decoded in this order (`FALLBACK_ENCODINGS` defaults to `cp1250, iso8859_2, cp1252`):
1. a BOM (UTF-8/16/32);
2. when the first 64 KiB hold NUL bytes: UTF-16 without a BOM, recognised by its NUL pattern, else
   refused `binary_content`;
3. strict UTF-8;
4. a binary check (`binary_content`);
5. charset-normalizer, restricted to `FALLBACK_ENCODINGS`;
6. the fallbacks in order;
7. the first fallback with replacement characters (noted `decode_replacement`: `partial`).

Restricting detection to plausible code pages keeps a short Czech file from being "detected" as an
unrelated one and turned into mojibake, which the categorizer would then score as damaged OCR. The
encoding used is reported per file. The stats stage decodes page files with the same
`FALLBACK_ENCODINGS`.

## 4. Limits and failure handling

Each file is processed on its own, and a failure costs only that file. It gets a row in
`ingest_report.csv` with its reason code, a `skipped_files_detail` entry in paradata, and the run
continues. A file that was read but may have lost text is `partial` (below): written and processed
downstream, its reasons listed. `--strict` (or `STRICT = true`) makes `text_split.py` exit 1 if any
file failed or was read partially, and `extract_TEXT_2_TXT.py` exit 1 if a page or a document record
failed; `--no-strict` overrides the config, and `run_pipeline.py` passes either flag through. A
strict failure stops the pipeline after that stage.

| `[TEXT_INGEST]` key  | Default | Guards against                                                                                                                                |
|----------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `MAX_FILE_MB`        | 256     | oversized files (`too_large`); also the most a gzip/bzip2/xz file may decompress to                                                           |
| `ZIP_MAX_MEMBERS`    | 10000   | ZIP containers and bundles with absurd member counts (`zip_limits_exceeded`)                                                                  |
| `ZIP_MAX_TOTAL_MB`   | 1024    | zip bombs: declared unpacked size, checked **before** anything is read; also the text a spreadsheet may expand to                             |
| `ZIP_MAX_MEMBER_MB`  | 256     | one huge member                                                                                                                               |
| `ZIP_MAX_RATIO`      | 200     | compression ratio of ZIP members over 1 MiB, and of a gzip/bzip2/xz stream (`zip_limits_exceeded`)                                            |
| `MAX_PAGES`          | 20000   | page-count bombs (`too_large`)                                                                                                                |
| `MAX_LINES_PER_PAGE` | 100000  | a real page over it is refused; a block over it (a sheet, a JSON child, a form-feed section) continues on pages labelled `<label>+1`, `+2`, … |
| `READER_TIMEOUT_S`   | 300     | a PDF that hangs PDFium. PDFs are read in a **separate process**, so a crash or hang costs one file (`timeout`, `reader_crashed`)             |

XML is parsed without entity resolution, DTD loading or network access, and with libxml2's size and
depth limits on. A document that declares entities is refused (`xml_entity_declaration`). Broken XML
gets one retry in recovery mode, noted `xml_recovered` (`partial`) — except the `<name>…</n>` quirk
of older TEITOK exports, which is repaired exactly (noted `name_close_repaired`) because recovery may
drop text. Encrypted inputs are refused (`encrypted`): password PDFs, ZIP members with the encryption
flag, ODF with `encryption-data`, and DRM-protected EPUB chapters (EPUB font obfuscation alone is
fine).

Input discovery is conservative. Only regular files at the top level of the input directory are
read. Symbolic links, FIFOs/devices, subdirectories, hidden files, Office lock files (`~$…`,
`.~lock.…#`) and OS metadata (`._*`, `Thumbs.db`, `desktop.ini`) are listed as `ignored`.
Document ids come from `canonical_doc_id()` (hub-shared), so `report.v2.pdf` and `report.txt.gz`
become `report`. Two files that map to the same id (case-insensitively) are a `doc_id_collision`: the
first in sorted order wins and the other is refused, never silently overwritten.

**Output directories.** Each document's pages are written into `.tmp-<doc>/`, the previous `<doc>/` is
renamed aside to `.old-<doc>/`, the new one renamed in, and the old one removed; the document record
is written before the pages are swapped in. A POSIX directory swap is not atomic, so a concurrent
reader can briefly miss `<doc>/`; the stage assumes it is the only writer. text_split only ever
replaces or removes a directory that holds nothing but that document's `<doc>-<n>.txt` files: an
input named `setup.txt` next to a `setup/` folder is refused (`output_failed`), never deleted. When a
document fails, its pages from an earlier run are removed (noted `stale_pages_removed`), so the later
stages never read pages its report calls failed. Pages of an input that was deleted from the input
directory, and of a collision's loser, are left as they are. A page whose text did not change keeps its file time
through the swap, and `extract_TEXT_2_TXT.py` rewrites a page only when its text changed: the extract and
classify stages resume by file time (an output is current while it is at least as new as its inputs), so a
full re-run over unchanged inputs skips them again, and a changed page is processed again (#31 Phase 5).

### Reports

`ingest_report.csv`, one row per input file:

| Column                                              | Content                                                                                             |
|-----------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| `filename`, `doc_id`                                | the input file and its document id                                                                  |
| `status`                                            | `ok`; `partial` (read, with a possible loss); `error` (refused or failed); `ignored` (not read, §4) |
| `reason`                                            | `error`: the reason code; `partial`: the lossy notes, `;`-joined; `ignored`: why, in words          |
| `kind`, `media_type`, `encoding`                    | the detected format (§1), its media type, the text encoding used                                    |
| `pages`, `lines`, `chars`                           | counts of the pages written and their non-blank lines and characters                                |
| `pages_no_text`, `pages_garbled`, `pages_ocr_layer` | pages without a text line; PDF pages whose text layer is `garbled` / `ocr`                          |
| `origin`, `sha256`                                  | the `source.origin` recorded (§6) and the file's SHA-256                                            |
| `notes`                                             | the reader's notes and the stage's own, `; `-joined                                                 |

The lossy notes are `xml_recovered`, `decode_replacement`, `jsonl_bad_records=N`,
`csv_unbalanced_quote` (a stray `"`: re-read without quoting), `tsv_bad_rows=N`,
`xlsx_bad_shared_string`, `sheet_repeat_capped` (an ODS repeat over 100), `zip_members_skipped=N`,
`epub_spine_skipped=N`, `email_bad_messages=N`, `lone_surrogates_dropped=N`, and the page flag
`page_load_failed`. Everything else in `notes` is informational (`encoding_detected`,
`decompressed:gzip`, `bundle_members=N`, `docx_footnotes=N`, `xlsx_hidden_sheets=N`, …).

`pages_report.csv`, one row per written page: `file`, `page` (1..N), `page_label`, `text_layer` and
`needs_ocr_reason` (PDF, §5), `lines` (non-blank), `images` (PDF image objects) and `flags`
(`;`-joined, §5).

### Reason codes

The HTTP column is the status the service answers with (§7).

| Code                                  | Meaning                                                                                                                          | HTTP |
|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|------|
| `empty_file`                          | zero bytes (also an empty compressed stream)                                                                                     | 422  |
| `too_large`                           | a size/page/line cap was exceeded                                                                                                | 422  |
| `binary_content`                      | not text and not a supported container                                                                                           | 400  |
| `legacy_office_unsupported`           | OLE2 `.doc/.xls/.ppt`; save it as DOCX/XLSX/PPTX                                                                                 | 400  |
| `image_needs_ocr`                     | an image (or a ZIP of images); run OCR first and feed its output (ALTO, PAGE XML, hOCR, Tesseract TSV, TXT)                      | 400  |
| `archive_unsupported`                 | an archive this repo does not read: tar, 7z, RAR, zstd, a ZIP without page files, a container inside a compressed file or bundle | 400  |
| `zip_limits_exceeded`                 | ZIP or decompression caps above                                                                                                  | 422  |
| `xml_entity_declaration`              | XML with `<!ENTITY>`                                                                                                             | 422  |
| `malformed`                           | broken for its format (bad JSON, unparseable XML/CSV, too deeply nested)                                                         | 422  |
| `corrupt`                             | the container or compressed stream could not be opened                                                                           | 422  |
| `encrypted`                           | password- or DRM-protected                                                                                                       | 422  |
| `timeout` / `reader_crashed`          | the isolated PDF reader hung or died                                                                                             | 422  |
| `dependency_missing`                  | `pypdfium2` (PDF) or `lxml` not installed                                                                                        | 400  |
| `decode_failed`                       | no configured encoding decodes the text                                                                                          | 422  |
| `no_text`                             | read fine, but no text lines. For a PDF: no text layer on any page, so run OCR first                                             | 422  |
| `unreadable`                          | the file could not be opened or read (permissions, an I/O error)                                                                 | 422  |
| `doc_id_collision` / `doc_id_invalid` | see above                                                                                                                        | —    |
| `output_failed`                       | writing the page files or the document record failed, or the page directory is not text_split's                                  | —    |

## 5. PDF text layers and page flags

`pages_report.csv` classifies every PDF page, using the thresholds of llm-enrich's `pdf_to_md`:

| `text_layer` | Test                                                                                        | Meaning                                                       |
|--------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------|
| `none`       | fewer than `PDF_MIN_TEXT_CHARS` (3) visible characters                                      | an image-only page: needs OCR                                 |
| `garbled`    | more than `PDF_GARBLE_THRESHOLD` (15%) U+FFFD / control / format / private-use / unassigned | a subset font without `/ToUnicode`: the layer does not decode |
| `ocr`        | at least `PDF_OCR_LAYER_MIN_RATIO` (50%) of the text objects are invisible (render mode 3)  | the classic OCR layer under a scanned image                   |
| `digital`    | anything else                                                                               | born-digital text                                             |

A garbled layer is still extracted, because its lines are exactly what the categorizer is meant to
flag as `Trash`. A document whose text-bearing pages are mostly `ocr` gets `source.origin`
`ocr:pdf-text-layer`; any other PDF gets `digital-born-pdf`. PDFium returns text in content-stream
order, so multi-column pages are not re-ordered (no column detection). That is the same limitation
as llm-enrich's `digital_to_json`. Both ratio thresholds must lie in [0, 1].

The `flags` column reports what the text layer looks like. The flags never set a category, since the
categories stay classify's and the shared vocabulary keeps the tools' category sets apart:

| Flag                                 | Set when                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
|--------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `mojibake_cp1252`                    | at least 20% of a page's lines read like CP1250 Czech decoded as CP1252 (`sondì èíslo`). The confusion table is derived from the two codecs, as in llm-enrich's `digital_to_json`; a line needs two misread characters or under 90% clean letters, a Czech letter both code pages share (á í ú ý š ž), no letter only Western text has (à ê û œ), and a clean CP1252 round trip, so French or Italian text is not flagged. Any kind; the note `mojibake_cp1252_pages=N` counts them |
| `mirrored_text=N` / `rotated_text=N` | PDF: N text objects whose matrix (composed with its form XObjects) mirrors the text (a negative scale; 180° counts as mirrored) or turns it by more than about 1°                                                                                                                                                                                                                                                                                                                   |
| `page_load_failed`                   | PDF: PDFium could not load the page (lossy: `partial`)                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `overflow`                           | a continuation page of a block over `MAX_LINES_PER_PAGE`                                                                                                                                                                                                                                                                                                                                                                                                                            |

## 6. Provenance: `source.origin`

With `[DOCUMENT].JSON_DIR` set, `text_split.py` writes the record's `source`: `sha256`, `filename`,
`media_type`, `origin`, and `page_count` for formats with real pages only (PDF, ALTO, PAGE XML,
hOCR, ABBYY, DjVu, Tesseract TSV, a bundle of such pages). llm-enrich records a DOCX as one page, and
an invented count would conflict in `set_source()`. The origin is **truthful per class**:

| Class                      | Formats                                                                                                                  | Default origin                                                                                                                                                                                              | Positional blocks written by this repo |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| OCR output                 | ALTO, PAGE XML, hOCR, ABBYY FineReader XML, DjVuXML, Tesseract TSV, PDF with an OCR layer; a ZIP bundle of one such kind | `ABBYY-ALTO`, `ocr:page-xml`, `ocr:hocr`, `ocr:abbyy-finereader`, `ocr:djvu`, `ocr:tesseract`, `ocr:pdf-text-layer`; for ALTO and hOCR the engine the file names, when known (below); the bundle's members' | yes                                    |
| text of unknown provenance | TXT, Markdown, CSV/TSV, JSON/JSONL, TEI, other XML, SRT/VTT; a bundle of mixed kinds                                     | `ocr:generic` (as json-keys)                                                                                                                                                                                | yes                                    |
| born-digital               | DOCX, ODT/ODS/ODP, XLSX, PPTX, EPUB, RTF, plain HTML, visible-text PDF, EML/MBOX                                         | `digital-born-<kind>`                                                                                                                                                                                       | **no**: `source` only                  |

A `digital-born-*` origin authorises llm-enrich's **digital-convert** to originate the record's
`pages`/`content`/`lines`/`tables` (`atrium_document` §1a), and the hub's digital end-to-end test
asserts that alto-postprocess does not write them. The shared check only *warns* when it is not
strict. `document_hook.write_document_block()` therefore holds those blocks back for every stage of
this repo, including the unchanged `classify_TEXT` and `aggregate_STAT`, with one warning per
document. The exception is when the record carries digital-convert's own `pages[].needs_ocr`
hand-off, which asks for exactly this repo's pass. The categorized CSV outputs are produced either
way; with a record configured, the ingest report notes `born-digital origin: the document record gets
source only`.

**The engine an OCR file names** (#31 Phase 5). ALTO's
`Description/OCRProcessing/…/processingSoftware/softwareName` (or `softwareCreator`) and hOCR's
`<meta name="ocr-system">` name the engine that wrote the file. When it is one this repo knows —
Tesseract → `ocr:tesseract`, PERO → `ocr:pero`, Kraken → `ocr:kraken`, Transkribus →
`ocr:transkribus`, eScriptorium → `ocr:escriptorium`, OCR-D → `ocr:ocrd`, Calamari → `ocr:calamari`,
ABBYY/FineReader → `ABBYY-ALTO` — that is the default origin; any other name keeps the format's
default. A real Tesseract ALTO used to be recorded as `ABBYY-ALTO`. A ZIP bundle takes its members'
origin when they agree. The ALTO methods (`page_split.py`) keep `ABBYY-ALTO`.

digital-convert reads only PDF and DOCX, so for the other born-digital kinds nothing in the ecosystem
writes the positional plane yet, and the note says so. The hub keeps its `digital-born…` prefix as is,
by decision (2026-09-25): narrowing it would change the §1a contract for all five repos, and the
per-kind override below is the per-site answer. `[DOCUMENT].SOURCE_ORIGIN_BY_KIND` lets an
operator state, per kind, what the files really are:

```ini
[DOCUMENT]
SOURCE_ORIGIN_BY_KIND = xlsx = ocr:generic, pptx = ocr:generic
```

Precedence: `--source-origin` > the `DOCUMENT_SOURCE_ORIGIN` env var > `SOURCE_ORIGIN_BY_KIND` for the
document's kind > `[DOCUMENT].SOURCE_ORIGIN` > the truthful default. The flag and the env var are
per-run statements about every input, so they win over the config; within the config the per-kind
entry is the more specific one. Kinds are those of §1 (`text_formats.READERS`). An unknown kind, a
kind given twice, an empty value, or an origin no originator claims (`ocr:…`, `vlm:…`, `ABBYY-ALTO`,
`digital-born-…`) stops the run with exit 2. **Keep it truthful.** `source.origin` is provenance
that other tools read, so use it when the files of that kind really are OCR or transcription
exports (a spreadsheet of transcribed lines, an OCR engine's DOCX export), not only to unlock the
blocks. A `pdf = …` entry overrides the per-document text-layer test.

## 7. The API service

`POST /process` accepts the same formats. `auto` keeps `.txt` → text and `.json` → json and decides
`.xml` and every other extension from the bytes: an uncompressed ALTO root keeps the ALTO path;
anything else readable, a compressed ALTO included, is a `document`. Documents are read with
`read_document_isolated()` and shaped with `shape_lines()`, the same code as the batch method, and
classified page by page in batches of 128 lines. Each result line carries `page`/`page_label`, and
`line_num` restarts per page. A `.txt` upload keeps its `plain_text` response, but it is decoded and
shaped like the batch path (any common encoding, cp1250 first).

The readers use the config's `[TEXT_INGEST]` settings and `[DOCUMENT].SOURCE_ORIGIN(_BY_KIND)`, read
from `LANGID_CONFIG` once at start-up; a malformed value fails the start. Blank lines are always
dropped here (`KEEP_BLANK_LINES` and `STRICT` are batch-only), and the `DOCUMENT_SOURCE_ORIGIN` env
var is not read. Errors name the reason code: `400` for a file of a kind this service does not read,
`422` for a supported kind that cannot be read, a document without a single text line included
(`no_text`); the HTTP column of §4 lists them. Born-digital uploads do not accrete into a
`document_record`.

## 8. Known limitations

* The ALTO methods: a namespace-less ALTO file, and any other `.xml` in the ALTO input directory
  (PAGE XML, TEI), is skipped by `page_split.py` with its root named, and gets no page files and no
  document record; the run goes on. Read such files with text-lines, or through the service. (Every
  namespaced ALTO version is split since #31 Phase 5.) The ALTO methods keep `ABBYY-ALTO` as the
  default origin whatever engine the file names; set `SOURCE_ORIGIN = ocr:<engine>` for other exports.
* The alto-tools extractor joins a line that ends in `-`, `–` or `—` with the next one and drops the
  dash (unchanged ALTO behaviour), so a page footer such as `— 1—` loses its last dash there.
* PDF: no OCR (image-only pages are reported, not recognised) and no multi-column reordering.
  Annotations and form fields are not read. A page's `/Rotate` is not part of `rotated_text`.
* DOCX/ODT: style-inherited page breaks, headers/footers and comments are not read (headers and
  footers are counted). Charts, SmartArt and embedded objects are not read.
* XLSX/ODS: formulas contribute their cached text result only; numbers and dates are not text lines.
  PPTX speaker notes are not read.
* CSV/TSV without a recognised text column: the header row is a line too.
* JSON that exports only words or characters without a line level (Google Vision's symbols) gives
  one line per word or character.
* E-mail: headers other than the Subject, and attachments, are not read. Subtitles: speaker names in
  `<v …>` are dropped.
* ZIP bundles and compressed files are one level deep: no PDF, office document or archive is read
  inside them, and `.tar.*`, 7z, RAR and zstd are refused.
* `mojibake_cp1252` is deliberately conservative and Czech-only; text encoded with a base-14 PDF font
  (`ZprÆva`, `(cid:236)`) is not flagged.
* `canonical_doc_id()` (hub-shared) strips a known suffix and otherwise truncates a multi-dot name at
  the first dot. Until the hub's `KNOWN_PIPELINE_SUFFIXES` carries the text-lines suffixes (`.pdf`,
  `.docx`, … — delivered with #31 Phase 5, active once the hub's `v1` is re-vendored), `scan.2019.pdf`
  is `scan`, so `report.v1.docx` and `report.v2.docx` collide; collisions are refused rather than
  merged. Compressed names keep the first-dot answer (`x.txt.gz` → `x`).
* `.doc`, `.xls`, `.ppt`, `.pages`, `.wpd`, `.xlsb` and images are refused with a reason, not
  converted.
