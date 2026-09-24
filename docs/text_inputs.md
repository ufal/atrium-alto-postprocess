# Text-bearing inputs other than ALTO XML (#31)

`--method text-lines` accepts any text-bearing file and turns it into an ordered list of **pages**,
each an ordered list of **lines**. Those lines become the rows of the categorized CSV output: first
`DOC_LINES_TEXT/<doc>.csv`, then, after Step 4, `DOC_LINE_CATEG/<doc>.csv`. This page is the reference for
what the readers in [`text_formats.py`](../text_formats.py) do. The pipeline mechanics are in the
[README](../README.md#any-other-text-bearing-input-pdf-docx-txt--31-).

```
TEXT/ (any mix) ──text_split.py──▶ PAGE_TEXT/<doc>/<doc>-<n>.txt + ingest_report.csv + pages_report.csv
                ──text_stats_create.py──▶ stats CSV (file,page,textlines,illustrations,graphics,strings,path)
                ──extract_TEXT_2_TXT.py──▶ PAGE_TXT_TEXT/<doc>/<doc>-<n>.txt + DOC_LINES_TEXT/<doc>.csv
                ──classify_TEXT.py / aggregate_STAT.py (unchanged)──▶ DOC_LINE_CATEG/, DOC_LINE_STATS/
```

ALTO XML keeps its own methods (`layoutreader`, `alto-tools`, `glm`), which reconstruct reading order
and dehyphenate. Generic OCR JSON keeps `json-keys`. text-lines reads both as well, in document
order, when they arrive mixed with other files.

## 1. How a file is recognised

The **content** decides. The extension is only used to choose between plain-text dialects:

| Order | Test                                                                                                                    | Result                                                                          |
|-------|-------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
| 1     | empty file                                                                                                              | refused `empty_file`                                                            |
| 2     | `%PDF-` in the first 1 KiB                                                                                              | **pdf**                                                                         |
| 3     | OLE2 signature `D0 CF 11 E0`                                                                                            | refused `legacy_office_unsupported` (.doc/.xls/.ppt)                            |
| 4     | ZIP signature → `mimetype` member (ODF/EPUB) or `_rels/.rels` main part (OOXML)                                         | **odt/ods/odp/epub** or **docx/xlsx/pptx**; any other ZIP `archive_unsupported` |
| 5     | PNG/JPEG/GIF/TIFF/WebP/JPEG 2000 signature                                                                              | refused `image_needs_ocr`                                                       |
| 6     | gzip/bzip2/xz/7z/rar/zstd signature                                                                                     | refused `archive_unsupported`                                                   |
| 7     | `{\rtf`                                                                                                                 | **rtf**                                                                         |
| 8     | NUL/control-heavy bytes that are not UTF-16                                                                             | refused `binary_content`                                                        |
| 9     | starts with `<`: root element `alto` / `PcGts` / `TEI` / `html` (hOCR if `ocr_page`/`ocr_line` classes) / other         | **alto / page-xml / tei / hocr / html / xml**                                   |
| 10    | `.jsonl`/`.ndjson` → **jsonl**; `.json` → **json**; other `{`/`[` files → json if it parses, jsonl if every record does |                                                                                 |
| 11    | `.csv` → **csv**; `.tsv`/`.tab` → **tsv**; `.md`/`.markdown` → **md**; anything else → **txt**                          |                                                                                 |

A mismatch between extension and content is recorded in the ingest report's `notes`, for example
`extension .txt but content is pdf`, and the file is still read by its content.

## 2. What a page and a line are, per format

| Kind          | Page (block)                                                                                                                                                      | Line                                                                                                                                                                                   | Library        |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| TXT           | form-feed (`\f`) section; otherwise the whole file                                                                                                                | physical line                                                                                                                                                                          | stdlib         |
| Markdown      | form-feed section                                                                                                                                                 | physical line, with front matter, code fences, comments, rules and link definitions dropped; heading/list/quote markers and inline markup stripped; table rows become tab-joined cells | stdlib         |
| CSV / TSV     | the file; or groups of a `page`/`page_num`/`page_number` column, in first-seen order                                                                              | the `text`/`line`/`content`/`transcription`/`sentence`/`string` column; otherwise all non-empty cells joined by tab                                                                    | stdlib `csv`   |
| JSON          | page list (`pages`, …) or page-tagged list (`Page`, `pageNumber`, …), the same detection as json-keys; otherwise each top-level child object/list that holds text | string leaf under a text key (`content`, `text`, `line`, …); when none exist, every string containing a letter (noted `json_all_strings`)                                              | stdlib         |
| JSON Lines    | record                                                                                                                                                            | as JSON                                                                                                                                                                                | stdlib         |
| ALTO v2/v3/v4 | `Page` (label `PHYSICAL_IMG_NR`)                                                                                                                                  | `TextLine`: `String@CONTENT` joined by spaces, `HYP` → `-`                                                                                                                             | lxml           |
| PAGE XML      | `Page` (label = image file stem)                                                                                                                                  | `TextLine` in `ReadingOrder` region order: first `TextEquiv/Unicode`, else its `Word`s                                                                                                 | lxml           |
| hOCR          | `ocr_page` (label from `ppageno`, 1-based)                                                                                                                        | `ocr_line`, `ocrx_line`, `ocr_caption`, `ocr_header`, `ocr_textfloat`                                                                                                                  | lxml.html      |
| HTML / XHTML  | CSS `page-break-before/after` (`break-before: page`); otherwise one page                                                                                          | block element (`p`, `h1`–`h6`, `li`, `td`, …) or `<br>`; `script`/`style`/`head` dropped                                                                                               | lxml.html      |
| TEI / TEITOK  | `<pb/>`, every one, a blank page too (label `pb@n`, else its number)                                                                                              | `<lb/>` and block ends (`p`, `head`, `l`, `item`, `cell`, `ab`, …); `<s>` too, except in tokenized TEITOK (`<tok>` + `<lb/>`), where a sentence is not a line; `teiHeader` skipped     | lxml           |
| other XML     | root child elements, when two or more hold text                                                                                                                   | an element with its own (mixed) text; pure containers are descended into                                                                                                               | lxml           |
| PDF           | PDF page (label = the PDF's page label, else its number)                                                                                                          | text-layer line, in PDFium's order                                                                                                                                                     | **pypdfium2**  |
| DOCX          | explicit page break, `pageBreakBefore`, non-continuous section break; in `auto` mode also Word's rendered breaks                                                  | paragraph (`w:br`/`w:cr` split it); table cells row by row; text boxes after their anchor paragraph                                                                                    | zipfile + lxml |
| ODT           | `fo:break-before/after="page"` paragraph styles; in `auto` mode also `text:soft-page-break`                                                                       | paragraph / heading (`text:line-break` splits it); table cells                                                                                                                         | zipfile + lxml |
| XLSX / ODS    | sheet, in workbook order (label = sheet name)                                                                                                                     | row: its text cells joined by tab; numbers, dates, booleans dropped                                                                                                                    | zipfile + lxml |
| PPTX / ODP    | slide, in presentation order                                                                                                                                      | paragraph                                                                                                                                                                              | zipfile + lxml |
| EPUB          | spine chapter (label = file stem)                                                                                                                                 | as HTML                                                                                                                                                                                | zipfile + lxml |
| RTF           | `\page`                                                                                                                                                           | `\par`, `\line`, `\row`; `\cell` → tab; header/footer/footnote/picture/field-instruction groups skipped; `\'xx` decoded with `\ansicpgN`                                               | stdlib         |

Pages are written as `<doc>-1 … <doc>-N` in reading order. The original label (sheet name, PDF
page label, JSON page number, `record[2]` …) is kept in `pages_report.csv` and in the line table's
`page_label` column. Every PDF page is written, including pages without text, so page numbers
stay faithful to the source. A block with no text (a sheet with only numbers, a JSON child with
only ids) is not a page.

**DOCX/ODT pagination** is set by `[TEXT_INGEST].PAGE_BREAKS`:
* `auto` (default): explicit breaks plus the breaks Word or LibreOffice rendered when the file was
  last saved. A rendered break right after an explicit one opens no extra page. A rendered break
  in the middle of a paragraph splits it where the page really ended.
* `explicit`: only the breaks the author inserted.
* `none`: one page.

DOCX break styles inherited through the style hierarchy are ignored (only direct formatting counts).
Deleted revisions and the VML fallback copies of text boxes are skipped. Headers, footers and
footnotes are not read.

## 3. Normalisation (the line invariant)

`text_split.py` writes each line after `normalize_line()`:
* every line separator except the page break becomes `\n` (`\r\n`, `\r`, `\v`, `\x1c–\x1e`, U+0085,
  U+2028/2029). `classify_TEXT` splits with `readlines()`, so any separator left inside a line would
  shift line numbers.
* NFC; the ligatures U+FB00–FB06 are expanded.
* zero-width, BOM and bidi controls are removed (ZWJ/ZWNJ kept); NBSP-like spaces become spaces.
* C0/C1 control characters except tab are removed.
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

Plain text is decoded in this order:
1. a BOM (UTF-8/16/32);
2. strict UTF-8;
3. UTF-16 without a BOM, recognised by its NUL pattern;
4. a binary check;
5. charset-normalizer, restricted to `FALLBACK_ENCODINGS` (default `cp1250, iso8859_2, cp1252`);
6. the fallbacks in order.

Restricting detection to plausible code pages keeps a short Czech file from being "detected" as an
unrelated one and turned into mojibake, which the categorizer would then score as damaged OCR. The
encoding used is reported per file.

## 4. Limits and failure handling

Each file is processed on its own, and a failure costs only that file. It gets a row in
`ingest_report.csv` with its reason code, a `skipped_files_detail` entry in paradata, and the run
continues. `--strict` (or `STRICT = true`) makes `text_split.py` exit 1 if any file failed.

| `[TEXT_INGEST]` key  | Default | Guards against                                                                                                                    |
|----------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------|
| `MAX_FILE_MB`        | 256     | oversized files (`too_large`)                                                                                                     |
| `ZIP_MAX_MEMBERS`    | 10000   | ZIP containers with absurd member counts (`zip_limits_exceeded`)                                                                  |
| `ZIP_MAX_TOTAL_MB`   | 1024    | zip bombs: declared unpacked size, checked **before** anything is read                                                            |
| `ZIP_MAX_MEMBER_MB`  | 256     | one huge member                                                                                                                   |
| `ZIP_MAX_RATIO`      | 200     | compression ratio of members over 1 MiB                                                                                           |
| `MAX_PAGES`          | 20000   | page-count bombs (`too_large`)                                                                                                    |
| `MAX_LINES_PER_PAGE` | 100000  | a real page over it is refused; a block over it continues on pages labelled `<label>+1`, `+2`, …                                  |
| `READER_TIMEOUT_S`   | 300     | a PDF that hangs PDFium. PDFs are read in a **separate process**, so a crash or hang costs one file (`timeout`, `reader_crashed`) |

XML is parsed without entity resolution, DTD loading or network access, and with libxml2's size and
depth limits on. A document that declares entities is refused (`xml_entity_declaration`). Broken XML
gets one retry in recovery mode, noted `xml_recovered` — except the `<name>…</n>` quirk of older
TEITOK exports, which is repaired exactly (noted `name_close_repaired`) because recovery may drop text. Encrypted inputs are refused (`encrypted`):
password PDFs, ZIP members with the encryption flag, ODF with `encryption-data`, and DRM-protected
EPUB chapters (EPUB font obfuscation alone is fine).

Input discovery is conservative. Only regular files at the top level of the input directory are
read. Symbolic links, FIFOs/devices, subdirectories, hidden files, Office lock files (`~$…`,
`.~lock.…#`) and OS metadata (`._*`, `Thumbs.db`, `desktop.ini`) are listed as `ignored`.
Document ids come from `canonical_doc_id()` (hub-shared), so `report.v2.pdf` becomes `report`.
Two files that map to the same id (case-insensitively) are a `doc_id_collision`: the first in sorted
order wins and the other is refused, never silently overwritten. Each document's pages are written
into a staging directory and swapped in, so a re-run never leaves stale pages from a longer
earlier version.

### Reason codes

| Code                                  | Meaning                                                                              |
|---------------------------------------|--------------------------------------------------------------------------------------|
| `empty_file`                          | zero bytes                                                                           |
| `too_large`                           | a size/page/line cap was exceeded                                                    |
| `binary_content`                      | not text and not a supported container                                               |
| `legacy_office_unsupported`           | OLE2 `.doc/.xls/.ppt`; save it as DOCX/XLSX/PPTX                                     |
| `image_needs_ocr`                     | an image; run OCR first and feed its output (ALTO, PAGE XML, hOCR, TXT)              |
| `archive_unsupported`                 | a ZIP/archive that is not DOCX/XLSX/PPTX/ODF/EPUB                                    |
| `zip_limits_exceeded`                 | ZIP caps above                                                                       |
| `xml_entity_declaration`              | XML with `<!ENTITY>`                                                                 |
| `malformed`                           | broken for its format (bad JSON, unparseable XML/CSV, too deeply nested)             |
| `corrupt`                             | the container could not be opened                                                    |
| `encrypted`                           | password- or DRM-protected                                                           |
| `timeout` / `reader_crashed`          | the isolated PDF reader hung or died                                                 |
| `dependency_missing`                  | `pypdfium2` (PDF) or `lxml` not installed                                            |
| `decode_failed`                       | no configured encoding decodes the text                                              |
| `no_text`                             | read fine, but no text lines. For a PDF: no text layer on any page, so run OCR first |
| `doc_id_collision` / `doc_id_invalid` | see above                                                                            |
| `output_failed`                       | writing the page files or the document record failed                                 |

## 5. PDF text layers

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
as llm-enrich's `digital_to_json`.

## 6. Provenance: `source.origin`

With `[DOCUMENT].JSON_DIR` set, `text_split.py` writes the record's `source`: `sha256`, `filename`,
`media_type`, `origin`, and `page_count` for formats with real pages only (PDF, ALTO, PAGE XML,
hOCR). llm-enrich records a DOCX as one page, and an invented count would conflict in `set_source()`.
The origin is **truthful per class** unless `--source-origin`, `DOCUMENT_SOURCE_ORIGIN` or
`[DOCUMENT].SOURCE_ORIGIN` overrides it:

| Class                      | Formats                                                                | Default origin                                                 | Positional blocks written by this repo |
|----------------------------|------------------------------------------------------------------------|----------------------------------------------------------------|----------------------------------------|
| OCR output                 | ALTO, PAGE XML, hOCR, PDF with an OCR layer                            | `ABBYY-ALTO`, `ocr:page-xml`, `ocr:hocr`, `ocr:pdf-text-layer` | yes                                    |
| text of unknown provenance | TXT, Markdown, CSV/TSV, JSON/JSONL, TEI, other XML                     | `ocr:generic` (as json-keys)                                   | yes                                    |
| born-digital               | DOCX, ODT/ODS/ODP, XLSX, PPTX, EPUB, RTF, plain HTML, visible-text PDF | `digital-born-<kind>`                                          | **no**: `source` only                  |

A `digital-born-*` origin authorises llm-enrich's **digital-convert** to originate the record's
`pages`/`content`/`lines`/`tables` (`atrium_document` §1a), and the hub's digital end-to-end test
asserts that alto-postprocess does not write them. The shared check only *warns* when it is not
strict. `document_hook.write_document_block()` therefore holds those blocks back for every stage of
this repo, including the unchanged `classify_TEXT` and `aggregate_STAT`, with one warning per
document. The exception is when the record carries digital-convert's own `pages[].needs_ocr`
hand-off, which asks for exactly this repo's pass. The categorized CSV outputs are produced either
way. If the files are known OCR output (for example a `.txt` or `.docx` export from an OCR engine),
say so with `--source-origin ocr:<engine>`.

## 7. The API service

`POST /process` accepts the same formats with `task_type=document`. `auto` decides `.xml` and any
unknown extension from the bytes; an ALTO root keeps the ALTO path. Documents are read with
`read_document_isolated()` and shaped with `shape_lines()`, the same code as the batch method, and
classified page by page in batches of 128 lines. Each result line carries `page`/`page_label`, and
`line_num` restarts per page. An unsupported file is a `400` and an unreadable one a `422`, both
naming the reason code. Born-digital uploads do not accrete into a `document_record`.

## 8. Known limitations

* PDF: no OCR (image-only pages are reported, not recognised) and no multi-column reordering.
  Annotations and form fields are not read.
* DOCX/ODT: style-inherited page breaks, headers/footers, footnotes/endnotes and comments are not
  read. Charts, SmartArt and embedded objects are not read.
* XLSX/ODS: formulas contribute their cached text result only; numbers and dates are not text lines.
* A folder of per-page files (for example one PAGE XML per scan) is read as one document per file,
  not as one multi-page document.
* `canonical_doc_id()` (hub-shared) truncates unknown multi-dot names at the first dot
  (`scan.2019.pdf` → `scan`). Collisions are refused rather than merged. Adding the office/PDF
  suffixes to the hub's `KNOWN_PIPELINE_SUFFIXES` is a cross-repo follow-up.
* `.doc`, `.xls`, `.ppt`, `.pages`, `.wpd` and images are refused with a reason, not converted.
