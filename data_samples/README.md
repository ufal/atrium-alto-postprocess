# data_samples (synthetic)

Synthetic demonstration data for the ALTO postprocessing pipeline. All content is
**fictional** — an invented site "Hradiště u Horní Mezí" (okr. Horní Mezi), with
made-up researchers Jan Novotný and Eva Procházková. No real archival records,
CTX identifiers, place names, or restricted ARUP/ARUB data are included.

Demo documents:
- CTX000000001 — 2 pages, clean OCR (baseline)
- CTX000000002 — 4 pages, mixed quality (Noisy/Trash lines, empty tokens)
- CTX000000003 — 1 page, short/poor (edge cases)

Inputs for the other two methods (#31) — same fictional world, new ids so nothing clashes:
- JSON/CTX000000015.json — an Azure-style `analyzeResult.pages` OCR export (`--method json-keys`)
- TEXT/ — one file per text-lines format (`--method text-lines`):
  - CTX000000004.txt — plain text in **cp1250**, two pages split by a form feed, OCR noise on page 2
  - CTX000000005.md — Markdown with front matter, a list, a table and a code fence
  - CTX000000006.csv — a `page,text` line table
  - CTX000000007.jsonl — three JSON Lines records (one page each)
  - CTX000000008.hocr — Tesseract-style hOCR, two `ocr_page`s
  - CTX000000009.xml — PAGE XML whose ReadingOrder differs from document order
  - CTX000000010.docx — explicit page break and a table (born-digital)
  - CTX000000011.pdf — born-digital PDF, 3 pages, the last with no text layer
  - CTX000000012.pdf — "scanned" PDF with an invisible OCR text layer (render mode 3)
  - CTX000000013.xlsx — two sheets = two pages; rows = lines, numbers dropped
  - CTX000000014.odt — a paragraph style with a page break before it
  - CTX000000016.tsv — Tesseract TSV, 2 pages, words grouped into lines (OCR noise on both pages)
  - CTX000000017.xml — ABBYY FineReader 10 XML, 2 pages, one character per `charParams`, a table block
  - CTX000000018.xml — DjVuXML, two `OBJECT` pages of `LINE`/`WORD`s
  - CTX000000019.srt — SubRip subtitles with `<i>` and `{\an8}` markup (dropped)
  - CTX000000020.vtt — WebVTT with a header, a `NOTE` block and `<v Speaker>` voices
  - CTX000000021.eml — a multipart e-mail (text + HTML alternative, Q-encoded Czech subject)
  - CTX000000022.mbox — a mailbox of two messages (one page each), a `>From ` quoted body line
  - CTX000000023.txt.gz — a gzip-compressed page transcript (read through)
  - CTX000000024.zip — a Transkribus-style export: two PAGE XML pages plus METS and doc metadata, read as one
    two-page document
  - CTX000000025.tsv — **real** Tesseract 5.3.4 TSV of two rendered Czech report pages (a line-end hyphen,
    page footers), the engine's own output
  - CTX000000026.hocr — the same two pages as **real** Tesseract hOCR (`ocr-system` names the engine, so the
    origin is `ocr:tesseract`)

  The binary members (DOCX, PDF, XLSX, ODT, `.txt.gz`, `.zip`) are generated, byte-for-byte reproducibly, by
  `tests/text_format_fixtures.write_samples()`; CTX000000025/26 by `tools/make_tesseract_samples.py` (needs
  `tesseract` with `ces`; output depends on its version, so they are committed as generated). The tests read
  only those two real samples here. The text-lines outputs go to PAGE_TEXT/ (pages +
  ingest_report.csv/pages_report.csv), PAGE_TXT_TEXT/ (classify-ready text) and DOC_LINES_TEXT/
  (per-document line tables), with the stats CSV `text_stats.csv` (`[EXTRACT].INPUT_CSV_TEXT`) — not committed.

Directory roles mirror the real pipeline: ALTO/ (source) -> A-PAGE/, PAGE_ALTO/
(page splits) -> PAGE_TXT*, (text extraction) -> DOC_LINE_CATEG*, DOC_LINE_STATS*
(classification + aggregation). The *_gpt variants correspond to the distilgpt2
perplexity model; the unsuffixed ones to Qwen2.5-0.5B.
