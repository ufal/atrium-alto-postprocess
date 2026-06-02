# data_samples (synthetic)

Synthetic demonstration data for the ALTO postprocessing pipeline. All content is
**fictional** — an invented site "Hradiště u Horní Mezí" (okr. Horní Mezi), with
made-up researchers Jan Novotný and Eva Procházková. No real archival records,
CTX identifiers, place names, or restricted ARUP/ARUB data are included.

Demo documents:
- CTX000000001 — 2 pages, clean OCR (baseline)
- CTX000000002 — 4 pages, mixed quality (Noisy/Trash lines, empty tokens)
- CTX000000003 — 1 page, short/poor (edge cases)

Directory roles mirror the real pipeline: ALTO/ (source) -> A-PAGE/, PAGE_ALTO/
(page splits) -> PAGE_TXT*, (text extraction) -> DOC_LINE_CATEG*, DOC_LINE_STATS*
(classification + aggregation). The *_gpt variants correspond to the distilgpt2
perplexity model; the unsuffixed ones to Qwen2.5-0.5B.
