---
name: atrium-alto-postprocess
description: Classifies OCR output quality line-by-line - uploads ALTO XML pages or plain-text files and returns per-line language identification (FastText), perplexity (Qwen2.5-0.5B), noise metrics, a composite quality score, and a Clear/Noisy/Trash/Non-text/Empty category, with LayoutReader reading-order reconstruction for ALTO input. Use this skill to filter and triage digitized historical documents after OCR, before NLP enrichment or translation.
---

# ATRIUM ALTO Postprocessing Skill 🧹

This skill provides agent access to the **ATRIUM ALTO Postprocessing** service -
language identification and quality classification for OCR'd document lines.
It follows a **server-client** design: a FastAPI server (in `service/`) runs the
models (LayoutReader, FastText, Qwen2.5-0.5B perplexity), and a zero-dependency
client script (`scripts/atrium_postprocess.py`) is the only thing the agent
calls directly.

## Operational Requirements ⚙️

- **Server**: a running instance is required. Default `http://localhost:8000`;
  override with `--base-url` or the `ATRIUM_AP_URL` environment variable.
- **Client dependencies**: none - `scripts/atrium_postprocess.py` uses only the
  Python 3 standard library.
- **Server dependencies**: Docker (recommended, compose `api` profile) or a
  Python venv provisioned by `setup/setup_api_server.sh` (installs
  `service/requirements.txt`, fetches LayoutReader helpers and the FastText
  binary).
- **First launch**: FastText `lid.176.bin` (~130 MB), LayoutReader, and the
  perplexity model are downloaded and cached. Warmup takes minutes, not
  seconds - do **not** treat a slow first start as failure.
- **Limits**: 10 MB per file (one ALTO page or one text file per request).

## Quality categories 🧹

| Category   | Meaning                                                        |
|------------|----------------------------------------------------------------|
| `Clear`    | clean, readable text - safe for downstream NLP                 |
| `Noisy`    | readable but degraded - usable with care                       |
| `Trash`    | OCR garbage - discard or re-OCR                                |
| `Non-text` | dates, numbers, codes - no linguistic content                  |
| `Empty`    | nothing left after cleaning                                    |

Each line also carries the raw signals behind the decision: `lang`/`lang_score`
(FastText), `perplexity` (Qwen2.5-0.5B; 0 = pre-filtered), `garbage_density`,
`sym_count`, `upper_count`, `repeated_count`, `ldl_fuses`, `gibberish`,
`word_weird`, and the composite `quality_score` [0-1] (> 0.75 Clear,
>= 0.45 Noisy, < 0.45 Trash).

## Workflows 🪄

### 1. Ensure the server is running

```bash
bash scripts/server.sh          # Docker Compose api profile (or local fallback)
bash scripts/server.sh --gpu    # Docker with GPU
bash scripts/server.sh --local  # force local uvicorn (no Docker)
```

Idempotent: exits immediately if GET /info already answers; waits for
first-run warmup.

### 2. Classify

```bash
# One ALTO XML page (layout reorder + per-line classification)
python3 scripts/atrium_postprocess.py small_data_samples/CTX000000001-1.alto.xml

# Plain-text lines, machine-readable CSV
python3 scripts/atrium_postprocess.py page.txt --format csv

# Several pages, raw JSON (full per-line metrics)
python3 scripts/atrium_postprocess.py scans/*.xml --format json

# Force the handling of a non-standard suffix
python3 scripts/atrium_postprocess.py export.dat --task-type text

# Discover capabilities and limits
python3 scripts/atrium_postprocess.py --info
```

### 3. Interpret output

Rows are FILE, LINE, LANG, QUALITY, CATEGORY, TEXT (text truncated in
table mode, complete in csv/json). The JSON response additionally
reports reading_order: layout-reader when LayoutReader reordering was
applied to ALTO lines, document otherwise.

## Agent Guidelines 🤖

1. Routing discipline: .xml goes through ALTO parsing + layout
reordering; .txt is classified line-by-line as-is. For other suffixes,
pass --task-type explicitly rather than renaming files.
2. Category use: feed only Clear (and, with care, Noisy) lines into
downstream NLP/translation; surface the quality_score when a decision is
borderline rather than asserting the category alone.
3. Prefer --format json when the result feeds further processing (it carries
every per-line metric, not just the summary columns).
4. For full request/response schemas, fetch GET /openapi.json from the
running server (Swagger UI at /docs).
5. Exit code 2 (unreachable): start the server (bash scripts/server.sh)
and retry once. Exit code 3 (server error): the client already retried
502/503/504 three times - check GET /health?deep=true and server logs;
do not loop.
6. Size limits: files over 10 MB are rejected - split multi-page ALTO
exports into single pages first, and tell the user you did so.
7. Do not bypass the API by importing the model code directly - the server is
the supported, resource-managed entry point for classification runs.

## Acknowledgements & Citations 🙏

The models and dataset are developed within the [ATRIUM](https://atrium-research.eu/)
project at ÚFAL, Charles University, with data hosted on
[LINDAT/CLARIAH-CZ](https://lindat.cz). If you use this service for research, cite the
repository's `CITATION.cff` and the LINDAT dataset record
(http://hdl.handle.net/20.500.12800/1-6184).
