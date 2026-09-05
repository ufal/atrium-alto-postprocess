"""
service/text_api.py
FastAPI wrapper for the ATRIUM text processing service.
"""

import json
import os
import shutil
import sys
import tempfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from atrium_document import canonical_doc_id
from atrium_paradata import ParadataLogger
from document_hook import PROGRAM_NAME, quality_band, write_document_block

# Add this file's own directory (service/) to sys.path BEFORE importing the
# sibling `text_inference` module, so the bare import resolves in every launch
# context: `python service/text_api.py` (the Docker entrypoint), `uvicorn
# service.text_api:app`, and pytest importing this module as `service.text_api`
# from the repo root. This bootstrap MUST run before the import below; the E402
# suppression on that import keeps Ruff's import sorter (I001) from hoisting it
# back above this code and re-breaking it (regression tracked in atrium-project#18).
_current_dir = Path(__file__).resolve().parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

# Both imports below are bare (service/ is on sys.path from the bootstrap above) and
# carry noqa: E402 so Ruff's import sorter does not hoist them above that bootstrap.
# `atrium_service` is the shared ATRIUM meta-contract helper (§4), byte-identical
# across every service and enforced by para-drift.reusable.yml.
from atrium_service import (  # noqa: E402
    add_cors,
    attach_health,
    build_info,
    read_tool_version,
    resolve_max_upload_mb,
)
from text_inference import text_manager  # noqa: E402

# Bare like its siblings above, for the same reason: `utils` is service/utils.py,
# reached through the sys.path bootstrap, so the import resolves under
# `python service/text_api.py` (the Docker entrypoint) as well as under
# `uvicorn service.text_api:app`.
from utils import parse_alto_page_labels  # noqa: E402

# Canonical upload limit (§4.5): MAX_UPLOAD_MB, with a MAX_UPLOAD_BYTES fallback.
MAX_UPLOAD_MB = resolve_max_upload_mb(25)
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle context manager — loads models synchronously before traffic."""
    try:
        text_manager.load_models()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise models on startup: {exc}") from exc
    yield


app = FastAPI(
    title="ATRIUM Text Processor",
    version=read_tool_version(Path(__file__).resolve().parent),
    lifespan=lifespan,
)

# CORS — standard §4.5 configuration; default "*" for parity with sibling services.
add_cors(app, methods=["GET", "POST"])


def _deep_health() -> str | None:
    """Deep readiness (§4.1): quality/language models are loaded."""
    if getattr(text_manager, "device", None) is None:
        return "text models not loaded"
    return None


attach_health(app, deep_check=_deep_health)

# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Resolve the absolute path to setup/para_config.txt
PARA_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "setup" / "para_config.txt")

# ---------------------------------------------------------------------------
# Accretion mapping (atrium-project#10 J1)
# ---------------------------------------------------------------------------
# One request describes exactly ONE page, and that is a property of the inference
# layer rather than a simplification made here:
#   * alto  — service/utils.parse_alto_xml_lines() takes its geometry from
#             `root.find(".//Page")`, i.e. the FIRST <Page> only, and
#             post_process_text() then reads the whole upload's lines back in that
#             one page's coordinate space;
#   * json  — process_json() is "one JSON file = one page" by the batch pipeline's
#             own convention (see its docstring);
#   * text  — a .txt upload carries no page concept at all.
#
# `page_metrics` used to be a hardcoded
# `[{"page": "1", "quality_score": result.get("doc_quality", 1.0)}]`, and
# `doc_quality` is a key `text_inference` returns on no path whatsoever — so every
# accreted record claimed a perfect 1.0 for exactly one page named "1", regardless
# of what was uploaded, appending a page row to any document whose own labels
# (PHYSICAL_IMG_NR) were anything else. Below, the page SET, the LABEL and every
# metric are all derived from the request: the label from the ALTO's own <Page>, the
# rows from the lines actually classified, the metrics from those lines' scores.
#
# Fallback label for the two formats that have no page identity of their own. "1" is
# not a guess there: it is the label page_split.split_json_document gives the single
# page of a Family-C JSON document, and a plain-text upload has exactly one page by
# definition.
SERVICE_PAGE_LABEL = "1"


def _lines_records_from_result(result: Dict[str, Any], page: str = SERVICE_PAGE_LABEL) -> List[Dict[str, Any]]:
    """Project `text_inference`'s classified lines onto atrium_document's `lines[]`
    shape: key fields `page`/`line`, plus this repo's owned fields (categ,
    quality_score, lang, text) — the same projection classify_TEXT.py's
    `_lines_records_from_df()` performs for the batch path.

    Reads **`cleaned_lines`**, which is the key `text_inference` returns on every
    path (`process_text_file`/`process_json`/`process_alto`, including its empty-
    document early return). The endpoint used to read `result["lines"]`, a key
    nothing ever writes, so the list was always empty and `write_document_block`'s
    `if records:` guard skipped the lines merge on every single call (J1).

    Note the two spellings: the inference layer calls the field `category`, the
    schema calls it `categ`. Values pass through VERBATIM — `"Garbage"` and
    `"Inverted"` are load-bearing (api_util/json_to_md.py's DROP_CATEGORIES keys
    off them), and a re-spelling would not fail validation, it would silently
    disable that filter.
    """
    records: List[Dict[str, Any]] = []
    for entry in result.get("cleaned_lines") or []:
        if not isinstance(entry, dict):
            continue
        line_num = entry.get("line_num")
        if line_num is None:
            # `line` is a required key field; a row without one cannot be merged
            # (merge_block would align every such row onto the same null key).
            continue
        record: Dict[str, Any] = {"page": page, "line": int(line_num)}
        for source_key, schema_field in (
            ("text", "text"),
            ("lang", "lang"),
            ("quality_score", "quality_score"),
            ("category", "categ"),
        ):
            value = entry.get(source_key)
            if value is not None:
                record[schema_field] = value
        records.append(record)
    return records


def _page_records_from_lines(line_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Real `pages[]` rows for the lines this request just classified.

    One row per page label actually present in `line_records`, with
    `quality_score` as the mean of that page's own line scores and `quality_band`
    reduced from its own Clear/Noisy/Trash counts by the same plurality vote
    aggregate_STAT.py applies to the batch path (document_hook.quality_band).

    `quality_band` is omitted when none of the three bands is represented (an
    all-`Empty`/all-`Non-text` page): the vote's tie-break favours "Clear", so
    counting zeros would report a pristine page for one with no text on it.
    """
    by_page: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for record in line_records:
        by_page.setdefault(record["page"], []).append(record)

    pages: List[Dict[str, Any]] = []
    for page, rows in by_page.items():
        record: Dict[str, Any] = {"page": page}
        scores = [r["quality_score"] for r in rows if isinstance(r.get("quality_score"), (int, float))]
        if scores:
            record["quality_score"] = round(sum(scores) / len(scores), 4)
        categs = [r.get("categ") for r in rows]
        clear, noisy, trash = categs.count("Clear"), categs.count("Noisy"), categs.count("Trash")
        if clear or noisy or trash:
            record["quality_band"] = quality_band(clear, noisy, trash)
        pages.append(record)
    return pages


def _accretion_records(task_type: str, upload_path: str, result: Dict[str, Any]):
    """The (pages, lines) contribution for one /process request, or ([], []) when it
    cannot be attributed to a page truthfully.

    A multi-page ALTO upload is the "cannot" case, and it is refused rather than
    guessed at: `parse_alto_xml_lines` flattens every page's `<TextLine>` into one
    list scaled by the FIRST page's dimensions, so `result` genuinely does not say
    which page a given line came from. Writing them all under one label would put
    misattributed rows into a record other tools then align their own fields onto —
    the silent-wrong-data failure this whole issue is about. The classified lines are
    still returned in the HTTP response; only the accretion is skipped, and loudly.
    Multi-page documents belong to the batch pipeline (page_split.py splits first).
    """
    page_labels = parse_alto_page_labels(upload_path) if task_type == "alto" else []
    if len(page_labels) > 1:
        print(
            f"[document] WARNING – upload has {len(page_labels)} <Page> elements: "
            f"/process classifies them as one flattened page, so no pages[]/lines[] "
            f"contribution can be attributed per page. Skipping the accretion for this "
            f"request — split the document first (page_split.py) and post one page per "
            f"request, or use the batch pipeline.",
            file=sys.stderr,
        )
        return [], []

    page = page_labels[0] if page_labels else SERVICE_PAGE_LABEL
    lines = _lines_records_from_result(result, page)
    return _page_records_from_lines(lines), lines


@app.get("/", response_model=None)
async def root() -> Union[HTMLResponse, Dict[str, str]]:
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return {"message": "Service running. Frontend not found."}


@app.get("/info")
async def info() -> Dict[str, Any]:
    return build_info(
        app,
        service="atrium-alto-postprocess",
        limits={"max_upload_mb": MAX_UPLOAD_MB},
        status="active",
        device=text_manager.device,
        supported_formats=["ALTO XML (.xml)", "Plain Text (.txt)", "Generic JSON (.json)"],
        quality_categories=["Clear", "Noisy", "Trash", "Non-text", "Empty"],
        line_fields=[
            "line_num",
            "text",
            "lang",
            "lang_score",
            "perplexity",
            "garbage_density",
            "sym_count",
            "upper_count",
            "repeated_count",
            "ldl_fuses",
            "gibberish",
            "word_weird",
            "quality_score",
            "category",
        ],
    )


@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    task_type: str = Form("auto"),
    document_record: UploadFile = File(None),  # Added optional input
) -> JSONResponse:
    """
    Upload an ALTO XML, plain-text, or generic JSON file.

    Returns a list of classified lines.  Each entry carries:

      line_num        (int)   – 1-based position after layout reordering
      text            (str)   – cleaned text with split-word merges applied
      lang            (str)   – ISO language code predicted by FastText
      lang_score      (float) – FastText confidence [0, 1]
      perplexity      (float) – Qwen2.5-0.5B perplexity; 0 for pre-filtered lines
      garbage_density (float) – ratio of non-alphanumeric noise characters
      sym_count       (int)   – tokens with strange/unexpected symbols
      upper_count     (int)   – tokens with mid-word uppercase artefacts
      repeated_count  (int)   – tokens with non-standard char repetition (>=40%)
      ldl_fuses       (int)   – tokens with letter-digit-letter fusions
      gibberish       (int)   – tokens lacking vowels or highly irregular ratios
      word_weird      (float) – mean per-word weirdness score [0, 1]
      quality_score   (float) – composite continuous quality score [0, 1]
      category        (str)   – Clear | Noisy | Trash | Non-text | Empty
                                Assigned dynamically using the unified penalty system.
    """
    if not file.filename:
        raise HTTPException(status_code=422, detail="Filename is missing from the upload.")
    if not file.content_type:
        raise HTTPException(status_code=422, detail="Content-Type is missing from the upload.")

    filename = file.filename.lower()
    # (atrium-project#10 D2) The record's key comes from the hub's one derivation,
    # on the ORIGINAL-CASE filename. `Path(filename).stem` on the lower-cased name
    # yielded `ctx000000001.alto` for this repo's own documented convention
    # `CTX000000001.alto.xml` — down-cased AND still carrying `.alto`, because
    # `stem` strips only the LAST extension — while page_split.py keys the same
    # document `CTX000000001`. Since DocumentRecord.__init__ sets `_data["doc_id"]`
    # unconditionally, uploading a real baseline re-keyed the accreted output.
    # Nothing else in the pipeline lower-cases, so neither does this.
    doc_id = canonical_doc_id(file.filename)
    if not doc_id:
        raise HTTPException(status_code=422, detail="Filename has no usable document id.")

    if task_type == "auto":
        if filename.endswith(".xml"):
            task_type = "alto"
        elif filename.endswith(".txt"):
            task_type = "text"
        elif filename.endswith(".json"):
            task_type = "json"
        else:
            raise HTTPException(
                status_code=400,
                detail="Cannot auto-detect file type. Set task_type='alto', 'text', or 'json'.",
            )

    # Initialize ParadataLogger with the required config argument and unified program name
    para_logger = ParadataLogger(config=PARA_CONFIG_PATH, program=PROGRAM_NAME)

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if os.path.getsize(tmp_path) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.")

        # Execute text inference
        if task_type == "alto":
            result = text_manager.process_alto(tmp_path)
        elif task_type == "json":
            result = text_manager.process_json(tmp_path)
        else:
            result = text_manager.process_text_file(tmp_path)

        result["filename"] = file.filename

        # --- Paradata Pair Accretion Hook ---
        if document_record:
            with tempfile.TemporaryDirectory() as doc_tmp_dir:
                baseline_path = os.path.join(doc_tmp_dir, f"{doc_id}.document.json")

                # Save the uploaded baseline JSON
                with open(baseline_path, "wb") as bf:
                    shutil.copyfileobj(document_record.file, bf)

                # (#10 J1) Real lines + real per-page rows, both derived from this
                # request — see _accretion_records for what was fabricated before.
                page_metrics, lines_metrics = _accretion_records(task_type, tmp_path, result)

                # Write the block using the repo-local hook. `pages`/`lines` are
                # field-split with page-classification/nlp-enrich (BLOCK_FIELD_OWNERS),
                # so this must merge — set_blocks would erase their co-owned fields
                # (category/category_confidence, teitok_surface, lemma/upos/feats, ...)
                # on any document that already carries a baseline record.
                write_document_block(
                    document_json_dir=doc_tmp_dir,
                    doc_id=doc_id,
                    run_id=para_logger.run_id,
                    paradata_ref="",  # Left empty for stateless API responses
                    merge_blocks={"pages": page_metrics, "lines": lines_metrics},
                )

                # Read back the accreted record and attach it to the response
                with open(baseline_path, "r", encoding="utf-8") as bf:
                    result["document_json_out"] = json.load(bf)
        # ------------------------------------

        para_logger.finalize()
        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# @app.post("/process")
# async def process_document(
#     file: UploadFile = File(...),
#     task_type: str = Form("auto"),
# ) -> JSONResponse:
#
#     # §4.4: missing upload metadata is a client error (422), not a server 500.
#     if not file.filename:
#         raise HTTPException(status_code=422, detail="Filename is missing from the upload.")
#     if not file.content_type:
#         raise HTTPException(status_code=422, detail="Content-Type is missing from the upload.")
#
#     filename = file.filename.lower()
#
#     if task_type == "auto":
#         if filename.endswith(".xml"):
#             task_type = "alto"
#         elif filename.endswith(".txt"):
#             task_type = "text"
#         elif filename.endswith(".json"):
#             task_type = "json"
#         else:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Cannot auto-detect file type. Set task_type='alto', 'text', or 'json'.",
#             )
#
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
#         shutil.copyfileobj(file.file, tmp)
#         tmp_path = tmp.name
#
#     try:
#         # §4.3/§4.4: enforce the canonical upload limit (413).
#         if os.path.getsize(tmp_path) > MAX_UPLOAD_BYTES:
#             raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.")
#
#         if task_type == "alto":
#             result = text_manager.process_alto(tmp_path)
#         elif task_type == "json":
#             result = text_manager.process_json(tmp_path)
#         else:
#             result = text_manager.process_text_file(tmp_path)
#
#         result["filename"] = file.filename
#         return JSONResponse(content=result)
#
#     except HTTPException:
#         # Never re-wrap an intentional 4xx (413/422) as a 500.
#         raise
#     except Exception as exc:
#         import traceback
#
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc
#
#     finally:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)


if __name__ == "__main__":
    import logging

    import uvicorn

    # (12-factor XI) Logs are an event stream: emit to stdout and let the
    # supervisor route them. The library modules only getLogger(); this is the
    # one place allowed to configure handlers.
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )

    # (12-factor VII) The service exports itself by binding a port, and which
    # port is configuration. These were hardcoded, which also meant `reload=True`
    # — a development convenience that watches the filesystem and respawns —
    # was what docker-compose ran as the `api` profile entrypoint.
    uvicorn.run(
        "text_api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").strip().lower() in ("true", "1", "yes", "on"),
    )
