"""
service/text_api.py
FastAPI wrapper for the ATRIUM text processing service.
"""

import os
import shutil
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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
        supported_formats=["ALTO XML (.xml)", "Plain Text (.txt)"],
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
) -> JSONResponse:
    """
    Upload an ALTO XML or plain-text file.

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
    # §4.4: missing upload metadata is a client error (422), not a server 500.
    if not file.filename:
        raise HTTPException(status_code=422, detail="Filename is missing from the upload.")
    if not file.content_type:
        raise HTTPException(status_code=422, detail="Content-Type is missing from the upload.")

    filename = file.filename.lower()

    if task_type == "auto":
        if filename.endswith(".xml"):
            task_type = "alto"
        elif filename.endswith(".txt"):
            task_type = "text"
        else:
            raise HTTPException(
                status_code=400,
                detail="Cannot auto-detect file type. Set task_type='alto' or 'text'.",
            )

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # §4.3/§4.4: enforce the canonical upload limit (413).
        if os.path.getsize(tmp_path) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.")

        if task_type == "alto":
            result = text_manager.process_alto(tmp_path)
        else:
            result = text_manager.process_text_file(tmp_path)

        result["filename"] = file.filename
        return JSONResponse(content=result)

    except HTTPException:
        # Never re-wrap an intentional 4xx (413/422) as a 500.
        raise
    except Exception as exc:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("text_api:app", host="0.0.0.0", port=8000, reload=True)
