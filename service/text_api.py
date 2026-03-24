"""
service/text_api.py
FastAPI wrapper for the ATRIUM text processing service.
"""
import os
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, Union

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Local import
from text_inference import text_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle context manager — loads models synchronously before traffic."""
    try:
        text_manager.load_models()
    except Exception as e:
        raise RuntimeError(f"Failed to initialise models on startup: {e}")
    yield


app = FastAPI(title="ATRIUM Text Processor", lifespan=lifespan)

# ---------------------------------------------------------------------------
# CORS — configurable via environment variable; defaults to localhost only
# ---------------------------------------------------------------------------
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8080,http://localhost:5500")
allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static frontend
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root() -> Union[HTMLResponse, Dict[str, str]]:
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return {"message": "Service running. Frontend not found."}


@app.get("/info")
async def info() -> Dict[str, Any]:
    return {
        "status": "active",
        "device": text_manager.device,
        "supported_formats": ["ALTO XML (.xml)", "Plain Text (.txt)"],
        "quality_categories": ["Clear", "Noisy", "Trash", "Non-text", "Empty"],
    }


@app.post("/process")
async def process_document(
    file: UploadFile = File(...),
    task_type: str = Form("auto"),
) -> JSONResponse:
    """
    Upload an ALTO XML or plain-text file.

    Returns a list of classified lines.  Each line carries:
      - line_num    : 1-based position
      - text        : cleaned text content
      - perplexity  : DistilGPT2 cross-entropy perplexity (0 for pre-filtered lines)
      - sym_count   : tokens with strange/unexpected symbols (see text_util_langID)
      - upper_count : tokens with mid-word uppercase artefacts
      - category    : one of Clear | Noisy | Trash | Non-text | Empty
    """
    filename = (file.filename or "").lower()

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
        if task_type == "alto":
            result = text_manager.process_alto(tmp_path)
        else:
            result = text_manager.process_text_file(tmp_path)

        result["filename"] = file.filename
        return JSONResponse(content=result)

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