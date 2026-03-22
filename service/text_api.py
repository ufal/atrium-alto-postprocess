"""
service/text_api.py
FastAPI wrapper for the text processing service.
"""
import os
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Local import
from text_inference import text_manager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle context manager to load models synchronously on startup."""
    # Pre-load heavy models into memory before accepting external traffic
    try:
        text_manager.load_models()
    except Exception as e:
        # If models fail to load, the service shouldn't start
        raise RuntimeError(f"Failed to initialize models during startup: {e}")
    yield
    # Clean up resources on shutdown if necessary

app = FastAPI(title="Atrium Text Processor", lifespan=lifespan)

# CORS setup - Secure configuration via environment variables
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES ---
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
async def root() -> Union[HTMLResponse, Dict[str, str]]:
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"message": "Service Running. Index not found."}

@app.get("/info")
async def info() -> Dict[str, Any]:
    return {
        "status": "active",
        "device": text_manager.device,
        "supported_formats": ["ALTO XML (.xml)", "Plain Text (.txt)"]
    }

@app.post("/process")
async def process_document(
        file: UploadFile = File(...),
        task_type: str = Form("auto")
) -> JSONResponse:
    """
    Upload a file (XML or TXT).
    Returns cleaned text lines with quality metrics.
    """
    filename = (file.filename or "").lower()

    # Auto-detect logic
    if task_type == "auto":
        if filename.endswith(".xml"):
            task_type = "alto"
        elif filename.endswith(".txt"):
            task_type = "text"
        else:
            raise HTTPException(400, "Unknown file extension. Specify task_type='alto' or 'text'")

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if task_type == "alto":
            result = text_manager.process_alto(tmp_path)
        else:
            result = text_manager.process_text_file(tmp_path)

        result['filename'] = file.filename
        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("text_api:app", host="0.0.0.0", port=8000, reload=True)