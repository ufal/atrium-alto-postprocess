"""
service/text_api.py
FastAPI wrapper for the text processing service.
"""
import os
import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Local import
from text_inference import text_manager

app = FastAPI(title="Atrium Text Processor")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- STATIC FILES ---
# Resolve path to frontend folder relative to this file
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
async def root():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return {"message": "Service Running. Index not found."}


@app.get("/info")
async def info():
    return {
        "status": "active",
        "device": text_manager.device,
        "supported_formats": ["ALTO XML (.xml)", "Plain Text (.txt)"]
    }


@app.post("/process")
async def process_document(
        file: UploadFile = File(...),
        task_type: str = Form("auto")
):
    """
    Upload a file (XML or TXT).
    Returns cleaned text lines with quality metrics.
    """
    filename = file.filename.lower()

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

    # Run from root of project with: python service/text_api.py
    uvicorn.run(app, host="0.0.0.0", port=8000)



    # export NODE_OPTIONS=--openssl-legacy-provider
    # npm install