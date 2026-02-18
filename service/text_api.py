import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse

# Import our manager
from text_inference import text_manager

app = FastAPI(title="Text Processing & Layout Service")


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
        task_type: str = Form("auto", description="auto, alto, or text")
):
    """
    Upload a file (XML or TXT).
    Returns cleaned text lines with quality metrics.
    """

    # 1. Validate File
    filename = file.filename.lower()

    # Auto-detect task
    if task_type == "auto":
        if filename.endswith(".xml"):
            task_type = "alto"
        elif filename.endswith(".txt"):
            task_type = "text"
        else:
            raise HTTPException(400, "Unknown file extension. Specify task_type='alto' or 'text'")

    # 2. Save to Temp (Critical for "Very Long" files)
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 3. Process based on type
        if task_type == "alto":
            result = text_manager.process_alto(tmp_path)
        else:
            result = text_manager.process_text_file(tmp_path)

        # Add metadata
        result['filename'] = file.filename

        return JSONResponse(content=result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # 4. Cleanup Temp File
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn

    # Run with: python text_api.py
    uvicorn.run(app, host="0.0.0.0", port=8000)