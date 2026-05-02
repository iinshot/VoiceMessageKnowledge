import os
import tempfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import sys
import traceback

sys.path.insert(0, "/app/Speech-to-Text-main")

app = FastAPI(title="STT Service")

MODEL_DIR = os.getenv("STT_MODEL_DIR")
SILERO_DIR = os.getenv("SILERO_DIR")
from speach_to_text import run_pipeline

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Только .wav файлы")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = run_pipeline(
            audio_path=tmp_path,
            model_path=MODEL_DIR,
            out_path=None,
            min_speakers=1,
            max_speakers=1,
            use_lm=False,
            use_punctuation=True,
        )

        segments = result.get("segments", [])
        text = " ".join(s["text"].strip() for s in segments if s.get("text", "").strip())

        return JSONResponse({"text": text})

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)