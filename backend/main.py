import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from preprocess import preprocess, save_steps

app = FastAPI(title="ColdLens API")

DEBUG_DIR = Path(__file__).resolve().parent.parent / "data" / "debug"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/preprocess")
async def preprocess_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    np_bytes = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "could not decode image"})

    _, steps = preprocess(image)

    run_dir = DEBUG_DIR / uuid.uuid4().hex
    saved_paths = save_steps(steps, run_dir)

    return {"run_id": run_dir.name, "steps": saved_paths}
