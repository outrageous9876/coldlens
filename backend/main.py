import base64
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from compliance import check_compliance
from extract import extract_fields
from fda import get_label
from ocr import run_ocr
from preprocess import preprocess, save_steps
from rag import answer
from recall import check_recalls

app = FastAPI(title="ColdLens API")

DEBUG_DIR = Path(__file__).resolve().parent.parent / "data" / "debug"
STATIC_DIR = Path(__file__).resolve().parent / "static"

# eval/RESULTS.md: full_no_threshold (grayscale + denoise + perspective
# correction + deskew, no adaptive threshold) beat every other variant,
# including the full pipeline - the threshold step hurt OCR as often as
# it helped. Keep /preprocess defaulting to all steps (it's for debug
# visualization of every step), but /analyze uses the measured-best set.
ANALYZE_STEPS = ["grayscale", "denoise", "perspective_correction", "deskew"]


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


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


@app.post("/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    contents = await file.read()
    np_bytes = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)

    if image is None:
        return JSONResponse(status_code=400, content={"error": "could not decode image"})

    final_image, _ = preprocess(image, enabled_steps=ANALYZE_STEPS)
    ocr_result = run_ocr(final_image)
    fields = extract_fields(ocr_result["full_text"])
    flags = check_compliance(fields)

    ok, encoded = cv2.imencode(".png", final_image)
    preprocessed_image = f"data:image/png;base64,{base64.b64encode(encoded).decode('utf-8')}" if ok else None

    return {
        "ocr": ocr_result,
        "fields": fields.model_dump(),
        "flags": flags,
        "preprocessed_image": preprocessed_image,
    }


NO_MATCH_NOTE = "openFDA only covers US-marketed products, so labels sold elsewhere often have no match."


def _no_label(product: str) -> JSONResponse:
    return JSONResponse(status_code=404, content={
        "error": f"no openFDA label found for {product!r}",
        "note": NO_MATCH_NOTE,
    })


@app.get("/label")
def label_endpoint(product: str = Query(..., min_length=2)):
    label = get_label(product)
    if label is None:
        return _no_label(product)
    return label


class AskRequest(BaseModel):
    product_name: str = Field(..., min_length=2)
    question: str = Field(..., min_length=3)


@app.post("/ask")
def ask_endpoint(request: AskRequest):
    label = get_label(request.product_name)
    if label is None:
        return _no_label(request.product_name)
    result = answer(request.question, label)
    return {
        "answer": result["answer"],
        "found": result["found"],
        "citations": result["citations"],
        "matched_label": {key: label[key] for key in ("brand_name", "generic_name", "manufacturer", "set_id")},
    }


@app.get("/recalls")
def recalls_endpoint(product: str = Query(..., min_length=2), batch: str | None = None):
    result = check_recalls(product, batch)
    result["note"] = NO_MATCH_NOTE
    return result
