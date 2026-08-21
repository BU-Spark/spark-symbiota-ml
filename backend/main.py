import datetime
import json
import os
import tempfile

import requests
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image
from pydantic import BaseModel

from transcription.claude_sonnet import run_claude_pipeline
from transcription.doc_intelligence import run_doc_intell_pipeline
from transcription.envelope import to_middleware_envelope
from transcription.google_vision import run_google_vision_pipeline

app = FastAPI()

DOWNLOAD_TIMEOUT = 30

PIPELINES = {
    "azure": run_doc_intell_pipeline,
    "google": run_google_vision_pipeline,
    "anthropic": run_claude_pipeline,
}

CALIBRATION_MAP = {"azure": "calibration.json",
                   "google": "calibration_google.json",
                   "anthropic": "calibration_anthropic.json"}


def _ceilings(pipeline: str) -> dict:
    """Highest confidence each field can emit. Fields with no map are absent."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "transcription", CALIBRATION_MAP[pipeline])
    try:
        cal = json.load(open(path, encoding="utf-8"))
    except Exception:
        return {}
    return {f: max(y for _, y in knots) for f, knots in cal.items()}


def _fetch(url: str, path: str):
    try:
        r = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not fetch image: {e}")
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Downloaded file is not a valid image")


MAX_UPLOAD_BYTES = 40 * 1024 * 1024


def _save_upload(upload: UploadFile, path: str):
    size = 0
    with open(path, "wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Image larger than 40MB")
            f.write(chunk)
    if not size:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")


def _transcribe(pipeline: str, url: str = None, upload: UploadFile = None) -> dict:
    # Keep the routes sync: blocking work then runs in a thread, not the event loop.
    if not url and upload is None:
        raise HTTPException(status_code=400, detail="Provide either ?url= or an uploaded file")
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        if upload is not None:
            _save_upload(upload, path)
        else:
            _fetch(url, path)
        result = PIPELINES[pipeline](path)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    try:
        data = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=502,
                            detail=f"{pipeline} pipeline error: {str(result)[:200]}")
    if isinstance(data, dict) and data.get("_error"):
        raise HTTPException(status_code=502, detail=f"{pipeline} pipeline error: {data['_error']}")
    return to_middleware_envelope(data, model=pipeline)


TOOL_HTML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool.html")
# Set REVIEWS_PATH to a mounted volume in a container.
REVIEWS = os.environ.get("REVIEWS_PATH") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "transcription", "results", "reviews.jsonl")


class Review(BaseModel):
    image: str
    threshold: float | None = None
    # {pipeline: {field: {"value": str, "confidence": float | None}}}
    readings: dict = {}
    # {field: value} as the reviewer resolved it
    truth: dict = {}
    # fields the reviewer accepted unchanged
    agreed: list = []


@app.post("/reviews")
def save_review(r: Review):
    """Append one reviewed specimen to the review log."""
    if not r.truth:
        raise HTTPException(status_code=400, detail="No fields resolved")
    row = r.model_dump()
    row["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(REVIEWS), exist_ok=True)
    with open(REVIEWS, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n = sum(1 for _ in open(REVIEWS, encoding="utf-8"))
    return {"saved": True, "fields": len(r.truth), "reviews_logged": n}


@app.get("/reviews")
def review_count():
    if not os.path.exists(REVIEWS):
        return {"reviews_logged": 0, "path": REVIEWS}
    return {"reviews_logged": sum(1 for _ in open(REVIEWS, encoding="utf-8")), "path": REVIEWS}


@app.get("/")
async def root():
    # Original contract. The middleware may health-check this -- do not change it.
    return {"message": "OCR service is running", "pipelines": sorted(PIPELINES)}


@app.get("/health")
async def health():
    return {"message": "OCR service is running", "pipelines": sorted(PIPELINES)}


@app.get("/tool", response_class=HTMLResponse)
def tool():
    with open(TOOL_HTML, encoding="utf-8") as f:
        return f.read()


@app.get("/pipelines")
async def pipelines():
    """Available pipelines and their per-field confidence ceilings."""
    return {
        "pipelines": [
            {"id": name,
             "confidence_fields": sorted(_ceilings(name)),
             "confidence_ceilings": _ceilings(name)}
            for name in sorted(PIPELINES)
        ],
        "note": "A field absent from confidence_ceilings ships an uncalibrated "
                "score: it can read 1.00 without evidence for it.",
    }


@app.post("/compare")
def compare(url: str = Query(None), pipelines: str = Query(None),
            file: UploadFile = File(None)):
    """Run several pipelines over one image. `pipelines` is comma-separated."""
    wanted = [p.strip() for p in (pipelines or "").split(",") if p.strip()] or list(PIPELINES)
    unknown = [p for p in wanted if p not in PIPELINES]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown pipeline(s): {unknown}")
    if not url and file is None:
        raise HTTPException(status_code=400, detail="Provide either ?url= or an uploaded file")

    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    try:
        if file is not None:
            _save_upload(file, path)
        else:
            _fetch(url, path)
        out = {}
        for name in wanted:
            try:
                raw = PIPELINES[name](path)
                data = json.loads(raw)
                out[name] = {"ok": True, "envelope": to_middleware_envelope(data, model=name)}
            except Exception as e:
                out[name] = {"ok": False, "error": str(e)[:200]}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return {"results": out, "ceilings": {n: _ceilings(n) for n in wanted}}


@app.post("/azure")
def azure(url: str = Query(None), file: UploadFile = File(None)):
    return _transcribe("azure", url, file)


@app.post("/google")
def google(url: str = Query(None), file: UploadFile = File(None)):
    return _transcribe("google", url, file)


@app.post("/anthropic")
def anthropic(url: str = Query(None), file: UploadFile = File(None)):
    return _transcribe("anthropic", url, file)
