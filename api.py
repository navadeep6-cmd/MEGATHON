"""REST API for category suggestions (human-in-the-loop).

    uvicorn app.api:app --host 127.0.0.1 --port 8000

POST /suggest         multipart: image + optional title, description, date, lat, lon, top_k
POST /feedback        JSON: {query_id, category, accepted, score?, type?}
POST /feedback/apply  apply logged feedback to the serving index now (no retraining)
POST /categories      JSON: {name}  - add a new category without retraining
GET  /health

This process is the single writer of the serving index while it runs.
"""

import io
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from catsuggest.config import Paths, load_config  # noqa: E402
from catsuggest.feedback import (apply_feedback, claim_serving_writer, log_feedback,  # noqa: E402
                                 release_serving_writer)
from catsuggest.suggester import Suggester  # noqa: E402

STATE = {}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    paths = Paths(cfg).ensure()
    claim_serving_writer(paths.serve_lock)
    STATE.update(cfg=cfg, paths=paths, suggester=Suggester(cfg, memory="serve"))
    yield
    release_serving_writer(paths.serve_lock)
    STATE.clear()


app = FastAPI(title="Commons category suggester", version="1.0.0", lifespan=lifespan)


def _opt_float(value: str) -> Optional[float]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        raise HTTPException(status_code=422, detail="invalid number: %r" % value)


class Feedback(BaseModel):
    query_id: str
    category: str
    accepted: bool
    score: Optional[float] = None
    type: Optional[str] = None


class NewCategory(BaseModel):
    name: str


@app.get("/health")
def health():
    sug = STATE.get("suggester")
    return {"status": "ok" if sug else "loading",
            "categories": len(sug.store) if sug else 0,
            "indexed_images": len(sug.memory) if sug else 0}


@app.post("/suggest")
def suggest(image: UploadFile = File(...), title: str = Form(""), description: str = Form(""),
            date: str = Form(""), lat: str = Form(""), lon: str = Form(""), top_k: int = Form(10)):
    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail="not a readable image")
    top_k = max(1, min(int(top_k), 50))
    return STATE["suggester"].suggest(img, title=title, description=description, date=date,
                                      lat=_opt_float(lat), lon=_opt_float(lon), top_k=top_k, cache_query=True)


@app.post("/feedback")
def feedback(fb: Feedback):
    event = log_feedback(STATE["paths"].feedback_log, fb.query_id, fb.category, fb.accepted,
                         fb.score, fb.type, "api")
    return {"logged": event}


@app.post("/feedback/apply")
def feedback_apply():
    return apply_feedback(STATE["suggester"], STATE["paths"])


@app.post("/categories")
def add_category(cat: NewCategory):
    try:
        cid = STATE["suggester"].add_category(cat.name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    STATE["suggester"].save()
    return {"id": cid, "name": STATE["suggester"].store.names[cid]}
