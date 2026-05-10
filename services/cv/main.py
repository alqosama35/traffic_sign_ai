import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(title="CV Baseline Service", version="1.0.0")

_CV_METRICS_PATH = Path(os.environ.get(
    "CV_METRICS_PATH",
    "/app/dashboard/data/cv_metrics.json",
))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def cv_metrics():
    """Pre-computed CV baseline metrics (SIFT + Naive Bayes on 10-class subset)."""
    if not _CV_METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="cv_metrics.json not found")
    return json.loads(_CV_METRICS_PATH.read_text())
