import json
import os
from pathlib import Path

import boto3
from fastapi import FastAPI, HTTPException

app = FastAPI(title="CV Baseline Service", version="1.0.0")

_CV_METRICS_PATH = Path(os.environ.get(
    "CV_METRICS_PATH",
    "/app/dashboard/data/cv_metrics.json",
))
_DATA_SOURCE = os.environ.get("DATA_SOURCE", "local")
_S3_BUCKET   = os.environ.get("S3_BUCKET", "")
_AWS_REGION  = os.environ.get("AWS_REGION", "us-east-1")


@app.on_event("startup")
async def _startup():
    if _DATA_SOURCE != "s3" or not _S3_BUCKET:
        return
    try:
        _CV_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        boto3.client("s3", region_name=_AWS_REGION).download_file(
            _S3_BUCKET, "dashboard/data/cv_metrics.json", str(_CV_METRICS_PATH))
        print("[cv] downloaded cv_metrics.json from S3", flush=True)
    except Exception as exc:
        print(f"[cv] S3 unavailable, using local file: {exc}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def cv_metrics():
    """Pre-computed CV baseline metrics (SIFT + Naive Bayes on 10-class subset)."""
    if not _CV_METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="cv_metrics.json not found")
    return json.loads(_CV_METRICS_PATH.read_text())
