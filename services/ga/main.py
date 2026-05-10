import json
import os
from pathlib import Path

import boto3
from fastapi import FastAPI, HTTPException

app = FastAPI(title="GA Results Service", version="1.0.0")

_EA_RESULTS_DIR = Path(os.environ.get("EA_RESULTS_DIR", "/app/ea/results"))
_DATA_SOURCE    = os.environ.get("DATA_SOURCE", "local")
_S3_BUCKET      = os.environ.get("S3_BUCKET", "")
_AWS_REGION     = os.environ.get("AWS_REGION", "us-east-1")
_STRIP          = {"best_chromosome", "selected_feature_indices"}


@app.on_event("startup")
async def _startup():
    if _DATA_SOURCE != "s3" or not _S3_BUCKET:
        return
    try:
        s3 = boto3.client("s3", region_name=_AWS_REGION)
        _EA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=_S3_BUCKET, Prefix="ea/results/"):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".json"):
                    dest = _EA_RESULTS_DIR / Path(key).name
                    s3.download_file(_S3_BUCKET, key, str(dest))
                    print(f"[ga] downloaded s3://{_S3_BUCKET}/{key}", flush=True)
    except Exception as exc:
        print(f"[ga] S3 unavailable, using local results: {exc}", flush=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/experiments")
def list_experiments():
    """All pre-computed GA experiment runs (large arrays stripped for the UI)."""
    if not _EA_RESULTS_DIR.exists():
        raise HTTPException(status_code=503, detail="EA results directory not found")
    runs = []
    for path in sorted(_EA_RESULTS_DIR.glob("*.json")):
        run = json.loads(path.read_text())
        for k in _STRIP:
            run.pop(k, None)
        runs.append(run)
    return runs
