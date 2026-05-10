import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException

app = FastAPI(title="GA Results Service", version="1.0.0")

_EA_RESULTS_DIR = Path(os.environ.get("EA_RESULTS_DIR", "/app/ea/results"))
_STRIP = {"best_chromosome", "selected_feature_indices"}


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
