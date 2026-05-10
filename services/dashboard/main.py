import json
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Dashboard Service", version="1.0.0")

# ── Internal service URLs ──────────────────────────────────────────────────────
INFERENCE_URL    = os.environ.get("INFERENCE_URL", "http://inference-svc:8001")
CV_URL           = os.environ.get("CV_URL",        "http://cv-svc:8002")
GA_URL           = os.environ.get("GA_URL",        "http://ga-svc:8003")
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "")

# ── Local file paths (mounted from host / S3 download) ────────────────────────
_REPORT_PATH    = Path(os.environ.get("REPORT_PATH",    "/app/training/outputs/logs/report.json"))
_CM_PATH        = Path(os.environ.get("CM_PATH",        "/app/training/outputs/plots/cm.png"))
_AUDIT_LOG_FILE = Path(os.environ.get("AUDIT_LOG_FILE", "/app/logs/predictions.jsonl"))
_STATIC_DIR     = Path(os.environ.get("STATIC_DIR",     "/app/dashboard"))


# ── Health — fan-out to all upstream services ──────────────────────────────────
@app.get("/health")
async def health():
    results: dict[str, str] = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in [("inference", INFERENCE_URL), ("cv", CV_URL), ("ga", GA_URL)]:
            try:
                r = await client.get(f"{url}/health")
                results[name] = "ok" if r.status_code == 200 else "degraded"
            except Exception:
                results[name] = "unreachable"
    results["dashboard"] = "ok"
    all_ok = all(v == "ok" for v in results.values())
    return JSONResponse(
        content={"status": "ok" if all_ok else "degraded", "services": results},
        status_code=200 if all_ok else 503,
    )


# ── Dashboard data endpoints ───────────────────────────────────────────────────
@app.get("/dashboard/data/ea")
async def ea_data():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{GA_URL}/experiments")
    if r.status_code != 200:
        raise HTTPException(status_code=503, detail=f"GA service error: {r.status_code}")
    return r.json()


@app.get("/dashboard/data/training-report")
def training_report():
    if not _REPORT_PATH.exists():
        raise HTTPException(status_code=503, detail="report.json not found")
    return json.loads(_REPORT_PATH.read_text())


@app.get("/dashboard/data/cv-metrics")
async def cv_metrics():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{CV_URL}/metrics")
    if r.status_code != 200:
        raise HTTPException(status_code=503, detail=f"CV service error: {r.status_code}")
    return r.json()


@app.get("/dashboard/data/confusion-matrix.png")
def confusion_matrix_png():
    if not _CM_PATH.exists():
        raise HTTPException(status_code=503, detail="Confusion matrix image not found")
    return FileResponse(str(_CM_PATH), media_type="image/png")


@app.get("/dashboard/data/recent-predictions")
def recent_predictions(limit: int = 20):
    """Last N predictions logged by the audit worker via RabbitMQ, newest first."""
    if not _AUDIT_LOG_FILE.exists():
        return []
    lines = [l for l in _AUDIT_LOG_FILE.read_text().strip().split("\n") if l]
    return list(reversed([json.loads(l) for l in lines[-limit:]]))


# ── Predict relay → inference-svc ─────────────────────────────────────────────
@app.post("/dashboard/predict")
async def dashboard_predict(file: UploadFile = File(...)):
    content = await file.read()
    headers = {"X-API-Key": INTERNAL_API_KEY} if INTERNAL_API_KEY else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            f"{INFERENCE_URL}/predict",
            files={"file": (file.filename, content, file.content_type)},
            headers=headers,
        )
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code,
                            detail=r.json().get("detail", "Inference failed"))
    return r.json()


# ── Static files (mounted last so API routes take priority) ───────────────────
if _STATIC_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(_STATIC_DIR), html=True),
              name="dashboard-static")
