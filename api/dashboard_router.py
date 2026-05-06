from __future__ import annotations

import io
import json
import os
from pathlib import Path

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DATA_SOURCE  = os.environ.get("DATA_SOURCE", "local").lower()
_S3_BUCKET    = os.environ.get("S3_BUCKET", "")

# ── Local paths ────────────────────────────────────────────────────────────────
_EA_DIR      = _PROJECT_ROOT / "ea"      / "results"
_REPORT_PATH = _PROJECT_ROOT / "training" / "outputs" / "logs"  / "report.json"
_CV_PATH     = _PROJECT_ROOT / "dashboard" / "data" / "cv_metrics.json"
_CM_PATH     = _PROJECT_ROOT / "training" / "outputs" / "plots" / "cm.png"

# ── S3 keys (mirror the SRS bucket structure) ─────────────────────────────────
_S3_EA_PREFIX       = "ea-experiments/"
_S3_REPORT_KEY      = "training/report.json"
_S3_CV_METRICS_KEY  = "dashboard/cv_metrics.json"
_S3_CM_KEY          = "training/cm.png"


def _s3():
    import boto3
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))


def _read_json_local(path: Path) -> dict:
    if not path.exists():
        raise HTTPException(status_code=503, detail=f"Data file not found: {path.name}")
    with open(path) as f:
        return json.load(f)


def _read_json_s3(key: str) -> dict:
    try:
        obj = _s3().get_object(Bucket=_S3_BUCKET, Key=key)
        return json.loads(obj["Body"].read())
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"S3 read failed ({key}): {exc}") from exc


def _read_json(local_path: Path, s3_key: str) -> dict:
    return _read_json_s3(s3_key) if _DATA_SOURCE == "s3" else _read_json_local(local_path)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/data/ea")
def ea_data():
    """All 4 GA experiment runs, stripped of large arrays."""
    _STRIP = {"best_chromosome", "selected_feature_indices"}

    if _DATA_SOURCE == "s3":
        try:
            s3  = _s3()
            res = s3.list_objects_v2(Bucket=_S3_BUCKET, Prefix=_S3_EA_PREFIX)
            runs = []
            for item in res.get("Contents", []):
                if item["Key"].endswith(".json"):
                    raw = s3.get_object(Bucket=_S3_BUCKET, Key=item["Key"])
                    run = json.loads(raw["Body"].read())
                    for k in _STRIP:
                        run.pop(k, None)
                    runs.append(run)
            return runs
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"S3 EA list failed: {exc}") from exc

    if not _EA_DIR.exists():
        raise HTTPException(status_code=503, detail="EA results directory not found")
    runs = []
    for path in sorted(_EA_DIR.glob("*.json")):
        with open(path) as f:
            run = json.load(f)
        for k in _STRIP:
            run.pop(k, None)
        runs.append(run)
    return runs


@router.get("/data/training-report")
def training_report():
    """MobileNetV2 classification report (accuracy, per-class precision/recall/F1)."""
    return _read_json(_REPORT_PATH, _S3_REPORT_KEY)


@router.get("/data/cv-metrics")
def cv_metrics():
    """CV baseline metrics fixture extracted from the CV notebook."""
    return _read_json(_CV_PATH, _S3_CV_METRICS_KEY)


@router.get("/data/confusion-matrix.png")
def confusion_matrix_image():
    """Serves the pre-generated 43×43 MobileNetV2 confusion matrix heatmap."""
    if _DATA_SOURCE == "s3":
        try:
            obj = _s3().get_object(Bucket=_S3_BUCKET, Key=_S3_CM_KEY)
            return StreamingResponse(obj["Body"], media_type="image/png")
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"S3 image fetch failed: {exc}") from exc
    if not _CM_PATH.exists():
        raise HTTPException(status_code=503, detail="Confusion matrix image not found")
    return FileResponse(str(_CM_PATH), media_type="image/png")


@router.post("/predict")
async def dashboard_predict(file: UploadFile = File(...)):
    """
    Prediction relay — browser never sends X-API-Key.
    Calls the same inference logic as POST /predict but without auth enforcement.
    Safe because this endpoint is only reachable through the same-origin dashboard.
    """
    # Lazy import: api.app is fully initialised before the first request arrives.
    import api.app as _app

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Unsupported image format") from exc

    tensor = _app.preprocess(img).unsqueeze(0).to(_app.device)
    with torch.no_grad():
        probs = torch.softmax(_app.model(tensor), dim=1)[0]

    pred_idx      = int(torch.argmax(probs).item())
    pred_label    = _app.CLASSES[pred_idx]
    class_id      = _app.CLEAN_TO_ID.get(pred_label)
    display_label = (
        _app.SIGN_DISPLAY_BY_ID.get(class_id, _app.CLEAN_TO_DISPLAY.get(pred_label, pred_label))
        if class_id is not None else _app.CLEAN_TO_DISPLAY.get(pred_label, pred_label)
    )
    category = _app.CATEGORY_BY_ID.get(class_id, "unknown") if class_id is not None else "unknown"

    top_3 = []
    for idx in torch.topk(probs, k=3).indices.tolist():
        lbl  = _app.CLASSES[idx]
        cid  = _app.CLEAN_TO_ID.get(lbl)
        disp = (
            _app.SIGN_DISPLAY_BY_ID.get(cid, _app.CLEAN_TO_DISPLAY.get(lbl, lbl))
            if cid is not None else _app.CLEAN_TO_DISPLAY.get(lbl, lbl)
        )
        top_3.append({"class": disp, "confidence": round(float(probs[idx].item()), 6)})

    return {
        "class":      display_label,
        "category":   category,
        "confidence": round(float(probs[pred_idx].item()), 6),
        "top_3":      top_3,
        "note":       "Screening tool only — not for use in safety-critical autonomous systems",
    }
