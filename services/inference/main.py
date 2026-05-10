from dotenv import load_dotenv
load_dotenv()

import io
import json
import math
import os
import time
import uuid
from pathlib import Path

import pika
import torch
from fastapi import BackgroundTasks, FastAPI, File, Header, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from services.inference.model_utils import (
    NUM_CLASSES,
    AWS_REGION,
    SIGN_DISPLAY_BY_ID,
    CATEGORY_BY_ID,
    CLEAN_TO_ID,
    CLEAN_TO_DISPLAY,
    _build_model,
    _maybe_download_from_s3,
    load_model,
    preprocess,
    _default_classes,
)

# ── Runtime flags ──────────────────────────────────────────────────────────────
TEST_MODE         = os.environ.get("MODEL_TEST_MODE",   "0") == "1"
ENABLE_CLOUDWATCH = os.environ.get("ENABLE_CLOUDWATCH", "0") == "1"

API_KEY          = os.environ.get("API_KEY", "").strip()
INTERNAL_API_KEY = os.environ.get("INTERNAL_API_KEY", "").strip()
RABBITMQ_URL     = os.environ.get("RABBITMQ_URL", "")

EXCHANGE = "traffic_sign"

# ── Thread tuning ──────────────────────────────────────────────────────────────
_cpu = int(os.environ.get("CPU_LIMIT", os.cpu_count() or 1))
torch.set_num_threads(_cpu)
torch.set_num_interop_threads(max(1, math.floor(_cpu / 2)))

# ── Model path ─────────────────────────────────────────────────────────────────
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[2]
            / "training" / "outputs" / "model" / "mobilenetv2_inference.pth"),
    )
)


def _load_or_stub():
    if TEST_MODE:
        print("MODEL_TEST_MODE=1 — stub model active (CI mode)")
        m = _build_model(NUM_CLASSES)
        m.eval()
        return m, torch.device("cpu"), _default_classes()
    _maybe_download_from_s3(MODEL_PATH)
    return load_model(MODEL_PATH)


# ── Auth ───────────────────────────────────────────────────────────────────────
def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    # Accept either the public key or the internal service key
    accepted = {k for k in [API_KEY, INTERNAL_API_KEY] if k}
    if x_api_key not in accepted:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── CloudWatch ─────────────────────────────────────────────────────────────────
_cw = None


def _cw_client():
    global _cw
    if _cw is None and ENABLE_CLOUDWATCH:
        try:
            import boto3
            _cw = boto3.client("cloudwatch", region_name=AWS_REGION)
        except Exception as e:
            print(f"CloudWatch unavailable: {e}")
    return _cw


def _emit(latency_ms: float, status_code: int) -> None:
    cw = _cw_client()
    if not cw:
        return
    try:
        cw.put_metric_data(
            Namespace="TrafficSignAPI",
            MetricData=[
                {"MetricName": "Latency",      "Value": latency_ms,             "Unit": "Milliseconds"},
                {"MetricName": "RequestCount", "Value": 1,                      "Unit": "Count"},
                {"MetricName": "ErrorCount",   "Value": int(status_code >= 500),"Unit": "Count"},
            ],
        )
    except Exception as e:
        print(f"CloudWatch emit error: {e}")


# ── RabbitMQ audit publish (fire-and-forget) ───────────────────────────────────
def _publish_audit_event(event: dict) -> None:
    """Non-fatal: inference must work even if the broker is down."""
    if not RABBITMQ_URL:
        return
    try:
        params = pika.URLParameters(RABBITMQ_URL)
        params.heartbeat = 30
        conn = pika.BlockingConnection(params)
        ch   = conn.channel()
        ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
        ch.exchange_declare(exchange="traffic_sign_dlx", exchange_type="direct", durable=True)
        ch.queue_declare(
            queue="prediction.audit",
            durable=True,
            arguments={
                "x-dead-letter-exchange":    "traffic_sign_dlx",
                "x-dead-letter-routing-key": "prediction.dead",
            },
        )
        ch.queue_bind(queue="prediction.audit", exchange=EXCHANGE,
                      routing_key="prediction.completed")
        ch.basic_publish(
            exchange=EXCHANGE,
            routing_key="prediction.completed",
            body=json.dumps(event),
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )
        conn.close()
    except Exception as exc:
        print(f"[inference] audit publish failed (non-fatal): {exc}", flush=True)


# ── App startup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Sign Inference Service",
    description="MobileNetV2-based GTSRB 43-class classifier.",
    version="1.0.0",
)

model, device, CLASSES = _load_or_stub()

with torch.no_grad():
    model(torch.zeros(1, 3, 224, 224).to(device))
print(f"Inference service ready. Auth: {'enabled' if API_KEY else 'disabled'}")


# ── Middleware ─────────────────────────────────────────────────────────────────
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start    = time.monotonic()
    response = await call_next(request)
    _emit((time.monotonic() - start) * 1000, response.status_code)
    return response


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "Traffic Sign Inference Service", "model": "mobilenet_v2",
            "classes": len(CLASSES), "docs": "/docs"}


@app.post("/predict")
async def predict(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)
    t0 = time.time()

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Unsupported image format") from exc

    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]

    pred_idx      = int(torch.argmax(probs).item())
    pred_label    = CLASSES[pred_idx]
    class_id      = CLEAN_TO_ID.get(pred_label)
    display_label = (
        SIGN_DISPLAY_BY_ID.get(class_id, CLEAN_TO_DISPLAY.get(pred_label, pred_label))
        if class_id is not None else CLEAN_TO_DISPLAY.get(pred_label, pred_label)
    )
    category = CATEGORY_BY_ID.get(class_id, "unknown") if class_id is not None else "unknown"

    top_3 = []
    for idx in torch.topk(probs, k=3).indices.tolist():
        lbl  = CLASSES[idx]
        cid  = CLEAN_TO_ID.get(lbl)
        disp = (
            SIGN_DISPLAY_BY_ID.get(cid, CLEAN_TO_DISPLAY.get(lbl, lbl))
            if cid is not None else CLEAN_TO_DISPLAY.get(lbl, lbl)
        )
        top_3.append({"class": disp, "confidence": round(float(probs[idx].item()), 6)})

    latency_ms = round((time.time() - t0) * 1000, 1)
    result = {
        "class":      display_label,
        "category":   category,
        "confidence": round(float(probs[pred_idx].item()), 6),
        "top_3":      top_3,
        "note":       "Screening tool only — not for use in safety-critical autonomous systems",
    }

    # Fire-and-forget: user already has their result, audit happens in background
    background_tasks.add_task(_publish_audit_event, {
        "request_id":      str(uuid.uuid4()),
        "timestamp":       t0,
        "predicted_class": result["class"],
        "category":        result["category"],
        "confidence":      result["confidence"],
        "top_3":           result["top_3"],
        "latency_ms":      latency_ms,
    })

    return result
