from dotenv import load_dotenv
load_dotenv()  # ← MUST be before every other import so os.environ reads .env

import io
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from torchvision.models import mobilenet_v2

# ── Runtime flags ──────────────────────────────────────────────────────────────
# Set these in your .env file locally or as env vars in production.
# MODEL_TEST_MODE=1  → random-weight stub model, no .pth file needed (CI only)
# ENABLE_CLOUDWATCH=1→ emit metrics to CloudWatch (production only)
TEST_MODE         = os.environ.get("MODEL_TEST_MODE",   "0") == "1"
ENABLE_CLOUDWATCH = os.environ.get("ENABLE_CLOUDWATCH", "0") == "1"
AWS_REGION        = os.environ.get("AWS_REGION", "us-east-1")

# API_KEY: if empty or not set → auth is disabled (fine for local dev).
#          if set              → every /predict call must supply it.
# There is no DEV_MODE flag — empty API_KEY IS dev mode. Simple and safe.
API_KEY = os.environ.get("API_KEY", "").strip()

# ── Thread tuning ──────────────────────────────────────────────────────────────
_cpu = int(os.environ.get("CPU_LIMIT", os.cpu_count() or 1))
torch.set_num_threads(_cpu)
torch.set_num_interop_threads(max(1, math.floor(_cpu / 2)))

# ── Model path ─────────────────────────────────────────────────────────────────
NUM_CLASSES = 43
MODEL_PATH  = Path(
    os.environ.get(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[1]
            / "training" / "outputs" / "model" / "mobilenetv2_inference.pth"),
    )
)

# ── Class maps ─────────────────────────────────────────────────────────────────
CLASS_NAMES_BY_ID = {
    0:"Speed limit 20", 1:"Speed limit 30", 2:"Speed limit 50",
    3:"Speed limit 60", 4:"Speed limit 70", 5:"Speed limit 80",
    6:"End speed 80",   7:"Speed limit 100",8:"Speed limit 120",
    9:"No passing",     10:"No passing >3.5t", 11:"Right of way jn.",
    12:"Priority road", 13:"Yield",         14:"Stop",
    15:"No vehicles",   16:"No veh >3.5t",  17:"No entry",
    18:"General caution",19:"Danger curve L",20:"Danger curve R",
    21:"Double curve",  22:"Bumpy road",    23:"Slippery road",
    24:"Narrow road R", 25:"Road works",    26:"Traffic signals",
    27:"Pedestrians",   28:"Children crossing",29:"Bicycles crossing",
    30:"Beware of ice", 31:"Wild animals",  32:"End restrictions",
    33:"Turn right ahead",34:"Turn left ahead",35:"Ahead only",
    36:"Ahead or right",37:"Ahead or left", 38:"Keep right",
    39:"Keep left",     40:"Roundabout",    41:"End no passing",
    42:"End no pass >3.5t",
}

SIGN_DISPLAY_BY_ID = {
    0:"Speed limit (20km/h)",   1:"Speed limit (30km/h)",
    2:"Speed limit (50km/h)",   3:"Speed limit (60km/h)",
    4:"Speed limit (70km/h)",   5:"Speed limit (80km/h)",
    6:"End of speed limit (80km/h)", 7:"Speed limit (100km/h)",
    8:"Speed limit (120km/h)",  9:"No passing",
    10:"No passing for vehicles over 3.5 tons",
    11:"Right-of-way at the next intersection",
    12:"Priority road",         13:"Yield",
    14:"Stop",                  15:"No vehicles",
    16:"No vehicles over 3.5 tons", 17:"No entry",
    18:"General caution",       19:"Dangerous curve to the left",
    20:"Dangerous curve to the right", 21:"Double curve",
    22:"Bumpy road",            23:"Slippery road",
    24:"Road narrows on the right", 25:"Road works",
    26:"Traffic signals",       27:"Pedestrians",
    28:"Children crossing",     29:"Bicycles crossing",
    30:"Beware of ice/snow",    31:"Wild animals crossing",
    32:"End of all speed and passing restrictions",
    33:"Turn right ahead",      34:"Turn left ahead",
    35:"Ahead only",            36:"Go straight or right",
    37:"Go straight or left",   38:"Keep right",
    39:"Keep left",             40:"Roundabout mandatory",
    41:"End of no passing",
    42:"End of no passing by vehicles over 3.5 tons",
}

CATEGORY_BY_ID = {
    **{i:"prohibition" for i in [0,1,2,3,4,5,7,8,9,10,15,16,17]},
    **{i:"danger"      for i in [18,19,20,21,22,23,24,25,26,27,28,29,30,31]},
    **{i:"mandatory"   for i in [33,34,35,36,37,38,39,40]},
    **{i:"other"       for i in [6,11,12,13,14,32,41,42]},
}

def clean_name(name: str) -> str:
    return name.replace(">", "").replace(" ", "_")

CLEAN_TO_DISPLAY = {clean_name(v): v for v in CLASS_NAMES_BY_ID.values()}
CLEAN_TO_ID      = {clean_name(v): k for k, v in CLASS_NAMES_BY_ID.items()}

# ── Model ──────────────────────────────────────────────────────────────────────
def _default_classes() -> list[str]:
    return sorted(clean_name(n) for n in CLASS_NAMES_BY_ID.values())

def _build_model(num_classes: int) -> nn.Module:
    m = mobilenet_v2(weights=None)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m

def _maybe_download_from_s3(local_path: Path) -> None:
    model_env = os.environ.get("MODEL_PATH", "")
    if not model_env.startswith("s3://"):
        return
    try:
        import boto3
        without_scheme = model_env[5:]
        bucket, _, key = without_scheme.partition("/")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading model from s3://{bucket}/{key} ...")
        boto3.client("s3", region_name=AWS_REGION).download_file(bucket, key, str(local_path))
        print("Model downloaded.")
    except Exception as exc:
        raise RuntimeError(f"S3 model download failed: {exc}") from exc

def load_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model not found: {checkpoint_path}")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model      = _build_model(NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes    = _default_classes()
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict   = checkpoint["model_state_dict"]
        class_to_idx = checkpoint.get("class_to_idx")
        if isinstance(class_to_idx, dict):
            classes = [n for n, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(stripped)
    return model.to(device).eval(), device, classes

def _load_or_stub():
    if TEST_MODE:
        print("MODEL_TEST_MODE=1 — stub model active (CI mode)")
        m = _build_model(NUM_CLASSES)
        m.eval()
        return m, torch.device("cpu"), _default_classes()
    _maybe_download_from_s3(MODEL_PATH)
    return load_model(MODEL_PATH)

# ── Preprocessing ──────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Auth ───────────────────────────────────────────────────────────────────────
# Rule: if API_KEY is empty → auth is off (local dev).
#       if API_KEY is set   → enforce it, return 401 on mismatch.
#       Never return 500 for auth reasons.
def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return   # no key configured → open access (local dev / CI)
    if x_api_key != API_KEY:
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

# ── App startup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Sign AI API",
    description="MobileNetV2-based GTSRB 43-class classifier.",
    version="1.0.0",
)

model, device, CLASSES = _load_or_stub()

# Warmup: eliminates first-request latency spike
with torch.no_grad():
    model(torch.zeros(1, 3, 224, 224).to(device))
print(f"API ready. Auth: {'enabled' if API_KEY else 'disabled (no API_KEY set)'}")

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
    return {"service": "Traffic Sign AI API", "model": "mobilenet_v2",
            "classes": len(CLASSES), "docs": "/docs"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    _check_api_key(x_api_key)

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

    return {
        "class":      display_label,
        "category":   category,
        "confidence": round(float(probs[pred_idx].item()), 6),
        "top_3":      top_3,
        "note":       "Screening tool only — not for use in safety-critical autonomous systems",
    }

# Local:  uvicorn api.app:app --reload --port 8000
# Docs:   http://localhost:8000/docs