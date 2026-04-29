import io
import os
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
  # parallelize across ops
# ── Constants ─────────────────────────────────────────────────────────────────
torch.set_num_threads(os.cpu_count())          # parallelize within one inference
torch.set_num_interop_threads(os.cpu_count())
NUM_CLASSES = 43

# FR-CI-02 / NFR-MAINT-03: MODEL_PATH must come from an environment variable,
# never be hardcoded. Falls back to the local training output for convenience
# when running without Docker (e.g. `uvicorn api.app:app --reload`).
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(
            Path(__file__).resolve().parents[1]
            / "training"
            / "outputs"
            / "model"
            / "mobilenetv2_inference.pth"
        ),
    )
)

# NFR-SEC-01: API key read from environment variable.
# Set API_KEY in your .env file locally or as a GitHub secret in CI.
# If API_KEY is empty the check is skipped — useful for local dev without auth.
API_KEY = os.environ.get("API_KEY", "")


# ── Class maps ────────────────────────────────────────────────────────────────
CLASS_NAMES_BY_ID = {
    0: "Speed limit 20",
    1: "Speed limit 30",
    2: "Speed limit 50",
    3: "Speed limit 60",
    4: "Speed limit 70",
    5: "Speed limit 80",
    6: "End speed 80",
    7: "Speed limit 100",
    8: "Speed limit 120",
    9: "No passing",
    10: "No passing >3.5t",
    11: "Right of way jn.",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "No veh >3.5t",
    17: "No entry",
    18: "General caution",
    19: "Danger curve L",
    20: "Danger curve R",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Narrow road R",
    25: "Road works",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice",
    31: "Wild animals",
    32: "End restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Ahead or right",
    37: "Ahead or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout",
    41: "End no passing",
    42: "End no pass >3.5t",
}

SIGN_DISPLAY_BY_ID = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "No vehicles over 3.5 tons",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road works",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing restrictions",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 tons",
}

CATEGORY_BY_ID = {
    **{i: "prohibition" for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16, 17]},
    **{i: "danger" for i in [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]},
    **{i: "mandatory" for i in [33, 34, 35, 36, 37, 38, 39, 40]},
    **{i: "other" for i in [6, 11, 12, 13, 14, 32, 41, 42]},
}


def clean_name(name: str) -> str:
    return name.replace(">", "").replace(" ", "_")


CLEAN_TO_DISPLAY = {clean_name(v): v for v in CLASS_NAMES_BY_ID.values()}
CLEAN_TO_ID = {clean_name(v): k for k, v in CLASS_NAMES_BY_ID.items()}


# ── Model helpers ─────────────────────────────────────────────────────────────
def _default_classes() -> list[str]:
    return sorted(clean_name(name) for name in CLASS_NAMES_BY_ID.values())


def _build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def load_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model file not found at: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    classes = _default_classes()

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_to_idx = checkpoint.get("class_to_idx")
        if isinstance(class_to_idx, dict):
            classes = [
                name
                for name, _ in sorted(
                    class_to_idx.items(), key=lambda item: item[1]
                )
            ]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(stripped)

    model = model.to(device)
    model.eval()
    return model, device, classes


# ── Preprocessing ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ── Auth helper ───────────────────────────────────────────────────────────────
def _check_api_key(x_api_key: str | None) -> None:
    """
    NFR-SEC-01: Reject requests without a valid X-API-Key header.
    If API_KEY env var is empty (local dev), auth is skipped entirely.
    """
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── App + model startup ───────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Sign AI API",
    description="MobileNetV2-based traffic sign classifier — GTSRB 43 classes.",
    version="1.0.0",
)

model, device, CLASSES = load_model(MODEL_PATH)

# Warmup: run one dummy inference at startup to pre-initialize
# PyTorch's CPU thread pool. Without this, the first real request
# pays a 1-2s penalty. After warmup, CPU inference is ~200-400ms.
_warmup_tensor = torch.zeros(1, 3, 224, 224).to(device)
with torch.no_grad():
    model(_warmup_tensor)
del _warmup_tensor


# ── Endpoints ─────────────────────────────────────────────────────────────────

# FR-API-02: Health check — always returns 200 {"status": "ok"}
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "service": "Traffic Sign AI API",
        "model": "mobilenet_v2",
        "num_classes": len(CLASSES),
        "docs": "/docs",
    }


# FR-API-01: POST /predict — accepts image, returns class + confidence + top_3
# FR-API-03: Response always includes the safety disclaimer note
# FR-API-04: Images are processed in memory and never persisted
# NFR-SEC-01: X-API-Key header required (skipped when API_KEY env var is empty)
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    x_api_key: str | None = Header(default=None),
):
    # Auth check
    _check_api_key(x_api_key)

    # Validate and decode image entirely in memory — FR-API-04
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=422, detail="Unsupported image format") from exc

    # Run inference
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    # Top prediction
    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())
    pred_label = CLASSES[pred_idx]
    class_id = CLEAN_TO_ID.get(pred_label)

    if class_id is None:
        display_label = CLEAN_TO_DISPLAY.get(pred_label, pred_label)
        category = "unknown"
    else:
        display_label = SIGN_DISPLAY_BY_ID.get(
            class_id, CLEAN_TO_DISPLAY.get(pred_label, pred_label)
        )
        category = CATEGORY_BY_ID.get(class_id, "unknown")

    # FR-API-01: top_3 — get 3 highest-confidence predictions
    top3_indices = torch.topk(probs, k=3).indices.tolist()
    top_3 = []
    for idx in top3_indices:
        lbl = CLASSES[idx]
        cid = CLEAN_TO_ID.get(lbl)
        if cid is None:
            disp = CLEAN_TO_DISPLAY.get(lbl, lbl)
        else:
            disp = SIGN_DISPLAY_BY_ID.get(cid, CLEAN_TO_DISPLAY.get(lbl, lbl))
        top_3.append({"class": disp, "confidence": float(probs[idx].item())})

    # FR-API-01 + FR-API-03: full SRS-compliant response
    return {
        "class": display_label,
        "category": category,
        "confidence": confidence,
        "top_3": top_3,
        "note": "Screening tool only — not for use in safety-critical autonomous systems",
    }


# Run locally:  uvicorn api.app:app --reload --port 8000
# Docs:         http://localhost:8000/docs