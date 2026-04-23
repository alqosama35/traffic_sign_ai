from pathlib import Path
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v2


NUM_CLASSES = 43
MODEL_PATH = Path(__file__).resolve().parents[1] / "training" / "outputs" / "model" / "mobilenetv2_inference.pth"


# Canonical GTSRB names used in data preparation.
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


def _default_classes() -> list[str]:
    # ImageFolder assigns indices by sorted folder names.
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
            classes = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Support checkpoints saved from DataParallel.
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(stripped)

    model = model.to(device)
    model.eval()

    return model, device, classes


preprocess = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

app = FastAPI(title="Traffic Sign AI API")
model, device, CLASSES = load_model(MODEL_PATH)


@app.get("/")
def root():
    return {
        "service": "Traffic Sign AI API",
        "model": "mobilenet_v2",
        "num_classes": len(CLASSES),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file.") from exc

    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(torch.max(probs).item())
    pred_label = CLASSES[pred_idx]
    class_id = CLEAN_TO_ID.get(pred_label)

    if class_id is None:
        display_label = CLEAN_TO_DISPLAY.get(pred_label, pred_label)
        category = "unknown"
    else:
        display_label = SIGN_DISPLAY_BY_ID.get(class_id, CLEAN_TO_DISPLAY.get(pred_label, pred_label))
        category = CATEGORY_BY_ID.get(class_id, "unknown")

    return {
        "sign": display_label,
        "category": category,
        "confidence": confidence,
    }


# Run: uvicorn api.app:app --reload --port 8000
# Docs: http://localhost:8000/docs
