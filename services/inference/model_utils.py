import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

NUM_CLASSES = 43
AWS_REGION  = os.environ.get("AWS_REGION", "us-east-1")

# ── Class maps (copied verbatim from api/app.py) ───────────────────────────────
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
    **{i:"prohibition" for i in [0,1,2,3,4,5,7,8,9,10,14,15,16,17]},       # speed limits + no-X + stop
    **{i:"danger"      for i in [11,13,18,19,20,21,22,23,24,25,26,27,28,29,30,31]},  # triangular warning signs + yield
    **{i:"mandatory"   for i in [33,34,35,36,37,38,39,40]},                  # blue circle direction signs
    **{i:"other"       for i in [6,12,32,41,42]},                             # end-restriction + priority road
}


def clean_name(name: str) -> str:
    return name.replace(">", "").replace(" ", "_")


CLEAN_TO_DISPLAY = {clean_name(v): v for v in CLASS_NAMES_BY_ID.values()}
CLEAN_TO_ID      = {clean_name(v): k for k, v in CLASS_NAMES_BY_ID.items()}


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


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
