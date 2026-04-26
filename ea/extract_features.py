from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import mobilenet_v2
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tqdm import tqdm

NUM_CLASSES = 43


def load_mobilenetv2(checkpoint_path: str | Path) -> nn.Module:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(checkpoint)}")

    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Support checkpoints saved from DataParallel.
        stripped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        model.load_state_dict(stripped)

    model.eval()
    return model


def build_feature_extractor(model: nn.Module):
    """Return a callable that extracts global-average-pooled backbone features.

    Output shape: (batch, feature_dim) — no classifier head involved.
    feature_dim is read from the model architecture, never hard-coded.
    """
    try:
        feature_dim = model.features[-1][0].out_channels
    except (AttributeError, IndexError):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feature_dim = int(model.features(dummy).mean([2, 3]).shape[1])

    def extract(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return model.features(x).mean([2, 3])

    extract.feature_dim = feature_dim
    return extract


_VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def extract_split(
    split_dir: str | Path,
    feature_fn,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract backbone features for every image in split_dir.

    Returns:
        X: float32 array of shape (N, feature_dim)
        y: int64 array of shape (N,)
    """
    dataset = ImageFolder(str(split_dir), transform=_VAL_TRANSFORMS)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    n_samples = len(dataset)
    desc = Path(split_dir).name
    with tqdm(loader, desc=f"Extracting {desc}", unit="batch",
              total=len(loader), postfix={"samples": n_samples}) as pbar:
        for images, labels in pbar:
            features = feature_fn(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0)
    return X, y


def apply_variance_filter(
    X: np.ndarray,
    target: int = 900,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep the top-`target` features ranked by variance across training samples.

    Uses a pure-numpy argsort on column variances — no sklearn dependency here.
    Returns exactly `target` features (or all features if X already has <= target).

    Args:
        X: float32 array of shape (N, n) — full feature matrix (training set).
        target: number of features to keep.

    Returns:
        X_filtered: float32 array of shape (N, target)
        mask: bool array of shape (n,) — True at the kept feature positions.
    """
    n = X.shape[1]
    if n <= target:
        return X.astype(np.float32), np.ones(n, dtype=bool)

    variances = X.var(axis=0)
    top_indices = np.argsort(variances)[-target:]
    mask = np.zeros(n, dtype=bool)
    mask[top_indices] = True
    return X[:, mask].astype(np.float32), mask


if __name__ == "__main__":
    _REPO_ROOT = Path(__file__).resolve().parents[1]
    _CHECKPOINT = _REPO_ROOT / "training" / "outputs" / "model" / "mobilenetv2_inference.pth"
    _DATA_DIR = Path(__file__).resolve().parent / "data"
    _DATA_DIR.mkdir(exist_ok=True)

    print("Loading MobileNetV2 checkpoint...")
    _model = load_mobilenetv2(_CHECKPOINT)
    _extract = build_feature_extractor(_model)
    print(f"Feature dimension n = {_extract.feature_dim}")

    t0 = time.time()

    print("\nExtracting train features...")
    X_train, y_train = extract_split(_REPO_ROOT / "dataset" / "train", _extract)
    np.save(_DATA_DIR / "X_train.npy", X_train)
    np.save(_DATA_DIR / "y_train.npy", y_train)
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")

    print("Extracting val features...")
    X_val, y_val = extract_split(_REPO_ROOT / "dataset" / "val", _extract)
    np.save(_DATA_DIR / "X_val.npy", X_val)
    np.save(_DATA_DIR / "y_val.npy", y_val)
    print(f"  X_val:   {X_val.shape}  y_val:   {y_val.shape}")

    print(f"\nExtraction time: {time.time() - t0:.1f}s")

    # Full-feature baseline (FR-EA-16)
    print("\nComputing full-feature baseline (LinearSVC)...")
    t1 = time.time()
    clf = LinearSVC(C=0.1, max_iter=2000)
    clf.fit(X_train, y_train)
    baseline_acc = float(accuracy_score(y_val, clf.predict(X_val)))
    print(f"Baseline val accuracy: {baseline_acc:.4f}  ({time.time() - t1:.1f}s)")

    (_DATA_DIR / "baseline_accuracy.txt").write_text(str(baseline_acc))
    print(f"Saved baseline_accuracy.txt → {baseline_acc}")
