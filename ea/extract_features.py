import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ── 1. Load MobileNetV2 ──────────────────────────────────────────────────────

def load_mobilenetv2(checkpoint_path):
    model = models.mobilenet_v2()
    model.classifier[1] = torch.nn.Linear(1280, 43)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ── 2. Build feature extractor ───────────────────────────────────────────────

def build_feature_extractor(model):
    def extract(x):
        with torch.no_grad():
            features = model.features(x)
            features = features.mean([2, 3])  # global average pool
        return features
    return extract

# ── 3. Extract features for one split ───────────────────────────────────────

def extract_split(split_dir, feature_fn, batch_size=64):
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageFolder(split_dir, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    all_labels = []

    for i, (images, labels) in enumerate(loader):
        feats = feature_fn(images)
        all_features.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
        print(f"  Batch {i+1}/{len(loader)} done", end='\r')

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = np.concatenate(all_labels, axis=0)
    return X, y

# ── 4. Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    checkpoint = 'training/outputs/model/mobilenetv2_inference.pth'
    train_dir  = 'dataset/train'
    val_dir    = 'dataset/val'
    out_dir    = 'ea/data'

    print("Loading MobileNetV2...")
    model = load_mobilenetv2(checkpoint)
    feature_fn = build_feature_extractor(model)

    print("\nExtracting TRAIN features...")
    t0 = time.time()
    X_train, y_train = extract_split(train_dir, feature_fn)
    print(f"\nTrain done in {time.time()-t0:.1f}s — shape: {X_train.shape}")

    print("\nExtracting VAL features...")
    t0 = time.time()
    X_val, y_val = extract_split(val_dir, feature_fn)
    print(f"\nVal done in {time.time()-t0:.1f}s — shape: {X_val.shape}")

    print("\nSaving .npy files...")
    np.save(os.path.join(out_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(out_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(out_dir, 'X_val.npy'),   X_val)
    np.save(os.path.join(out_dir, 'y_val.npy'),   y_val)
    print("Saved X_train, y_train, X_val, y_val to ea/data/")

    print("\nComputing full-feature baseline (LinearSVC)...")
    t0 = time.time()
    clf = LinearSVC(C=0.1, max_iter=2000)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    baseline_acc = accuracy_score(y_val, preds)
    print(f"Baseline accuracy: {baseline_acc:.4f} (took {time.time()-t0:.1f}s)")

    with open(os.path.join(out_dir, 'baseline_accuracy.txt'), 'w') as f:
        f.write(str(baseline_acc))
    print("Saved baseline_accuracy.txt")

    print("\n=== Phase 2 Complete ===")
    print(f"Feature dimension n = {X_train.shape[1]}")
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples:   {X_val.shape[0]}")
    print(f"Baseline val accuracy: {baseline_acc:.4f}")