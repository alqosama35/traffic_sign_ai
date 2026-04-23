# Codebase Review — Traffic Sign Intelligence Platform
**Reviewer:** AI Engineer  
**Date:** 2026-04-23  
**SRS Version:** 1.1  
**Branch:** main

---

## 1. Executive Summary

The codebase is in **early-phase development**. Only Phase 1 (data pipeline) and part of Phase 2 (ML training for MobileNetV2 only) are implemented. Five of the seven member work packages are entirely absent. Several SRS requirements are directly violated in the existing code, and multiple bugs exist that must be fixed before the project can advance.

---

## 2. Implementation Status by Module

| Module | SRS Requirements | Status | Notes |
|---|---|---|---|
| Data Pipeline | FR-D-01 → FR-D-08 | ⚠ Partial | FR-D-07, FR-D-08 missing; FR-D-01 ID wrong |
| CV Baseline | FR-CV-01 → FR-CV-09 | ❌ Not Started | No files exist |
| Transfer Learning | FR-AML-01 → FR-AML-13 | ⚠ Partial | Only MobileNetV2; 7 of 13 requirements missing |
| GA Engine | FR-EA-01 → FR-EA-16 | ❌ Not Started | No files exist |
| Inference API | FR-API-01 → FR-API-07 | ❌ Not Started | No files exist |
| Containerization / CI-CD | FR-CI-01 → FR-CI-05 | ❌ Not Started | No Dockerfile, no workflow |
| Cloud + Dashboard | FR-CLD-01 → FR-CLD-07, FR-DASH-01 → FR-DASH-06 | ❌ Not Started | No S3, no CloudWatch, no dashboard |

---

## 3. SRS Compliance Violations

### 3.1 FR-D-01 — Wrong Kaggle Dataset ID
**File:** `setup_dataset.py:49`

```python
# Current (WRONG)
path = kagglehub.dataset_download(
    "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
)

# SRS requires
path = kagglehub.dataset_download(
    "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign-recognition-benchmark"
)
```

The truncated slug is a different Kaggle resource. This may silently download the wrong dataset or fail at runtime.

**Fix:** Update the slug to match exactly what is stated in FR-D-01.

---

### 3.2 FR-D-02 — Train/Val/Test Split Ratio Does Not Match 70/15/15
**File:** `setup_dataset.py:66`, `training/training.ipynb`

The code splits only the Kaggle training set (70% train + 15% val from it) and uses the Kaggle test set as-is. The resulting distribution based on actual output is:

| Split | Images | Actual % |
|---|---|---|
| Train | 33,318 | 64.3% |
| Val | 5,891 | 11.4% |
| Test | 12,630 | 24.4% |

SRS FR-D-02 requires **70% / 15% / 15%**. The test set is 24.4%, not 15%.

**Fix:** Either sub-sample the Kaggle test set to 15% of total, or re-split the entire combined dataset (train+test together) into 70/15/15 using `random.seed(42)` for reproducibility.

---

### 3.3 FR-AML-03 — Backbone Unfreeze Strategy Incorrect
**File:** `training/training.ipynb`, cell-13 and the unfreeze inside the loop

SRS requires: *"freeze the backbone for the first 5 epochs, then unfreeze the **last 2 blocks** for fine-tuning"*

Current implementation unfreezes **the entire backbone**:
```python
# Freeze (correct)
for param in model.features.parameters():
    param.requires_grad = False

# Unfreeze at epoch 5 (WRONG — unfreezes everything)
if epoch == 4:
    for param in model.features.parameters():
        param.requires_grad = True
```

MobileNetV2's `model.features` has 19 sub-modules (indices 0–18). Only the last 2 blocks should be unfrozen:
```python
# Correct fix
if epoch == 4:
    for param in model.features[16:].parameters():  # last 2 blocks
        param.requires_grad = True
```

This is an academic requirement violation (FR-AML-03) and also affects the architecture comparison table.

---

### 3.4 FR-AML-07 — Model Saved with Wrong Filename
**File:** `training/training.ipynb`, cell-27

SRS requires the model to be saved as `traffic_sign_model_mobilenetv2.pth`. The notebook saves it as `mobilenetv2_inference.pth`:

```python
# Current (WRONG)
torch.save(model.state_dict(), "outputs/model/mobilenetv2_inference.pth")

# SRS requires (FR-AML-07)
torch.save(model.state_dict(), "outputs/model/traffic_sign_model_mobilenetv2.pth")
```

This will break the S3 upload path (FR-CLD-02), the API model-load step (FR-CLD-03), and the CI/CD pipeline.

---

### 3.5 FR-AML-02 + FR-AML-13 — EfficientNet-B0 Not Trained
**File:** `training/training.ipynb`

The SRS explicitly requires both models to be trained and compared:
- `traffic_sign_model_efficientnetb0.pth` must exist (FR-AML-07)
- Architecture comparison table: MobileNetV2 vs EfficientNet-B0 accuracy + latency (FR-AML-13)

Neither the training code nor any checkpoint for EfficientNet-B0 exists. This is a hard requirement — it gates the four-way model comparison table on the dashboard (FR-DASH-04).

---

### 3.6 FR-AML-12 — Top-5 Confused Pairs Not Extracted
**File:** `training/training.ipynb`

The confusion matrix is saved as an image (`cm.png`) but no code identifies the top-5 most confused sign pairs programmatically and exports them. This analysis is required for both the training report and the dashboard.

**Fix:** After computing `cm`, add:
```python
np.fill_diagonal(cm, 0)
confused_pairs = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1], cm.shape))[0][:5]
```

---

### 3.7 FR-D-07 + FR-D-08 — S3 Upload and Augmentation Comparison Missing
**File:** `setup_dataset.py`, `training/training.ipynb`

- **FR-D-07:** No S3 upload step for the processed dataset. No boto3 usage anywhere.
- **FR-D-08:** No second training run without augmentation and no accuracy delta report. The `config.json` does not record whether augmentation was on or off.

---

### 3.8 NFR-MAINT-01 — Directory Structure Doesn't Match SRS
**Project root**

SRS (NFR-MAINT-01) requires:
```
trainer/    api/    dashboard/
```

Current structure:
```
training/   EDA/    docs/
```

- `training/` should be `trainer/`
- `api/` directory does not exist
- `dashboard/` directory does not exist
- `cv_baseline/` directory does not exist (required by architecture diagram)
- `ea/` directory does not exist (required by architecture diagram)

This naming mismatch will break CI/CD scripts, Docker build contexts, and any automation that references these paths.

---

### 3.9 NFR-MAINT-02 — No requirements.txt
**Project root**

No `requirements.txt` or `pyproject.toml` exists. Packages used across files include `torch`, `torchvision`, `pandas`, `PIL`, `kagglehub`, `sklearn`, `seaborn`, `cv2`, `numpy` — none are pinned. This violates NFR-MAINT-02 and makes the environment non-reproducible.

---

## 4. Bugs in Existing Code

### 4.1 Early Stopping Logic Broken — Counter Never Increments
**File:** `training/training.ipynb`, cell-17

The `counter` variable is declared and compared but never incremented when validation accuracy does not improve. The check `if counter >= patience` will never trigger:

```python
# What exists (BROKEN — counter stays 0 forever)
best_acc = 0
patience = 3
counter = 0

# ... inside loop ...
if counter >= patience:
    print("⛔ Early stopping")
    break
```

The increment logic and the best-model save logic (visible in cell outputs as "✅ Saved best model") is not visible in the cell source, suggesting it may have been accidentally deleted. The correct pattern:

```python
if acc > best_acc:
    best_acc = acc
    counter = 0
    torch.save(model.state_dict(), "outputs/model/best_model.pth")
    print("✅ Saved best model")
else:
    counter += 1
```

---

### 4.2 Hardcoded Absolute Path in EDA Notebook — Will Fail on Any Other Machine
**File:** `EDA/EDA.ipynb`, cell `e245ed10`

```python
BASE_DIR = Path(r"D:/AML_Project")
```

The project lives at `D:/University/y3t2/traffic_sign_ai`, not `D:/AML_Project`. This path is also machine-specific. Every `cv2.imread()` call downstream will return `None` and silently produce blank plots.

**Fix:** Use a path relative to the notebook's location:
```python
BASE_DIR = Path(__file__).parent.parent  # or use os.getcwd() logic
```
Or better, load images via the `image_path` column which already contains relative paths, and resolve from the repo root.

---

### 4.3 Learning Rate Inconsistency Between Config and Code
**File:** `training/training.ipynb` cell-15 vs `training/outputs/logs/config.json`

- Notebook optimizer uses `lr=3e-4` (0.0003)
- `config.json` records `"learning_rate": 0.001`

This means the experiment log does not accurately reflect how the model was trained. Any reproducibility attempt based on `config.json` will produce a different model.

**Fix:** Set `LR = 3e-4` as a constant at the top of the notebook and use it in both the optimizer and the config dict.

---

### 4.4 `organize_test()` Ignores Subdirectory Structure of Kaggle Test Set
**File:** `setup_dataset.py:106–128`

The Kaggle GTSRB test set ships with images in a flat directory. The code does `os.path.basename(row["Path"])` to get just the filename, but the `row["Path"]` in the Kaggle `Test.csv` contains a relative path like `Test/00000.png`. This works only if images are already extracted into a flat folder at `test_path`. If the Kaggle download preserves subdirectory structure, `os.path.join(test_path, img_name)` will produce a wrong path.

**Fix:** Use `row["Path"]` directly relative to the `data/` base:
```python
src = os.path.join(base_path, row["Path"])
```

---

### 4.5 `build_annotation()` O(n×m) Reverse Lookup
**File:** `setup_dataset.py:161–165`

For every image, the code iterates over all 43 `CLASS_NAMES` entries to reverse-map the class name back to an ID. With 51,839 images this is ~2.2M dictionary iterations. 

**Fix:** Pre-build the reverse map once:
```python
name_to_id = {clean_name(v): k for k, v in CLASS_NAMES.items()}
class_id = name_to_id.get(class_name)
```

---

## 5. Missing Implementations Summary

The following modules have zero implementation and must be built from scratch:

### 5.1 CV Baseline (`cv_baseline/`) — FR-CV-01 → FR-CV-09
Required components:
- Gaussian/Median filter preprocessing
- Harris Corner Detector with threshold tuning
- Gaussian/Laplacian pyramid (≥3 levels)
- SIFT feature extraction and matching with quantitative metric
- K-means or Watershed segmentation with IoU metric
- Naive Bayes classifier on SIFT feature vectors
- Full evaluation report (accuracy, per-class precision/recall/F1, IoU, matching accuracy)

### 5.2 GA Feature Selection Engine (`ea/`) — FR-EA-01 → FR-EA-16
Required components (all custom NumPy, no DEAP):
- Binary chromosome over MobileNetV2 penultimate-layer features
- Tournament Selection (k=3) and Roulette Wheel Selection
- Single-Point Crossover and Uniform Crossover
- Bit-Flip Mutation (rate 0.01/bit)
- Fitness Sharing for diversity
- 4 experiment configurations: {Tournament, Roulette} × {Single-Point, Uniform}
- JSON experiment logs uploaded to S3

### 5.3 Inference API (`api/`) — FR-API-01 → FR-API-07
Required components:
- FastAPI `POST /predict` with X-API-Key auth (HTTP 401 on failure)
- Sign category lookup table for all 43 GTSRB class IDs
- `GET /health` endpoint
- Safety disclaimer in every response
- In-memory preprocessing (no disk writes)
- `test_api.py` with ≥10 test images

### 5.4 Containerization + CI/CD (`.github/`, root) — FR-CI-01 → FR-CI-05
Required:
- `Dockerfile` (Python 3.10 slim)
- `docker-compose.yml` (port 8000, model path via env var)
- `.github/workflows/deploy.yml` (build → test → push ECR → deploy ECS)
- Image tagged as `<ecr-repo>:<git-sha>`

### 5.5 Cloud + Dashboard (`dashboard/`) — FR-CLD-01 → FR-CLD-07, FR-DASH-01 → FR-DASH-06
Required:
- S3 bucket with `models/`, `ea-experiments/`, `dataset/` prefixes
- IAM role with least-privilege S3 `GetObject` on `models/` only
- CloudWatch metrics + alarm (error rate > 1%)
- Billing alerts at $10 / $20
- Dashboard: GA convergence chart, feature reduction table, confusion matrix, 4-way comparison, live prediction tab

---

## 6. Prioritized Fix List

Fixes are ordered by dependency — earlier items unblock later ones.

| Priority | File | Issue | Impact |
|---|---|---|---|
| P0 | `setup_dataset.py:49` | Wrong Kaggle dataset ID | Wrong dataset downloaded |
| P0 | `training/training.ipynb` | Early stopping counter never increments | Training unreliable |
| P0 | `training/training.ipynb` | Model saved as wrong filename | Breaks API, S3, CI/CD |
| P0 | `EDA/EDA.ipynb` | Hardcoded `D:/AML_Project` path | All image loads fail |
| P1 | `setup_dataset.py` | Split ratio is ~64/11/25, not 70/15/15 | FR-D-02 violation |
| P1 | `training/training.ipynb` | All backbone unfrozen, not last 2 blocks | FR-AML-03 violation |
| P1 | `training/training.ipynb` | LR in config.json doesn't match code | Experiment non-reproducible |
| P1 | `training/training.ipynb` | Top-5 confused pairs not extracted | FR-AML-12 missing |
| P1 | Project root | No `requirements.txt` | NFR-MAINT-02 violation |
| P2 | Project root | Directory names don't match SRS | NFR-MAINT-01 violation |
| P2 | `setup_dataset.py:161` | O(n×m) reverse lookup in annotation builder | Performance |
| P3 | `training/training.ipynb` | EfficientNet-B0 not trained | FR-AML-02, FR-AML-13 |
| P3 | `setup_dataset.py` | No S3 upload step | FR-D-07 |
| P3 | `training/training.ipynb` | No augmentation-off baseline run | FR-D-08 |

---

## 7. What Is Working Correctly

| Item | Requirement | Result |
|---|---|---|
| WeightedRandomSampler | FR-D-05 | ✅ Correctly computes per-class weights |
| Augmentation pipeline | FR-D-06 | ✅ Flip, rotation ±15°, ColorJitter all present |
| ImageNet normalization | FR-D-03 | ✅ Mean [0.485, 0.456, 0.406] applied correctly |
| CrossEntropyLoss with class weights | FR-AML-04 | ✅ Per-class weights from bincount |
| StepLR scheduler | FR-AML-05 | ✅ step_size=3, gamma=0.3 |
| MobileNetV2 accuracy ≥ 90% | FR-AML-08 | ✅ 94.1% test accuracy |
| Top-5 accuracy ≥ 98% | FR-AML-11 | ✅ 99.6% |
| 43×43 confusion matrix | FR-AML-09 | ✅ Saved as cm.png |
| Full classification report | FR-AML-10 | ✅ Saved as report.json (all 43 classes) |
| Loss + accuracy curves | FR-AML-06 | ✅ Saved as curves.png |
| Reproducibility seed | Config | ✅ SEED=42 applied to random, numpy, torch |
| Class distribution EDA | FR-D-04 | ✅ Imbalance ratio, bar charts, heatmap all present |
