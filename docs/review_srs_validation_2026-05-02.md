
══════════════════════════════════════════════════════════════
  CODE REVIEW REPORT — SRS Requirements Validation
  Reviewed: Full codebase vs docs/srs.md          Date: 2026-05-02
  Domain: AI · API · INFRA · TEST · DS            Language: Python 3.10+
══════════════════════════════════════════════════════════════

## Summary

  Total issues found: 20
  🔴 Critical:  3    🟠 High: 7    🟡 Medium: 7    🟢 Low: 3    🔵 Info: 0

  Overall health: CRITICAL

  The API layer (FR-API), CI/CD pipeline (FR-CI), and MobileNetV2 training
  (FR-AML core) are solid and largely meet their requirements. However, three
  complete modules are either entirely absent or non-functional: (1) the entire
  EA/GA module has no Python source files in the repository — only compiled .pyc
  bytecache and pre-computed .npy data exist; (2) the dashboard module does not
  exist at all; and (3) two core CV requirements (Harris Corner Detector and SIFT
  matching) are empty stub files. These gaps represent roughly 40% of the graded
  SRS surface area and are blockers for every course except AML and Cloud.

────────────────────────────────────────────────────────────
## Requirements Coverage Matrix
────────────────────────────────────────────────────────────

| Module     | Requirements          | Status                         |
|------------|-----------------------|--------------------------------|
| FR-D       | FR-D-01 → FR-D-08     | Partial (FR-D-06, FR-D-08 gap) |
| FR-CV      | FR-CV-01 → FR-CV-09   | Partial (FR-CV-02, FR-CV-04 ❌)|
| FR-AML     | FR-AML-01 → FR-AML-12 | Partial (FR-AML-03, FR-AML-12) |
| FR-EA      | FR-EA-01 → FR-EA-16   | ❌ Source files missing         |
| FR-API     | FR-API-01 → FR-API-07 | ✅ All met                      |
| FR-CI      | FR-CI-01 → FR-CI-05   | ✅ All met                      |
| FR-CLD     | FR-CLD-01 → FR-CLD-07 | Infra-only (can't verify)      |
| FR-DASH    | FR-DASH-01 → FR-DASH-06| ❌ Module does not exist        |

────────────────────────────────────────────────────────────
## Issues — Ranked by Severity
────────────────────────────────────────────────────────────

### 🔴 CRITICAL — Issue #1: EA Module Source Files Completely Missing
**Location:** `ea/` directory — all FR-EA-* requirements
**Dimension:** AI-ML / Correctness

**Problem:**
The `ea/` directory contains zero Python source files. Only `__pycache__/` bytecode
(.pyc files compiled from Python 3.13) and pre-computed NumPy arrays remain. The
`__pycache__` names reveal modules that once existed:
`ga.py`, `experiment.py`, `fitness.py`, `population.py`, `diversity.py`,
`logger.py`, `extract_features.py`, `operators/selection.py`,
`operators/crossover.py`, `operators/mutation.py`, `operators/survivor.py`
— but all are deleted from the repository. FR-EA-01 through FR-EA-16 (the entire
EA course component) cannot be validated, reviewed, re-run, or graded. The data
artifacts (`X_train.npy`, `X_val.npy`, `feature_filter_mask.npy`) confirm the GA
was run at some point, but without source code, that work is irreproducible.

**Example of the problem:**
```
ea/
├── data/
│   ├── X_train.npy          ← feature vectors extracted (present)
│   ├── X_val.npy            ← (present)
│   ├── feature_filter_mask.npy  ← GA result chromosome (present)
│   ├── baseline_accuracy.txt    ← 0.9918 (present)
│   └── .gitkeep
├── operators/
│   └── __pycache__/         ← bytecode only, no .py
└── __pycache__/             ← bytecode only, no .py
```
**NO .py SOURCE FILES ANYWHERE IN ea/**

**Best Fix:**
Recover from git history (`git log --all --full-history -- "ea/*.py"`) or from
the original developer's machine. The .pyc files cannot be reverse-engineered to
clean source. Every FR-EA requirement must be satisfied by inspectable Python
source code, not bytecode.

```bash
# Attempt recovery:
git log --all --oneline -- "ea/*.py" "ea/operators/*.py"
git show <commit>:ea/ga.py > ea/ga.py
```

---

### 🔴 CRITICAL — Issue #2: Dashboard Module Entirely Absent
**Location:** Project root — all FR-DASH-* requirements
**Dimension:** AI-ML / Correctness

**Problem:**
No `dashboard/` directory exists. FR-DASH-01 through FR-DASH-06 are completely
unimplemented: no GA convergence chart, no feature reduction table, no confusion
matrix display, no model comparison table, no live prediction tab, and no S3 data
reader. The system architecture in the SRS explicitly defines a `dashboard/`
module as a first-class component hosted on AWS. The NFR-MAINT-01 directory
structure requires it. Without the dashboard, the examiner has no visual evidence
of EA convergence results, model comparison, or any cross-module integration.

**Best Fix:**
Implement a static HTML dashboard (acceptable per `cloud_proposal.md §10` which
allows "reduced to static HTML with embedded charts"). Minimum viable dashboard:
one HTML file + JS that reads the EA JSON logs from S3 and renders:
1. GA convergence line chart (Chart.js or Plotly CDN)
2. Feature reduction table
3. Confusion matrix heatmap (embedded PNG or Plotly)
4. Model comparison table (static or S3-sourced)
5. Live predict form calling `/predict`

---

### 🔴 CRITICAL — Issue #3: Harris Corner Detector and SIFT Matching Are Empty Stubs
**Location:** `cv/harris.py` (1 blank line), `cv/matching.py` (1 blank line)
**Dimension:** AI-ML / Correctness

**Problem:**
Two files required by core CV course requirements are completely empty:
- `cv/harris.py` — FR-CV-02 requires Harris Corner Detector implementation,
  keypoint visualization, and threshold tuning analysis.
- `cv/matching.py` — FR-CV-04 requires SIFT feature matching across image pairs,
  match visualizations, and a quantitative matching accuracy metric
  (correct-match ratio or RANSAC inlier ratio).

These are not partially implemented — they have literally no code. Since
`run_pipeline.py` never imports either file, they are also untested. The CV
course evaluation report (FR-CV-07) is also incomplete because it cannot include
SIFT matching accuracy or Harris threshold analysis without these modules.

**Example of the problem:**
```python
# cv/harris.py — actual file content:
(empty — 1 blank line)

# cv/matching.py — actual file content:
(empty — 1 blank line)
```

**Best Fix:**
```python
# cv/harris.py — minimal compliant implementation
import cv2
import numpy as np

def detect_harris(img, block_size=2, ksize=3, k=0.04, threshold=0.01):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    response = cv2.cornerHarris(gray, block_size, ksize, k)
    response = cv2.dilate(response, None)
    corners = np.zeros_like(gray)
    corners[response > threshold * response.max()] = 255
    return corners, response

# cv/matching.py — minimal compliant implementation
import cv2

def match_sift(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    ratio = len(good) / len(matches) if matches else 0.0
    vis = cv2.drawMatchesKnn(img1, kp1, img2, kp2,
                             [[m] for m in good], None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return good, ratio, vis
```

---

### 🟠 HIGH — Issue #4: CV Pipeline Evaluates on Training Data (Data Leakage)
**Location:** `run_pipeline.py:16–52`
**Dimension:** AI-ML / Correctness

**Problem:**
The CV baseline pipeline loads images from `data/*/*.png` (the raw Kaggle training
folder), trains the Naive Bayes classifier on those images, and then immediately
calls `clf.predict(X)` on the same features it just trained on. The reported
accuracy is training-set accuracy, not test-set accuracy. FR-CV-08 requires the
baseline to be the lower-bound comparison in the three-way model comparison table,
but an inflated training accuracy is meaningless for that purpose. The correct
evaluation set is `dataset/test/`.

**Example of the problem:**
```python
# run_pipeline.py — trains and evaluates on the SAME data
clf = NB()
clf.train(X, labels)        # trained on data/*/*.png
pred = clf.predict(X)       # evaluated on SAME data/* — WRONG
acc, report = evaluate(labels, pred)
```

**Best Fix:**
```python
# Split into train/test sets using dataset/train and dataset/test
from pathlib import Path

def load_split(split_dir):
    images, labels = [], []
    for path in Path(split_dir).glob("*/*.png"):
        label = path.parent.name
        img = cv2.imread(str(path))
        images.append(preprocess(img))
        labels.append(label)
    return images, labels

train_images, train_labels = load_split("dataset/train")
test_images, test_labels   = load_split("dataset/test")

train_descs = [extract_sift(img)[1] for img in train_images]
test_descs  = [extract_sift(img)[1] for img in test_images]

bovw = BoVW(k=20)
bovw.fit(train_descs)                       # fit on train only

X_train = [bovw.transform(d) for d in train_descs]
X_test  = [bovw.transform(d) for d in test_descs]

clf = NB()
clf.train(X_train, train_labels)
pred = clf.predict(X_test)                  # evaluate on test
acc, report = evaluate(test_labels, pred)
```

---

### 🟠 HIGH — Issue #5: requirements.txt Missing PyTorch and TorchVision
**Location:** `requirements.txt`
**Dimension:** Reliability / Maintainability

**Problem:**
`requirements.txt` lists only: numpy, fastapi, uvicorn, python-multipart,
python-dotenv, pillow, boto3, requests. It does not include `torch` or
`torchvision` — the two most critical dependencies for the entire API service.
A developer following NFR-MAINT-02 (pin all dependencies in requirements.txt)
and running `pip install -r requirements.txt` to set up locally will get an
ImportError on first run. The comment claims the file is "only used locally",
making this gap even worse: it's the exact use case that is broken.

**Best Fix:**
```
# requirements.txt — add the CPU torch build for local dev
--index-url https://download.pytorch.org/whl/cpu
torch==2.3.0+cpu
torchvision==0.18.0+cpu
numpy<2
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
python-dotenv==1.0.1
pillow==10.3.0
boto3==1.34.0
requests==2.31.0
```

---

### 🟠 HIGH — Issue #6: FR-AML-03 Violated — Full Backbone Unfrozen, Not Last 2 Blocks
**Location:** `training/training.ipynb` cell id="cell-13" and cell id="cell-17"
**Dimension:** AI-ML / Correctness

**Problem:**
FR-AML-03 requires: "freeze the backbone for the first 5 epochs, then unfreeze
the last 2 blocks for fine-tuning." The training notebook freezes all of
`model.features` initially (correct) but at epoch 5 unfreezes ALL of
`model.features` instead of only the last 2 InvertedResidual blocks. This
violates the letter of the SRS requirement and may cause the examiner to flag it.
Unfreezing all 19 InvertedResidual blocks simultaneously also risks catastrophic
forgetting of lower-level features.

**Example of the problem:**
```python
# cell-13: freezes all features — correct
for param in model.features.parameters():
    param.requires_grad = False

# cell-17, epoch==4: unfreezes ALL features — violates FR-AML-03
if epoch == 4:
    for param in model.features.parameters():  # ALL blocks, not last 2
        param.requires_grad = True
```

**Best Fix:**
```python
# Unfreeze only model.features[-2:] (the last 2 InvertedResidual blocks)
if epoch == 4:
    for param in model.features[-2:].parameters():
        param.requires_grad = True
    optimizer.add_param_group({
        "params": model.features[-2:].parameters(),
        "lr": 3e-5
    })
    print("🔥 Last 2 backbone blocks unfrozen (FR-AML-03)")
```

---

### 🟠 HIGH — Issue #7: FR-AML-12 Missing — Top-5 Most Confused Sign Pairs Not Computed
**Location:** `training/training.ipynb` — no such cell exists
**Dimension:** AI-ML / Completeness

**Problem:**
FR-AML-12 requires: "identify and report the top-5 most confused sign pairs from
the confusion matrix." The training notebook generates the 43×43 confusion matrix
and saves it as a heatmap PNG, but never extracts the off-diagonal pairs with the
highest counts. This is a required deliverable in the AML acceptance criteria and
a specific output that examiners expect to see in the training report.

**Best Fix:**
Add a cell after the confusion matrix generation:
```python
import numpy as np

# Zero the diagonal (true positives are not confused pairs)
cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)

# Get top-5 off-diagonal entries
flat_idx = np.argsort(cm_no_diag.ravel())[-5:][::-1]
print("Top-5 most confused sign pairs:")
for idx in flat_idx:
    true_cls = class_labels[idx // 43]
    pred_cls = class_labels[idx % 43]
    count = cm_no_diag[idx // 43, idx % 43]
    print(f"  True: {true_cls} → Predicted: {pred_cls} ({count} errors)")
```

---

### 🟠 HIGH — Issue #8: FR-D-08 Missing — No Augmentation Comparison Run
**Location:** `training/training.ipynb` — not present
**Dimension:** AI-ML / Completeness

**Problem:**
FR-D-08 requires "two comparable model runs — one with the augmentation pipeline
active and one without — and report the accuracy delta." The notebook only
contains one training run (with augmentation). There is no no-augmentation
baseline run, no accuracy delta calculation, and no comparison table. This is
explicitly required by the SRS acceptance criteria table in §9.1.

**Best Fix:**
Add a second training cell that uses `val_transform` for both train and val
(effectively removing augmentation), trains for the same epochs, evaluates on
the test set, and prints:
```python
print(f"Augmented test acc:    {acc_augmented:.4f}")
print(f"No-augmentation acc:   {acc_no_aug:.4f}")
print(f"Augmentation delta:    {acc_augmented - acc_no_aug:+.4f}")
```

---

### 🟠 HIGH — Issue #9: Model Filename Mismatch Between SRS and Code
**Location:** `training/training.ipynb:cell-27`, `docker-compose.yml:18`, `api/app.py:41`
**Dimension:** Reliability / Maintainability

**Problem:**
FR-AML-07 specifies the exported model file must be `traffic_sign_model_mobilenetv2.pth`.
The training notebook saves two files: `mobilenetv2_full.pth` (full checkpoint) and
`mobilenetv2_inference.pth` (inference state dict). Neither matches the SRS-required
name. The docker-compose.yml default path is `/app/models/traffic_sign_model_mobilenetv2.pth`
but the actual file is `mobilenetv2_inference.pth`. This inconsistency means the S3
bucket structure in §5.2, docker-compose default, and the actual file on disk are
all misaligned.

**Best Fix:**
Rename the saved file in the notebook to match the SRS:
```python
torch.save(
    model.state_dict(),
    "outputs/model/traffic_sign_model_mobilenetv2.pth"   # FR-AML-07
)
```
Then update `api/app.py` default MODEL_PATH to match:
```python
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[1]
            / "training" / "outputs" / "model"
            / "traffic_sign_model_mobilenetv2.pth"),
    )
)
```

---

### 🟠 HIGH — Issue #10: Timing-Unsafe API Key Comparison
**Location:** `api/app.py:174`
**Dimension:** Security

**Problem:**
The API key comparison `if x_api_key != API_KEY` uses Python's built-in string
equality, which may short-circuit on the first differing character. This enables
a timing side-channel attack: an attacker sending many requests can measure
response latency to incrementally guess the correct API key one character at a
time. For a production endpoint (NFR-SEC-01), the key comparison must be
constant-time regardless of where the mismatch occurs.

**Example of the problem:**
```python
def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if x_api_key != API_KEY:   # ← NOT constant-time
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

**Best Fix:**
```python
import hmac

def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    provided = (x_api_key or "").encode()
    expected = API_KEY.encode()
    if not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

**References:** OWASP A07 — Timing oracle on secret comparison.

---

### 🟡 MEDIUM — Issue #11: FR-D-06 Missing RandomHorizontalFlip Augmentation
**Location:** `training/training.ipynb:cell-3`
**Dimension:** AI-ML / Correctness

**Problem:**
FR-D-06 requires three augmentations: brightness/contrast jitter ✅, random
rotation ±15° ✅, and random horizontal flip ❌. The training transform uses
`ColorJitter(0.2, 0.2)` and `RandomRotation(15)` but lacks `RandomHorizontalFlip`.
This is a minor gap but directly violates a stated SRS requirement. Note: for
asymmetric signs (e.g., turn-right vs turn-left), horizontal flip can create
incorrect labels — this is a known trade-off worth documenting if left out.

**Best Fix:**
```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),           # FR-D-06
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

### 🟡 MEDIUM — Issue #12: FR-CV-07 Incomplete — No IoU or SIFT Matching Accuracy
**Location:** `cv/evaluation.py`, `run_pipeline.py`
**Dimension:** AI-ML / Completeness

**Problem:**
FR-CV-07 requires the CV baseline evaluation to report: overall accuracy ✅,
per-class precision/recall/F1 ✅, IoU for the segmentation step ❌, and descriptor
matching accuracy for SIFT ❌. The `cv/evaluation.py` module only wraps
`sklearn.metrics.accuracy_score` and `classification_report`. Neither IoU for the
K-means segmentation output nor a SIFT matching accuracy metric is computed or
reported anywhere in the pipeline.

**Best Fix:**
Add IoU calculation in `cv/evaluation.py`:
```python
import numpy as np

def compute_iou(seg_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(seg_mask, gt_mask).sum()
    union = np.logical_or(seg_mask, gt_mask).sum()
    return float(intersection / union) if union > 0 else 0.0
```
And integrate SIFT matching accuracy from `cv/matching.py` (once implemented)
into the evaluation report.

---

### 🟡 MEDIUM — Issue #13: FR-CV-09 Incomplete — No Scale-Space Visualization
**Location:** `cv/pyramid.py`, `run_pipeline.py`
**Dimension:** AI-ML / Completeness

**Problem:**
FR-CV-09 requires: "implement a Gaussian or Laplacian image pyramid at ≥3 levels,
demonstrate scale-space analysis, and produce a scale-space visualization." The
`cv/pyramid.py` module does build 3 Gaussian levels using `cv2.pyrDown` ✅, but:
(1) no visualization is produced or saved, (2) the pyramid function is never called
from `run_pipeline.py`, and (3) there is no Laplacian pyramid (only Gaussian).
The SRS acceptance criteria specifically require a visualization to be "included in
the CV evaluation report."

**Best Fix:**
Add visualization to `cv/pyramid.py` and call it from the pipeline:
```python
import cv2
import matplotlib.pyplot as plt

def pyramid(img, levels=3):
    result = [img]
    current = img
    for _ in range(levels):
        current = cv2.pyrDown(current)
        result.append(current)
    return result

def visualize_pyramid(levels, save_path="outputs/pyramid_visualization.png"):
    fig, axes = plt.subplots(1, len(levels), figsize=(15, 5))
    for i, (ax, lvl) in enumerate(zip(axes, levels)):
        ax.imshow(cv2.cvtColor(lvl, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Level {i} ({lvl.shape[1]}×{lvl.shape[0]})")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Pyramid visualization saved to {save_path}")
```

---

### 🟡 MEDIUM — Issue #14: FR-EA-13/FR-EA-14 — No JSON Experiment Logs in Repository
**Location:** `ea/data/` directory
**Dimension:** AI-ML / Completeness

**Problem:**
FR-EA-13 requires each experiment run to export a JSON log (run_id, operator
config, fitness per generation, best chromosome, selected indices, reduction
ratio). FR-EA-14 requires all 4 logs to be uploaded to `s3://<bucket>/ea-experiments/`.
The SRS §5.2 defines exact filenames: `run_tournament_singlepoint.json`,
`run_tournament_uniform.json`, `run_roulette_singlepoint.json`,
`run_roulette_uniform.json`. None of these exist in the repository. Even if the
EA code is recovered (Issue #1), without these log files the dashboard (Issue #2)
has no data to display and S3 requirements cannot be satisfied.

**Best Fix:**
Once EA source is recovered, each experiment runner must produce and save JSON:
```python
log = {
    "run_id": "tournament_singlepoint_001",
    "selection": "tournament",
    "crossover": "single_point",
    "population_size": 50,
    "generations_run": generations_run,
    "fitness_per_generation": fitness_history,
    "best_fitness": float(best_fitness),
    "best_chromosome": best_chromosome.tolist(),
    "selected_feature_indices": selected_indices.tolist(),
    "total_features": total_features,
    "selected_features": len(selected_indices),
    "reduction_ratio": 1 - len(selected_indices) / total_features
}
with open(f"ea/data/run_{selection}_{crossover}.json", "w") as f:
    json.dump(log, f, indent=2)
```

---

### 🟡 MEDIUM — Issue #15: Category Names Don't Match SRS Specification
**Location:** `api/app.py:91–96`
**Dimension:** API / Correctness

**Problem:**
FR-API-01 and the SRS §5.1 example show category values of: "prohibition",
"warning", "mandatory", "informational". The API returns: "prohibition" ✅,
"danger" ❌ (should be "warning"), "mandatory" ✅, "other" ❌ (should be
"informational"). Test `test_api.py` only asserts `category` is a "non-empty
string" — it never validates the actual category values — so this mismatch passes
CI but fails the SRS specification.

**Best Fix:**
```python
CATEGORY_BY_ID = {
    **{i: "prohibition"   for i in [0,1,2,3,4,5,7,8,9,10,15,16,17]},
    **{i: "warning"       for i in [18,19,20,21,22,23,24,25,26,27,28,29,30,31]},
    **{i: "mandatory"     for i in [33,34,35,36,37,38,39,40]},
    **{i: "informational" for i in [6,11,12,13,14,32,41,42]},
}
```
Also update `test_api.py` to assert `category in {"prohibition","warning","mandatory","informational"}`.

---

### 🟡 MEDIUM — Issue #16: No S3 Upload Scripts for Dataset or Model Artifacts
**Location:** Project root — no upload scripts exist
**Dimension:** INFRA / Completeness

**Problem:**
FR-D-07 requires the processed dataset to be stored in S3 so it is "downloaded
once and never re-fetched." FR-AML-07 requires the model .pth file to be uploaded
to S3. FR-CLD-02 requires an S3 bucket with `models/`, `ea-experiments/`, and
`dataset/` prefixes. No upload script, Makefile target, or documented command
exists for any of these. The API has an S3 download path (`_maybe_download_from_s3`)
but nothing to put the artifacts there in the first place.

**Best Fix:**
Add a `scripts/upload_artifacts.py` (or document the CLI commands) at minimum:
```bash
# Upload model
aws s3 cp training/outputs/model/traffic_sign_model_mobilenetv2.pth \
    s3://<bucket>/models/traffic_sign_model_mobilenetv2.pth

# Upload EA logs
aws s3 sync ea/data/ s3://<bucket>/ea-experiments/ \
    --exclude "*.npy" --exclude ".gitkeep"
```

---

### 🟡 MEDIUM — Issue #17: NFR-MAINT-01 Directory Structure Non-Compliant
**Location:** Project root directory layout
**Dimension:** Maintainability

**Problem:**
NFR-MAINT-01 specifies: "three top-level directories: `trainer/`, `api/`, `dashboard/`".
The actual structure uses `training/` (not `trainer/`), has no `dashboard/`
directory, and adds `cv/`, `ea/`, `EDA/`, `testing/` directories that are not in
the SRS. While this is a pragmatic choice, course evaluators may specifically check
the directory layout against the SRS.

---

### 🟢 LOW — Issue #18: docker-compose Default MODEL_PATH Points to Non-Existent Container Path
**Location:** `docker-compose.yml:18`
**Dimension:** Reliability

**Problem:**
The default MODEL_PATH in docker-compose is `/app/models/traffic_sign_model_mobilenetv2.pth`.
The Dockerfile only copies `COPY api/ ./api/` — no model files are bundled. So
running `docker-compose up` without setting MODEL_PATH will start a container that
immediately crashes with FileNotFoundError. This will confuse any developer who
tries the "quick start" without reading the documentation.

**Best Fix:**
Update the default to use the S3 path placeholder, making the intent explicit:
```yaml
- MODEL_PATH=${MODEL_PATH:-s3://your-bucket/models/traffic_sign_model_mobilenetv2.pth}
```
Or add a volume mount in the override file and document it clearly in README.

---

### 🟢 LOW — Issue #19: Kaggle Credentials Not Documented for setup_dataset.py
**Location:** `setup_dataset.py:49`, `.env.example`
**Dimension:** Reliability / Documentation

**Problem:**
`setup_dataset.py` calls `kagglehub.dataset_download()` which silently requires
`KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables (or `~/.kaggle/kaggle.json`).
These are not mentioned in `.env.example`, not documented anywhere in the project,
and the script will fail with an authentication error on first run. A developer
with no prior Kaggle experience will have no idea why it fails.

**Best Fix:**
Add to `.env.example`:
```bash
# Required for setup_dataset.py — get from https://www.kaggle.com/settings → API
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

---

### 🟢 LOW — Issue #20: CloudWatch ErrorCount Counts Only 5xx, Not 4xx
**Location:** `api/app.py:200`
**Dimension:** Reliability / Monitoring

**Problem:**
`_emit` counts an error only when `status_code >= 500`. The CloudWatch alarm
(FR-CLD-06) triggers when "HTTP error rate exceeds 1% over any 5-minute window."
If the alarm is configured against the `ErrorCount` metric, it would miss a flood
of 401 (auth failures) or 422 (bad images) events. Under an attack scenario, 100%
of requests returning 401 would appear as 0% error rate in CloudWatch.

**Best Fix:**
```python
_emit(
    latency_ms=(time.monotonic() - start) * 1000,
    status_code=response.status_code,
    is_error=response.status_code >= 400   # count 4xx and 5xx
)
```
And update `_emit` signature accordingly.

────────────────────────────────────────────────────────────
## What's Done Well
────────────────────────────────────────────────────────────

- **API layer is production-quality** — `api/app.py` correctly handles auth, in-memory
  preprocessing (FR-API-04), proper 401/422 error codes, CloudWatch emission,
  S3 model download, warmup inference, and non-root Docker user. All 7 FR-API
  requirements are met.

- **test_api.py is thorough** — 15 tests covering health check, 12 color-variant
  predict calls, invalid image (422 path), and wrong API key (401 path). Correctly
  differentiates dev mode (no key) vs prod mode. FR-API-06 satisfied with margin.

- **CI/CD pipeline is well-structured** — deploy.yml correctly chains all 4 stages
  with `needs:`, uses MODEL_TEST_MODE=1 so CI doesn't need a .pth file, applies
  ECR lifecycle cleanup, implements deployment rollback on health check failure,
  and avoids AWS credentials leaking over SSH by relying on EC2 IAM instance profile
  (FR-CI-04 and NFR-SEC-04 both satisfied).

- **MobileNetV2 training achieves strong results** — Test accuracy 95% (FR-AML-08 ✅),
  top-5 accuracy 99.6% (FR-AML-11 ✅). SEED=42 set across random/numpy/torch for
  reproducibility. CosineAnnealingLR and CrossEntropyLoss with class weights
  correctly applied.

- **Dockerfile is well-layered** — PyTorch (190MB) installed in a dedicated layer so
  it survives code changes. Non-root `appuser`, HEALTHCHECK with curl, and
  multi-stage retry flags (`--retries 5 --timeout 120`) for flaky network.

────────────────────────────────────────────────────────────
## Refactor Roadmap (Priority Order)
────────────────────────────────────────────────────────────

  P0 — Fix before any demo or submission:
    [ ] Recover EA Python source files from git history or developer machine (Issue #1)
    [ ] Implement dashboard module — minimum static HTML (Issue #2)
    [ ] Implement cv/harris.py and cv/matching.py (Issue #3)
    [ ] Fix run_pipeline.py to evaluate on dataset/test/, not data/ (Issue #4)
    [ ] Generate and commit 4 EA JSON experiment logs (Issue #14)

  P1 — Fix this sprint:
    [ ] Add torch/torchvision to requirements.txt (Issue #5)
    [ ] Fix FR-AML-03: unfreeze only last 2 backbone blocks (Issue #6)
    [ ] Add top-5 confused pairs cell to training.ipynb (Issue #7)
    [ ] Add no-augmentation comparison run to training.ipynb (Issue #8)
    [ ] Rename model file to traffic_sign_model_mobilenetv2.pth (Issue #9)
    [ ] Fix API key comparison to hmac.compare_digest (Issue #10)
    [ ] Fix category names to "warning"/"informational" (Issue #15)
    [ ] Add S3 upload scripts or document CLI commands (Issue #16)

  P2 — Fix next time you touch this code:
    [ ] Add RandomHorizontalFlip to train_transform (Issue #11)
    [ ] Add IoU and SIFT matching accuracy to cv/evaluation.py (Issue #12)
    [ ] Add pyramid visualization and integrate into pipeline (Issue #13)
    [ ] Fix docker-compose default MODEL_PATH (Issue #18)
    [ ] Document Kaggle credentials in .env.example (Issue #19)
    [ ] Expand CloudWatch ErrorCount to include 4xx (Issue #20)

────────────────────────────────────────────────────────────
## Domain-Specific Checklist
────────────────────────────────────────────────────────────

### Security Hardening
  [x] All user inputs validated at system boundaries (image format checked)
  [x] No AWS credentials in source files or Dockerfiles
  [x] Auth check on /predict endpoint
  [ ] API key comparison is constant-time (Issue #10)
  [x] No image data persisted (FR-API-04 / NFR-SEC-03)

### AI / ML
  [x] Train/val/test split done before preprocessing (setup_dataset.py)
  [x] Random seeds set for reproducibility (SEED=42 across all libs)
  [x] Evaluation on held-out test set (12,630 images) — for AML module
  [ ] CV baseline evaluation on held-out test set (Issue #4 — currently train set)
  [x] Model artifacts saved (mobilenetv2_inference.pth)
  [ ] Model filename matches SRS spec (Issue #9)
  [ ] EA experiment logs exported as JSON (Issue #14)
  [ ] 4 complete GA experiment runs with operator coverage (Issue #1 — source missing)
  [ ] Augmentation comparison run (Issue #8)

### Infrastructure / DevOps
  [x] Secrets managed via GitHub Actions secrets and env vars
  [x] EC2 deployment uses IAM instance profile (no credential leakage)
  [x] Container health check defined in Dockerfile
  [x] Restart policy configured (restart: always)
  [x] ECR lifecycle policy keeps only last 10 images
  [ ] Dashboard deployed alongside API (Issue #2)
  [ ] S3 artifacts populated (Issues #14, #16)

══════════════════════════════════════════════════════════════
```
