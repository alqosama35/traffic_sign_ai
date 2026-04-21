# Project Requirements Validation
## Traffic Sign Intelligence Platform — 4-Course Coverage Check

> **Last updated:** 2026-04-21
> **Based on:** `srs.md` v1.1 (supersedes `cloud_proposal.md`), cross-checked against `projects.md` (CV + EA course sheets), `aml.md` (AML project guidelines), and `cloud_project_registration.md` (Cloud registration SRS).
> **All gaps resolved** — CI/CD (FR-CI-03), dual architecture (FR-AML-02), Gaussian/Laplacian Pyramid (FR-CV-09), IoU + matching accuracy (FR-CV-07), sign category in API (FR-API-01).

---

## Course 1: Cloud Computing

| # | Requirement | Source | Status | Evidence in SRS |
|---|---|---|---|---|
| 1.1 | Application deployed on a cloud provider (AWS/GCP/Azure) | Cloud reg. FR-03, SRS FR-CLD-01 | ✅ Satisfied | AWS EC2 (t2.medium) or ECS Fargate hosting the FastAPI inference container |
| 1.2 | Use of cloud object/blob storage | Cloud reg. FR-04, SRS FR-CLD-02 | ✅ Satisfied | AWS S3 — stores model `.pth` files, EA experiment JSON logs, versioned dataset |
| 1.3 | Containerization with Docker | Cloud reg. FR-08, SRS FR-CI-01 | ✅ Satisfied | `Dockerfile` using Python 3.10 slim base image |
| 1.4 | Container registry usage | SRS FR-CI-05 | ✅ Satisfied | Docker images tagged `<ecr-repo>:<git-sha>` and pushed to AWS ECR |
| 1.5 | Monitoring and logging | Cloud reg. FR-09, SRS FR-CLD-05 | ✅ Satisfied | AWS CloudWatch — request count, p50/p95 latency, HTTP error rate, memory usage |
| 1.6 | Access control / identity management | Cloud reg. NFR-07, SRS FR-CLD-04, NFR-SEC-02 | ✅ Satisfied | AWS IAM role with least-privilege S3 `GetObject` on `models/` prefix only; no hardcoded credentials |
| 1.7 | Publicly accessible API / endpoint | Cloud reg. FR-01, SRS FR-CLD-01 | ✅ Satisfied | `/predict` publicly accessible on AWS during demo; `GET /health` also exposed |
| 1.8 | API security | Cloud reg. NFR-07, SRS NFR-SEC-01, FR-API-06 | ✅ Satisfied | `X-API-Key` header required on `/predict`; HTTP 401 on missing/invalid key |
| 1.9 | Cloud contribution is meaningful, not decorative | SRS §2.2 | ✅ Satisfied | Three substantive purposes: reproducible experiment storage (S3), scalable inference (ECS/EC2), observability (CloudWatch) |
| 1.10 | Reproducible, auditable results via cloud storage | Cloud reg. NFR-08, SRS FR-EA-14 | ✅ Satisfied | All 4 GA run JSON logs uploaded to `s3://<bucket>/ea-experiments/` with full parameters |
| 1.11 | Cost awareness / resource management | SRS FR-CLD-07 | ✅ Satisfied | Billing alerts at $10/$20; EC2 terminated after demo; AWS budget cap $20 |
| 1.12 | CI/CD or deployment pipeline *(was Gap 1)* | Cloud reg. FR-08, SRS FR-CI-03 | ✅ Satisfied | GitHub Actions workflow (`deploy.yml`): build → test → push ECR → deploy; blocks merge on failure |
| 1.13 | Docker image build time | Cloud reg. NFR-05 | ✅ Satisfied | Target <5 minutes (Python 3.10 slim base, pinned `requirements.txt`) |
| 1.14 | `docker-compose.yml` for local dev | SRS FR-CI-02, NFR-PORT-02 | ✅ Satisfied | `docker-compose.yml` exposes port 8000, model path via env var, runs API locally without AWS |

**Cloud Score: 14/14 fully satisfied**

---

## Course 2: Advanced Machine Learning (AML)

| # | Requirement | Source | Status | Evidence in SRS |
|---|---|---|---|---|
| 2.1 | Transfer learning from a pre-trained model | aml.md (all projects), SRS FR-AML-01 | ✅ Satisfied | MobileNetV2 pre-trained on ImageNet, fine-tuned on GTSRB |
| 2.2 | Fine-tuning strategy described | SRS FR-AML-03 | ✅ Satisfied | Freeze backbone for 5 epochs, then unfreeze last 2 blocks |
| 2.3 | Handling class imbalance | SRS FR-D-05, FR-AML-04 | ✅ Satisfied | `WeightedRandomSampler` + CrossEntropyLoss with per-class weights |
| 2.4 | Data augmentation pipeline | SRS FR-D-06, FR-AML-01 | ✅ Satisfied | Brightness/contrast jitter, rotation ±15°, horizontal flip |
| 2.5 | Augmentation comparison (with vs without) | aml.md Augmentation Designer role | ✅ Satisfied | Augmentation Designer role explicitly compares augmented vs non-augmented runs (FR-D-06) |
| 2.6 | Training loop with loss and accuracy curves | aml.md deliverables, SRS FR-AML-06 | ✅ Satisfied | Loss/accuracy curves (train + val) exported per epoch |
| 2.7 | Learning rate scheduling | SRS FR-AML-05 | ✅ Satisfied | StepLR or CosineAnnealingLR |
| 2.8 | Model evaluation — overall accuracy | SRS FR-AML-08 | ✅ Satisfied | MobileNetV2 target ≥90% on GTSRB test set |
| 2.9 | Model evaluation — per-class metrics | SRS FR-AML-10 | ✅ Satisfied | Full classification report: per-class precision, recall, F1 for all 43 classes |
| 2.10 | Confusion matrix | SRS FR-AML-09 | ✅ Satisfied | 43×43 heatmap; top-5 most confused sign pairs identified (FR-AML-12) |
| 2.11 | Top-k accuracy metric | SRS FR-AML-11 | ✅ Satisfied | Top-5 accuracy target ≥98% |
| 2.12 | Baseline comparison | SRS FR-CV-08, FR-DASH-04 | ✅ Satisfied | Three-way comparison: CV baseline vs MobileNetV2 vs EfficientNet-B0 vs GA-optimized |
| 2.13 | Model comparison (at least 2 architectures) *(was Gap 2)* | aml.md (Project 3 trainer role), SRS FR-AML-02, FR-AML-13 | ✅ Satisfied | MobileNetV2 **and** EfficientNet-B0 both required to produce `.pth` checkpoints; architecture comparison table (accuracy + latency) is now a hard requirement |
| 2.14 | REST API serving the model | aml.md deliverables, SRS FR-API-01 | ✅ Satisfied | FastAPI `POST /predict` — class, confidence, top-3 alternatives |
| 2.15 | Dataset preprocessing (resize, normalize) | SRS FR-D-02, FR-D-03 | ✅ Satisfied | Resize to 224×224, ImageNet normalization, 70/15/15 train/val/test split |
| 2.16 | EDA notebook with 8+ visualizations | aml.md (required deliverable) | ✅ Satisfied | 8+ visualizations: sample grids, class distribution bar chart, image size histogram, similar sign pairs (FR-D-04) |
| 2.17 | Test script (`test_api.py`) | aml.md (required deliverable), SRS FR-API-06 | ✅ Satisfied | `test_api.py` sends 10+ representative images; asserts HTTP 200, valid class, confidence ∈ [0,1], top_3 length = 3 |
| 2.18 | Sign category in API response | aml.md Project 3 API spec, SRS FR-API-01 | ✅ Satisfied | `FR-API-01` now requires `"category"` field (e.g., "prohibition", "warning") derived from a static 43-class lookup table; `test_api.py` asserts `category` is a non-empty string |
| 2.19 | Presentation / live demo | aml.md (required deliverable) | ✅ Satisfied | 10-minute demo with live API test — covered in SRS §9 acceptance criteria and dashboard Live Prediction tab |

**AML Score: 18/19 fully satisfied, 1 partial**

---

## Course 3: Evolutionary Algorithms (EA)

| # | Requirement | Source | Status | Evidence in SRS |
|---|---|---|---|---|
| 3.1 | Custom EA implementation — no black-box library | projects.md EA req, SRS FR-EA-01 | ✅ Satisfied | Explicitly stated: custom Python GA + NumPy; no DEAP or scipy.optimize |
| 3.2 | Clear chromosome / solution representation | projects.md §d, SRS FR-EA-02 | ✅ Satisfied | Binary chromosome of length *n*; each bit = include/exclude one feature dimension |
| 3.3 | Fitness function defined and justified | projects.md §d, SRS FR-EA-03 | ✅ Satisfied | Validation set classification accuracy of k-NN or linear SVM on GA-selected subset |
| 3.4 | Population initialization | projects.md §d, SRS FR-EA-04 | ✅ Satisfied | 50 individuals, uniformly initialized at random |
| 3.5 | Selection operator implemented | projects.md §d, SRS FR-EA-05 | ✅ Satisfied | Tournament selection (k=3) |
| 3.6 | At least 2 selection strategies compared | projects.md §e, SRS FR-EA-06, FR-EA-12 | ✅ Satisfied | Tournament vs Roulette Wheel selection — explicitly required; results table across 4 configs |
| 3.7 | Crossover operator implemented | projects.md §d, SRS FR-EA-07 | ✅ Satisfied | Single-point crossover |
| 3.8 | At least 2 crossover operators compared | projects.md §e, SRS FR-EA-08, FR-EA-12 | ✅ Satisfied | Single-point vs Uniform crossover — explicitly required; 4 experiment configurations |
| 3.9 | Mutation operator implemented | projects.md §d, SRS FR-EA-09 | ✅ Satisfied | Bit-flip mutation at rate 0.01 per bit |
| 3.10 | Termination condition defined | projects.md §d, SRS FR-EA-10 | ✅ Satisfied | 100 generations OR no improvement for 20 consecutive generations |
| 3.11 | Diversity preservation mechanism | projects.md §h, SRS FR-EA-11 | ✅ Satisfied | Fitness sharing within niches — explicitly required |
| 3.12 | At least 4 experimental configurations | projects.md deliverables, SRS FR-EA-12 | ✅ Satisfied | 2 selection strategies × 2 crossover operators = 4 complete runs |
| 3.13 | Convergence curve (fitness vs generation) | projects.md deliverables, SRS FR-DASH-01 | ✅ Satisfied | GA convergence curves displayed on dashboard for all 4 configs |
| 3.14 | Feature reduction quantified | projects.md deliverables, SRS FR-EA-15 | ✅ Satisfied | Target ≥30% feature reduction; feature reduction ratio in JSON log (FR-EA-13) |
| 3.15 | Accuracy vs reduction trade-off analysis | SRS FR-EA-16, FR-DASH-02 | ✅ Satisfied | Accuracy drop <3 pp vs full-feature model; feature reduction table on dashboard |
| 3.16 | Operator comparison results table | projects.md deliverables, SRS FR-DASH-02 | ✅ Satisfied | Table comparing all 4 experiment configurations on dashboard |
| 3.17 | EA applied to a real, meaningful problem | projects.md context, SRS §2.1 | ✅ Satisfied | Feature selection on 1,280-dimension deep learning feature vectors for traffic sign classification |
| 3.18 | Problem formally defined as optimization type | projects.md §a | ✅ Satisfied | Defined as a constrained optimisation problem: maximise classification accuracy while minimising feature count |
| 3.19 | Comprehensive report with visual aids | projects.md deliverables | ✅ Satisfied | JSON experiment logs + dashboard charts + convergence curves + operator comparison table |

**EA Score: 19/19 fully satisfied**

---

## Course 4: Computer Vision (CV)

| # | Requirement | Source | Status | Evidence in SRS |
|---|---|---|---|---|
| 4.1 | Image preprocessing pipeline (≥2 filter types) | projects.md CV req #1, SRS FR-CV-01 | ✅ Satisfied | Gaussian filter **and** Median filter both applied; results compared visually and numerically |
| 4.2 | Corner / keypoint detection | projects.md CV req #2, SRS FR-CV-02 | ✅ Satisfied | Harris Corner Detector implemented; keypoints displayed on images |
| 4.3 | Harris threshold tuning analysis | projects.md CV req #2 | ✅ Satisfied | Threshold tuning analysis documented as part of Harris implementation (FR-CV-02) |
| 4.4 | Multi-Scale Analysis — Gaussian/Laplacian Pyramid | projects.md CV req #3 (1 mark), SRS FR-CV-09 | ✅ Satisfied | `FR-CV-09` requires Gaussian or Laplacian pyramid at ≥3 scales, scale-space visualization, and inclusion in the CV evaluation report; owned by Member 2 |
| 4.5 | Feature extraction | projects.md CV req #4, SRS FR-CV-03 | ✅ Satisfied | SIFT feature extraction implemented |
| 4.6 | Feature matching | projects.md CV req #4, SRS FR-CV-04 | ✅ Satisfied | SIFT feature matching across image pairs with match visualization |
| 4.7 | Image segmentation | projects.md CV req #5, SRS FR-CV-05 | ✅ Satisfied | K-means clustering or Watershed algorithm to isolate sign regions |
| 4.8 | Classical classifier (non-DL) | projects.md CV req #6, SRS FR-CV-06 | ✅ Satisfied | Naive Bayes classifier trained on SIFT feature vectors |
| 4.9 | Dataset: Traffic Sign Detection (as per course topic) | projects.md CV §4, SRS §7 | ✅ Satisfied | GTSRB — German Traffic Sign Recognition Benchmark (50,000+ images, 43 classes) |
| 4.10 | Baseline accuracy reported | projects.md CV evaluation, SRS FR-CV-07 | ✅ Satisfied | CV baseline expected ~60–70%; overall accuracy + per-class precision, recall, F1 |
| 4.11 | IoU metric for segmentation | projects.md CV evaluation, SRS FR-CV-07 | ✅ Satisfied | `FR-CV-07` now explicitly requires IoU (Intersection over Union) for the segmentation step in the CV evaluation report |
| 4.12 | Matching accuracy metric | projects.md CV evaluation, SRS FR-CV-04 | ✅ Satisfied | `FR-CV-04` now requires a quantitative matching accuracy metric (correct-match ratio or RANSAC inlier ratio) with match visualization |
| 4.13 | Comparison against a more advanced method | SRS FR-CV-08, FR-DASH-04 | ✅ Satisfied | Three-way comparison: CV baseline < GA-optimized < full DL model (MobileNetV2 / EfficientNet-B0) |
| 4.14 | Data visualizations (samples, distributions) | SRS FR-D-04, FR-D-06 | ✅ Satisfied | 8+ visualizations: sample grids, class distribution charts (Member 2 deliverable) |
| 4.15 | Edge case / robustness testing | SRS §10 (constraints), cloud_reg §3 | ✅ Satisfied | Partially occluded signs, night-time photos, non-German variants tested via real-world phone photos |
| 4.16 | Real-world test beyond benchmark | cloud_reg §3, SRS §10 | ✅ Satisfied | ~20–30 real-world phone photos collected by team members; informal robustness check |
| 4.17 | Discussion of data limitations | SRS §10, §7 | ✅ Satisfied | Class imbalance, lighting variance, and German-only sign limitation documented |
| 4.18 | Final integrated pipeline | projects.md CV req #7 | ✅ Satisfied | All CV modules combined into one working application feeding the three-way comparison |

**CV Score: 14/18 fully satisfied, 3 partial/not addressed**

---

## Overall Summary

| Course | Requirements Satisfied | Partial / Gap | Total | Coverage |
|---|---|---|---|---|
| Cloud Computing | 14 ✅ | 0 | 14 | 100% |
| Advanced ML | 19 ✅ | 0 | 19 | 100% |
| Evolutionary Algorithms | 19 ✅ | 0 | 19 | 100% |
| Computer Vision | 18 ✅ | 0 | 18 | 100% |
| **Total** | **70 ✅** | **0** | **70** | **100%** |

---

## Gaps and Recommendations

All gaps are resolved. No open items remain.

| Gap | Description | Resolution | SRS Requirement |
|---|---|---|---|
| Gap 1 (CI/CD pipeline) | GitHub Actions workflow was "if time allows" | Now a hard requirement; blocks merge on failure | FR-CI-03 |
| Gap 2 (Dual architecture) | EfficientNet-B0 was optional fallback | Both models required to produce `.pth` and comparison table | FR-AML-02 |
| Gap 3 (Pyramid / scale-space) | Gaussian/Laplacian pyramid entirely absent from SRS | FR-CV-09 added; Member 2 owns ≥3-level pyramid + visualization | FR-CV-09 |
| Gap 4 (IoU + matching accuracy) | Segmentation IoU and SIFT matching accuracy not specified | FR-CV-07 and FR-CV-04 updated to require both metrics explicitly | FR-CV-07, FR-CV-04 |
| Gap 5 (Sign category in API) | `"category"` field missing from `/predict` response | FR-API-01 updated; static 43-class lookup table required; test asserts field | FR-API-01 |

---

## Stretch Requirements (Not Graded — Optional)

| # | Requirement | Source | Status |
|---|---|---|---|
| S.1 | Real-time webcam inference via API | cloud_reg FR-11 | Not started — post-demo if time allows |
| S.2 | Dashboard side-by-side comparison of GA vs PSO vs DE | cloud_reg FR-12 | Not started — requires implementing PSO/DE beyond GA |

---

*Generated: 2026-04-21 | Based on: `srs.md` v1.0, `projects.md` (CV + EA course sheets), `aml.md` (AML project guidelines), `cloud_project_registration.md`*
