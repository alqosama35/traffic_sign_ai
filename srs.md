# Software Requirements Specification (SRS)
## Traffic Sign Intelligence Platform
### Version 1.0 — Spring 2026

---

| Field | Value |
|---|---|
| **Project Title** | Traffic Sign Intelligence Platform |
| **Course Mapping** | Cloud Computing · Advanced ML · Evolutionary Algorithms · Computer Vision |
| **Team Type** | AI-Only |
| **Document Version** | 1.0 |
| **Date** | 2026-04-18 |

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Overall Description](#2-overall-description)
3. [Functional Requirements](#3-functional-requirements)
   - 3.1 Data Pipeline
   - 3.2 Computer Vision Baseline (CV)
   - 3.3 Transfer Learning Trainer (AML)
   - 3.4 GA Feature Selection Engine (EA)
   - 3.5 Inference API
   - 3.6 Containerization and CI/CD *(Gap 1 fix)*
   - 3.7 Cloud Deployment
   - 3.8 Results Dashboard
4. [Non-Functional Requirements](#4-non-functional-requirements)
5. [External Interface Requirements](#5-external-interface-requirements)
6. [System Architecture](#6-system-architecture)
7. [Data Requirements](#7-data-requirements)
8. [Security Requirements](#8-security-requirements)
9. [Evaluation and Acceptance Criteria](#9-evaluation-and-acceptance-criteria)
10. [Constraints and Assumptions](#10-constraints-and-assumptions)
11. [Team Roles and Responsibilities](#11-team-roles-and-responsibilities)

---

## 1. Introduction

### 1.1 Purpose

This document specifies the functional and non-functional requirements for the Traffic Sign Intelligence Platform — a cloud-deployed AI service that classifies traffic sign images using transfer learning, evolutionary feature selection, and a classical computer vision baseline. It serves as the binding technical reference for the development team and the evaluation reference for course supervisors.

### 1.2 Scope

The platform accepts a traffic sign image via a REST API, returns a classification result with confidence and top-3 alternatives, and exposes a results dashboard showing model evaluation metrics and evolutionary algorithm experiment results. All components are containerized and deployed on AWS.

### 1.3 Definitions

| Term | Definition |
|---|---|
| GTSRB | German Traffic Sign Recognition Benchmark — 50,000+ labeled traffic sign images across 43 classes |
| GA | Genetic Algorithm — the evolutionary search method used for feature selection |
| Transfer Learning | Reusing a model pre-trained on ImageNet and fine-tuning it on GTSRB |
| Feature Vector | Numerical representation of an image extracted from the penultimate layer of the trained DL model |
| Chromosome | A binary string in the GA where each bit represents inclusion/exclusion of one feature dimension |
| Fitness | Validation classification accuracy of a classifier trained on a GA-selected feature subset |
| ECS Fargate | AWS serverless container hosting service |
| ECR | AWS Elastic Container Registry — stores Docker images |

### 1.4 Overview

The system is divided into three offline services (trainer, EA engine, CV baseline) and one cloud-facing runtime service (inference API + dashboard). Requirements are grouped by module and tagged with the course they satisfy.

---

## 2. Overall Description

### 2.1 Product Perspective

The platform bridges four academic courses into one end-to-end system:

- **Computer Vision (CV):** classical hand-crafted feature pipeline as a comparison baseline
- **Advanced ML (AML):** deep transfer learning model as the primary classifier
- **Evolutionary Algorithms (EA):** GA-based feature selection to reduce dimensionality with minimal accuracy loss
- **Cloud Computing:** AWS infrastructure that deploys, stores, monitors, and makes the system publicly accessible

### 2.2 Product Functions (Summary)

1. Preprocess and augment the GTSRB dataset
2. Train a classical CV baseline (SIFT + Naive Bayes)
3. Fine-tune a deep learning model (MobileNetV2 **and** EfficientNet-B0)
4. Run GA feature selection experiments across 4 operator configurations
5. Serve predictions via a REST API deployed on AWS
6. Store all model artifacts and experiment logs in S3
7. Monitor API health and performance via CloudWatch
8. Display results on a web dashboard
9. Build, test, and deploy automatically via a CI/CD pipeline

### 2.3 User Classes

| User | Interaction |
|---|---|
| **Examiner / Supervisor** | Uses the live dashboard and Swagger UI to evaluate all four academic contributions |
| **Autonomous Driving Engineer** | Calls `/predict` via REST API for sign classification |
| **Researcher / Student** | Reads GA experiment logs and model comparison results from the dashboard |
| **Developer (team)** | Runs training scripts, pushes Docker images, manages AWS resources |

### 2.4 Operating Environment

- **Runtime platform:** AWS EC2 (t2.medium) or ECS Fargate
- **Container runtime:** Docker
- **Language:** Python 3.10+
- **Storage:** AWS S3
- **Monitoring:** AWS CloudWatch
- **CI/CD:** GitHub Actions

---

## 3. Functional Requirements

---

### 3.1 Data Pipeline

**FR-D-01:** The system shall download the GTSRB dataset from Kaggle (`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign-recognition-benchmark`).

**FR-D-02:** The system shall split the dataset into train (70%), validation (15%), and test (15%) subsets, using the pre-existing Kaggle train/test split as the base.

**FR-D-03:** All images shall be resized to 224×224 pixels and normalized to the ImageNet mean and standard deviation.

**FR-D-04:** The system shall analyze and document class distribution across all 43 classes and report the imbalance ratio.

**FR-D-05:** The system shall apply `WeightedRandomSampler` in the PyTorch DataLoader to compensate for class imbalance during training.

**FR-D-06:** The augmentation pipeline shall apply the following transforms to training images:
- Random brightness and contrast jitter
- Random rotation (±15°)
- Random horizontal flip

**FR-D-07:** The processed dataset shall be stored in S3 so it is downloaded once and never re-fetched during training or inference.

---

### 3.2 Computer Vision Baseline *(CV course)*

**FR-CV-01:** The system shall implement an image preprocessing pipeline using Gaussian and/or Median filtering to reduce noise.

**FR-CV-02:** The system shall apply the Harris Corner Detector to identify keypoints in traffic sign images.

**FR-CV-03:** The system shall extract SIFT (Scale-Invariant Feature Transform) features from the preprocessed images.

**FR-CV-04:** The system shall perform SIFT feature matching across image pairs, draw match visualizations, and report a quantitative matching accuracy metric (correct-match ratio or RANSAC inlier ratio) for a representative sample of image pairs.

**FR-CV-05:** The system shall apply image segmentation using K-means clustering or the Watershed algorithm to isolate sign regions.

**FR-CV-06:** The system shall train a Naive Bayes classifier on the extracted SIFT feature vectors.

**FR-CV-07:** The CV baseline shall be evaluated on the GTSRB test set and report: overall accuracy, per-class precision, recall, F1, IoU (Intersection over Union) for the segmentation step, and descriptor matching accuracy for the SIFT matching step.

**FR-CV-08:** The CV baseline accuracy shall be reported as the lower-bound comparison point in the three-way model comparison table.

**FR-CV-09:** The system shall implement a Gaussian or Laplacian image pyramid on sample GTSRB images, demonstrate scale-space analysis at a minimum of 3 pyramid levels, and produce a scale-space visualization showing sign detection at multiple scales. This visualization shall be included in the CV evaluation report.

---

### 3.3 Transfer Learning Trainer *(AML course)*

**FR-AML-01:** The system shall fine-tune MobileNetV2 pre-trained on ImageNet on the GTSRB training set.

**FR-AML-02 *(Gap 2 fix):*** The system shall **also** fine-tune EfficientNet-B0 pre-trained on ImageNet on the same GTSRB training set. Both runs shall complete and produce a saved `.pth` checkpoint. The MobileNetV2 model is the production model; the EfficientNet-B0 result is required for the architecture comparison table.

**FR-AML-03:** The training strategy shall freeze the backbone for the first 5 epochs, then unfreeze the last 2 blocks for fine-tuning.

**FR-AML-04:** The loss function shall be CrossEntropyLoss with per-class weights inversely proportional to class frequency.

**FR-AML-05:** The learning rate scheduler shall use StepLR or CosineAnnealingLR.

**FR-AML-06:** The trainer shall record and export loss and accuracy curves (train and validation) per epoch.

**FR-AML-07:** The trained model shall be exported as `traffic_sign_model_mobilenetv2.pth` and `traffic_sign_model_efficientnetb0.pth` and uploaded to S3.

**FR-AML-08:** The MobileNetV2 model shall achieve ≥90% overall accuracy on the GTSRB test set.

**FR-AML-09:** The system shall generate a 43×43 confusion matrix heatmap for the MobileNetV2 model.

**FR-AML-10:** The system shall produce a full classification report (per-class precision, recall, F1) for all 43 classes.

**FR-AML-11:** The system shall compute and report top-5 accuracy; target ≥98%.

**FR-AML-12:** The system shall identify and report the top-5 most confused sign pairs from the confusion matrix.

**FR-AML-13:** The system shall produce an architecture comparison table: MobileNetV2 vs EfficientNet-B0, reporting test accuracy and inference latency per image.

---

### 3.4 GA Feature Selection Engine *(EA course)*

**FR-EA-01:** The GA implementation shall be written entirely in custom Python + NumPy. No black-box optimization library (e.g., DEAP, scipy.optimize) shall be used.

**FR-EA-02:** Each chromosome shall be a binary string of length *n*, where *n* equals the dimensionality of the feature vector extracted from the penultimate layer of the trained AML model. Each bit encodes inclusion (1) or exclusion (0) of the corresponding feature dimension.

**FR-EA-03:** The fitness function shall be the validation set classification accuracy of a k-NN or linear SVM classifier trained on the feature subset selected by the chromosome.

**FR-EA-04:** Population size shall be 50 individuals, initialized uniformly at random.

**FR-EA-05:** The system shall implement Tournament Selection with tournament size k=3.

**FR-EA-06:** The system shall implement Roulette Wheel (fitness-proportionate) Selection as a second strategy.

**FR-EA-07:** The system shall implement Single-Point Crossover.

**FR-EA-08:** The system shall implement Uniform Crossover as a second crossover operator.

**FR-EA-09:** The system shall implement Bit-Flip Mutation with probability 0.01 per bit.

**FR-EA-10:** Termination shall occur after 100 generations or after 20 consecutive generations with no improvement in best fitness, whichever comes first.

**FR-EA-11:** The system shall implement Fitness Sharing within niches to maintain population diversity and prevent premature convergence.

**FR-EA-12:** The system shall run at least 4 complete experiment configurations: {Tournament, Roulette Wheel} × {Single-Point, Uniform} crossover.

**FR-EA-13:** For each experiment run, the system shall export a JSON log containing: run ID, operator configuration, fitness per generation, best chromosome, selected feature indices, best validation accuracy, and feature reduction ratio.

**FR-EA-14:** All JSON experiment logs shall be uploaded to the designated S3 prefix (`s3://<bucket>/ea-experiments/`).

**FR-EA-15:** The GA-optimized model shall achieve ≥88% validation accuracy while using ≤70% of the original feature dimensions (≥30% reduction).

**FR-EA-16:** The accuracy drop from the full-feature model to the GA-optimized model shall be <3 percentage points.

---

### 3.5 Inference API

**FR-API-01:** The system shall expose a `POST /predict` endpoint that accepts a multipart image upload and returns a JSON response containing: predicted class label, sign category (e.g., "prohibition", "warning", "mandatory", "informational"), confidence score (0–1), and top-3 alternative predictions with their confidence scores. The category shall be derived from a static lookup table mapping each of the 43 GTSRB class IDs to its sign category.

**FR-API-02:** The system shall expose a `GET /health` endpoint returning HTTP 200 and `{"status": "ok"}` when the service is running.

**FR-API-03:** The API response shall include the disclaimer: `"note": "Screening tool only — not for use in safety-critical autonomous systems"`.

**FR-API-04:** All input images shall be preprocessed in memory (resize, normalize) and discarded after inference. No image data shall be persisted.

**FR-API-05:** The API shall serve interactive documentation at `GET /docs` (Swagger UI).

**FR-API-06:** A `test_api.py` script shall send at least 10 representative test images to `/predict` and assert that: HTTP status is 200, `class` field is a string, `category` field is a non-empty string, `confidence` is between 0 and 1, `top_3` contains exactly 3 entries.

**FR-API-07:** API response time shall be <500ms per request under normal load.

---

### 3.6 Containerization and CI/CD *(Gap 1 fix)*

**FR-CI-01:** The inference service shall be packaged in a `Dockerfile` using a Python 3.10 slim base image.

**FR-CI-02:** A `docker-compose.yml` shall define the API service, expose port 8000, and mount the model path as an environment variable.

**FR-CI-03 *(Gap 1 fix — CI/CD pipeline):*** A GitHub Actions workflow file (`.github/workflows/deploy.yml`) shall automate the following pipeline on every push to the `main` branch:
1. **Build:** `docker build` the inference image
2. **Test:** run `test_api.py` against the locally started container
3. **Push:** push the tagged image to AWS ECR (using `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` GitHub secrets)
4. **Deploy:** trigger an ECS service update (or SSH into EC2 and `docker pull` + restart) to deploy the new image

**FR-CI-04:** The workflow shall fail and block merge if any of the four pipeline stages fail.

**FR-CI-05:** Docker image tags shall follow the convention `<ecr-repo>:<git-sha>` to ensure every deployed version is traceable to a commit.

---

### 3.7 Cloud Deployment *(Cloud course)*

**FR-CLD-01:** The FastAPI inference container shall be hosted on AWS EC2 (t2.medium) or AWS ECS Fargate and be publicly accessible via HTTP during the demo.

**FR-CLD-02:** Model artifacts (`traffic_sign_model_mobilenetv2.pth`, `traffic_sign_model_efficientnetb0.pth`) and EA experiment JSON logs shall be stored in a dedicated AWS S3 bucket.

**FR-CLD-03:** The inference container shall load the model file from S3 at startup using read-only IAM role credentials.

**FR-CLD-04:** An AWS IAM role with least-privilege S3 read-only access shall be attached to the inference service. No hardcoded credentials shall appear in code or Dockerfiles.

**FR-CLD-05:** AWS CloudWatch shall collect and display: API request count, p50/p95 response latency, HTTP error rate, and container memory usage.

**FR-CLD-06:** A CloudWatch alarm shall trigger when the API HTTP error rate exceeds 1% over any 5-minute window.

**FR-CLD-07:** AWS billing alerts shall be configured at $10 and $20 thresholds. The EC2 instance shall be terminated after the final demo.

---

### 3.8 Results Dashboard

**FR-DASH-01:** The dashboard shall display a GA convergence page showing fitness (validation accuracy) vs generation number for all 4 experiment configurations on a single chart.

**FR-DASH-02:** The dashboard shall display a feature reduction table: full-feature accuracy, GA-selected accuracy, number of features used, and percentage reduction — for all 4 GA configurations.

**FR-DASH-03:** The dashboard shall display the 43×43 confusion matrix heatmap for the MobileNetV2 model.

**FR-DASH-04:** The dashboard shall display the three-way model comparison table: CV Baseline vs MobileNetV2 vs EfficientNet-B0 vs GA-Optimized model, reporting test accuracy and (for DL models) inference latency.

**FR-DASH-05:** The dashboard shall include a Live Prediction tab with an image upload form that calls `/predict` and displays the result inline.

**FR-DASH-06:** Dashboard data shall be read from S3 experiment logs (not from a live database) so the dashboard remains functional even when the trainer is offline.

---

## 4. Non-Functional Requirements

### 4.1 Performance

| ID | Requirement |
|---|---|
| NFR-PERF-01 | API `/predict` response time shall be <500ms per request (p95) under normal load |
| NFR-PERF-02 | API uptime shall be 100% during the live demo session |
| NFR-PERF-03 | CloudWatch HTTP error rate shall be <1% over 100 consecutive test requests |
| NFR-PERF-04 | GA fitness evaluation per generation shall complete in <60 seconds (enforced by using k-NN/linear SVM, not the full DL model) |

### 4.2 Reliability

| ID | Requirement |
|---|---|
| NFR-REL-01 | The Docker container shall restart automatically on crash (ECS restart policy or `restart: always` in docker-compose) |
| NFR-REL-02 | All EA experiment results shall be persisted to S3 before the experiment process exits |

### 4.3 Maintainability

| ID | Requirement |
|---|---|
| NFR-MAINT-01 | The codebase shall be organized in three top-level directories: `trainer/`, `api/`, `dashboard/` |
| NFR-MAINT-02 | A `requirements.txt` (or `pyproject.toml`) shall pin all Python dependency versions |
| NFR-MAINT-03 | Environment-specific values (S3 bucket name, model path, API key) shall be passed via environment variables, never hardcoded |

### 4.4 Security

| ID | Requirement |
|---|---|
| NFR-SEC-01 | The `/predict` endpoint shall require a valid API key passed in the `X-API-Key` request header; requests without a valid key shall return HTTP 401 |
| NFR-SEC-02 | IAM roles shall follow the principle of least privilege; the inference container shall have S3 read access only |
| NFR-SEC-03 | No uploaded image data shall be written to disk or stored in any persistent store |
| NFR-SEC-04 | AWS credentials shall never appear in source code, Dockerfiles, or committed `.env` files |

### 4.5 Portability

| ID | Requirement |
|---|---|
| NFR-PORT-01 | The Docker container shall run identically on a developer's local machine and on AWS |
| NFR-PORT-02 | The `docker-compose.yml` shall allow running the full API service locally without an AWS dependency (model loaded from a local path via env var override) |

---

## 5. External Interface Requirements

### 5.1 REST API

**Endpoint:** `POST /predict`

```
Request
  Content-Type: multipart/form-data
  Header: X-API-Key: <token>
  Body: file=<image bytes>

Response 200
{
  "class": "Speed limit (30km/h)",
  "category": "prohibition",
  "confidence": 0.97,
  "top_3": [
    {"class": "Speed limit (30km/h)", "confidence": 0.97},
    {"class": "Speed limit (50km/h)", "confidence": 0.02},
    {"class": "Speed limit (80km/h)", "confidence": 0.01}
  ],
  "note": "Screening tool only — not for use in safety-critical autonomous systems"
}

Response 401  { "detail": "Invalid or missing API key" }
Response 422  { "detail": "Unsupported image format" }
```

**Endpoint:** `GET /health`

```
Response 200  { "status": "ok" }
```

### 5.2 S3 Bucket Structure

```
s3://<bucket>/
  models/
    traffic_sign_model_mobilenetv2.pth
    traffic_sign_model_efficientnetb0.pth
  ea-experiments/
    run_tournament_singlepoint.json
    run_tournament_uniform.json
    run_roulette_singlepoint.json
    run_roulette_uniform.json
  dataset/
    train/   val/   test/
```

### 5.3 EA Experiment Log Schema

```json
{
  "run_id": "tournament_singlepoint_001",
  "selection": "tournament",
  "crossover": "single_point",
  "population_size": 50,
  "generations_run": 87,
  "fitness_per_generation": [0.71, 0.74, ...],
  "best_fitness": 0.903,
  "best_chromosome": [1, 0, 1, ...],
  "selected_feature_indices": [0, 2, 5, ...],
  "total_features": 1280,
  "selected_features": 510,
  "reduction_ratio": 0.398
}
```

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        OFFLINE (local / batch)                  │
│                                                                 │
│  trainer/                    ea/                  cv_baseline/  │
│  ┌──────────────────┐   ┌────────────────┐   ┌──────────────┐  │
│  │ Data Pipeline    │   │ GA Engine      │   │ SIFT+NB      │  │
│  │ Augmentation     │──▶│ Chromosome     │   │ Harris       │  │
│  │ MobileNetV2      │   │ Fitness Eval   │   │ K-means seg  │  │
│  │ EfficientNet-B0  │   │ 4 Configs      │   │ Eval report  │  │
│  └────────┬─────────┘   └───────┬────────┘   └──────────────┘  │
│           │ .pth                │ JSON logs                     │
└───────────┼─────────────────────┼───────────────────────────────┘
            │                     │
            ▼                     ▼
┌──────────────────────── AWS S3 ─────────────────────────────────┐
│  models/  ·  ea-experiments/  ·  dataset/                       │
└──────────────────────────────────────────────────────────────────┘
            │                     │
            ▼                     ▼
┌─────────────────────── AWS ECS / EC2 ───────────────────────────┐
│                                                                 │
│  api/                              dashboard/                   │
│  ┌──────────────────────────┐      ┌───────────────────────┐   │
│  │ FastAPI                  │      │ Static HTML/JS         │   │
│  │ POST /predict            │      │ GA convergence charts  │   │
│  │ GET  /health             │      │ Confusion matrix       │   │
│  │ GET  /docs               │      │ Model comparison table │   │
│  └──────────────────────────┘      │ Live prediction tab    │   │
│                                    └───────────────────────┘   │
│  CloudWatch: latency · errors · memory                          │
└─────────────────────────────────────────────────────────────────┘
            ▲
            │  GitHub Actions CI/CD
            │  build → test → push ECR → deploy
            │
┌─────── GitHub repo ─────────────────────────────────────────────┐
│  .github/workflows/deploy.yml                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Data Requirements

| Attribute | Value |
|---|---|
| Dataset | GTSRB — German Traffic Sign Recognition Benchmark |
| Source | Kaggle: `meowmeowmeowmeowmeow/gtsrb-german-traffic-sign-recognition-benchmark` |
| Size | 50,000+ images, 43 classes |
| Input format | JPEG, variable sizes (15×15 to 250×250 px) |
| Normalized size | 224×224 px (all models) |
| Split | Train 70% / Val 15% / Test 15% |
| Privacy | No personal data; all images are public road sign photos |
| License | Public, academic use permitted |
| Class imbalance | ~200 images (rare classes) to ~2,000 (common classes); mitigated with WeightedRandomSampler |

---

## 8. Security Requirements

| Requirement | Detail |
|---|---|
| API authentication | `X-API-Key` header required on `/predict`; HTTP 401 on failure |
| IAM least privilege | Inference container: S3 `GetObject` on `models/` prefix only |
| No data persistence | Images processed in memory and immediately discarded |
| Secrets management | All credentials via environment variables or AWS Secrets Manager; none in code or Docker images |
| GDPR | No personal data collected or stored; no consent workflow required |

---

## 9. Evaluation and Acceptance Criteria

### 9.1 AML Acceptance Criteria

| Metric | Threshold |
|---|---|
| MobileNetV2 test accuracy | ≥ 90% |
| EfficientNet-B0 test accuracy | Reported (no minimum — comparison only) |
| Top-5 accuracy | ≥ 98% |
| Confusion matrix | Produced and visualized (43×43) |
| Classification report | All 43 classes, precision/recall/F1 |

### 9.2 EA Acceptance Criteria

| Metric | Threshold |
|---|---|
| Configurations run | 4 (2 selection × 2 crossover) |
| Feature reduction | ≥ 30% reduction from full feature set |
| Accuracy degradation | < 3 percentage points vs full-feature model |
| Convergence curve | Visible convergence (fitness improves over generations) |
| Experiment logs in S3 | All 4 JSON logs present and valid |

### 9.3 CV Acceptance Criteria

| Metric | Threshold |
|---|---|
| CV baseline accuracy | Reported (expected 60–70%) |
| SIFT + Naive Bayes pipeline | End-to-end functional on GTSRB test set |
| Full evaluation report | Precision, recall, F1 per class; IoU for segmentation; matching accuracy for SIFT |
| Gaussian/Laplacian Pyramid | Implemented at ≥3 scales; scale-space visualization produced |
| Harris threshold analysis | Threshold tuning effect documented with visual comparison |

### 9.4 Cloud Acceptance Criteria

| Metric | Threshold |
|---|---|
| API publicly accessible | Live URL responds during demo |
| `/predict` response time | < 500ms (p95) |
| CloudWatch error rate | < 1% over 100 requests |
| CI/CD pipeline | All 4 stages pass on `main` branch push |
| S3 artifacts present | Model `.pth` files + all EA logs accessible |

---

## 10. Constraints and Assumptions

| Type | Detail |
|---|---|
| **Budget** | AWS spend capped at $20 via billing alerts; EC2 terminated post-demo |
| **Timeline** | Single semester (Spring 2026); GA experiments start Week 2 in parallel with model training |
| **Library constraint** | GA must be implemented from scratch in Python + NumPy — no DEAP or scipy.optimize |
| **Model serving** | Only MobileNetV2 is served in production; EfficientNet-B0 is trained offline for comparison |
| **Dataset** | GTSRB covers German signs only; generalization to other countries is a known limitation |
| **Real-world testing** | Phone-camera photos used as informal robustness checks only; not part of official evaluation |
| **Disclaimer** | All API responses must include the safety disclaimer; the system is not validated for autonomous vehicle use |

---

*This SRS supersedes the informal proposal (cloud_proposal.md) and incorporates fixes for all identified requirement gaps: the CI/CD pipeline (Gap 1) is a hard requirement under FR-CI-03; the dual-architecture comparison (Gap 2) is a hard requirement under FR-AML-02; the Gaussian/Laplacian image pyramid and scale-space analysis (Gap 3) is a hard requirement under FR-CV-09; IoU and matching accuracy metrics (Gap 4) are required under FR-CV-07; and the sign category field in the API response (Gap 5) is a hard requirement under FR-API-01.*

---

## 11. Team Roles and Responsibilities

### 11.1 Role Summary

| Member | Role | SRS Requirements Owned |
|---|---|---|
| Member 1 | Data Engineer | FR-D-01 → FR-D-07 |
| Member 2 | CV Engineer | FR-CV-01 → FR-CV-09 |
| Member 3 | ML Engineer | FR-AML-01 → FR-AML-13 |
| Member 4 | EA Engineer | FR-EA-01 → FR-EA-16 |
| Member 5 | API Developer | FR-API-01 → FR-API-07 |
| Member 6 | DevOps / CI-CD Engineer | FR-CI-01 → FR-CI-05 |
| Member 7 | Cloud + Dashboard Engineer | FR-CLD-01 → FR-CLD-07, FR-DASH-01 → FR-DASH-06 |

---

### 11.2 Detailed Role Definitions

#### Member 1 — Data Engineer
**Owns:** FR-D-01 → FR-D-07

| Deliverable | Linked Requirement |
|---|---|
| Download GTSRB from Kaggle and verify integrity | FR-D-01 |
| Organize train (70%) / val (15%) / test (15%) splits | FR-D-02 |
| Resize all images to 224×224 and apply ImageNet normalization | FR-D-03 |
| Class balance analysis report (imbalance ratio per class) | FR-D-04 |
| Configure `WeightedRandomSampler` for the training DataLoader | FR-D-05 |
| 8+ EDA visualizations: sample grids, class distribution bar chart, image size histogram | FR-D-06, FR-D-04 |
| Upload organized dataset to S3 (`s3://<bucket>/dataset/`) | FR-D-07 |

**Depends on:** nothing — starts immediately.
**Hands off to:** M2 (raw dataset), M3 (splits + sampler config).

---

#### Member 2 — CV Engineer
**Owns:** FR-CV-01 → FR-CV-09

| Deliverable | Linked Requirement |
|---|---|
| Gaussian / Median filter preprocessing pipeline with visual and numerical comparison | FR-CV-01 |
| Harris Corner Detector implementation, keypoint visualization, and threshold tuning analysis | FR-CV-02 |
| Gaussian or Laplacian image pyramid (≥3 scales) with scale-space visualization | FR-CV-09 |
| SIFT feature extraction | FR-CV-03 |
| SIFT feature matching across image pairs with match visualization and matching accuracy metric | FR-CV-04 |
| Image segmentation using K-means or Watershed | FR-CV-05 |
| Naive Bayes classifier trained on SIFT feature vectors | FR-CV-06 |
| CV baseline evaluation report: accuracy, per-class precision/recall/F1, IoU (segmentation), matching accuracy (SIFT) | FR-CV-07 |
| Baseline accuracy figure for the three-way comparison table | FR-CV-08 |

**Depends on:** M1 (dataset splits).
**Hands off to:** M7 (baseline accuracy numbers for the dashboard comparison table).

---

#### Member 3 — ML Engineer
**Owns:** FR-AML-01 → FR-AML-13

| Deliverable | Linked Requirement |
|---|---|
| Augmentation pipeline (brightness/contrast jitter, rotation ±15°, horizontal flip) | FR-AML-01, FR-AML-04 |
| MobileNetV2 fine-tuning: freeze backbone 5 epochs, unfreeze last 2 blocks | FR-AML-01, FR-AML-03 |
| EfficientNet-B0 fine-tuning on same GTSRB split *(Gap 2 fix)* | FR-AML-02 |
| CrossEntropyLoss with per-class weights + `WeightedRandomSampler` | FR-AML-04, FR-AML-05 |
| LR scheduler (StepLR or CosineAnnealingLR) | FR-AML-06 |
| Loss and accuracy curves (train + val) exported per epoch | FR-AML-07 |
| Export `traffic_sign_model_mobilenetv2.pth` and `traffic_sign_model_efficientnetb0.pth` to S3 | FR-AML-08 |
| MobileNetV2 test accuracy ≥ 90% | FR-AML-09 |
| 43×43 confusion matrix heatmap | FR-AML-10 |
| Full classification report (precision, recall, F1) for all 43 classes | FR-AML-11 |
| Top-5 accuracy ≥ 98% | FR-AML-12 |
| Architecture comparison table: MobileNetV2 vs EfficientNet-B0 (accuracy + latency) | FR-AML-13 |

**Depends on:** M1 (dataset splits + sampler config).
**Hands off to:** M4 (trained model for feature vector extraction), M5 (`.pth` file for inference).

---

#### Member 4 — EA Engineer
**Owns:** FR-EA-01 → FR-EA-16

| Deliverable | Linked Requirement |
|---|---|
| Custom GA in Python + NumPy — no external optimization library | FR-EA-01 |
| Binary chromosome representation over penultimate-layer feature vectors | FR-EA-02 |
| Fitness function: validation accuracy of k-NN or linear SVM on selected subset | FR-EA-03 |
| Population of 50, randomly initialized | FR-EA-04 |
| Tournament Selection (k=3) | FR-EA-05 |
| Roulette Wheel Selection (second strategy) | FR-EA-06 |
| Single-Point Crossover | FR-EA-07 |
| Uniform Crossover (second operator) | FR-EA-08 |
| Bit-Flip Mutation (rate 0.01/bit) | FR-EA-09 |
| Termination: 100 generations or 20 generations without improvement | FR-EA-10 |
| Fitness sharing for diversity preservation | FR-EA-11 |
| 4 complete experiment runs: {Tournament, Roulette} × {Single-Point, Uniform} | FR-EA-12 |
| JSON log per run (fitness curve, best chromosome, selected indices, reduction ratio) | FR-EA-13 |
| Upload all 4 JSON logs to `s3://<bucket>/ea-experiments/` | FR-EA-14 |
| GA model accuracy ≥ 88%, feature reduction ≥ 30% | FR-EA-15 |
| Accuracy drop vs full-feature model < 3 percentage points | FR-EA-16 |

**Depends on:** M3 (trained model to extract feature vectors from).
**Hands off to:** M7 (JSON experiment logs for the dashboard).

---

#### Member 5 — API Developer
**Owns:** FR-API-01 → FR-API-07

| Deliverable | Linked Requirement |
|---|---|
| `POST /predict` endpoint: accepts image, returns class + category + confidence + top-3 | FR-API-01 |
| `GET /health` endpoint returning `{"status": "ok"}` | FR-API-02 |
| Safety disclaimer included in every `/predict` response | FR-API-03 |
| In-memory image preprocessing (resize, normalize); no image persistence | FR-API-04 |
| Swagger UI at `GET /docs` | FR-API-05 |
| `test_api.py`: 10+ test images, asserts status 200, valid class, non-empty category, confidence ∈ [0,1], top_3 length = 3 | FR-API-06 |
| Response time < 500ms verified locally before handoff | FR-API-07 |

**Depends on:** M3 (`.pth` model file).
**Hands off to:** M6 (complete API codebase ready to containerize).

---

#### Member 6 — DevOps / CI-CD Engineer *(Gap 1 fix — new dedicated role)*
**Owns:** FR-CI-01 → FR-CI-05

| Deliverable | Linked Requirement |
|---|---|
| `Dockerfile` using Python 3.10 slim base image | FR-CI-01 |
| `docker-compose.yml`: exposes port 8000, model path via env var | FR-CI-02 |
| `.github/workflows/deploy.yml` with 4 mandatory stages: build → test → push ECR → deploy | FR-CI-03 |
| Pipeline configured to fail and block merge on any stage failure | FR-CI-04 |
| Docker image tagged as `<ecr-repo>:<git-sha>` | FR-CI-05 |

**Depends on:** M5 (API codebase to containerize).
**Hands off to:** M7 (built Docker image in ECR ready for AWS deployment).

---

#### Member 7 — Cloud + Dashboard Engineer
**Owns:** FR-CLD-01 → FR-CLD-07, FR-DASH-01 → FR-DASH-06

| Deliverable | Linked Requirement |
|---|---|
| AWS EC2 (t2.medium) or ECS Fargate hosting the inference container | FR-CLD-01 |
| S3 bucket created with `models/`, `ea-experiments/`, `dataset/` prefixes | FR-CLD-02 |
| Inference container loads model from S3 at startup via IAM role | FR-CLD-03 |
| IAM role with least-privilege S3 `GetObject` access on `models/` only | FR-CLD-04 |
| CloudWatch collecting request count, p50/p95 latency, error rate, memory | FR-CLD-05 |
| CloudWatch alarm: triggers when error rate > 1% over any 5-minute window | FR-CLD-06 |
| Billing alerts at $10 and $20; EC2 teardown plan documented | FR-CLD-07 |
| Dashboard: GA convergence chart (4 configs on one chart) | FR-DASH-01 |
| Dashboard: feature reduction table (accuracy, features used, % reduction per config) | FR-DASH-02 |
| Dashboard: 43×43 confusion matrix heatmap | FR-DASH-03 |
| Dashboard: three-way comparison table (CV Baseline, MobileNetV2, EfficientNet-B0, GA-Optimized) | FR-DASH-04 |
| Dashboard: Live Prediction tab calling `/predict` inline | FR-DASH-05 |
| Dashboard reads from S3 experiment logs — no live database dependency | FR-DASH-06 |

**Depends on:** M6 (Docker image in ECR), M4 (EA JSON logs in S3), M2 (CV baseline accuracy), M3 (model evaluation metrics).

---

### 11.3 Implementation Order

**Approach: Hybrid (sequential phases with internal parallelism)**

Pure sequential execution wastes team capacity — members sit idle waiting for unrelated work to finish. Pure parallel execution is impossible because hard data dependencies exist: M4 cannot run the GA without M3's trained model, M5 cannot build the API without the `.pth` file, and M6 cannot containerize code that does not yet exist. The hybrid approach resolves this by enforcing sequential ordering only where a true dependency exists, and running members in parallel everywhere else.

- **Phase 1 is forced sequential** — M1 is the single foundation; no dataset means no work for anyone.
- **Phase 2 is parallel** — M2 (CV baseline) and M3 (ML training) both depend only on M1's output and share no code or artifacts with each other.
- **Phase 3 is parallel** — M4 (GA engine) and M5 (API) both unblock on M3's `.pth` files and are fully independent of each other.
- **Phase 4 is sequential** — M6 cannot containerize an API that does not exist yet.
- **Phase 5 is a joining phase** — M7 waits for both M6 (Docker image) and M4 (EA logs), the two slowest parallel tracks, before AWS deployment and dashboard work can begin.

**Critical path risk:** M4 (GA experiments — 4 runs × 100 generations) is the longest task in Phase 3 and directly gates M7. Mitigation: pre-compute feature vectors from M3's model offline before GA starts, so each fitness evaluation runs a lightweight k-NN/SVM only, not the full deep model (FR-EA-03).

| Phase | Members Working | Execution Mode | Condition to Start |
|---|---|---|---|
| **Phase 1** | M1 | Sequential | Immediately |
| **Phase 2** | M2, M3 | Parallel | M1 delivers dataset splits to S3 |
| **Phase 3** | M4, M5 | Parallel | M3 delivers `.pth` files to S3 |
| **Phase 4** | M6 | Sequential | M5 delivers complete API codebase |
| **Phase 5** | M7 | Sequential (join) | M6 delivers Docker image in ECR AND M4 delivers EA logs to S3 |

> M2 (CV baseline) completes independently within Phase 2. Its output (baseline accuracy + evaluation report) is not needed until M7 builds the dashboard in Phase 5.

### 11.4 Handoff Artifacts

| From | To | Artifact |
|---|---|---|
| M1 → M2, M3 | Dataset splits in `s3://<bucket>/dataset/` |
| M3 → M4 | `traffic_sign_model_mobilenetv2.pth` in S3 (for feature extraction) |
| M3 → M5 | `traffic_sign_model_mobilenetv2.pth` in S3 (for inference) |
| M4 → M7 | 4× JSON experiment logs in `s3://<bucket>/ea-experiments/` |
| M2 → M7 | CV baseline accuracy + evaluation report |
| M3 → M7 | Confusion matrix, classification report, architecture comparison table |
| M5 → M6 | Complete FastAPI codebase (`api/` directory) |
| M6 → M7 | Docker image pushed to AWS ECR (`<ecr-repo>:<git-sha>`) |
