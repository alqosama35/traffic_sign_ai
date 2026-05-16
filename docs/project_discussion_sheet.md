# PROJECT DISCUSSION – DETAILED SHEET

| Course | Semester | Team ID | Discussion Time | TA | Final Grade |
|---|---|---|---|---|---|
| Cloud Computing / Integrated Projects | Spring 2026 | *(fill in)* | *(fill in)* | *(fill in)* | |

---

## 1. Project and Team Snapshot

### Project Identity

| Field | Value |
|---|---|
| **Project No.** | *(fill in)* |
| **Project Title** | Traffic Sign Intelligence Platform |
| **Team Type** | ☑ AI |
| **Idea Source** | ☑ Bank (AML Project 3 + CV Project 4 + EA Project 3, integrated into one platform) |
| **Related Courses** | Cloud Computing · Advanced ML (AML) · Evolutionary Algorithms (EA) · Computer Vision (CV) |
| **Project Bank Ref.** | AML-P3 (Traffic Sign Recognition) · CV-P4 (Traffic Sign Detection) · EA-P3 (Feature Selection with GAs) |
| **One-line Summary** | A cloud-deployed AI service that classifies 43 categories of German traffic signs using MobileNetV2 transfer learning, GA-based feature selection, and a classical SIFT + Naive Bayes baseline, served via FastAPI on AWS EC2 with S3 storage and CloudWatch monitoring. |

---

### Contacts / Links

| Field | Value |
|---|---|
| **Team Leader Email** | *(fill in)* |
| **GitHub Link** | *(fill in)* |
| **Demo / Deployment Link** | *(fill in — EC2 public IP or domain after deployment)* |

---

### Project Snapshot

**Problem being solved:**
Traffic sign recognition for autonomous driving requires solving four layers at once: feature selection (EA), a classical CV baseline, deep learning via transfer learning (AML), and reliable cloud deployment so predictions are served publicly and all experiments are stored and reproducible. This project integrates all four into one platform.

**Target users:**
- Primary: researchers and students benchmarking traffic sign recognition pipelines
- Secondary: autonomous driving engineers needing a tested, cloud-deployed inference API
- Tertiary: course supervisors validating that all four academic contributions are measurable and non-overlapping

**Minimum viable scope:**
1. GTSRB dataset downloaded, split (70/15/15), and preprocessed (224×224, ImageNet normalization)
2. MobileNetV2 fine-tuned to ≥90% test accuracy, exported as `traffic_sign_model_mobilenetv2.pth`
3. GA feature selection run with at least 4 operator configurations (2 selection × 2 crossover)
4. FastAPI `/predict` endpoint returning class label, category, confidence, and top-3 alternatives
5. Docker container building and running locally (Dockerfile + docker-compose.yml)
6. AWS EC2 deployment live — `/predict` endpoint publicly accessible
7. S3 bucket storing the model `.pth` and all 4 GA experiment JSON logs
8. CloudWatch collecting API latency, error rate, and memory metrics
9. Results dashboard showing GA convergence curves, confusion matrix, and three-way model comparison
10. `test_api.py` passing all assertions (15 test cases)

**Bonus / Enrichment:**
- Real-time webcam inference demo via the dashboard Live Prediction tab
- EfficientNet-B0 vs MobileNetV2 architecture comparison (stretch — cut first if time is tight)
- Multiple GA variant comparison: GA vs DE vs PSO on the same feature selection task
- Automated S3-triggered retraining pipeline when new labelled data is uploaded
- Second mutation operator (Swap Mutation) with ≥2 additional GA configurations (recommended in EA course)
- Two survivor selection strategies (generational replacement vs elitism) compared (recommended in EA course)
- Gaussian/Laplacian image pyramid at ≥3 scales with scale-space visualization (CV bonus)
- Harris corner threshold tuning analysis (CV bonus)

---

## 2. Team Details

| # | Student Name | Student ID | Main Role / Contribution | Attendance |
|---|---|---|---|---|
| 1 | *(fill in)* | *(fill in)* | **Data Engineer** — GTSRB download, train/val/test split (70/15/15), class balance analysis, WeightedRandomSampler, augmentation pipeline (brightness, contrast, rotation), 8+ EDA visualizations, dataset uploaded to S3 | ☐ P ☐ A |
| 2 | *(fill in)* | *(fill in)* | **CV Engineer** — Gaussian/Median filter preprocessing, Harris Corner Detector, Gaussian/Laplacian pyramid (≥3 scales), SIFT extraction and matching with visualization, K-means segmentation, Naive Bayes classifier, full CV evaluation report (accuracy, F1, IoU, matching accuracy) | ☐ P ☐ A |
| 3 | *(fill in)* | *(fill in)* | **ML Engineer** — MobileNetV2 fine-tuning (freeze 5 epochs, unfreeze last 2 blocks), CrossEntropyLoss with class weights, CosineAnnealingLR schedule, loss/accuracy curves, export `.pth`, confusion matrix (43×43), full classification report, top-5 accuracy | ☐ P ☐ A |
| 4 | *(fill in)* | *(fill in)* | **EA Engineer** — Custom GA in Python/NumPy (no DEAP), binary chromosome over MobileNetV2 penultimate-layer features, k-NN/SVM fitness function, Tournament + Roulette selection, Single-Point + Uniform crossover, Bit-Flip mutation, fitness sharing for diversity, 4 experiment configs, JSON logs uploaded to S3 | ☐ P ☐ A |
| 5 | *(fill in)* | *(fill in)* | **API Developer** — FastAPI `/predict` (class + category + confidence + top-3 + disclaimer) and `/health` endpoints, Swagger UI at `/docs`, in-memory preprocessing, `test_api.py` with 15 test cases, response time <500ms verified locally | ☐ P ☐ A |
| 6 | *(fill in)* | *(fill in)* | **DevOps / CI-CD Engineer** — Dockerfile (Python 3.10 slim), docker-compose.yml (port 8000, env vars), GitHub Actions deploy.yml (build → test → push ECR → deploy EC2 via SSH), image tagging as `<ecr-repo>:<git-sha>` | ☐ P ☐ A |
| 7 | *(fill in)* | *(fill in)* | **Cloud + Dashboard Engineer** — AWS EC2 t2.medium setup, S3 bucket + IAM roles (least-privilege S3 GetObject), CloudWatch metrics (latency, error rate, memory) + alarm (>1% error rate / 5 min), billing alerts ($10/$20), results dashboard (GA convergence, confusion matrix, three-way comparison table, Live Prediction tab) | ☐ P ☐ A |
| 8 | *(fill in — if applicable)* | *(fill in)* | *(fill in)* | ☐ P ☐ A |

> **Interpretation note:** This is an AI team. Cloud depth is expected in hosting inference/experiments, data/result storage, deployment, monitoring, and cost feasibility.

---

## 3. Evaluation, Cloud Review, and Decision

### A. General Project Evaluation

| Criterion / focus | What to check | Score (0–4) | Notes |
|---|---|---|---|
| **Problem clarity** | Is the problem clear and meaningful? | | Traffic sign recognition for autonomous driving — well-defined, safety-relevant, maps to GTSRB benchmark with 43 measurable classes |
| **Scope realism** | Can the team finish it realistically? | | 7-member team with dedicated roles; MVP is 10 deliverables; cloud infra is the remaining work (all code complete) |
| **Course alignment** | Does it fit the claimed courses naturally? | | 4-course integration is non-overlapping: CV = SIFT baseline, AML = MobileNetV2, EA = GA feature selection, Cloud = AWS deployment/storage/monitoring |
| **Technical depth** | Is there enough substance for this team type? | | AI team: three-way model comparison, 4 GA operator configurations, custom GA from scratch, FastAPI + Docker + AWS + CloudWatch |
| **Evaluation / testing readiness** | Are metrics, experiments, or tests defined? | | AML: ≥90% accuracy, confusion matrix, F1; EA: ≥30% feature reduction, <3% accuracy drop, convergence curves; Cloud: <500ms latency, <1% error rate; `test_api.py` with 15 assertions |
| **Discussion performance** | Did the team explain and defend the idea well? | | *(TA fills in during discussion)* |
| **Total General Score / 24** | | | |

*Scoring scale: 0 = Not demonstrated, 1 = Weak, 2 = Basic, 3 = Good, 4 = Strong*

---

### B. Cloud Computing Evaluation

| Cloud criterion | What to check | Student Response | Score (0–4) | Notes |
|---|---|---|---|---|
| **Architecture relevance** | Is the cloud design meaningful for the project? | The cloud layer serves three substantive purposes: (1) **Reproducible experiment storage** — every GA run (fitness curves, best chromosome, selected features, accuracy) is uploaded to S3 as a structured JSON log so experiments can be audited without relying on a local machine. (2) **Scalable inference serving** — FastAPI on EC2 t2.medium makes `/predict` publicly accessible during the demo and beyond. (3) **Observability** — CloudWatch monitors request latency, error rate, and memory, with an alarm on >1% error rate. Cloud is not decorative — it enables the academic contributions to be verifiable and reproducible. | | |
| **Service selection** | Are chosen cloud services appropriate and justified? | **EC2 t2.medium** — chosen over ECS Fargate for simplicity; sufficient for single-container inference workload. **S3** — stores model `.pth`, all 4 GA JSON logs, and the dataset (downloaded once, never re-fetched). **CloudWatch** — collects custom metrics emitted by the FastAPI app (namespace: `TrafficSignAPI`); also collects memory via the unified CloudWatch agent installed on EC2. **ECR** — stores Docker images tagged by git SHA for full traceability. **IAM** — EC2 instance role scoped to S3 `GetObject` on `models/` prefix only; no hardcoded credentials anywhere. | | |
| **Deployment / containers** | Is there a clear deployment and packaging approach? | `Dockerfile` uses Python 3.10 slim base image. `docker-compose.yml` exposes port 8000 and passes model path via env var (works locally without AWS dependency). GitHub Actions `deploy.yml` has 4 mandatory stages: build → test (runs `test_api.py` against the local container) → push to ECR → deploy to EC2 via SSH (`docker pull` + restart). Pipeline fails and blocks merge if any stage fails. Images are tagged `<ecr-repo>:<git-sha>` so every deployed version is traceable to a commit. | | |
| **Networking / security** | Is exposure and protection in the cloud understood? | The EC2 security group allows inbound HTTP on port 8000 (restricted to demo period). The `/predict` endpoint requires an `X-API-Key` header — requests without a valid key return HTTP 401. IAM roles follow least-privilege: the EC2 instance role has S3 `GetObject` access on `models/` only (not write, not full bucket). No credentials are in source code, Dockerfiles, or committed `.env` files — all passed as GitHub Actions secrets (9 secrets configured: `AWS_ACCOUNT_ID`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `EC2_SSH_KEY`, `EC2_HOST`, `EC2_USER`, `MODEL_PATH`, `API_KEY`, `S3_BUCKET`). | | |
| **Storage / data handling** | Are storage and data choices sensible? | S3 bucket (`traffic-sign-ai-<name>`, `us-east-1`, all public access blocked) has three prefixes: `models/` for `.pth` files, `ea-experiments/` for the 4 GA JSON logs, and `dataset/` for the preprocessed GTSRB splits. The dashboard reads experiment logs directly from S3 at load time (no live database dependency). Uploaded inference images are processed in memory and immediately discarded — no user data is persisted. Dataset is stored in S3 once and never re-downloaded during training or inference (FR-D-07). | | |
| **Scalability / reliability** | Has the team considered scale and availability? | The Docker container is configured with `restart: always` (docker-compose) so it restarts automatically on crash. EA experiment results are persisted to S3 before the process exits (no data loss on crash). For demo purposes, a single EC2 t2.medium is sufficient — the system is not designed for horizontal scaling beyond the demo, which is an acknowledged constraint. The model loads from S3 at startup (cold-start is mitigated by keeping the container warm). EC2 will be terminated after the final demo to avoid unnecessary cost. | | |
| **Monitoring / observability** | Is there a credible logging / monitoring plan? | The FastAPI app emits custom CloudWatch metrics under the namespace `TrafficSignAPI`: request count, p50/p95 response latency, HTTP error rate, and container memory usage (via the CloudWatch unified agent on EC2). A CloudWatch alarm triggers when the error rate exceeds 1% over any 5-minute window. Billing alerts are set at $10 and $20. The CloudWatch dashboard is demonstrated live during the demo (Step 10 of the demo scenario). | | |
| **Cost awareness** | Is there awareness of cost and free-tier feasibility? | AWS spend is capped at $20 via billing alerts. EC2 t2.medium (~$0.046/hr) is used only during the demo period and terminated immediately after. S3 storage costs are negligible (<1 GB of model + logs). CloudWatch custom metrics: first 10 metrics are free-tier; we use 4 custom metrics. Total estimated demo cost: <$5 for a 2–3 day demo window. | | |
| **Total Cloud Score / 32** | | | | |

---

## Supplementary: Three-Way Model Comparison (for discussion reference)

| Model | Description | Expected Accuracy | Features Used |
|---|---|---|---|
| **CV Baseline** | SIFT features + Naive Bayes (classical pipeline) | ~60–70% | Hand-crafted SIFT descriptors |
| **MobileNetV2 (Full)** | Transfer learning on ImageNet, fine-tuned on GTSRB | ≥90% | All penultimate-layer features (1280-dim) |
| **GA-Optimized** | MobileNetV2 features filtered by GA-selected subset | ≥88% (target) | ≤70% of original features (≥30% reduction) |

---

## Supplementary: GA Experiment Configurations (for discussion reference)

| Run ID | Selection | Crossover | Expected Best Fitness |
|---|---|---|---|
| `run_tournament_singlepoint` | Tournament (k=3) | Single-Point | ≥88% |
| `run_tournament_uniform` | Tournament (k=3) | Uniform | ≥88% |
| `run_roulette_singlepoint` | Roulette Wheel | Single-Point | ≥88% |
| `run_roulette_uniform` | Roulette Wheel | Uniform | ≥88% |

GA parameters: population = 50, max generations = 100, early stop = 20 generations no improvement, mutation rate = 0.01/bit, fitness sharing for diversity preservation.

---

## Supplementary: Demo Walkthrough (10-Step Scenario)

| Step | Action | Expected Output |
|---|---|---|
| 1 | Examiner opens the live AWS-hosted URL | Dashboard homepage loads with project title, model accuracy summary, and navigation tabs |
| 2 | Examiner navigates to the **Live Prediction** tab | Image upload form appears |
| 3 | Examiner uploads a photo of a speed limit sign (30 km/h) | API calls `/predict`, model processes the image |
| 4 | System returns prediction result | "Speed limit (30km/h)" · Confidence: ~97% · Top-3: [30km/h, 50km/h, 80km/h] |
| 5 | Examiner uploads a partially occluded sign | Lower confidence + top-3 alternatives — demonstrating honest uncertainty |
| 6 | Examiner navigates to **EA Experiments** tab | GA convergence curves: fitness vs generation for 4 configurations |
| 7 | Examiner views the **Feature Reduction Table** | Table: full-feature accuracy (~92%), GA-selected accuracy (~90%), features used (~40% of original) |
| 8 | Examiner navigates to **Model Evaluation** tab | 43×43 confusion matrix heatmap, per-class accuracy, top-5 most confused sign pairs |
| 9 | Examiner opens Swagger UI (`/docs`) | Live API docs — examiner sends a raw POST directly from the browser |
| 10 | Presenter shows CloudWatch dashboard | Real-time API latency graph, request count, zero errors over the demo session |

---

## Supplementary: Risk and Mitigation Summary

| Risk | Severity | Mitigation |
|---|---|---|
| Class imbalance (some classes 10× fewer samples) | High | `WeightedRandomSampler` in PyTorch DataLoader |
| GA convergence time (fitness = train classifier per generation) | High | k-NN/linear SVM for fitness (not full DL model); PCA-reduced feature vectors |
| AWS cost overrun | Medium | Billing alerts at $10 and $20; EC2 terminated after demo |
| Integration failure between AML model and API | Medium | `.pth` export format and preprocessing pipeline agreed in Week 1; local Docker test before AWS push |
| EA experiments slow (4 runs × 100 generations) | Medium | Pre-compute feature vectors offline; GA only runs the classifier, not the full model |

---

*Document generated from: `docs/cloud_proposal.md`, `docs/srs.md`, `docs/projects.md`*
*Last updated: 2026-05-12*
