# Cloud Computing Spring 2026
# Standard Project Idea Proposal — Part I (Student Submission)

---

| Field | Value |
|---|---|
| **Team name / ID** | *(fill in team name / ID)* |
| **Team type** | AI-only |
| **Course mapping** | Cloud Computing + Advanced ML + Evolutionary Algorithms + Computer Vision |
| **Proposed project title** | Traffic Sign Intelligence Platform |
| **Team members and IDs** | *(fill in names and IDs)* |

---

## 1. Problem Statement and Target Users

**What problem are you solving?**

Traffic sign recognition is a safety-critical task in autonomous driving and driver-assistance systems. Three distinct technical problems exist side-by-side: (1) identifying *which* visual features are most discriminative for traffic sign classification — a feature selection problem suited to evolutionary search; (2) building a classical computer vision baseline using hand-crafted feature detectors; and (3) training a high-accuracy deep learning model using transfer learning. A fourth challenge ties them all together: deploying the resulting system reliably and reproducibly on cloud infrastructure so that predictions are served at scale and all experimental results are stored, monitored, and reproducible.

No single course solves all four layers. This project bridges them into one platform.

**Who are the target users or stakeholders?**

- Primary: researchers and students benchmarking traffic sign recognition pipelines
- Secondary: autonomous driving engineers who need a tested, cloud-deployed inference API for sign classification
- Tertiary: course supervisors validating that all four academic contributions are measurable and non-overlapping

**Why is this problem worth solving in this semester project?**

The GTSRB (German Traffic Sign Recognition Benchmark) is one of the most studied real-world computer vision datasets and maps directly to two already-chosen course projects (CV: Traffic Sign Detection, EA: Feature Selection in Image Analysis). Extending those projects into a cloud-deployed, EA-optimized deep learning platform produces a complete, end-to-end system with high demonstration value, clear evaluation metrics, and a coherent academic narrative across all four courses in one semester.

---

## 2. Project Summary in Plain Language

The Traffic Sign Intelligence Platform is a cloud-deployed AI service that classifies traffic sign images into 43 categories using transfer learning on the GTSRB dataset. A user uploads a photo of a traffic sign and receives an instant prediction with a confidence score and top-3 alternatives through a REST API. Behind the scenes, a Genetic Algorithm runs feature selection experiments over extracted visual features to determine which feature subsets produce the highest classification accuracy, and the results of these experiments are stored and visualised on a results dashboard. The entire system is containerized with Docker and hosted on AWS, with model files stored in S3 and API performance monitored via CloudWatch. The platform produces three comparable outputs: a classical CV baseline (SIFT + Naive Bayes), a full-feature deep learning model, and a GA-optimized feature-subset model — giving a clear, measurable comparison across all three approaches.

---

## 3. Core Features / Modules

| # | Module | Description | MVP? |
|---|---|---|---|
| 1 | **Data Pipeline** | GTSRB download, resize to 224×224, train/val/test split, class balance analysis | ✅ Yes |
| 2 | **Augmentation Pipeline** | Brightness/contrast jitter, rotation, horizontal flip to handle sun/shadow variance | ✅ Yes |
| 3 | **Transfer Learning Trainer** | MobileNetV2 or EfficientNet-B0 fine-tuned on GTSRB; exports `traffic_sign_model.pth` | ✅ Yes |
| 4 | **GA Feature Selection Engine** | Genetic Algorithm applied to extracted feature vectors; compares GA-selected vs full-feature accuracy | ✅ Yes |
| 5 | **FastAPI Inference Service** | `/predict` endpoint accepting image uploads; returns class, confidence, top-3; `/health` endpoint | ✅ Yes |
| 6 | **Dockerization** | `Dockerfile` + `docker-compose.yml` packaging the FastAPI app and model | ✅ Yes |
| 7 | **AWS Deployment** | EC2 or ECS Fargate hosting; S3 for model and experiment storage; CloudWatch for monitoring | ✅ Yes |
| 8 | **Results Dashboard** | Web page showing GA convergence curves, feature reduction stats, confusion matrix | ✅ Yes |
| 9 | **Classical CV Baseline** | SIFT + Naive Bayes pipeline from the CV course project (comparison baseline) | ✅ Yes |
| 10 | **Automated Test Script** | `test_api.py` sending sample images and validating responses | ✅ Yes |
| 11 | Real-time webcam inference | Live camera feed fed to the API | Stretch |
| 12 | Multi-GA-variant comparison | Compare GA, DE, and PSO on the same feature selection task | Stretch |
| 13 | Auto-retraining pipeline | Trigger model retraining when new labelled data is uploaded to S3 | Stretch |

---

## 4. Course Mapping and Academic Contribution

| Course | Contribution to This Project |
|---|---|
| **Advanced ML (AML)** | Transfer learning setup (MobileNetV2 / EfficientNet-B0), training loop, loss/accuracy curves, evaluation (confusion matrix, per-class F1, top-5 accuracy), FastAPI deployment template, Docker containerization |
| **Evolutionary Algorithms (EA)** | GA implementation for feature selection: chromosome = binary feature mask over extracted feature vectors, fitness function = validation classification accuracy, operators = tournament selection + single-point crossover + bit-flip mutation. At least 2 selection strategies and 2 crossover operators compared as required by EA course guidelines. Diversity preservation via fitness sharing. |
| **Computer Vision (CV)** | Classical baseline pipeline: image preprocessing (Gaussian/Median filters), Harris Corner Detector, SIFT feature extraction and matching, image segmentation (K-means or watershed), Naive Bayes classifier. This baseline provides the comparison point against the deep learning model. |
| **Cloud Computing** | AWS deployment (EC2/ECS), S3 model and experiment artifact storage, CloudWatch monitoring and logging, Docker containerization, IAM role-based access control, API security. Cloud contribution is architectural — not decorative — because it enables reproducible storage of all EA experiment runs, scalable inference serving, and observability. |

**Academic non-overlap guarantee:** EA produces a feature selector. CV produces a hand-crafted baseline. AML produces a deep learning model. Cloud deploys, monitors, and stores everything. No two courses repeat the same technique.

---

## 5. Technical Approach and Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch 2.x + TorchVision (transfer learning) |
| API Framework | FastAPI + Uvicorn |
| Image Processing | Pillow, OpenCV (CV baseline), NumPy |
| EA Implementation | Custom Python GA + NumPy (no black-box library — course requirement) |
| Metrics | scikit-learn (confusion matrix, classification report, ROC-AUC) |
| Visualization | Matplotlib, Seaborn |
| Containerization | Docker, docker-compose |
| Cloud Provider | AWS (EC2 or ECS Fargate, S3, CloudWatch, IAM) |
| CI / Deployment | Manual Docker push + AWS CLI (or GitHub Actions if time allows) |
| Dashboard | Lightweight HTML/JS served by FastAPI static files, or Streamlit |

**Architecture style:** Modular — three independently runnable services:
1. `trainer/` — offline training and EA experiment scripts
2. `api/` — FastAPI inference service (the only cloud-facing runtime service)
3. `dashboard/` — results viewer reading from S3 experiment logs

This separation ensures the cloud deployment is lean (only the inference API runs 24/7) while all training and EA experiments run offline or in batch jobs.

---

## 6. Data, Inputs, and Test Scenarios

**Dataset:** GTSRB — German Traffic Sign Recognition Benchmark
- **Source:** Kaggle (`meowmeowmeowmeowmeow/gtsrb-german-traffic-sign-recognition-benchmark`) — publicly available, no license restrictions for academic use
- **Size:** 50,000+ images across 43 classes (speed limits, stop, yield, no entry, warning signs, etc.)
- **Format:** JPEG images, variable sizes (15×15 to 250×250 px) — resized to 224×224 for transfer learning
- **Split:** Pre-existing train/test split; we define a validation subset (15%) from train
- **Privacy:** No personal data. All images are photos of road signs in public spaces.

**Data limitations:**
- Severe class imbalance: some signs have ~200 images, others ~2,000 → handled with `WeightedRandomSampler`
- Images captured under varying lighting (sun, shadow, rain) → handled with brightness/contrast augmentation
- No license plate or face data — no GDPR concern

**Test scenarios:**
- Standard GTSRB test set (official benchmark)
- Real-world photos taken by team members (phone camera) — informal robustness check
- Edge cases: partially occluded signs, night-time photos, non-German sign variants

---

## 7. AI / Model / Algorithm Plan

### AML Component — Transfer Learning

| Item | Detail |
|---|---|
| Base model | MobileNetV2 (primary) and EfficientNet-B0 (comparison) — both pre-trained on ImageNet |
| Strategy | Freeze backbone for first 5 epochs, then unfreeze last 2 blocks for fine-tuning |
| Loss function | CrossEntropyLoss with class weights for imbalance |
| Sampler | WeightedRandomSampler |
| LR schedule | StepLR or CosineAnnealingLR |
| Export | `traffic_sign_model.pth` |
| Target | >90% overall accuracy on GTSRB test set |

### EA Component — GA Feature Selection

| Item | Detail |
|---|---|
| **Representation** | Binary chromosome of length *n* where each bit = include/exclude one feature dimension from the penultimate layer feature vector of the trained AML model (or HOG/SIFT features for classical mode) |
| **Fitness function** | Validation set classification accuracy of a lightweight classifier (k-NN or linear SVM) trained on the selected feature subset |
| **Population** | 50 individuals, initialized randomly |
| **Selection** | Tournament selection (k=3) — also tested: Roulette Wheel selection (comparison run as per EA course requirement) |
| **Crossover** | Single-point crossover — also tested: Uniform crossover (comparison run) |
| **Mutation** | Bit-flip mutation (rate 0.01 per bit) |
| **Termination** | 100 generations or no improvement for 20 consecutive generations |
| **Diversity** | Fitness sharing within niches to prevent premature convergence (EA course requirement) |
| **Experiments** | At least 4 runs: 2 selection strategies × 2 crossover operators |

### Three-Way Comparison

| Model | Description | Expected Accuracy |
|---|---|---|
| CV Baseline | SIFT features + Naive Bayes (from CV course) | ~60–70% |
| Full DL Model | MobileNetV2 transfer learning on all features | >90% |
| GA-Optimized Model | Transfer learning features filtered by GA-selected subset | >88% with fewer features |

---

## 8. Cloud and Deployment Plan

**Why the cloud contribution is meaningful, not decorative:**

The cloud layer serves three substantive purposes:
1. **Reproducible experiment storage** — every GA run (fitness curves, best chromosome, selected feature indices, accuracy results) is uploaded to S3 as a structured JSON log. This means experiments can be re-examined, compared, and audited without relying on a local machine.
2. **Scalable inference serving** — the FastAPI app is deployed on AWS EC2 (or ECS Fargate) so the `/predict` endpoint is publicly accessible during the demo and beyond. Model cold-start is avoided by keeping the container warm.
3. **Observability** — CloudWatch monitors API request latency, error rate, and memory usage. Alerts are configured for >1% error rate.

| Cloud Component | Service | Purpose |
|---|---|---|
| Inference API hosting | AWS EC2 t2.medium or ECS Fargate | Serve FastAPI /predict endpoint |
| Model storage | AWS S3 | Store `traffic_sign_model.pth` and versioned checkpoints |
| Experiment logs | AWS S3 | Store all GA run outputs (JSON) for dashboard access |
| Monitoring & logging | AWS CloudWatch | Track API latency, error rate, memory; alert on anomalies |
| Access control | AWS IAM roles | Least-privilege S3 read access for the inference container |
| Container registry | AWS ECR or Docker Hub | Store and pull the built Docker image |

**Security considerations:**
- API key header required for `/predict` (simple token auth)
- IAM roles scoped to S3 read-only for inference service
- No user data stored — images are processed in memory and discarded

---

## 9. Evaluation Plan

### Model Evaluation (AML layer)

| Metric | Target | Comparison |
|---|---|---|
| Overall test accuracy | >90% | MobileNetV2 vs EfficientNet-B0 |
| Per-class accuracy | Report all 43 classes | Identify top-5 most confused sign pairs |
| Precision / Recall / F1 | Full classification report | DL model vs CV baseline vs GA-optimized |
| Top-5 accuracy | >98% | Standard benchmark metric |
| Confusion matrix | 43×43 heatmap | Visualize most common misclassifications |

### EA Evaluation (GA layer)

| Metric | Target |
|---|---|
| GA convergence curve | Fitness (accuracy) vs generation number — must show clear convergence |
| Feature reduction ratio | GA-selected feature count / total features — target >30% reduction |
| Accuracy vs reduction tradeoff | GA model accuracy loss must be <3% vs full-feature model |
| Operator comparison | Table comparing all 4 experiment configurations (2 selection × 2 crossover) |

### Cloud Evaluation (Cloud layer)

| Metric | Target |
|---|---|
| API response time | <500ms per inference request |
| Uptime during demo | 100% |
| CloudWatch error rate | <1% over 100 test requests |

---

## 10. Scope Control and Minimum Viable Version

### Minimum Viable Version (must work at final demo)

1. GTSRB dataset downloaded, split, and preprocessed
2. MobileNetV2 model trained to >90% accuracy and exported as `.pth`
3. GA feature selection experiment run with at least 2 operator configurations
4. FastAPI `/predict` endpoint working and returning class + confidence
5. Docker container building and running locally
6. AWS deployment live — endpoint publicly accessible
7. S3 storing the model and at least one set of GA experiment logs
8. CloudWatch showing at least API latency metrics
9. Results dashboard showing GA convergence curve and confusion matrix
10. `test_api.py` script passing all assertions

### Stretch Features (if time permits)

- Real-time webcam inference demo
- EfficientNet-B0 vs MobileNetV2 comparison results on dashboard
- Multiple GA variant comparison (GA vs DE vs PSO)
- Automated S3-triggered retraining pipeline

### Cut first if time is tight

1. Multi-architecture model comparison (keep MobileNetV2 only)
2. Extended dashboard — reduce to static HTML with embedded charts
3. Multiple GA variants — keep only one configuration comparison
4. Real-time webcam feed

---

## 11. Team Role Split

| Member | Role | Main Responsibilities |
|---|---|---|
| Member 1 | **Data Manager** | Download GTSRB, organize train/val/test (70/15/15), analyze class balance, document data sources |
| Member 2 | **EDA & Visualizer** | Sample grids, class distribution charts, 8+ visualizations for report |
| Member 3 | **Augmentation Designer & Model Trainer** | Build augmentation pipeline (brightness, contrast, rotation), MobileNetV2 transfer learning, training loop, loss/accuracy curves, export `.pth` |
| Member 4 | **EA Researcher** | GA implementation (chromosome, fitness, selection, crossover, mutation, diversity), 4 experiment configurations, convergence plots |
| Member 5 | **API Developer** | FastAPI `/predict` + `/health` endpoints, preprocessing pipeline, Swagger docs, `test_api.py` |
| Member 6 | **Cloud Deployer** | Dockerfile, AWS EC2/ECS setup, S3 bucket + IAM roles, CloudWatch alarms, results dashboard, demo slides, presentation |

**Integration points:**
- Member 3 (Trainer) exports `.pth` → used by Member 4 (EA, feature extraction) and Member 5 (API)
- Member 4 (EA) exports experiment JSON logs → stored by Member 6 (Cloud, S3) and displayed in dashboard
- Member 5 (API) provides Docker-ready app → handed to Member 6 for AWS deployment
- Member 2 (EDA) provides visualizations → used in both Member 6's dashboard and the final report

---

### Alternative Role Split — 7 Members

| Member | Role | Main Responsibilities |
|---|---|---|
| Member 1 | **Data Manager** | Download GTSRB, organize train/val/test (70/15/15), analyze class balance, document data sources |
| Member 2 | **EDA & Visualizer** | Sample grids, class distribution charts, 8+ visualizations for report |
| Member 3 | **Augmentation Designer** | Build augmentation pipeline (brightness, contrast, rotation), compare augmented vs non-augmented model performance |
| Member 4 | **Model Trainer** | MobileNetV2 transfer learning, training loop, loss/accuracy curves, export `.pth` |
| Member 5 | **EA Researcher** | GA implementation (chromosome, fitness, selection, crossover, mutation, diversity), 4 experiment configurations, convergence plots |
| Member 6 | **API Developer** | FastAPI `/predict` + `/health` endpoints, preprocessing pipeline, Swagger docs, `test_api.py` |
| Member 7 | **Cloud Deployer** | Dockerfile, AWS EC2/ECS setup, S3 bucket + IAM roles, CloudWatch alarms, results dashboard, demo slides, presentation |

**Integration points:**
- Member 4 (Trainer) exports `.pth` → used by Member 5 (EA, feature extraction) and Member 6 (API)
- Member 5 (EA) exports experiment JSON logs → stored by Member 7 (Cloud, S3) and displayed in dashboard
- Member 6 (API) provides Docker-ready app → handed to Member 7 for AWS deployment
- Member 2 (EDA) provides visualizations → used in both Member 7's dashboard and the final report

---

## 12. Demo Scenario

| Step | Action | Expected Output |
|---|---|---|
| 1 | Examiner opens the live AWS-hosted URL in a browser | Dashboard homepage loads showing project title, model accuracy summary, and navigation tabs |
| 2 | Examiner navigates to the **Live Prediction** tab | Image upload form appears |
| 3 | Examiner uploads a photo of a speed limit sign (30 km/h) | API calls `/predict`, model processes the image |
| 4 | System returns prediction result | "Speed limit (30km/h)" · Confidence: 97% · Top-3: [30km/h, 50km/h, 80km/h] |
| 5 | Examiner uploads an ambiguous or partially occluded sign | System returns lower confidence + top-3 alternatives — demonstrating honest uncertainty |
| 6 | Examiner navigates to the **EA Experiments** tab | GA convergence curves appear: fitness (validation accuracy) vs generation number for 4 configurations |
| 7 | Examiner views the **Feature Reduction Table** | Table shows: full-feature accuracy (92%), GA-selected accuracy (90%), features used (40% of original) |
| 8 | Examiner navigates to the **Model Evaluation** tab | 43×43 confusion matrix heatmap, per-class accuracy bar chart, top-5 most confused sign pairs |
| 9 | Examiner opens Swagger UI (`/docs`) | Live API documentation — examiner sends a raw POST request directly from the browser |
| 10 | Presenter shows CloudWatch dashboard | Real-time API latency graph, request count, zero errors over the demo session |

---

## 13. Risks, Ethics, and Constraints

| Risk | Severity | Mitigation |
|---|---|---|
| **Class imbalance** (some classes have 10× fewer samples) | High | Use `WeightedRandomSampler` in PyTorch DataLoader; monitor per-class recall during training |
| **GA convergence time** (fitness evaluation = training a classifier per generation) | High | Use a lightweight k-NN or linear SVM for fitness evaluation (not the full deep model); limit chromosome length by using PCA-reduced feature vectors |
| **AWS cost overrun** | Medium | Use free-tier eligible services (EC2 t2.micro for demo, S3 standard tier); set AWS billing alerts at $10 and $20; tear down EC2 after demo |
| **Integration failure between AML model and API** | Medium | Define the `.pth` export format and preprocessing pipeline in Week 1; Member 6 writes a local Docker test before pushing to AWS |
| **Timeline risk: EA experiments are slow** | Medium | Start GA experiments in Week 2 in parallel with model training; pre-compute feature vectors offline so GA only runs the classifier, not the full model |
| **GTSRB dataset size on Kaggle** | Low | 50K images (~600MB) — download once, store in S3, never re-download |
| **Image quality of real-world test photos** | Low | Note in report that model was trained on GTSRB (German signs); performance on other country signs may differ — include as a limitation |

**Ethical and legal considerations:**
- Dataset is fully public and has no privacy concerns (no faces, no license plates, only road sign photos)
- The system is a classification aid, not a safety-critical autonomous decision maker — include an appropriate disclaimer in the API response: `"Screening tool only — not for use in safety-critical autonomous systems"`
- No personal user data is stored or transmitted; uploaded images are processed in memory and immediately discarded

---

## Student Submission Package Checklist

- [x] Completed Part I in full
- [x] Proposed project title and clear problem statement
- [x] Course mapping explained (4 courses, non-overlapping contributions)
- [x] Dataset / input source identified (GTSRB, Kaggle, public)
- [x] Evaluation plan with metrics (accuracy, F1, GA convergence, API latency)
- [x] MVP scope clearly stated
- [x] Team role split included (7 members, clear responsibilities)
- [x] Demo scenario included (10-step end-to-end walkthrough)

---

# Part II — Supervisor Scoring Checklist

*(To be completed by the supervisor after reviewing Part I)*

## A. Approval Gate Checklist

| Approval condition | Yes | No | Notes / conditions |
|---|---|---|---|
| Problem is clearly defined and user value is understandable. | [ ] | [ ] | |
| Scope is realistic for the timeline and team size. | [ ] | [ ] | |
| Project has enough technical depth for the team type. | [ ] | [ ] | |
| Inputs / dataset / scenarios are available and usable. | [ ] | [ ] | |
| Evaluation metrics and comparison plan are clear. | [ ] | [ ] | |
| Work can be divided fairly across team members. | [ ] | [ ] | |
| The project supports a stable end-to-end demo. | [ ] | [ ] | |
| Course mapping is natural and academically defensible. | [ ] | [ ] | |
| Documentation requirements can be satisfied. | [ ] | [ ] | |
| The proposal avoids obvious ethical, legal, or privacy problems. | [ ] | [ ] | |
| Cloud contribution is meaningful and not merely decorative. | [ ] | [ ] | |
| The team has a clear minimum viable version and fallback plan. | [ ] | [ ] | |

## B. Suggested Scoring Rubric (100 Marks)

| Criterion | Weight | Score | Supervisor notes |
|---|---|---|---|
| Problem definition and scope | 10 | | |
| Technical depth and course alignment | 15 | | |
| System or model design quality | 15 | | |
| Data / inputs / preprocessing quality | 10 | | |
| Evaluation plan and measurable success criteria | 10 | | |
| Feasibility and timeline realism | 10 | | |
| Team role distribution and fairness | 10 | | |
| Cloud architecture / deployment relevance | 10 | | |
| Documentation readiness | 5 | | |
| Demo readiness | 5 | | |

## C. Team-Type Check — AI-Only

Must include: a defined modeling or optimization problem, a baseline, measurable evaluation, experiments, and analysis.

This proposal includes:
- Three-way comparison (CV baseline vs DL model vs GA-optimized model) ✓
- Defined GA representation, fitness function, and operators ✓
- Measurable evaluation metrics for both the model and the EA component ✓
- Cloud contribution beyond token hosting (S3 experiment storage, CloudWatch monitoring, IAM security) ✓

## D. Final Decision

| | |
|---|---|
| **Decision** | [ ] Approve  [ ] Approve with conditions  [ ] Revise and resubmit  [ ] Reject |
| **Required conditions** | |
| **Most important scope correction** | |
| **Supervisor comments** | |
| **Supervisor name / signature / date** | |
