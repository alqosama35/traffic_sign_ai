# Traffic Sign AI — Project Context

## Source of Truth for Requirements

Requirements come from **two source documents** (in priority order):

1. `docs/projects.md` — course project requirements for EA, CV, and AML courses
2. `docs/cloud_proposal.md` — cloud computing course requirements and overall system design

The `docs/srs.md` and `docs/requirements_validation.md` are **derived documents** — they synthesize the above two sources. If there is a conflict between the SRS/validation and the source documents, the source documents win.

## Project Overview

**Traffic Sign Intelligence Platform** — a cloud-deployed AI system classifying traffic signs (GTSRB dataset, 43 classes) that satisfies 4 courses simultaneously:

| Course | Contribution |
|---|---|
| Advanced ML (AML) | Transfer learning (MobileNetV2), FastAPI inference, Docker |
| Evolutionary Algorithms (EA) | GA-based feature selection on extracted feature vectors |
| Computer Vision (CV) | Classical pipeline: SIFT + Naive Bayes baseline |
| Cloud Computing | AWS EC2/ECS + S3 + CloudWatch deployment |

## Key Architecture Decisions

- **Production model: MobileNetV2** (the only model served by the API and deployed to AWS)
- **EfficientNet-B0: NOT required** — `projects.md` says "MobileNetV2 **or** EfficientNet-B0" (either/or). `cloud_proposal.md` lists EfficientNet-B0 as a stretch feature and first item to cut under "Cut first if time is tight". The SRS over-specified by making it mandatory — this was a documentation error.
- **Three-way comparison**: CV Baseline vs MobileNetV2 vs GA-Optimized (not four-way; EfficientNet-B0 is stretch only)

## MVP Scope (from cloud_proposal.md §10)

1. GTSRB dataset downloaded, split, preprocessed
2. MobileNetV2 trained to >90% accuracy, exported as `.pth`
3. GA feature selection with at least 2 operator configurations
4. FastAPI `/predict` endpoint working (class + confidence + top-3)
5. Docker container building and running locally
6. AWS deployment live and publicly accessible
7. S3 storing model and at least one GA experiment log set
8. CloudWatch showing API latency metrics
9. Results dashboard showing GA convergence curve and confusion matrix
10. `test_api.py` passing all assertions

## What to Cut First (if time is tight)

1. Multi-architecture model comparison — keep MobileNetV2 only
2. Extended dashboard — reduce to static HTML with embedded charts
3. Multiple GA variants — keep only one configuration comparison
4. Real-time webcam feed

## Don't Implement Without Approval

Do not implement, refactor, or add anything beyond answering questions unless the user explicitly approves. Always explain what would be done and wait for confirmation first.
