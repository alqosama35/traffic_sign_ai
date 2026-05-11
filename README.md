# Traffic Sign AI — Traffic Sign Intelligence Platform

A course project platform for traffic sign classification and analytics using the **GTSRB (43 classes)** dataset.

This repo contains:
- **Production inference model:** **MobileNetV2 (PyTorch)** served via **FastAPI**
- **Dashboard:** static HTML/JS served by a FastAPI “dashboard service”
- **CV baseline:** pre-computed metrics (SIFT + Naive Bayes) exposed as a small FastAPI service
- **EA/GA:** pre-computed GA experiment logs exposed as a small FastAPI service
- **Cloud hooks:** optional **S3** artifact loading and **CloudWatch** latency metrics
- **Audit logging (microservices):** async prediction audit events via **RabbitMQ** → JSONL logs

> Note: The deployed/served model is MobileNetV2. Any other architecture is out of scope unless you explicitly add it.

---

## Repository layout

- `api/` — **monolith** FastAPI app (inference + dashboard router)
- `services/` — **microservices** implementation (gateway + inference + dashboard + cv + ga + audit worker)
- `dashboard/` — static dashboard site (HTML/CSS/JS)
- `training/outputs/` — model + report + plots used by API/dashboard
  - `training/outputs/model/mobilenetv2_inference.pth`
  - `training/outputs/logs/report.json`
  - `training/outputs/plots/cm.png`
- `ea/results/` — GA experiment JSON logs (consumed by GA service / dashboard)
- `dataset/` — processed dataset layout (`train/`, `val/`, `test/`) created by `setup_dataset.py`
- `test_api.py` — automated API test script (course deliverable)

---

## Quickstart (recommended): microservices with Docker Compose

This runs:
- `gateway` (nginx) on your host
- `inference-svc` (FastAPI + MobileNetV2) on internal port `8001`
- `cv-svc` on internal port `8002`
- `ga-svc` on internal port `8003`
- `dashboard-svc` on internal port `8004`
- `rabbitmq` + `audit-worker` for async audit logging

### 1) Configure environment

Copy `.env.example` → `.env` and edit secrets as needed:

```bash
# Windows PowerShell
Copy-Item .env.example .env
```

Key variables:
- `API_KEY`: if empty, auth is disabled (ok for local dev)
- `INTERNAL_API_KEY`: used by `dashboard-svc` to call `inference-svc` without exposing the public key
- `MODEL_TEST_MODE=1`: use a stub model (CI/dev) without needing a `.pth`
- `DATA_SOURCE=s3` + `S3_BUCKET=...`: load dashboard artifacts / GA logs / CV metrics from S3

### 2) Build + run

```bash
docker compose up --build
```

Local dev uses `docker-compose.override.yml`, so the gateway listens on:
- http://localhost:8080

Useful URLs:
- API docs (inference via gateway): http://localhost:8080/docs
- Dashboard (static + data API): http://localhost:8080/dashboard/
- Aggregate health check: http://localhost:8080/health
- RabbitMQ UI (dev only): http://localhost:15672 (default user `guest`)

### 3) Try a prediction

```bash
curl -X POST "http://localhost:8080/predict" \
  -F "file=@path/to/image.jpg"
```

If you configured `API_KEY`, add the header:

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@path/to/image.jpg"
```

---

## Local dev (no Docker)

The repo’s `requirements.txt` is kept intentionally minimal; **PyTorch and some service-only deps are installed in Dockerfiles** for better layer caching.

### Option A: Run the monolith API

The monolith app is `api/app.py`:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Install PyTorch (CPU) explicitly for local runs
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.0+cpu torchvision==0.18.0+cpu

# Uses training/outputs/model/mobilenetv2_inference.pth by default
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Open:
- http://localhost:8000/docs
- http://localhost:8000/health

### Option B: Run the inference microservice directly

```bash
# same venv setup as above
pip install -r requirements.txt

# Extra deps used by services/*
pip install pika==1.3.2
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.0+cpu torchvision==0.18.0+cpu

uvicorn services.inference.main:app --reload --host 0.0.0.0 --port 8001
```

### Option C: Run the monolith container (fallback)

There is also a single-container “monolith” Dockerfile at the repo root (named `dockerfile`):

```bash
docker build -f dockerfile -t traffic-sign-monolith .
docker run --rm -p 8000:8000 traffic-sign-monolith
```

---

## API

### Endpoints (inference)

- `GET /` — basic metadata
- `GET /health` → `{ "status": "ok" }`
- `POST /predict` — multipart upload field name: `file`

`POST /predict` response schema:

```json
{
  "class": "Speed limit (30km/h)",
  "category": "prohibition",
  "confidence": 0.987654,
  "top_3": [
    {"class": "...", "confidence": 0.9},
    {"class": "...", "confidence": 0.05},
    {"class": "...", "confidence": 0.02}
  ],
  "note": "Screening tool only — not for use in safety-critical autonomous systems"
}
```

### Auth behavior

- If `API_KEY` is **empty or unset**, auth is **disabled**.
- If `API_KEY` is **set**, `/predict` requires `X-API-Key`.
- In the microservices stack, `inference-svc` also accepts `INTERNAL_API_KEY` for internal calls.

---

## Dashboard

### Gateway paths

When running Compose, nginx routes:
- `/dashboard/` → dashboard static site + dashboard data API
- `/cv/` → CV baseline service
- `/ga/` → GA results service

### Dashboard data endpoints (via gateway)

- `GET /dashboard/data/training-report`
- `GET /dashboard/data/confusion-matrix.png`
- `GET /dashboard/data/cv-metrics`
- `GET /dashboard/data/ea`
- `GET /dashboard/data/recent-predictions?limit=20`
- `POST /dashboard/predict` (relay used by the browser UI)

---

## Dataset setup

If you need to rebuild the processed dataset splits, run:

```bash
python setup_dataset.py
```

What it does:
- Downloads GTSRB via `kagglehub` into `data/`
- Creates `dataset/train`, `dataset/val`, `dataset/test` with 224×224 resized images
- Writes `dataset/annotations.csv` and `dataset/test_labeled.csv`

Important notes:
- `setup_dataset.py` deletes existing `data/` and `dataset/` folders on re-run.
- Kaggle access may require credentials (either `~/.kaggle/kaggle.json` or env vars such as `KAGGLE_USERNAME`/`KAGGLE_KEY`).

---

## Tests

### `test_api.py` (automated API test)

By default the script targets `http://localhost:8000`:

```bash
python test_api.py
```

To point it at a different base URL:

```bash
# PowerShell
$env:API_BASE_URL = "http://localhost:8000"
$env:API_KEY = ""  # or set if your server enforces auth
python test_api.py
```

Notes:
- `test_api.py` is designed to run against an inference-style API whose `GET /health` returns exactly `{ "status": "ok" }`.
- If you run the full microservices stack behind the gateway, prefer testing the **inference service directly** (or the monolith) rather than the gateway health endpoint.

---

## CV baseline pipeline (optional)

`run_pipeline.py` contains a classical CV pipeline (SIFT + BoVW + Naive Bayes) for experimentation.

```bash
python run_pipeline.py
```

It expects the raw Kaggle layout under `data/`.

---

## Production notes (AWS)

- Set `ENABLE_CLOUDWATCH=1` to emit latency/request/error metrics to CloudWatch under namespace `TrafficSignAPI`.
- To load artifacts from S3, set `DATA_SOURCE=s3`, `S3_BUCKET=...`, and `AWS_REGION=...`.
- The inference service supports downloading a model at startup when `MODEL_PATH` starts with `s3://...`.

---

## License

Course project / educational use.
