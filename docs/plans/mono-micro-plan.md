# Microservices Decomposition Plan — Traffic Sign AI

## Context

The project is currently a single FastAPI monolith (`api/app.py` + `api/dashboard_router.py`)
running in one Docker container. The goal is to split it into independently deployable services
communicating over HTTP, with an nginx API gateway as the single external entry point, and
RabbitMQ as a message broker for **async prediction audit logging**: every `/predict` call
publishes a `prediction.completed` event after responding to the user (non-blocking), and a
lightweight `audit-worker` consumes it and writes structured JSONL logs. The dashboard then
reads those logs to display recent predictions — a realistic analytics pipeline that runs
comfortably on a t3.small EC2 instance.

---

## Target Architecture

```
Internet :80
    │
    ▼
┌───────────────────────────────────────────┐
│            nginx  (gateway)               │
│  /predict       → inference-svc:8001      │
│  /health        → dashboard-svc:8004      │
│  /cv/*          → cv-svc:8002             │
│  /ga/*          → ga-svc:8003             │
│  /dashboard/*   → dashboard-svc:8004      │
└──┬──────────┬────────┬──────────┬─────────┘
   │          │        │          │
[8001]     [8002]   [8003]     [8004]
inference   cv-svc   ga-svc  dashboard-svc
   │                                │
   │  publish (BackgroundTask)      │ reads prediction log
   │  prediction.completed          │ for /recent-predictions
   ▼                                │
┌──────────────┐                    │
│  RabbitMQ    │                    │
│  :5672       │                    │
└──────┬───────┘                    │
       │ consume                    │
       ▼                            │
  [audit-worker] ── writes ──→  prediction-logs volume
  appends JSONL log                 (shared named volume)
```

**Internal Docker network**: all services on the `internal` bridge network;
only the gateway exposes a host port. RabbitMQ is also internal-only.

---

## New Directory Structure

```
services/
├── gateway/
│   ├── Dockerfile          # FROM nginx:1.27-alpine
│   └── nginx.conf          # upstream blocks + location routing
├── inference/
│   ├── Dockerfile          # python:3.10-slim + torch CPU + fastapi
│   ├── main.py             # /predict, /health, /  — publishes audit event after each prediction
│   └── model_utils.py      # class maps, _build_model, load_model, preprocess
├── cv/
│   ├── Dockerfile          # python:3.10-slim + fastapi (tiny)
│   └── main.py             # /health, /metrics
├── ga/
│   ├── Dockerfile          # python:3.10-slim + fastapi (tiny, read-only)
│   └── main.py             # /health, /experiments
├── audit/
│   ├── Dockerfile          # python:3.10-slim + pika only (ultra-lightweight)
│   └── worker.py           # RabbitMQ consumer — appends JSONL prediction log
└── dashboard/
    ├── Dockerfile          # python:3.10-slim + fastapi + httpx
    └── main.py             # /health (fan-out), /dashboard/data/*, /dashboard/predict
```

Files **kept unchanged**: `api/`, `cv/`, `ea/`, `training/`, `dashboard/` (static),
`test_api.py`, `requirements.txt`, `dockerfile` (monolith fallback).

Files **replaced**: `docker-compose.yml`, `docker-compose.override.yml`,
`.github/workflows/deploy.yml`.

Files **extended** (not replaced): `.env.example`.

---

## Message Broker Design (RabbitMQ)

### Why prediction audit logging

Every `/predict` call already tracks latency for CloudWatch. Adding a RabbitMQ event
after each prediction enables a decoupled audit trail and a live "recent predictions"
dashboard panel — without adding any latency to the user's response. The publish
happens in a FastAPI `BackgroundTask` (fire-and-forget): the user gets their result
immediately, then the event goes to the broker.

This is lightweight enough to run on t3.small and demonstrates real production
patterns: audit trails, analytics pipelines, and event-driven dashboard updates.

### Exchange and Queue Topology

```
Exchange: traffic_sign   (type: topic, durable: true)
  │
  ├── routing key: prediction.completed ──→  Queue: prediction.audit  (durable: true)
  │                                                │
  │                                        x-dead-letter-exchange: traffic_sign_dlx
  │                                        x-dead-letter-routing-key: prediction.dead
  │
  └── (future: cv.requested, ga.completed, ...)

Exchange: traffic_sign_dlx  (type: direct, durable: true)
  └── routing key: prediction.dead ──→  Queue: prediction.dead  (durable: true)
```

**Design decisions and why:**
- **Topic exchange** (not direct/fanout): routing key `<domain>.<event>` lets future
  services subscribe to `prediction.*` without touching existing code.
- **Durable exchange + queues**: survive RabbitMQ restart without losing log events.
- **Persistent messages** (`delivery_mode=2`): survive broker restart.
- **Dead Letter Queue (DLQ)** on `prediction.dead`: if the audit-worker fails to process
  a message, it lands here for inspection instead of being silently dropped.
- **`prefetch_count=10`**: audit-worker can batch-process log writes (not a heavy job),
  so a higher prefetch is fine — improves throughput.
- **Manual ack**: worker acks only after writing to disk. If it crashes mid-write,
  the message is re-queued automatically.
- **Connection retry with backoff**: RabbitMQ takes several seconds to start.
  Both inference-svc (publisher) and audit-worker (consumer) retry with exponential backoff.
- **Non-fatal publish failure**: if RabbitMQ is down, inference-svc logs a warning
  and returns the prediction result to the user anyway. Audit logging must never break inference.

### Shared Log Volume

`audit-worker` and `dashboard-svc` share a named Docker volume (`prediction-logs`)
mounted at `/app/logs/`. audit-worker appends to `predictions.jsonl`; dashboard-svc
reads the last N lines for `GET /dashboard/data/recent-predictions`.

Event schema (one JSON object per line):
```json
{
  "request_id": "uuid",
  "timestamp": 1715000000.123,
  "logged_at": "2025-05-08T12:00:00",
  "predicted_class": "Speed limit 30",
  "category": "prohibition",
  "confidence": 0.987,
  "top_3": [...],
  "latency_ms": 42
}
```

---

## Service Specifications

### 1. Gateway — `services/gateway/`

**nginx.conf** — routing rules:
```nginx
upstream inference  { server inference-svc:8001; }
upstream cv         { server cv-svc:8002; }
upstream ga         { server ga-svc:8003; }
upstream dashboard  { server dashboard-svc:8004; }

server {
    listen 80;
    client_max_body_size 10M;

    location = /predict    { proxy_pass http://inference; proxy_set_header Host $host; proxy_read_timeout 30s; }
    location = /           { proxy_pass http://inference; }
    location /docs         { proxy_pass http://inference; }
    location /openapi.json { proxy_pass http://inference; }
    location /cv/          { proxy_pass http://cv/; }
    location /ga/          { proxy_pass http://ga/; }
    location /dashboard/   { proxy_pass http://dashboard/dashboard/; proxy_set_header Host $host; proxy_read_timeout 30s; }
    location = /health     { proxy_pass http://dashboard/health; }
}
```

`/predict` routes directly to inference-svc — `test_api.py` requires zero changes.

### 2. Inference Service — `services/inference/` (port 8001)

**Source**: extracted from `api/app.py`.

`model_utils.py` — copy in from `api/app.py`:
- Class dicts (`CLASS_NAMES_BY_ID`, `SIGN_DISPLAY_BY_ID`, `CATEGORY_BY_ID`,
  `CLEAN_TO_ID`, `CLEAN_TO_DISPLAY`) — lines 46–100
- `_build_model()`, `load_model()`, `_maybe_download_from_s3()` — extract as-is
- `preprocess` transforms

`main.py` — imports from `model_utils.py`, keeps:
- `POST /predict` — identical auth, inference, CloudWatch, top-3 logic
- `GET /health` — `{"status": "ok"}`
- `GET /` — metadata
- Remove: `app.include_router(dashboard_router)` and static file mount

**`INTERNAL_API_KEY`** env var: inference-svc accepts this in addition to `API_KEY`,
so dashboard-svc's internal relay calls pass auth without exposing the public key.

**New: RabbitMQ audit event publish** — after each successful prediction, fire-and-forget
using FastAPI `BackgroundTasks` so publish latency never reaches the user:

```python
import uuid, time, pika, json
from fastapi import BackgroundTasks

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "")  # empty = disable (dev mode)
EXCHANGE     = "traffic_sign"

def _publish_audit_event(event: dict):
    """Non-fatal fire-and-forget — inference must work even if broker is down."""
    if not RABBITMQ_URL:
        return
    try:
        params = pika.URLParameters(RABBITMQ_URL)
        params.heartbeat = 30
        conn = pika.BlockingConnection(params)
        ch = conn.channel()
        ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
        ch.queue_declare(
            queue="prediction.audit", durable=True,
            arguments={
                "x-dead-letter-exchange": "traffic_sign_dlx",
                "x-dead-letter-routing-key": "prediction.dead",
            }
        )
        ch.queue_bind(queue="prediction.audit", exchange=EXCHANGE,
                      routing_key="prediction.completed")
        ch.basic_publish(
            exchange=EXCHANGE,
            routing_key="prediction.completed",
            body=json.dumps(event),
            properties=pika.BasicProperties(
                delivery_mode=2,                       # persistent
                content_type="application/json",
            ),
        )
        conn.close()
    except Exception as exc:
        print(f"[inference] audit publish failed (non-fatal): {exc}", flush=True)

@app.post("/predict")
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...),
                  x_api_key: str = Header(default="")):
    _check_api_key(x_api_key)
    t0 = time.time()
    # ... existing inference logic (unchanged) ...
    latency_ms = round((time.time() - t0) * 1000, 1)

    background_tasks.add_task(_publish_audit_event, {
        "request_id": str(uuid.uuid4()),
        "timestamp": t0,
        "predicted_class": result["class"],
        "category":        result["category"],
        "confidence":      result["confidence"],
        "top_3":           result["top_3"],
        "latency_ms":      latency_ms,
    })
    return result
```

### 3. CV Service — `services/cv/` (port 8002)

`cv/cv_project.py` is Colab-only and cannot run in a container. This service
returns the pre-computed fixture (`dashboard/data/cv_metrics.json`).

`main.py`:
```python
@app.get("/metrics")
def cv_metrics():
    return json.loads(Path(CV_METRICS_PATH).read_text())
    # CV_METRICS_PATH → /app/dashboard/data/cv_metrics.json
```

Endpoints: `GET /health`, `GET /metrics`

### 4. GA Service — `services/ga/` (port 8003)

Read-only service — lists pre-computed experiment results. No heavy computation,
no RabbitMQ dependency.

`main.py`:
```python
_STRIP = {"best_chromosome", "selected_feature_indices"}  # large arrays

@app.get("/experiments")
def list_experiments():
    runs = []
    for path in sorted(EA_RESULTS_DIR.glob("*.json")):
        run = json.loads(path.read_text())
        for k in _STRIP: run.pop(k, None)
        runs.append(run)
    return runs
```

Endpoints: `GET /health`, `GET /experiments`

### 5. Audit Worker — `services/audit/` (RabbitMQ consumer)

Runs as a standalone container (`audit-worker`). Ultra-lightweight: only `pika`
installed, no torch or heavy dependencies.

`services/audit/worker.py`:
```python
import pika, json, time, os, sys
from pathlib import Path
from datetime import datetime, timezone

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
LOG_FILE     = Path(os.environ.get("AUDIT_LOG_FILE", "/app/logs/predictions.jsonl"))
EXCHANGE     = "traffic_sign"


def _connect_with_retry(max_attempts=10, base_delay=3):
    for attempt in range(max_attempts):
        try:
            params = pika.URLParameters(RABBITMQ_URL)
            params.heartbeat = 60
            params.blocked_connection_timeout = 10
            return pika.BlockingConnection(params)
        except Exception as exc:
            wait = base_delay * (2 ** min(attempt, 4))
            print(f"[audit] RabbitMQ not ready ({exc}), retry in {wait}s …", flush=True)
            time.sleep(wait)
    raise RuntimeError("Could not connect to RabbitMQ after retries")


def callback(ch, method, properties, body):
    try:
        event = json.loads(body)
        event["logged_at"] = datetime.now(timezone.utc).isoformat()
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event) + "\n")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"[audit] {event.get('predicted_class')} ({event.get('confidence', 0):.1%})", flush=True)
    except Exception as exc:
        print(f"[audit] error: {exc}", file=sys.stderr, flush=True)
        # nack without requeue → message goes to DLQ
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)


def main():
    conn = _connect_with_retry()
    ch = conn.channel()

    # Declare topology (idempotent)
    ch.exchange_declare(exchange=EXCHANGE, exchange_type="topic", durable=True)
    ch.exchange_declare(exchange="traffic_sign_dlx", exchange_type="direct", durable=True)
    ch.queue_declare(
        queue="prediction.audit", durable=True,
        arguments={
            "x-dead-letter-exchange": "traffic_sign_dlx",
            "x-dead-letter-routing-key": "prediction.dead",
        }
    )
    ch.queue_bind(queue="prediction.audit", exchange=EXCHANGE,
                  routing_key="prediction.completed")
    ch.queue_declare(queue="prediction.dead", durable=True)
    ch.queue_bind(queue="prediction.dead", exchange="traffic_sign_dlx",
                  routing_key="prediction.dead")

    ch.basic_qos(prefetch_count=10)  # audit writes are cheap, batch them
    ch.basic_consume(queue="prediction.audit", on_message_callback=callback)
    print("[audit] waiting for prediction events …", flush=True)
    ch.start_consuming()


if __name__ == "__main__":
    main()
```

**`services/audit/Dockerfile`**:
```dockerfile
FROM python:3.10-slim
RUN adduser --disabled-password --gecos "" appuser
WORKDIR /app
RUN pip install --no-cache-dir pika==1.3.2
COPY services/audit/ ./services/audit/
RUN chown -R appuser:appuser /app
USER appuser
CMD ["python", "-m", "services.audit.worker"]
```

### 6. Dashboard Service — `services/dashboard/` (port 8004)

Extracted from `api/dashboard_router.py`. Replaces direct file reads with httpx
calls to sibling services. Adds `POST /ga/run` relay and status polling.

`main.py` key patterns:
```python
# Fan-out health check (includes RabbitMQ status via ga-svc)
@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=5.0) as client:
        for name, url in [("inference", INFERENCE_URL), ("cv", CV_URL), ("ga", GA_URL)]:
            try:
                r = await client.get(f"{url}/health")
                results[name] = r.json().get("status", "ok") if r.status_code == 200 else "degraded"
            except Exception:
                results[name] = "unreachable"
    status = "ok" if all(v == "ok" for v in results.values()) else "degraded"
    return JSONResponse({"status": status, "services": results},
                        status_code=200 if status == "ok" else 503)

# EA data via GA service
@app.get("/dashboard/data/ea")
async def ea_data():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{GA_URL}/experiments")
    return r.json()

# CV metrics via CV service
@app.get("/dashboard/data/cv-metrics")
async def cv_metrics():
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{CV_URL}/metrics")
    return r.json()

# Training report and confusion matrix — read from mounted local files
@app.get("/dashboard/data/training-report")
def training_report():
    return json.loads(Path(REPORT_PATH).read_text())

@app.get("/dashboard/data/confusion-matrix.png")
def confusion_matrix():
    return FileResponse(CM_PATH, media_type="image/png")

# Relay predict → inference-svc
@app.post("/dashboard/predict")
async def dashboard_predict(file: UploadFile = File(...)):
    content = await file.read()
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(f"{INFERENCE_URL}/predict",
                              files={"file": (file.filename, content, file.content_type)},
                              headers={"X-API-Key": INTERNAL_API_KEY})
    return r.json()

# Trigger GA experiment (relay to ga-svc which publishes to RabbitMQ)
@app.post("/dashboard/ga/run")
async def trigger_ga(config: str = "tournament_singlepoint_generational_bitflip_42"):
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(f"{GA_URL}/run", params={"config": config})
    return r.json()

# Poll GA job status
@app.get("/dashboard/ga/status/{job_id}")
async def ga_status(job_id: str):
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.get(f"{GA_URL}/status/{job_id}")
    return r.json()
```

**New endpoint — recent predictions** (reads the shared audit log volume):
```python
AUDIT_LOG_FILE = Path(os.environ.get("AUDIT_LOG_FILE", "/app/logs/predictions.jsonl"))

@app.get("/dashboard/data/recent-predictions")
def recent_predictions(limit: int = 20):
    if not AUDIT_LOG_FILE.exists():
        return []
    lines = [l for l in AUDIT_LOG_FILE.read_text().strip().split("\n") if l]
    return list(reversed([json.loads(l) for l in lines[-limit:]]))  # newest first
```

Static files served via `app.mount("/dashboard", StaticFiles(...))` — same URL
contract as the monolith; existing dashboard JS requires zero changes. The new
`/dashboard/data/recent-predictions` endpoint can be wired to a new "Recent" tab
in the dashboard JS if desired.

---

## docker-compose.yml (full replacement)

```yaml
networks:
  internal:
    driver: bridge

volumes:
  rabbitmq-data:       # RabbitMQ persistence — queues survive restarts
  prediction-logs:     # shared between audit-worker (writer) and dashboard-svc (reader)

services:

  # ── API Gateway ─────────────────────────────────────────────────────────
  gateway:
    build: { context: ., dockerfile: services/gateway/Dockerfile }
    ports: ["80:80"]
    depends_on: [inference-svc, cv-svc, ga-svc, dashboard-svc]
    networks: [internal]
    restart: always

  # ── Message Broker ───────────────────────────────────────────────────────
  rabbitmq:
    image: rabbitmq:3.13-management-alpine
    expose: ["5672"]                  # AMQP — internal only
    environment:
      - RABBITMQ_DEFAULT_USER=${RABBITMQ_USER:-guest}
      - RABBITMQ_DEFAULT_PASS=${RABBITMQ_PASS:-guest}
    volumes:
      - rabbitmq-data:/var/lib/rabbitmq
    networks: [internal]
    restart: always
    healthcheck:
      test: ["CMD", "rabbitmq-diagnostics", "ping"]
      interval: 15s
      timeout: 10s
      retries: 5
      start_period: 30s
    deploy:
      resources:
        limits: { cpus: "0.5", memory: 256M }

  # ── Inference Service ────────────────────────────────────────────────────
  inference-svc:
    build: { context: ., dockerfile: services/inference/Dockerfile }
    expose: ["8001"]
    environment:
      - MODEL_PATH=${MODEL_PATH}
      - API_KEY=${API_KEY}
      - INTERNAL_API_KEY=${INTERNAL_API_KEY:-${API_KEY}}
      - AWS_REGION=${AWS_REGION:-us-east-1}
      - ENABLE_CLOUDWATCH=${ENABLE_CLOUDWATCH:-0}
      - CPU_LIMIT=2
      - MODEL_TEST_MODE=${MODEL_TEST_MODE:-0}
      - RABBITMQ_URL=amqp://${RABBITMQ_USER:-guest}:${RABBITMQ_PASS:-guest}@rabbitmq:5672/
    depends_on:
      rabbitmq:
        condition: service_healthy
    networks: [internal]
    restart: always
    deploy:
      resources:
        limits:       { cpus: "2", memory: 2G }
        reservations: { cpus: "0.5", memory: 512M }

  # ── CV Service ───────────────────────────────────────────────────────────
  cv-svc:
    build: { context: ., dockerfile: services/cv/Dockerfile }
    expose: ["8002"]
    environment:
      - CV_METRICS_PATH=/app/dashboard/data/cv_metrics.json
    networks: [internal]
    restart: always
    deploy:
      resources:
        limits: { cpus: "0.25", memory: 128M }

  # ── GA Service (read-only) ────────────────────────────────────────────────
  ga-svc:
    build: { context: ., dockerfile: services/ga/Dockerfile }
    expose: ["8003"]
    environment:
      - EA_RESULTS_DIR=/app/ea/results
    networks: [internal]
    restart: always
    deploy:
      resources:
        limits: { cpus: "0.25", memory: 128M }

  # ── Audit Worker (RabbitMQ consumer) ─────────────────────────────────────
  audit-worker:
    build: { context: ., dockerfile: services/audit/Dockerfile }
    environment:
      - RABBITMQ_URL=amqp://${RABBITMQ_USER:-guest}:${RABBITMQ_PASS:-guest}@rabbitmq:5672/
      - AUDIT_LOG_FILE=/app/logs/predictions.jsonl
    volumes:
      - prediction-logs:/app/logs
    depends_on:
      rabbitmq:
        condition: service_healthy
    networks: [internal]
    restart: on-failure
    deploy:
      resources:
        limits: { cpus: "0.1", memory: 64M }   # extremely lightweight

  # ── Dashboard Service ────────────────────────────────────────────────────
  dashboard-svc:
    build: { context: ., dockerfile: services/dashboard/Dockerfile }
    expose: ["8004"]
    environment:
      - INFERENCE_URL=http://inference-svc:8001
      - CV_URL=http://cv-svc:8002
      - GA_URL=http://ga-svc:8003
      - INTERNAL_API_KEY=${INTERNAL_API_KEY:-${API_KEY}}
      - REPORT_PATH=/app/training/outputs/logs/report.json
      - CM_PATH=/app/training/outputs/plots/cm.png
      - STATIC_DIR=/app/dashboard
      - AUDIT_LOG_FILE=/app/logs/predictions.jsonl
    volumes:
      - prediction-logs:/app/logs:ro   # read-only — only audit-worker writes here
    networks: [internal]
    restart: always
    deploy:
      resources:
        limits: { cpus: "0.5", memory: 256M }
```

---

## docker-compose.override.yml (local dev, full replacement)

```yaml
services:
  gateway:
    ports: ["8080:80"]       # avoid requiring root locally

  rabbitmq:
    ports:
      - "15672:15672"        # management UI: http://localhost:15672  (dev only)
      - "5672:5672"          # expose AMQP locally for debugging/inspection

  inference-svc:
    command: uvicorn services.inference.main:app --host 0.0.0.0 --port 8001 --reload
    volumes:
      - ./services/inference:/app/services/inference
      - ./training/outputs/model/mobilenetv2_inference.pth:/app/models/mobilenetv2_inference.pth:ro
    environment:
      - MODEL_PATH=/app/models/mobilenetv2_inference.pth
      - API_KEY=
      - INTERNAL_API_KEY=
      - RABBITMQ_URL=amqp://guest:guest@rabbitmq:5672/

  cv-svc:
    command: uvicorn services.cv.main:app --host 0.0.0.0 --port 8002 --reload
    volumes:
      - ./services/cv:/app/services/cv
      - ./dashboard/data:/app/dashboard/data:ro

  ga-svc:
    command: uvicorn services.ga.main:app --host 0.0.0.0 --port 8003 --reload
    volumes:
      - ./services/ga:/app/services/ga
      - ./ea/results:/app/ea/results:ro

  audit-worker:
    volumes:
      - ./services/audit:/app/services/audit   # hot-swap worker code

  dashboard-svc:
    command: uvicorn services.dashboard.main:app --host 0.0.0.0 --port 8004 --reload
    volumes:
      - ./services/dashboard:/app/services/dashboard
      - ./dashboard:/app/dashboard:ro
      - ./training/outputs/logs:/app/training/outputs/logs:ro
      - ./training/outputs/plots:/app/training/outputs/plots:ro
    environment:
      - INTERNAL_API_KEY=
```

---

## Model Checkpoint Strategy

| Environment | How inference-svc gets the .pth file |
|---|---|
| Local dev | Bind mount in override: `./training/outputs/model/mobilenetv2_inference.pth:/app/models/...` |
| CI (test stage) | `MODEL_TEST_MODE=1` — no checkpoint needed, stub model used |
| Production EC2 | `MODEL_PATH=s3://bucket/models/mobilenetv2_inference.pth` — existing `_maybe_download_from_s3()` handles it at startup |

---

## CI/CD Changes — `.github/workflows/deploy.yml`

**Stage 1 (Build)** — matrix build, 6 service images in parallel:
```yaml
strategy:
  matrix:
    include:
      - service: gateway
        dockerfile: services/gateway/Dockerfile
      - service: inference
        dockerfile: services/inference/Dockerfile
      - service: cv
        dockerfile: services/cv/Dockerfile
      - service: ga
        dockerfile: services/ga/Dockerfile
      - service: audit
        dockerfile: services/audit/Dockerfile
      - service: dashboard
        dockerfile: services/dashboard/Dockerfile
steps:
  - docker build -f ${{ matrix.dockerfile }} -t $REGISTRY/traffic-sign-${{ matrix.service }}:$TAG .
```

**RabbitMQ**: uses official `rabbitmq:3.13-management-alpine` — no ECR push needed,
Docker Hub pulls it on deploy.

**Stage 2 (Test)** — two sub-steps:
1. Inference only: start inference-svc with `MODEL_TEST_MODE=1` and
   `RABBITMQ_URL=""` (disables broker publish) on port 8001, run `test_api.py`
   with `API_BASE_URL=http://localhost:8001` — 14 tests unchanged, zero edits.
2. Full-stack smoke test: `docker compose up -d`, poll until RabbitMQ healthcheck
   passes, `curl http://localhost/health`, one predict call through gateway, then
   check `curl http://localhost/dashboard/data/recent-predictions` returns `[]`
   (log empty) or a single entry if the predict call was processed.

**Stage 3 (Push)** — push 6 images to ECR.
Prerequisite: create 6 ECR repos once manually:
`traffic-sign-gateway`, `traffic-sign-inference`, `traffic-sign-cv`,
`traffic-sign-ga`, `traffic-sign-audit`, `traffic-sign-dashboard`.

**Stage 4 (Deploy)** — SSH to EC2, pull all 6 images, scp updated compose file,
`docker compose up -d`. RabbitMQ pulls from Docker Hub automatically.

`.env.example` additions:
```
INTERNAL_API_KEY=change-me-secret-for-internal-calls
RABBITMQ_USER=guest
RABBITMQ_PASS=change-me-in-production
```

---

## Implementation Order

1. `services/inference/model_utils.py` — extract class maps + model loading from `api/app.py:46–180`
2. `services/inference/main.py` — copy `api/app.py`, add BackgroundTask audit publish, remove dashboard imports
3. `services/inference/Dockerfile`
4. `services/cv/main.py` + Dockerfile
5. `services/ga/main.py` + Dockerfile (read-only, no pika dependency)
6. `services/audit/worker.py` + Dockerfile (pika only, ultra-tiny image)
7. `services/dashboard/main.py` — httpx relay calls + `/dashboard/data/recent-predictions` endpoint
8. `services/dashboard/Dockerfile`
9. `services/gateway/nginx.conf` + Dockerfile
10. Replace `docker-compose.yml` and `docker-compose.override.yml`
11. Local smoke test: `docker compose up --build` → all curl checks below pass
12. Replace `.github/workflows/deploy.yml`
13. Extend `.env.example`

---

## Verification

```bash
# Start full stack (gateway on :8080 locally via override)
docker compose up --build -d

# Wait for RabbitMQ to be healthy
docker compose ps rabbitmq    # should show "healthy"

# Per-service health checks
curl http://localhost:8080/health      # aggregate fan-out — all "ok"
curl http://localhost:8080/cv/health   # cv-svc "ok"
curl http://localhost:8080/ga/health   # ga-svc "ok"

# Inference — existing test suite unchanged
python test_api.py   # API_BASE_URL=http://localhost:8080, 14/14 pass

# Dashboard data flows (gateway → dashboard-svc → ga-svc / cv-svc)
curl http://localhost:8080/dashboard/data/ea
curl http://localhost:8080/dashboard/data/cv-metrics
curl http://localhost:8080/dashboard/data/training-report
curl http://localhost:8080/dashboard/data/confusion-matrix.png -o /tmp/cm.png

# Dashboard predict relay (no auth)
curl -X POST http://localhost:8080/dashboard/predict -F file=@test_images/stop.jpg

# RabbitMQ audit flow (the message broker feature)
# Step 1: make a prediction — this fires a background event to RabbitMQ
curl -X POST http://localhost:8080/predict -F file=@test_images/stop.jpg
# Step 2: wait ~1 second for audit-worker to consume it
sleep 1
# Step 3: check the audit log via dashboard endpoint
curl http://localhost:8080/dashboard/data/recent-predictions
# → [{"predicted_class":"Stop","confidence":0.99,"latency_ms":45,"logged_at":"..."}]

# Make several predictions, verify the log grows (up to 20 shown, newest first)
for i in 1 2 3; do curl -s -X POST http://localhost:8080/predict -F file=@test_images/stop.jpg; done
sleep 2
curl http://localhost:8080/dashboard/data/recent-predictions | python -m json.tool | head -20

# RabbitMQ management UI (dev override exposes port 15672)
# Open: http://localhost:15672  (login: guest / guest)
# Verify: prediction.audit queue, prediction.dead queue, message rates graph

# Graceful degradation: stop ga-svc — prediction and audit still work
docker compose stop ga-svc
curl http://localhost:8080/health            # → {"status": "degraded", "services": {"ga": "unreachable"}}
curl -X POST http://localhost:8080/predict -F file=@test_images/stop.jpg  # → still 200

# DLQ test: kill audit-worker mid-message (or send malformed body via management UI)
# Verify message lands in prediction.dead, not silently discarded
```

**CI**: existing `test_api.py` 14 assertions run unchanged against inference-svc
directly (port 8001, `MODEL_TEST_MODE=1`, `RABBITMQ_URL=""`) — zero test file changes.
