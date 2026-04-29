# DevOps Review — Traffic Sign AI
**Reviewed commit:** `833572c` (devOps)
**Review date:** 2026-04-29
**Reviewer scope:** Dockerfile · docker-compose.yml · docker-compose.override.yml · `.github/workflows/deploy.yml` · requirements.txt · test_api.py · api/app.py (DevOps-relevant sections)
**SRS sections referenced:** FR-CI-01–05 · FR-CLD-01–07 · FR-API-06 · NFR-REL-01 · NFR-SEC-01–04 · NFR-PORT-01–02

---

## Summary

The DevOps work in this commit establishes a solid foundation — the four-stage CI/CD pipeline exists, docker-compose is split correctly for dev/prod, and the API code handles env-var injection and in-memory image processing correctly. However there are **5 critical issues** that will break the pipeline or create security gaps, plus a number of medium/low issues. Every finding below includes the exact fix.

---

## Issue Index

| # | File | Severity | Title |
|---|---|---|---|
| 1 | CI workflow | **CRITICAL** | Test stage always fails — model not in Docker image |
| 2 | CI workflow | **CRITICAL** | AWS credentials sent to EC2 over SSH — use IAM role |
| 3 | CI workflow | **CRITICAL** | `-o StrictHostKeyChecking=no` used after ssh-keyscan |
| 4 | requirements.txt | **CRITICAL** | `boto3` missing — S3 model loading (FR-CLD-03) not wired |
| 5 | api/app.py | **CRITICAL** | API_KEY env check is skippable in prod — NFR-SEC-01 gap |
| 6 | test_api.py | HIGH | `category` field never asserted — FR-API-06 gap |
| 7 | test_api.py | HIGH | No 401 test for missing/invalid API key — NFR-SEC-01 gap |
| 8 | Dockerfile | HIGH | Dockerfile filename is lowercase — non-standard |
| 9 | Dockerfile | HIGH | No `HEALTHCHECK` instruction |
| 10 | repo root | HIGH | No `.dockerignore` — full repo sent as build context |
| 11 | CI workflow | HIGH | `IMAGE_TAG` comment says short SHA but full 40-char SHA used |
| 12 | CI workflow | HIGH | CloudWatch integration absent — FR-CLD-05/06 not met |
| 13 | Dockerfile | MEDIUM | Container runs as root — no non-root user |
| 14 | docker-compose*.yml | MEDIUM | `version:` key is deprecated in Compose v2 |
| 15 | CI workflow | MEDIUM | No `timeout-minutes` on any job |
| 16 | CI workflow | MEDIUM | No ECR lifecycle policy — images accumulate indefinitely |
| 17 | CI workflow | MEDIUM | No rollback on failed deployment health check |
| 18 | api/app.py | MEDIUM | `os.cpu_count()` ignores container CPU limits |
| 19 | Dockerfile | LOW | Stale comment references `api/main.py` |
| 20 | Dockerfile | LOW | Base image not pinned to digest hash |
| 21 | docker-compose.yml | LOW | No CPU/memory resource limits |
| 22 | repo root | LOW | No `.env.example` — local setup undocumented |

---

## Critical Issues

---

### Issue 1 — Test stage always fails: model not inside Docker image
**File:** `.github/workflows/deploy.yml` · Stage 2 · Test  
**SRS:** FR-CI-03 §2, FR-CI-04

**What happens:** The Dockerfile only copies `api/` into the image. The test job starts the container with:
```
-e MODEL_PATH=/app/models/traffic_sign_model_mobilenetv2.pth
```
That path does not exist inside the image. `load_model()` in `api/app.py` raises `FileNotFoundError` at startup, the container exits immediately, the health-check loop times out after 30 s, and the pipeline fails every single run.

**Fix — three options (pick one):**

**Option A (recommended for CI): add a `--model-check` / dummy-model flag in the API**

Add a 10-line dummy model that is used only when `MODEL_TEST_MODE=1` is set:

```python
# api/app.py — top of file
import os
TEST_MODE = os.environ.get("MODEL_TEST_MODE", "0") == "1"

def _load_or_stub():
    if TEST_MODE:
        model = _build_model(NUM_CLASSES)   # random weights, no .pth needed
        model.eval()
        device = torch.device("cpu")
        return model, device, _default_classes()
    return load_model(MODEL_PATH)

model, device, CLASSES = _load_or_stub()
```

Then in the workflow test step, start the container with the extra flag:
```yaml
- name: Start container
  run: |
    docker run -d \
      --name inference-api \
      -p 8000:8000 \
      -e MODEL_TEST_MODE=1 \
      $ECR_REGISTRY/$ECR_REPOSITORY:latest
```

**Option B: download model from S3 before starting container** (requires S3 credentials in CI — heavier setup)

**Option C: bake a tiny dummy `.pth` into the image** for CI only — acceptable but adds weight to the image.

---

### Issue 2 — AWS credentials passed to EC2 via SSH environment
**File:** `.github/workflows/deploy.yml` · Stage 4 · Deploy  
**SRS:** NFR-SEC-04, FR-CLD-04

**What happens:** The deploy step sets `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` as env vars on the GitHub runner and then SSHs into EC2 and runs `aws ecr get-login-password`. Those runner-side env vars are **not automatically forwarded** into the SSH session. The `aws ecr get-login-password` on EC2 will fail because no credentials are in scope.

Even if you explicitly pass them (e.g. via `ssh -o SendEnv=...`), passing IAM keys over SSH is against FR-CLD-04 / NFR-SEC-04.

**Fix:** Attach an IAM instance profile to the EC2 instance with `ecr:GetAuthorizationToken` + `ecr:BatchGetImage` permissions. Then on EC2 the AWS CLI picks up credentials from the instance metadata service automatically — no keys needed:

```bash
# On EC2, no credentials needed:
aws ecr get-login-password --region us-east-1 | docker login ...
```

In the GitHub Actions workflow, remove `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` from the deploy step's `env:` block (keep them only for the push-to-ecr step which runs on the GitHub runner, not EC2).

---

### Issue 3 — StrictHostKeyChecking=no used after ssh-keyscan
**File:** `.github/workflows/deploy.yml` · Stage 4 · Deploy  
**SRS:** NFR-SEC-04

**What happens:** The workflow correctly runs `ssh-keyscan -H ${{ secrets.EC2_HOST }} >> ~/.ssh/known_hosts` to populate known hosts, then immediately uses `-o StrictHostKeyChecking=no` which **disables** the known_hosts check entirely and opens the connection to MITM attacks.

**Fix:** Remove the `-o StrictHostKeyChecking=no` flag. The `ssh-keyscan` step already handles host verification.

```yaml
# Before (insecure):
run: |
  ssh -i ~/.ssh/deploy_key \
      -o StrictHostKeyChecking=no \
      ${{ secrets.EC2_USER }}@${{ secrets.EC2_HOST }} \
      "..."

# After (secure):
run: |
  ssh -i ~/.ssh/deploy_key \
      ${{ secrets.EC2_USER }}@${{ secrets.EC2_HOST }} \
      "..."
```

---

### Issue 4 — `boto3` missing from requirements.txt
**File:** `requirements.txt`  
**SRS:** FR-CLD-03 ("The inference container shall load the model file from S3 at startup using read-only IAM role credentials")

**What happens:** `boto3` is not listed in `requirements.txt`. S3 model loading is not implemented in `api/app.py` either — `load_model()` reads from a local `MODEL_PATH` only. FR-CLD-03 is entirely unmet.

**Fix (two parts):**

1. Add to `requirements.txt`:
```
boto3==1.34.0
```

2. Add S3 download logic to `api/app.py`:
```python
import boto3

def _download_model_from_s3_if_needed(path: Path) -> None:
    """Download model from S3 if MODEL_PATH starts with s3://."""
    model_path_str = os.environ.get("MODEL_PATH", "")
    if not model_path_str.startswith("s3://"):
        return
    # s3://bucket/key  →  bucket, key
    without_scheme = model_path_str[5:]
    bucket, _, key = without_scheme.partition("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(path))

# Call before load_model():
_download_model_from_s3_if_needed(MODEL_PATH)
model, device, CLASSES = load_model(MODEL_PATH)
```

Set the env var to point at S3:
```
MODEL_PATH=s3://your-bucket/models/traffic_sign_model_mobilenetv2.pth
```

The IAM role attached to the EC2/ECS instance provides the credentials automatically (no hardcoded keys — FR-CLD-04 satisfied).

---

### Issue 5 — API_KEY is skippable — NFR-SEC-01 not enforced in production
**File:** `api/app.py`  
**SRS:** NFR-SEC-01 ("The `/predict` endpoint shall require a valid API key … requests without a valid key shall return HTTP 401")

**What happens:** The auth check is:
```python
API_KEY = os.environ.get("API_KEY", "")
def _check_api_key(x_api_key): 
    if API_KEY and x_api_key != API_KEY:   # ← skipped when API_KEY is empty string
        raise HTTPException(status_code=401, ...)
```
If the `API_KEY` env var is not set (or set to an empty string), every request passes without authentication. In production this means the public endpoint is unprotected.

**Fix:** Require the key to be set at startup. Use a separate `DEV_MODE` flag to disable auth locally:

```python
API_KEY = os.environ.get("API_KEY", "")
DEV_MODE = os.environ.get("DEV_MODE", "0") == "1"

def _check_api_key(x_api_key: str | None) -> None:
    if DEV_MODE:
        return
    if not API_KEY:
        raise RuntimeError("API_KEY env var must be set in production. Set DEV_MODE=1 to skip auth locally.")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

In `docker-compose.override.yml` (local dev), add `- DEV_MODE=1`. In production, ensure `API_KEY` is always set.

---

## High Issues

---

### Issue 6 — `category` field not asserted in test_api.py
**File:** `test_api.py`  
**SRS:** FR-API-01, FR-API-06

**What happens:** The docstring at the top of `test_api.py` claims `'category' field is a non-empty string` is checked, but there is no actual assertion in the test body for `category`.

**Fix:** Add after the `confidence` assertion in the predict loop:

```python
category = body.get("category")
ok = assert_true(
    "'category' is non-empty str",
    isinstance(category, str) and len(category) > 0,
    f"got: {category!r}",
) and ok
```

---

### Issue 7 — No 401 test for missing/invalid API key
**File:** `test_api.py`  
**SRS:** NFR-SEC-01, FR-API-06

**What happens:** `test_api.py` never tests that the API rejects requests with a bad or missing `X-API-Key` header. NFR-SEC-01 acceptance is not verified by the test suite.

**Fix:** Add a new test case (e.g. Test 14):

```python
print("\n[14] POST /predict  missing API key → expect 401")
total += 1
# Send request with wrong key — should get 401 (when API_KEY is set)
files = {"file": ("test.jpg", make_image_bytes((100, 100, 100)), "image/jpeg")}
r = requests.post(f"{BASE_URL}/predict", files=files,
                  headers={"X-API-Key": "wrong-key"}, timeout=10)
ok = assert_true("status code is 401 or 200 (dev mode)",
                 r.status_code in (401, 200),
                 f"got: {r.status_code}")
if ok:
    print("  PASS")
else:
    failures += 1
```

> Note: the assertion allows 200 in dev mode (when `DEV_MODE=1`). In CI, start the container with `API_KEY=test-key` and check for 401 explicitly.

---

### Issue 8 — Dockerfile filename is lowercase
**File:** `dockerfile`  
**SRS:** FR-CI-01 ("packaged in a `Dockerfile`")

**What happens:** The file is named `dockerfile` (all lowercase). Docker on Linux is case-sensitive and `docker build .` searches for `Dockerfile` first. While Docker will usually still find a lowercase `dockerfile`, it is non-standard, breaks tooling (VS Code syntax highlighting, hadolint, etc.), and violates the FR-CI-01 wording.

**Fix:** Rename the file:
```bash
git mv dockerfile Dockerfile
```

---

### Issue 9 — No `HEALTHCHECK` instruction in Dockerfile
**File:** `dockerfile`  
**SRS:** NFR-REL-01

**What happens:** Docker and ECS cannot detect if the process inside the container has deadlocked or is stuck in a bad state without a `HEALTHCHECK`. ECS will show the container as "RUNNING" even if the FastAPI process is frozen.

**Fix:** Add before the `CMD` line:
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
```

This also requires `curl` to be installed — add it before the `COPY requirements.txt` step:
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
```

---

### Issue 10 — No `.dockerignore` file
**File:** (missing from repo root)  
**SRS:** NFR-PORT-01

**What happens:** Without `.dockerignore`, the entire repository is sent as the Docker build context. This includes the dataset (potentially GBs), training outputs, Jupyter notebooks, Python caches, and git history. This severely slows down every `docker build` call.

**Fix:** Create `.dockerignore` in the repo root:
```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/

# Training outputs / datasets
data/
dataset/
training/outputs/
*.pth
*.pkl
*.h5

# Notebooks
*.ipynb
.ipynb_checkpoints/

# Git and CI
.git/
.github/

# Dev tools
.env
*.log
```

---

### Issue 11 — IMAGE_TAG comment says "first 7 chars" but full 40-char SHA is used
**File:** `.github/workflows/deploy.yml`

**What happens:** The workflow comment says:
> "We use the first 7 chars (short SHA) — still unique, much more readable"

But the actual value is:
```yaml
IMAGE_TAG: ${{ github.sha }}   # ← full 40-char SHA
```
The tag will be 40 characters, not 7. Not pipeline-breaking but tags become unreadable (e.g. `abc1234def5678...`).

**Fix:**
```yaml
# In the build step, compute the short SHA:
- name: Build Docker image
  run: |
    SHORT_SHA="${GITHUB_SHA:0:7}"
    docker build \
      -t $ECR_REGISTRY/$ECR_REPOSITORY:${SHORT_SHA} \
      -t $ECR_REGISTRY/$ECR_REPOSITORY:latest \
      .
    echo "IMAGE_TAG=${SHORT_SHA}" >> $GITHUB_ENV
```

Then remove the `IMAGE_TAG` from the top-level `env:` block and reference `$IMAGE_TAG` as before (it is now set via `GITHUB_ENV`).

---

### Issue 12 — CloudWatch integration absent
**File:** `api/app.py`, `.github/workflows/deploy.yml`  
**SRS:** FR-CLD-05, FR-CLD-06

**What happens:** FR-CLD-05 requires CloudWatch to collect request count, p50/p95 latency, HTTP error rate, and memory usage. FR-CLD-06 requires a CloudWatch alarm on error rate. Neither is implemented — there is no `boto3` CloudWatch client, no middleware logging metrics, and no Terraform/CloudFormation/AWS CLI commands in the workflow to create the alarm.

**Fix (minimum viable):**

Add a FastAPI middleware to emit latency and error metrics to CloudWatch:

```python
# api/app.py
import time
import boto3

cw = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1"))

@app.middleware("http")
async def cloudwatch_metrics(request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    latency_ms = (time.monotonic() - start) * 1000
    is_error = response.status_code >= 500

    if os.environ.get("ENABLE_CLOUDWATCH", "0") == "1":
        cw.put_metric_data(
            Namespace="TrafficSignAPI",
            MetricData=[
                {"MetricName": "Latency",   "Value": latency_ms, "Unit": "Milliseconds"},
                {"MetricName": "ErrorCount","Value": int(is_error), "Unit": "Count"},
                {"MetricName": "RequestCount","Value": 1, "Unit": "Count"},
            ],
        )
    return response
```

Add a one-time CLI command in the workflow or a setup script to create the alarm (FR-CLD-06):
```bash
aws cloudwatch put-metric-alarm \
  --alarm-name "API-HighErrorRate" \
  --metric-name ErrorCount \
  --namespace TrafficSignAPI \
  --statistic Sum \
  --period 300 \
  --threshold 1 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

---

## Medium Issues

---

### Issue 13 — Container runs as root
**File:** `dockerfile`

Running as root inside a container means a container breakout gives an attacker root on the host. Add a non-root user:

```dockerfile
# After WORKDIR /app:
RUN adduser --disabled-password --gecos "" appuser \
 && chown -R appuser:appuser /app
USER appuser
```

---

### Issue 14 — `version:` key is deprecated in Compose v2
**Files:** `docker-compose.yml`, `docker-compose.override.yml`

Docker Compose v2 (the current default) no longer uses the `version:` field and emits a deprecation warning. Remove it from both files:

```yaml
# Remove this line from both compose files:
version: "3.8"
```

---

### Issue 15 — No `timeout-minutes` on CI jobs
**File:** `.github/workflows/deploy.yml`

GitHub Actions default timeout is 6 hours. A hung build (e.g. pip install stalls, SSH hangs) will block the runner for 6 hours and consume billing minutes.

**Fix:** Add timeouts to each job:
```yaml
build:
  timeout-minutes: 20
test:
  timeout-minutes: 10
push-to-ecr:
  timeout-minutes: 15
deploy:
  timeout-minutes: 10
```

---

### Issue 16 — No ECR lifecycle policy — images accumulate
**File:** `.github/workflows/deploy.yml`

Every push creates a new image in ECR tagged with the git SHA. After 50 pushes there are 50 images. ECR storage is billed per GB.

**Fix:** Add a lifecycle policy (one-time setup, can be in a setup script):
```bash
aws ecr put-lifecycle-policy \
  --repository-name traffic-sign-api \
  --lifecycle-policy-text '{
    "rules": [{
      "rulePriority": 1,
      "description": "Keep last 10 images",
      "selection": {"tagStatus": "any", "countType": "imageCountMoreThan", "countNumber": 10},
      "action": {"type": "expire"}
    }]
  }'
```

---

### Issue 17 — No rollback on failed post-deploy health check
**File:** `.github/workflows/deploy.yml` · Stage 4

If the health check at Step 4c fails, the pipeline is marked failed but the broken container is left running on EC2 (the old container was stopped in Step 4b). The live endpoint is now down.

**Fix:** In the deploy SSH script, record the previous image tag before pulling, and roll back on failure:

```bash
# Before docker stop:
PREV_TAG=$(docker inspect inference-api --format='{{.Config.Image}}' 2>/dev/null || echo "none")

# After starting new container, check health:
sleep 5
if ! curl -sf http://localhost:8000/health; then
  echo "Health check failed — rolling back to $PREV_TAG"
  docker stop inference-api && docker rm inference-api
  docker run -d --name inference-api --restart always \
    -p 8000:8000 -e MODEL_PATH=... $PREV_TAG
  exit 1
fi
```

---

### Issue 18 — `os.cpu_count()` ignores container CPU limits
**File:** `api/app.py`

```python
torch.set_num_threads(os.cpu_count())
```

Inside a container, `os.cpu_count()` returns the **host's** CPU count, not the container's CPU allocation. On a 64-core host, this would spawn 64 threads for a container allocated only 1 vCPU, causing CPU thrashing and higher latency.

**Fix:**
```python
import math
cpu_limit = int(os.environ.get("CPU_LIMIT", os.cpu_count() or 1))
torch.set_num_threads(cpu_limit)
torch.set_num_interop_threads(max(1, math.floor(cpu_limit / 2)))
```

Set `CPU_LIMIT=2` in the docker-compose.yml or ECS task definition to match the actual allocation.

---

## Low Issues

---

### Issue 19 — Stale comment in Dockerfile references wrong module path
**File:** `dockerfile`

```dockerfile
# CMD ["uvicorn", "api.main:app", ...   ← comment says api/main.py
CMD ["uvicorn", "api.app:app", ...]     # ← correct: api/app.py
```

The comment above the `CMD` line says `api/main.py` but the module is `api/app.py`. Fix the comment.

---

### Issue 20 — Base image not pinned to digest hash
**File:** `dockerfile`

```dockerfile
FROM python:3.10-slim
```

If the `python:3.10-slim` tag is updated upstream (e.g. a security patch that breaks something), your builds silently change behavior. Pin the digest for reproducibility:

```dockerfile
FROM python:3.10-slim@sha256:<current-digest>
```

Get the current digest: `docker pull python:3.10-slim && docker inspect python:3.10-slim --format='{{index .RepoDigests 0}}'`

---

### Issue 21 — No CPU/memory resource limits in docker-compose
**File:** `docker-compose.yml`

Without limits, the container can consume all host resources. Add:
```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 512M
```

---

### Issue 22 — No `.env.example` file
**File:** (missing)

The compose file references `${MODEL_PATH}`, `${API_KEY}`, etc. There is no `.env.example` showing which variables must be set. A new team member has no guidance.

**Fix:** Create `.env.example`:
```
# Copy to .env and fill in values
MODEL_PATH=/app/models/traffic_sign_model_mobilenetv2.pth
API_KEY=change-me
AWS_REGION=us-east-1
DEV_MODE=1           # set to 0 in production
ENABLE_CLOUDWATCH=0  # set to 1 in production
```

---

## SRS Requirements Coverage Gap

| SRS Requirement | Status | Issue # |
|---|---|---|
| FR-CI-01 Dockerfile, Python 3.10 slim | Partial — lowercase filename | #8 |
| FR-CI-02 docker-compose, port 8000, MODEL_PATH env | Met | — |
| FR-CI-03 4-stage pipeline on push to main | Partial — test stage broken | #1 |
| FR-CI-04 Fail and block on any stage failure | Met | — |
| FR-CI-05 Image tagged `<ecr-repo>:<git-sha>` | Partial — full SHA not short SHA | #11 |
| FR-CLD-03 Load model from S3 at startup | **Not met** | #4 |
| FR-CLD-04 Least-privilege IAM, no hardcoded creds | Partial — creds over SSH | #2 |
| FR-CLD-05 CloudWatch metrics | **Not met** | #12 |
| FR-CLD-06 CloudWatch alarm | **Not met** | #12 |
| NFR-REL-01 `restart: always` | Met | — |
| NFR-SEC-01 X-API-Key on /predict | Partial — bypassable | #5 |
| NFR-SEC-04 No credentials in code/Dockerfiles | Partial — SSH creds leak | #2 |
| NFR-PORT-02 Compose works locally without AWS | Met via override file | — |
| FR-API-06 test_api.py assertions complete | Partial — missing category, 401 | #6, #7 |

---

## Priority Fix Order

1. **Issue 1** — fix the broken test stage (CI is entirely non-functional without this)
2. **Issue 4** — add `boto3` + S3 model loading (major SRS feature gap)
3. **Issue 2 + 3** — fix SSH deploy security (credentials + StrictHostKeyChecking)
4. **Issue 5** — enforce API_KEY in production
5. **Issue 12** — CloudWatch integration (required for Cloud course grade)
6. **Issue 8 + 9 + 10** — rename Dockerfile, add HEALTHCHECK, add .dockerignore
7. **Issues 6 + 7** — complete test_api.py assertions
8. **Issues 13–22** — medium/low improvements
