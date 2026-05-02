```
══════════════════════════════════════════════════════════════
  CODE REVIEW REPORT
  Reviewed: API Engineer scope — api/app.py · test_api.py ·
            dockerfile · docker-compose.yml ·
            docker-compose.override.yml · requirements.txt ·
            .github/workflows/deploy.yml (Stages 1-2)
  SRS Scope: FR-API-01 → FR-API-07, NFR-SEC-01/03/04,
             NFR-MAINT-02, FR-CI-01 (impacts API deliverable)
  Date: 2026-05-02
  Domain: API · BE · INFRA · TEST
  Language: Python 3.10 / FastAPI / Docker / GitHub Actions
══════════════════════════════════════════════════════════════

## Summary

  Total issues found: 12
  🔴 Critical:  1    🟠 High: 1    🟡 Medium: 5    🟢 Low: 2    🔵 Info: 3

  Overall health: CRITICAL

  The FastAPI logic in app.py is well-structured and satisfies the
  core SRS requirements (FR-API-01 through FR-API-06) with clean
  in-memory preprocessing, proper auth gating, a static lookup
  table for categories, and a complete disclaimer. However one
  show-stopping defect exists: the Dockerfile is named `dockerfile`
  (all lowercase), which means `docker build .` will silently fail
  to find it on Linux (the GitHub Actions ubuntu-latest runner),
  blocking every CI stage. A secondary high-severity gap is that
  `torch` and `torchvision` are missing from requirements.txt,
  breaking local development. Several medium issues — non-timing-safe
  key comparison, category-name mismatch vs SRS, and a latency
  assertion that never fails — round out the findings.

────────────────────────────────────────────────────────────
## Issues — Ranked by Severity
────────────────────────────────────────────────────────────

### 🔴 CRITICAL — Issue #1: Lowercase `dockerfile` breaks all CI stages on Linux
**Location:** `dockerfile` (root) + `deploy.yml:38` (`docker build .`)
**Dimension:** Infrastructure / Reliability

**Problem:**
Docker's default context lookup expects the file to be named `Dockerfile` (capital D). On Linux — where GitHub Actions `ubuntu-latest` runners run — filesystems are case-sensitive, so `dockerfile` (lowercase) is an entirely different filename. Running `docker build .` on the runner will fail with "no such file or directory: Dockerfile", blocking Stages 1–4 of the CI pipeline and therefore every push to `main`. This violates FR-CI-01 ("packaged in a `Dockerfile`") and NFR-PORT-01.

**Example of the problem:**
```yaml
# deploy.yml:35-39 — ubuntu-latest runner, case-sensitive filesystem
- name: Build Docker image
  run: |
    docker build \            # looks for ./Dockerfile
      -t $ECR_REGISTRY/...   # finds nothing → fatal error
      .
# File on disk is: dockerfile (lowercase) — NEVER found
```

**Best Fix:**
Rename the file from `dockerfile` to `Dockerfile` (capital D). This is also required by Docker convention and the SRS literal text. On Windows the rename is effectively invisible, but git will track it correctly via `git mv`:

```bash
git mv dockerfile Dockerfile
git commit -m "fix: rename dockerfile → Dockerfile for Linux CI compatibility"
```

---

### 🟠 HIGH — Issue #2: `torch` and `torchvision` absent from requirements.txt
**Location:** `requirements.txt` (entire file)
**Dimension:** Maintainability / Reliability

**Problem:**
The file comment states "requirements.txt is now only used locally (pip install -r requirements.txt)", but `torch` and `torchvision` — the two heaviest and most critical dependencies — are absent. Any developer running `pip install -r requirements.txt` locally will end up with a broken environment where `api/app.py` fails immediately at import with `ModuleNotFoundError: No module named 'torch'`. This also violates NFR-MAINT-02 ("shall pin all Python dependency versions"). Note that the Dockerfile never calls `pip install -r requirements.txt` either — it installs everything directly — making requirements.txt serve neither purpose correctly.

**Example of the problem:**
```
# requirements.txt — missing the most important deps
numpy<2
fastapi==0.111.0
# torch          ← NOT HERE
# torchvision    ← NOT HERE
```

**Best Fix:**
Add CPU-only torch pins matching the Dockerfile versions:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.0+cpu
torchvision==0.18.0+cpu
numpy==1.26.4
fastapi==0.111.0
uvicorn[standard]==0.29.0
python-multipart==0.0.9
python-dotenv==1.0.1
pillow==10.3.0
boto3==1.34.0
requests==2.31.0
```

---

### 🟡 MEDIUM — Issue #3: API key comparison is not timing-safe
**Location:** `api/app.py:174`
**Dimension:** Security

**Problem:**
`x_api_key != API_KEY` uses Python's built-in string equality, which short-circuits at the first differing byte. An attacker can measure response-time variance (timing attack) to determine the key length and common prefix, reducing the brute-force search space. NFR-SEC-01 requires the endpoint to reject invalid keys with 401; the intent is strong auth, but the implementation is subtly weak. Since this endpoint is publicly accessible on AWS (FR-CLD-01), the timing signal is real.

**Example of the problem:**
```python
# api/app.py:174
if x_api_key != API_KEY:          # short-circuits → timing leak
    raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

**Best Fix:**
Use `hmac.compare_digest` for constant-time comparison:

```python
import hmac

def _check_api_key(x_api_key: str | None) -> None:
    if not API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
```

**References:** OWASP ASVS V2.9 — Cryptographic Verification; Python `hmac.compare_digest` docs.

---

### 🟡 MEDIUM — Issue #4: Category values deviate from SRS specification
**Location:** `api/app.py:91-96`
**Dimension:** Correctness

**Problem:**
FR-API-01 names the expected categories as: `"prohibition"`, `"warning"`, `"mandatory"`, `"informational"`. The implementation uses `"danger"` (for warning/hazard signs, classes 18–31) and `"other"` (for regulatory/info signs like Stop, Priority Road, End restrictions). While the SRS uses "e.g." which allows flexibility, `test_api.py` only asserts non-empty string — meaning no automated check will catch the mismatch. An examiner comparing the live API response to the SRS will see a discrepancy. The SRS §5.1 example response shows `"category": "prohibition"` matching the spec names exactly.

**Example of the problem:**
```python
# api/app.py:91-96
CATEGORY_BY_ID = {
    **{i:"prohibition" for i in [0,1,2,3,4,5,7,8,9,10,15,16,17]},
    **{i:"danger"      for i in [18,...,31]},   # SRS says "warning"
    **{i:"mandatory"   for i in [33,...,40]},
    **{i:"other"       for i in [6,11,12,13,14,32,41,42]},  # SRS says "informational"
}
```

**Best Fix:**
Rename to match the SRS-specified vocabulary and tighten the test assertion:

```python
CATEGORY_BY_ID = {
    **{i: "prohibition"   for i in [0,1,2,3,4,5,7,8,9,10,15,16,17]},
    **{i: "warning"       for i in [18,19,20,21,22,23,24,25,26,27,28,29,30,31]},
    **{i: "mandatory"     for i in [33,34,35,36,37,38,39,40]},
    **{i: "informational" for i in [6,11,12,13,14,32,41,42]},
}
```

Also update `test_api.py` to assert the value is one of the four allowed strings:
```python
VALID_CATEGORIES = {"prohibition", "warning", "mandatory", "informational"}
ok = assert_true("'category' is a known value",
                 body.get("category") in VALID_CATEGORIES,
                 f"got: {body.get('category')!r}") and ok
```

---

### 🟡 MEDIUM — Issue #5: FR-API-07 latency test is informational only — never fails
**Location:** `test_api.py:144-146`
**Dimension:** Testability

**Problem:**
FR-API-07 is an acceptance criterion: "API response time shall be <500ms per request." The test prints a warning when latency exceeds 500ms but does **not** increment `failures` and does not call `sys.exit(1)`. This means a 5-second API passes all tests. FR-CI-04 requires the pipeline to "fail and block merge if any stage fails" — but this failure path is permanently disabled.

**Example of the problem:**
```python
# test_api.py:144-146
latency_ok = elapsed_ms < 500
print(f"  latency: {elapsed_ms:.0f}ms {'✓' if latency_ok else '⚠ above 500ms ...'}")
# ↑ missing: if not latency_ok: failures += 1
```

**Best Fix:**
Treat latency as a real failure but only when running against a non-stub server (to avoid flaky failures in CI where MODEL_TEST_MODE=1 and container startup adds overhead):

```python
latency_ok = elapsed_ms < 500
if not latency_ok and not os.environ.get("MODEL_TEST_MODE"):
    print(f"  FAIL  latency {elapsed_ms:.0f}ms exceeds 500ms limit (FR-API-07)")
    failures += 1
else:
    marker = "✓" if latency_ok else "⚠ (stub mode — not enforced)"
    print(f"  latency: {elapsed_ms:.0f}ms {marker}")
```

---

### 🟡 MEDIUM — Issue #6: `DEV_MODE=1` in docker-compose.override.yml is a dead variable
**Location:** `docker-compose.override.yml:15`
**Dimension:** Maintainability

**Problem:**
The override file sets `DEV_MODE=1` with the comment "disables API key enforcement locally". However, `api/app.py` never reads `DEV_MODE` — dev mode is implied by an empty `API_KEY` env var. The variable is a vestige of a previous design and its misleading comment creates confusion about how auth is actually controlled. Any future developer seeing this might assume `DEV_MODE=1` is meaningful and build incorrect tooling around it.

**Example of the problem:**
```yaml
# docker-compose.override.yml:15
- DEV_MODE=1   # comment says disables auth, but app.py never reads DEV_MODE
```

```python
# api/app.py:28-29 — auth controlled by empty API_KEY, not DEV_MODE
API_KEY = os.environ.get("API_KEY", "").strip()
# DEV_MODE is never referenced anywhere in app.py
```

**Best Fix:**
Remove the dead `DEV_MODE=1` line and document the actual auth mechanism:

```yaml
# docker-compose.override.yml
environment:
  # Auth is disabled when API_KEY is not set (empty string = open access)
  # To test auth locally, add: - API_KEY=local-test-key
  - MODEL_PATH=${MODEL_PATH:-/app/models/traffic_sign_model_mobilenetv2.pth}
  - ENABLE_CLOUDWATCH=0
  - CPU_LIMIT=2
```

---

### 🟡 MEDIUM — Issue #7: `torch.load()` missing `weights_only` — insecure deserialization
**Location:** `api/app.py:133`
**Dimension:** Security / AI-ML

**Problem:**
`torch.load(checkpoint_path, map_location=device)` loads the checkpoint as an unrestricted pickle, allowing arbitrary Python objects to execute code on deserialization. PyTorch 2.4+ will warn and eventually default to `weights_only=True`. While the model file comes from a trusted S3 bucket, the absence of `weights_only` means any supply-chain compromise of the S3 artifact could achieve remote code execution inside the container. OWASP A08 — Insecure Deserialization.

**Example of the problem:**
```python
# api/app.py:133
checkpoint = torch.load(checkpoint_path, map_location=device)
# No weights_only — loads raw pickle, arbitrary code can execute
```

**Best Fix:**
Try `weights_only=True` first (safe for state-dict checkpoints). Fall back for dict-with-metadata checkpoints only when needed:

```python
try:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
except Exception:
    # weights_only=True fails for checkpoints with non-tensor metadata (class_to_idx etc.)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
```

**References:** OWASP A08:2021 — Software and Data Integrity Failures; PyTorch security advisory on `torch.load`.

---

### 🟢 LOW — Issue #8: `numpy<2` is a range constraint, not a pinned version
**Location:** `requirements.txt:3`
**Dimension:** Maintainability

**Problem:**
NFR-MAINT-02 requires all dependency versions to be pinned. `numpy<2` allows any numpy 1.x release. A new numpy 1.x patch could introduce a breaking change or ABI shift that breaks torchvision or PIL. Reproducible builds require exact pins.

**Best Fix:**
```
numpy==1.26.4
```
(1.26.4 is the last stable 1.x release, compatible with PyTorch 2.3.x and Pillow 10.x.)

---

### 🟢 LOW — Issue #9: `note` disclaimer not asserted for exact text in tests
**Location:** `test_api.py:139-143`
**Dimension:** Testability

**Problem:**
FR-API-03 specifies the exact disclaimer: `"Screening tool only — not for use in safety-critical autonomous systems"`. The test only asserts `len(body.get("note","")) > 0` — a note with a single space would pass. The acceptance criterion for this field cannot be verified by the test suite as written.

**Best Fix:**
```python
EXPECTED_NOTE = "Screening tool only — not for use in safety-critical autonomous systems"
ok = assert_true("'note' matches SRS disclaimer",
                 body.get("note") == EXPECTED_NOTE,
                 f"got: {body.get('note')!r}") and ok
```

---

### 🔵 INFO — Issue #10: Model loaded at module level, not in FastAPI lifespan
**Location:** `api/app.py:213`
**Dimension:** Reliability

**Problem:**
`model, device, CLASSES = _load_or_stub()` executes when the module is imported. If the model download or load fails, the error surfaces as a startup exception with no graceful shutdown or health-check feedback. FastAPI's `lifespan` context manager is the idiomatic pattern — it cleanly separates startup from request handling and allows `GET /health` to return 503 instead of a crash. No action required for MVP scope, but worth addressing if the service is extended.

---

### 🔵 INFO — Issue #11: CloudWatch `ErrorCount` only counts 5xx — not 4xx
**Location:** `api/app.py:200`
**Dimension:** Observability

**Problem:**
`int(status_code >= 500)` emits `ErrorCount=1` only for server errors. Auth failures (401) and bad-input errors (422) are 4xx and are not counted. FR-CLD-05 specifies "HTTP error rate" which conventionally includes both 4xx and 5xx. Systematic auth abuse (e.g., high rate of 401s) will be invisible in the CloudWatch alarm. Raise with M7 when configuring the CloudWatch alarm threshold.

---

### 🔵 INFO — Issue #12: Test images are synthetic color patches, not representative signs
**Location:** `test_api.py:36-44`
**Dimension:** Testability

**Problem:**
FR-API-06 requires "at least 10 **representative** test images." All 12 images are 224×224 solid-color squares with a centered ellipse — no traffic sign structure at all. The stub model (MODEL_TEST_MODE=1) returns random outputs, so they serve CI purposes correctly. However, for a human examiner reviewing the test suite, solid-color patches are not "representative" of GTSRB images. A handful of real GTSRB JPEG samples stored in `testing/fixtures/` would both satisfy the SRS wording and provide real regression coverage when the real model is loaded. Informational — does not block the demo.

────────────────────────────────────────────────────────────
## What's Done Well
────────────────────────────────────────────────────────────

- **Complete SRS field coverage in `/predict` response.** All five required fields (`class`, `category`, `confidence`, `top_3`, `note`) are returned in every response, matching the §5.1 schema exactly, including the exact disclaimer text (FR-API-03).
- **AUTH design is clean and safe for both dev and CI modes.** Empty `API_KEY` = open access (local), non-empty = enforced (prod/CI). No separate flags needed. The CI workflow sets `API_KEY=ci-test-key` so the 401 code path is actually exercised in CI.
- **Image never touches disk.** `io.BytesIO` processing and PIL in-memory decode fully satisfy FR-API-04 and NFR-SEC-03 with no special handling needed.
- **Dockerfile is properly hardened.** Least-privilege non-root `appuser`, HEALTHCHECK with curl, layered pip installs for cache efficiency, and Python 3.10-slim base (FR-CI-01 base image requirement met, pending filename rename fix).
- **CloudWatch metrics emitted per request via middleware, not ad-hoc.** The `metrics_middleware` intercepts every response regardless of path, ensuring consistent latency and error-count telemetry. The lazy-init `_cw_client()` pattern keeps the import side-effect-free in non-cloud environments.

────────────────────────────────────────────────────────────
## Refactor Roadmap (Priority Order)
────────────────────────────────────────────────────────────

  P0 — Fix before merge:
    [ ] Rename `dockerfile` → `Dockerfile` (Issue #1 — CI is broken)
    [ ] Add `torch==2.3.0+cpu` and `torchvision==0.18.0+cpu` to requirements.txt (Issue #2)

  P1 — Fix this sprint:
    [ ] Replace `!=` key comparison with `hmac.compare_digest()` (Issue #3)
    [ ] Rename category values to match SRS: "warning" / "informational" (Issue #4)
    [ ] Make FR-API-07 latency check a real test failure (Issue #5)
    [ ] Remove dead `DEV_MODE=1` from docker-compose.override.yml (Issue #6)
    [ ] Add `weights_only=True` to `torch.load()` call (Issue #7)

  P2 — Fix next time you touch this code:
    [ ] Pin `numpy==1.26.4` (Issue #8)
    [ ] Assert exact disclaimer text in test_api.py (Issue #9)

────────────────────────────────────────────────────────────
## SRS Requirements Compliance Matrix
────────────────────────────────────────────────────────────

  | Requirement  | Status      | Notes                                              |
  |--------------|-------------|----------------------------------------------------|
  | FR-API-01    | ✅ PASS      | class, category, confidence, top_3 all returned    |
  | FR-API-02    | ✅ PASS      | GET /health → 200 {"status":"ok"}                  |
  | FR-API-03    | ✅ PASS      | Exact disclaimer text present in every response    |
  | FR-API-04    | ✅ PASS      | In-memory only, no disk writes                     |
  | FR-API-05    | ✅ PASS      | FastAPI auto-provides /docs (Swagger UI)            |
  | FR-API-06    | ✅ PASS      | 12 image tests + 3 edge cases; all 5 SRS assertions|
  | FR-API-07    | ⚠️ PARTIAL  | Tested but not a hard failure (Issue #5)           |
  | NFR-SEC-01   | ⚠️ PARTIAL  | Auth present but timing-unsafe (Issue #3)          |
  | NFR-SEC-03   | ✅ PASS      | No disk writes                                     |
  | NFR-SEC-04   | ✅ PASS      | All secrets via env vars                           |
  | NFR-MAINT-02 | ⚠️ PARTIAL  | torch/torchvision missing, numpy unpinned (Issues #2, #8) |
  | FR-CI-01     | ❌ FAIL     | `dockerfile` lowercase breaks Linux CI (Issue #1)  |

────────────────────────────────────────────────────────────
## Domain-Specific Checklist
────────────────────────────────────────────────────────────

### Security Hardening
  [x] All user inputs validated at system boundaries (image format, auth header)
  [x] No secrets or credentials in source files (all via env vars)
  [ ] Constant-time comparison for API key (Issue #3)
  [x] Parameterized queries — N/A (no database)

### Performance
  [x] Model warmup run eliminates first-request cold-start spike
  [x] Torch inference under no_grad() context
  [ ] Latency SLA enforced in test suite (Issue #5)

### Infrastructure
  [x] Dockerfile uses non-root user
  [x] HEALTHCHECK defined in Dockerfile
  [x] restart: always in docker-compose.yml (NFR-REL-01)
  [ ] Dockerfile filename correct for Linux CI (Issue #1)
  [x] No hardcoded credentials in Dockerfile or YAML files

══════════════════════════════════════════════════════════════
```
