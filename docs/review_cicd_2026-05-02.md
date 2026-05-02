
══════════════════════════════════════════════════════════════
  CODE REVIEW REPORT
  Reviewed: CI/CD — Dockerfile, docker-compose.yml,
            docker-compose.override.yml, .dockerignore,
            .env.example, .github/workflows/deploy.yml,
            test_api.py, api/app.py
  Date: 2026-05-02
  Domain: INFRA · TEST · API                Language: YAML · Python · Bash
══════════════════════════════════════════════════════════════

## SRS Requirements Compliance Matrix

All FR-CI and relevant NFR requirements are checked below.

| Req ID        | Description                                                    | Status |
|---------------|----------------------------------------------------------------|--------|
| FR-CI-01      | Dockerfile uses Python 3.10-slim base image                    | ✅ PASS |
| FR-CI-02      | docker-compose.yml defines API service, port 8000, MODEL_PATH  | ✅ PASS |
| FR-CI-03      | .github/workflows/deploy.yml with 4 stages on push to main     | ✅ PASS |
| FR-CI-04      | Pipeline fails and blocks merge if any stage fails             | ✅ PASS |
| FR-CI-05      | Image tagged `<ecr-repo>:<git-sha>`                            | ✅ PASS |
| NFR-REL-01    | Container restarts automatically on crash                      | ✅ PASS |
| NFR-MAINT-02  | requirements.txt pins all Python dependency versions           | ⚠️ PARTIAL |
| NFR-MAINT-03  | Env-specific values via env vars, never hardcoded              | ✅ PASS |
| NFR-PORT-01   | Container runs identically locally and on AWS                  | ✅ PASS |
| NFR-PORT-02   | docker-compose.yml allows local running without AWS dependency | ✅ PASS |
| NFR-SEC-04    | AWS credentials absent from source code and Dockerfiles        | ✅ PASS |

**All 5 functional CI/CD requirements (FR-CI-01 through FR-CI-05) are fully satisfied.**
NFR-MAINT-02 has a partial gap — see Issue #1.

---

## Summary

  Total issues found: 9
  🔴 Critical:  0    🟠 High: 2    🟡 Medium: 4    🟢 Low: 2    🔵 Info: 1

  Overall health: NEEDS WORK

The CI/CD pipeline is structurally complete and satisfies all five FR-CI
requirements. The four-stage GitHub Actions workflow (build → test → push →
deploy) is well-designed, the rollback logic is a genuine safety net, and the
use of an IAM instance profile instead of passing AWS credentials over SSH is
an improvement on what the SRS specified. The two high-severity issues are a
real security correctness risk (single-quote injection in the SSH heredoc) and
a maintenance time bomb (Dockerfile bypasses requirements.txt entirely). The
medium issues are mostly hygiene: inconsistent resource limits, redundant
library in the image, and accumulated disk waste on EC2. None of these block
the demo but two of them should be fixed before any production-grade handoff.

────────────────────────────────────────────────────────────
## Issues — Ranked by Severity
────────────────────────────────────────────────────────────

### 🟠 HIGH — Issue #1: Dockerfile Ignores requirements.txt — Silent Drift Risk

**Location:** `dockerfile` lines 11–32  
**Dimension:** Maintainability / Infra

**Problem:**
`requirements.txt` is copied into the image (`COPY requirements.txt .`) but
is never consumed with `pip install -r requirements.txt`. All dependencies are
installed via hardcoded `RUN pip install` commands. This satisfies today's
installs but creates a silent drift: anyone who adds a dependency to
`requirements.txt` (the normal workflow) will get it in their virtualenv but
NOT in the Docker image — leading to runtime `ModuleNotFoundError` in
production while local development succeeds. NFR-MAINT-02 requires that
requirements.txt pins all dependencies; the Dockerfile making it advisory-only
undermines that contract.

**Example of the problem:**
```dockerfile
COPY requirements.txt .          # copied but never used
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    ...                          # diverges silently from requirements.txt
```

**Best Fix:**
Keep the layer-split strategy for torch caching (it legitimately saves 3+
minutes on cache hits) but explicitly install the remaining dependencies from
`requirements.txt` in Layer 3 so the file is the single source of truth.

```dockerfile
COPY requirements.txt .

# Layer 1: torch CPU-only (heaviest; separate layer so it caches independently)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0+cpu torchvision==0.18.0+cpu

# Layer 2: everything else from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
```

Then ensure `requirements.txt` lists all remaining packages with pinned
versions. This way a `requirements.txt` update always flows through to Docker.

---

### 🟠 HIGH — Issue #2: Single-Quote Shell Injection in SSH Heredoc for Secrets

**Location:** `.github/workflows/deploy.yml` lines 211–212  
**Dimension:** Security / Reliability

**Problem:**
`MODEL_PATH` and `API_KEY` GitHub secret values are injected into the SSH
remote command using single-quote quoting:
`MODEL_PATH_VAL='${{ secrets.MODEL_PATH }}'`. GitHub Actions expands
`${{ secrets.MODEL_PATH }}` before the shell evaluates it. If either secret
value contains a single-quote character (e.g., a path like `s3://bucket/it's/
model.pth` or an API key generated with `'` in it), the single-quote quoting
breaks and the remaining shell commands execute in the wrong context — or fail
silently. The secret value is also visible in EC2 process tables (`ps aux`)
since it is passed as a command-line argument.

**Example of the problem:**
```yaml
MODEL_PATH_VAL='${{ secrets.MODEL_PATH }}'
# If secrets.MODEL_PATH = "s3://bucket/model's.pth":
# → shell sees: MODEL_PATH_VAL='s3://bucket/model's.pth'
#               which is a syntax error / broken quoting
```

**Best Fix:**
Pass secrets as SSH environment variables (using `SetEnv` or the `-o SendEnv`
flag with `AcceptEnv` configured on the server), or write them to a temporary
env-file on the runner and `scp` that file. The simplest safe pattern is to
pass them via `-e VAR=value` to `docker run` using env vars already present
in the runner environment, never interpolating them into the heredoc body.

```yaml
- name: Deploy on EC2
  env:
    DEPLOY_MODEL_PATH: ${{ secrets.MODEL_PATH }}
    DEPLOY_API_KEY:    ${{ secrets.API_KEY }}
  run: |
    ssh -i ~/.ssh/deploy_key \
        -o SendEnv="DEPLOY_MODEL_PATH DEPLOY_API_KEY" \
        ${{ secrets.EC2_USER }}@${{ secrets.EC2_HOST }} \
        'docker run -d --name inference-api --restart always -p 8000:8000 \
         -e MODEL_PATH="$DEPLOY_MODEL_PATH" \
         -e API_KEY="$DEPLOY_API_KEY" ...'
```
(Requires `AcceptEnv DEPLOY_*` in `/etc/ssh/sshd_config` on EC2.)
Alternatively, use AWS SSM Parameter Store to retrieve secrets on the EC2
side at deploy time, keeping them entirely out of the SSH command.

**References:** OWASP A03 (Injection); CWE-78 (OS Command Injection)

---

### 🟡 MEDIUM — Issue #3: No Memory/CPU Resource Limits on EC2 docker run

**Location:** `.github/workflows/deploy.yml` lines 231–240  
**Dimension:** Reliability / Infra

**Problem:**
`docker-compose.yml` defines sensible resource limits (`cpus: "2"`,
`memory: 2G`) but the EC2 deploy step starts the container via a raw
`docker run` command that omits both `--cpus` and `--memory` flags. A
container without a memory limit can OOM the host, killing the EC2 instance
or causing the kernel OOM killer to terminate the container mid-inference.
On a t2.medium (4 GB RAM) this is a real risk if any other process competes
for memory.

**Example of the problem:**
```bash
docker run -d \
  --name inference-api \
  --restart always \
  -p 8000:8000 \
  -e MODEL_PATH=... \
  $FULL_IMAGE   # no --memory or --cpus
```

**Best Fix:**
Mirror the docker-compose resource constraints in the `docker run` invocation.

```bash
docker run -d \
  --name inference-api \
  --restart always \
  --memory 2g --memory-swap 2g \
  --cpus 2 \
  -p 8000:8000 \
  -e MODEL_PATH=... \
  $FULL_IMAGE
```

---

### 🟡 MEDIUM — Issue #4: `IMAGE_TAG` Recomputed Independently in Three Separate Jobs

**Location:** `.github/workflows/deploy.yml` lines 32, 122, 189  
**Dimension:** Maintainability / Reliability

**Problem:**
The same `${GITHUB_SHA:0:7}` computation is duplicated in the `build`,
`push-to-ecr`, and `deploy` jobs. This works today because `GITHUB_SHA` is
immutable within a workflow run, but any future change to the tag format
(e.g., switching to a longer SHA, adding a branch prefix, or using a semantic
version) must be applied in three places and can diverge if one is missed.
GitHub Actions provides `outputs` specifically to share computed values
between jobs without duplication.

**Example of the problem:**
```yaml
# Line 32  (build job)
- run: echo "IMAGE_TAG=${GITHUB_SHA:0:7}" >> $GITHUB_ENV
# Line 122 (push-to-ecr job)
- run: echo "IMAGE_TAG=${GITHUB_SHA:0:7}" >> $GITHUB_ENV
# Line 189 (deploy job)
- run: echo "IMAGE_TAG=${GITHUB_SHA:0:7}" >> $GITHUB_ENV
```

**Best Fix:**
Compute the tag once in the `build` job, expose it as a job output, and
reference it downstream.

```yaml
jobs:
  build:
    outputs:
      image_tag: ${{ steps.tag.outputs.value }}
    steps:
      - id: tag
        run: echo "value=${GITHUB_SHA:0:7}" >> $GITHUB_OUTPUT

  push-to-ecr:
    needs: [test]
    env:
      IMAGE_TAG: ${{ needs.build.outputs.image_tag }}
    # ... (no separate computation step)
```

---

### 🟡 MEDIUM — Issue #5: `requests` Library Installed in Production Image

**Location:** `dockerfile` line 32  
**Dimension:** Maintainability / Performance (image size)

**Problem:**
`requests==2.31.0` is installed in the production Docker image but is only
used in `test_api.py` (the CI test script). `api/app.py` does not import
`requests` anywhere. Including it inflates the image size and widens the
attack surface (every additional library is a potential CVE vector). In CI,
`test_api.py` installs its dependencies separately (`pip install requests
pillow`) so removing `requests` from the Dockerfile would not break the
test stage.

**Best Fix:**
Remove `requests==2.31.0` from the production Docker install. It is already
correctly installed in the test stage via:
```yaml
- name: Install test deps
  run: pip install requests pillow
```

---

### 🟡 MEDIUM — Issue #6: No Old Image Cleanup on EC2 After Successful Deploy

**Location:** `.github/workflows/deploy.yml` (deploy job, after health check)  
**Dimension:** Reliability / Infra

**Problem:**
Each deployment pulls a new image and starts the new container but never
prunes old images from the EC2 instance. A t2.medium's root EBS volume
defaults to 8 GB. MobileNetV2 CPU image is approximately 2.0–2.5 GB after
layers. After as few as 3–4 deploys the disk fills up, causing the next
`docker pull` to fail with a "no space left on device" error and triggering
a failed deployment with no rollback path.

**Best Fix:**
Add a cleanup step after a confirmed healthy deployment.

```bash
# After HEALTHY=1 is confirmed
docker image prune -f --filter "until=24h"
# Or more targeted: remove all images except the current and previous tag
docker images $REGISTRY/$REPO --format "{{.ID}} {{.Tag}}" | \
  grep -v "$TAG\|latest" | awk '{print $1}' | xargs -r docker rmi -f || true
```

---

### 🟢 LOW — Issue #7: `.env.example` Contains Personal Developer Path

**Location:** `.env.example` line 1  
**Dimension:** Maintainability

**Problem:**
The example value for `MODEL_PATH` is a hardcoded personal local path
(`D:/fuck_ai/project/traffic_sign_ai/training/outputs/model/
mobilenetv2_inference.pth`). Any team member who copies this file verbatim
will have a broken `MODEL_PATH` that silently defaults to a non-existent
location. The path also leaks the developer's local directory name into the
repository history.

**Best Fix:**
```bash
# .env.example
MODEL_PATH=/path/to/training/outputs/model/mobilenetv2_inference.pth
# or for S3: MODEL_PATH=s3://<your-bucket>/models/traffic_sign_model_mobilenetv2.pth
API_KEY=change-me-before-production
AWS_REGION=us-east-1
ENABLE_CLOUDWATCH=0
CPU_LIMIT=2
```

---

### 🟢 LOW — Issue #8: ECR Lifecycle Policy Applied on Every Push

**Location:** `.github/workflows/deploy.yml` lines 152–167  
**Dimension:** Reliability / Infra

**Problem:**
`aws ecr put-lifecycle-policy` is called on every successful push. The
`put-lifecycle-policy` call is idempotent so it does not cause any error, but
it does consume IAM permission (`ecr:PutLifecyclePolicy`) that the push
credentials must have. This is an unnecessary permission for routine CI and
increases the blast radius of a compromised `AWS_ACCESS_KEY_ID`. The policy
should be set once as part of infrastructure setup (Terraform / console),
not on every code push.

**Best Fix:**
Move the ECR lifecycle policy configuration to a one-time infrastructure
setup step (IaC, CloudFormation, or manual console configuration) and remove
the `aws ecr put-lifecycle-policy` step from the deploy workflow.

---

### 🔵 INFO — Issue #9: test_api.py Uses Synthetic Images, Not GTSRB Samples

**Location:** `test_api.py` lines 36–44  
**Dimension:** Testability

**Observation:**
FR-API-06 requires "at least 10 representative test images." The test script
generates 12 synthetic colored squares with an ellipse overlay rather than
using actual GTSRB test set images. In CI this is intentional — `MODEL_TEST_MODE=1`
uses random weights so GTSRB images would not produce meaningful classifications
anyway. The test validates the API contract (HTTP codes, response schema,
auth) correctly. No action required for CI, but for a production smoke test or
the final demo, running `test_api.py` against actual GTSRB samples with the
real model would provide higher confidence that end-to-end classification
works as expected.

---

────────────────────────────────────────────────────────────
## What's Done Well
────────────────────────────────────────────────────────────

- **Rollback is implemented and correct.** The deploy job captures the previous
  running image ID before stopping it, performs a post-deploy health check with
  10 retries, and automatically restores the previous image if the new container
  fails to become healthy. Most student CI/CD pipelines skip this entirely.

- **IAM role on EC2 instead of GitHub secrets for deployment pull.** FR-CI-03
  only requires `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` for the ECR push.
  The implementation correctly uses those only for the push job and relies on
  the EC2 instance profile for the `docker pull` inside the SSH script. This
  prevents long-lived credentials from being exposed over SSH.

- **MODEL_TEST_MODE=1 stub enables clean CI without a model file.** Rather than
  requiring the `.pth` file to be baked into the image or downloaded during CI,
  the app supports a test mode with random-weight MobileNetV2. This means the
  entire build → test → push pipeline runs without any AWS dependency or large
  binary artifact.

- **Non-root user in Dockerfile.** The Dockerfile creates `appuser` and drops
  privileges before running uvicorn, reducing the damage radius of a container
  escape. Many Dockerfiles skip this step.

- **Layer ordering in Dockerfile is correct.** Torch (190 MB) is installed first
  in its own layer; application code is copied last. This ensures torch is never
  re-downloaded when only Python files change, which is the dominant change
  pattern in this project.

- **Deployment timeout guards.** Both the test job (20 retries × 3s) and the
  deploy health check (10 retries × 3s) have bounded retry loops with explicit
  failure messages and container log dumps on failure. This makes CI failures
  debuggable without SSH access.

────────────────────────────────────────────────────────────
## Refactor Roadmap (Priority Order)
────────────────────────────────────────────────────────────

  P0 — Fix before next production push:
    [ ] Issue #2: Harden SSH heredoc to prevent single-quote injection on secret values
    [ ] Issue #1: Make Dockerfile install from requirements.txt (keep torch layer-split)
    [ ] Issue #3: Add --memory and --cpus flags to docker run in deploy script

  P1 — Fix this sprint:
    [ ] Issue #4: Compute IMAGE_TAG once in build job, expose as output, reference downstream
    [ ] Issue #5: Remove `requests` from Dockerfile (test stage already installs it separately)
    [ ] Issue #6: Add `docker image prune` step after confirmed healthy deployment

  P2 — Fix next time you touch this code:
    [ ] Issue #7: Replace personal path in .env.example with a generic placeholder
    [ ] Issue #8: Move ECR lifecycle policy to one-time IaC setup, remove from workflow

────────────────────────────────────────────────────────────
## Domain-Specific Checklist
────────────────────────────────────────────────────────────

### Security Hardening
  [x] No secrets or credentials in source files (GitHub secrets used correctly)
  [x] Non-root user in Dockerfile
  [x] IAM role on EC2 (no long-lived credentials on the server)
  [ ] Secret values injection into SSH heredoc needs single-quote protection (Issue #2)
  [x] .env file excluded from Docker build context (.dockerignore)
  [x] .env excluded from git (via .dockerignore + should be in .gitignore)

### Infrastructure / DevOps
  [x] All 4 CI/CD stages present (FR-CI-03)
  [x] Pipeline blocks merge on any stage failure (FR-CI-04, `needs:` chain)
  [x] Image tagged with git SHA (FR-CI-05)
  [x] Container health check defined (Dockerfile HEALTHCHECK + runtime retry loops)
  [x] Restart policy set (NFR-REL-01: `restart: always`)
  [x] Rollback implemented on health check failure
  [ ] Resource limits missing from deploy docker run (Issue #3)
  [ ] Old image pruning not implemented on EC2 (Issue #6)
  [x] Docker build context exclusions correct (.dockerignore)
  [x] ECR image lifecycle policy in place

### Performance
  [x] Dockerfile layer order optimised for cache (torch first, code last)
  [x] Model warmup pass at startup eliminates first-request latency spike
  [ ] `requests` library included in production image unnecessarily (Issue #5)

══════════════════════════════════════════════════════════════
```
