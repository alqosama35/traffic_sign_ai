# Implementation Plan — Cloud Deployment (Traffic Sign AI)

## Goal

Provision all AWS infrastructure and wire up the CI/CD pipeline so the already-complete API, dashboard, and EA artifacts are live, monitored, and satisfy every FR-CLD and FR-CI requirement.

## Constraints & assumptions

- EC2 t2.medium chosen over ECS Fargate; deploy.yml already uses SSH-based EC2 deployment.
- All AWS provisioning done manually via the AWS console or CLI (no Terraform/CDK).
- Region: `us-east-1` (matches deploy.yml default; do not change mid-project).
- S3 bucket name must be globally unique — choose `traffic-sign-ai-<yourname>` and use it everywhere.
- AWS spend cap: $20. Billing alerts at $10 and $20 are created first (Phase 1).
- The API, Dockerfile, docker-compose, CI/CD YAML, EA logs, trained model, and dashboard are all code-complete. No training, EA, or API code needs to be written.
- One small code addition is needed: the deploy stage in `deploy.yml` must pass `DATA_SOURCE=s3` and `S3_BUCKET` to the container so the dashboard reads from S3 (FR-DASH-06). This is the only file change in this plan.

## Architecture overview

The container image is built and tested by GitHub Actions, pushed to ECR, then SSH-deployed to a single EC2 t2.medium. The container loads the MobileNetV2 model from S3 at startup using the EC2 instance's IAM role (no hardcoded credentials). The dashboard reads EA logs and training reports from S3 through the same IAM role. CloudWatch receives latency, request count, and error metrics from the running container via the `boto3` client already embedded in `api/app.py`.

## Components

### S3 Bucket
- **Responsibility:** Persistent store for the model artifact, EA experiment logs, training report, confusion matrix, and CV metrics.
- **Inputs / Outputs:** Receives uploads from the developer CLI; serves GET requests from the running container and the dashboard.
- **Key decisions:** Block all public access. No versioning needed for this project. Use a single bucket with prefixes matching the SRS layout.

### ECR Repository
- **Responsibility:** Stores Docker images tagged by git SHA.
- **Inputs / Outputs:** Receives `docker push` from GitHub Actions; serves `docker pull` from EC2.
- **Key decisions:** Lifecycle policy (already in deploy.yml) keeps only the last 10 images to avoid storage costs.

### IAM — GitHub CI user
- **Responsibility:** Allows GitHub Actions to push images to ECR only.
- **Key decisions:** Scoped to ECR push operations. No S3, no EC2 API access. Credentials stored as GitHub secrets.

### IAM — EC2 instance role
- **Responsibility:** Allows the EC2 instance to pull images from ECR and read all S3 prefixes needed by the API and dashboard.
- **Key decisions:** Read-only S3 (`GetObject` + `ListBucket` on the traffic-sign bucket only). ECR pull only. No write access anywhere. Attached as an instance profile so no credentials appear in code.

### EC2 t2.medium
- **Responsibility:** Runs the Docker container with `--restart always`, serves HTTP on port 8000.
- **Key decisions:** Ubuntu 22.04 LTS. Security group allows inbound port 8000 from anywhere (demo requirement) and SSH from developer IP only.

### CloudWatch
- **Responsibility:** Collect the three custom metrics emitted by the app (`Latency`, `RequestCount`, `ErrorCount` in namespace `TrafficSignAPI`), display them on a dashboard, and alarm on high error rate.
- **Key decisions:** One alarm using metric math `ErrorCount / (RequestCount + 1) * 100 > 1` over a 5-minute Sum period. Memory monitoring via the CloudWatch unified agent installed on EC2.

---

## Implementation phases

### Phase 1 — Billing protection + IAM (size: S)

Do this before any EC2 or S3 usage to eliminate surprise charges.

- **Billing alerts (FR-CLD-07):**
  1. AWS Console → Billing → Budgets → Create a Cost budget.
  2. Set amount $10, alert at 80% actual and 100% forecasted → add your email.
  3. Repeat for $20.

- **IAM user for GitHub Actions:**
  1. IAM → Users → Create user: `traffic-sign-github-ci`.
  2. Attach managed policy `AmazonEC2ContainerRegistryPowerUser`.
  3. Security credentials → Create access key (CLI use case) → save the key ID and secret.

- **IAM role for EC2:**
  1. IAM → Roles → Create role → AWS service → EC2.
  2. Attach managed policy `AmazonEC2ContainerRegistryReadOnly` (ECR pull).
  3. Add an inline policy named `TrafficSignS3Read`:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [{
         "Effect": "Allow",
         "Action": ["s3:GetObject", "s3:ListBucket"],
         "Resource": [
           "arn:aws:s3:::traffic-sign-ai-<yourname>",
           "arn:aws:s3:::traffic-sign-ai-<yourname>/*"
         ]
       }]
     }
     ```
  4. Name the role `TrafficSignEC2Role`.

Done-when: Billing budget shows two active alerts. IAM user exists with access key saved. IAM role `TrafficSignEC2Role` exists with both ECR and S3 policies attached.

---

### Phase 2 — S3 bucket + artifact upload (size: S)

- **Create bucket (FR-CLD-02):**
  1. S3 → Create bucket → name: `traffic-sign-ai-<yourname>` → region: `us-east-1`.
  2. Block all public access: ON. Versioning: OFF. Encryption: default (SSE-S3).

- **Upload artifacts** using AWS CLI (from repo root):
  ```
  aws s3 cp training/outputs/model/mobilenetv2_inference.pth \
      s3://traffic-sign-ai-<yourname>/models/traffic_sign_model_mobilenetv2.pth

  aws s3 cp ea/results/ \
      s3://traffic-sign-ai-<yourname>/ea-experiments/ --recursive

  aws s3 cp training/outputs/logs/report.json \
      s3://traffic-sign-ai-<yourname>/training/report.json

  aws s3 cp training/outputs/plots/cm.png \
      s3://traffic-sign-ai-<yourname>/training/cm.png

  aws s3 cp dashboard/data/cv_metrics.json \
      s3://traffic-sign-ai-<yourname>/dashboard/cv_metrics.json
  ```

Done-when: `aws s3 ls s3://traffic-sign-ai-<yourname>/models/` shows the `.pth` file. `aws s3 ls s3://traffic-sign-ai-<yourname>/ea-experiments/` shows 4 JSON files.

---

### Phase 3 — ECR repository + local Docker verification (size: S)

- **Create ECR repository:**
  1. ECR → Create repository → Private → name: `traffic-sign-api` → region: `us-east-1`.
  2. Note the repository URI: `<account-id>.dkr.ecr.us-east-1.amazonaws.com/traffic-sign-api`.

- **Verify Docker locally before trusting CI/CD (FR-CI-01, FR-CI-02):**
  ```
  docker build -t traffic-sign-api-local .
  docker run -d --name local-test -p 8000:8000 \
    -e MODEL_TEST_MODE=1 \
    traffic-sign-api-local
  # Wait ~10 seconds for startup
  python test_api.py
  docker stop local-test && docker rm local-test
  ```
  All 15 tests must pass before proceeding. Fix any failures here rather than in CI.

Done-when: ECR repo exists. `python test_api.py` passes 15/15 locally against the container.

---

### Phase 4 — EC2 instance setup (size: M)

- **Launch instance:**
  1. EC2 → Launch instance → Ubuntu Server 22.04 LTS → t2.medium.
  2. Key pair: create new → name `traffic-sign-key` → download `.pem` file → store safely.
  3. Security group: add two inbound rules:
     - SSH (22) from My IP only.
     - Custom TCP (8000) from anywhere (`0.0.0.0/0`).
  4. IAM instance profile: `TrafficSignEC2Role` (Advanced details → IAM instance profile).
  5. Launch. Note the public IPv4 address.

- **Install Docker on the instance:**
  ```
  chmod 400 traffic-sign-key.pem
  ssh -i traffic-sign-key.pem ubuntu@<ec2-public-ip>

  # Inside EC2:
  sudo apt update && sudo apt install -y docker.io awscli
  sudo usermod -aG docker ubuntu
  sudo systemctl enable docker && sudo systemctl start docker
  # Log out and back in so group change takes effect
  exit
  ssh -i traffic-sign-key.pem ubuntu@<ec2-public-ip>
  docker info   # must not say "permission denied"
  ```

- **Install CloudWatch agent for memory metrics (FR-CLD-05):**
  ```
  # Inside EC2:
  wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
  sudo dpkg -i amazon-cloudwatch-agent.deb

  # Minimal config — reports MemoryUsed % every 60 seconds
  sudo tee /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json > /dev/null <<'EOF'
  {
    "metrics": {
      "append_dimensions": { "InstanceId": "${aws:InstanceId}" },
      "metrics_collected": {
        "mem": { "measurement": ["mem_used_percent"], "metrics_collection_interval": 60 }
      }
    }
  }
  EOF

  sudo /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
    -a fetch-config -m ec2 \
    -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s
  ```

Done-when: EC2 is running, public IPv4 noted, `docker info` works without sudo, CloudWatch agent is running (`sudo systemctl status amazon-cloudwatch-agent` shows active).

---

### Phase 5 — deploy.yml patch + GitHub secrets (size: S)

- **One code change to `deploy.yml`** — add `DATA_SOURCE` and `S3_BUCKET` to the `docker run` command in the deploy stage so the dashboard reads from S3 (FR-DASH-06). In the `Deploy on EC2` step, inside the SSH heredoc, add two lines to the `docker run` block:

  Current:
  ```
  -e ENABLE_CLOUDWATCH=1 \
  -e CPU_LIMIT=2 \
  ```
  Change to:
  ```
  -e ENABLE_CLOUDWATCH=1 \
  -e CPU_LIMIT=2 \
  -e DATA_SOURCE=s3 \
  -e S3_BUCKET='${{ secrets.S3_BUCKET }}' \
  ```
  Same two lines must be added to the rollback `docker run` block inside the same step.

- **Add GitHub repository secrets** (Settings → Secrets and variables → Actions → New secret):

  | Secret name | Value |
  |---|---|
  | `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID |
  | `AWS_ACCESS_KEY_ID` | Access key ID for `traffic-sign-github-ci` |
  | `AWS_SECRET_ACCESS_KEY` | Secret access key for `traffic-sign-github-ci` |
  | `EC2_SSH_KEY` | Full contents of `traffic-sign-key.pem` (including `-----BEGIN...` and `-----END...` lines) |
  | `EC2_HOST` | Public IPv4 address of your EC2 instance |
  | `EC2_USER` | `ubuntu` |
  | `MODEL_PATH` | `s3://traffic-sign-ai-<yourname>/models/traffic_sign_model_mobilenetv2.pth` |
  | `API_KEY` | A strong random string — generate with `openssl rand -hex 32` |
  | `S3_BUCKET` | `traffic-sign-ai-<yourname>` |

Done-when: All 9 secrets appear in the GitHub secrets list. `deploy.yml` has the two new `-e` lines in both the primary and rollback `docker run` blocks.

---

### Phase 6 — First CI/CD run (size: S)

- Commit the `deploy.yml` change and push to `main`:
  ```
  git add .github/workflows/deploy.yml
  git commit -m "feat: pass DATA_SOURCE and S3_BUCKET to production container"
  git push origin main
  ```
- Open GitHub → Actions → the triggered workflow run.
- Watch each stage complete in order:
  1. Stage 1 · Build — `docker build` (~4–8 min first time, cached after).
  2. Stage 2 · Test — starts container, runs `test_api.py` (~2 min).
  3. Stage 3 · Push to ECR — authenticates and pushes (~2 min).
  4. Stage 4 · Deploy to EC2 — SSH in, pull, restart container, health check (~2 min).
- If any stage fails: click it, read the step logs, fix the root cause (most common issues: wrong secret value, wrong EC2 IP, security group blocking SSH from GitHub runner IPs).
- After Stage 4 succeeds, verify from your machine:
  ```
  curl http://<ec2-public-ip>:8000/health
  # Expected: {"status":"ok"}
  ```

Done-when: All 4 stages green. `curl /health` returns 200 from your local machine.

---

### Phase 7 — CloudWatch alarms + end-to-end verification (size: S)

- **Generate some traffic** so CloudWatch has data to show:
  ```
  API_BASE_URL=http://<ec2-public-ip>:8000 API_KEY=<your-api-key> python test_api.py
  ```

- **Create CloudWatch alarm (FR-CLD-06):**
  1. CloudWatch → Alarms → Create alarm → Select metric.
  2. Custom namespaces → `TrafficSignAPI` → Metrics with no dimensions → select both `ErrorCount` and `RequestCount`.
  3. Choose `Math expression` → enter: `m1 / (m2 + 1) * 100` where `m1 = ErrorCount` (Sum) and `m2 = RequestCount` (Sum).
  4. Period: 5 minutes. Threshold: > 1. Treat missing data as: not breaching.
  5. Name: `TrafficSign-HighErrorRate`. Notification: optional SNS email.

- **Create CloudWatch dashboard (FR-CLD-05):**
  1. CloudWatch → Dashboards → Create dashboard → name: `TrafficSignAPI`.
  2. Add widgets:
     - Line: `TrafficSignAPI / Latency` (p95, 5-min)
     - Number: `TrafficSignAPI / RequestCount` (Sum, 5-min)
     - Number: `TrafficSignAPI / ErrorCount` (Sum, 5-min)
     - Line: `CWAgent / mem_used_percent` (Average, 1-min) — from the CloudWatch agent on EC2
  3. Save dashboard.

- **Verify dashboard tab (FR-DASH-01 through FR-DASH-06):**
  Open `http://<ec2-public-ip>:8000/dashboard` in a browser.
  - GA tab: convergence chart for 4 configs, feature reduction table.
  - Model tab: accuracy stats, confusion matrix image.
  - Comparison tab: three-way table (CV Baseline / MobileNetV2 / GA-Optimized).
  - Live Prediction tab: upload a sign image, result appears inline.

- **Verify Swagger UI (FR-API-05):**
  Open `http://<ec2-public-ip>:8000/docs` — all endpoints visible and testable.

Done-when: CloudWatch alarm exists in OK state. Dashboard shows all 4 metrics with data. All 4 dashboard tabs load. `/docs` works. `test_api.py` passes 15/15 against the live URL.

---

## What to cut first

1. CloudWatch memory widget — if the CloudWatch agent is complex to set up, skip it. The SRS acceptance criteria (§9.4) does not list memory as a graded metric; it is only mentioned in FR-CLD-05.
2. S3 dataset upload — `s3://<bucket>/dataset/` is referenced in the SRS but the container does not use it at runtime. Upload is optional if bandwidth is an issue (600 MB).
3. CloudWatch SNS email notification — the alarm itself is required; the email action is a convenience.

## Open questions

- **EC2 teardown**: The SRS requires the instance to be terminated after the demo. Decide who is responsible for stopping the instance to avoid ongoing charges after the demo date.
- **HTTPS**: The SRS does not require TLS, so HTTP on port 8000 is acceptable for the demo. If an examiner's browser blocks mixed-content uploads, you may need an HTTPS terminator (e.g., Nginx with a self-signed cert).
- **cv_metrics.json content**: `dashboard/data/cv_metrics.json` is present but check it contains real CV baseline accuracy numbers from the CV notebook, not placeholder values, before the demo.
