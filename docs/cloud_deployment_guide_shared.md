# Cloud Deployment Guide — EC2 + ECR + S3 + CloudWatch

A step-by-step guide for deploying a Dockerized ML API to AWS using GitHub Actions CI/CD. Covers IAM, S3, ECR, EC2, and CloudWatch from scratch with no Terraform or CDK — everything is done via the AWS Console or CLI.

---

## Prerequisites

Before starting, make sure you have:

- An AWS account with billing access
- AWS CLI installed and configured locally (`aws configure`)
- Docker installed locally
- A GitHub repository with your project code, a `Dockerfile`, and a working `docker-compose` or equivalent
- A CI/CD workflow file (e.g., `.github/workflows/deploy.yml`) that builds, tests, and deploys the Docker image
- Your trained model artifact and any other static files the container needs at runtime

---

## Architecture Overview

```
GitHub Actions
  ├── Build Docker image
  ├── Run test suite against container
  ├── Push image → ECR (tagged by git SHA)
  └── SSH deploy → EC2

EC2 (t2.medium, Ubuntu 22.04)
  └── Docker container (--restart always)
        ├── Loads model artifact from S3 via IAM role (no hardcoded credentials)
        ├── Serves HTTP API on configured port
        └── Emits custom metrics → CloudWatch via boto3

S3 Bucket
  └── Stores model artifact, experiment logs, reports, dashboard data

CloudWatch
  └── Custom metrics dashboard + high-error-rate alarm
```

The EC2 instance authenticates to both ECR and S3 using an **IAM instance role** — no credentials are stored in the container or code.

---

## Implementation Phases

### Phase 1 — Billing Protection + IAM

Do this **before** provisioning any EC2 or S3 resources.

#### Billing alerts

1. AWS Console → **Billing → Budgets → Create a Cost budget**
2. Set amount to your chosen limit (e.g. $10), alert at 80% actual and 100% forecasted → add your email
3. Create a second budget at your hard cap (e.g. $20)

#### IAM user for GitHub Actions

This user only needs permission to push images to ECR — nothing else.

1. **IAM → Users → Create user**: name it something like `<project>-github-ci`
2. Attach managed policy: `AmazonEC2ContainerRegistryPowerUser`
3. **Security credentials → Create access key** (CLI use case) → save the key ID and secret immediately (you will not see the secret again)

#### IAM role for EC2

The instance role lets your container pull images and read from S3 without any hardcoded credentials.

1. **IAM → Roles → Create role → AWS service → EC2**
2. Attach managed policy: `AmazonEC2ContainerRegistryReadOnly` (ECR pull)
3. Add an **inline policy** (e.g. named `<Project>S3Read`):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::<your-bucket-name>",
      "arn:aws:s3:::<your-bucket-name>/*"
    ]
  }]
}
```

4. Name the role something like `<Project>EC2Role`

**Done when:** Two billing budgets are active. IAM user exists with access key saved. IAM role exists with both ECR and S3 policies attached.

---

### Phase 2 — S3 Bucket + Artifact Upload

#### Create the bucket

1. **S3 → Create bucket**
   - Name: `<your-project>-<yourname>` (must be globally unique)
   - Region: match the region you will use everywhere (e.g. `us-east-1`)
   - Block all public access: **ON**
   - Versioning: OFF (unless you need it)
   - Encryption: default (SSE-S3)

#### Upload your artifacts

Use the AWS CLI from your project root. Adjust paths and prefixes to match your project layout:

```bash
# Model artifact
aws s3 cp <local-path-to-model-file> s3://<your-bucket-name>/models/<model-filename>

# Experiment logs (directory)
aws s3 cp <local-path-to-logs-dir>/ s3://<your-bucket-name>/experiments/ --recursive

# Training report / metrics
aws s3 cp <local-path-to-report-file> s3://<your-bucket-name>/training/<report-filename>

# Any other static files the container or dashboard needs
aws s3 cp <local-path> s3://<your-bucket-name>/<prefix>/<filename>
```

**Done when:** `aws s3 ls s3://<your-bucket-name>/models/` shows your model file.

---

### Phase 3 — ECR Repository + Local Docker Verification

#### Create the ECR repository

1. **ECR → Create repository → Private**
   - Name: `<your-project>-api`
   - Region: same as your S3 bucket
2. Note the **repository URI**: `<account-id>.dkr.ecr.<region>.amazonaws.com/<your-project>-api`

#### Verify Docker locally before trusting CI/CD

Run your full test suite against the container locally first. This saves time debugging in CI.

```bash
docker build -t <your-project>-api-local .

docker run -d --name local-test -p <port>:<port> \
  -e MODEL_TEST_MODE=1 \
  <your-project>-api-local

# Wait ~10 seconds for startup, then run your test suite:
python test_api.py   # or whatever your test command is

docker stop local-test && docker rm local-test
```

All tests must pass before proceeding. Fix any failures here rather than chasing them in CI logs.

**Done when:** ECR repo exists. All local container tests pass.

---

### Phase 4 — EC2 Instance Setup

#### Launch the instance

1. **EC2 → Launch instance**
   - AMI: Ubuntu Server 22.04 LTS
   - Instance type: t2.medium (adjust if your model needs more RAM)
2. **Key pair**: Create new → download the `.pem` file → store it safely (you cannot re-download it)
3. **Security group** — add two inbound rules:
   - SSH (port 22) from **My IP only**
   - Custom TCP (your app port, e.g. 8000) from anywhere (`0.0.0.0/0`) for demo access
4. **IAM instance profile**: select the EC2 role you created in Phase 1 (Advanced details → IAM instance profile)
5. Launch. Note the **public IPv4 address**.

#### Install Docker on the instance

```bash
chmod 400 <your-key>.pem
ssh -i <your-key>.pem ubuntu@<ec2-public-ip>

# Inside EC2:
sudo apt update && sudo apt install -y docker.io awscli
sudo usermod -aG docker ubuntu
sudo systemctl enable docker && sudo systemctl start docker

# Log out and back in so the group change takes effect
exit
ssh -i <your-key>.pem ubuntu@<ec2-public-ip>
docker info   # must not say "permission denied"
```

#### Install CloudWatch agent for memory metrics (optional but recommended)

```bash
# Inside EC2:
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb

# Minimal config — reports memory used % every 60 seconds
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

**Done when:** EC2 is running, public IPv4 noted, `docker info` works without sudo, CloudWatch agent status shows active.

---

### Phase 5 — GitHub Secrets

Add all required secrets under **Settings → Secrets and variables → Actions → New secret**:

| Secret name | Value |
|---|---|
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID |
| `AWS_ACCESS_KEY_ID` | Access key ID for your GitHub CI IAM user |
| `AWS_SECRET_ACCESS_KEY` | Secret access key for your GitHub CI IAM user |
| `EC2_SSH_KEY` | Full contents of your `.pem` file (including `-----BEGIN...` and `-----END...` lines) |
| `EC2_HOST` | Public IPv4 address of your EC2 instance |
| `EC2_USER` | `ubuntu` |
| `MODEL_PATH` | S3 URI to your model file, e.g. `s3://<your-bucket>/models/<model-file>` |
| `API_KEY` | A strong random string — generate with `openssl rand -hex 32` |
| `S3_BUCKET` | Your bucket name |

Add any additional secrets your `deploy.yml` references (e.g. `DATA_SOURCE`, environment flags).

**Done when:** All secrets appear in the GitHub secrets list.

---

### Phase 6 — First CI/CD Run

Push any pending change to trigger the pipeline:

```bash
git add .github/workflows/deploy.yml   # or whatever file you changed
git commit -m "chore: configure production environment variables"
git push origin main
```

Open **GitHub → Actions** and watch the triggered run. A typical pipeline has four stages:

1. **Build** — `docker build` (~4–8 min first time, faster with layer caching)
2. **Test** — starts container, runs test suite (~2 min)
3. **Push to ECR** — authenticates and pushes the image (~2 min)
4. **Deploy to EC2** — SSH in, pull new image, restart container, health check (~2 min)

If a stage fails, click it, read the step logs, and fix the root cause. Common issues:
- Wrong secret value (especially the SSH key — paste the full PEM including header/footer lines)
- Wrong EC2 IP (re-check after instance stop/start; the public IP changes unless you use an Elastic IP)
- Security group blocking SSH from GitHub runner IP ranges

After Stage 4 succeeds, verify from your local machine:

```bash
curl http://<ec2-public-ip>:<port>/health
# Expected: {"status":"ok"} or your equivalent health response
```

**Done when:** All pipeline stages green. Health check returns 200 from your machine.

---

### Phase 7 — CloudWatch Alarms + End-to-End Verification

#### Generate traffic so CloudWatch has data

```bash
# Run your test suite against the live URL
API_BASE_URL=http://<ec2-public-ip>:<port> API_KEY=<your-api-key> python test_api.py
```

#### Create a CloudWatch alarm

1. **CloudWatch → Alarms → Create alarm → Select metric**
2. Navigate to your custom metrics namespace (set in your app code)
3. Select your error count and request count metrics
4. Choose **Math expression**: `m1 / (m2 + 1) * 100` where `m1 = ErrorCount` (Sum) and `m2 = RequestCount` (Sum)
5. Period: 5 minutes. Threshold: > 1 (i.e. alarm if error rate exceeds 1%). Treat missing data as: not breaching
6. Give the alarm a descriptive name. Add an SNS email notification if you want alerts

#### Create a CloudWatch dashboard

1. **CloudWatch → Dashboards → Create dashboard** → give it a name
2. Add widgets for your key metrics:
   - Line chart: API latency (p95, 5-min period)
   - Number: request count (Sum, 5-min)
   - Number: error count (Sum, 5-min)
   - Line chart: `CWAgent / mem_used_percent` (Average, 1-min) — from the CloudWatch agent
3. Save the dashboard

#### Final verification checklist

- [ ] CloudWatch alarm exists and is in OK state
- [ ] CloudWatch dashboard shows all metrics with data points
- [ ] All API endpoints are reachable from a browser (`/health`, `/docs`, `/predict`)
- [ ] Your application's dashboard or UI loads correctly
- [ ] Your full test suite passes against the live URL

---

## What to Skip if Time is Tight

1. **CloudWatch memory widget** — the CloudWatch agent setup can be skipped if you're under time pressure; the alarm and API metrics are more important
2. **Large dataset upload to S3** — if your container doesn't need the dataset at runtime, skip this; you can always upload it later
3. **CloudWatch SNS email notification** — the alarm itself matters; the email action is optional
4. **Elastic IP** — your EC2 public IP will change on stop/start, but for a demo you can just update the secrets each time

---

## Common Issues & Tips

| Symptom | Likely cause | Fix |
|---|---|---|
| Pipeline fails at Push to ECR | IAM user missing ECR permissions | Verify `AmazonEC2ContainerRegistryPowerUser` is attached |
| Pipeline fails at Deploy — SSH timeout | GitHub runner IP not in security group | EC2 security group SSH rule must be `0.0.0.0/0` (or use GitHub's published IP ranges) |
| Container starts but model fails to load | EC2 role missing S3 read permission | Check the inline policy ARNs match your actual bucket name |
| `docker info` says permission denied | ubuntu user not in docker group | `sudo usermod -aG docker ubuntu` then log out and back in |
| EC2 public IP changed | Instance was stopped and restarted | Allocate an Elastic IP and associate it, or update `EC2_HOST` secret |
| CloudWatch shows no metrics | Container not emitting metrics | Check your app's metric-publishing code and that the IAM role allows `cloudwatch:PutMetricData` |

---

## Security Notes

- **Never hardcode AWS credentials** in your code or Dockerfile — use IAM roles for EC2 and GitHub secrets for CI
- The EC2 IAM role should be **read-only** for S3 and ECR — no write access needed at runtime
- SSH access should be restricted to known IPs; only open port 22 to `0.0.0.0/0` if you cannot predict GitHub runner IPs
- Store your `.pem` key file securely and never commit it to the repository
- Rotate your `API_KEY` and GitHub CI access keys if they are ever exposed

---

## After the Demo — Teardown

To avoid ongoing charges after your demo:

1. **Stop or terminate the EC2 instance** (terminate = permanent, stop = paused but still costs for EBS)
2. **Delete the ECR images** if storage costs are a concern (or let the lifecycle policy clean up old tags)
3. **S3** charges very little for storage but delete the bucket if you no longer need it
4. **CloudWatch** custom metrics are retained for 15 months but incur minimal cost
