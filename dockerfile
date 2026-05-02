# syntax=docker/dockerfile:1
FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" appuser

WORKDIR /app

COPY requirements.txt .

# Layer 1: torch CPU-only (190MB — heaviest, cached separately)
# If this succeeds once, it NEVER re-downloads unless this line changes.
RUN pip install --upgrade pip --quiet \
    && pip install --no-cache-dir --retries 5 --timeout 120 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.3.0+cpu torchvision==0.18.0+cpu

# Layer 2: AWS + numpy (medium weight, separate cache layer)
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    "numpy<2" \
    boto3==1.34.0

# Layer 3: Web framework + everything else (small, fast)
RUN pip install --no-cache-dir --retries 5 --timeout 120 \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    python-multipart==0.0.9 \
    python-dotenv==1.0.1 \
    pillow==10.3.0 \
    requests==2.31.0

# Copy app code last — never busts any of the pip cache layers above
COPY api/ ./api/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]