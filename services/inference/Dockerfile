FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" appuser

WORKDIR /app

# Layer 1: PyTorch CPU (heaviest, cached separately)
RUN pip install --upgrade pip --quiet \
    && pip install --no-cache-dir --retries 5 --timeout 120 \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.3.0+cpu torchvision==0.18.0+cpu

# Layer 2: AWS + numpy
RUN pip install --no-cache-dir "numpy<2" boto3==1.34.0

# Layer 3: web framework + broker client + utilities
RUN pip install --no-cache-dir \
    fastapi==0.111.0 "uvicorn[standard]==0.29.0" \
    python-multipart==0.0.9 python-dotenv==1.0.1 \
    pillow==10.3.0 pika==1.3.2

COPY services/inference/ ./services/inference/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "services.inference.main:app", "--host", "0.0.0.0", "--port", "8001"]
