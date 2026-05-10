FROM python:3.10-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && adduser --disabled-password --gecos "" appuser

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.111.0 "uvicorn[standard]==0.29.0" \
    httpx==0.27.0 python-multipart==0.0.9 pillow==10.3.0

COPY services/dashboard/ ./services/dashboard/
COPY dashboard/ ./dashboard/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

CMD ["uvicorn", "services.dashboard.main:app", "--host", "0.0.0.0", "--port", "8004"]
