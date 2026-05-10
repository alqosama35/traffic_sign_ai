FROM python:3.10-slim

RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

RUN pip install --no-cache-dir fastapi==0.111.0 "uvicorn[standard]==0.29.0"

COPY services/cv/ ./services/cv/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')"

CMD ["uvicorn", "services.cv.main:app", "--host", "0.0.0.0", "--port", "8002"]
