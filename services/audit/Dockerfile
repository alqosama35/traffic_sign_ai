FROM python:3.10-slim

RUN adduser --disabled-password --gecos "" appuser

WORKDIR /app

RUN pip install --no-cache-dir pika==1.3.2

COPY services/audit/ ./services/audit/

RUN chown -R appuser:appuser /app
USER appuser

CMD ["python", "-m", "services.audit.worker"]
