# ============================================================
# FR-CI-01: Inference Service Dockerfile
# Base image: Python 3.10 slim (lightweight, production-ready)
# ============================================================

# PART 1 — Base image + working directory
FROM python:3.10-slim

# Set /app as the working directory inside the container.
# All subsequent COPY, RUN, and CMD instructions operate from here.
WORKDIR /app

# PART 2 — Install dependencies
# IMPORTANT: Copy requirements.txt FIRST, before any other code.
# This is a Docker layer-caching trick: if your code changes but
# requirements.txt does not, Docker reuses the cached pip install
# layer and skips reinstalling — saving minutes on every CI build.
COPY requirements.txt .

# Upgrade pip, then install all dependencies.
# --no-cache-dir: do not store pip's download cache inside the image
#                 (keeps the image smaller).
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# PART 3 — Copy application code + launch the server
# Copy the entire api/ directory into the container's /app folder.
# This happens AFTER pip install so that code changes don't
# invalidate the dependency cache layer.
COPY api/ ./api/

# Document that the container listens on port 8000.
# This is required for docker-compose and AWS ECS to route traffic correctly.
EXPOSE 8000

# Launch the FastAPI app using uvicorn.
# - api.main:app  → looks for the FastAPI `app` object in api/main.py
# - --host 0.0.0.0 → listen on all interfaces (required to be reachable
#                    from outside the container)
# - --port 8000   → matches the port in docker-compose.yml (FR-CI-02)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]