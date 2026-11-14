# ==============================================================================
# Multi-Stage Production Dockerfile for MRI Brain Tumor Detection Platform
# Target Deployment: Google Cloud Run (Docker Runtime)
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build React Production SPA Frontend
# ------------------------------------------------------------------------------
FROM node:24-slim AS frontend-builder

WORKDIR /app/frontend

# Copy package manifests and install dependencies deterministically via npm ci
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# Copy frontend source code and compile production assets with Vite
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Runtime Environment with Python 3.11 & Machine Learning Stack
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# Set environment variables for optimized container runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install required system packages
# - build-essential: compile any binary wheel dependencies
# - curl: container health checks and probes
# - git: versioning utilities
# - libgl1 & libglib2.0-0: image codecs / graphical libraries for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 for secure container execution
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH

WORKDIR /app

# Upgrade pip and install production-only Python dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-deploy.txt

# Copy application source code, models, and output metrics
COPY --chown=appuser:appuser . /app

# Copy the compiled React application from Stage 1 into frontend/dist
COPY --from=frontend-builder --chown=appuser:appuser /app/frontend/dist /app/frontend/dist

# Switch to non-root user
USER appuser

# Expose default Google Cloud Run port
EXPOSE 8080

# Container health check (using dynamic PORT with 8080 fallback)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/api/health || exit 1

# Launch unified FastAPI server (serves React frontend + REST API)
# Uses exec form in shell to ensure proper SIGTERM signal forwarding on Cloud Run
CMD ["sh", "-c", "exec uvicorn integration_api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
