# ==============================================================================
# Multi-Stage Production Dockerfile for MRI Brain Tumor Detection Platform
# Target Deployment: Hugging Face Spaces (Docker SDK) / Container Cloud
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Build React Production SPA Frontend
# ------------------------------------------------------------------------------
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend

# Install dependencies first for optimal Docker layer caching
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --prefer-offline || npm install

# Copy frontend source code and compile production assets
COPY frontend/ ./
RUN npm run build

# ------------------------------------------------------------------------------
# Stage 2: Runtime Environment with Python & Machine Learning Stack
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

# Install required system packages
# - build-essential: compile any C extensions
# - curl: container health checks
# - git: repo operations
# - libgl1 & libglib2.0-0: image codecs / OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (standard for Hugging Face Spaces security)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code, models, output metrics, and sample data
COPY --chown=user:user . /app

# Copy the compiled React application from Stage 1 into frontend/dist
COPY --from=frontend-builder --chown=user:user /app/frontend/dist /app/frontend/dist

# Switch to non-root user
USER user

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Container health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Launch unified FastAPI server (serves React frontend + REST API)
CMD ["sh", "-c", "uvicorn integration_api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
