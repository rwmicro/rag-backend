# ==============================================================================
# RAG Backend - Optimized Multi-Stage Dockerfile with GPU/CPU Support
# ==============================================================================
# Based on official Docker + NVIDIA best practices
# Supports both CUDA GPU and CPU-only deployments
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Runtime with GPU Support (NVIDIA CUDA)
# ------------------------------------------------------------------------------
FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04 AS runtime-gpu

# Set non-interactive installation
ARG DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 and build dependencies (Python 3.12 is default in Ubuntu 24.04)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    build-essential \
    git \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Copy requirements and install Python packages
WORKDIR /tmp
COPY requirements.txt .

# Install Python dependencies (using --break-system-packages is safe in Docker)
RUN python3.12 -m pip install --break-system-packages --no-cache-dir --upgrade --ignore-installed pip setuptools wheel && \
    python3.12 -m pip install --break-system-packages --no-cache-dir -r requirements.txt && \
    PIP_BREAK_SYSTEM_PACKAGES=1 python3.12 -m spacy download en_core_web_sm

# Create app user (security best practice)
RUN useradd -m appuser && \
    mkdir -p /app /data && \
    chown -R appuser:appuser /app /data

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Fix permissions for all files
RUN chmod -R u+r,g+r /app && \
    find /app -type d -exec chmod u+x,g+x {} \;

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Run application with GPU support
CMD ["python3.12", "-m", "uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]

# ------------------------------------------------------------------------------
# Stage 2: Runtime CPU-only (smaller image)
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime-cpu

# Set non-interactive installation
ARG DEBIAN_FRONTEND=noninteractive

# Install build and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
WORKDIR /tmp
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Create app user
RUN useradd -m appuser && \
    mkdir -p /app /data && \
    chown -R appuser:appuser /app /data

WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Fix permissions for all files
RUN chmod -R u+r,g+r /app && \
    find /app -type d -exec chmod u+x,g+x {} \;

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=""

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Run application (CPU mode)
CMD ["uvicorn", "rag.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
