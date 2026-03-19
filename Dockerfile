# syntax=docker/dockerfile:1
# FinAI Dockerfile — multi-stage build for FinGPT pipeline.
#
# Two targets:
#   builder  (default) — full ML toolchain, used to run pipeline scripts
#   runtime  — slim image with only openvino-genai + fastapi, for serving
#
# Build:
#   docker build -t fingpt:latest --target builder .
#   docker build -t fingpt:runtime --target runtime .
#
# Runtime image does NOT include models — mount them from host:
#   docker run --rm --device=/dev/dri:/dev/dri \
#              -v ./models/openvino:/app/models/openvino:ro \
#              -v ./configs:/app/configs:ro \
#              -v ./server.py:/app/server.py:ro \
#              -p 8000:8000 \
#              fingpt:runtime python server.py
#
# Notes:
#   --device passes Intel GPU/NPU (Linux). On Windows/WSL2, GPU passthrough is
#   automatic if "WSL2 integrated GPU scheduling" is enabled in Docker Desktop.

# ─── Shared base ───────────────────────────────────────────────────────────────
FROM python:3.11-slim AS shared-deps

SHELL ["/bin/bash", "-c"]
WORKDIR /app

# Install system deps.
# NOTE: on Debian 12 (bookworm) the package names changed:
#   libgl1-mesa-glx -> libgl1 (or libgl1-mesa-dri)
#   libnuma-dev      -> libnuma1 (runtime) + libnuma-dev (dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cpio \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libnuma1 \
        ocl-icd-libopencl1 \
        opencl-headers \
    && rm -rf /var/lib/apt/lists/*

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
# Used to run pipeline scripts (download, merge, convert).
# Models are written to host via volume mounts.
FROM shared-deps AS builder

# Copy requirements first for Docker layer cache
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV HF_HUB_DOWNLOAD_TIMEOUT=300

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
# Slim serving image. OpenVINO model is copied directly from build context.
FROM shared-deps AS runtime

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "openvino>=2025.0.0" \
        "openvino-genai>=2025.0.0" \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        gradio \
        huggingface_hub \
        python-dotenv && \
    pip cache purge

# Copy OpenVINO model + app files directly from build context
COPY models/openvino/ /app/models/openvino/
COPY configs/         /app/configs/
COPY scripts/         /app/scripts/
COPY server.py        /app/server.py
COPY app.py           /app/app.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "scripts/04_run_inference.py"]
