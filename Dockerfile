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
# syntax=docker/dockerfile:1
FROM python:3.11-slim AS shared-deps

SHELL ["/bin/bash", "-c"]
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        cpio \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libgomp1 \
        libnuma-dev \
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
# Slim serving image. Models are baked in via COPY from builder,
# so no volume mount of models/ is needed at runtime.
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

# Copy ONLY the converted OpenVINO model from builder (small, ~1-2 GB)
# If builder ran scripts/03_convert_openvino.py, the model is at /app/models/openvino
COPY --from=builder /app/models/openvino  /app/models/openvino
COPY --from=builder /app/configs          /app/configs
COPY --from=builder /app/scripts          /app/scripts
COPY --from=builder /app/server.py        /app/server.py
COPY --from=builder /app/app.py           /app/app.py

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["python", "scripts/04_run_inference.py"]
