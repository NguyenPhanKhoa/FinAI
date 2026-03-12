# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinAI deploys FinGPT (Llama 3.1 8B Instruct + FinGPT LoRA fine-tune) on an Intel Core Ultra 7 258V NPU using OpenVINO with INT4 symmetric quantization. The target hardware has a 47 TOPS NPU (Lunar Lake), 32GB RAM, and Intel Arc 140V GPU.

## Pipeline

The deployment is a 4-step sequential pipeline — each script must complete before the next:
1. `scripts/01_download_models.py` — downloads base model + LoRA from HuggingFace
2. `scripts/02_merge_lora.py` — merges LoRA weights into base model (PEFT)
3. `scripts/03_convert_openvino.py` — converts to OpenVINO IR with INT4 sym quantization
4. `scripts/04_run_inference.py` — runs inference on NPU

## Serving & Deployment

- `server.py` — OpenAI-compatible API server (FastAPI + uvicorn) that serves the model via `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, and `/v1/models` endpoints. Supports streaming (SSE) for TUI clients.
- `app.py` — Gradio web UI that connects to `server.py` (does not load the model directly).
- OpenClaw integration: provider config in `~/.openclaw/openclaw.json` with `"api": "openai-completions"` pointing to `http://127.0.0.1:8000/v1`.

## Commands

```bash
# Setup
powershell -ExecutionPolicy Bypass -File setup.ps1

# Hardware check
python scripts/check_hardware.py

# Full pipeline
python scripts/01_download_models.py
python scripts/02_merge_lora.py
python scripts/03_convert_openvino.py

# Run inference
python scripts/04_run_inference.py              # CLI interactive
python scripts/04_run_inference.py --device CPU  # fallback
python server.py                                 # OpenAI-compatible API server
python app.py                                    # Gradio web UI (requires server.py)

# Testing
python tests/test_fingpt.py                      # Run test suite against live server
```

## Architecture

- All config lives in `configs/model_config.json` — model IDs, quantization params, inference settings, and paths
- Models are stored in `models/` (gitignored): `base/`, `merged/`, `openvino/` subdirs
- Quantization must be INT4 symmetric with ratio 1.0 for NPU compatibility
- For 7B+ models use channel-wise quantization (`group_size: -1`); for <=5B use `group_size: 128`
- Device can be overridden to CPU or GPU via `--device` flag on inference scripts and server
- NPU default prompt limit is 1024 tokens; server uses `MAX_PROMPT_LEN=8192` for OpenClaw agent prompts
- System prompts exceeding 2000 chars are truncated to fit NPU generation budget
- Server uses Llama 3.1 Instruct prompt template with `<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>` tokens

## Key Constraints

- Python 3.11 required (3.14 lacks prebuilt wheels for ML packages)
- OpenClaw sends extra fields in requests — Pydantic models use `extra="allow"`
- OpenClaw sends `content` as string, list, or null — `get_text()` helper handles all formats
- Streaming (SSE) is required for OpenClaw TUI to display responses
