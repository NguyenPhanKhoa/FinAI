# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FinAI deploys FinGPT (Llama 2 7B + LoRA fine-tune) on an Intel Core Ultra 7 258V NPU using OpenVINO with INT4 symmetric quantization. The target hardware has a 47 TOPS NPU (Lunar Lake), 32GB RAM, and Intel Arc 140V GPU.

## Pipeline

The deployment is a 4-step sequential pipeline — each script must complete before the next:
1. `scripts/01_download_models.py` — downloads base model + LoRA from HuggingFace
2. `scripts/02_merge_lora.py` — merges LoRA weights into base model (PEFT)
3. `scripts/03_convert_openvino.py` — converts to OpenVINO IR with INT4 sym quantization
4. `scripts/04_run_inference.py` — runs inference on NPU

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
python app.py                                    # Gradio web UI
```

## Architecture

- All config lives in `configs/model_config.json` — model IDs, quantization params, inference settings, and paths
- Models are stored in `models/` (gitignored): `base/`, `merged/`, `openvino/` subdirs
- Quantization must be INT4 symmetric with ratio 1.0 for NPU compatibility
- For 7B+ models use channel-wise quantization (`group_size: -1`); for ≤5B use `group_size: 128`
- Device can be overridden to CPU or GPU via `--device` flag on inference scripts
- NPU has a 1024-token prompt length limit by default
