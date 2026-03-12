# FinAI — FinGPT on Intel NPU

Deploy [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) (Llama 2 7B + LoRA) on Intel Core Ultra 7 258V NPU using OpenVINO with INT4 symmetric quantization.

## Hardware Requirements

- **CPU**: Intel Core Ultra (Series 2 / Lunar Lake) with NPU
- **RAM**: 16 GB+ (32 GB recommended)
- **Storage**: ~30 GB free (for model downloads and conversion)
- **NPU Driver**: [Intel NPU Driver for Windows](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)

## Quick Start

```powershell
# 1. Setup
powershell -ExecutionPolicy Bypass -File setup.ps1

# 2. Configure
copy .env.example .env
# Edit .env and add your HuggingFace token

# 3. Activate environment
.\.venv\Scripts\Activate.ps1

# 4. Verify hardware
python scripts/check_hardware.py

# 5. Pipeline (run in order)
python scripts/01_download_models.py
python scripts/02_merge_lora.py
python scripts/03_convert_openvino.py

# 6. Run
python scripts/04_run_inference.py          # CLI chat
python app.py                               # Web UI
```

## Model Pipeline

```
Llama 2 7B (base) + FinGPT LoRA adapter
        ↓ merge (PEFT)
Merged FP16 model
        ↓ convert (optimum-cli)
OpenVINO IR + INT4 symmetric quantization
        ↓ deploy
Intel NPU via openvino-genai
```

## Project Structure

```
├── app.py                      # Gradio web UI
├── configs/
│   └── model_config.json       # Model and quantization settings
├── scripts/
│   ├── check_hardware.py       # Hardware compatibility check
│   ├── 01_download_models.py   # Download base + LoRA from HF
│   ├── 02_merge_lora.py        # Merge LoRA into base model
│   ├── 03_convert_openvino.py  # Convert to OpenVINO IR INT4
│   └── 04_run_inference.py     # CLI inference on NPU
├── models/                     # (gitignored) model files
│   ├── base/                   # Downloaded base + LoRA
│   ├── merged/                 # Merged FP16 model
│   └── openvino/               # Final INT4 OpenVINO IR
├── requirements.txt
├── setup.ps1                   # Windows setup script
└── .env.example                # Environment template
```

## Configuration

Edit `configs/model_config.json` to change:
- **Model**: swap `base_model` and `lora_model` for other FinGPT variants
- **Quantization**: adjust `group_size` (128 for ≤5B models, -1 for 7B+)
- **Inference**: change `device` to `CPU` or `GPU` as fallback

## Device Fallback

If NPU inference has issues, you can fall back to other devices:

```bash
python scripts/04_run_inference.py --device CPU
python scripts/04_run_inference.py --device GPU
python app.py --device CPU
```
