# FinAI — FinGPT on Intel NPU

Deploy [FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) (Llama 3.1 8B + LoRA) on Intel Core Ultra 7 258V NPU using OpenVINO with INT4 symmetric quantization. Includes an OpenAI-compatible API server, Gradio web UI, and OpenClaw AI Agent integration.

## Features

- **NPU Acceleration** — Runs on Intel NPU (47 TOPS) with INT4 quantization via OpenVINO
- **OpenAI-Compatible API** — Drop-in replacement server for any OpenAI client
- **Streaming Support** — SSE streaming for real-time token delivery
- **Financial AI** — Sentiment analysis, market forecasting, risk assessment, financial text processing
- **Multiple Interfaces** — CLI, Web UI (Gradio), API server, OpenClaw TUI
- **Automated Testing** — 14-case test suite covering all capabilities and endpoints

## Limitations

- **No real-time data** — The model has no internet access and cannot fetch live news, stock prices, or market data. It only analyzes text you provide to it.
- **Training data cutoff: 2023** — The base model (Llama 3.1) and FinGPT LoRA were trained on data up to 2023. The model has no knowledge of events after that date.
- **Not financial advice** — Outputs are for informational/educational purposes only and should not be used as the sole basis for investment decisions.

## What You Can Ask

You provide the text, the model provides the analysis.

**Works well:**
- "Analyze the sentiment of this headline: [paste news]"
- "Summarize this earnings report: [paste text]"
- "Extract key financial metrics from: [paste data]"
- "What are the risks of investing in a single sector?"
- "Explain the P/E ratio and how to use it"
- "What factors affect bond prices when interest rates rise?"
- "Compare growth investing vs value investing"

**Won't work:**
- "What is AAPL's stock price today?" (no internet access)
- "What happened in the 2025 market crash?" (training cutoff: 2023)
- "Should I buy TSLA right now?" (no live data to base this on)

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | Intel Core Ultra (Lunar Lake) with NPU | Intel Core Ultra 7 258V |
| **RAM** | 16 GB | 32 GB |
| **Storage** | 30 GB free | 50 GB free |
| **NPU Driver** | [Intel NPU Driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html) | Latest version |
| **OS** | Windows 11 | Windows 11 |

## Quick Start

```powershell
# 1. Setup environment (Python 3.11 required)
powershell -ExecutionPolicy Bypass -File setup.ps1

# 2. Configure HuggingFace token
copy .env.example .env
# Edit .env and add your HuggingFace token (needs Llama 3.1 access)

# 3. Activate environment
.\.venv\Scripts\Activate.ps1

# 4. Verify hardware
python scripts/check_hardware.py

# 5. Run pipeline (in order)
python scripts/01_download_models.py
python scripts/02_merge_lora.py
python scripts/03_convert_openvino.py

# 6. Start server
python server.py

# 7. Launch web UI (in another terminal)
python app.py
```

## Model Pipeline

```
Meta Llama 3.1 8B Instruct (base)  +  FinGPT LoRA adapter
                    ↓ merge (PEFT)
              Merged FP16 model (~16 GB)
                    ↓ convert (optimum-cli)
        OpenVINO IR + INT4 symmetric quantization (~4.5 GB)
                    ↓ deploy
              Intel NPU via openvino-genai
```

## Usage

### API Server

The API server is OpenAI-compatible, so any OpenAI client can connect:

```bash
python server.py                    # default: NPU, port 8000
python server.py --device CPU       # fallback to CPU
python server.py --port 9000        # custom port
```

**Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions (streaming + non-streaming) |
| `/v1/responses` | POST | OpenAI Responses API |
| `/v1/completions` | POST | Legacy completions |
| `/v1/models` | GET | List available models |
| `/health` | GET | Server health check |

**Example request:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fingpt-llama3.1-8b-npu",
    "messages": [{"role": "user", "content": "What is the sentiment of: AAPL beat earnings expectations by 5%"}],
    "stream": false
  }'
```

### Web UI (Gradio)

```bash
python app.py                       # opens at http://localhost:7860
python app.py --share               # public link
```

Requires `server.py` to be running.

### CLI Interactive

```bash
python scripts/04_run_inference.py              # NPU
python scripts/04_run_inference.py --device CPU  # fallback
```

### OpenClaw AI Agent

Connect FinGPT as an AI agent in [OpenClaw](https://openclaw.com):

1. Start the server: `python server.py`
2. Add provider in `~/.openclaw/openclaw.json`:
```json
{
  "models": {
    "providers": {
      "fingpt-npu": {
        "baseUrl": "http://127.0.0.1:8000/v1",
        "apiKey": "no-key-needed",
        "api": "openai-completions",
        "models": [{
          "id": "fingpt-llama3.1-8b-npu",
          "name": "FinGPT Llama 3.1 8B (NPU)",
          "contextWindow": 32768
        }]
      }
    }
  }
}
```
3. Set as default model in `agents.defaults.model.primary`: `"fingpt-npu/fingpt-llama3.1-8b-npu"`

## Testing

Run the automated test suite (requires `server.py` running):

```bash
python tests/test_fingpt.py
```

**Test coverage (14 tests):**

| Category | Tests | Description |
|----------|-------|-------------|
| Sentiment Analysis | 3 | Positive, negative, and neutral news |
| Financial Knowledge | 3 | P/E ratio, bull/bear markets, diversification |
| Market Analysis | 2 | Revenue vs price divergence, recession indicators |
| Financial Text Processing | 2 | Earnings summarization, metric extraction |
| Risk Assessment | 1 | Concentration risk awareness |
| Domain Boundary | 1 | Non-financial request handling |
| Streaming (SSE) | 1 | Streaming endpoint validation |
| Responses API | 1 | /v1/responses endpoint validation |

Results are saved to `test_results/` as both `.txt` report and `.json` data.

## Project Structure

```
FinAI/
├── server.py                      # OpenAI-compatible API server (FastAPI)
├── app.py                         # Gradio web UI (connects to server.py)
├── configs/
│   └── model_config.json          # Model, quantization, and inference settings
├── scripts/
│   ├── check_hardware.py          # Hardware compatibility check
│   ├── 01_download_models.py      # Download base + LoRA from HuggingFace
│   ├── 02_merge_lora.py           # Merge LoRA into base model (PEFT)
│   ├── 03_convert_openvino.py     # Convert to OpenVINO IR INT4
│   └── 04_run_inference.py        # CLI inference on NPU
├── tests/
│   └── test_fingpt.py             # Automated test suite (14 cases)
├── test_results/                  # Test reports (.txt + .json)
├── models/                        # (gitignored) model files
│   ├── base/                      # Downloaded base + LoRA
│   ├── merged/                    # Merged FP16 model
│   └── openvino/                  # Final INT4 OpenVINO IR
├── FinAI_Presentation.pptx        # Project presentation (10 slides)
├── FinAI_Report_VN.docx           # Vietnamese project report
├── README.md                      # English documentation
├── README_VN.md                   # Vietnamese documentation
├── requirements.txt
├── setup.ps1                      # Windows setup script
├── .env.example                   # Environment template
└── CLAUDE.md                      # Claude Code project instructions
```

## Configuration

Edit `configs/model_config.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `base_model` | `meta-llama/Llama-3.1-8B-Instruct` | HuggingFace base model |
| `lora_model` | `FinGPT/fingpt-mt_llama3-8b_lora` | FinGPT LoRA adapter |
| `weight_format` | `int4` | Quantization format |
| `symmetric` | `true` | Symmetric quantization (required for NPU) |
| `group_size` | `-1` | Channel-wise for 7B+ models |
| `device` | `NPU` | Inference device (NPU/CPU/GPU) |
| `max_new_tokens` | `512` | Maximum generation length |
| `temperature` | `0.7` | Sampling temperature |

## Device Fallback

If NPU has issues, fall back to CPU or GPU:

```bash
python server.py --device CPU
python scripts/04_run_inference.py --device GPU
```

## Tech Stack

- **Model**: Meta Llama 3.1 8B Instruct + FinGPT LoRA (financial fine-tune)
- **Runtime**: OpenVINO GenAI with INT4 symmetric quantization
- **Hardware**: Intel Core Ultra 7 258V NPU (47 TOPS, Lunar Lake)
- **Server**: FastAPI + Uvicorn (OpenAI-compatible API)
- **Web UI**: Gradio
- **Agent**: OpenClaw AI Agent platform
- **Language**: Python 3.11

## License

This project uses Meta Llama 3.1 (subject to [Meta's license](https://llama.meta.com/llama3/license/)) and FinGPT LoRA weights from [AI4Finance](https://github.com/AI4Finance-Foundation/FinGPT).
