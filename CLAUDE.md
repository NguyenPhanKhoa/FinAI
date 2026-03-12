# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

**FinGPT Finance AI Agent** — a local-first financial AI system that aggregates live financial news (Vietnam + global), performs sentiment analysis, stock forecasting, and trading signals using a real FinGPT model (Llama-2-7B + FinGPT LoRA) running on Intel NPU via IPEX-LLM. No cloud LLM costs.

## Commands

```bash
# One-time setup
python merge.py         # Merge FinGPT LoRA into Llama-2-7b (downloads ~14GB, run once)

# Start servers
python npu_server.py    # FastAPI LLM server on port 11435 (IPEX-LLM, Intel NPU)
node middleware.js      # Express middleware on port 11436 (OpenClaw bridge)

# CLI tools
npm run news                                 # Fetch Vietnam + global news
node bins/fin-news.js company AAPL           # Company-specific news (Finnhub)
node bins/fin-analyze.js sentiment "text"    # Sentiment: positive/neutral/negative
node bins/fin-analyze.js forecast TICKER     # Price forecast: up/down/stable
node bins/fin-analyze.js signal TICKER       # Trading signal: BUY/SELL/HOLD
node bins/fin-analyze.js qa "question"       # Financial Q&A
node bins/fin-search.js "query"              # DuckDuckGo search
node src/scheduler.js                        # Daily briefing cron (8AM Vietnam time)
```

## Architecture

Two-server architecture with a middleware pattern:

```
OpenClaw (Client, port 11436)
    ↓
middleware.js — keyword detection, executes fin-* commands, injects live data into context
    ↓
npu_server.py (FastAPI, port 11435) — FinGPT Llama-2-7B INT4 via IPEX-LLM
    ↓
Intel NPU (primary) → CPU INT4 (fallback)
```

**Data flow:** User message → middleware detects keywords → runs `bins/` CLI scripts → fetches live data → injects as system context → forwards enriched request to NPU server → LLM summarizes → response returned.

**Keyword triggers in middleware:** `news`, `sentiment`, `forecast`, `signal`, `search` (+ Vietnamese synonyms like `tin tức`, `phân tích`, `dự báo`).

## Key Files

| File | Role |
|------|------|
| `merge.py` | One-time script: merges `FinGPT/fingpt-mt_llama2-7b_lora` into Llama-2-7b base |
| `middleware.js` | Express server (port 11436); intercepts messages, runs enrichment |
| `npu_server.py` | FastAPI server (port 11435); IPEX-LLM inference on Intel NPU → CPU fallback |
| `src/fingpt.js` | Ollama API client for FinGPT (sentiment, forecast, signal, QA functions) |
| `src/news.js` | News aggregation from Vietnam RSS feeds, global RSS feeds, Finnhub |
| `src/stocks.js` | Stock quotes and historical data via `yahoo-finance2` |
| `src/scheduler.js` | `node-cron` job for daily 8AM briefing |
| `bins/fin-*.js` | CLI entry points used both standalone and spawned by middleware |
| `skill.json` | OpenClaw skill metadata |
| `models/fingpt-llama2-7b-merged/` | Merged FinGPT model (output of merge.py) |

## Environment Variables

Copy `.env.example` → `.env`:
- `FINNHUB_API_KEY` — for company-specific news and stock data (free tier)
- `NEWSAPI_KEY` — optional, 100 req/day free tier

Yahoo Finance and DuckDuckGo require no keys.

## Model Setup

```bash
# 1. Install Python deps
pip install --pre --upgrade ipex-llm[npu]
pip install peft transformers accelerate fastapi uvicorn huggingface_hub

# 2. Login to HuggingFace (Llama 2 is gated — request access first)
huggingface-cli login

# 3. Merge model (one time)
python merge.py

# 4. Start
python npu_server.py
```

Merged model saved at `models/fingpt-llama2-7b-merged/`. The server auto-selects: NPU → CPU.

## Data Sources

- **Vietnam news (RSS):** CafeF, VnEconomy, VNExpress, VietnamBiz, TNCK
- **Global news (RSS):** Reuters, CNBC, BBC, MarketWatch
- **Stock data:** Yahoo Finance (free, no key)
- **Company news:** Finnhub API
- **Search:** DuckDuckGo (free, no key)
