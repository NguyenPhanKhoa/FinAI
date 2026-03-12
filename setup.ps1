# FinGPT Setup Script for Windows (IPEX-LLM + Intel NPU)
# Run this in PowerShell as Administrator

Write-Host "=== FinGPT NPU Setup ===" -ForegroundColor Cyan
Write-Host ""

# Step 1: Python dependencies for IPEX-LLM NPU
Write-Host "[1/4] Installing Python dependencies (IPEX-LLM for NPU)..." -ForegroundColor Yellow
Write-Host "      This installs ipex-llm, peft, transformers, fastapi, uvicorn" -ForegroundColor Gray
pip install --pre --upgrade ipex-llm[npu]
pip install peft transformers accelerate fastapi uvicorn huggingface_hub
Write-Host "      Python dependencies installed." -ForegroundColor Green

# Step 2: HuggingFace login (required for Llama 2 gated model)
Write-Host ""
Write-Host "[2/4] HuggingFace login (needed to download Llama 2)..." -ForegroundColor Yellow
Write-Host "      If you haven't already, request access at:" -ForegroundColor Gray
Write-Host "      https://huggingface.co/meta-llama/Llama-2-7b-chat-hf" -ForegroundColor Gray
huggingface-cli login
Write-Host "      Logged in." -ForegroundColor Green

# Step 3: Merge FinGPT LoRA into base model (downloads ~14GB on first run)
Write-Host ""
Write-Host "[3/4] Merging FinGPT LoRA into Llama-2-7b (~14GB download, one time)..." -ForegroundColor Yellow
Set-Location $PSScriptRoot
python merge.py
Write-Host "      Model merged and saved to models/fingpt-llama2-7b-merged/" -ForegroundColor Green

# Step 4: Node.js dependencies
Write-Host ""
Write-Host "[4/4] Installing Node.js dependencies..." -ForegroundColor Yellow
npm install
Write-Host "      Dependencies installed." -ForegroundColor Green

Write-Host ""
Write-Host "=== Setup Complete! ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start:" -ForegroundColor White
Write-Host "  1. python npu_server.py     (starts FinGPT on NPU at port 11435)" -ForegroundColor Gray
Write-Host "  2. node middleware.js        (starts OpenClaw bridge at port 11436)" -ForegroundColor Gray
Write-Host ""
Write-Host "CLI tools still work as before:" -ForegroundColor White
Write-Host "  npm run news                 Fetch Vietnam + global news" -ForegroundColor Gray
Write-Host "  node bins/fin-analyze.js sentiment `"Apple beats earnings`"" -ForegroundColor Gray
Write-Host "  node bins/fin-analyze.js signal AAPL" -ForegroundColor Gray
