# FinAI Setup Script for Windows
# Run: powershell -ExecutionPolicy Bypass -File setup.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  FinAI - FinGPT on Intel NPU Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "[ERROR] Python not found. Install Python 3.10+ from python.org" -ForegroundColor Red
    exit 1
}
$pyVersion = python --version
Write-Host "[OK] $pyVersion" -ForegroundColor Green

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}
Write-Host "[OK] Virtual environment ready" -ForegroundColor Green

# Activate and install
Write-Host "Installing dependencies..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Copy .env.example to .env and add your HF_TOKEN"
Write-Host "  2. Activate venv:  .\.venv\Scripts\Activate.ps1"
Write-Host "  3. Check hardware: python scripts/check_hardware.py"
Write-Host "  4. Download models: python scripts/01_download_models.py"
Write-Host "  5. Merge LoRA:     python scripts/02_merge_lora.py"
Write-Host "  6. Convert model:  python scripts/03_convert_openvino.py"
Write-Host "  7. Run inference:  python scripts/04_run_inference.py"
Write-Host "  8. Or run Web UI:  python app.py"
