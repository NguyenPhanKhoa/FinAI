# FinAI Docker Runner Script
# Automates Docker Desktop installation + full FinGPT pipeline inside containers.
# Automatically skips steps that are already complete.
#
# Usage:
#   .\run.ps1                          # Interactive menu (shows step status)
#   .\run.ps1 -FullPipeline            # Run pipeline, auto-skip completed steps
#   .\run.ps1 -FullPipeline -CleanUp   # Also delete old models after each step
#   .\run.ps1 -Step 3                  # Convert to OpenVINO only
#   .\run.ps1 -Inference               # Run interactive inference
#   .\run.ps1 -Server                  # Start API server
#   .\run.ps1 -Build                   # Build Docker image only
#
# Prerequisites:
#   - .env file with HF_TOKEN (copy from .env.example if missing)
#   - Docker Desktop installed (script will install if missing on Windows)

param(
    [switch]$FullPipeline,
    [ValidateSet("1","2","3","4")]
    [string]$Step,
    [switch]$Inference,
    [switch]$Server,
    [switch]$Build,
    [switch]$CleanUp
)

$ErrorActionPreference = "Stop"
$SCRIPT_DIR = $PSScriptRoot
$HUGGINGFACE_TOKEN = $env:HF_TOKEN

# --- Helpers -----------------------------------------------------------------

function Write-Step($msg) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host "========================================`n" -ForegroundColor Cyan
}

function Write-Success($msg) {
    Write-Host "[OK] $msg" -ForegroundColor Green
}

function Write-Fail($msg) {
    Write-Host "[FAIL] $msg" -ForegroundColor Red
}

function Write-Info($msg) {
    Write-Host "[INFO] $msg" -ForegroundColor Yellow
}

function Get-DiskUsage($path) {
    try {
        $size = (Get-ChildItem $path -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1GB
        return "{0:N1} GB" -f $size
    } catch { return "?" }
}

function Remove-ModelDir($path, $label) {
    if (Test-Path $path) {
        $size = Get-DiskUsage $path
        $confirm = Read-Host "Delete $label ($size)? This frees disk space. (y/n)"
        if ($confirm -eq "y") {
            Remove-Item -Path $path -Recurse -Force
            Write-Success "Deleted $label"
        } else {
            Write-Info "Kept $label"
        }
    }
}

function Test-EnvFile {
    $envFile = Join-Path $SCRIPT_DIR ".env"
    if (-not (Test-Path $envFile)) {
        $exampleFile = Join-Path $SCRIPT_DIR ".env.example"
        if (Test-Path $exampleFile) {
            Copy-Item $exampleFile $envFile
            Write-Host "[INFO] Created .env from .env.example" -ForegroundColor Yellow
            Write-Host "       Please edit .env and add your HF_TOKEN!" -ForegroundColor Yellow
        }
    }
}

function Get-HuggingFaceToken {
    if ($HUGGINGFACE_TOKEN) { return $HUGGINGFACE_TOKEN }

    $envFile = Join-Path $SCRIPT_DIR ".env"
    if (Test-Path $envFile) {
        $content = Get-Content $envFile -Raw
        if ($content -match 'HF_TOKEN\s*=\s*"?([^"\r\n]+)"?') {
            $t = $matches[1].Trim()
            if ($t -and $t -ne "your_huggingface_token_here") { return $t }
        }
    }

    Write-Host "`n[HuggingFace Token Required]" -ForegroundColor Yellow
    Write-Host "Get your free token at: https://huggingface.co/settings/tokens" -ForegroundColor Yellow
    Write-Host "Also accept the license: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct`n" -ForegroundColor Yellow
    $token = Read-Host "Paste your HF_TOKEN"
    return $token.Trim()
}

function Test-DockerInstalled {
    try {
        $null = docker version --format "{{.Server.Version}}" 2>$null
        return $true
    } catch { return $false }
}

function Install-DockerDesktop {
    Write-Host "`nDocker Desktop not found. Downloading..." -ForegroundColor Yellow
    $url = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"
    $downloadPath = "$env:TEMP\DockerDesktopInstaller.exe"
    try {
        Write-Host "Downloading (~500MB)..." -ForegroundColor Cyan
        Invoke-WebRequest -Uri $url -OutFile $downloadPath -UseBasicParsing
        Write-Host "Starting installer..." -ForegroundColor Cyan
        Start-Process $downloadPath -Verb RunAs
        Write-Host "`nAfter Docker Desktop starts, re-run: .\run.ps1 -FullPipeline" -ForegroundColor Yellow
        exit 0
    } catch {
        Write-Fail "Download failed. Install manually: $url"
        exit 1
    }
}

function Start-DockerIfNeeded {
    if ($IsWindows) {
        try {
            $svc = Get-Service -Name "com.docker.service" -ErrorAction SilentlyContinue
            if ($svc -and $svc.Status -ne "Running") {
                Write-Host "Starting Docker service..." -ForegroundColor Cyan
                Start-Service "com.docker.service"
                Start-Sleep -Seconds 5
            }
        } catch { }
    }
    $attempts = 0
    while (-not (Test-DockerInstalled)) {
        $attempts++
        if ($attempts -gt 30) {
            Write-Fail "Docker is not running. Start Docker Desktop and retry."
            exit 1
        }
        Write-Host "Waiting for Docker... ($attempts/30)" -ForegroundColor Yellow
        Start-Sleep -Seconds 2
    }
    Write-Success "Docker is running"
}

function Build-DockerImage {
    Write-Step "Building Docker image (~10-20 min first run, then cached)"
    $null = docker compose build 2>&1 | Tee-Object -Variable output
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Docker build failed."
        Write-Host $output
        exit 1
    }
    Write-Success "Docker image built"
}

function Invoke-DockerRun {
    param([string]$Command, [string]$Token)

    $runningOnLinux = $IsLinux -or (Test-Path "/dev/dri")

    $baseArgs = @(
        "run", "--rm",
        "--network=host",
        "--shm-size=4gb",
        "-e", "HF_TOKEN=$Token",
        "-e", "PYTHONUNBUFFERED=1",
        "-v", "${SCRIPT_DIR}\models:/app/models",
        "-v", "${SCRIPT_DIR}\.cache\huggingface:/app/.cache/huggingface:rw",
        "-v", "${SCRIPT_DIR}\configs:/app/configs:ro",
        "-v", "${SCRIPT_DIR}\scripts:/app/scripts:ro",
        "-v", "${SCRIPT_DIR}\server.py:/app/server.py:ro",
        "-v", "${SCRIPT_DIR}\app.py:/app/app.py:ro",
        "-v", "${SCRIPT_DIR}\check_hardware.py:/app/check_hardware.py:ro",
        "fingpt:latest",
        $Command
    )

    if ($runningOnLinux) {
        $baseArgs = @("run", "--rm", "--device=/dev/dri:/dev/dri") + $baseArgs
    }

    docker $baseArgs
}

function Invoke-DockerServe {
    $runningOnLinux = $IsLinux -or (Test-Path "/dev/dri")

    $baseArgs = @(
        "run", "--rm",
        "--network=host",
        "--shm-size=2gb",
        "-e", "PYTHONUNBUFFERED=1",
        "-v", "${SCRIPT_DIR}\models:/app/models",
        "-v", "${SCRIPT_DIR}\configs:/app/configs:ro",
        "-v", "${SCRIPT_DIR}\scripts:/app/scripts:ro",
        "-v", "${SCRIPT_DIR}\server.py:/app/server.py:ro",
        "-v", "${SCRIPT_DIR}\app.py:/app/app.py:ro",
        "fingpt:latest",
        "python", "server.py"
    )

    if ($runningOnLinux) {
        $baseArgs = @("run", "--rm", "--device=/dev/dri:/dev/dri") + $baseArgs
    }

    docker $baseArgs
}

# --- Step Handlers ------------------------------------------------------------

function Step-DownloadModels {
    param([string]$Token)
    $basePath = Join-Path $SCRIPT_DIR "models\base\llama3.1-8b"
    if (Test-Path "$basePath\config.json") {
        Write-Info "Step 1 SKIPPED  --  base model already downloaded"
        return
    }
    Write-Step "Step 1/3: Downloading base model + LoRA from HuggingFace"
    Invoke-DockerRun -Command "python scripts/01_download_models.py" -Token $Token
    if ($CleanUp) {
        Remove-ModelDir (Join-Path $SCRIPT_DIR "models\base") "base model download cache"
    }
}

function Step-MergeLora {
    $mergedPath = Join-Path $SCRIPT_DIR "models\merged\config.json"
    if (Test-Path $mergedPath) {
        Write-Info "Step 2 SKIPPED  --  merged model already exists"
        if ($CleanUp) {
            Remove-ModelDir (Join-Path $SCRIPT_DIR "models\base") "base model (already merged)"
        }
        return
    }
    Write-Step "Step 2/3: Merging LoRA into base model"
    Invoke-DockerRun -Command "python scripts/02_merge_lora.py" -Token ""
    if ($CleanUp) {
        Remove-ModelDir (Join-Path $SCRIPT_DIR "models\base") "base model (already merged)"
    }
}

function Step-ConvertOpenVINO {
    $ovPath = Join-Path $SCRIPT_DIR "models\openvino\openvino_config.json"
    if (Test-Path $ovPath) {
        Write-Info "Step 3 SKIPPED  --  OpenVINO model already converted"
        if ($CleanUp) {
            Remove-ModelDir (Join-Path $SCRIPT_DIR "models\merged") "merged model (already converted)"
        }
        return
    }
    Write-Step "Step 3/3: Converting to OpenVINO IR (INT4 symmetric)"
    Invoke-DockerRun -Command "python scripts/03_convert_openvino.py" -Token ""
    if ($CleanUp) {
        Remove-ModelDir (Join-Path $SCRIPT_DIR "models\merged") "merged model (already converted)"
    }
}

function Step-RunInference {
    Write-Step "Running FinGPT inference (NPU/CPU)"
    Invoke-DockerRun -Command "python scripts/04_run_inference.py" -Token ""
}

function Step-RunServer {
    Write-Step "Starting FinGPT API Server"
    Write-Host "API:  http://localhost:8000/v1/chat/completions" -ForegroundColor Cyan
    Write-Host "Docs: http://localhost:8000/docs`n" -ForegroundColor Cyan
    Invoke-DockerServe
}

# --- Pipeline Status ----------------------------------------------------------

function Show-PipelineStatus {
    $s1 = Test-Path (Join-Path $SCRIPT_DIR "models\base\llama3.1-8b\config.json")
    $s2 = Test-Path (Join-Path $SCRIPT_DIR "models\merged\config.json")
    $s3 = Test-Path (Join-Path $SCRIPT_DIR "models\openvino\openvino_config.json")

    Write-Host "`n  Pipeline status:" -ForegroundColor White
    $s1s = if ($s1) { "[DONE]" } else { "[     ]" }
    $s2s = if ($s2) { "[DONE]" } else { "[     ]" }
    $s3s = if ($s3) { "[DONE]" } else { "[     ]" }
    $c1  = if ($s1) { "Green"  } else { "DarkGray" }
    $c2  = if ($s2) { "Green"  } else { "DarkGray" }
    $c3  = if ($s3) { "Green"  } else { "DarkGray" }
    Write-Host "    Step 1 Download  $s1s   models/base/llama3.1-8b/" -ForegroundColor $c1
    Write-Host "    Step 2 Merge     $s2s   models/merged/"           -ForegroundColor $c2
    Write-Host "    Step 3 Convert   $s3s   models/openvino/"          -ForegroundColor $c3

    $baseSize   = Get-DiskUsage (Join-Path $SCRIPT_DIR "models\base")
    $mergedSize = Get-DiskUsage (Join-Path $SCRIPT_DIR "models\merged")
    $ovSize     = Get-DiskUsage (Join-Path $SCRIPT_DIR "models\openvino")
    Write-Host "`n  Disk usage:" -ForegroundColor White
    Write-Host "    models/base/    $baseSize"
    Write-Host "    models/merged/   $mergedSize"
    Write-Host "    models/openvino/ $ovSize"
}

# --- Show Cleanup Menu ---------------------------------------------------------

function Show-CleanupMenu {
    Write-Host "`n--- Delete old models (free disk space) ---" -ForegroundColor Yellow
    Write-Host "  1) Delete base model (llama3.1-8b + lora)   [frees ~16GB]" -ForegroundColor White
    Write-Host "  2) Delete merged model                       [frees ~16GB]" -ForegroundColor White
    Write-Host "  3) Delete HuggingFace cache                  [frees varies]" -ForegroundColor White
    Write-Host "  4) Delete all (keep openvino only)           [frees ~32GB]" -ForegroundColor White
    Write-Host "  5) Cancel" -ForegroundColor White
    $c = Read-Host "Select"
    switch ($c) {
        "1" { Remove-ModelDir (Join-Path $SCRIPT_DIR "models\base") "base model" }
        "2" { Remove-ModelDir (Join-Path $SCRIPT_DIR "models\merged") "merged model" }
        "3" { Remove-ModelDir (Join-Path $SCRIPT_DIR ".cache\huggingface") "HF cache" }
        "4" {
            Remove-ModelDir (Join-Path $SCRIPT_DIR "models\base") "base model"
            Remove-ModelDir (Join-Path $SCRIPT_DIR "models\merged") "merged model"
            Remove-ModelDir (Join-Path $SCRIPT_DIR ".cache\huggingface") "HF cache"
        }
    }
}

# --- Full Pipeline -------------------------------------------------------------

function Run-FullPipeline {
    param([string]$Token, [bool]$Clean)

    Write-Step "Full Pipeline"
    if ($Clean) {
        Write-Host "Cleanup mode ON  --  old models deleted after each step`n" -ForegroundColor Yellow
    }

    $s1done = Test-Path (Join-Path $SCRIPT_DIR "models\base\llama3.1-8b\config.json")
    $s2done = Test-Path (Join-Path $SCRIPT_DIR "models\merged\config.json")
    $s3done = Test-Path (Join-Path $SCRIPT_DIR "models\openvino\openvino_config.json")

    if ($s3done) {
        Write-Host "All steps already complete! models/openvino/ is ready." -ForegroundColor Green
        $buildRuntime = Read-Host "Build slim runtime image (fingpt:runtime) now? (y/n)"
        if ($buildRuntime -eq "y") { Build-RuntimeImage }
        Write-Host "Run inference: .\run.ps1 -Inference" -ForegroundColor Cyan
        Write-Host "Start server:  .\run.ps1 -Server" -ForegroundColor Cyan
        return
    }

    if (-not (docker image inspect fingpt:latest -f "{{.Id}}" 2>$null)) {
        Build-DockerImage
    }

    if (-not $s1done) { Step-DownloadModels -Token $Token }
    if (-not $s2done) { Step-MergeLora }
    Step-ConvertOpenVINO

    Write-Step "Pipeline complete!"
    Write-Host "Run inference: .\run.ps1 -Inference" -ForegroundColor Cyan
    Write-Host "Start server:  .\run.ps1 -Server" -ForegroundColor Cyan
    Write-Host "Free disk:     .\run.ps1 (option 9)" -ForegroundColor Cyan

    $buildRuntime = Read-Host "`nBuild slim runtime image (fingpt:runtime) now? (y/n)"
    if ($buildRuntime -eq "y") { Build-RuntimeImage }
}

function Build-RuntimeImage {
    Write-Step "Building slim runtime image (fingpt:runtime)"
    $ovPath = Join-Path $SCRIPT_DIR "models\openvino\openvino_config.json"
    if (-not (Test-Path $ovPath)) {
        Write-Fail "models/openvino/ not found. Run Step 3 first."
        return
    }
    Write-Host "Embedding models/openvino/ into the image (~1-2 GB)`n" -ForegroundColor Yellow
    docker build --target runtime -t fingpt:runtime . 2>&1 | Tee-Object -Variable output
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "Runtime image build failed."
        Write-Host $output
        exit 1
    }
    Write-Success "fingpt:runtime built successfully"
    Write-Host "`nRun server (no volume mount needed):" -ForegroundColor Cyan
    Write-Host "  docker run --rm --device=/dev/dri:/dev/dri -p 8000:8000 fingpt:runtime python server.py" -ForegroundColor White
}

# --- Interactive Menu ---------------------------------------------------------

function Show-Menu {
    $token = Get-HuggingFaceToken

    Show-PipelineStatus

    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  FinAI Docker Runner Menu" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor White
    Write-Host "  1) Build Docker image                (one-time)" -ForegroundColor White
    Write-Host "  2) Full pipeline                     (auto-skip done steps)" -ForegroundColor White
    Write-Host "  3) Full pipeline + cleanup           (free disk after each step)" -ForegroundColor White
    Write-Host "  4) Download models                   (Step 1)" -ForegroundColor White
    Write-Host "  5) Merge LoRA                       (Step 2)" -ForegroundColor White
    Write-Host "  6) Convert to OpenVINO              (Step 3)" -ForegroundColor White
    Write-Host "  7) Run inference                    (Step 4)" -ForegroundColor White
    Write-Host "  8) Start API server                (port 8000)" -ForegroundColor White
    Write-Host "  9) Delete old models                (free disk)" -ForegroundColor White
    Write-Host "  Q) Quit" -ForegroundColor White
    Write-Host ""
    $choice = Read-Host "Select option"

    switch ($choice) {
        "1"  { Build-DockerImage; Show-Menu }
        "2"  { Run-FullPipeline -Token $token -Clean $false; Show-Menu }
        "3"  { Run-FullPipeline -Token $token -Clean $true; Show-Menu }
        "4"  { Step-DownloadModels -Token $token; Show-Menu }
        "5"  { Step-MergeLora; Show-Menu }
        "6"  { Step-ConvertOpenVINO; Show-Menu }
        "7"  { Step-RunInference; Show-Menu }
        "8"  { Step-RunServer }
        "9"  { Show-CleanupMenu; Show-Menu }
        "Q"  { exit 0 }
        "q"  { exit 0 }
        default { Write-Host "Invalid." -ForegroundColor Red; Show-Menu }
    }
}

# --- Main --------------------------------------------------------------------

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  FinAI Docker Runner" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-DockerInstalled)) {
    Write-Host "`n[INFO] Docker not detected." -ForegroundColor Yellow
    $confirm = Read-Host "Install Docker Desktop now? (y/n)"
    if ($confirm -ne "y") { exit 0 }
    Install-DockerDesktop
}
Start-DockerIfNeeded

Test-EnvFile

if ($Build) {
    Build-DockerImage
} elseif ($FullPipeline) {
    $token = Get-HuggingFaceToken
    Run-FullPipeline -Token $token -Clean $CleanUp.IsPresent
} elseif ($Step) {
    $token = Get-HuggingFaceToken
    switch ($Step) {
        "1" { Step-DownloadModels -Token $token }
        "2" { Step-MergeLora }
        "3" { Step-ConvertOpenVINO }
        "4" { Step-RunInference }
    }
} elseif ($Inference) {
    Step-RunInference
} elseif ($Server) {
    Step-RunServer
} else {
    Show-Menu
}
