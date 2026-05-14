# Install deps (first run) and start the FastAPI backend.
# Idempotent: safe to run repeatedly.

$ErrorActionPreference = "Stop"

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "python not found on PATH. Install Python 3.10+ first."
    exit 1
}

if (-not (Test-Path "$PSScriptRoot\backend\venv")) {
    Write-Host "Creating backend venv..."
    python -m venv "$PSScriptRoot\backend\venv"
}

& "$PSScriptRoot\backend\venv\Scripts\Activate.ps1"
pip install -q -r "$PSScriptRoot\backend\requirements.txt"

# Warn (but don't fail) if the CTC ONNX is missing — the live-letter
# WebSocket still works without it, but /api/transcribe-landmarks will
# 500 on the first call.
$ctc = "$PSScriptRoot\backend\models\artifacts\asl_ctc.onnx"
if (-not (Test-Path $ctc)) {
    Write-Warning "CTC model not found at $ctc"
    Write-Warning "The phrase-transcribe path won't work until you copy it from openhand-model/exports/ctc/."
}

# Run from the repo root so the package import path resolves.
Set-Location $PSScriptRoot
uvicorn backend.main:app --host 0.0.0.0 --port 8273 --reload
