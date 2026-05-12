# Install deps (first run) and start the FastAPI server
Set-Location "$PSScriptRoot\backend"

if (-not (Get-Command pip -ErrorAction SilentlyContinue)) {
    Write-Error "pip not found. Install Python 3.10+ first."
    exit 1
}

if (-not (Test-Path "$PSScriptRoot\backend\venv")) {
    python -m venv "$PSScriptRoot\backend\venv"
}

& "$PSScriptRoot\backend\venv\Scripts\Activate.ps1"
pip install -q -r requirements.txt
Set-Location "$PSScriptRoot"
uvicorn backend.main:app --host 0.0.0.0 --port 8273 --reload
