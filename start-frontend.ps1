# Install deps (first run) and start the Vite dev server.
# Idempotent: safe to run repeatedly.

$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\frontend"

if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend dependencies..."
    npm install
}

# Fetch MediaPipe .task files into frontend/public/models/ if missing.
node ./scripts/download_mediapipe_models.mjs

npm run dev
