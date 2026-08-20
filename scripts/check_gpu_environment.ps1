$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $env:UV_CACHE_DIR = Join-Path $projectRoot ".cache\uv"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot ".tools\python"
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot ".venv"
    uv run --no-sync efficient-llm doctor --output results/raw/environment.json
} finally {
    Pop-Location
}
