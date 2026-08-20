$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $env:UV_CACHE_DIR = Join-Path $projectRoot ".cache\uv"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot ".tools\python"
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot ".venv"
    uv run --no-sync ruff check .
    uv run --no-sync mypy
    uv run --no-sync pytest -m "not gpu"
    uv run --no-sync efficient-llm --help | Out-Null
    uv run --no-sync efficient-llm train --config configs/train/lora_r8.yaml --dry-run | Out-Null
} finally {
    Pop-Location
}
