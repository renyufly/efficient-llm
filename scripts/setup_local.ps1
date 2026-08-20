$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $env:UV_CACHE_DIR = Join-Path $projectRoot ".cache\uv"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot ".tools\python"
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $projectRoot ".venv"
    uv python install 3.11
    uv venv --python 3.11
    uv pip install -r requirements/dev.txt
} finally {
    Pop-Location
}

