#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export UV_CACHE_DIR="$project_root/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$project_root/.tools/python"
export UV_PROJECT_ENVIRONMENT="$project_root/.venv"
uv run --no-sync efficient-llm doctor --output results/raw/environment.json
