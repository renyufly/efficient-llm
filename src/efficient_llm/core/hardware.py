"""CPU-safe environment inspection; NVIDIA probing is optional and defensive."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psutil

from efficient_llm.core.paths import project_root


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args], cwd=project_root(), capture_output=True, check=True,
                text=True, timeout=5,
            )
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _gpu_metadata() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}
    fields = "uuid,name,driver_version,memory.total,power.limit"
    try:
        summary = subprocess.run(
            [executable], capture_output=True, check=False, text=True, timeout=10
        )
        query = subprocess.run(
            [executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, check=True, text=True, timeout=10,
        )
        processes = subprocess.run(
            [executable, "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, check=False, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "reason": f"nvidia-smi probe failed: {exc}"}
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", summary.stdout)
    gpus = []
    for line in query.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 5:
            keys = ["uuid", "name", "driver_version", "memory_mib", "power_w"]
            gpus.append(dict(zip(keys, values, strict=True)))
    return {
        "available": bool(gpus),
        "cuda_version": cuda_match.group(1) if cuda_match else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "devices": gpus,
        "active_compute_processes": [
            line for line in processes.stdout.splitlines() if line.strip()
        ],
    }


def collect_environment() -> dict[str, Any]:
    root = project_root()
    disk = psutil.disk_usage(str(root))
    memory = psutil.virtual_memory()
    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "git": _git_metadata(),
        "cpu": {
            "logical_count": psutil.cpu_count(),
            "physical_count": psutil.cpu_count(logical=False),
            "processor": platform.processor(),
        },
        "ram": {"total_bytes": memory.total, "available_bytes": memory.available},
        "disk": {
            "path": str(Path(root).anchor),
            "total_bytes": disk.total,
            "free_bytes": disk.free,
        },
        "gpu": _gpu_metadata(),
    }
