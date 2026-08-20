# Efficient LLM Post-training & Inference Systems

A reproducible, single-RTX-4090 research pipeline for studying quality, memory,
latency, throughput, and cost trade-offs across LoRA/QLoRA, deployment INT4, and
Hugging Face/vLLM inference.

Phase 0 provides the CPU-only engineering and reproducibility foundation. Later
phases must follow `plan.md`; no command creates paid resources.

## Local setup (project-isolated)

The checked-in `uv.toml` keeps uv downloads under `.tools/` and `.cache/`.

```powershell
./scripts/setup_local.ps1
uv run --no-sync efficient-llm --help
uv run --no-sync pytest -m "not gpu"
```

Use `scripts/run_cpu_smoke.ps1` for the complete Phase 0 verification. Generated
experiment outputs are append-only and gitignored by default.
