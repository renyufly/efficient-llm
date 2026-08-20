# Implementation Progress

## 2026-08-20 — Phase 0: Repository Foundation and Reproducibility Gate

**Status:** Complete  
**Compute used:** Local CPU only; 0 GPU hours; no cloud or paid resources created.

### Execution basis

- Followed `plan.md` Phase 0 (lines 495–536) as the execution specification.
- Used `Idea.md` only as project background; frozen decisions and directory/interface
  choices follow `plan.md` when the two differ.
- Starting repository contained only `Idea.md`, `plan.md`, and `LICENSE`; no unrelated
  user changes were present. Starting commit: `26bfec1`.

### Environment isolation

- Used local `uv 0.9.22` exclusively for Python setup and package installation.
- Installed CPython 3.11.14 under project-local `.tools/python`.
- Created the virtual environment under project-local `.venv`.
- Kept uv downloads under project-local `.cache/uv`.
- Added `.python-version`, `uv.toml`, and `scripts/setup_local.ps1` so setup is repeatable.
- GPU dependencies remain separated in `requirements/gpu-train.txt` and
  `requirements/gpu-serve.txt`; CPU imports do not load vLLM or bitsandbytes.

Compatibility note: uv 0.9.22 accepts `cache-dir` in `uv.toml`, but does not accept a
`python-install-dir` TOML key. The Python install directory is therefore set explicitly
to `.tools/python` through `UV_PYTHON_INSTALL_DIR` in every setup/run script. Verification
commands use `uv run --no-sync` so test execution never performs an implicit dependency
update.

### Implemented

1. Created the `src/` package layout, planned config/requirements/result/document
   directories, editable package metadata, CLI entry point, README, and gitignore rules.
2. Added YAML loading, deterministic `${...}` interpolation, Pydantic schemas, canonical
   SHA-256 config hashes, and validation for schema version, model revision, allowed
   ranks, alpha/rank ratio, exact effective batch size, packing report, and distinct
   quality/performance EOS policies.
3. Added a LoRA/QLoRA comparison validator that permits only the frozen loading/
   quantization and run-label differences and rejects unauthorized experiment drift.
4. Added deterministic run IDs, random seeding with optional NumPy/PyTorch lazy support,
   output-directory collision protection, and atomic text/JSON writes.
5. Implemented `doctor` metadata for OS, Python, git commit/dirty state, CPU, RAM, disk,
   and defensive NVIDIA discovery (UUID, driver, CUDA version, memory, power limit, and
   active compute processes when available).
6. Added the complete planned CLI tree. Phase 0 supports help, doctor, and config
   dry-runs; later-phase commands exit 4 with an explicit unimplemented message instead
   of reporting false success.
7. Added pytest unit/integration markers, ruff, strict mypy, a PowerShell CPU smoke entry,
   GPU script placeholders that fail explicitly, architecture/protocol/runbook/findings
   documentation, and a sanitized environment-report example.

### Acceptance evidence

Executed from `D:\Code\efficient-llm` using the project-local uv environment:

```text
./scripts/run_cpu_smoke.ps1
  ruff:  All checks passed
  mypy:  Success; 18 source files checked
  pytest -m "not gpu": 15 passed in 0.66s

CLI config dry-runs: 13/13 passed
doctor report: results/raw/phase0_environment.json
doctor Python: 3.11.14
doctor GPU available: false
second doctor write to same path: exit 2; original report preserved
```

Tests explicitly cover invalid rank, invalid alpha ratio, invalid batch relation, missing
model revision, quality EOS misuse, LoRA/QLoRA drift, stable hashing/interpolation,
atomic writes, non-empty output collision, no-NVIDIA doctor behavior, CLI dry-run, and
later-phase placeholder failure.

### Progress

| Phase | State | Notes |
|---|---|---|
| Phase 0 | Complete | All specified CPU acceptance checks passed |
| Phase 1 | Not started | Must begin with fixed dataset revisions and manifests |
| Phase 2–12 | Not started | Preserved in dependency order from `plan.md` |

No frozen experiment variable was changed. Config revisions currently use the explicit
`main` dry-run value and must be replaced by immutable model/dataset revisions before any
formal later-phase run, as required by the protocol.

