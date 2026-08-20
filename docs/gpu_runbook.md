# GPU Runbook

Phase 0 does not install GPU libraries or start paid resources. On the approved Linux
RTX 4090 host, explicitly point caches and outputs at the persistent project SSD before
running the Phase 4 smoke gate. Freeze library versions only after that compatibility
test succeeds.

