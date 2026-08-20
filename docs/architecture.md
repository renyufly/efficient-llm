# Architecture

The package follows a dependency-light `src/` layout. `core` owns configuration,
schemas, project paths, reproducibility, and hardware metadata. It imports no model
training or serving library. Later modules depend inward on these contracts:

```text
core <- data <- evaluation <- training
  ^                         <- quantization
  +---- serving <- benchmark <- analysis
```

Data manifests are the only allowed boundary between raw preparation and experiments.
Training selects checkpoints from validation only. Quantization consumes one canonical
merged checkpoint. Benchmarking writes request events; aggregation is a separate,
rebuildable step.

All configured paths are project-relative unless one of the documented
`EFFICIENT_LLM_*_ROOT` variables is explicitly set. Raw experiment outputs are
append-only. GPU dependencies are isolated in their requirement files and must be
imported lazily in their owning modules.

