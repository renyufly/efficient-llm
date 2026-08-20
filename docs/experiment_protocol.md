# Frozen Experiment Protocol

The project answers three orthogonal questions:

1. Base vs LoRA vs QLoRA with fixed data, prompt, evaluation, and final inference dtype.
2. BF16/FP16 vs one deployment INT4 built from the same canonical merged checkpoint.
3. Hugging Face vs vLLM at concurrency 1, followed by vLLM concurrency scaling.

Frozen identifiers are Qwen/Qwen3-1.7B-Base and Qwen/Qwen3-1.7B; data seed 2026;
training seeds 42 and 1234; LoRA ranks 8/16 with alpha/rank 2 and dropout 0.05;
effective batch 32; at most three epochs; LR 2e-4 with cosine schedule and 0.03
warmup. Initial sequence length is 512 and changes globally to 768 only if the
Phase 1 token report crosses the frozen 5% threshold.

Quality generation is greedy with at most 256 new tokens and respects EOS. Performance
generation uses exactly 512 input and 128 output tokens, ignores EOS, and tests
concurrency 1/4/8/16 with 100 requests, 10 warm-ups, and three repetitions.

Test and SVAMP never select checkpoints. Formal results require a resolved config,
config hash, immutable dataset/model revisions, git and hardware metadata, and raw
per-sample or per-request records. Paid GPU use always requires explicit user approval.

