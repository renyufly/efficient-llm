# Efficient LLM Post-training & Inference Systems — Implementation Roadmap

> 本文件是 Codex 的执行规格，不是项目介绍。后续实现必须按 Phase 的依赖顺序推进；每完成一个 Phase，先运行该阶段的验收命令并保存证据，再进入下一阶段。若实际工具版本导致接口变化，只允许做局部兼容调整，不得改变已冻结的实验变量和比较关系。

## 1. 项目目标、边界与完成定义

### 1.1 核心研究问题

在一张低成本 RTX 4090 24GB 上，如何通过 LoRA/QLoRA 后训练、部署量化和推理引擎优化，在以下指标之间取得可解释的权衡：

- 质量：GSM8K / SVAMP 数值 Exact Match、解析率、格式合规率、通用能力回归。
- 内存：训练 Peak VRAM、权重磁盘大小、推理 Peak VRAM、理论 KV Cache。
- 性能：训练耗时、TTFT、TPOT、E2E Latency、吞吐量、成功率。
- 成本：GPU 小时、失败实验成本、可复现性。

项目产出必须形成三组正交、可归因的实验，而不是把工具堆在一起：

1. 后训练：Base vs LoRA vs QLoRA；固定数据、Prompt、评测协议和最终推理精度。
2. 部署量化：同一最佳 merged checkpoint 的 BF16/FP16 vs 一种 INT4；固定引擎和 workload。
3. 推理引擎：同一 checkpoint、同一精度的 Hugging Face vs vLLM；先做并发 1 等价对照，再做 vLLM 并发扩展。

### 1.2 范围分层

#### Core / MVP（必须完成）

- 可测试的 Mini Transformer 学习模块，但它不阻塞主实验。
- GSM8K/SVAMP 数据处理、固定划分、manifest、token 长度分析。
- 统一的生成、答案解析、质量评测和逐样本结果格式。
- Qwen3-1.7B Base 与同尺寸 Instruct reference baseline。
- LoRA rank 8/16；LoRA 与 QLoRA 公平对照；关键配置 seed 42/1234。
- 最佳 adapter 合并、合并前后一致性检查。
- BF16/FP16 与一种部署 INT4 的质量、大小、显存对照。
- KV Cache calculator。
- Hugging Face 与 vLLM serving；固定 token workload 下的并发 1/4/8/16 benchmark。
- 汇总表、图、实验结论、复现说明。

#### Recommended Extensions（Core MVP 后，优先级从高到低）

- 输入长度 128/512/1024、输出长度 32/128 的性能矩阵。
- Request-rate/RPS sweep、P99。
- LoRA rank 32（仅当 rank 8/16 的趋势值得验证）。
- bitsandbytes INT8。
- PyTorch Profiler 与更细的 prefill/decode 分析。
- 更完整的错误分类和回归测试。

#### Bonus / Future Work（不得阻塞 Core）

- Algorithm：DPO、合成数据、reasoning SFT、advanced evaluation。
- Systems：FlashAttention 原理实验、Triton、CUDA kernel。
- Distributed：DDP/FSDP/ZeRO 概念与小规模验证。
- Quantization：GPTQ/AWQ/GGUF 多格式横评。
- Serving：SGLang/TensorRT-LLM/llama.cpp、speculative decoding、KV Cache quantization、prefix caching。
- 明确不做：大规模预训练、7B+ full fine-tuning、多机多卡、Megatron-LM、RLHF/PPO、复杂自研 kernel、TP/PP、MoE。

### 1.3 Core 完成定义

Core 只有在以下条件全部满足时才算完成：

- 每项正式结果可追溯到 config、git commit、数据 manifest、模型 revision、环境和硬件元数据。
- 三组正交实验均有原始逐样本/逐请求记录、聚合指标和分析，不以“脚本运行成功”代替实验结论。
- 本地 CPU 测试全部通过；正式 GPU 数据来自同型号 RTX 4090，性能数据尽量在同一实例租期完成。
- 所有付费 GPU 操作均需用户明确授权；本计划本身不授权租用、启动或计费。
- README、`docs/findings.md` 能解释实验为何这样设计、数据说明什么、异常为何发生、下一步验证什么。

## 2. 冻结决策与合理调整

### 2.1 冻结协议

| 项目 | 决策 |
|---|---|
| Base | `Qwen/Qwen3-1.7B-Base` |
| Instruct reference | `Qwen/Qwen3-1.7B`，使用官方 chat template，仅作参考 |
| 任务 | 英文小学数学短 CoT，结尾为 `Final answer:` |
| 训练数据 | GSM8K train 清洗后固定 2,000 条 |
| Validation | GSM8K train 清洗后固定 256 条，与训练集不重叠 |
| Smoke train | 固定 256 条 |
| 快速测试 | GSM8K official test 固定 500 条 |
| 最终测试 | GSM8K official test 1,319 条 |
| OOD | SVAMP full set，禁止用于调参/选 checkpoint |
| 回归集 | ARC-Easy 200 + HellaSwag 200，zero-shot log-likelihood multiple choice |
| Data seed | 2026 |
| Train seeds | 初筛 42；关键 LoRA vs QLoRA 复跑 1234 |
| LoRA | rank 8/16，alpha/rank=2，dropout=0.05，bias=none，all-linear，排除 embedding/lm_head |
| 训练 | 最多 3 epochs，LR 2e-4，cosine，warmup 0.03，effective batch 32，gradient checkpointing |
| 长度 | 初始 512；若有效样本中超过 512 的比例 >5%，统一改为 768 |
| 质量生成 | greedy、do_sample=false、max_new_tokens=256、允许 EOS |
| 性能 workload | input 512、output 128、ignore EOS、并发 1/4/8/16、每点 100 请求、10 warm-up、3 repetitions |
| 正式硬件 | 独占 RTX 4090 24GB；48GB RAM；缓存与产物放 200GB SSD |
| INT4 | GPTQ/W4A16 + Marlin 优先；兼容性失败才整体回退 AWQ |

### 2.2 对 Idea.md 的实施性调整

1. **Mini Transformer 与主线解耦。** 它用于验证 attention、causal loss、generation 和 KV Cache 概念，可与数据/评测基础设施并行完成；不因生成质量不佳延迟真实模型实验。
2. **使用单一 Python package 和薄 CLI。** 不分别维护 `lora.py`/`qlora.py` 的重复训练逻辑；由同一 trainer 读取 `method: lora|qlora`，减少公平性对照中的代码漂移。
3. **Transformers + PEFT + TRL 是可复现基线，Unsloth 是可选加速适配器。** 先保证标准栈可运行；只有 Unsloth 在冻结版本和目标硬件上通过一致性 smoke test 后才启用，且所有严格对照使用同一后端。
4. **`requirements.txt` 拆为 CPU 与 GPU extras。** vLLM、bitsandbytes、量化工具常受 CUDA/平台约束，不能阻塞 Windows/CPU 本地开发。正式版本锁在 GPU smoke test 后生成 lock/环境快照。
5. **部署 INT4 不并行维护 GPTQ 与 AWQ 两套正式结果。** 先执行兼容性 probe，记录失败证据；成功则 GPTQ，失败则 AWQ。这样控制 GPU 成本和比较变量。
6. **SFT 不单列为第三种训练方法。** 本项目的 SFT 通过 LoRA/QLoRA 参数高效实现；“Base vs LoRA vs QLoRA”已经覆盖后训练问题，避免引入高成本 full SFT。
7. **质量评测与性能评测使用不同生成模式并严格隔离。** 质量允许 EOS；性能必须固定 token 数并忽略 EOS。两类结果不得写入同一统计口径。

## 3. 总体架构与依赖关系

### 3.1 数据与产物流

```text
raw datasets + revisions
          |
          v
prepare/validate/tokenize ----> data manifest + fixed IDs + hashes
          |
          +------------------------+
          |                        |
          v                        v
baseline evaluation          LoRA / QLoRA training
                                   |
                                   v
                         validation checkpoint selection
                                   |
                                   v
                         adapter/merged consistency
                                   |
                                   v
                         canonical merged BF16 checkpoint
                                   |
                         +---------+---------+
                         |                   |
                         v                   v
                      BF16/FP16          INT4 quantize
                         |                   |
                         +---------+---------+
                                   |
                    quality eval + HF/vLLM serving
                                   |
                                   v
                         raw benchmark events
                                   |
                                   v
                      aggregate + plots + findings
```

依赖原则：

- `core` 只定义配置、schema、ID、hash、路径和公共工具，不依赖训练/serving 库。
- `data` 依赖 `core`；其输出是其他模块唯一允许读取的数据入口。
- `evaluation` 依赖 `core` 和 `data`，不依赖 trainer；任何模型后端通过统一 generator/scorer 接口接入。
- `training` 依赖 `core`、`data`、`evaluation`；checkpoint 选择只能读取 validation 指标。
- `quantization` 只接受已选中的 canonical merged checkpoint。
- `serving` 提供启动配置和 OpenAI-compatible client，不包含 benchmark 统计逻辑。
- `benchmark` 只消费生成后端或 HTTP endpoint，并写原始事件；`analysis` 再聚合，防止边测边改统计口径。

### 3.2 目标目录结构

```text
efficient-llm/
├── Idea.md
├── plan.md
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements/
│   ├── cpu.txt
│   ├── gpu-train.txt
│   ├── gpu-serve.txt
│   └── dev.txt
├── configs/
│   ├── base.yaml
│   ├── data/gsm8k.yaml
│   ├── eval/math.yaml
│   ├── eval/regression.yaml
│   ├── train/lora_r8.yaml
│   ├── train/lora_r16.yaml
│   ├── train/qlora_r8.yaml
│   ├── train/qlora_r16.yaml
│   ├── quantize/int4.yaml
│   ├── serve/hf.yaml
│   ├── serve/vllm.yaml
│   └── benchmark/core.yaml
├── src/efficient_llm/
│   ├── cli.py
│   ├── core/
│   │   ├── config.py
│   │   ├── schemas.py
│   │   ├── paths.py
│   │   ├── reproducibility.py
│   │   └── hardware.py
│   ├── data/
│   │   ├── prepare.py
│   │   ├── templates.py
│   │   ├── tokenize.py
│   │   └── validate.py
│   ├── mini_llm/
│   │   ├── config.py
│   │   ├── model.py
│   │   ├── attention.py
│   │   ├── rope.py
│   │   ├── train.py
│   │   └── generate.py
│   ├── evaluation/
│   │   ├── answers.py
│   │   ├── generation.py
│   │   ├── math_eval.py
│   │   ├── regression_eval.py
│   │   ├── bootstrap.py
│   │   └── aggregate.py
│   ├── training/
│   │   ├── model_loader.py
│   │   ├── collator.py
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   ├── selection.py
│   │   └── merge.py
│   ├── quantization/
│   │   ├── compatibility.py
│   │   ├── gptq.py
│   │   ├── awq.py
│   │   └── validate.py
│   ├── serving/
│   │   ├── hf_backend.py
│   │   ├── vllm_launcher.py
│   │   └── openai_client.py
│   ├── inference/
│   │   └── kv_cache.py
│   ├── benchmark/
│   │   ├── workload.py
│   │   ├── online.py
│   │   ├── metrics.py
│   │   ├── gpu_monitor.py
│   │   └── aggregate.py
│   └── analysis/
│       ├── tables.py
│       └── plots.py
├── scripts/
│   ├── check_gpu_environment.ps1
│   ├── check_gpu_environment.sh
│   ├── run_cpu_smoke.ps1
│   ├── run_gpu_smoke.sh
│   └── run_core_benchmark.sh
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/
│   ├── raw/                 # gitignored
│   ├── processed/           # 默认 gitignored；可发布小型 manifest/IDs
│   └── manifests/
├── artifacts/               # adapters/merged/quantized，gitignored
├── results/
│   ├── raw/
│   ├── summaries/
│   ├── predictions/
│   └── figures/
└── docs/
    ├── architecture.md
    ├── experiment_protocol.md
    ├── gpu_runbook.md
    ├── findings.md
    └── interview_notes.md
```

所有目录通过配置中的项目根路径和环境变量解析，禁止在代码中硬编码个人绝对路径。云端建议设置：

```text
EFFICIENT_LLM_DATA_ROOT=/mnt/ssd/efficient-llm/data
EFFICIENT_LLM_ARTIFACT_ROOT=/mnt/ssd/efficient-llm/artifacts
EFFICIENT_LLM_RESULT_ROOT=/mnt/ssd/efficient-llm/results
HF_HOME=/mnt/ssd/cache/huggingface
TORCH_HOME=/mnt/ssd/cache/torch
```

不得自动改写用户环境；runbook 仅给出显式导出命令。

## 4. 公共接口、配置与数据契约

### 4.1 统一 CLI

安装后提供 `efficient-llm`，同时允许 `python -m efficient_llm.cli`。Core 命令固定为：

```bash
efficient-llm doctor --output results/raw/environment.json
efficient-llm data prepare --config configs/data/gsm8k.yaml
efficient-llm data inspect --manifest data/manifests/gsm8k.json
efficient-llm mini train --config configs/mini.yaml
efficient-llm evaluate math --config configs/eval/math.yaml --model <model-ref>
efficient-llm evaluate regression --config configs/eval/regression.yaml --model <model-ref>
efficient-llm train --config configs/train/lora_r8.yaml
efficient-llm select --runs <run-dir...> --output results/summaries/selection.json
efficient-llm merge --adapter <path> --output <path>
efficient-llm quantize probe --config configs/quantize/int4.yaml
efficient-llm quantize run --config configs/quantize/int4.yaml
efficient-llm serve hf --config configs/serve/hf.yaml
efficient-llm serve vllm --config configs/serve/vllm.yaml
efficient-llm kv-cache --model-config <config.json> --seq-len 512 --batch-size 1 --dtype bf16
efficient-llm benchmark online --config configs/benchmark/core.yaml --endpoint <url>
efficient-llm results aggregate --results results/raw
efficient-llm results plot --summaries results/summaries
```

每个会执行计算的命令都支持：

- `--dry-run`：解析并打印完全展开的配置、输入和输出路径，不加载大模型。
- `--run-id`：可显式指定；默认由 UTC 时间、方法、seed、短 config hash 生成。
- `--resume`：只恢复同 config hash 的未完成 run；不允许静默覆盖。
- `--limit N`：CPU/GPU smoke test 限制样本数。
- 退出码：0 成功；2 配置/数据错误；3 环境或硬件不满足；4 模型/后端错误；5 验收指标失败。

### 4.2 配置模型

使用 YAML 供人编辑，加载后用强类型 dataclass/Pydantic 校验，保存为展开后的 `resolved_config.yaml`。禁止在运行时从多个隐式默认来源拼接关键实验参数。

```yaml
schema_version: 1
run:
  name: qlora-r16-seed42
  seed: 42
  output_dir: artifacts/runs/${run.name}
  overwrite: false
model:
  id: Qwen/Qwen3-1.7B-Base
  revision: <frozen-commit-or-tag>
  trust_remote_code: false
  dtype: bfloat16
  attn_implementation: sdpa
data:
  manifest: data/manifests/gsm8k.json
  train_split: train_2000
  validation_split: validation_256
  max_seq_length: 512
  assistant_only_loss: true
  packing: false
training:
  method: qlora              # lora | qlora
  backend: transformers      # transformers | unsloth
  epochs: 3
  learning_rate: 2.0e-4
  micro_batch_size: 1        # GPU smoke test 后确定
  gradient_accumulation_steps: 32
  effective_batch_size: 32
  optimizer: paged_adamw_8bit
  scheduler: cosine
  warmup_ratio: 0.03
  weight_decay: 0.0
  gradient_checkpointing: true
  compute_dtype: bfloat16
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  bias: none
  target_modules: all-linear
  exclude_modules: [embed_tokens, lm_head]
quantization:
  load_in_4bit: true
  quant_type: nf4
  double_quant: true
selection:
  metric: normalized_numeric_exact_match
  tie_breakers: [parse_rate, earlier_epoch]
```

配置加载时必须验证：

- `effective_batch_size == micro_batch_size * gradient_accumulation_steps * world_size`。
- `alpha / rank == 2`；rank 只能是冻结实验允许值。
- LoRA 与 QLoRA 对照除加载精度和必要量化字段外，公平性字段完全相同；提供 config diff 校验器。
- 正式 run 的 model revision、dataset revision 不能为空。
- `packing=true` 只有存在已通过的 packing correctness report 才允许。
- 质量评测不能设置 `ignore_eos=true`；性能评测必须设置固定输入/输出 token 数。

### 4.3 数据集 manifest 与样本格式

`data/manifests/gsm8k.json` 是不可变输入契约：

```json
{
  "schema_version": 1,
  "dataset": "openai/gsm8k",
  "config": "main",
  "revision": "<revision>",
  "data_seed": 2026,
  "processor_version": "<git-commit>",
  "tokenizer": {"id": "Qwen/Qwen3-1.7B-Base", "revision": "<revision>"},
  "max_seq_length": 512,
  "splits": {
    "smoke_256": {"path": "...jsonl", "count": 256, "sha256": "..."},
    "train_2000": {"path": "...jsonl", "count": 2000, "sha256": "..."},
    "validation_256": {"path": "...jsonl", "count": 256, "sha256": "..."},
    "test_500": {"path": "...jsonl", "count": 500, "sha256": "..."},
    "test_full": {"path": "...jsonl", "count": 1319, "sha256": "..."},
    "svamp_full": {"path": "...jsonl", "count": 1000, "sha256": "..."}
  },
  "filter_stats": {},
  "length_stats": {}
}
```

统一 JSONL 样本：

```json
{
  "sample_id": "gsm8k:train:<source-id>",
  "source": "gsm8k",
  "source_split": "train",
  "question": "...",
  "reasoning": "...",
  "answer_raw": "...",
  "answer_normalized": "42",
  "prompt": "Problem:\n...",
  "response": "Reasoning:\n...\n\nFinal answer: 42",
  "text": "<prompt><response>",
  "prompt_token_count": 0,
  "response_token_count": 0,
  "total_token_count": 0
}
```

处理顺序固定：下载指定 revision → 保留 source ID → 去除 `<<expression=result>>` → 拆分 `####` → 规范模板 → 数值答案校验 → token 统计 → 记录并移除超长/无效样本 → 从合格 GSM8K train 中先划 validation，再抽 train/smoke。`smoke_256` 应是 `train_2000` 的固定子集；validation 必须不相交。禁止静默截断 response。

### 4.4 训练 run 产物

每个 run 目录固定包含：

```text
artifacts/runs/<run-id>/
├── resolved_config.yaml
├── metadata.json
├── checkpoints/epoch-{1,2,3}/
├── adapter/                    # selected adapter
├── trainer_state.json
├── metrics.jsonl               # 每 step/epoch 追加事件
├── validation_predictions.jsonl
├── memory_samples.csv
├── environment.json
└── COMPLETE                    # 仅所有写入和校验成功后原子创建
```

`metadata.json` 至少记录 git commit/dirty 状态、config hash、model/data revisions、GPU UUID/model/power limit、driver/CUDA/library versions、开始结束 UTC、阶段耗时和失败原因。正式结果若来自 dirty tree，报告中必须显式标识。

### 4.5 预测与质量汇总格式

逐样本 `predictions.jsonl`：

```json
{
  "run_id": "...",
  "model_ref": "...",
  "dataset": "gsm8k",
  "split": "test_full",
  "sample_id": "...",
  "prompt": "...",
  "raw_output": "...",
  "parsed_answer": "42",
  "gold_answer": "42",
  "is_correct": true,
  "parse_success": true,
  "format_compliant": true,
  "input_tokens": 128,
  "output_tokens": 37,
  "truncated": false,
  "generation_seconds": 0.0
}
```

答案规范化必须精确处理：千位逗号、货币符号、单位尾缀、正负号、整数/等价小数、分数和百分数；内部用 `Decimal`/`Fraction`，不得用二进制 float 做相等判断。多个 `Final answer:` 时取最后一个合法数值计算正确率，但 `format_compliant=false`。无可解析答案记错。

汇总 JSON 必含样本数、EM、bootstrap 95% CI、parse rate、format compliance、平均输出 tokens、truncation rate、失败计数和 config hash。Bootstrap 默认按样本有放回重采样并使用固定 seed，重复次数写入配置。

### 4.6 Benchmark 原始事件格式

每个请求一行 JSONL，不先聚合：

```json
{
  "benchmark_run_id": "...",
  "engine": "vllm",
  "model_ref": "...",
  "precision": "bf16",
  "concurrency": 8,
  "repetition": 1,
  "request_id": "...",
  "scheduled_at_ns": 0,
  "first_token_at_ns": 0,
  "completed_at_ns": 0,
  "input_tokens": 512,
  "output_tokens": 128,
  "success": true,
  "http_status": 200,
  "error": null
}
```

从客户端单调时钟计算：`TTFT = first_token - scheduled`；`E2E = completed - scheduled`；`TPOT = (E2E - TTFT)/(output_tokens-1)`。聚合输出 P50/P95、output tok/s、total tok/s、请求成功率；只有 128 tokens 完整输出且成功的请求进入延迟分位数，同时单独报告失败/短输出，禁止悄悄丢弃。

## 5. Phase / Milestone 执行计划

## Phase 0 — Repository Foundation and Reproducibility Gate

**Classification:** Core / MVP  
**Compute:** Local CPU，0 GPU  
**Dependencies:** 无

### Goal

建立可安装、可配置、可测试、不会覆盖实验结果的最小工程骨架，为后续 CPU/GPU 阶段提供统一契约。

### Tasks

1. 创建上述目录、`pyproject.toml`、分层 requirements、`.gitignore`、基础 README。
2. 实现配置加载/校验、run ID/config hash、项目路径解析、随机种子、原子写入。
3. 实现 `doctor`：采集 OS/Python/git/CPU/RAM/disk；存在 GPU 时额外采集 UUID、driver、CUDA、power limit 和活动进程。
4. 建立 CLI 框架及所有子命令占位；未实现命令应明确报错，不能假成功。
5. 配置 pytest、ruff（或同等轻量 lint）、类型检查；建立 unit/integration 标记。
6. 写 `docs/architecture.md`、`docs/experiment_protocol.md` 初版，记录冻结变量和正交实验。

### Core files / modules

`pyproject.toml`、`requirements/*`、`src/efficient_llm/cli.py`、`core/config.py`、`core/schemas.py`、`core/paths.py`、`core/reproducibility.py`、`core/hardware.py`、`tests/`。

### Key implementation choices

- Python 3.11 作为首选；若 GPU 库兼容性要求其他小版本，仅在 smoke test 后统一调整。
- 采用 `src/` layout，避免从仓库根目录误导入。
- 原始结果 append-only；聚合结果可重建。任何命令发现目标目录非空且无 `--resume` 时失败。
- CPU 栈不能 import vLLM/bitsandbytes；GPU 依赖只在相应函数中延迟导入。

### Deliverables

- 可 editable install 的 package。
- `efficient-llm --help`、`doctor`、`--dry-run`。
- CPU CI/本地测试入口和环境报告样例。

### Acceptance Criteria

- 全新 CPU 环境安装 `requirements/cpu.txt` 与 `requirements/dev.txt` 后，`pytest -m "not gpu"` 通过。
- 所有 YAML 均能 dry-run；非法 rank、batch 关系、缺失 revision 会失败并指出字段。
- 连续两次使用同一 run ID 不会覆盖第一次结果。
- `doctor` 在无 NVIDIA GPU 时正常返回 `gpu.available=false`，而不是崩溃。

---

## Phase 1 — Data Pipeline, Token Analysis, and Fixed Splits

**Classification:** Core / MVP  
**Compute:** Local CPU；需要网络下载时只下载数据/tokenizer，不租 GPU  
**Dependencies:** Phase 0

### Goal

产生冻结、可审计、无泄漏的数据集输入，并据真实 token 分布决定全项目唯一的 `max_seq_length`。

### Tasks

1. 下载指定 revision 的 GSM8K、SVAMP、ARC-Easy、HellaSwag；保留 source ID 和 license/source metadata。
2. 按数据契约清洗 GSM8K：删除 calculator annotations，解析 `####`，构造固定 prompt/response。
3. 实现精确数值 parser 的训练标签校验；无效样本进入 rejection report。
4. 使用 Qwen3 tokenizer 统计 prompt/response/total 的 P50/P90/P95/P99/max 和 `>512` 比例。
5. 按 seed 2026 先划 validation 256，再抽 train 2,000；生成 smoke 256、test 500 固定 IDs；验证所有必要的不相交关系。
6. 若 `>512` 的有效训练样本比例大于 5%，将全项目配置改为 768；否则保持 512。记录决策文件，后续不得因单个方法更改。
7. 写 manifest、sha256、split JSONL、过滤报告和数据 README。

### Core files / modules

`configs/data/gsm8k.yaml`、`data/prepare.py`、`data/templates.py`、`data/tokenize.py`、`data/validate.py`、`data/manifests/*`。

### Key implementation choices

- 划分基于排序后的稳定 source ID + 独立 seeded RNG，不能依赖上游返回顺序。
- Prompt 和 response 分字段保存；collator 后续通过 token 边界将 prompt labels 设为 `-100`。
- 处理后公开数据前先检查上游 license；若不适合提交完整文本，仓库只提交 IDs、manifest 和重建脚本。

### Deliverables

- 数据 manifest、固定 IDs、处理后 JSONL、rejection report、长度统计与决策。
- `results/figures/token_length_distribution.*`。

### Acceptance Criteria

- 重跑两次产生相同 split IDs 和 sha256。
- train=2,000、validation=256、smoke=256、test500=500、test_full=1,319；SVAMP 数量以固定 revision 实际值为准并记录。
- train/validation 不相交；GSM8K test 不进入训练；SVAMP 不出现在任何选择配置中。
- 每个 response 恰有一个规范模板生成的末尾 `Final answer:`；没有 calculator annotation。
- 长度决策自动测试覆盖比例恰好 5%、略高于 5% 两个边界。

---

## Phase 2 — Evaluation Harness First

**Classification:** Core / MVP  
**Compute:** Local CPU；模型生成留待 GPU，但 parser/bootstrap 可完全本地验证  
**Dependencies:** Phase 1

### Goal

在训练前冻结评测口径，防止看到结果后修改答案解析或指标。

### Tasks

1. 实现数值提取、规范化、multiple `Final answer:` 规则、格式合规和 truncation 标记。
2. 实现数学逐样本 evaluator、聚合器和 deterministic bootstrap 95% CI。
3. 定义 generator protocol：输入 prompts，输出文本/token counts/timing；实现 mock generator 和 HF backend adapter。
4. 实现 Base/LoRA/QLoRA 共用的固定 base prompt；实现 Instruct reference 的官方 chat-template adapter并标注“不完全公平参考”。
5. 实现 ARC-Easy/HellaSwag 200 fixed-ID 的 zero-shot log-likelihood scorer；禁止生成 CoT。
6. 建立至少 30 个答案 parser golden cases：逗号、小数、分数、百分数、负数、货币、单位、多答案、不可解析。
7. 保存评测协议 hash；正式 prediction 文件携带该 hash。

### Core files / modules

`evaluation/answers.py`、`generation.py`、`math_eval.py`、`regression_eval.py`、`bootstrap.py`、`aggregate.py`、`configs/eval/*`。

### Key implementation choices

- Parser 必须在首个模型结果产生前冻结；之后 bug fix 需升级 `evaluator_version` 并对所有模型重评。
- 主指标为 normalized numeric exact match；parse rate 不能代替 accuracy。
- Bootstrap CI 和两个训练 seed 的差异分别报告，禁止混成单个“误差条”。

### Deliverables

- 可独立运行的数学/回归评测 CLI。
- prediction/summary schema、parser golden fixture、mock 评测报告。

### Acceptance Criteria

- Golden cases 100% 通过；同一输入和 bootstrap seed 输出字节级一致的 summary。
- 无法解析答案计为错误；多个 marker 数学答案取最后合法值且格式不合规。
- 用 fixture predictions 手算的 EM、parse rate、P50/P95 token stats 与程序一致。
- HF backend 在本地构造的小型随机 causal LM fixture 上完成 2 条端到端生成，不下载大模型。

---

## Phase 3 — Mini Transformer and KV Cache Learning Modules

**Classification:** Core learning module；不阻塞 Phase 4 以后主线  
**Compute:** Local CPU，0 GPU  
**Dependencies:** Phase 0；可与 Phase 1–2 并行但 Codex 默认按顺序完成

### Goal

用最小实现验证 decoder-only Transformer、causal loss、自回归生成和 GQA/KV Cache 内存公式，不追求文本质量。

### Tasks

1. 实现 token embedding、RMSNorm、RoPE、causal self-attention、可配置 MHA/GQA、SwiGLU MLP、residual block、LM head。
2. 实现 shifted cross entropy、teacher forcing、optimizer loop、checkpoint 和 greedy generation。
3. 用小型内置字符/整数 token 数据训练数百 step，记录 loss 曲线并生成样例。
4. 实现 KV Cache calculator：手工参数模式和读取 Hugging Face `config.json` 模式。
5. 公式固定为 `2 * layers * batch * seq_len * num_kv_heads * head_dim * bytes_per_element`，输出 bytes/MiB/GiB；注明 tensor parallel 未计入 Core。
6. 绘制 context length 与 batch size 对 KV Cache 内存的曲线，展示 GQA 相对 MHA 的差异。

### Core files / modules

`mini_llm/*`、`inference/kv_cache.py`、`configs/mini.yaml`、对应 unit tests。

### Key implementation choices

- 默认模型约 5M 参数，CPU 数分钟级；不得扩大到烧 GPU。
- Attention 测试与显式矩阵参考实现对齐；causal mask 测试证明未来 token 不影响过去 logits。
- KV Cache calculator 使用配置中的 `num_key_value_heads`；缺失时回退 `num_attention_heads` 并输出说明。

### Deliverables

- Mini model/train/generate 命令、loss 曲线、生成样例。
- KV Cache CLI、CSV 和曲线。

### Acceptance Criteria

- shape、causal mask、RoPE、RMSNorm、loss shift 测试通过。
- 固定小数据上末段平均 loss 明显低于首段平均 loss；只要求趋势，不设生成质量阈值。
- cached 与 uncached greedy decode 在容差内 logits/token 一致。
- KV 公式对手算 fixture 精确一致；seq length 或 batch 翻倍时内存线性翻倍，kv heads 缩减比例正确。

---

## Phase 4 — Model Loading, Collation, and CPU/GPU Smoke Gates

**Classification:** Core / MVP  
**Compute:** 先 Local CPU；随后免费 GPU 或获授权的 RTX 4090 短 smoke  
**Dependencies:** Phase 1–2

### Goal

在正式训练前证明真实 tokenizer、模型加载、assistant-only loss、LoRA 注入、QLoRA 4-bit 路径和结果记录正确。

### Tasks

1. 实现统一 model loader：Base、Instruct、adapter、merged、quantized 引用；显式 dtype/device map/revision。
2. 实现 response-only collator，验证 prompt token labels 全为 `-100`，response 与 EOS 参与 loss。
3. 实现 all-linear target discovery，排除 embedding/lm_head；将实际 target module 列表写入 metadata。
4. 在本地 tiny config 随机模型上跑 forward/backward、LoRA 注入、保存/加载 adapter。
5. 在 GPU 上分别对 LoRA BF16 和 QLoRA NF4 跑 8–20 optimizer steps、2–8 个评测样本。
6. 探测能容纳的 micro batch，计算 accumulation 保持 effective batch=32；记录 peak memory，不改其他冻结参数。
7. 验证 packing off；若计划开启 packing，另做 packed/unpacked label 与 token-count 正确性报告，且所有正式对照统一切换。
8. 验证 Transformers backend；若要使用 Unsloth，对相同 adapter config 做 loss/输出/target modules 一致性 probe 后再决定全实验后端。

### Core files / modules

`training/model_loader.py`、`collator.py`、`trainer.py`、`scripts/run_cpu_smoke.ps1`、`scripts/run_gpu_smoke.sh`。

### Key implementation choices

- BF16 为 RTX 4090 默认 compute dtype；环境探测不支持时才统一改 FP16并记录。
- QLoRA 仅冻结 base 权重采用 bitsandbytes NF4 + double quant；不能把它称为部署 INT4。
- 不允许 `device_map=auto` 隐式 CPU offload 后仍宣称是可比训练；检测到 offload 应失败或明确标为非正式 smoke。

### Deliverables

- CPU smoke report、GPU compatibility report、micro-batch decision、后端决策、packing 决策。
- 最终 resolved LoRA/QLoRA configs。

### Acceptance Criteria

- CPU tiny model 端到端训练、adapter reload 通过；prompt labels mask 测试通过。
- GPU LoRA/QLoRA 均完成短 run，无 NaN/Inf/OOM，metadata/metrics/memory samples 完整。
- LoRA 与 QLoRA config fairness diff 只显示允许字段。
- trainable parameter 数与 target list 被记录；embedding/lm_head 不可训练。
- 未获用户明确授权时，到此停止付费 GPU 操作并报告本地已就绪状态。

---

## Phase 5 — Pre-training Baselines

**Classification:** Core / MVP  
**Compute:** RTX 4090 约 0.5–1 h，必须获明确授权  
**Dependencies:** Phase 2、Phase 4

### Goal

在任何正式 fine-tuning 前取得可复现的 Base 与 Instruct reference 质量基线，并验证真实模型评测链路。

### Tasks

1. 对 Base 使用固定 completion prompt；对 Instruct 使用官方 chat template，分别保存 prompt rendering 样例。
2. 先跑 GSM8K test500 快速评测；链路无误后跑 GSM8K full、SVAMP full、ARC/HellaSwag fixed subsets。
3. 记录模型权重大小、加载 dtype、PyTorch allocated/reserved、`nvidia-smi` peak、生成耗时和 tokens/s。
4. 保存全部原始输出、解析结果、CI 和环境元数据。
5. 标记 Instruct 为 reference，不将其 prompt-level 分数写入 Base/LoRA/QLoRA 严格主表。

### Core files / modules

`configs/base.yaml`、`configs/eval/*`、`evaluation/*`、`serving/hf_backend.py`、`results/predictions/*`。

### Key implementation choices

- Baseline 质量生成遵循 greedy/正常 EOS/max 256；不与固定 token 性能 workload 混用。
- 先 Base 后 Instruct，任何 parser 改动都需两者全量重评并升级 evaluator version。

### Deliverables

- `results/summaries/baseline.json/csv`、逐样本 predictions、显存/速度记录。

### Acceptance Criteria

- test500 无缺失/重复 sample ID，成功率 100%；全量运行前 parser/format 分布人工抽查至少 20 条。
- full 结果数量精确匹配 manifest，summary 可由 raw predictions 完全重建。
- Base 与 Instruct 的配置、prompt 类型和“不完全公平”标签清楚可见。

---

## Phase 6 — LoRA Rank Study

**Classification:** Core / MVP  
**Compute:** RTX 4090 约 1.5–2 h，必须获明确授权  
**Dependencies:** Phase 5

### Goal

以相同 SFT 协议训练 LoRA rank 8/16，量化 rank 对 trainable parameters、训练显存、时间和质量的影响。

### Tasks

1. 固定数据顺序、seed 42、epochs/LR/scheduler/effective batch/length/target modules；只改变 rank 和对应 alpha。
2. 每 epoch 保存 adapter，在 validation 上评测；按 EM → parse rate → earlier epoch 选择 checkpoint。
3. 分别记录 load/train/eval/save 时间、step loss、tokens、peak allocated/reserved 和 `nvidia-smi` peak。
4. 对 selected adapter 跑 test500；只在链路/结果合理后跑 full GSM8K/SVAMP/regression。
5. 比较 trainable params 理论值与实际值；解释 rank 增加是否带来质量收益。

### Core files / modules

`configs/train/lora_r8.yaml`、`lora_r16.yaml`、`training/trainer.py`、`callbacks.py`、`selection.py`。

### Key implementation choices

- 默认 all-linear；若 smoke 证明不可行，可在正式 run 前整体改 attention-only，并让所有 LoRA/QLoRA config 一起改变。不得只改一个方法。
- 最多 3 epochs；test/SVAMP 不得参与选 checkpoint。
- Rank 32 是 Recommended Extension，不在此阶段自动执行。

### Deliverables

- 两个完整 run、selected adapters、validation/test predictions、rank 对照 summary。

### Acceptance Criteria

- 两个 run 的 fairness diff 只包含 rank/alpha/run name。
- 每 epoch checkpoint 与 validation 指标齐全，selection 可独立重放并选出同一 checkpoint。
- 所有训练样本数、optimizer steps、有效 tokens/data order 一致。
- 结果表至少含 trainable params、peak VRAM、train time、GSM8K EM/CI、parse rate。

---

## Phase 7 — QLoRA Fair Comparison and Replication

**Classification:** Core / MVP（核心训练实验）  
**Compute:** RTX 4090 约 2–3 h，必须获明确授权  
**Dependencies:** Phase 6

### Goal

在只改变 base 权重加载精度及必要 NF4 配置的条件下，测量 QLoRA 相比 BF16 LoRA 的训练内存节省、速度变化和质量差异。

### Tasks

1. 选择主比较 rank：默认 rank 16；若 Phase 6 validation 明确支持 rank 8，则预先记录选择理由后使用 rank 8。不能看 test 后决定。
2. 用 seed 42 训练匹配 rank 的 QLoRA，并与对应 LoRA 做严格 config diff。
3. 对 LoRA/QLoRA 两者用 seed 1234 各独立复跑一次。
4. 按相同规则选 checkpoint，并运行 test500/full、SVAMP、regression。
5. 分别报告每 seed 的结果、跨 seed 差异与样本 bootstrap CI；不将两者混合。
6. 对 selected adapters 做 adapter reload 稳定性测试，为 merge 阶段准备 canonical candidate。

### Core files / modules

`configs/train/qlora_r*.yaml`、`training/model_loader.py`、`trainer.py`、`selection.py`、`analysis/tables.py`。

### Key implementation choices

- LoRA 和 QLoRA compute dtype 一致；不同项仅 BF16 base vs NF4 frozen base 及直接必要的量化 loader config。
- 正式性能/内存结论只来自 RTX 4090；免费 T4/L4 结果只能标 smoke。
- 最佳“方法”由 validation 决定，不能根据 test 选。

### Deliverables

- seed 42/1234 的匹配 LoRA/QLoRA runs。
- 后训练主表与 seed/CI 分解图；最佳方法选择记录。

### Acceptance Criteria

- 四个关键 run 的数据顺序、steps、tokens、target modules、batch、optimizer/scheduler 一致。
- QLoRA metadata 证明 `nf4`、double quant、BF16 compute；LoRA metadata 证明 BF16 base。
- 报告同时给出绝对 Peak VRAM 和相对节省百分比、训练时间、EM/CI/parse rate。
- 最佳 adapter 的选择只引用 validation 文件。

---

## Phase 8 — Merge, Consistency, and Canonical Checkpoint

**Classification:** Core / MVP  
**Compute:** CPU 可做部分；RTX 4090 做正式一致性评测，约 0.5 h  
**Dependencies:** Phase 7

### Goal

把 validation 选出的唯一最佳 adapter 合并为 canonical 浮点 checkpoint，确保后续 BF16 与 INT4 来自同一权重源。

### Tasks

1. 加载 Base + selected adapter，固定 tokenizer/special tokens，merge 并保存 safetensors。
2. 保存 lineage：base revision、adapter hash/config、merge library versions、merged file hashes。
3. 在固定 20 个样本上比较 unmerged 与 merged greedy token IDs/logits（容差内）和评测输出。
4. 对 merged checkpoint 跑 test500 consistency eval；异常时停止量化，定位 dtype/tokenizer/merge 问题。
5. 将通过验收的路径写入 `artifacts/canonical.json`；后续量化只读取此文件。

### Core files / modules

`training/merge.py`、`quantization/validate.py`、`artifacts/canonical.json`。

### Key implementation choices

- BF16 merged checkpoint 默认不上传，因为可由 Base + adapter 重建；但本地保留 hash。
- “一致”优先比较 greedy token IDs/最终 EM；浮点 logits 使用明确 atol/rtol，不要求字节相同。

### Deliverables

- Canonical merged checkpoint、lineage、hash、consistency report。

### Acceptance Criteria

- 20 个固定样本 greedy token IDs 或最终文本无非预期差异；test500 EM/parse rate 不出现超过预设容差的变化（默认 EM 差 0，若浮点边界造成差异必须逐样本解释）。
- tokenizer/config/special token files齐全，可在新进程离线重载。
- `canonical.json` 只指向一个已校验 checkpoint。

---

## Phase 9 — Deployment INT4 Quantization

**Classification:** Core / MVP  
**Compute:** RTX 4090 约 1–2 h，必须获明确授权  
**Dependencies:** Phase 8

### Goal

从 canonical merged checkpoint 生成一种 vLLM 可服务的 W4A16 checkpoint，并与浮点版本进行可归因比较。

### Tasks

1. 运行 `quantize probe`：在少量层/校准样本上检查 Qwen3、工具版本、Marlin kernel 和 vLLM load compatibility。
2. 默认 GPTQ W4A16 + Marlin；若 probe 失败，保存日志、错误分类和环境证据，然后将全阶段切换 AWQ，不能同时做两套正式实验。
3. 校准集只从固定 train split 抽取，保存 IDs；禁止使用 validation/test/SVAMP。
4. 量化 canonical checkpoint，保存 quantization config、tool versions、calibration IDs、文件 hash。
5. 验证 reload、生成、vLLM serving；跑 GSM8K test500，再跑 full/SVAMP/regression。
6. 记录理论权重大小、磁盘大小、HF/vLLM 推理 peak VRAM；性能正式对照留到 Phase 11。

### Core files / modules

`configs/quantize/int4.yaml`、`quantization/compatibility.py`、`gptq.py`、`awq.py`、`validate.py`。

### Key implementation choices

- NF4 QLoRA 是训练加载方式；GPTQ/AWQ 是部署 checkpoint，两者命名和结果表必须分开。
- INT4 质量对照使用同一 canonical checkpoint、prompt、eval backend 和生成配置。
- INT8 不属于 Core。

### Deliverables

- 一种正式 INT4 checkpoint、compatibility decision、lineage、质量/大小/显存 summary。

### Acceptance Criteria

- INT4 可在新进程重载并通过 20 样本生成及 vLLM health check。
- 校准 IDs 与 validation/test/SVAMP 零交集。
- BF16 与 INT4 全量评测数量、协议 hash 一致；报告质量 delta、CI、parse/format delta。
- 磁盘大小和 peak VRAM 有可重建测量方法，不能只引用理论 bit 数。

---

## Phase 10 — Serving Backends and Benchmark Correctness

**Classification:** Core / MVP  
**Compute:** 本地用 mock server；RTX 4090 做服务 smoke 约 0.5–1 h  
**Dependencies:** Phase 8–9

### Goal

建立语义一致的 HF 与 vLLM 推理入口、流式客户端和固定 token workload，先证明测量正确再进行正式 benchmark。

### Tasks

1. 实现 HF in-process/backend endpoint，支持逐 token 时间戳；实现 vLLM OpenAI-compatible launcher/client。
2. 固定 tokenizer 和输入 token IDs：从非评测敏感的合成/训练文本构造恰好 512 tokens 的 workload；保存 IDs/hash。
3. 性能配置：greedy、sampling off、output 128、ignore EOS；确认两个引擎实际输入/输出 token 数一致。
4. 实现 async closed-loop concurrency runner、warm-up 隔离、请求超时、错误记录和重复实验。
5. 实现独立 GPU monitor，采样 `nvidia-smi`，同时在 HF 后端记录 PyTorch allocated/reserved。
6. 用可控 mock streaming server 测 TTFT/TPOT 公式、并发调度、失败/短输出处理。
7. 在 RTX 4090 对同一 BF16 checkpoint 做 HF vs vLLM concurrency=1 小样本等价 smoke；再验证 INT4 vLLM。

### Core files / modules

`serving/*`、`benchmark/workload.py`、`online.py`、`metrics.py`、`gpu_monitor.py`、`configs/serve/*`、`configs/benchmark/core.yaml`。

### Key implementation choices

- TTFT 必须来自客户端收到首个流式 token，不用服务端总耗时代替。
- 使用 `time.perf_counter_ns()` 单调时钟；wall-clock 只记录 run 起止。
- Core 使用 closed-loop fixed concurrency；RPS/open-loop sweep 是 extension，指标不可混报。
- HF vs vLLM 并发 1 的目的为语义与低并发开销对照；并发扩展主要研究 vLLM。

### Deliverables

- HF/vLLM 启动命令、health/readiness check、固定 workload、mock 测试、smoke report。

### Acceptance Criteria

- Mock server 的已知延迟与测得 TTFT/TPOT 在容差内；请求错误不被吞掉。
- 两个真实后端都返回 512 input/128 output tokens；warm-up 不进入正式原始事件。
- HF/vLLM concurrency=1 输出 token IDs 或生成参数等价性有报告；无法逐 token 等价时必须说明后端数值差异，但 workload 长度必须一致。
- GPU monitor 采样覆盖整个正式窗口，并能关联 benchmark run ID。

---

## Phase 11 — Formal Inference Benchmark

**Classification:** Core / MVP（核心 Systems 实验）  
**Compute:** RTX 4090 约 2–3 h，必须获明确授权；尽量同一实例租期完成  
**Dependencies:** Phase 10

### Goal

在受控硬件和固定 workload 上测量 BF16/INT4、HF/vLLM 与 vLLM 并发扩展的 TTFT、TPOT、E2E、吞吐和显存权衡。

### Tasks

1. 租用前执行 hardware gate：独占 GPU、无其他进程、正常 power limit、driver/CUDA、SSD/缓存路径、磁盘空间、网络和时间同步。
2. 冻结环境，记录 `pip freeze`/容器信息、GPU UUID；服务启动后等待 readiness 并进行 10 warm-up。
3. 低并发等价实验：同一 canonical BF16/FP16 checkpoint，HF vs vLLM，concurrency=1，100 请求 × 3 repetitions。
4. vLLM 并发实验：concurrency 1/4/8/16，每点 100 请求、10 warm-up、3 repetitions；BF16 与 INT4 各跑完整矩阵。
5. 随机化或交错配置执行顺序，避免热状态/温度/后台漂移总偏向同一配置；每 repetition 记录 GPU 温度/时钟/power。
6. 每个点输出 raw events、GPU samples、server log；聚合 P50/P95 TTFT/TPOT/E2E、output/total tok/s、success rate、peak VRAM。
7. 对异常点只允许按预先规则重跑：服务错误、成功率不足、GPU 有外部进程、token 长度不符；保留原失败 run，不覆盖。

### Core files / modules

`scripts/run_core_benchmark.sh`、`benchmark/*`、`results/raw/benchmarks/*`、`analysis/tables.py`。

### Key implementation choices

- 正式结论只使用同一型号 RTX 4090；其他 GPU 仅 smoke。
- 模型加载/服务启动不计请求 latency，但单独报告 startup/model load time。
- Peak VRAM 同时报 `nvidia-smi` 和框架可用指标；说明两者口径不同。
- INT4 不预设更快；分析 kernel、dequantization、batch/concurrency 和 memory headroom。

### Deliverables

- 完整 benchmark raw bundle、四并发点汇总、HF/vLLM 表、BF16/INT4 表、环境证据。

### Acceptance Criteria

- 每个正式配置恰有 3 个有效 repetitions，每个 100 请求；另有明确隔离的 warm-up。
- 成功率、实际 token 数、失败原因完整；成功率低于 99% 的点不得作为正常性能结论，必须修复或标为失败。
- 所有分位数可从 raw JSONL 独立重算；聚合测试和手算 fixture 一致。
- 表格能明确回答：并发提高后吞吐与单请求 latency 如何变化；INT4 节省多少 VRAM、质量/性能代价是什么；HF 与 vLLM 的比较边界是什么。

---

## Phase 12 — Final Analysis, Documentation, and Reproduction Package

**Classification:** Core / MVP  
**Compute:** Local CPU；必要的缺失数据补跑需单独授权  
**Dependencies:** Phase 5–11

### Goal

把实验数据转化为可审计的结论和面试可讲述的工程证据，不把不等价变量塞进同一消融表。

### Tasks

1. 自动生成三张主表：post-training、deployment quantization、inference engine/concurrency。
2. 绘制：质量 vs training VRAM、BF16 vs INT4 size/VRAM/quality、concurrency vs TTFT/TPOT/throughput、KV Cache vs context/batch。
3. 选择少量代表性端到端配置画 Pareto 图；在图注中说明哪些变量同时变化，避免伪因果。
4. 编写 `docs/findings.md`：hypothesis、method、controls、results、unexpected findings、limitations、cost、next experiment。
5. 完成 README quickstart、CPU/GPU 分层安装、数据重建、训练、量化、serving、benchmark 和结果复算命令。
6. 写 `docs/gpu_runbook.md`：租用前检查、SSD 路径、执行顺序、结果同步、关机清单；不得包含自动租用逻辑。
7. 整理公开产物：adapters、INT4 checkpoint、manifest/IDs、逐样本预测、configs、summaries、环境信息；检查 license、secret 和绝对路径。
8. 形成 `docs/interview_notes.md`，用本项目数据回答 Transformer/LoRA/QLoRA/quantization/KV Cache/vLLM/实验设计问题。

### Core files / modules

`analysis/*`、`results/summaries/*`、`results/figures/*`、`README.md`、`docs/*`。

### Key implementation choices

- 图表由 raw data 脚本生成，不手工编辑数值。
- 训练 seed 差异和样本 bootstrap CI 分图或分列。
- 明确限制：单模型系列、单硬件型号、小训练子集、单任务主线、单部署 INT4 格式。

### Deliverables

- 完整 README、findings、实验表图、复现包、GPU 成本表、面试笔记。

### Acceptance Criteria

- 从干净 checkout 按 README 可完成 CPU 测试、data dry-run/重建、结果聚合；GPU 步骤有清晰先决条件。
- 所有公开 summary 可从提交/发布的 raw 结果一键重建，数值一致。
- 三组主实验没有混淆变量；每条关键结论都链接到 config 与 raw run IDs。
- 仓库扫描无 token/secret、个人绝对路径、大型未跟踪 checkpoint；license/source attribution 完整。

## 6. Recommended Extension Phases

### Extension R1 — Expanded Length and Request-rate Matrix

在 Core vLLM benchmark 稳定后增加 input 128/1024、output 32、P99 和 open-loop RPS。复用同一 raw event schema，但添加 `workload_mode: closed_loop|open_loop`，两种模式分别汇总。验收要求每个长度组合实际 token 数正确，且能将长 prompt 对 TTFT、长输出对 E2E/TPOT 的影响分开解释。

### Extension R2 — Rank 32 and INT8

只有 rank 8/16 结果显示尚未饱和或存在反常趋势时训练 rank 32；保持 alpha/rank=2。INT8 仅从 canonical checkpoint 生成，并加入部署量化正交表。两项均需单独 GPU 成本估计和用户授权。

### Extension R3 — Profiling and Error Analysis

对代表性 BF16/INT4、低/高并发点使用 PyTorch Profiler 或 Nsight 做短 profile；不要 profile 全 benchmark。质量侧对 arithmetic、reasoning、format、truncation、OOD 等错误做固定 taxonomy 和双人/复核式抽样规则。

## 7. Bonus / Future Tracks

Core 完成后只选择一条与目标岗位一致的路线：

- **Post-training/Algorithm:** DPO → synthetic data/filtering → reasoning dataset → robustness evaluation。
- **AI Systems:** profiler → FlashAttention IO 分析 → Triton RMSNorm/softmax → CUDA fundamentals。
- **Inference Optimization:** 多种 weight-only quantization → vLLM/SGLang → speculative decoding → KV/prefix cache optimization。
- **Training Infrastructure:** DDP/FSDP 理论 → 两卡短 ZeRO/FSDP 实验；不得以多卡规模替代实验质量。

每条 Bonus 都必须沿用 Core 的 run metadata、raw-event-first 和正交比较原则，并另建设计文档；不得直接塞入 Core config。

## 8. 测试策略与质量门

### 8.1 测试层级

| 层级 | 环境 | 内容 |
|---|---|---|
| Unit | CPU、无网络 | parser、hash、split、mask、attention、KV 公式、metrics、config fairness |
| Integration | CPU、尽量无网络 | tiny random model 的 train/save/load/generate、mock streaming server |
| Data | CPU、有缓存/网络 | revision、count、ID overlap、hash、token length |
| GPU smoke | 免费 GPU/授权 4090 | LoRA/QLoRA 数步、dtype、OOM、INT4/vLLM compatibility |
| Formal | 授权 4090 | 冻结 config 的训练、质量评测、benchmark |

### 8.2 每阶段通用停止条件

- 数据 hash/revision 与 manifest 不符。
- 对照 config 出现未授权差异。
- NaN/Inf、隐式 CPU offload、外部 GPU 进程或异常 power limit。
- 输出 token 数不符、成功率不足、原始事件缺失。
- 选 checkpoint 的代码读取 test/SVAMP。
- 需要付费 GPU 但没有用户明确执行指令。

遇到停止条件时保留失败 run 和日志，修复后生成新 run ID；不得覆盖或删改失败证据。

## 9. GPU 执行与成本控制

### 9.1 GPU 准入门

进入首次付费 GPU 前，必须已有：

- CPU tests 全绿；数据 manifest/hash 冻结；parser golden tests 全绿。
- 所有 GPU 命令 dry-run 通过；模型/数据/cache/output 明确写入 200GB SSD。
- smoke config 限制 steps/samples；正式 config 与 smoke config 分离。
- 用户明确授权启动付费实例。

### 9.2 建议批次

1. **Smoke batch（免费 GPU 优先）：** 真实模型加载、LoRA/QLoRA 数步、micro batch、量化/vLLM probe。
2. **Training batch（4090）：** baseline → LoRA r8/r16 seed42 → QLoRA matching seed42 → LoRA/QLoRA seed1234；及时同步 adapters/raw metrics。
3. **Deployment batch（4090）：** merge check → INT4 probe/quantize → quality evaluation。
4. **Benchmark batch（同一 4090 实例）：** HF/vLLM BF16 等价 → vLLM BF16/INT4 concurrency matrix → 同步结果 → 停机。

初始总预算仍以约 12–18 GPU 小时估算，Phase 4 smoke 后用实测 tokens/s 和阶段耗时修订。任何 Recommended/Bonus 实验不自动消耗 Core buffer。

## 10. Codex 执行规则

1. 每次只开始一个未完成的 Core Phase；Phase 3 可被标记为不阻塞，但仍需在最终 Core 前完成。
2. 开始 Phase 前读取本文件、上一阶段 deliverables 和当前 git status；不得覆盖用户的无关改动。
3. 先实现最小闭环和测试，再扩大数据/模型；能在 CPU fixture 发现的问题不得留给 GPU。
4. 每完成一个 Phase：运行验收、更新 `docs/` 中相应记录、保存命令与结果路径、提交一份简短 phase report。
5. 不因“更方便”修改冻结协议。需要修改时，在 `docs/experiment_protocol.md` 添加 dated decision record，说明证据、影响范围和是否需要重跑既有结果。
6. 任何正式实验必须先生成 resolved config 和 config hash；任何汇总必须从 raw 文件构建。
7. 不自动创建云资源、不自动产生费用、不自动发布模型/数据；这些动作分别需要用户明确授权。

## 11. 推荐实施顺序总览

```text
Phase 0  工程与复现骨架（CPU）
   |
Phase 1  数据、固定划分、token 长度决策（CPU）
   |
Phase 2  先冻结评测（CPU）
   |\
   | Phase 3  Mini Transformer + KV Calculator（CPU，不阻塞主线）
   |
Phase 4  模型/Collator/LoRA/QLoRA Smoke Gate（CPU -> 短 GPU）
   |
Phase 5  Base + Instruct Baseline（GPU）
   |
Phase 6  LoRA Rank 8/16（GPU）
   |
Phase 7  QLoRA + 两 seed 公平复跑（GPU）
   |
Phase 8  最佳 Adapter Merge 与一致性（CPU/GPU）
   |
Phase 9  单一路线部署 INT4（GPU）
   |
Phase 10 Serving 与 Benchmark 正确性（CPU -> 短 GPU）
   |
Phase 11 正式 HF/vLLM/并发 Benchmark（同一 GPU）
   |
Phase 12 汇总、分析、复现与公开材料（CPU）
   |
Recommended Extensions -> 任选一条 Bonus Track
```

执行优先级始终是：可复现质量/训练显存 → 同 checkpoint BF16 vs INT4 → 同 checkpoint/精度 HF vs vLLM → 核心复跑 → 扩展实验。项目的成功标准是形成可信、可解释、可重跑的 Quality / Memory / Performance trade-off 证据，而不是训练更大的模型或调用更多框架。
