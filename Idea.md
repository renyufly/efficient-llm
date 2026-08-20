# Efficient LLM Post-training & Inference Systems

## 0. 项目定位

### 项目名称

**Efficient LLM Post-training & Inference Systems**

可用于简历的中文定位：

> **有限 GPU 资源下的大语言模型高效后训练、量化与推理优化**

这个项目的目的不是单纯“微调一个模型”，也不是简单把 Unsloth、bitsandbytes、vLLM 等工具全部调用一遍。

项目核心问题是：

> **在有限单卡 GPU 资源下，如何通过高效微调、模型压缩和推理优化，在模型效果、GPU 显存、训练成本和推理性能之间取得合理 trade-off？**

整个项目采用：

```text
Hypothesis
    ↓
Experiment
    ↓
Measurement
    ↓
Comparison
    ↓
Analysis
```

而不是：

```text
安装 Library
    ↓
运行 Demo
    ↓
项目结束
```

------

# 1. 项目背景与个人约束

## 已有项目

目前已有：

```text
Project 1
RAG + Agent AI System

涉及：
- Retrieval
- Reranking
- RAG
- Tool Calling
- Agent Workflow
- LLM Application
```

因此本项目不再重复 AI Application Layer，而是重点覆盖：

```text
LLM Model
+
Post-training
+
Inference
+
AI Systems
```

最终形成：

```text
              AI / LLM 技术栈

              Application
                  │
          RAG + Agent Project
                  │
                  ▼
             LLM Serving
                  │
           Inference Systems
                  │
        Quantization / Runtime
                  │
             Post-training
                  │
             Transformer
```

------

# 2. 项目约束

本项目明确遵守以下原则：

- 无本地 GPU
- 尽量使用免费 GPU
- 必须租 GPU 时优先低价单卡
- 不依赖多机多卡
- 不进行大规模预训练
- 不进行长时间 full fine-tuning
- GPU 只负责必要计算
- 本地 CPU 完成绝大部分代码开发和数据处理
- 重点是理解机制、实验设计和性能分析
- GPU 时间不是项目成果
- 不追求 SOTA
- 不追求模型越大越好

------

# 3. 项目最终能力目标

项目完成之后，需要能够较深入地解释以下四个层次。

## 3.1 Transformer / LLM Fundamentals

理解：

```text
Tokenizer
Embedding
Self-Attention
Causal Mask
Multi-Head Attention
RoPE
RMSNorm
FFN
Residual Connection
LM Head
Cross Entropy
Autoregressive Generation
```

------

## 3.2 Post-training

重点掌握：

```text
SFT
LoRA
QLoRA
PEFT
LoRA Rank
LoRA Alpha
Target Modules
Gradient Accumulation
Gradient Checkpointing
Mixed Precision
NF4
```

理解：

> 为什么不用 Full Fine-tuning？

> LoRA 为什么可以显著降低 Trainable Parameters？

> QLoRA 为什么能够降低训练显存？

------

## 3.3 Quantization

掌握：

```text
FP32
FP16
BF16

INT8
INT4

PTQ
GPTQ
AWQ
NF4
```

理解：

> Quantization 为什么降低显存？

> Quantization 为什么可能降低模型质量？

> INT4 为什么不一定比 BF16 推理更快？

------

## 3.4 LLM Inference Systems

重点掌握：

```text
Prefill
Decode
KV Cache

TTFT
TPOT
Latency
Throughput

Batching
Continuous Batching
PagedAttention

vLLM
Model Serving
```

最终可以分析：

```text
Model Quality
      ↑
      │
      │
      └────────────→ Compute Cost

Memory
Latency
Throughput
Quality
```

之间的关系。

------

# 4. 项目总体 Architecture

```text
                     Dataset
                        │
                        ▼
                  Base LLM
                        │
                Baseline Eval
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
   Post-training               Inference Study
          │                           │
      SFT / LoRA                   Prefill
       QLoRA                       Decode
          │                        KV Cache
          │                           │
          ▼                           │
     Fine-tuned Model                │
          │                           │
          └─────────────┬─────────────┘
                        │
                        ▼
                  Quantization
                 BF16 / INT8 / INT4
                        │
                        ▼
                 vLLM Serving
                        │
                        ▼
                    Benchmark
                        │
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
       Quality        Memory       Performance
                                     │
                               TTFT / TPOT
                               Throughput
                               Latency
```

------

# 5. 推荐技术栈

核心：

```text
Python
PyTorch

Hugging Face
├── Transformers
├── Datasets
├── PEFT
└── TRL

Unsloth
bitsandbytes

vLLM
```

评测：

```text
lm-evaluation-harness
自定义 evaluation script
```

性能分析：

```text
nvidia-smi

Core 完整版 / 后续：
PyTorch Profiler
Nsight Systems
Nsight Compute
```

日志：

```text
JSON / CSV
TensorBoard

Core MVP 之后可选：
Weights & Biases
```

------

# 6. 模型选择策略（已确认）

## Core 模型

```text
Base：Qwen/Qwen3-1.7B-Base

Reference Instruct：Qwen/Qwen3-1.7B
```

主要实验使用 Base 模型进行 SFT。Instruct 模型只作为同尺寸参考 Baseline，不进入全部消融实验，也不与 Base 系列进行完全公平的 Prompt-level 对照。

模型规模固定为 1.7B，不因普通显存问题降级；必要时减少 Micro Batch Size、增加 Gradient Accumulation，或在 Token 长度分析后调整 Max Sequence Length。

选择理由：

- 对 16GB / 24GB 单卡相对友好
- Hugging Face / PEFT / bitsandbytes / vLLM 支持良好
- 具备 GQA，适合后续分析 KV Cache
- Base 与同尺寸 Instruct checkpoint 均可用
- License 适合公开实验产物

项目重点仍然是实验设计和分析，而不是追求更大的模型。

------

# 7. GPU 策略

## 7.1 Local CPU

完成：

```text
代码开发

Dataset Download
Dataset Cleaning
Dataset Sampling

Prompt Formatting
Tokenizer Analysis

Evaluation Pipeline
Plotting

Result Analysis

Unit Test

Mini Transformer
```

------

## 7.2 免费 GPU

优先使用：

```text
Google Colab

Kaggle Notebook

Unsloth 官方 Notebook / 支持环境

其他提供免费 GPU 的 Notebook 平台
```

主要跑：

```text
QLoRA
短时间 Evaluation
Quantization Test
```

------

## 7.3 低价单卡 GPU

本项目正式 GPU 已选择：

```text
RTX 4090 24GB
```

租用参考价格为约 2 元人民币/小时。A10 24GB 的参考价格约为 3.5 元人民币/小时；在显存容量相同的情况下，RTX 4090 对本项目具有更高的性价比，因此不再将 A10 作为首选正式平台。

可使用免费 L4 / T4 完成本地流程迁移或 Smoke Test，但所有正式训练显存、量化和推理性能结论统一来自 RTX 4090。

原则：

> **不要租 GPU 写代码。**

推荐流程：

```text
Local Development
       ↓
Git Push
       ↓
Cloud GPU
       ↓
git clone / git pull
       ↓
Install
       ↓
Run Experiment
       ↓
Save Result
       ↓
Shutdown GPU
```

------

# 8. GPU 时间与硬件预算（已确认）

GPU 预算优先保持较低，但不再预先设置不可调整的绝对硬上限。先通过 Smoke Test 和 Token 统计估算，再根据核心实验需要调整。

正式硬件配置：

```text
GPU：RTX 4090 24GB
CPU Memory：48GB DDR4
Storage：50GB System Disk + 200GB SSD
Reference Price：约 2 CNY / GPU Hour
```

RTX 4090 支持本项目所需的 BF16、bitsandbytes NF4、AWQ/GPTQ/Marlin 和 vLLM。所有正式性能结论必须来自同一型号 GPU；优先在同一个实例租用周期内集中完成正式 Benchmark，并记录 GPU UUID、Power Limit、Driver、CUDA 和软件环境。

48GB CPU 内存足够完成 1.7B 模型的数据处理、Adapter Merge、量化和评测。50GB 系统盘只用于操作系统和基础环境；Hugging Face Cache、PyTorch Cache、pip Cache、Docker Data、训练 Checkpoint、Merged Model、量化模型和结果必须放到 200GB SSD，避免系统盘被缓存占满。

租用前必须确认：完整独占 24GB GPU、无其他用户进程、无异常低功率限制、Linux 环境、200GB SSD 在实验期间持久可用，以及可以访问 Hugging Face。硬件确认只冻结实验方案，不代表立即租用、启动实例或产生费用；只有收到明确的执行指令后才进入云 GPU 阶段。

计划：

| 实验         | GPU 时间        |
| ------------ | --------------- |
| Baseline     | 0.5～1 h        |
| LoRA         | 1.5～2 h        |
| QLoRA        | 2～3 h          |
| Quantization | 1～2 h          |
| Evaluation   | 1～2 h          |
| vLLM         | 1～2 h          |
| Benchmark    | 2～3 h          |
| Debug Buffer | 2～3 h          |
| **Total**    | **先以约 12～18 h 估算，Smoke Test 后修订** |

原则：

> GPU 时间应该花在获得实验数据，而不是 debugging。

------

# 9. Phase 0 — Project Setup

时间：

**0.5～1 天**

GPU：

**0**

建立 Repository：

```text
efficient-llm/
│
├── README.md
├── requirements.txt
│
├── configs/
│
├── data/
│
├── mini_llm/
│
├── finetuning/
│
├── quantization/
│
├── inference/
│
├── evaluation/
│
├── benchmark/
│
├── results/
│
└── docs/
```

建立统一实验配置：

```yaml
model:
dataset:
precision:
batch_size:
learning_rate:
lora_rank:
max_seq_len:
seed:
```

------

# 10. Phase 1 — Transformer Fundamentals

时间：

**3～4 天**

GPU：

**0**

目标：

> 确保后续 LoRA、Quantization、KV Cache、FlashAttention 等知识不是黑盒。

------

## 10.1 Attention

理解：Attention(Q,K,V)

$$ 
Attention(Q,K,V)
softmax\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

掌握：

```text
Q
K
V

Attention Score

Softmax

Causal Mask

Multi-head Attention
```

------

## 10.2 Decoder-only Transformer

理解：

```text
Tokens
 ↓
Embedding
 ↓
Transformer Block
 │
 ├── RMSNorm
 ├── Attention
 ├── Residual
 ├── RMSNorm
 ├── MLP
 └── Residual
 ↓
LM Head
 ↓
Logits
```

重点掌握：

```text
RoPE
RMSNorm
Causal Mask
GQA / MQA 基础概念
```

------

## 10.3 Training Objective

理解：

$$
P(x_t|x_1,...,x_{t-1})
$$

以及：

$$
L=-\sum_t \log P(x_t|x_{<t})
$$

重点：

```text
Next-token Prediction
Cross Entropy
Teacher Forcing
```

------

# 11. Phase 2 — Mini Transformer From Scratch

时间：

**2～3 天**

GPU：

**0**

项目定位：

> 这是 Learning Module，不是最终项目主成果。

实现：

```python
Embedding

Attention

MLP

TransformerBlock

MiniGPT
```

完成：

```text
Forward

Loss

Backward

Optimizer

Generation
```

规模：

```text
约 5M ～ 30M Parameters
```

只训练：

```text
几百 ～ 几千 Step
```

目标：

看到：

```text
Training Loss ↓
```

并成功生成文本即可。

禁止：

> 为了让 MiniGPT 生成高质量文本而烧 GPU。

------

# 12. Phase 3 — Define Research Question

这是项目和普通 Demo 最大的区别。

核心问题：

> **How can we efficiently post-train and serve an LLM under limited single-GPU resources while balancing model quality, memory consumption, and inference performance?**

整个项目围绕三个 trade-off：

```text
Quality

Memory

Performance
```

最终主要比较拆成三个正交问题：

```text
Post-training：Base vs LoRA vs QLoRA

Deployment Quantization：同一最佳 Merged Checkpoint 的 BF16/FP16 vs INT4

Inference Engine：同一 Checkpoint、同一精度的 Standard HF vs vLLM
```

INT8 仅在预算允许时追加，不进入 Core MVP。训练方法、量化精度和推理引擎不能作为同一级互斥配置混合比较。

------

# 13. Phase 4 — Dataset & Baseline（已确认）

时间：

**2～3 天**

GPU：

**约 0.5 h**

## Dataset

主任务固定为：

> **英文小学数学应用题的短 CoT 数学推理，输出简洁推理过程和可解析的最终数值答案。**

| 用途 | 数据集 | 规模 |
| ---- | ------ | ---- |
| Smoke Experiment | GSM8K Train 固定子集 | 256 |
| 正式训练 | GSM8K Train 固定子集 | 2,000 |
| Validation | 从 GSM8K Train 固定划出 | 256 |
| 快速域内评测 | GSM8K Official Test 固定子集 | 500 |
| 最终域内评测 | GSM8K Official Test | 1,319 |
| OOD 评测 | SVAMP Full Set | 约 1,000 |

主指标为 `Normalized Numeric Exact Match`。SVAMP 只作为 OOD 挑战集，不参与训练、Prompt 调试、超参数选择或 Checkpoint 选择。

数据划分固定使用 `data_seed = 2026`，与训练随机种子分离。先完成清洗、校验和 Token 长度过滤，再划分 Validation 和抽取正式训练集。保存原始样本 ID、数据集 revision、处理配置和处理后 hash。快速 Test 500 条同样固定样本 ID。

### 数据处理规则

- 使用 GSM8K 原始推理，不调用教师模型重写
- 删除 `<<expression=result>>` 计算器标注
- 将 `#### answer` 转换为 `Final answer: answer`
- 保留简洁自然语言推理
- 不使用 Socratic 版本
- 不使用 Calculator Tool 辅助训练或评测
- 不静默截断答案；超长样本过滤并单独记录
- 使用 Qwen3 Tokenizer 统计 P50 / P90 / P95 / P99 / Max 长度
- 初始目标 `max_seq_length = 512`
- 若超过 512 的有效样本比例大于 5%，则考虑调整为 768

固定训练模板：

```text
Problem:
{question}

Provide a concise solution and finish with "Final answer:".

Reasoning:
{reasoning}

Final answer: {answer}
```

只对 Assistant / Response 部分计算 Loss，不对用户问题部分计算 Loss。

------

## Baseline Evaluation

必须在 Fine-tuning 之前进行。

记录：

```text
Model Quality

Model Size

Precision

VRAM

Latency

Tokens/s
```

保存：

```text
results/baseline.json
```

Baseline 包括 Base 模型和同尺寸 Instruct Reference。Instruct 模型使用其官方 Chat Template，只作为参考，不进入 Base / LoRA / QLoRA 的严格公平主对照。

------

# 14. Phase 5 — LoRA

时间：

**2～3 天**

GPU：

**约 1.5～2 h**

理解：

$$
W'=W+\Delta W
$$

其中：

$$
\Delta W=BA
$$

且：

$$
r \ll d
$$

------

## 实验

不需要做大量 Grid Search。

已确认：

```text
LoRA r = 8

LoRA r = 16

仅在 r=8 / 16 结果有分析价值时追加：
LoRA r = 32
```

固定参数：

| Rank | Alpha | Alpha / Rank |
| ---- | ----- | ------------ |
| 8    | 16    | 2            |
| 16   | 32    | 2            |
| 32（可选） | 64 | 2         |

```text
Target Modules: all-linear，排除 Embedding 和 LM Head
LoRA Dropout: 0.05
Bias: none
Train LM Head: false
```

如显存或训练速度不理想，可整体改为 attention-only，但所有 Rank、LoRA 和 QLoRA 必须使用相同 Target Modules。

严格统一：

```text
Dataset

Epoch

Learning Rate

Seed

Data and Data Order

Alpha / Rank

Dropout and Target Modules

Optimizer / Scheduler / Warmup

Effective Batch Size

Max Sequence Length

Gradient Checkpointing

Training Tokens

Validation and Checkpoint Selection
```

重点比较：

| Rank | Trainable Params | Peak VRAM | Train Time | Quality |
| ---- | ---------------- | --------- | ---------- | ------- |
| 8    |                  |           |            |         |
| 16   |                  |           |            |         |
| 32   |                  |           |            |         |

研究问题：

> Rank 增大是否一定提升模型质量？

> Trainable Parameters 增加了多少？

------

# 15. Phase 6 — QLoRA

时间：

**2～3 天**

GPU：

**约 2 h**

这是核心 Phase。

掌握：

```text
4-bit Base Model

NF4

Double Quantization

LoRA Adapter

Gradient Checkpointing
```

------

## 实验

比较：

```text
LoRA BF16（若硬件原生支持）

vs

QLoRA NF4 Base + 相同 Compute Dtype
```

QLoRA 的 NF4 是训练时对冻结 Base 权重的量化方式，不等同于最终部署 INT4 checkpoint。

控制：

```text
Dataset
Rank
Epoch
LR
```

测：

```text
Peak Training VRAM

Training Time

Quality

Trainable Parameters
```

结果：

| Method | Base Precision | Peak VRAM | Train Time | Quality |
| ------ | -------------- | --------- | ---------- | ------- |
| LoRA   | BF16           |           |            |         |
| QLoRA  | 4-bit          |           |            |         |

LoRA 与 QLoRA 允许不同的只有 Base 权重加载精度及其直接必需的量化配置。其余训练变量必须一致。

初筛使用 seed 42。最重要的 LoRA vs QLoRA 使用 seed 1234 独立复跑一次，共两个训练 seed。Bootstrap CI 反映测试样本不确定性，多 seed 差异反映训练随机性，两者分别报告。

核心问题：

> QLoRA 降低了多少 Training Memory？

> 是否明显影响最终模型表现？

------

# 16. Phase 7 — Quantization（已确认）

时间：

**2～3 天**

GPU：

**约 1～2 h**

Core 主要比较：

```text
BF16

INT4
```

方案定义：

```text
QLoRA Training：bitsandbytes NF4
Baseline Inference：BF16（硬件不支持时明确改为 FP16）
Deployment INT4：优先 GPTQ/W4A16 + Marlin，AWQ 作为兼容性回退
Optional：bitsandbytes INT8
```

RTX 4090 已确认。部署 INT4 的默认路线为 GPTQ/W4A16 + Marlin；进入量化阶段时先执行小型兼容性验证，如果目标模型、工具版本或 vLLM 后端存在阻塞，再回退到 AWQ。该切换由兼容性证据决定，不需要提前扩大为两套完整实验。第一版只正式评测一种部署 INT4，不同时堆叠多个量化格式。

量化对象是根据 Validation 选出的最佳微调方法：先将其 Adapter Merge 到 Base，得到唯一的浮点 checkpoint，再由同一个 checkpoint 生成部署 INT4。这样 BF16/FP16 与 INT4 对照只改变量化方式。

------

## 测量指标

```text
Model Size

VRAM

Task Quality

Latency

Throughput
```

表格：

| Precision | Model Size | VRAM | Quality | tok/s |
| --------- | ---------- | ---- | ------- | ----- |
| BF16      |            |      |         |       |
| INT8（可选） |         |      |         |       |
| INT4      |            |      |         |       |

核心分析：

> INT4 相比 BF16 节省多少显存？

> Accuracy / Score 损失多少？

> 为什么更低 bit 不一定意味着更低 Latency？

------

# 17. Phase 8 — LLM Inference Fundamentals

时间：

**2 天**

GPU：

**0**

理解：

```text
Prompt
  ↓
Tokenization
  ↓
Prefill
  ↓
KV Cache
  ↓
Decode
  ↓
Token
  ↓
Decode
  ↓
Token
```

------

## Prefill

特点：

```text
Processing whole prompt

Highly parallel

Often compute-intensive
```

理解：

> 为什么长 Prompt 会影响 TTFT？

------

## Decode

特点：

```text
One token at a time

Autoregressive

Memory intensive
```

理解：

> 为什么 Decode 很难像 Training 那样高度并行？

------

# 18. Phase 9 — KV Cache

时间：

**1～2 天**

GPU：

**0**

理解没有 Cache 时：

```text
Step 1:
Token 1

Step 2:
Token 1 2

Step 3:
Token 1 2 3

Step 4:
Token 1 2 3 4
```

会重复计算历史 token。

KV Cache 保存：

```text
K1 V1

K2 V2

K3 V3
```

------

## 必须自己实现一个 KV Cache Memory Calculator

输入：

```text
num_layers

num_kv_heads

head_dim

sequence_length

batch_size

dtype
```

输出：

```text
KV Cache Memory
```

研究：

```text
Sequence Length ↑

Batch Size ↑

KV Cache Memory ↑
```

最好生成：

```text
KV Cache Memory vs Context Length
```

曲线。

这个实验基本不需要 GPU，但面试价值很高。

------

# 19. Phase 10 — vLLM Serving

时间：

**2～3 天**

GPU：

**约 1～2 h**

部署：

```text
LLM
 ↓
vLLM
 ↓
OpenAI-compatible API
 ↓
Client
```

理解：

```text
Continuous Batching

PagedAttention

KV Cache Management

Request Scheduling
```

------

# 20. Phase 11 — Inference Benchmark

时间：

**3～4 天**

GPU：

**约 2～3 h**

这是本项目最重要的 Systems Experiment。

## Benchmark A — Online Concurrency

Core MVP 测试：

```text
1
4
8
16
```

固定：

```text
Input Length
Output Length

Request Count

Warm-up
```

测：

```text
Latency

TTFT

TPOT

Throughput

Peak VRAM
```

研究：

> 并发增加后 Throughput 为什么增加？

> 为什么单请求 Latency 可能增加？

------

## Benchmark B — Input Length

Core 完整版测试：

```text
128
512
1024
```

研究：

```text
Input Length
      ↓
Prefill Cost
      ↓
TTFT
```

------

## Benchmark C — Output Length

Core 完整版测试：

```text
32
128
```

研究：

```text
Output Length
      ↓
Decode Steps
      ↓
Total Latency
```

### Core MVP 固定 Workload

```text
Input Length: 512 tokens
Output Length: 128 tokens
Concurrency: 1 / 4 / 8 / 16
Requests per Point: 100
Warm-up Requests: 10
Independent Repetitions: 3
Sampling: disabled
Ignore EOS: true（仅性能测试，用于固定输出长度）
```

质量评测使用真实输出并允许正常 EOS；性能评测使用可复现的固定 Token Workload，两者严格分开。

HF vs vLLM 先做相同浮点 checkpoint、相同精度下的低并发等价对照，再研究 vLLM 并发 1/4/8/16。如有余量，再测试不同 Request Rate / RPS。

------

# 21. Phase 12 — Final Ablation Study（重新设计）

将所有实验结果串起来。

不再把训练方法、权重精度和推理引擎混在同一组互斥 Configuration 中。拆成三组正交实验：

### A. 后训练实验

```text
Base vs LoRA vs QLoRA
```

固定最终评测精度、Prompt、数据和推理后端，比较 Quality、Peak Training VRAM、Train Time 和 Trainable Parameters。

### B. 部署量化实验

```text
同一个最佳 Merged Checkpoint：BF16/FP16 vs INT4
```

固定推理引擎和 Workload，比较 Quality、Model Size、VRAM、Latency 和 Throughput。INT8 仅在预算允许时追加。

### C. 推理引擎实验

```text
同一个 Checkpoint、同一种精度：HF vs vLLM
```

先比较低并发等价场景，再研究 vLLM 在线并发扩展。

最终端到端总结只选择少量代表性组合绘制 Quality / Memory / Cost / Performance 的 Pareto Trade-off，不把互不等价的变量强行放进一张消融表。

最终回答：

> 如何在有限 GPU 环境下最大化 LLM 的 Quality / Cost / Performance Ratio？

------

# 22. Evaluation Philosophy

评测必须贯穿整个项目，而不是最后运行一次。

流程：

```text
Base
 ↓
Eval A
 ↓
LoRA
 ↓
Eval B
 ↓
QLoRA
 ↓
Eval C
 ↓
Quantization
 ↓
Eval D
 ↓
Inference Engine
 ↓
Benchmark E
```

必须区分两类指标。

## Model Quality

主任务：

```text
GSM8K Normalized Numeric Exact Match
SVAMP Normalized Numeric Exact Match
Final Answer Parse Rate
Output Format Compliance
Average Output Tokens
Truncation Rate
Bootstrap 95% Confidence Interval
```

生成协议固定为 Greedy Decoding、`temperature=0`、不 Sampling、不 Self-consistency、不调用 Calculator、固定 Max Output Tokens 和 Stop Conditions。Base / LoRA / QLoRA 使用相同 Prompt。

答案使用精确数值规范化，处理千位逗号、整数与等价小数、分数、百分数、负数、货币符号和单位；不做模糊语义匹配。无法解析计为错误并单独报告 Parse Rate。若出现多个 `Final answer:`，数学正确率使用最后一个合法答案，但格式合规记为失败。保存全部原始输出和解析结果。

通用能力回归集固定为 ARC-Easy 200 条和 HellaSwag 200 条，使用固定样本 ID、Zero-shot Log-likelihood Multiple-choice Accuracy，不生成 CoT、不参与调参。

------

## System Performance

```text
Peak VRAM

Model Memory

Training Time

TTFT

TPOT

Latency

Throughput
```

性能指标定义：

- TTFT：客户端发送请求至收到首个流式 Token
- TPOT：`(E2E Latency - TTFT) / (Output Tokens - 1)`
- Throughput：分别报告 Output Token/s 和 Total Token/s
- Latency：报告 E2E P50 / P95，必要时追加 P99
- VRAM：同时记录 PyTorch Allocated / Reserved 与 `nvidia-smi` Peak
- Model Size：分别报告磁盘大小和理论权重大小
- Training Time：区分模型加载、训练、评测和保存
- Serving：额外报告成功请求率与实际输入/输出 Token 数

------

# 23. 最终 GitHub Repo

```text
efficient-llm/
│
├── README.md
│
├── requirements.txt
│
├── configs/
│   ├── baseline.yaml
│   ├── lora.yaml
│   └── qlora.yaml
│
├── data/
│   ├── prepare.py
│   ├── inspect.py
│   └── README.md
│
├── mini_llm/
│   ├── attention.py
│   ├── model.py
│   └── train.py
│
├── finetuning/
│   ├── lora.py
│   └── qlora.py
│
├── quantization/
│   ├── int8.py
│   └── int4.py
│
├── inference/
│   ├── hf_generate.py
│   ├── kv_cache_calculator.py
│   └── vllm_server.sh
│
├── benchmark/
│   ├── latency.py
│   ├── throughput.py
│   ├── memory.py
│   └── benchmark_config.yaml
│
├── evaluation/
│   └── evaluate.py
│
├── results/
│   ├── baseline.csv
│   ├── finetuning.csv
│   ├── quantization.csv
│   └── inference.csv
│
└── docs/
    ├── architecture.md
    ├── experiment_design.md
    └── findings.md
```

------

# 24. 五周执行计划

## Week 1 — Fundamentals

完成：

```text
Transformer

Attention

RoPE

RMSNorm

Causal LM

Mini Transformer

Project Repo
```

GPU：

```text
0 h
```

------

## Week 2 — Post-training

完成：

```text
Dataset

Baseline

SFT

LoRA

QLoRA
```

理解：

```text
PEFT

LoRA Rank

LoRA Alpha

Target Modules

NF4

Gradient Checkpointing
```

GPU：

```text
约 4 h
```

------

## Week 3 — Quantization + Evaluation

完成：

```text
BF16

INT8

INT4

Quality Comparison

Memory Comparison
```

GPU：

```text
约 2～3 h
```

------

## Week 4 — Inference Systems

学习：

```text
Prefill

Decode

KV Cache

TTFT

TPOT

Continuous Batching

PagedAttention

vLLM
```

完成：

```text
KV Cache Calculator

vLLM Serving

Inference Benchmark
```

GPU：

```text
约 3～4 h
```

------

## Week 5 — Analysis + Interview

整理：

```text
Ablation Study

Plots

README

Architecture

Experiment Findings
```

准备：

```text
Transformer Interview Questions

Fine-tuning Questions

Quantization Questions

Inference Questions

AI Systems Questions
```

GPU：

```text
0～1 h
```

------

# 25. 项目 MVP

如果时间有限，只要完成下面这些，就已经可以作为完整简历项目：

```text
Transformer Fundamentals

        +

Baseline Evaluation

        +

LoRA / QLoRA

        +

BF16 / INT4 Quantization

        +

KV Cache

        +

vLLM

        +

TTFT / TPOT / Throughput Benchmark

        +

Final Ablation Study
```

这部分定义为：

# Core Project

也是求职阶段最应该优先完成的部分。

------

# 26. 暂时不进入 Core Project 的内容

第一版明确不做：

```text
Large-scale Pretraining

Full Fine-tuning of 7B+

Multi-node Training

Megatron-LM

Large-scale Distributed Training

RLHF / PPO

CUDA Kernel Development

Complex Triton Kernels

Tensor Parallelism

Pipeline Parallelism

Large-scale MoE
```

这些不是“不重要”。

而是：

> **在当前求职阶段，它们相对于时间和 GPU 成本的 ROI 没有 Core Project 高。**

------

# 27. BONUS PHASES

以下内容全部属于未来扩展。

Core Project 完成之后，根据目标岗位选择其中一条路线继续。

------

# BONUS A — Post-training / Algorithm Track

适合：

```text
LLM Algorithm

Post-training

Alignment

Reasoning

Research Intern
```

------

## Bonus A1 — DPO

增加：

```text
SFT Model
    ↓
Preference Dataset
    ↓
DPO
    ↓
Evaluation
```

比较：

```text
Base

SFT

SFT + DPO
```

重点理解：

```text
Chosen / Rejected

Reference Model

Preference Optimization

KL Regularization
```

研究：

> SFT 和 Preference Optimization 分别解决什么问题？

------

## Bonus A2 — Synthetic Data

研究：

```text
Teacher LLM
    ↓
Synthetic Dataset
    ↓
Filtering
    ↓
SFT
    ↓
Evaluation
```

研究：

```text
Synthetic Data Quantity

Synthetic Data Quality

Filtering Strategy
```

------

## Bonus A3 — Reasoning

选择 Math / Logic / Code Reasoning 数据。

研究：

```text
Base

vs

SFT

vs

Reasoning SFT

vs

Preference Optimization
```

分析：

```text
Accuracy

Reasoning Length

Failure Mode
```

------

## Bonus A4 — Advanced Evaluation

建立：

```text
Evaluation Suite

├── Accuracy
├── Robustness
├── Out-of-domain
├── Regression Test
└── Error Analysis
```

目标：

> 从“模型跑分”升级到“模型行为分析”。

------

# BONUS B — AI Systems / Inference Track

适合：

```text
AI Systems

LLM Inference

Inference Optimization

ML Systems

LLM Runtime
```

这是非常推荐的未来方向。

------

## Bonus B1 — PyTorch Profiler

分析：

```text
CPU Time

CUDA Time

Kernel Time

Memory

Operator Breakdown
```

学习：

> 系统优化必须先 Profile，再 Optimize。

------

## Bonus B2 — FlashAttention

学习：

```text
Standard Attention

vs

FlashAttention
```

重点不是调用：

```python
flash_attention=True
```

而是理解：

```text
HBM

SRAM

IO Complexity

Tiling

Kernel Fusion
```

回答：

> FlashAttention 为什么叫 IO-aware Attention？

------

## Bonus B3 — Triton Kernel

第一批可以做：

```text
Vector Add

Softmax

RMSNorm
```

然后：

```text
PyTorch

vs

torch.compile

vs

Triton
```

Benchmark：

```text
Latency

Bandwidth

Tensor Shape Scaling
```

------

## Bonus B4 — Transformer Kernel Optimization

进一步做：

```text
RMSNorm

Fused Activation

Fused MLP

Attention Component
```

重点：

```text
Memory Access

Kernel Launch

Fusion

Arithmetic Intensity
```

------

## Bonus B5 — CUDA

如果准备专门投：

```text
AI Systems

GPU Systems

Inference Engine

CUDA Engineer
```

再进一步学习：

```text
CUDA Programming Model

Thread

Warp

Block

Grid

Shared Memory

Registers

Global Memory

Memory Coalescing

Occupancy
```

------

# BONUS C — Distributed Training Track

适合：

```text
Training Infrastructure

Distributed Training

ML Systems

AI Infrastructure
```

这条路线不建议第一版实际烧很多 GPU。

------

## Bonus C1 — Distributed Concepts

先理解：

```text
Data Parallel

DDP

FSDP

ZeRO

Tensor Parallel

Pipeline Parallel
```

------

## Bonus C2 — DeepSpeed ZeRO

如果未来有两张廉价 GPU 可以短时间使用，可以实验：

```text
ZeRO-1

ZeRO-2

ZeRO-3
```

记录：

```text
Memory

Communication

Training Speed
```

------

## Bonus C3 — FSDP

比较：

```text
DDP

vs

FSDP
```

理解：

```text
Parameter Sharding

Gradient Sharding

Optimizer State Sharding
```

------

# BONUS D — Advanced Quantization Track

适合：

```text
Inference Optimization

Compression

Edge LLM

Deployment
```

扩展：

```text
GPTQ

AWQ

GGUF

Weight-only Quantization
```

可以比较：

```text
BF16

INT8

bitsandbytes INT4

GPTQ

AWQ
```

重点研究：

```text
Accuracy

Memory

Kernel Support

Latency
```

------

# BONUS E — Advanced Serving Track

进一步研究：

```text
vLLM

SGLang

TensorRT-LLM

llama.cpp
```

不需要全部深入。

可以挑两个比较：

```text
vLLM

vs

SGLang
```

研究：

```text
TTFT

TPOT

Throughput

Concurrency

VRAM
```

------

# BONUS F — Speculative Decoding

学习：

```text
Draft Model
     ↓
Generate Candidate Tokens
     ↓
Target Model Verify
     ↓
Accept / Reject
```

研究：

> 为什么 Speculative Decoding 可以减少 Decode Cost？

比较：

```text
Normal Autoregressive Decode

vs

Speculative Decode
```

------

# BONUS G — Long Context / KV Cache Optimization

适合：

```text
Inference Systems

Serving

Long-context LLM
```

研究：

```text
Context Length

KV Cache Memory

Latency

Throughput
```

进一步理解：

```text
GQA

MQA

KV Cache Quantization

Prefix Caching
```

------

# 28. 推荐的 Bonus 优先级

如果以后继续扩展，不建议同时做所有 Bonus。

## 如果目标是 LLM Algorithm

顺序：

```text
Core Project

    ↓

DPO

    ↓

Synthetic Data

    ↓

Reasoning

    ↓

Advanced Evaluation
```

------

## 如果目标是 AI Systems

顺序：

```text
Core Project

    ↓

Profiling

    ↓

FlashAttention

    ↓

Triton

    ↓

CUDA
```

------

## 如果目标是 Inference Optimization

顺序：

```text
Core Project

    ↓

Advanced Quantization

    ↓

vLLM / SGLang Comparison

    ↓

Speculative Decoding

    ↓

KV Cache Optimization

    ↓

Triton
```

------

## 如果目标是 Training Infrastructure

顺序：

```text
Core Project

    ↓

DDP / FSDP Theory

    ↓

DeepSpeed

    ↓

FSDP Experiment

    ↓

Tensor Parallelism

    ↓

Large-scale Training Systems
```

------

# 29. 最推荐的发展路线

目前不需要立即决定：

> 我到底是 Algorithm 还是 Systems？

先完成 Core：

```text
              Transformer
                   │
                   ▼
            LoRA / QLoRA
                   │
                   ▼
            Quantization
                   │
                   ▼
              KV Cache
                   │
                   ▼
                vLLM
                   │
                   ▼
              Benchmark
```

之后根据兴趣分叉：

```text
                      Core
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
    Algorithm Track             Systems Track
         │                           │
        DPO                       Profiling
         │                           │
 Synthetic Data                FlashAttention
         │                           │
     Reasoning                    Triton
         │                           │
   Evaluation                     CUDA
```

这个结构允许在项目过程中再判断自己真正喜欢哪个方向。

------

# 30. 项目最终成功标准

项目完成不以：

> “成功微调模型”

作为标准。

而是至少能够回答：

### Transformer

```text
为什么 Attention 是 O(n²)？

为什么 Decoder-only LLM 需要 Causal Mask？

RoPE 在做什么？

GQA 为什么减少 KV Cache？
```

### Fine-tuning

```text
LoRA 为什么降低 Trainable Parameters？

Rank 代表什么？

Alpha 有什么作用？

QLoRA 为什么省显存？

为什么使用 NF4？
```

### Quantization

```text
INT8 / INT4 如何降低显存？

为什么 Quantization 会造成 Quality Loss？

为什么 INT4 不一定比 BF16 更快？
```

### Inference

```text
Prefill 和 Decode 有什么区别？

为什么 Decode 往往 Memory Bound？

KV Cache 为什么有用？

KV Cache 为什么成为 Serving 的显存瓶颈？
```

### Systems

```text
TTFT 是什么？

TPOT 是什么？

Latency 和 Throughput 有什么区别？

Continuous Batching 为什么提高 Throughput？

PagedAttention 解决什么问题？

FlashAttention 和 PagedAttention 有什么区别？
```

### Experiment

最重要：

> **你为什么设计这个实验？**

> **你的数据说明了什么？**

> **结果是否符合你的预期？**

> **为什么出现这个结果？**

> **如果有更多 GPU，你下一步会验证什么？**

------

# 31. 最终项目 Narrative

面试时，这个项目应该可以用一句话总结：

> **I built an experimental LLM post-training and inference pipeline under constrained single-GPU resources, comparing LoRA/QLoRA, quantization, KV-cache behavior and vLLM serving, and systematically evaluated the trade-offs among model quality, GPU memory, TTFT, TPOT and throughput.**

而不是：

> I fine-tuned a model using Unsloth.

------

# 32. 简历项目组合最终定位

```text
PROJECT 1

RAG + Agent AI System

展示：
LLM Application Engineering

Retrieval
Reranking
Agent
Tool Calling
Workflow
Evaluation
```

以及：

```text
PROJECT 2

Efficient LLM Post-training
& Inference Systems

展示：
LLM Model + AI Systems

Transformer
LoRA
QLoRA
Quantization
KV Cache
vLLM
Benchmark
```

两个项目组合形成：

```text
          AI Algorithm / LLM Profile

     Application          Model / Systems
          │                     │
          │                     │
     RAG + Agent        Post-training
          │                     │
          │               Quantization
          │                     │
          │                Inference
          │                     │
          └──────────┬──────────┘
                     │
                     ▼
              End-to-End LLM
                Understanding
```

------

# 33. 一句话执行原则

> **Core Project 做深，不做大；实验做完整，不堆工具；GPU 用来验证假设，而不是用来证明项目规模。**

完成 Core Project 后，再根据实际面试岗位选择 Algorithm、Inference、AI Systems 或 Distributed Training 中的一条 Bonus Track 深挖。

------

# 34. 已冻结的 Core 实验协议

本节汇总当前已经确认的执行决策。若前文的概念性示例与本节冲突，以本节为准。

## 34.1 项目重点与范围层级

项目兼顾 Post-training 与 Inference Systems，但 Systems 稍重，重点保留：

```text
Quantization
KV Cache
TTFT / TPOT
vLLM Serving
Concurrency Benchmark
```

执行范围分为：

```text
Core MVP
    ↓
Core 完整版
    ↓
Bonus
```

Mini Transformer 保留为独立 Learning Module，但不得阻塞 Post-training、Quantization 和 Inference 主线。

GPU 预算按实际 Smoke Test 结果调整，优先级依次为：

1. Base / LoRA / QLoRA 的质量与训练显存
2. 同一 checkpoint 的 BF16/FP16 vs 一种部署 INT4 的质量与推理显存
3. HF vs vLLM 的并发性能
4. 核心实验复跑
5. Rank 32、INT8 和更多长度点

## 34.2 已确认硬件

```text
GPU：RTX 4090 24GB
Rental Price：约 2 CNY / GPU Hour
CPU Memory：48GB DDR4
System Disk：50GB
Data SSD：200GB
```

选择 RTX 4090 而不是约 3.5 元/小时的 A10 24GB，原因是两者显存容量相同，而 4090 在当前租价下具有更高的训练与推理性价比。正式 LoRA/QLoRA 显存测量、BF16 vs INT4、HF vs vLLM 及全部并发 Benchmark 均在 RTX 4090 上完成。

所有模型缓存、Docker Data、Checkpoint、Merged/Quantized Model 和结果写入 200GB SSD，不依赖 50GB 系统盘。正式实验前检查 GPU 独占性、Power Limit、Driver/CUDA、磁盘持久性和网络访问，并记录完整环境元数据。

硬件选择只代表方案冻结，不授权自动租用或启动实例。云 GPU 资源只有在收到明确执行指令后才会创建和计费。

## 34.3 默认训练协议

初始默认值：

```yaml
data_seed: 2026
initial_train_seed: 42
replication_train_seed: 1234
max_epochs: 3
learning_rate: 2e-4
optimizer: paged_adamw_8bit
lr_scheduler: cosine
warmup_ratio: 0.03
weight_decay: 0.0
effective_batch_size: 32
gradient_checkpointing: true
max_seq_length: 512  # Token 分析后可能调整为 768
max_new_tokens: 256
lora_dropout: 0.05
lora_bias: none
train_lm_head: false
assistant_only_loss: true
```

Micro Batch Size 根据显存调整，并通过 Gradient Accumulation 保持 Effective Batch Size = 32。Sequence Packing 只有在完成正确性验证后才启用，并且所有严格对照配置必须保持一致。

每个 Epoch 结束后在 Validation 上评测并保存 checkpoint。按以下顺序选择最佳 checkpoint：

1. Validation Normalized Numeric Exact Match 更高
2. Exact Match 相同时，Parse Rate 更高
3. 再相同时，选择更早的 checkpoint

Test 和 SVAMP 不参与 checkpoint 或超参数选择。所有配置最多训练 3 Epoch，不为单个配置追加额外 Epoch。

## 34.4 LoRA / QLoRA 公平性

LoRA 与 QLoRA 必须保持一致：

- 数据及数据顺序
- 训练 Seed
- Rank / Alpha / Dropout
- Target Modules
- Optimizer / Learning Rate / Scheduler / Warmup
- Effective Batch Size
- Max Sequence Length
- Gradient Checkpointing
- Training Tokens
- Validation 频率与 Checkpoint 选择规则
- 训练 Compute Dtype（硬件允许时）

允许不同的只有 Base 权重加载精度及其直接必需的量化配置。

Adapter 未 Merge 和 Merge 后各进行一次一致性检查，确认同一输入下输出与评测结果不存在非预期偏差。

## 34.5 质量评测协议

主任务：GSM8K；OOD：SVAMP；回归集：ARC-Easy + HellaSwag。

```text
Decoding: Greedy
Temperature: 0
Sampling: false
Self-consistency: false
Calculator: false
Max New Tokens: 256
```

数学评测使用真实输出并允许 EOS。多个 `Final answer:` 时取最后一个合法值计算数学正确率，但格式合规率记为失败。无法解析时 Exact Match 记为错误。

最终报告同时包含 Bootstrap 95% CI 和核心实验两个训练 Seed 的结果差异，明确二者代表不同的不确定性来源。

## 34.6 性能评测协议

质量评测与性能评测严格分离。性能评测使用固定 Token Workload、关闭 Sampling，并忽略 EOS 以保证输出长度一致。

Core MVP：

```text
Input Length: 512
Output Length: 128
Concurrency: 1 / 4 / 8 / 16
Requests per Point: 100
Warm-up: 10
Repetitions: 3
Report: TTFT / TPOT / E2E / Throughput / Peak VRAM / Success Rate
Percentiles: P50 / P95
```

Core 完整版再增加 Input 128/1024、Output 32、Request Rate/RPS、P99 和更多长度组合。

所有正式性能数据必须在同一张、同型号 GPU 上获得，并记录 GPU 型号、Driver、CUDA、PyTorch、Transformers、bitsandbytes、vLLM 和量化工具版本。

## 34.7 公开产物

计划公开：

- LoRA Adapter
- QLoRA Adapter
- 最终部署 INT4 Checkpoint
- 数据处理脚本和处理后数据
- 固定样本 ID、数据 Manifest、Revision 与 Hash
- 全部逐样本预测和解析结果
- 实验配置、随机种子、汇总 CSV / JSON
- 环境版本、GPU 信息和复现实验说明

原始 Qwen Base 不重复上传。BF16 Merged Checkpoint 默认不重复上传，因为可以由公开 Base 与 Adapter 重建。

## 34.8 明确推迟到 Core MVP 之后的项目

```text
LoRA Rank 32
bitsandbytes INT8
多个部署量化格式
更多 Input / Output Length
Request Rate Sweep
PyTorch Profiler
GSM-Symbolic
其他模型系列
Bonus Track
```

## 34.9 执行就绪状态

当前 Core MVP 的项目目标、模型、数据集、数据切分、训练协议、质量评测、硬件、量化候选路线和性能 Benchmark 均已冻结，可以进入执行层面。

以下事项不阻塞启动：

- 最终 `max_seq_length` 由 Token 长度统计按既定 5% 规则决定
- Micro Batch Size 和 Gradient Accumulation 由 RTX 4090 Smoke Test 决定，但 Effective Batch Size 固定为 32
- Sequence Packing 仅在正确性验证通过后统一启用
- GPTQ/W4A16 + Marlin 若兼容性失败则按既定规则回退 AWQ
- 公开产物的具体 Hugging Face / GitHub 发布位置在发布阶段确定
- Rank 32、INT8、Profiler 和其他扩展均在 Core MVP 完成后重新评估

进入执行层面不代表立即租用 GPU。先完成本地 Phase 0、数据准备、Tokenizer 长度分析、评测脚本和 CPU Smoke Test；只有本地流程通过且收到明确指令后，才启动付费 RTX 4090 实例。
