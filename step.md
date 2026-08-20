# 实现进度

## 2026-08-20 — Phase 0：仓库基础与可复现性门禁

**状态：** 已完成  
**使用的计算资源：** 仅本地 CPU；GPU 使用时间为 0 小时；未创建云端或付费资源。

### 执行依据

- 严格按照 `plan.md` 中 Phase 0（第 495–536 行）作为执行规范。
- `Idea.md` 仅作为项目背景参考；当两者存在差异时，冻结决策以及目录、接口选择均以
  `plan.md` 为准。
- 开始执行时，仓库仅包含 `Idea.md`、`plan.md` 和 `LICENSE`，不存在无关的用户改动。
  起始提交为 `26bfec1`。

### 环境隔离

- Python 环境搭建和依赖安装仅使用本地 `uv 0.9.22`。
- CPython 3.11.14 安装在项目内的 `.tools/python`。
- 虚拟环境创建在项目内的 `.venv`。
- uv 下载缓存保存在项目内的 `.cache/uv`。
- 新增 `.python-version`、`uv.toml` 和 `scripts/setup_local.ps1`，确保环境可重复搭建。
- GPU 依赖继续隔离在 `requirements/gpu-train.txt` 和 `requirements/gpu-serve.txt`；
  CPU 代码导入时不会加载 vLLM 或 bitsandbytes。

兼容性说明：uv 0.9.22 支持在 `uv.toml` 中配置 `cache-dir`，但不支持
`python-install-dir` TOML 键。因此，每个环境搭建或运行脚本均通过
`UV_PYTHON_INSTALL_DIR` 将 Python 安装目录显式设为 `.tools/python`。验收命令使用
`uv run --no-sync`，确保测试执行期间不会隐式更新依赖。

### 已实现内容

1. 创建 `src/` package 布局、规划中的配置/依赖/结果/文档目录、editable package
   元数据、CLI 入口、README 和 gitignore 规则。
2. 实现 YAML 加载、确定性的 `${...}` 插值、Pydantic schema、规范化 SHA-256 配置
   哈希，以及 schema 版本、模型 revision、允许的 rank、alpha/rank 比例、精确有效
   batch size、packing 报告和质量/性能 EOS 策略隔离校验。
3. 实现 LoRA/QLoRA 配置对照校验器，仅允许冻结协议规定的加载方式、量化字段和运行
   标签差异，并拒绝未经授权的实验变量漂移。
4. 实现确定性 run ID、随机种子设置（可选延迟支持 NumPy/PyTorch）、输出目录冲突
   保护，以及文本/JSON 原子写入。
5. 实现 `doctor` 环境元数据采集，包括操作系统、Python、Git commit/dirty 状态、
   CPU、RAM 和磁盘；当 NVIDIA 环境可用时，防御性采集 UUID、驱动、CUDA 版本、
   显存、功耗限制和活动计算进程。
6. 建立规划中的完整 CLI 命令树。Phase 0 支持 help、doctor 和配置 dry-run；后续
   Phase 的命令会明确提示尚未实现并以退出码 4 结束，避免假成功。
7. 配置 pytest unit/integration 标记、ruff、严格 mypy、PowerShell CPU smoke 入口、
   会明确失败的 GPU 占位脚本、architecture/protocol/runbook/findings 文档，以及脱敏
   的环境报告示例。

### 验收证据

以下命令在 `D:\Code\efficient-llm` 中使用项目本地 uv 环境执行：

```text
./scripts/run_cpu_smoke.ps1
  ruff：全部检查通过
  mypy：检查 18 个源文件，全部通过
  pytest -m "not gpu"：15 个测试通过，用时 0.66 秒

CLI 配置 dry-run：13/13 通过
doctor 报告：results/raw/phase0_environment.json
doctor Python：3.11.14
doctor GPU available：false
第二次向相同路径写入 doctor 报告：退出码 2；原报告保持不变
```

测试明确覆盖：非法 rank、非法 alpha 比例、错误 batch 关系、缺失模型 revision、
质量评测 EOS 策略误用、LoRA/QLoRA 配置漂移、稳定哈希与插值、原子写入、非空输出
目录冲突、无 NVIDIA 环境下的 doctor 行为、CLI dry-run，以及后续 Phase 占位命令
必须失败的行为。

### 当前进度

| Phase | 状态 | 说明 |
|---|---|---|
| Phase 0 | 已完成 | 所有规定的 CPU 验收检查均已通过 |
| Phase 1 | 尚未开始 | 必须从冻结数据集 revision 和 manifest 开始 |
| Phase 2–12 | 尚未开始 | 保持 `plan.md` 规定的依赖顺序 |

本阶段未修改任何冻结实验变量。当前配置 revision 使用显式的 `main` dry-run 值；根据
冻结协议，在后续任何正式运行之前，必须将其替换为不可变的模型和数据集 revision。
