# inference_scaling

本仓库实现并评测语言模型推理时的分布重加权方法，包括 Metropolis--Hastings（MH）、重要性采样
（IS）、off-policy rollout replay、动态候选、SMC，以及 GRPO 训练基线。质量实验和执行层实验分别
报告准确率、pass@k、FLOPs、墙钟与复用率。

## 文档入口

| 内容 | 文档 |
| --- | --- |
| 质量、pass@k 与计算量 | [GSM8K 方法质量与计算量实验](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md) |
| 墙钟、吞吐与 rollout 复用 | [RTX 3090 推理执行与 rollout 复用实验](docs/reports/RTX3090_ROLLOUT_INFRA.md) |
| 算法定义、公式与关键实现 | [推理算法实现](docs/methods/ALGORITHMS.md) |
| 调度、缓存、后端与计量 | [推理基础设施实现](docs/methods/INFRASTRUCTURE.md) |
| 数据、设置与复现命令 | [GSM8K 统一实验设计](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md) |
| vLLM 安装与成对测量 | [vLLM 推理运行时](docs/methods/VLLM_RUNTIME.md) |
| 机器可读结果 | [结果索引](results/README.md) |
| 全部文档 | [文档导航](docs/README.md) |

## 已实现算法

| 标识 | 候选来源 | rollout 或 proposal | 统计作用 |
| --- | --- | --- | --- |
| `mh` | 当前完整序列 | base 后缀 proposal | 采样幂分布或显式奖励目标 |
| `conditional-is` | base | base completion | 估计条件能量并重采样候选 |
| `base-replay` | base | 历史 off-policy rollout + fresh tail | 校正并复用历史 rollout |
| `dynamic-is` | base/辅助模型混合 | 动态 proposal + 外层 IS | 修正候选来源并分配 rollout 预算 |
| `progressive-is` | base | pilot + 独立 evaluation | 按方差与成本冻结 evaluation 预算 |
| `smc-forest` | base 粒子 | 可继承的条件后缀 reservoir | 逐 block 重采样并复用条件后缀 |

方法标签的参数、奖励、候选、rollout、概率修正和解码规则见
[实验标签定义](docs/methods/ALGORITHMS.md#alg-report-labels)。

## 主要结果

质量实验使用 32 道固定 GSM8K test 题、`Qwen2.5-1.5B-Instruct` 和单张 RTX 3090。下表各方法的
奖励目标可能不同；共享奖励比较见[完整报告](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md#共享奖励目标)。

| 方法 | 正确数 / 32 | pass@1 |
| --- | ---: | ---: |
| Base | 13 | 40.625% |
| 幂分布 MH | 12 | 37.500% |
| 标准条件 IS | 21 | 65.625% |
| 0.5B rollout proposal 条件 IS | 15 | 46.875% |
| GRPO 参数 + 随机采样 | 22 | 68.750% |

- 标准条件 IS 与 GRPO 参数随机采样相差 -3.125 个百分点，配对 95% 区间为
  `[-12.500, 6.250]`；标准条件 IS 的推理 FLOPs 约为后者的 54 倍。
- 0.5B rollout proposal 条件 IS 相对标准条件 IS 低 18.75 个百分点，FLOPs 为 `1.804×`。
- 精确奖励实验中，verifier-MH 与 verifier-IS 分别得到 78.125% 和 75.000%。
- 删除 1.5B 后缀重评分后，精确奖励路径仍为 20/32，估算 FLOPs 降低 74.0%，墙钟增加 16.7%。

执行层结果统一采用“优化路径 / 对照路径”；小于 1 表示相应成本下降。

| 优化路径 | 对照路径 | 墙钟因子 | 逻辑 FLOPs 因子 |
| --- | --- | ---: | ---: |
| 连续批处理 | 同方法逐 prompt | 0.206×–0.952× | 1.003×–1.177× |
| warm replay 在线阶段 | fresh-only | 0.859× | 0.766× |
| 流式 IS，0.2 s verifier | 整批完成后提交 verifier | 0.671× | 1.000× |
| MH proposal-tree 预取，0.2 s 奖励 | 普通 MH | 0.817× | 1.267× |
| delayed acceptance，0.2 s 奖励 | 普通 MH | 0.827× | 1.000× |
| replay 混合 MH proposal，在线 | base suffix proposal | 0.534× | 1.003× |
| SMC 条件后缀复用 | 相同 SMC 的 fresh-only 路径 | 0.856× | 0.963× |

实验臂设置和成本分母见[执行标签定义](docs/methods/INFRASTRUCTURE.md#infra-report-labels)。

## 安装与测试

基础环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
```

RTX 3090 实验环境：

```powershell
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -m pip install -e ".[dev,gpu,training]"
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

PyTorch wheel 提供 CUDA 运行时；`nvcc` 仅用于编译 CUDA 扩展。正式 RTX 3090 质量网格使用 FP32。
vLLM 运行环境和 Linux/WSL2 安装步骤见 [vLLM 推理运行时](docs/methods/VLLM_RUNTIME.md)。

## 快速检查

有限状态算法检查：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\toy_mh.py
.\.venv\Scripts\python experiments\toy_conditional_is.py
.\.venv\Scripts\python experiments\toy_base_replay.py
.\.venv\Scripts\python experiments\toy_dynamic_is.py
```

真实模型检查结果见 [RTX 3090 复现记录](docs/validation/RTX3090_REPRODUCTION.md)；端到端集成结果见
[GSM8K quick 集成检查](docs/validation/GSM8K_QUICK_VALIDATION.md)。

## 复现实验

准备数据与模型：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\prepare_gsm8k.py `
  --config configs\gsm8k_3090_aligned.toml
```

训练 GRPO：

```powershell
.\.venv\Scripts\python experiments\train_gsm8k_grpo.py --resume auto
```

运行质量、replay、动态候选、批处理与消融网格：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\run_gsm8k_suite.py `
  --config configs\gsm8k_3090_aligned.toml `
  --tag validated `
  --summary-root results\gsm8k_3090 `
  --with-matched-target `
  --with-replay `
  --with-dynamic-is `
  --with-async `
  --with-ablations `
  --with-budget-curve `
  --with-length-ablation `
  --ablation-limit 8
```

运行 Transformers 执行层实验：

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python experiments\benchmark_rollout_infra.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers.json

.\.venv\Scripts\python experiments\benchmark_is_mh_reuse.py `
  --backend transformers --dtype bfloat16 --section all --seed 20260812 `
  --output results\infra\rtx3090_transformers_is_mh_seed20260812.json
```

完整后处理、pass@k、绘图和恢复命令见
[GSM8K 统一实验设计](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md)。

## 结果与数据

| 路径 | 内容 |
| --- | --- |
| `results/gsm8k_3090/` | 正式质量与计算量汇总 |
| `results/infra/` | 执行层重复运行与聚合 |
| `results/training/` | GRPO 训练摘要 |
| `results/validation/` | 工程和集成检查 |
| `results/gsm8k/` | 可恢复的逐题中间记录 |

`validated` 表示题目网格、manifest 和输入哈希通过后处理检查。统计结论的适用范围记录在对应报告中。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `src/inference_scaling/` | 算法、后端、调度、replay 和指标 |
| `configs/` | 模型、数据与预算配置 |
| `experiments/` | 训练、运行、汇总与绘图入口 |
| `tests/` | 分布、实现一致性与结果处理测试 |
| `docs/` | 方法、实验协议、报告与验证记录 |
| `results/` | 纳入版本控制的机器可读汇总 |
