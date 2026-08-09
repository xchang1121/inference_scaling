# inference_scaling

本仓库研究如何在不修改或少量修改语言模型参数的情况下，直接控制推理时输出分布，并统一比较
Metropolis--Hastings（MH）、重要性采样（IS）、off-policy rollout replay 与 GRPO 的质量和计算量。

当前最完整的实测结论见
[GSM8K 单卡对齐实验](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md)。实验协议、机器可读结果和工程验证
已分别归档，不再混在 README 中。

## 从哪里开始

| 目的 | 入口 |
| --- | --- |
| 阅读主要实验设置、数据和结论 | [GSM8K 单卡对齐实验](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md) |
| 复现实验或核对公平性约束 | [GSM8K 统一实验设计](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md) |
| 对照数学对象与代码入口 | [算法映射](docs/methods/ALGORITHM_MAP.md) |
| 查看批处理、KV 复用和计量方式 | [推理性能设计](docs/methods/PERFORMANCE_DESIGN.md) |
| 查找全部文档 | [文档导航](docs/README.md) |
| 查找机器可读结果 | [结果索引](results/README.md) |

## 已实现的方法

| 标识 | 候选来源 | rollout / proposal | 作用 |
| --- | --- | --- | --- |
| `mh` | 当前完整序列 | 基座模型后缀 proposal | 直接采样固定长度幂分布或显式奖励目标 |
| `conditional-is` | 基座模型 | on-policy completion | 用条件能量重新分配基座候选的选择概率 |
| `base-replay` | 基座模型 | 历史 off-policy rollout + fresh tail | 在不改变候选来源的前提下复用 rollout |
| `dynamic-is` | base/辅助 proposal 混合 | 动态 proposal + 外层 IS | 支持候选层修正与方差—成本预算分配 |

四条路径共享后端、请求级随机数、概率评分、token/FLOPs 账本和诊断接口。算法实现位于
`src/inference_scaling/algorithms/`；GSM8K 对照实现位于 `experiments/`。

## 当前主要结果

以下数据来自 32 道固定 GSM8K 测试题、`Qwen2.5-1.5B-Instruct` 和单张 RTX 3090。各方法的奖励目标
并不完全相同，因此这张表比较任务质量与成本；共享奖励目标的受控比较见完整报告。

| 方法 | pass@1 | 每 32 题推理 PFLOPs |
| --- | ---: | ---: |
| Base | 40.625% | 0.0279 |
| 幂分布 MH | 37.500% | 1.3077 |
| 标准条件 IS | 65.625% | 1.3706 |
| 0.5B proposal 条件 IS | 46.875% | 2.4724 |
| GRPO 随机采样 | 68.750% | 0.0254 |

结论可以概括为：

- 标准条件 IS 的 pass@1 与本地 GRPO 接近，但单次推理约使用 GRPO 的 54 倍 FLOPs；GRPO 另有一次性
  15.646 PFLOPs 训练成本。
- 当前 0.5B off-policy proposal 既没有保持标准 IS 的质量，也没有减少 FLOPs；主模型精确重评分和
  有限样本权重方差抵消了小模型生成的收益。
- warm rollout replay 相对相同总 rollout 预算的 fresh-only 路径减少 23.4% 在线 FLOPs 和 14.1%
  墙钟；包含缓存构建时首次查询更贵，本轮从第 7 次重复查询开始回本。
- 连续批处理相对同一方法的逐 prompt 执行得到 1.050×–4.845× 墙钟吞吐提升，但它主要改善硬件
  利用率，不等同于减少算法 FLOPs。

这些结果是单卡、有限题目上的实测，不代表完整 1,319 题评测或完整序列分布等价。统计区间、pass@k、
共享目标实验、消融和限制均保留在主报告中。

## 安装与测试

基础开发环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
```

需要真实模型训练或推理时，先安装与本机驱动兼容的 CUDA PyTorch wheel，再安装 GPU 依赖。以下是已验证
环境使用的安装方式：

```powershell
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -m pip install -e ".[dev,gpu,training]"
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

PyTorch wheel 自带 CUDA 运行时；普通推理不要求本机 `nvcc` 与 `torch.version.cuda` 相同。当前
RTX 3090 结果默认使用 FP32，因为 BF16 下生成概率与批量重评分的偏差足以影响重要性权重。

## 快速功能检查

无需下载模型即可运行有限状态实验：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\toy_mh.py
.\.venv\Scripts\python experiments\toy_conditional_is.py
.\.venv\Scripts\python experiments\toy_base_replay.py
.\.venv\Scripts\python experiments\toy_dynamic_is.py
```

真实模型的小规模后端检查及其已验证结果见
[RTX 3090 复现记录](docs/validation/RTX3090_REPRODUCTION.md)。quick 配置仅用于确认所有实验路径能够贯通，
记录在 [GSM8K quick 集成检查](docs/validation/GSM8K_QUICK_VALIDATION.md)，不用于最终方法排序。

## 复现 GSM8K 实验

准备固定版本的数据与模型：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\prepare_gsm8k.py `
  --config configs\gsm8k_3090_aligned.toml
```

如需重新训练 GRPO 对照：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\train_gsm8k_grpo.py --resume auto
```

运行 3090 对齐主网格。该命令耗时较长，但逐题结果可恢复；原始记录写入被 Git 忽略的工作目录，
replay 与批处理汇总写入正式结果目录：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\run_gsm8k_suite.py `
  --config configs\gsm8k_3090_aligned.toml `
  --tag validated `
  --summary-root results\gsm8k_3090 `
  --with-matched-target `
  --with-replay `
  --with-async `
  --with-ablations `
  --with-budget-curve `
  --with-length-ablation `
  --ablation-limit 8
```

主表、计算量汇总、分布审计、pass@k、消融和绘图需要在网格完成后运行只读后处理器。完整命令、输出
文件和恢复规则见 [GSM8K 统一实验设计](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md)。

## 数据与结果管理

- `results/gsm8k_3090/`：主实验的正式机器可读汇总；
- `results/training/`：GRPO 训练摘要；
- `results/validation/`：quick 与后端工程检查，不作为最终结论；
- `results/gsm8k/`、`*.chunks.jsonl`、模型 checkpoint 和运行日志：可恢复的中间产物，默认不提交。

`validated` 表示网格、manifest 和输入哈希已经通过对应后处理器检查，不表示统计结论可以外推到其他
模型、硬件或数据规模。每个结果文件的用途见 [结果索引](results/README.md)。

## 正确性约束

- base replay 的候选块始终由基座模型生成；历史数据只辅助估计候选权重。
- 每条历史 rollout 保存实际 behavior policy 及其真实采样概率，包括温度和截断设置。
- 当前决策消费过的数据不会再次进入未来 evaluation pool；只有选择完成后独立生成的 reserve rollout
  才能成为新的 evaluation record。
- 异步执行使用请求级随机数流，并逐方法报告 token、答案与共同前缀一致性；不同 CUDA batch 形状导致
  输出分叉时，只把耗时比解释为相同 workload 的吞吐对比。
- 计算量以实际 forward token slots 和模型参数量估算 FLOPs；wall time、显存和能耗仅作硬件诊断。

## 仓库结构

- `src/inference_scaling/`：算法、后端、调度器、replay 存储和指标；
- `configs/`：固定的数据、模型与实验预算；
- `experiments/`：训练、运行、汇总与绘图入口；
- `tests/`：有限状态分布测试、实现一致性与结果处理测试；
- `docs/`：方法说明、实验协议、主报告与验证记录；
- `results/`：纳入版本控制的小型汇总，原始运行产物默认忽略。
