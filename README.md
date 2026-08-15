# inference_scaling

本仓库研究语言模型的免训练推理扩展：保持基础模型参数固定，通过 Metropolis--Hastings（MH）、
重要性采样（IS）、off-policy rollout replay 和序贯蒙特卡洛（SMC）重新分配生成概率，并与
Group Relative Policy Optimization（GRPO）训练基线比较质量、计算量和墙钟。

## 目标分布与方法

给定提示 $`x`$、基础模型分布 $`p(y\mid x)`$、序列奖励 $`r(y)`$ 和奖励温度 $`\tau`$，主要目标为：

```math
\pi_r(y\mid x)
=\frac{p(y\mid x)\exp\{r(y)/\tau\}}
       {\sum_{y'}p(y'\mid x)\exp\{r(y')/\tau\}}.
```

该目标只改变完整序列的相对权重。仓库实现三条主要路径：

| 路径 | 核心操作 | off-policy / replay 处理 | 主要实现 |
| --- | --- | --- | --- |
| [后缀 MH](docs/methods/ALGORITHMS.md#alg-power-mh) | 重生成随机后缀，再按 Hastings 比接受或拒绝 | proposal 的正反概率进入接受率 | [`mh.py`](src/inference_scaling/algorithms/mh.py) |
| [条件能量 IS](docs/methods/ALGORITHMS.md#alg-conditional-is) | 为下一 block 生成候选，用 rollout 估计条件能量后重采样 | completion 来自其他模型时乘 $`p/q`$ | [`conditional_energy.py`](src/inference_scaling/algorithms/conditional_energy.py) |
| [rollout replay](docs/methods/ALGORITHMS.md#alg-base-replay) 与[动态候选](docs/methods/ALGORITHMS.md#alg-dynamic-is) | 复用历史 completion，并按方差和成本分配 fresh rollout | behavior 概率、fresh-tail 校正和外层 $`p/q_c`$ | [`base_replay.py`](src/inference_scaling/algorithms/base_replay.py)、[`dynamic_is.py`](src/inference_scaling/algorithms/dynamic_is.py) |

progressive IS、流式奖励、SMC rollout forest、delayed-acceptance MH、历史后缀 proposal、批处理、
KV 复用和 vLLM 后端均在同一份[算法基础、原理与实现文档](docs/methods/ALGORITHMS.md)中按“目标—算法—实现—误差与
成本”组织。

## 文档

| 文档 | 内容 |
| --- | --- |
| [算法基础、原理与实现](docs/methods/ALGORITHMS.md) | 基础知识、数学目标、算法步骤、收敛性质、关键代码、执行优化和 vLLM 配置 |
| [GSM8K 实验设计](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md) | 数据、模型、预算、指标、成本分母、命令和产物 |
| [方法质量与计算量](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md) | 准确率、pass@k、共享奖励、off-policy、replay 与消融 |
| [推理执行与 rollout 复用](docs/reports/RTX3090_ROLLOUT_INFRA.md) | 墙钟、FLOPs、吞吐、缓存成本和复用率 |
| [GSM8K 集成检查](docs/validation/GSM8K_QUICK_VALIDATION.md) | 8 题端到端路径和 32 题批处理检查 |
| [RTX 3090 复现记录](docs/validation/RTX3090_REPRODUCTION.md) | CUDA、概率评分、KV、MH、IS 与 replay 检查 |
| [机器可读结果](results/README.md) | 正式汇总、训练摘要和验证产物索引 |

## 主要结果

质量实验使用 32 道固定 GSM8K test 题、`Qwen2.5-1.5B-Instruct` 和单张 RTX 3090。主表中的方法
可能使用不同奖励；统一正确性奖励的比较见[共享奖励实验](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md#共享奖励目标)。

| 方法 | 正确数 / 32 | pass@1 | 推理 PFLOPs |
| --- | ---: | ---: | ---: |
| Base | 13 | 40.625% | 0.0279 |
| 幂分布 MH | 12 | 37.500% | 1.3077 |
| 标准条件 IS | 21 | 65.625% | 1.3706 |
| 0.5B rollout proposal 条件 IS | 15 | 46.875% | 2.4724 |
| GRPO 参数 + 随机采样 | 22 | 68.750% | 0.0254 |

标准条件 IS 与 GRPO 参数随机采样相差 -3.125 个百分点，题目级配对 95% 区间为
`[-12.500, 6.250]`。精确奖励实验中，verifier-MH 与标准 verifier-IS 分别得到 78.125% 和
75.000%。完整统计解释见[质量报告](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md)。

执行层因子定义为“优化路径 / 对照路径”；小于 1 表示相应成本下降。

| 优化路径 | 对照路径 | 墙钟因子 | 主模型 FLOPs 因子 |
| --- | --- | ---: | ---: |
| 连续批处理 | 同方法逐 prompt | 0.206×–0.952× | 1.003×–1.177× |
| warm replay 在线阶段 | fresh-only | 0.859× | 0.766× |
| 流式 IS，0.2 s verifier | 整批完成后提交 verifier | 0.671× | 1.000× |
| MH proposal-tree 预取，0.2 s 奖励 | 普通 MH | 0.817× | 1.267× |
| delayed acceptance，0.2 s 奖励 | 普通 MH | 0.827× | 1.000× |
| replay 混合 MH proposal，在线 | base suffix proposal | 0.534× | 1.003× |
| SMC 条件后缀复用 | 相同 SMC 的 fresh-only 路径 | 0.856× | 0.963× |

## 安装

Transformers 环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

RTX 3090 实验环境：

```powershell
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -m pip install -e ".[dev,gpu,training]"
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

vLLM `0.25.x` 使用 Linux GPU wheel。Windows 主机在 WSL2 的 Linux 文件系统中创建独立环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,vllm]"
```

## 复现

准备数据并训练 GRPO：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\prepare_gsm8k.py `
  --config configs\gsm8k_3090_aligned.toml
.\.venv\Scripts\python experiments\train_gsm8k_grpo.py --resume auto
```

运行质量、replay、动态候选、批处理和消融网格：

```powershell
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

运行执行层消融：

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python experiments\benchmark_rollout_infra.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers.json

.\.venv\Scripts\python experiments\benchmark_is_mh_reuse.py `
  --backend transformers --dtype bfloat16 --section all --seed 20260812 `
  --output results\infra\rtx3090_transformers_is_mh_seed20260812.json
```

完整运行顺序、pass@k、重评分消融和绘图命令见
[GSM8K 实验设计](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md)。

## 测试与目录

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python -m pytest
```

| 路径 | 内容 |
| --- | --- |
| `src/inference_scaling/` | 算法、后端、调度、replay 和计算账本 |
| `configs/` | 模型、数据与预算配置 |
| `experiments/` | 训练、运行、汇总与绘图入口 |
| `tests/` | 分布、实现一致性和结果处理测试 |
| `docs/` | 算法与实现、实验协议、报告和验证记录 |
| `results/` | 纳入版本控制的机器可读汇总 |
