# inference_scaling

本仓库同时实现自回归语言模型（AR-LLM）和掩码扩散语言模型（dLLM）的训练与推理扩展。AR-LLM
使用 Qwen2.5 与 Group Relative Policy Optimization（GRPO）；dLLM 使用 LLaDA-MoE 与
Variance-Reduced Preference Optimization（VRPO）。两侧共享 GSM8K 数据、奖励、统计量、计算账本和
可续跑调度，并分别实现 Metropolis--Hastings（MH）、重要性采样（IS）与 rollout replay。

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
| [后缀 MH](docs/methods/ALGORITHMS.md#alg-power-mh) | 重生成随机后缀或扩散 block，再按 Hastings 比接受或拒绝 | proposal 的正反概率进入接受率 | [共享接受核](src/inference_scaling/shared/mh.py)、[AR 适配](src/inference_scaling/arllm/algorithms/mh.py)、[dLLM 适配](src/inference_scaling/dllm/algorithms/search.py) |
| [条件能量 IS](docs/methods/ALGORITHMS.md#alg-conditional-is) | 为下一 block 生成候选，用 rollout 估计条件能量后重采样 | completion 来自其他模型时乘 $`p/q`$ | [AR 实现](src/inference_scaling/arllm/algorithms/conditional_energy.py)、[dLLM 实现](src/inference_scaling/dllm/algorithms/is_sampling.py) |
| [rollout replay](docs/methods/ALGORITHMS.md#alg-base-replay) 与[动态候选](docs/methods/ALGORITHMS.md#alg-dynamic-is) | 复用历史 completion，并按方差和成本分配 fresh rollout | behavior 概率、fresh-tail 校正和外层 $`p/q_c`$ | [AR replay](src/inference_scaling/arllm/algorithms/base_replay.py)、[dLLM replay](src/inference_scaling/dllm/replay.py) |

共享算法层不依赖模型的生成方向。条件 IS 使用统一的逐步候选、rollout 权重与重采样接口；MH 使用统一的
目标密度差、正反 proposal 比和接受/拒绝核。AR-LLM 与 dLLM 目录只实现 token 后缀、掩码 block 或扩散
轨迹的生成和概率评分。

训练对照采用 GRPO 与 [VRPO](https://arxiv.org/abs/2505.19223)。VRPO 以掩码扩散 ELBO 代替序列
log-likelihood：每个偏好对采样 8 个独立掩码比例、每个比例采样 1 个 mask，并让当前策略与冻结 reference
使用相同 mask。LoRA adapter 与被禁用 adapter 的 reference 共用一份常驻基础模型。

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

AR-LLM 与官方 LLaDA-MoE 使用不同的 Transformers 版本。单独运行一侧时可直接使用当前 Python；完整成对
运行时使用两个解释器。

### 当前 Python

AR-LLM 依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,gpu,training]"
```

LLaDA-MoE 与 VRPO 依赖应安装到另一个 Python，或在只运行 dLLM 时安装到当前 Python：

```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,dllm,dllm-training]"
python experiments\dllm\download_llada.py `
  --config configs\gsm8k_llada_moe_3090.toml --source modelscope
```

### 已有的 `.venv`

仓库根目录已有 `.venv` 时可直接作为控制器、dLLM 解释器或测试解释器，无需激活：

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev,dllm,dllm-training]"
.\.venv\Scripts\python experiments\dllm\run_llada_suite.py --profile smoke
.\.venv\Scripts\python -m pytest
```

### 两个显式解释器

解释器可以来自系统安装、已有 `.venv`、Conda 或其他 Python 安装。变量值既可为绝对路径，也可为 `PATH`
中的可执行文件名：

```powershell
$env:AR_PYTHON = "C:\path\to\ar-python.exe"
$env:DLLM_PYTHON = ".\.venv\Scripts\python.exe"

& $env:AR_PYTHON -m pip install -e ".[dev,gpu,training]"
& $env:DLLM_PYTHON -m pip install -e ".[dev,dllm,dllm-training]"
```

### Linux / WSL2 vLLM

vLLM `0.25.x` 使用 Linux GPU wheel。Windows 主机在 WSL2 的 Linux 文件系统中使用兼容的 Python：

```bash
python3.12 -m pip install --upgrade pip
python3.12 -m pip install -e ".[dev,vllm]"
```

## 统一复现入口

[`run_reproduction.py`](experiments/run_reproduction.py) 调度两侧的准备、训练和推理。两个 Python 路径分别
指向上述解释器。`smoke` 使用 1 题、缩短预算、AR 一次 GRPO 更新和 dLLM 的 CPU VRPO 反向传播预检；
每个真实 LLaDA 推理子进程结束后释放模型显存。

解释器选择顺序为：CLI 的 `--ar-python` / `--dllm-python`、环境变量 `AR_PYTHON` / `DLLM_PYTHON`、
启动统一入口的当前 Python。单侧运行可省略两个解释器参数：

```powershell
python experiments\run_reproduction.py `
  --family dllm --stage inference --profile smoke --tag local-dllm
```

环境变量方式无需在命令中重复路径：

```powershell
python experiments\run_reproduction.py `
  --family both --stage all --profile smoke --tag local-check `
  --ar-methods base mh conditional_is rl_sample `
  --dllm-methods base trajectory_power_mh conditional_is_reduced_layer_proposal `
  --components quality replay
```

大显存机器上的完整训练和推理使用相同入口。dLLM 阶段依次构造公开训练集偏好对、续跑 VRPO LoRA、加载
adapter，并运行配置中的推理方法；AR 阶段依次准备数据与权重、续跑 GRPO 和运行所选实验族。

CLI 显式路径会覆盖环境变量：

```powershell
python experiments\run_reproduction.py `
  --family both --stage all --profile full --tag full-reproduction `
  --ar-python $env:AR_PYTHON `
  --dllm-python $env:DLLM_PYTHON
```

主要 CLI 参数：

| 参数 | 作用 |
| --- | --- |
| `--family arllm\|dllm\|both` | 运行一侧或成对运行 |
| `--stage prepare\|train\|inference\|all` | 选择模型准备、RL 训练、推理或完整流水线 |
| `--profile smoke\|full` | 低成本实现检查或正式配置 |
| `--ar-methods ...`、`--dllm-methods ...` | 选择具体推理方法 |
| `--components ...` | 选择质量、matched target、replay、动态 IS、异步、pass@k、消融、infra 等实验族 |
| `--ar-python ...`、`--dllm-python ...` | 覆盖环境变量与当前解释器 |
| `--limit`、`--max-train-steps` 等 | 覆盖样本数和训练预算 |
| `--dry-run` | 只写入清单并打印下一层命令，不启动训练或推理 |

AR-LLM 的 `quality` 组件覆盖 base、beam、Best-of-$`N`$、MH、两种条件 IS、三种 verifier
方法以及 GRPO 随机/贪心解码；其余组件覆盖 replay、动态 IS、异步执行、pass@$`k`$、消融、分布审计、
infra 和 vLLM。dLLM 入口覆盖对应的 base、block beam、Best-of-$`N`$、trajectory-MH、条件 IS、
低层 proposal 的截断/无截断/无校正版本、verifier-MH/IS、VRPO 随机/贪心解码及 trajectory replay。

两侧也可独立启动：

```powershell
& $env:AR_PYTHON experiments\arllm\run_arllm_suite.py `
  --stage all --profile full --components quality replay dynamic_is async passk infra

& $env:DLLM_PYTHON experiments\dllm\run_llada_suite.py `
  --profile full --vrpo train --tag full-dllm
```

所有入口写入命令清单和已完成子任务数；原有单实验脚本继续可用。完整统计口径见
[GSM8K 实验设计](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md)。

## 测试与目录

```powershell
python -m pytest

# 或使用仓库中已有的解释器
.\.venv\Scripts\python -m pytest
```

| 路径 | 内容 |
| --- | --- |
| `src/inference_scaling/arllm/` | AR-LLM 的 MH、IS、replay、Transformers 与 vLLM 后端 |
| `src/inference_scaling/dllm/` | LLaDA-MoE 的 block 生成、MH、IS、replay 与 VRPO |
| `src/inference_scaling/shared/` | 两侧共用的逐步生成、IS/replay 权重、MH 接受核、数据评测、随机数和计算账本 |
| `configs/` | 模型、数据与预算配置 |
| `experiments/arllm/`、`experiments/dllm/` | 两侧独立复现入口与 dLLM 训练脚本 |
| `experiments/run_reproduction.py` | 成对调度 AR-LLM 与 dLLM 的统一入口 |
| `tests/` | 分布、实现一致性和结果处理测试 |
| `docs/` | 算法与实现、实验协议、报告和验证记录 |
| `results/` | 纳入版本控制的机器可读汇总 |

顶层旧模块名保留为 AR-LLM 兼容导出，原有导入路径和单实验脚本无需修改；新增代码使用上述三类目录。
