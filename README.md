# inference_scaling

本仓库研究如何在不修改或少量修改语言模型参数的情况下，直接控制推理时输出分布，并统一比较
Metropolis--Hastings（MH）、重要性采样（IS）、off-policy rollout replay 与 GRPO 的质量和计算量。

实测结论分为两份互补报告：[GSM8K 方法质量与计算量实验](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md)
侧重方法质量与算法预算，[RTX 3090 推理基础设施优化汇总](docs/reports/RTX3090_ROLLOUT_INFRA.md)侧重墙钟、
FLOPs、吞吐和复用影响。实验协议、机器可读结果和工程验证已分别归档。

## 从哪里开始

| 目的 | 入口 |
| --- | --- |
| 比较方法准确率、pass@k 和共享目标 | [GSM8K 方法质量与计算量实验](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md) |
| 比较基础设施优化的墙钟、FLOPs 与复用影响 | [RTX 3090 推理基础设施优化汇总](docs/reports/RTX3090_ROLLOUT_INFRA.md) |
| 复现实验或核对公平性约束 | [GSM8K 统一实验设计](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md) |
| 对照数学对象与代码入口 | [算法映射](docs/methods/ALGORITHM_MAP.md) |
| 查看批处理、KV 复用和计量方式 | [推理性能设计](docs/methods/PERFORMANCE_DESIGN.md) |
| 查看 rollout 生成、复用与验证优化 | [rollout 生成与复用](docs/methods/ROLLOUT_ACCELERATION.md) |
| 使用或成对测量 vLLM | [vLLM 推理运行时](docs/methods/VLLM_RUNTIME.md) |
| 查找全部文档 | [文档导航](docs/README.md) |
| 查找机器可读结果 | [结果索引](results/README.md) |

## 已实现的方法

| 标识 | 候选来源 | rollout / proposal | 作用 |
| --- | --- | --- | --- |
| `mh` | 当前完整序列 | 基座模型后缀 proposal | 直接采样固定长度幂分布或显式奖励目标 |
| `conditional-is` | 基座模型 | on-policy completion | 用条件能量重新分配基座候选的选择概率 |
| `base-replay` | 基座模型 | 历史 off-policy rollout + fresh tail | 在不改变候选来源的前提下复用 rollout |
| `dynamic-is` | base/辅助 proposal 混合 | 动态 proposal + 外层 IS | 支持候选层修正与方差—成本预算分配 |
| `progressive-is` | 基座模型 | pilot 后冻结独立 evaluation 预算 | 根据实际 rollout 成本分配预算而不让 pilot 进入最终估计 |
| `smc-forest` | 基座模型粒子 | 可继承的条件后缀 reservoir | 逐 block 重采样并复用仍满足条件分布的 rollout 后缀 |

这些路径共享后端、请求级随机数、概率评分、token/FLOPs 账本和诊断接口。算法实现位于
`src/inference_scaling/algorithms/`；GSM8K 对照实现位于 `experiments/`。

## 方法效果概览

以下数据来自 32 道固定 GSM8K 测试题、`Qwen2.5-1.5B-Instruct` 和单张 RTX 3090。各方法的奖励目标
并不完全相同，因此这里只比较单次生成的任务准确率；共享奖励目标的受控比较见完整报告。

| 方法 | 正确数 / 32 | pass@1 |
| --- | ---: | ---: |
| Base | 13 | 40.625% |
| 幂分布 MH | 12 | 37.500% |
| 标准条件 IS | 21 | 65.625% |
| 0.5B proposal 条件 IS | 15 | 46.875% |
| GRPO 随机采样 | 22 | 68.750% |

结论可以概括为：

- 标准条件 IS 与本地 GRPO 相差 -3.125 个百分点，配对区间跨 0；当前样本只支持二者准确率接近，
  不支持完整输出分布相同。
- 0.5B off-policy proposal 条件 IS 比标准版本低 18.75 个百分点，当前实现没有保持质量。
- 幂分布 MH 没有相对 Base 提升 GSM8K 准确率；在统一正确性奖励的 oracle 诊断中，MH 与标准 IS
  则得到接近的点估计。
- warm replay、动态候选和方差—成本分配的质量差异区间均跨 0，但样本量不足以宣称质量等价。

这些结果是单卡、有限题目上的实测，不代表完整 1,319 题评测或完整序列分布等价。统计区间、pass@k、
共享目标实验、质量消融和限制均保留在准确率报告中。

## Infra 优化概览

下表统一使用“优化路径 / 对照路径”；小于 1 表示减少。不同实验组的绝对时间不能横比，完整 setting、
冷启动成本和误差线见 [RTX 3090 推理基础设施优化汇总](docs/reports/RTX3090_ROLLOUT_INFRA.md)。

| 优化 | 对照 | 墙钟因子 | 逻辑 FLOPs 因子 | 当前结论 |
| --- | --- | ---: | ---: | --- |
| 连续批处理 | 同方法逐 prompt | 0.206×–0.952× | 1.003×–1.177× | 提升硬件利用率，不减少算法计算量 |
| warm replay 在线阶段 | fresh-only | 0.859× | 0.766× | 热缓存有效；第 7 次重复查询才覆盖建库成本 |
| 部分 rollout 续跑 | 丢弃部分 token 后重启 | 0.793× | 3.346× | 少生成 23.1%，但 token-only 恢复重复 prefill |
| 流式 IS（0.2 s verifier） | 整批完成后提交 verifier | 0.671× | 1.000× | 提前消化有限 CPU worker 队列 |
| 历史树，始终草稿 | 无草稿 | 2.162× | 1.660× | 低接受率造成明显退化 |
| 历史树，负载感知 | 无草稿 | 0.986× | 1.006× | 主要用于保护长尾吞吐 |
| MH proposal-tree 预取（0.2 s 奖励） | 普通 MH | 0.817× | 1.267× | 用作废分支的 FLOPs 隐藏奖励延迟 |
| delayed acceptance（0.2 s 奖励） | 普通 MH | 0.827× | 1.000× | 精确奖励调用降到 0.556× |
| replay 混合 MH proposal，在线 | base suffix proposal | 0.534× | 1.003× | 历史命中把串行生成改成并行评分 |
| SMC 条件后缀复用 | 同一 SMC 不复用 | 0.856× | 0.963× | 同时减少墙钟、主模型计算和 fresh rollout |

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

vLLM 是独立的可选运行时，固定在 `vllm>=0.25,<0.26`。其官方 GPU wheel 要求 Linux，并不原生支持
Windows；Windows 主机请在 WSL2 中新建 Linux 虚拟环境，不要复用上面的 Windows `.venv`：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,vllm]"
python -c "import torch, vllm; print(torch.cuda.is_available(), vllm.__version__)"
```

安装、精确评分边界、显存划分和配置说明见
[vLLM 推理运行时](docs/methods/VLLM_RUNTIME.md)。

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
  --with-dynamic-is `
  --with-async `
  --with-ablations `
  --with-budget-curve `
  --with-length-ablation `
  --ablation-limit 8
```

`--with-dynamic-is` 会同时读取 `configs/gsm8k_3090_dynamic_is.toml`，其中只保存动态候选 mixture、缓存
条数、独立 design 样本数和每候选总 rollout 数，因而不会改变已经固定的主网格配置指纹。

在 Linux/WSL2 上使用异步 vLLM 时，为同一命令增加 `--backend vllm`。若要先回答“vLLM 相对
Transformers 到底快多少”，建议运行会严格核对 setting 的成对入口：

```bash
export PYTHONPATH=src
python experiments/run_vllm_backend_benchmark.py \
  --config configs/gsm8k_3090_aligned.toml \
  --limit 32 \
  --workers 8 \
  --tag rtx3090
```

这里的加速分母始终是同模型、同 dtype、同请求网格和同硬件下的 Transformers 并发路径；既有 3090
结果尚未用 vLLM 重跑，不能直接当作 vLLM 结果。

rollout 基础设施使用单独的可恢复 benchmark 入口；`--section decode` 与
`--section algorithm` 可以分开运行：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\benchmark_rollout_infra.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers.json
```

Linux/WSL2 上把后端改为 `vllm` 即可使用同一 workload 和结果 schema。实现原理、配置与哪些成本必须
分列见 [rollout 生成与复用](docs/methods/ROLLOUT_ACCELERATION.md)。

部分 rollout、流式 frozen-design IS、随机历史草稿和三种 MH 执行优化使用独立入口：

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python experiments\benchmark_is_mh_reuse.py `
  --backend transformers --dtype bfloat16 --section all --seed 20260812 `
  --output results\infra\rtx3090_transformers_is_mh_seed20260812.json
```

该实验中的 0.2 s verifier 是明确标注的受控 infra 诊断，不用于方法准确率排序。三 seed 聚合命令和
每项优化的分母见 [RTX 3090 推理基础设施优化汇总](docs/reports/RTX3090_ROLLOUT_INFRA.md)。

主表、计算量汇总、分布审计、pass@k、消融和绘图需要在网格完成后运行只读后处理器。完整命令、输出
文件和恢复规则见 [GSM8K 统一实验设计](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md)。

## 数据与结果管理

- `results/gsm8k_3090/`：主实验的正式机器可读汇总；
- `results/training/`：GRPO 训练摘要；
- `results/validation/`：quick 与后端工程检查，不作为最终结论；
- `results/infra/`：RTX 3090 rollout 基础设施的原始重复运行与机器可读聚合；
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
