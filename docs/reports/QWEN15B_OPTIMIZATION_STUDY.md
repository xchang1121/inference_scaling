# Qwen2.5-1.5B 推理扩展优化研究

## 研究范围

本轮研究只评估 Qwen2.5-1.5B-Instruct 的自回归推理。Qwen2.5-0.5B-Instruct 仅可作为候选或 rollout
proposal，其 forward token、FLOPs、显存与墙钟分别记录。dLLM/LLaDA 不进入本轮实验，也不据此形成效果结论。

数据使用固定 revision 的公开 GSM8K 测试集，硬件为一张 RTX 3090 24 GiB，主后端为 Transformers。基础
设置、提示模板、答案提取和计算账本沿用 [GSM8K 实验设计](../experiments/GSM8K_EXPERIMENT_DESIGN.md)。

## 选择规则

候选方法先在固定的 8 题、2 个独立 draw 上筛选。通过筛选的方法在固定的 32 题、4 个独立 draw 上确认，
并使用 10,000 次成对 bootstrap 计算区间。执行优化需要保持同一请求的采样分布；算法改动则同时报告目标
分布是否保持、有限预算偏差和实际任务质量。

进入最终组合需要满足下列条件之一：

1. 准确率相对基线的成对差值不低于 `-3.125` 个百分点，同时墙钟或主模型 FLOPs 至少下降 5%；
2. 墙钟与主模型 FLOPs均未增加超过 5%，同时准确率至少提高 `3.125` 个百分点；
3. 对只改变调度的精确执行优化，耦合随机流下输出一致，且墙钟或主模型 FLOPs 至少下降 5%。

replay、缓存和 proposal 构建分别报告冷启动与稳态成本。只在特定奖励延迟、重复请求次数或后端成立的结果
标记为“条件启用”，不进入通用默认链。筛选未通过的方法保留实现与记录，但默认入口不调度。

机器可读状态位于
[`attempt_registry.json`](../../results/arllm/qwen15b_optimization/attempt_registry.json)。状态含义如下：

| 状态 | 含义 |
| --- | --- |
| `planned` / `screening` | 尚未形成最终结论 |
| `accepted` / `accepted_existing` | 通过本轮确认或已有同口径结果，允许进入默认组合 |
| `conditional` | 仅在登记的前提下有收益 |
| `rejected` | 未通过收益判据，保留在非默认实验路径 |

## 文献依据与候选方法

### 迭代 SIR

普通条件 IS 在每个 block 生成有限候选池并执行一次 Sampling-Importance-Resampling（SIR）。有限候选数下，
该输出只是目标条件分布的 self-normalized 近似。iterated SIR（i-SIR）把上一轮选中的“候选 block + 用于估计
其权重的 rollout”作为下一轮候选池中的一个元素，再从 proposal 生成其余元素并重采样。扩展状态上的 proposal
密度乘以非负无偏权重估计后，其候选边缘分布恰为所需条件目标；因此任意有限池大小下，i-SIR 转移都保持该
目标不变。候选仍全部来自基础模型，off-policy completion 只通过 $`p/q`$ 修正候选权重。

[Samsonov et al. (2022)](https://papers.neurips.cc/paper_files/paper/2022/file/21c86d5b10cdc28664ccdadf0a29065a-Paper-Conference.pdf)
给出独立 proposal i-SIR 核、可逆性和有限迭代的 TV 几何界；
[Laitinen and Vihola (2025)](https://arxiv.org/abs/2512.00220)进一步研究 proposal 数与并行成本的权衡。
本轮在相同候选-rollout 组总数下比较 `(pool, updates)=(9,1),(5,2),(3,4)`，从而分离一次性大池与多轮复用。

公共 i-SIR 转移、Qwen block 适配、off-policy rollout 权重和显式 TV 界已经实现，并通过有限状态详细平衡、
on-policy/off-policy 候选边缘分布、rollout 生命周期与总长度测试。方法标识为 `iterated_conditional_is`，
只可显式选择，不属于默认方法集。一次 8 题筛选可用以下命令启动：

```powershell
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.gsm8k_reproduction `
  --config configs\gsm8k_quick.toml `
  --method iterated_conditional_is `
  --conditional-reward frozen_consensus `
  --iterated-pool-size 3 --iterated-updates 4 `
  --tag qwen15b-isir-n3-u4 --limit 8
```

### MH 后缀长度 proposal

固定生成长度下，每个后缀起点都定义一个保持目标分布不变的 MH 核。按与当前序列无关、全支持的固定概率
混合这些核，所得转移仍保持同一目标分布。短后缀权重较高可降低每次 proposal 的生成 token；保留完整后缀
的正概率则维持全局移动能力。本轮比较均匀起点、按后缀长度倒数加权和二进制多尺度长度，报告接受率、每次
接受改变的 token 数、墙钟、FLOPs 及有限轮次质量。

该消融依据 MH 转移核的混合闭包；通用接受率与不变性由
[Tierney (1994)](https://projecteuclid.org/journals/annals-of-statistics/volume-22/issue-4/Markov-Chains-for-Exploring-Posterior-Distributions/10.1214/aos/1176325750.full)
给出。代码仍对每个实际后缀使用完整正反 proposal 概率，不使用未经校正的截断。

### rollout 方差与提前停止

scrambled randomized quasi-Monte Carlo（RQMC）使每条随机流保持正确边缘分布，同时让同一候选的多条 rollout
在单位立方体上覆盖得更均匀。[Buchholz and Chopin (2018)](https://proceedings.mlr.press/v80/buchholz18a/buchholz18a.pdf)
给出 RQMC 与重要性采样/SMC 的组合。离散长序列和很小的 rollout 数可能削弱收益，因此该方法只作为消融，
以条件权重方差、ESS、准确率和墙钟决定是否保留。

当奖励和重要性比具有已知上下界时，可以在未完成全部 rollout 前计算每个候选最终权重的区间；只有一个候选
在所有剩余取值下仍会被同一固定重采样随机数选中时才停止。该规则要求逐次验证与完整计算得到相同 selected
index。若当前 self-consistency 奖励不能提供足够紧的界，结果预计表现为停止率低，并登记为未通过。

### 自回归执行优化

现有后端已实现连续批处理、共同前缀 KV 复用、评分缓存、流式奖励、冻结 replay-mixture MH proposal 和
SMC 条件后缀复用。相关依据包括 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、
[PagedAttention](https://doi.org/10.1145/3600006.3613165)、[Sarathi-Serve](https://arxiv.org/abs/2308.16369)
和[精确 speculative decoding](https://proceedings.mlr.press/v202/leviathan23a.html)。

新增消融使用 Qwen2.5-0.5B-Instruct 批量生成草稿，再由 1.5B target 按精确 speculative sampling 规则验证。
该路径的目标是减少 1.5B 逐 token decode 次数；0.5B 草稿计算不并入 1.5B FLOPs。验收同时检查合计 FLOPs、
草稿接受长度和单卡墙钟，防止用额外小模型计算掩盖总体成本。

## 已有消融的初始分类

| 方法 | 初始结论 | 当前默认状态 | 依据 |
| --- | --- | --- | --- |
| 连续批处理 | 正收益 | 启用 | 多个 workload 墙钟下降，输出统计量固定 |
| warm replay | 稳态正收益 | 启用；冷启动单列 | 稳态 FLOPs 与墙钟下降，约 7 次重复请求摊销构建成本 |
| 冻结 replay-mixture MH | 正收益 | 启用 | 完整 mixture Hastings 校正下在线墙钟下降 |
| 流式奖励 | 条件收益 | 关闭 | verifier 延迟足够大时可与生成重叠；廉价奖励无收益 |
| delayed-acceptance MH | 条件收益 | 关闭 | 需要昂贵精确奖励和有效 surrogate |
| MH proposal 预取 | 条件收益 | 关闭 | 以更多 proposal FLOPs 隐藏奖励等待 |
| 历史 token tree | 无稳定收益 | 关闭 | 墙钟区间覆盖无变化且 FLOPs 增加 |
| progressive rollout 分配 | 负收益 | 关闭 | 墙钟与 FLOPs 同时增加 |
| 方差--成本动态分配 | 无收益 | 关闭 | 质量与墙钟未改善，FLOPs 增加 |
| 无条件历史树 | 负收益 | 关闭 | 墙钟与 FLOPs 明显增加 |
| Transformers partial resume | 成本口径下负收益 | 关闭 | 墙钟下降但重复 prefill 使 FLOPs 超过三倍 |

上述初始分类来自[方法质量与计算量](GSM8K_3090_ALIGNED_RESULTS.md)和
[推理执行与 rollout 复用](RTX3090_ROLLOUT_INFRA.md)中的已完成 RTX 3090 结果。后续各项 Qwen 1.5B
消融将在本文件追加设置、数值、区间和进入最终组合的决定；中间调试日志不作为实验结果。
