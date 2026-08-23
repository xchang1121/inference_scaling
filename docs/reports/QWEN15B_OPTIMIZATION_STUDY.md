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
只可显式选择，不属于默认方法集。可续跑入口依次执行三种候选池结构、两个独立 draw 和聚合程序：

```powershell
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_isir_screen `
  --config configs\gsm8k_quick.toml --draws 2 `
  --tag qwen15b-isir-screen
```

入口将逐题原始记录保存在忽略目录 `results/gsm8k/`，把可复核的汇总和运行清单写入
`results/arllm/qwen15b_optimization/`。

筛选使用固定 8 题、2 个 draw、每个 block 9 个不同候选-rollout 状态。结果如下；FLOPs 和墙钟为两个
draw 的总和，因实际 EOS 与补全长度不同而存在小幅差异。

| pool $`N`$ | 更新 $`n`$ | 准确率 | 相对大池准确率 | 主模型 PFLOPs | 相对大池 FLOPs | 墙钟（秒） | 相对大池墙钟 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 1 | 18.75% | 0.00 pp | 0.5846 | 1.000 | 244.9 | 1.000 |
| 5 | 2 | 18.75% | 0.00 pp | 0.5909 | 1.011 | 245.9 | 1.004 |
| 3 | 4 | 25.00% | +6.25 pp，95% 区间 `[0,18.75]` | 0.5788 | 0.990 | 261.0 | 1.066 |

同一 8 题的已有标准条件 IS 为 37.50%、0.1239 PFLOPs 和 98.6 秒（一个 draw）。按 draw 归一化后，
`(N=3,n=4)` 的主模型 FLOPs 为标准条件 IS 的 2.34 倍，墙钟为 1.32 倍，准确率低 12.5 个百分点。
三种 i-SIR 结构均未通过质量--成本筛选，状态记为 `rejected`，不执行 32 题确认。有限池不变性实现仍作为
显式实验方法保留；其失败来源是当前可部署的冻结 pilot 奖励和额外候选成本，不是否定式 (8c)--(8d) 的
有限状态正确性。

### MH 后缀长度 proposal

固定生成长度下，每个后缀起点都定义一个保持目标分布不变的 MH 核。按与当前序列无关、全支持的固定概率
混合这些核，所得转移仍保持同一目标分布。短后缀权重较高可降低每次 proposal 的生成 token；保留完整后缀
的正概率则维持全局移动能力。本轮比较均匀起点、按后缀长度倒数加权和二进制多尺度长度，报告接受率、每次
接受改变的 token 数、墙钟、FLOPs 及有限轮次质量。

若 $`K_\ell`$ 表示固定重生成最后 $`\ell`$ 个 token 的 Hastings 核，$`\rho(\ell)`$ 与当前序列无关，
则实际核为

```math
K_\rho=\sum_{\ell=1}^{L}\rho(\ell)K_\ell,
\qquad
\pi K_\rho=\sum_{\ell=1}^{L}\rho(\ell)\pi K_\ell=\pi.
```

实现中的三种 $`\rho`$ 均对每个长度赋予正概率。`uniform` 保留既有随机流；`inverse_length` 采用
$`1/\ell`$ 权重；`multiscale` 以 10% 均匀概率保证全支持，再把 90% 概率分配到二进制尺度和完整后缀。
普通、批处理、delayed-acceptance、proposal 预取和冻结 replay-mixture 路径共用这一配置。有限状态测试已
验证归一化与全支持、批处理逐步一致性，以及三种分布都收敛到相同的可枚举目标。

可续跑筛选入口对两个 draw 使用正序与逆序执行，固定题目、$`\alpha`$、block size 和每个 block 的 MH
更新数，只改变后缀长度分布：

```powershell
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_mh_suffix_screen `
  --config configs\gsm8k_quick.toml --draws 2 `
  --tag qwen15b-mh-suffix-screen
```

该消融依据 MH 转移核的混合闭包；通用接受率与不变性由
[Tierney (1994)](https://projecteuclid.org/journals/annals-of-statistics/volume-22/issue-4/Markov-Chains-for-Exploring-Posterior-Distributions/10.1214/aos/1176325750.full)
给出。代码仍对每个实际后缀使用完整正反 proposal 概率，不使用未经校正的截断。

8 题、2 个 draw 的筛选结果如下。墙钟与计算量均为 16 次观测合计；相对值以同轮 `uniform` 为分母。

| 后缀分布 | 准确率 | 相对准确率 | 生成 token | 相对生成 token | 主模型 PFLOPs | 相对 FLOPs | 墙钟（秒） | 相对墙钟 | 接受率 | 平均 proposal 长度 | 接受后改变 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform` | 12.50% | 0.00 pp | 10,783 | 1.000 | 0.2120 | 1.000 | 320.9 | 1.000 | 74.61% | 34.79 | 6.32 |
| `inverse_length` | 18.75% | +6.25 pp，95% 区间 `[0,18.75]` | 5,697 | 0.528 | 0.2093 | 0.987 | 179.7 | 0.560 | 90.62% | 14.75 | 3.72 |
| `multiscale` | 31.25% | +18.75 pp，95% 区间 `[0,43.75]` | 7,582 | 0.703 | 0.2062 | 0.973 | 257.2 | 0.801 | 86.33% | 22.62 | 4.54 |

`inverse_length` 与 `multiscale` 均通过筛选。短后缀提高接受率并减少逐 token decode；当前 Transformers
实现会为每次 proposal 重新提交保留前缀，所以线性 dense-FLOPs 账本下降小于生成 token 和墙钟下降。
两种策略进入 32 题、4 个 draw 的确认。若两者均通过确认，默认策略按准确率排序，准确率相同时选择墙钟较低者；
另一策略仍作为显式速度优先配置保留。

确认结果如下。每行含 128 次观测，准确率差的区间以 GSM8K 题目为聚类单位；墙钟包含四轮不同系统负载，
逐 draw 数值在机器可读结果中保留。

| 后缀分布 | 准确率 | 相对准确率 | 生成 token | 相对生成 token | 主模型 PFLOPs | 相对 FLOPs | 墙钟（秒） | 相对墙钟 | 接受率 | 平均 proposal 长度 | 接受后改变 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform` | 12.50% | 0.00 pp | 90,544 | 1.000 | 1.7720 | 1.000 | 2,483.1 | 1.000 | 70.26% | 36.63 | 6.45 |
| `inverse_length` | 13.28% | +0.78 pp，95% 区间 `[-4.69,6.25]` | 45,455 | 0.502 | 1.7573 | 0.992 | 1,342.6 | 0.541 | 88.18% | 14.55 | 2.80 |
| `multiscale` | 17.97% | +5.47 pp，95% 区间 `[-2.34,14.84]` | 61,271 | 0.677 | 1.7647 | 0.996 | 1,473.5 | 0.593 | 82.57% | 22.21 | 4.24 |

两个非均匀分布均满足确认门槛。`multiscale` 按预先登记的准确率优先规则成为默认 MH 后缀分布；
`inverse_length` 作为速度优先配置保留。四轮 uniform 墙钟分别为 518.9、441.0、800.0 和 723.2 秒，表明
绝对墙钟受到系统负载影响；生成 token 减少 32.3%、四轮聚合墙钟减少 40.7% 以及相邻执行位置的同向结果共同
支持 `multiscale` 的成本收益。线性 dense-FLOPs 仅下降 0.41%，原因仍是 Transformers 路径重复提交保留前缀。

确认运行命令为：

```powershell
$env:PYTHONNOUSERSITE = "1"
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_mh_suffix_screen `
  --config configs\gsm8k_quick.toml --phase confirmation `
  --limit 32 --draws 4 --tag qwen15b-mh-suffix-confirm-149c545
```

`PYTHONNOUSERSITE=1` 用于隔离本机用户目录中的 NumPy 2.4.3，避免与 Conda 环境内按 NumPy 1.x 编译的
SciPy/scikit-learn 冲突；该设置不改变模型或算法。

### rollout 方差与提前停止

scrambled randomized quasi-Monte Carlo（RQMC）使每条随机流保持正确边缘分布，同时让同一候选的多条 rollout
在单位立方体上覆盖得更均匀。[Buchholz and Chopin (2018)](https://proceedings.mlr.press/v80/buchholz18a/buchholz18a.pdf)
给出 RQMC 与重要性采样/SMC 的组合。仓库已实现经过数字扰动的 Sobol 点集和逐 token 逆 CDF：候选生成、
rollout 数、proposal、$`p/q`$ 和重采样随机数均保持不变，只替换 rollout 使用的均匀数。每条 rollout 的
边缘分布仍为原 proposal，因此条件权重估计保持无偏；点集内部不再独立，ESS 只作为权重离散程度的描述量。

实现限定于 Transformers 与表格后端。vLLM 当前不能注入逐 token 均匀数，因此显式拒绝该模式。消融使用
独立 pilot 冻结的逐序列奖励；批内自一致性奖励会随 rollout 相关性改变，不能用于这一成对比较。离散长序列和
很小的 rollout 数可能削弱收益，筛选将以多个独立 scramble 下的候选权重方差、准确率、生成 token、FLOPs
和墙钟决定是否进入确认。

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
