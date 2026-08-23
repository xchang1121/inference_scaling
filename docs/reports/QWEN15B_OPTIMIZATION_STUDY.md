# Qwen2.5-1.5B 推理扩展优化研究

## 研究范围

本轮研究只评估 Qwen2.5-1.5B-Instruct 的自回归推理。Qwen2.5-0.5B-Instruct 仅可作为候选或 rollout
proposal，其参与前向计算的 token 位置数、FLOPs、显存与墙钟分别记录。dLLM/LLaDA 不进入本轮实验，也不据此形成效果结论。

数据使用固定 revision 的公开 GSM8K 测试集，硬件为一张 RTX 3090 24 GiB，主后端为 Transformers。基础
设置、提示模板、答案提取和计算量统计沿用 [GSM8K 实验设计](../experiments/GSM8K_EXPERIMENT_DESIGN.md)。

## 选择规则

候选方法先在固定的 8 题、2 次独立重复上筛选。通过筛选的方法在固定的 32 题、4 次独立重复上确认，
并使用 10,000 次成对自助法（bootstrap）重采样计算区间。执行优化需要保持同一请求的采样分布；算法改动则同时报告目标
分布是否保持、有限预算偏差和实际任务质量。

进入最终组合需要满足下列条件之一：

1. 准确率相对基线的成对差值不低于 `-3.125` 个百分点，同时墙钟或主模型 FLOPs 至少下降 5%；
2. 墙钟与主模型 FLOPs 均未增加超过 5%，同时准确率至少提高 `3.125` 个百分点；
3. 对只改变调度的精确执行优化，每个请求使用相同随机数时输出一致，且墙钟或主模型 FLOPs 至少下降 5%。

replay、缓存和 proposal 构建分别报告历史库构建成本与在线成本。只在特定奖励延迟、重复请求次数或后端成立
的结果标记为“条件启用”，不进入通用默认路径。筛选未通过的方法保留实现与记录，但默认入口不调度。

机器可读状态位于
[`attempt_registry.json`](../../results/arllm/qwen15b_optimization/attempt_registry.json)。状态含义如下：

| 状态 | 含义 |
| --- | --- |
| `planned` / `screening` | 尚未形成最终结论 |
| `accepted` / `accepted_existing` | 通过本轮确认，或已有采用相同实验设置和统计方法的结果，允许进入默认组合 |
| `conditional` | 仅在登记的前提下有收益 |
| `rejected` | 未通过收益判据，保留在非默认实验路径 |

## 文献依据与候选方法

### 迭代 SIR

普通条件 IS 在每个生成块产生有限候选池并执行一次采样—重要性加权—重采样
（Sampling-Importance-Resampling，SIR）。有限候选数下，
该输出只是使用归一化权重得到的有限样本近似。iterated SIR（i-SIR）把上一轮选中的“候选块 + 用于估计
其权重的 rollout”作为下一轮候选池中的一个元素，再从 proposal 生成其余元素并重采样。扩展状态上的 proposal
概率乘以非负无偏权重估计后，其候选边缘分布恰为所需条件目标；因此任意有限池大小下，i-SIR 转移都保持该
目标不变。候选仍全部来自基础模型，off-policy 补全只通过 $`p/q`$ 修正候选权重。

[Samsonov et al. (2022)](https://papers.neurips.cc/paper_files/paper/2022/file/21c86d5b10cdc28664ccdadf0a29065a-Paper-Conference.pdf)
给出独立 proposal i-SIR 核、可逆性和有限迭代下总变差距离（TV）的几何收敛界；
[Laitinen and Vihola (2025)](https://arxiv.org/abs/2512.00220)进一步研究 proposal 数与并行成本的权衡。
本轮在相同候选-rollout 组总数下比较 `(pool, updates)=(9,1),(5,2),(3,4)`，用于区分一次使用大候选池与
多轮复用的影响。

公共 i-SIR 转移、Qwen 分块适配、off-policy rollout 权重和显式 TV 界已经实现，并通过有限状态细致平衡、
on-policy/off-policy 候选边缘分布、rollout 记录状态与总长度测试。方法标识为 `iterated_conditional_is`，
需显式选择，默认方法集不调度。可续跑入口依次执行三种候选池结构、两次独立重复和聚合程序：

```powershell
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_isir_screen `
  --config configs\gsm8k_quick.toml --draws 2 `
  --tag qwen15b-isir-screen
```

入口将逐题原始记录保存在忽略目录 `results/gsm8k/`，把可复核的汇总和运行清单写入
`results/arllm/qwen15b_optimization/`。

筛选使用固定 8 题、2 次独立重复、每个生成块 9 个不同候选-rollout 状态。结果如下；FLOPs 和墙钟为两次
重复的总和，因实际 EOS 与补全长度不同而存在小幅差异。

| 候选池大小 $`N`$ | 更新 $`n`$ | 准确率 | 相对大池准确率 | 主模型 PFLOPs | 相对大池 FLOPs | 墙钟（秒） | 相对大池墙钟 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 1 | 18.75% | 0.00 pp | 0.5846 | 1.000 | 244.9 | 1.000 |
| 5 | 2 | 18.75% | 0.00 pp | 0.5909 | 1.011 | 245.9 | 1.004 |
| 3 | 4 | 25.00% | +6.25 pp，95% 区间 `[0,18.75]` | 0.5788 | 0.990 | 261.0 | 1.066 |

同一 8 题的已有标准条件 IS 为 37.50%、0.1239 PFLOPs 和 98.6 秒（一次重复）。按重复次数归一化后，
`(N=3,n=4)` 的主模型 FLOPs 为标准条件 IS 的 2.34 倍，墙钟为 1.32 倍，准确率低 12.5 个百分点。
三种 i-SIR 结构均未通过质量—成本筛选，状态记为 `rejected`，不执行 32 题确认。有限候选池下保持目标分布
不变的实现仍作为显式实验方法保留，式 (8c)--(8d) 的有限状态正确性仍成立；本组质量和成本结果来自冻结的
初始估计奖励与额外候选计算。

### MH 后缀长度 proposal

固定生成长度下，每个后缀起点都定义一个保持目标分布不变的 MH 核。按与当前序列无关、全支持的固定概率
混合这些核，所得转移仍保持同一目标分布。短后缀权重较高可降低每次 proposal 的生成 token；保留完整后缀
的正概率则维持全局移动能力。本轮比较均匀起点、按后缀长度倒数加权和以 2 的幂为主的多尺度长度，报告接受率、每次
接受改变的 token 数、墙钟、FLOPs 及有限轮次质量。

若 $`K_\ell`$ 表示固定重生成最后 $`\ell`$ 个 token 的 Hastings 核，$`\rho(\ell)`$ 与当前序列无关，
则实际核为

```math
K_\rho=\sum_{\ell=1}^{L}\rho(\ell)K_\ell,
\qquad
\pi K_\rho=\sum_{\ell=1}^{L}\rho(\ell)\pi K_\ell=\pi.
```

实现中的三种 $`\rho`$ 均对每个长度赋予正概率。`uniform` 保留既有随机数序列；`inverse_length` 采用
$`1/\ell`$ 权重；`multiscale` 以 10% 均匀概率保证全支持，再把 90% 概率分配到 2 的幂长度和完整后缀。
普通、批处理、两阶段延迟接受、proposal 预取和冻结历史混合 proposal 路径共用这一配置。有限状态测试已
验证归一化与全支持、批处理逐步一致性，以及三种分布都收敛到相同的可枚举目标。

可续跑筛选入口对两次重复使用正序与逆序执行，固定题目、$`\alpha`$、块大小和每个生成块的 MH
更新数，只改变后缀长度分布：

```powershell
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_mh_suffix_screen `
  --config configs\gsm8k_quick.toml --draws 2 `
  --tag qwen15b-mh-suffix-screen
```

若干 MH 转移核保持同一个目标分布时，按固定概率混合这些转移核后仍保持该分布。本消融使用这一性质；
通用接受率与不变性由
[Tierney (1994)](https://projecteuclid.org/journals/annals-of-statistics/volume-22/issue-4/Markov-Chains-for-Exploring-Posterior-Distributions/10.1214/aos/1176325750.full)
给出。代码仍对每个实际后缀使用完整正反 proposal 概率，不使用未经校正的截断。

8 题、2 次独立重复的筛选结果如下。墙钟与计算量均为 16 次观测合计；相对值以同轮 `uniform` 为分母。

| 后缀分布 | 准确率 | 相对准确率 | 生成 token | 相对生成 token | 主模型 PFLOPs | 相对 FLOPs | 墙钟（秒） | 相对墙钟 | 接受率 | 平均 proposal 长度 | 接受后改变 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform` | 12.50% | 0.00 pp | 10,783 | 1.000 | 0.2120 | 1.000 | 320.9 | 1.000 | 74.61% | 34.79 | 6.32 |
| `inverse_length` | 18.75% | +6.25 pp，95% 区间 `[0,18.75]` | 5,697 | 0.528 | 0.2093 | 0.987 | 179.7 | 0.560 | 90.62% | 14.75 | 3.72 |
| `multiscale` | 31.25% | +18.75 pp，95% 区间 `[0,43.75]` | 7,582 | 0.703 | 0.2062 | 0.973 | 257.2 | 0.801 | 86.33% | 22.62 | 4.54 |

`inverse_length` 与 `multiscale` 均通过筛选。短后缀提高接受率并减少逐 token 生成；当前 Transformers
实现会为每次 proposal 重新提交保留前缀，所以按参数量和 token 数估算的 FLOPs 降幅小于生成 token
和墙钟降幅。两种策略进入 32 题、4 次独立重复的确认。若两者均通过确认，默认策略按准确率排序，准确率相同时选择墙钟较低者；
另一策略仍作为显式速度优先配置保留。

确认结果如下。每行含 128 次观测，准确率差的区间以 GSM8K 题目为聚类单位；墙钟包含四轮不同系统负载，
每次重复的数值在机器可读结果中保留。

| 后缀分布 | 准确率 | 相对准确率 | 生成 token | 相对生成 token | 主模型 PFLOPs | 相对 FLOPs | 墙钟（秒） | 相对墙钟 | 接受率 | 平均 proposal 长度 | 接受后改变 token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform` | 12.50% | 0.00 pp | 90,544 | 1.000 | 1.7720 | 1.000 | 2,483.1 | 1.000 | 70.26% | 36.63 | 6.45 |
| `inverse_length` | 13.28% | +0.78 pp，95% 区间 `[-4.69,6.25]` | 45,455 | 0.502 | 1.7573 | 0.992 | 1,342.6 | 0.541 | 88.18% | 14.55 | 2.80 |
| `multiscale` | 17.97% | +5.47 pp，95% 区间 `[-2.34,14.84]` | 61,271 | 0.677 | 1.7647 | 0.996 | 1,473.5 | 0.593 | 82.57% | 22.21 | 4.24 |

两个非均匀分布均满足确认门槛。`multiscale` 按预先登记的准确率优先规则成为默认 MH 后缀分布；
`inverse_length` 作为速度优先配置保留。四轮 uniform 墙钟分别为 518.9、441.0、800.0 和 723.2 秒，表明
绝对墙钟受到系统负载影响；生成 token 减少 32.3%、四轮聚合墙钟减少 40.7% 以及相邻执行位置的同向结果共同
支持 `multiscale` 的成本收益。按参数量和 token 数估算的 FLOPs 仅下降 0.41%，原因仍是 Transformers 路径
重复提交保留前缀。

确认运行命令为：

```powershell
$env:PYTHONNOUSERSITE = "1"
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_mh_suffix_screen `
  --config configs\gsm8k_quick.toml --phase confirmation `
  --limit 32 --draws 4 --tag qwen15b-mh-suffix-confirm-149c545
```

`PYTHONNOUSERSITE=1` 用于隔离本机用户目录中的 NumPy 2.4.3，避免与 Conda 环境内按 NumPy 1.x 编译的
SciPy/scikit-learn 冲突；该设置不改变模型或算法。

<a id="qwen15b-mh-stack"></a>
### 多尺度后缀与冻结 replay 的组合

多尺度后缀分布和冻结 replay proposal 可以直接组合。前者只改变后缀长度的固定混合权重；后者只改变给定
长度下的后缀 proposal，并将新旧后缀的完整混合概率写入 Hastings 比。两项改动分别保持同一目标分布，
组合后的转移核仍保持该分布。历史库必须在链开始前冻结，并与当前提示、模型和采样策略匹配。在线执行顺序见
[默认后缀 MH](../methods/ALGORITHMS.md#alg-qwen-default-mh)。

组合消融固定一条 GSM8K 提示、BF16、长度 32、4 条链、每链 8 次更新、8 条历史序列和奖励温度 0.3，
只改变后缀长度分布与是否使用冻结 replay。四组配置使用相同的三个随机种子配对比较；下表给出均值与样本标准差：

| proposal | 后缀分布 | 在线墙钟（秒） | 主模型 PFLOPs | 接受率 | 历史 proposal 比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| 基础模型 | `uniform` | 12.555 ± 0.610 | 0.016783 | 61.46% | 0% |
| 基础模型 | `multiscale` | 9.012 ± 1.469 | 0.016783 | 69.79% | 0% |
| 冻结 replay 混合 | `uniform` | 6.550 ± 2.064 | 0.016826 | 70.83% | 34.38% |
| 冻结 replay 混合 | `multiscale` | 4.484 ± 0.463 | 0.016822 | 80.21% | 30.21% |

组合路径相对 `base + uniform` 的在线墙钟因子为 `0.357 ± 0.026×`，主模型 FLOPs 因子为
`1.002 ± 0.001×`。在无 replay 路径上加入多尺度调度，墙钟因子为 `0.716×`；在多尺度路径上加入
replay，墙钟因子为 `0.503×`；在 uniform replay 上加入多尺度调度，墙钟因子为 `0.754×`。两项收益在
本组中可以叠加。

replay 缓存构建平均为 0.657 秒和 0.001136 PFLOPs。三个随机种子中，“缓存构建 + 首次组合查询”的
墙钟都低于一次 `base + uniform` 查询；在线主模型 FLOPs 略高，计入缓存构建后，任意查询次数下的平均
FLOPs 均高于基线。因此该组合登记为墙钟默认路径，FLOPs 优先场景仍使用基础模型 proposal。机器可读
结果由下列命令生成：

```powershell
$env:PYTHONNOUSERSITE = "1"
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_mh_stack `
  --config configs\gsm8k_quick.toml `
  --output results\arllm\qwen15b_optimization\mh_replay_multiscale_stack.json
```

### rollout 方差与提前停止

随机化拟蒙特卡洛（randomized quasi-Monte Carlo，RQMC）保持每条 rollout 的 proposal 边缘分布，同时改变同一候选内多条
rollout 的联合分布。本轮比较两种实现。`scrambled_sobol` 为每个 token 位置注入经过随机置乱的 Sobol
坐标；`arithmetic_lattice` 使用一维随机平移等距格点，并通过算术采样（Arithmetic Sampling）的递归区间映射将每个
格点递推为完整序列。
[Arithmetic Sampling](https://proceedings.mlr.press/v202/vilnis23a.html) 和
[QuasiMoTTo](https://arxiv.org/abs/2607.01179) 给出后一构造；
[Buchholz and Chopin (2018)](https://proceedings.mlr.press/v80/buchholz18a/buchholz18a.pdf)讨论 RQMC 与
重要性采样/SMC 的组合。

两种设计均保持候选、rollout 数、proposal、$`p/q`$ 和候选重采样随机数不变。每条 rollout 的边缘分布
仍为原 proposal，故条件权重的算术平均保持无偏。点集内部不独立，有效样本量（ESS）和单个点集的对数权重离散度只作
描述性指标；严格方差需要在固定候选上重复独立随机平移或随机置乱。消融使用独立初始估计样本固定的逐序列奖励，
避免批内自一致性奖励随 rollout 相关结构改变。Transformers 与表格后端支持这两种随机数构造；vLLM 当前不支持
请求级均匀数注入并显式拒绝该模式。

第一次重复的执行顺序为独立同分布采样（IID）、Sobol、格点，第二次重复完全反转；每个候选均使用 4 条 rollout：

```powershell
$env:PYTHONNOUSERSITE = "1"
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_rqmc_screen `
  --config configs\gsm8k_quick.toml --draws 2 --rollout-count 4 `
  --tag qwen15b-rqmc-screen
```

入口将逐题记录写入忽略目录 `results/gsm8k/`，将汇总与可续跑清单写入
`results/arllm/qwen15b_optimization/`。汇总核对每个成对运行的第一步候选 token 完全一致；单个点集内部的
对数权重离散程度与 ESS 均按描述性指标报告，不作为跨随机化方差估计。

8 题、2 次独立重复的筛选结果如下。准确率与成本均为 16 次观测的聚合值；三组均使用由初始估计样本确定的
固定众数奖励，
因此绝对准确率不与使用批内自一致性奖励的既有条件 IS 结果横向比较。

| rollout 均匀数 | 准确率 | 相对 IID 准确率 | 描述性 ESS | 对数权重离散度 | 生成 token | 主模型 PFLOPs | 相对 FLOPs | 墙钟（秒） | 相对墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IID | 18.75% | 0.00 pp | 2.678 | 7.342 | 69,718 | 0.3768 | 1.000 | 167.2 | 1.000 |
| 随机置乱 Sobol | 18.75% | 0.00 pp，95% 区间 `[0,0]` | 2.680 | 6.721 | 69,841 | 0.3848 | 1.021 | 169.5 | 1.013 |
| 随机平移等距格点 | 18.75% | 0.00 pp，95% 区间 `[-18.75,18.75]` | 2.852 | 5.204 | 69,036 | 0.3924 | 1.042 | 174.6 | 1.044 |

Sobol 和随机平移等距格点分别把点集内部的对数权重离散度降低 8.5% 和 29.1%；后者的描述性 ESS 提高
6.5%。在第一步候选 token 完全一致的 16 对观测中，Sobol 与随机平移等距格点的最大权重候选一致率分别为 43.75%
和 62.50%，两者实际所选候选的索引一致率均为 56.25%。这些差异表明 RQMC 确实改变了有限 rollout
权重估计，但没有提高本轮准确率。

随机平移等距格点减少 1.0% 生成 token，同时因后续候选选择和前缀长度改变而使统计的主模型 FLOPs 增加 4.2%；
逐步概率排序与区间更新使墙钟增加 4.4%。Sobol 同样没有质量或成本收益。两种方法均未通过登记门槛，
不执行 32 题确认，状态记为 `rejected`。默认条件 IS 继续使用 IID rollout；两种 RQMC 路径保留为显式
非默认消融。

当奖励和重要性比具有已知上下界时，可以在未完成全部 rollout 前计算每个候选最终权重的区间；只有一个候选
在所有剩余取值下仍会被同一固定重采样随机数选中时才停止。该规则要求逐次验证与完整计算得到相同的候选
索引。仓库已实现解析区间判定、分批 rollout 与越界检查。筛选使用由初始估计样本冻结的二值奖励，在
$`\tau=0.1`$ 的 on-policy 条件下声明对数权重界 $`[0,10]`$；完整路径与提前停止路径共享候选、rollout
随机种子和候选重采样均匀数。批内自一致性奖励不满足逐条固定奖励条件，不用于该消融。

筛选使用每批 2 条 rollout、每个候选最多 4 条；第一次重复先运行完整路径，第二次重复先运行提前停止路径：

```powershell
$env:PYTHONNOUSERSITE = "1"
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_bounded_stop_screen `
  --config configs\gsm8k_quick.toml --draws 2 --rollout-count 4 `
  --evaluation-batch-size 2 --log-weight-lower 0 --log-weight-upper 10 `
  --tag qwen15b-bounded-stop-screen
```

汇总要求每对运行的候选 token、各步骤所选候选的索引和最终输出逐项相同，再比较 rollout 跳过率、生成 token、
主模型 FLOPs 与墙钟。分批路径若未能提前停止，会重复提交前缀；这一额外成本计入结果。

8 题、2 次独立重复的筛选结果如下。两组的 16 对候选 token、各步骤所选候选的索引和最终输出均逐项相同，
准确率同为 18.75%。

| 执行方式 | rollout 计划/执行/跳过 | 评估批次数 | 生成 token | 相对 token | 主模型 PFLOPs | 相对 FLOPs | 墙钟（秒） | 相对墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 完整评估 | 822 / 822 / 0 | 63 | 69,718 | 1.000 | 0.3768 | 1.000 | 312.2 | 1.000 |
| 有界精确停止 | 822 / 754 / 68 | 102 | 66,629 | 0.956 | 0.4387 | 1.164 | 436.5 | 1.398 |

有界停止在 10 个候选选择步骤提前确定候选，跳过 8.27% rollout，并减少 4.43% 生成 token；但评估批次数
增加 61.9%，重复执行前缀预填充使主模型 FLOPs 增加 16.4%、墙钟增加 39.8%。两次采用相反执行顺序的重复中，
FLOPs 分别增加 11.9% 和 21.4%，墙钟分别增加 30.8% 和 51.8%，负收益不由单一执行顺序造成。该实现证明
了选中候选的精确提前停止规则，但在当前 Transformers 后端未通过成本门槛，不执行确认，状态记为
`rejected`。默认路径继续一次批量完成全部 rollout；有界停止仅保留为显式非默认实验。

### 自回归执行优化

现有后端已实现连续批处理、共同前缀 KV 复用、评分缓存、流式奖励、冻结历史混合 MH proposal 和
SMC 条件后缀复用。相关依据包括 [Orca](https://www.usenix.org/conference/osdi22/presentation/yu)、
[PagedAttention](https://doi.org/10.1145/3600006.3613165)、[Sarathi-Serve](https://arxiv.org/abs/2308.16369)
和[精确推测解码](https://proceedings.mlr.press/v202/leviathan23a.html)。

Qwen2.5-0.5B-Instruct 生成长度为 $`K`$ 的草稿，Qwen2.5-1.5B-Instruct 按精确推测采样
规则批量验证。接受草稿 token 的概率为 $`\min(1,p/q)`$；拒绝时从归一化的 $`(p-q)_+`$ 分布抽取替代
token，因此输出边缘分布与普通 1.5B 采样相同。主模型、草稿模型及合计 FLOPs 分列。

8 题、2 次独立重复、BF16、最长 128 token 的结果如下：

| 执行路径 | 墙钟（秒） | 输出 token/s | 主模型 FLOPs 因子 | 合计 FLOPs 因子 | 草稿接受率 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5B 普通生成 | 70.16 | 27.69 | 1.000× | 1.000× | — | 3922 MiB |
| 0.5B 草稿，$`K=2`$ | 74.24 | 26.36 | 1.055× | 1.439× | 86.50% | 4109 MiB |
| 0.5B 草稿，$`K=4`$ | 80.97 | 24.06 | 1.089× | 1.462× | 81.14% | 4147 MiB |
| 0.5B 草稿，$`K=8`$ | 89.26 | 22.29 | 1.170× | 1.555× | 72.66% | 4161 MiB |

$`K=2`$ 是最接近基线的草稿路径，但墙钟增加 5.8%、输出吞吐下降 4.8%、主模型 FLOPs 增加 5.5%，且
合计 FLOPs 增加 43.9%。同时加载两个模型也增加显存占用。该实现保留为显式实验后端；Qwen 1.5B 默认使用
只对 1.5B 主模型做连续批处理。表中准确率不用于选择执行后端：两条路径保证相同采样分布；有限样本下的
逐条输出可以不同。

<a id="qwen15b-is-stack"></a>
### IS replay、候选缓存与连续批处理的组合

组合实验只使用 Qwen2.5-1.5B 产生候选和新 rollout。Qwen2.5-0.5B 只产生 off-policy 历史记录，并在
在线阶段计算新补全在实际生成分布下的概率；两种模型参与前向计算的 token 位置数与 FLOPs 分列。五个实验组
依次为顺序纯新生成、连续批处理纯新生成、顺序已有历史 replay、顺序已有历史 replay 加候选缓存，以及已有
历史 replay 加候选缓存和连续批处理。候选、历史记录、重评分、新样本校正项与独立预留样本的完整顺序见
[默认条件 IS 与已有历史 replay](../methods/ALGORITHMS.md#alg-qwen-default-is)。

候选缓存复用 replay 匹配键构建时已经生成的同一组基础模型候选，省略在线阶段的重复候选生成。该改动不改变
候选、权重或重采样；顺序 replay 的缓存前后输出必须逐 token 相同。连续批处理为每个请求保留独立随机数序列，并要求
与对应顺序组逐 token 相同。缓存构建、在线阶段和包含建库的首次查询成本分别报告；另行给出的重复查询
“同一历史记录重复查询多少次后平均成本低于对照”的数字只作诊断，默认记录管理规则仍要求每条最终估计记录只使用一次。
正式运行采用 FP32，以保证保存的实际生成概率能在不同评分批次中通过数值复核。

可续跑入口为：

```powershell
$env:PYTHONNOUSERSITE = "1"
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_qwen15b_is_stack `
  --config configs\gsm8k_quick.toml `
  --output results\arllm\qwen15b_optimization\is_replay_batching_stack.json
```

实验固定 4 道题、3 个随机种子、FP32、4 个并发提示任务、最长 64 token、32-token 生成块、4 个候选，
每个非终止候选使用 1 条历史 rollout 和 1 条新 rollout。固定的 GSM8K 精确答案 verifier 只用于保持 replay
奖励不变；本组用于比较执行等价性和成本，不替代 32 题质量实验。三个随机种子的均值如下：

| 路径 | 缓存构建（秒） | 在线墙钟（秒） | 在线 1.5B PFLOPs | 在线 0.5B PFLOPs | 在线合计 PFLOPs | 含建库的首次查询墙钟（秒） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 纯新生成，顺序 | 0 | 8.336 | 0.016022 | 0 | 0.016022 | 8.336 |
| 纯新生成，连续批处理 | 0 | 4.131 | 0.017205 | 0 | 0.017205 | 4.131 |
| 已有历史 replay，顺序 | 5.905 | 8.351 | 0.014490 | 0.002562 | 0.017052 | 14.255 |
| 已有历史 replay，候选缓存 | 5.860 | 5.817 | 0.011696 | 0.002562 | 0.014258 | 11.677 |
| 已有历史 replay，候选缓存与连续批处理 | 2.498 | 3.133 | 0.012794 | 0.002801 | 0.015596 | 5.631 |

候选缓存相对顺序已有历史 replay 的在线墙钟因子为 `0.697 ± 0.009×`，1.5B FLOPs 因子为
`0.807 ± 0.004×`。在候选缓存上加入连续批处理，墙钟因子为 `0.537 ± 0.092×`；填充使该局部
对比的 1.5B FLOPs 和总 FLOPs 均为 `1.094×`。全部优化组合相对已经启用连续批处理的纯新生成路径，
墙钟因子为 `0.754 ± 0.093×`，1.5B FLOPs 因子为 `0.744×`，合计 FLOPs 因子为
`0.907×`。纯新生成顺序/批处理、replay 候选缓存前后、replay 顺序/批处理三类配对在三个随机种子中均逐
token 一致。平均 rollout 复用率为 31.62%。

若当前请求还需生成并评分历史记录，包含建库的首次查询相对连续批处理纯新生成路径的墙钟和总 FLOPs 分别为
`1.361×` 和 `1.941×`。默认路径只在存在匹配且尚未使用的最终估计记录时启用已有历史 replay；否则使用
连续批处理纯新生成路径。默认记录管理规则要求每条最终估计记录只使用一次；反复使用同一历史记录时计算出的成本
下降不计入默认结果。该组合登记为 `accepted`，适用范围是历史记录已经存在的在线阶段。dLLM 未进入本实验。

默认入口的接入方式如下：统一 AR CLI 默认传递 `--mh-suffix-schedule multiscale`，并同步用于单次质量评测
和 pass@$`k`$ 的批处理 MH；`replay` 组件把建库阶段返回的候选传给在线 `base_replay_step`。没有匹配且尚未使用的
历史记录时继续执行纯新生成路径。跨提示合批由连续批处理后端完成。配置文件保留均匀后缀基线，便于独立复现实验对照。

## 优化组合与方法状态

| 方法 | 初始结论 | 当前默认状态 | 依据 |
| --- | --- | --- | --- |
| 连续批处理 | 正收益 | 启用 | 多种请求集合的墙钟下降，输出统计量固定 |
| 已有历史 replay | 在线阶段正收益 | 有匹配且尚未使用的历史记录时启用 | 在线 FLOPs 与墙钟下降；新建历史记录的建库成本单列 |
| IS replay + 候选缓存 + 连续批处理 | 正收益 | 有匹配且尚未使用的历史记录时启用 | 相对连续批处理纯新生成路径，在线墙钟 `0.754×`、1.5B FLOPs `0.744×`、总 FLOPs `0.907×` |
| MH 多尺度后缀 | 正收益 | 启用 | 32 题确认中生成 token 下降 32.3%，聚合墙钟下降 40.7% |
| 多尺度后缀 + 冻结 replay MH | 正收益 | 有匹配历史时启用 | 完整 Hastings 校正下在线墙钟因子 `0.357×`；FLOPs 因子 `1.002×` |
| 迭代 SIR | 负收益 | 关闭 | 最优筛选组准确率低于标准条件 IS，主模型 FLOPs 为其 2.34 倍 |
| Sobol / 随机平移等距格点 rollout | 负收益 | 关闭 | 权重离散度下降，但准确率不变，墙钟因子分别为 `1.013×` 和 `1.044×` |
| 有界权重精确停止 | 负收益 | 关闭 | 跳过 8.27% rollout，但分批执行前缀预填充使墙钟因子达到 `1.398×` |
| 0.5B 精确推测解码 | 负收益 | 关闭 | 最优 $`K=2`$ 的墙钟因子 `1.058×`，合计 FLOPs 因子 `1.439×` |
| 流式奖励 | 条件收益 | 关闭 | verifier 延迟足够大时可与生成重叠；近零延迟奖励无收益 |
| 两阶段延迟接受 MH | 条件收益 | 关闭 | 需要高成本精确奖励和有效的近似奖励 |
| MH proposal 预取 | 条件收益 | 关闭 | 使用额外 proposal FLOPs，将 proposal 生成与奖励等待并行 |
| 历史 token 树 | 无稳定收益 | 关闭 | 墙钟区间覆盖无变化且 FLOPs 增加 |
| 初始估计与最终估计分离的 rollout 分配 | 负收益 | 关闭 | 墙钟与 FLOPs 同时增加 |
| 方差—成本动态分配 | 无收益 | 关闭 | 质量与墙钟未改善，FLOPs 增加 |
| 始终使用历史 token 树 | 负收益 | 关闭 | 墙钟与 FLOPs 明显增加 |
| Transformers 部分 rollout 续跑 | FLOPs 增加 | 关闭 | 墙钟下降但重复执行前缀预填充使 FLOPs 超过三倍 |

表中已有方法来自[方法质量与计算量](GSM8K_3090_ALIGNED_RESULTS.md)、
[推理执行与 rollout 复用](RTX3090_ROLLOUT_INFRA.md)及本轮 Qwen 1.5B 消融。机器可读登记表保存比较对象、
默认状态和结果文件路径；未通过的方法仍可显式运行，不进入默认组合。
