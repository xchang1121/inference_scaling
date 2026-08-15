# 推理扩展算法：基础、原理与实现

本文档集中说明仓库中全部推理算法及其执行实现。每个方法按目标分布、有限预算算法、关键代码、统计性质和
成本来源组织；批处理、KV 复用、异步奖励、vLLM 和计算账本统一列在第 17 节。实验数值见
[GSM8K 方法质量与计算量](../reports/GSM8K_3090_ALIGNED_RESULTS.md)和
[推理执行与 rollout 复用](../reports/RTX3090_ROLLOUT_INFRA.md)。

## 1. 统一记号与实现边界

给定 token 化提示 $`x`$，记基础模型的完整生成分布为

```math
p(y\mid x)=\prod_{t=1}^{|y|}p(y_t\mid x,y_{<t}).
```

在已经生成前缀 $`g`$ 时，下一段候选记为 $`z`$，候选后的补全记为 $`u`$。奖励写作
$`r(g,z,u)`$，奖励温度写作 $`\tau\gt 0`$。仓库中最常用的显式奖励目标是

```math
\pi_r(y\mid x)
=\frac{p(y\mid x)\exp\{r(y)/\tau\}}
       {\sum_{y'}p(y'\mid x)\exp\{r(y')/\tau\}}.

```

<p align="right">式 (1)</p>

另一类目标是幂分布

```math
\pi_\alpha(y\mid x)
=\frac{p(y\mid x)^\alpha}{\sum_{y'}p(y'\mid x)^\alpha},
\qquad \alpha>0.

```

<p align="right">式 (2)</p>

本文使用三类性质：

- **平稳分布精确**：MH 转移核保持指定目标不变；有限更新轮次仍有链的收敛误差。
- **估计量无偏**：普通 IS 或 replay 恒等式对条件能量给出无偏估计；有限候选数下的归一化重采样仍是近似。
- **执行等价**：批处理、流式完成和预取保持随机请求与统计量固定。

下文中，MH 指 Metropolis--Hastings，IS 指 Importance Sampling（重要性采样），SIR 指
Sampling-Importance-Resampling（采样--重要性加权--重采样），SMC 指 Sequential Monte Carlo
（序贯蒙特卡洛），GRPO 指 Group Relative Policy Optimization（组相对策略优化）。

重要性修正要求 $`p(y)\gt 0\Rightarrow q(y)\gt 0`$。硬 top-k/top-p 可能破坏该条件；权重截断以偏差换取
有限权重范围。

<a id="alg-overview"></a>
## 2. 方法总览

| 方法 | 采样或估计对象 | 有限预算下的性质 | 主要实现 |
| --- | --- | --- | --- |
| Base / greedy / beam / Best-of-$`N`$ | 基础模型采样或确定性搜索 | 基线分布或奖励最大化 | `experiments/gsm8k_reproduction.py` |
| 幂分布 MH | 式 (2) | 转移核精确；有限更新存在收敛误差 | `algorithms/mh.py::run_mh_chain` |
| 奖励目标 MH | 式 (1) | 转移核精确；每次 proposal 通常需完整奖励 | `algorithms/mh.py::run_reward_mh_chain` |
| 条件能量 IS | 式 (1) 的逐 block SIR | $`K,M\to\infty`$ 时趋近目标 | `algorithms/conditional_energy.py` |
| off-policy 条件 IS | 同上，补全来自其他 proposal | 未截断普通 IS 对能量无偏 | `algorithms/conditional_energy.py` |
| proposal-energy | $`p(z)\,\mathbb E_q[e^{r/\tau}\mid z]`$ | 有意改变目标的消融 | 同上，`apply_importance_correction=False` |
| base 候选 rollout replay | 式 (1) 的逐 block SIR | history + fresh-tail 能量估计无偏 | `algorithms/base_replay.py` |
| 动态候选 IS | 辅助候选、外层 IS、replay | 使用实际候选 proposal 的 $`p/q_c`$ | `algorithms/dynamic_is.py` |
| progressive IS | pilot 分配预算，独立 evaluation 估计 | 最终估计仅使用 evaluation | `algorithms/progressive_is.py` |
| frozen streaming IS | 固定设计的异步到达版本 | 固定样本集合上的顺序不变性 | `algorithms/streaming_is.py` |
| SMC rollout forest | block 级粒子近似 | 有限粒子、有限 lookahead 的 SMC 近似 | `algorithms/smc_forest.py` |
| delayed-acceptance MH | 式 (1) | 两阶段接受率保持目标不变 | `algorithms/mh_acceleration.py` |
| replay-mixture MH | 式 (1) | 冻结混合 proposal 的正反概率均进入 Hastings 比 | `algorithms/mh_acceleration.py` |
| GRPO | 参数化策略的训练近似 | 受模型族、优化轮次与采样预算影响 | `experiments/train_gsm8k_grpo.py` |

表中的相对源码路径均位于 [`src/inference_scaling`](../../src/inference_scaling/)。

<a id="alg-sources"></a>
### 方法来源

| 方法族 | 主要文献 | 本仓库中的关系 |
| --- | --- | --- |
| beam search | [Freitag and Al-Onaizan (2017)](https://aclanthology.org/W17-3207/) | 作为确定性搜索基线 |
| self-consistency | [Wang et al. (2023)](https://openreview.net/pdf?id=1PL1NIMMrw) | 作为并行采样基线与可部署奖励信号 |
| Metropolis--Hastings | [Hastings (1970)](https://doi.org/10.1093/biomet/57.1.97) | 用于幂分布和显式奖励目标的后缀转移 |
| 重要性采样与 defensive mixture | [Hesterberg (1995)](https://doi.org/10.1080/00401706.1995.10484303) | 用于条件能量、外层候选修正和完整支持集 proposal |
| off-policy 修正 | [Precup, Sutton, and Singh (2000)](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf) | 用真实 behavior 概率修正异分布 rollout |
| 经验回放 | [Lin (1992)](https://doi.org/10.1007/BF00992699) | 历史 completion 经式 (13) 校正后进入条件能量估计 |
| GRPO | [Shao et al. (2024)](https://arxiv.org/abs/2402.03300) | 使用同一基础模型训练的参数更新基线 |
| 最优分层分配 | [Neyman (1934)](https://doi.org/10.1111/j.2397-2335.1934.tb04184.x)、[Étoré and Jourdain (2010)](https://doi.org/10.1007/s11009-008-9108-0) | 推导式 (19) 的方差--成本预算规则 |
| SMC | [Del Moral, Doucet, and Jasra (2006)](https://doi.org/10.1111/j.1467-9868.2006.00553.x)、[Lew et al. (2023)](https://arxiv.org/abs/2306.03081) | 用于逐 block 粒子传播和条件后缀 reservoir |
| delayed-acceptance MCMC | [Christen and Fox (2005)](https://doi.org/10.1198/106186005X76983) | 通过两阶段接受率减少精确奖励调用 |
| 连续批处理与 KV block | [Orca，Yu et al. (2022)](https://www.usenix.org/conference/osdi22/presentation/yu)、[PagedAttention，Kwon et al. (2023)](https://doi.org/10.1145/3600006.3613165) | 跨 prompt 调度、共同前缀 prefill 和 vLLM APC |
| speculative decoding | [Leviathan, Kalman, and Matias (2023)](https://proceedings.mlr.press/v202/leviathan23a.html)、[REST，He et al. (2024)](https://aclanthology.org/2024.naacl-long.88/) | 历史 token tree、target verification 和残差抽样 |
| 异步生成与消费 | [IMPALA，Espeholt et al. (2018)](https://proceedings.mlr.press/v80/espeholt18a.html)、[SAO，Hou et al. (2026)](https://arxiv.org/abs/2607.07508) | completion callback、部分 rollout 和低优先级 run-ahead |
| MCMC prefetch | [Brockwell (2006)](https://doi.org/10.1198/106186006X100579) | 奖励等待期间预取接受和拒绝分支 |

下文给出分块条件能量、fresh-tail replay、动态候选和冻结 evaluation 的公式与实现。

<a id="alg-report-labels"></a>
### 实验标签定义

标签按参数、目标、候选、rollout、概率修正和输出规则定义。`verifier` 表示 GSM8K 标准答案的确定性
数值检查器。

#### 直接生成与训练基线

| 报告名称 | 使用的参数 | 推理时的实际操作 |
| --- | --- | --- |
| Base | 原始 1.5B 基础模型 $`\theta_0`$ | 每步按 $`p_{\theta_0}(\cdot\mid x,y_{\lt t})`$ 随机抽取一个 token；主实验温度为 1 |
| Beam-8 | 原始 1.5B 基础模型 $`\theta_0`$ | 保留累计 log-probability 最高的 8 条前缀，最终返回最高分序列 |
| 自一致性投票-8 | 原始 1.5B 基础模型 $`\theta_0`$ | 独立随机生成 8 条序列，返回数值答案众数对应的序列 |
| GRPO 参数 + 随机采样 | GRPO 训练后的 1.5B 参数 $`\theta_{\mathrm{GRPO}}`$ | $`y_t\sim p_{\theta_{\mathrm{GRPO}}}(\cdot\mid x,y_{\lt t})`$，温度为 1 |
| GRPO 参数 + 贪心解码 | GRPO 训练后的 1.5B 参数 $`\theta_{\mathrm{GRPO}}`$ | $`y_t=\arg\max_v p_{\theta_{\mathrm{GRPO}}}(v\mid x,y_{\lt t})`$ |

#### MH、条件 IS 与 replay 标签

| 报告名称 | 候选或状态 | rollout / proposal | 奖励与概率修正 |
| --- | --- | --- | --- |
| 幂分布 MH | 当前完整序列 | 1.5B 基础模型生成后缀 proposal | 目标为 $`p_{\theta_0}(y\mid x)^\alpha`$ |
| verifier-MH | 当前完整序列 | 1.5B 基础模型生成后缀 proposal | 使用终局正确性奖励，目标为式 (1) |
| 标准条件 IS | 候选 block 来自 1.5B base | rollout 来自 1.5B base | 数值众数奖励；修正因子为 1 |
| 0.5B rollout proposal 条件 IS | 候选 block 来自 1.5B base | rollout 来自 0.5B 模型 $`q`$ | 数值众数奖励；乘 $`p_{\theta_0}/q`$；默认截断 log-ratio |
| 0.5B rollout proposal IS（无截断） | 与上一行相同 | 与上一行相同 | 使用未经截断的 $`p_{\theta_0}/q`$ |
| 0.5B rollout 奖励重加权 | 候选 block 来自 1.5B base | rollout 来自 0.5B 模型 | 数值众数奖励；目标为式 (12) |
| 标准 verifier-IS | 候选和 rollout 都来自 1.5B base | 1.5B on-policy rollout | 数值正确性奖励；估计式 (7) |
| 0.5B rollout proposal verifier-IS | 候选 block 来自 1.5B base | 0.5B rollout | 数值正确性奖励；乘 $`p_{\theta_0}/q`$ |
| 0.5B rollout verifier 奖励重加权 | 候选 block 来自 1.5B base | 0.5B rollout | 数值正确性奖励；目标为式 (12) |
| fresh-only | 候选来自 1.5B base | 本轮 rollout 全部新生成 | history 数为 0 |
| warm replay | 候选来自 1.5B 基础模型 | 消费已建库 history，并生成独立 fresh tail | 按式 (14) 修正 history；`warm` 表示历史库已存在 |
| base 候选 + 固定 fresh | 候选来自 1.5B base | 每个候选使用固定数量 fresh rollout | 动态候选实验的基线 |
| 动态候选 + 固定 replay | 候选来自式 (15) 的 base/辅助模型 mixture | 命中时使用固定 history/fresh 数量 | 候选层乘式 (16)，rollout 层使用式 (14) |
| 动态候选 + 方差--成本分配 | 与上一行相同 | history/fresh 数量由独立 design 数据决定 | 最终 evaluation 预算按式 (19) 冻结 |

<a id="alg-baselines"></a>
## 3. 生成与训练基线

### 3.1 Base、greedy、beam 与 Best-of-$`N`$

`base` 按配置温度从基础模型抽样；`greedy` 逐 token 取最大概率项；beam search 保留累计对数概率最高的
若干前缀。

Best-of-$`N`$ 先独立生成 $`y_1,\ldots,y_N\sim p`$，再按奖励或 self-consistency 规则选择一个序列：

```math
\widehat y=\arg\max_{1\le i\le N}\widehat r(y_i).

```

<p align="right">式 (3)</p>

式 (3) 随 $`N`$ 增大趋向奖励最大化。数值答案众数相同时，实验按模型 log-probability 确定性破同票。

### 3.2 GRPO 对照

本仓库中的 GRPO 使用同一基础模型和 GSM8K 数值正确性奖励进行参数训练。若忽略参数化限制，一个带 KL
正则的理想策略优化问题具有式 (1) 的形式；实际 GRPO 只通过有限 rollout、组内相对优势和有限梯度更新去近似
该目标。报告分别列出训练 FLOPs 与训练后采样 FLOPs；单次推理成本指训练完成后的生成成本。

训练得到固定策略 $`p_{\theta_{\mathrm{GRPO}}}`$。实验分别采用温度 1 随机采样和逐 token argmax 解码。

训练入口为 [`experiments/train_gsm8k_grpo.py`](../../experiments/train_gsm8k_grpo.py)，精确数值奖励实现为
[`evaluation/grpo_reward.py`](../../src/inference_scaling/evaluation/grpo_reward.py)。

<a id="alg-power-mh"></a>
## 4. 幂分布后缀 MH

固定生成长度为 $`L`$。当前状态为 $`y=(y_1,\ldots,y_L)`$，每次更新从所有后缀起点
$`s\in\{0,\ldots,L-1\}`$ 中均匀抽取一个，保留 $`y_{\lt s}`$，再从 proposal
$`q_s(\cdot\mid x,y_{\lt s})`$ 生成新后缀 $`v`$。接受概率为

```math
A(y\to y')=
\min\left\{1,
\exp\left[
\alpha\bigl(\log p(v\mid x,y_{<s})-\log p(y_{\ge s}\mid x,y_{<s})\bigr)
+\log q_s(y_{\ge s}\mid x,y_{<s})-\log q_s(v\mid x,y_{<s})
\right]\right\}.

```

<p align="right">式 (4)</p>

候选前缀相同，切点选择概率正反相同，因此式 (4) 为完整 Hastings 比。温度 proposal 的逐前缀归一化
常数进入 $`q_s`$ 的正反概率。

实现按 `block_size` 逐步扩展到 $`L`$，并在每个长度执行 `steps_per_block` 次后缀更新。最终长度上的有限更新
结果仍含 MCMC 误差。由于切点 $`s=0`$ 能以正概率重生成整段，且未截断 softmax proposal 在有限词表、
固定长度空间上处处为正，转移矩阵任意两行都有正重叠。写

```math
\delta(K)=1-\min_{y,y'}\sum_v\min\{K(y,v),K(y',v)\}<1,
```

则最终长度的核满足

```math
\left\|\mu K^n-\pi_\alpha\right\|_{\mathrm{TV}}
\le \delta(K)^n.

```

<p align="right">式 (5)</p>

真实模型实验报告更新轮次、接受率和输出诊断；收缩常数需要显式转移矩阵 $`K`$。

代码中的接受率与式 (4) 一一对应：

```python
log_acceptance = min(
    0.0,
    alpha * (new_base_logprob - old_base_logprob)
    + old_proposal_logprob - new_proposal_logprob,
)
accepted = log(uniform) <= log_acceptance
```

EOS 由 [`AbsorbingEOSBackend`](../../src/inference_scaling/backends/absorbing.py) 转换为固定长度吸收状态；
终止判断作用于生成区间，EOS 后占位 token 的条件概率为 1。

<a id="alg-reward-mh"></a>
## 5. 奖励目标后缀 MH

对式 (1)，相同后缀 proposal 的接受率为

```math
A_r(y\to y')=\min\left\{1,
\exp\left[
\log\frac{p(y'_{\ge s}\mid x,y_{<s})}{p(y_{\ge s}\mid x,y_{<s})}
+\frac{r(y')-r(y)}{\tau}
+\log\frac{q_s(y_{\ge s}\mid x,y_{<s})}{q_s(y'_{\ge s}\mid x,y_{<s})}
\right]\right\}.

```

<p align="right">式 (6)</p>

当 $`q_s=p(\cdot\mid x,y_{\lt s})`$ 时，基础模型与 proposal 项抵消，只剩
$`\min\{1,e^{(r(y')-r(y))/\tau}\}`$。代码仍保留展开后的四项，因而同样支持任意可精确评分、具有完整
support 的温度 proposal。与式 (5) 相同，整段重生成使有限状态链在通常条件下几何收敛到 $`\pi_r`$。

奖励在实现中是完整生成序列的函数。数值正确性、外部 verifier 等只能在完整 proposal 后得到时，每次普通
MH 更新都要完成整段后缀并调用奖励；降低这部分成本的方法见
[两阶段 MH](#alg-delayed-mh)与[proposal-tree 预取](#infra-mh-prefetch)。

<a id="alg-conditional-is"></a>
## 6. 条件能量 IS

在已生成前缀 $`g`$ 之后，式 (1) 对下一 block $`z`$ 的条件分布可写为

```math
\pi_r(z\mid x,g)\propto p(z\mid x,g)h(g,z),
\qquad
h(g,z)=\mathbb E_{u\sim p(\cdot\mid x,g,z)}
\left[e^{r(g,z,u)/\tau}\right].

```

<p align="right">式 (7)</p>

标准条件 IS 的一次 guidance step 为：

1. 生成 $`M`$ 个候选 $`z_m\sim p(\cdot\mid x,g)`$；
2. 对每个候选生成 $`K`$ 条 on-policy 补全 $`u_{mk}\sim p(\cdot\mid x,g,z_m)`$；
3. 计算式 (8)：

```math
\widehat h_m=\frac1K\sum_{k=1}^K e^{r(g,z_m,u_{mk})/\tau};

```

<p align="right">式 (8)</p>

4. 以 $`\widehat h_m/\sum_j\widehat h_j`$ 的概率选择候选并追加到 $`g`$，随后进入下一 block。

候选来自 $`p`$，因此 SIR 选择权重为条件能量 $`\widehat h_m`$。当
$`K\to\infty`$ 时式 (8) 收敛到 $`h`$；当候选数 $`M\to\infty`$ 时，sampling-importance-resampling
输出趋近式 (7)。有限 $`K,M`$ 以及逐 block 重复选择共同构成实际近似误差。

关键实现直接在 log 域求均值并重采样：

```python
log_energy = logmeanexp(rollout.log_weight for rollout in evaluations)
probabilities = softmax([candidate.log_energy for candidate in candidates])
selected_index = rng.choice(len(candidates), p=probabilities)
```

实现入口为
[`run_conditional_is`](../../src/inference_scaling/algorithms/conditional_energy.py)，候选与所有 rollout 都按
异构请求展平为批次；执行细节见[重复前缀 KV 复用](#infra-prefix-kv)。

<a id="alg-offpolicy-is"></a>
## 7. off-policy 补全与主模型重评分

若补全由 proposal $`q(u\mid x,g,z)`$ 生成，则式 (7) 改写为

```math
h(g,z)=\mathbb E_{u\sim q}
\left[
e^{r(g,z,u)/\tau}
\frac{p(u\mid x,g,z)}{q(u\mid x,g,z)}
\right].

```

<p align="right">式 (9)</p>

对应普通 IS 估计量为

```math
\widehat h_m=\frac1K\sum_{k=1}^K
\exp\left\{
\frac{r_{mk}}{\tau}
+\log p(u_{mk}\mid x,g,z_m)
-\log q(u_{mk}\mid x,g,z_m)
\right\}.

```

<p align="right">式 (10)</p>

式 (10) 未截断时对 $`h(g,z_m)`$ 无偏。实践中 proposal 可以是 0.5B 模型，候选 $`z_m`$ 仍完全由
1.5B 基础模型生成；“1.5B 重评分”只是在小模型补全完成后，用基础模型一次批量前向计算式 (10) 中的
$`\log p(u_{mk}\mid x,g,z_m)`$。生成时已保存的 $`\log q`$ 不需要再次计算。

```python
raw_log_ratio = base_logprob - proposal_logprob
applied_log_ratio = raw_log_ratio
if importance_log_ratio_clip is not None:
    applied_log_ratio = clip(raw_log_ratio, -clip_value, clip_value)
log_weight = reward / reward_temperature + applied_log_ratio
```

截断 $`\operatorname{clip}(\log p/q,-c,c)`$ 将式 (9) 改为有偏估计。报告记录 raw ratio、applied ratio、
截断次数和 effective sample size（ESS）。

<a id="alg-proposal-energy"></a>
### 7.1 proposal-energy 目标

设置 `apply_importance_correction=False` 时，权重仅为 $`e^{r/\tau}`$：

```math
\widehat h^{(q)}(g,z)=\frac1K\sum_{k=1}^K e^{r(g,z,u_k)/\tau},
\qquad u_k\sim q(\cdot\mid x,g,z).

```

<p align="right">式 (11)</p>

此时逐 block 目标为

```math
p(z\mid x,g)\,
\mathbb E_{u\sim q(\cdot\mid x,g,z)}[e^{r(g,z,u)/\tau}],

```

<p align="right">式 (12)</p>

式 (12) 使用 1.5B 候选、0.5B 补全和奖励权重，主模型重评分成本为 0。该路径记为 proposal-energy
目标。两种路径的质量与
分模型 FLOPs 见[1.5B 重评分消融](../reports/GSM8K_3090_ALIGNED_RESULTS.md#15b-rescoring-ablation)。

<a id="alg-base-replay"></a>
## 8. base 候选上的 rollout replay

历史 completion 来自一个可精确评分的 behavior mixture $`b(u\mid x,g,z)`$。令

```math
w(u)=\frac{p(u\mid x,g,z)}{b(u\mid x,g,z)},
\qquad A(u)=e^{r(g,z,u)/\tau},
```

并取截断常数 $`c\gt 0`$。实现使用恒等式

```math
\mathbb E_b[\min\{c,w(u)\}A(u)]
+\mathbb E_p\left[\left(1-\frac{c}{w(u)}\right)_+A(u)\right]
=\mathbb E_p[A(u)].

```

<p align="right">式 (13)</p>

逐点验证式 (13)：当 $`w\le c`$ 时，左边第一项在同一离散样本空间上贡献 $`pA`$，第二项为 0；当
$`w\gt c`$ 时，两项分别贡献 $`cbA`$ 与 $`(p-cb)A`$。因此，使用 $`H`$ 条 history 与 $`F`$ 条独立 fresh
base rollout 的估计量

```math
\widehat h=
\frac1H\sum_{i=1}^H\min\left\{c,\frac{p(u_i)}{b(u_i)}\right\}A(u_i)
+\frac1F\sum_{j=1}^F
\left(1-c\frac{b(v_j)}{p(v_j)}\right)_+A(v_j)

```

<p align="right">式 (14)</p>

对式 (7) 的条件能量无偏。`corrected_replay_log_energy` 在 log 域分别计算两项均值，再执行 `logaddexp`：

```python
history_term = min(log(c), log_p - log_b) + reward / tau
if log_p - log_b <= log(c):
    fresh_term = float("-inf")
else:
    fresh_term = log1p(-exp(log(c) + log_b - log_p)) + reward / tau
log_energy = logaddexp(logmeanexp(history_terms), logmeanexp(fresh_terms))
```

当 $`H=0`$ 时，算法使用 fresh base rollout 的式 (8)。历史中存在多个 behavior 版本时，$`b`$
按本轮冻结 claim 中各 behavior 的条数构成显式 mixture；每条保存概率还会重新评分校验。

<a id="alg-replay-lifecycle"></a>
### 8.1 replay 数据生命周期

实现将记录分为三个集合：

1. `design`：已经消费的记录，用于估计方差和成本；
2. `evaluation`：仅暴露 key、behavior id 和数量；在设计冻结后最多消费一次；
3. `reserved`：已从 evaluation 中原子取出、数值保持隐藏的 claim。

当前选择所用的 fresh rollout 在本轮结束后进入 `design`。只有候选选择完成后，针对新前缀独立生成的
reserve rollout 才写入未来的 `evaluation`。关键代码约束如下：

```python
claim = store.freeze_claims([key], history_count)[0]  # 只返回数量与 behavior
history = store.reveal_and_consume(claim)             # 一次性揭示并转入 design
for record in current_fresh:
    store.add_design(record)
# 选择完成后，独立 reserve 才进入 evaluation：
store.add_evaluation(independent_reserve_record)
```

该生命周期同时用于 `base-replay` 和 `dynamic-is`。存储实现见
[`replay.py`](../../src/inference_scaling/replay.py)。

<a id="alg-dynamic-is"></a>
## 9. 动态候选 proposal 与外层 IS

动态版本从含基础模型分量的混合 proposal（defensive mixture）抽取候选：

```math
q_c(z\mid x,g)=(1-\lambda)p(z\mid x,g)+\lambda a(z\mid x,g),
\qquad 0\le\lambda<1,

```

<p align="right">式 (15)</p>

其中 $`a`$ 可以是辅助模型或依赖先前候选的 proposal。基础分量给出
$`p(z)\gt 0\Rightarrow q_c(z)\gt 0`$。每个候选使用其实际 proposal 计算外层比值

```math
\rho(z)=\frac{p(z\mid x,g)}{q_c(z\mid x,g)}.

```

<p align="right">式 (16)</p>

候选最终 log weight 为

```math
\log W_m=\log\rho(z_m)+\log\widehat h(g,z_m),

```

<p align="right">式 (17)</p>

其中 $`\widehat h`$ 可由 fresh rollout 或式 (14) 的 replay 估计得到。静态辅助 proposal 会按真实 sampling
policy 分组批量生成，并在 base 与 auxiliary 下分别批量评分；依赖先前候选的 factory 则保留必要的串行依赖。

```python
proposal_logprob = logaddexp(
    log(1.0 - mixture) + base_logprob,
    log(mixture) + auxiliary_logprob,
)
outer_log_ratio = base_logprob - proposal_logprob
candidate_log_weight = outer_log_ratio + replay_estimate.log_energy
```

有限候选下仍是 self-normalized SIR 近似。外层比值只修正候选来源；补全层仍需单独执行
off-policy/replay 修正。

<a id="alg-budget-allocation"></a>
### 9.1 方差—成本最优预算分配

对候选 $`i`$ 和来源 $`s\in\{\text{history},\text{fresh}\}`$，记单样本标准差为
$`\sigma_{i,s}`$、成本为 $`c_{i,s}`$、外层比值为 $`\rho_i`$、分配数量为 $`n_{i,s}`$。忽略整数与容量约束时，
实现近似最小化

```math
\sum_{i,s}\frac{\rho_i^2\sigma_{i,s}^2}{n_{i,s}}
\quad\text{s.t.}\quad
\sum_{i,s}c_{i,s}n_{i,s}\le C.

```

<p align="right">式 (18)</p>

拉格朗日一阶条件给出

```math
n_{i,s}\propto \frac{\rho_i\sigma_{i,s}}{\sqrt{c_{i,s}}}.

```

<p align="right">式 (19)</p>

代码先按式 (19) 求连续解，再施加逐候选 history 上限、相同 replay key 的共享容量、每个非终止候选的
最少 fresh 数量，最后执行确定性的 floor + largest-remainder 舍入。方差与成本只能来自 `design`；
`rollout_budget_provider` 的输入为候选、终止标记、库存数量和 design 统计量。

<a id="alg-progressive-is"></a>
## 10. progressive pilot / evaluation IS

当不同候选的补全长度、模型成本或权重方差差异较大时，固定 $`K`$ 可能浪费预算。progressive 版本先为每个
候选生成少量 pilot rollout，估计

```math
\widehat\sigma_i=\operatorname{Std}
\left[\exp\{\ell_{ik}-\max_{j,k}\ell_{jk}\}\right],
\qquad
\ell_{ik}=r_{ik}/\tau+\log p(u_{ik})-\log q(u_{ik}),

```

<p align="right">式 (20)</p>

并用生成 token 数乘 proposal/base 参数量估计相对成本。随后按式 (19) 冻结 evaluation 数量，再独立生成
新的 evaluation rollout。最终能量只使用 evaluation：

```math
\widehat h_i^{\mathrm{final}}
=\frac1{K_i^{\mathrm{eval}}}
\sum_{k=1}^{K_i^{\mathrm{eval}}}e^{\ell_{ik}^{\mathrm{eval}}}.

```

<p align="right">式 (21)</p>

pilot 可作为 speculative draft 的历史材料；式 (21) 仅使用独立 evaluation。终止候选的能量为确定值，
复用一次奖励计算。

<a id="alg-streaming-is"></a>
## 11. frozen-design streaming IS

streaming IS 使用式 (10)、(14) 或 (21)，并允许已冻结的 fresh 样本按任意完成顺序到达。状态机为：

1. 冻结前加入允许的 history contribution；
2. `freeze` 一次性声明每个候选的 fresh sample id；
3. `consume_fresh` 可按任意顺序提交，但拒绝未知 id、重复 id 和候选错配；
4. 所有声明样本到齐后，`select` 返回最终选择。

每个候选在固定 multiset 上计算 `logmeanexp`，因此结果与到达顺序无关。GPU 完成回调可立即启动 CPU
verifier。实现见
[`streaming_is.py`](../../src/inference_scaling/algorithms/streaming_is.py)，墙钟重叠见
[流式奖励计算](#infra-streaming-reward)。

<a id="alg-smc-forest"></a>
## 12. SMC rollout forest

Sequential Monte Carlo（SMC，序贯蒙特卡洛）版本维护 $`P`$ 个前缀粒子。定义理想 lookahead

```math
h(s)=\mathbb E_{u\sim p(\cdot\mid x,s)}[e^{r(s,u)/\tau}].

```

<p align="right">式 (22)</p>

从父粒子 $`s`$ 按基础模型生成下一 block $`z`$ 后，中间目标
$`p(s,z\mid x)h(s,z)`$ 相对 proposal 的增量权重为

```math
\Delta(s\to sz)=\frac{h(sz)}{h(s)},
\qquad
\log\Delta=\log h(sz)-\log h(s).

```

<p align="right">式 (23)</p>

实现用有限 rollout reservoir 的 `logmeanexp` 估计 $`h`$，按式 (23) 计算 branch 权重，再执行 systematic
resampling。增量在路径上望远镜相消；在精确 lookahead、足够粒子与完整长度下对应式 (1) 的序贯构造。

若父粒子的某条历史完整补全以新 block $`z`$ 开头，删掉该 block 后的剩余后缀仍是 $`p(\cdot\mid x,s,z)`$
下的有效条件 rollout，可以继承到子 branch。一个 branch 对应多个粒子时，reservoir 在粒子间分桶，
随后用 fresh rollout 补足。

有限粒子数、有限 branch factor 和有限 rollout reservoir 产生 SMC 近似误差。实现同时报告 ESS、fresh
与 reused rollout 数。

<a id="alg-delayed-mh"></a>
## 13. 两阶段 delayed-acceptance MH

设便宜 surrogate 奖励为 $`\widetilde r(y)`$。第一阶段用
$`p(y)e^{\widetilde r(y)/\tau}`$ 的完整 Hastings 比接受 proposal；只有通过时才计算精确奖励。第二阶段接受率为

```math
A_2(y\to y')=
\min\left\{1,
\exp\left[
\frac{r(y')-r(y)-\widetilde r(y')+\widetilde r(y)}{\tau}
\right]\right\}.

```

<p align="right">式 (24)</p>

两阶段乘积满足式 (1) 的详细平衡。surrogate 在链运行期间固定；自适应 surrogate 需要扩展状态或额外校正。

```python
stage_one = min(0.0, proposal_and_base_terms + surrogate_delta / tau)
if log(u1) <= stage_one:
    exact_proposed = reward(proposal)
    stage_two = min(0.0, (exact_delta - surrogate_delta) / tau)
    accepted = log(u2) <= stage_two
```

该路径减少精确奖励调用，proposal 生成 FLOPs 保持不变；对应消融见
[Delayed acceptance](../reports/RTX3090_ROLLOUT_INFRA.md#infra-report-delayed-acceptance)。

<a id="alg-replay-mh"></a>
## 14. replay-mixture MH proposal

冻结历史后缀经验分布 $`h_{\mathrm{emp}}`$，并与基础模型组成混合 proposal

```math
q_s(v\mid x,y_{<s})=(1-\lambda)p(v\mid x,y_{<s})
+\lambda h_{\mathrm{emp}}(v\mid x,y_{<s}),
\qquad 0\le\lambda<1.

```

<p align="right">式 (25)</p>

历史命中时可读取现成 suffix，并通过一次并行评分获得 $`p(v)`$；未命中时从基础模型生成。无论来源如何，
式 (6) 都使用旧后缀与新后缀在混合分布 (25) 下的精确概率。基础分量保证完整支持集，经验库在链开始前
冻结，因而该 proposal 仍定义普通 MH 转移核。

```python
old_q = replay_proposal.logprob(prefix, old_suffix, base_logprob=old_p)
draw = replay_proposal.draw(prefix, suffix_length, seed=seed)
log_acceptance = min(
    0.0,
    new_p - old_p + reward_delta / tau + old_q - draw.proposal_logprob,
)
```

这里的 replay 改变 proposal、再由 Hastings 比校正；它与式 (14) 中直接复用 rollout 估计条件能量是两种
不同机制。

<a id="alg-rewards"></a>
## 15. 已实现的奖励信号

条件 IS 与奖励 MH 接受任意有限序列奖励。GSM8K 实验提供下列实现：

| 奖励 | 定义或实现 | 作用范围 |
| --- | --- | --- |
| 数值正确性 verifier | 解析最终数值，与标准答案比较，取 0/1 | 共享目标诊断；会读取标准答案 |
| cumulative self-consistency | 按已经评估的数值结果累计众数，匹配众数取 1 | 部署质量实验；奖励来源为生成结果 |
| 平均 token log-probability | 选中 token 的平均 log-probability | 置信度消融 |
| 平均负熵 | $`\lvert y\rvert^{-1}\sum_t\sum_v p_t(v)\log p_t(v)`$ | 置信度消融 |
| self-certainty | $`-\lvert y\rvert^{-1}\sum_t \lvert V\rvert^{-1}\sum_v[\log\lvert V\rvert+\log p_t(v)]`$ | 置信度消融 |

后三类在每个 guidance step 内对全部 candidate rollout 做 min-max 归一化。熵类奖励需要全词表概率，
由精确 Transformers scoring backend 计算。self-consistency
实现见 [`evaluation/consensus.py`](../../src/inference_scaling/evaluation/consensus.py)。

<a id="alg-correctness-matrix"></a>
## 16. 正确性与近似来源

| 设置 | 统计性质 | 诊断 |
| --- | --- | --- |
| 增加 MH 更新轮次 | 目标固定；有限链误差下降 | 更新数、接受率、链间结果 |
| 增加条件 IS 的 $`M,K`$ | 渐近目标固定；有限 SIR 误差下降 | 每候选 rollout、ESS、FLOPs |
| off-policy 补全 + 未截断 $`p/q`$ | 式 (7) 的条件能量无偏 | 两侧 log-probability、ESS、support |
| 截断 log importance ratio | 有偏稳定化估计 | raw/applied ratio、截断次数 |
| proposal-energy | 目标为式 (12) | `score_calls=0`、分模型 FLOPs |
| replay 恒等式 + 独立 fresh tail | 式 (7) 的条件能量无偏 | behavior 版本、claim、fresh/history 数 |
| 动态候选 + 外层 $`p/q_c`$ | 候选来源已校正；保留有限 SIR 误差 | 候选来源、outer ratio、共享容量 |
| pilot 决定 evaluation 数量 | 最终估计仅使用独立 evaluation | pilot/evaluation 分离、冻结预算 |
| 流式到达、连续批处理、预取 | 统计量固定，执行顺序变化 | 请求 id、seed、token/FLOPs、废弃工作 |

<a id="alg-runtime"></a>
## 17. 共同执行实现

算法层固定候选、rollout、proposal 概率、请求 seed 和样本 multiplicity；执行层负责合批、KV、缓存、
异步回调和设备调度。该边界使同一统计设计能够运行在 Transformers 或 vLLM 后端。

### 17.1 后端接口、随机数与计算量

算法只依赖两类请求：

```python
GenerationRequest(prefix, max_new_tokens, sampling, seed, request_id)
ScoreRequest(prefix, continuations, sampling)
```

每个生成请求保存独立 seed 和 uniform stream。Transformers 使用 FP64 cumulative probability 执行
inverse-CDF；物理 batch 改变时，请求仍使用相同随机阈值。CUDA batch 形状引起的 logits 数值差异通过
exact-token match、共同前缀和最终数值结果记录。

模型 $`j`$ 的稠密前向主干计算量估计为：

```math
\widehat F_{\mathrm{forward}}=2\sum_j N_jS_j,
```

其中 $`N_j`$ 为参数量，$`S_j`$ 为实际 forward token slots。prefill、decode、完整序列评分和 target
speculative verification 分别计数；墙钟、显存和吞吐单独报告。

<a id="infra-prefix-kv"></a>
### 17.2 批处理、KV 与概率评分

| 机制 | 实现 | 收益与成本 |
| --- | --- | --- |
| 跨 prompt 连续批处理 | 兼容的 `sample_batch` / `score_batch` 在等待窗口内合并 | 提高 GPU 利用率；可能增加 padding |
| rollout 展平 | 不同候选的异构请求组成一个物理 batch，结果按索引还原 | 删除逐候选同步点 |
| 向量化 MH | 独立链按 stage/update 锁步，只合并同一步 proposal | 保留每条链的 cut、seed 和 uniform |
| 重复前缀 KV | 唯一前缀只 prefill 一次，再复制 KV 和末位置 logits | 增加 KV 复制；减少重复 prefill |
| 生成概率直返 | 同一 logits 保存实际 proposal 与基础 policy 概率 | on-policy IS 和 MH 省去重复评分 |
| 评分缓存 | `(policy, prefix, continuation)` 作为有界 LRU key | 将确定性重复评分变为查表 |
| 评分 microbatch | `max_score_batch_size` 与 `logits_to_keep` | 限制长序列全词表 logits 的显存峰值 |

若第 $`i`$ 个唯一前缀长 $`L_i`$、重复 $`K_i`$ 次，省去的非 padding prefill slots 为：

```math
S_{\mathrm{saved}}=\sum_i(K_i-1)L_i.
```

关键实现位于
[`batching.py`](../../src/inference_scaling/backends/batching.py)、
[`cache.py`](../../src/inference_scaling/backends/cache.py)和
[`transformers_backend.py`](../../src/inference_scaling/backends/transformers_backend.py)。

### 17.3 历史 token tree 与部分 rollout

`RolloutTokenTree` 保存“后缀 context → 下一 token 计数”。确定性模式提出最高频 token；随机模式从经验
proposal $`q_t`$ 抽取草稿 $`a`$，按下式接受：

```math
\Pr(\mathrm{accept}\ a)=\min\left\{1,\frac{p_t(a)}{q_t(a)}\right\}.
```

拒绝后从归一化残差抽取替代 token：

```math
\frac{(p_t(v)-q_t(v))_+}{\sum_w(p_t(w)-q_t(w))_+}.
```

接受路径贡献 $`\min(p_t,q_t)`$，拒绝路径贡献 $`p_t-\min(p_t,q_t)`$，总概率为 target $`p_t`$。
Transformers 一次验证 `prefix + drafts`，并在拒绝点裁剪 `DynamicCache`。草稿长度由 active batch
$`b`$ 的分段函数 $`K(b)`$ 控制，避免大 batch 下的低接受率验证开销。

`AsyncRolloutBroker` 将长生成拆成固定 token chunk。达到所需完整轨迹数后，过量提交产生的部分轨迹保存
token、behavior/reference 概率、continuation seed 和剩余长度；下一次从
`original prefix + saved tokens` 继续。Transformers 恢复时重新 prefill，vLLM 可命中 Automatic
Prefix Caching（APC）。

<a id="infra-streaming-reward"></a>
### 17.4 流式奖励与 run-ahead

支持完成回调的后端在每条序列结束时立即提交 CPU/verifier 任务：

```python
def completed(index, sample):
    futures[index] = executor.submit(reward, prompts[index], sample.token_ids)

samples = sample_batch_with_callback(backend, requests, completed)
rewards = tuple(future.result() for future in futures)
```

`FrozenStreamingISEstimator` 在生成前冻结 request id；样本可按任意顺序到达，最终权重只取决于固定样本
multiset。`LowPriorityRunAheadBackend` 在奖励等待空泡中按有界 chunk 生成未来草稿，前台请求在当前 chunk
结束后取得调度权；后台 token、前台等待和最终 drain 分列计量。

<a id="infra-mh-prefetch"></a>
### 17.5 MH 的执行优化

| 路径 | 执行方式 | 统计校正 |
| --- | --- | --- |
| proposal-tree 预取 | 奖励等待期间分别从接受状态和拒绝状态生成下一 proposal | Hastings 判断只消费实际分支 |
| delayed acceptance | surrogate 第一阶段早拒绝，精确奖励执行第二阶段 | 式 (24) 补回 exact-surrogate 差 |
| replay-mixture proposal | base 与冻结历史后缀组成 mixture | 新旧后缀的混合概率都进入式 (6) |

预取用额外 proposal 隐藏奖励延迟；delayed acceptance 减少精确奖励调用；replay-mixture 将历史命中的
自回归生成替换为 teacher-forced 批量评分。报告同时列出作废分支、精确调用、cache build、FLOPs 和墙钟。

<a id="infra-vllm"></a>
### 17.6 Transformers 与 vLLM

`runtime.backend` 和命令行 `--backend` 使用同一组标识：

| 标识 | 引擎 | 适用路径 |
| --- | --- | --- |
| `transformers` | 显式 KV、批处理和完整概率评分 | 参考实现、概率诊断、全词表奖励 |
| `vllm` | 常驻 `AsyncLLM` | 连续调度、APC 和异步完成回调 |
| `vllm-sync` | 离线 `LLM` | 同步接口和原生 beam |

| 能力 | Transformers | vLLM |
| --- | --- | --- |
| 调度 | 显式 batch 与连续批处理 wrapper | 常驻 `AsyncLLM` 原生 continuous scheduler |
| 前缀复用 | 单批唯一前缀 prefill + KV repeat | 跨调用 APC |
| 生成概率 | 实际 policy 与基础 policy 同时返回 | `processed_logprobs` |
| continuation 评分 | 任意可表示 sampling policy | 温度 1 原生；其余委托 exact Transformers backend |
| 历史草稿 | 确定性或随机 token tree | global suffix proposer |
| broker 恢复 | token 状态 + prefill | token 状态 + APC |

24 GiB 单卡同时驻留 1.5B base 和 0.5B rollout proposal 的配置为：

```toml
[runtime]
backend = "vllm"
device = "cuda"
dtype = "float32"

[vllm]
asynchronous = true
enable_prefix_caching = true
exact_scoring_backend = "none"
tensor_parallel_size = 1
data_parallel_size = 1

[vllm.base]
gpu_memory_utilization = 0.62
max_num_seqs = 48
max_num_batched_tokens = 12288

[vllm.proposal]
gpu_memory_utilization = 0.28
max_num_seqs = 24
max_num_batched_tokens = 6144

[vllm.engine_kwargs]
enable_chunked_prefill = true
```

entropy、self-certainty、非单位温度 behavior 和硬 top-k/top-p 的精确重评分通过
Transformers 后端委托：

```toml
[vllm]
exact_scoring_backend = "transformers"
exact_scoring_device = "cpu"
exact_scoring_dtype = "float32"
```

CPU 委托不占用 vLLM 的 GPU 显存；GPU 委托需要相应降低各引擎的 `gpu_memory_utilization`。快照分别记录
native/delegated sequences、token slots 与 FLOPs。单方法和成对后端测速入口为：

```bash
export PYTHONPATH=src
python experiments/gsm8k_reproduction.py \
  --config configs/gsm8k_3090_aligned.toml \
  --backend vllm --method conditional_is --tag vllm-smoke --limit 8

python experiments/run_vllm_backend_benchmark.py \
  --config configs/gsm8k_3090_aligned.toml \
  --limit 32 --workers 8 --tag rtx3090
```

成对测速固定数据、模型、算法、dtype、worker、GPU 数和代码版本，分别记录逐提示与并发墙钟、forward
slots、token 一致率和数值结果一致率。vLLM `0.25.x` 的 Linux/WSL2 安装命令见仓库
[README](../../README.md#安装)。

<a id="infra-report-labels"></a>
### 17.7 执行标签定义

| 报告名称 | 启用的机制 | 对照路径 |
| --- | --- | --- |
| 部分 rollout 续跑 | 保存未完成 token 并从保存前缀继续 | 丢弃部分轨迹并从原始前缀重生成 |
| 流式 IS，便宜 verifier | rollout 完成后立即提交奖励 worker | 整批完成后提交；verifier 延迟为 0 |
| 流式 IS，0.2 s verifier | 同上，每次 verifier 加入 0.2 s 延迟 | 相同 verifier 延迟的整批提交 |
| 确定性历史草稿 | token tree 提出最高频 token | 普通自回归解码 |
| 精确随机历史草稿 | 经验 proposal + target 残差校正 | 普通自回归解码 |
| MH proposal-tree 预取，便宜奖励 | 为接受和拒绝状态各预取下一 proposal | 普通 MH；reward 延迟为 0 |
| MH proposal-tree 预取，0.2 s 奖励 | 同上，每次 reward 加入 0.2 s 延迟 | 相同奖励延迟的普通 MH |
| delayed acceptance，0.2 s 精确奖励 | surrogate 早拒绝，精确奖励第二阶段 | 每个 proposal 直接调用精确奖励 |
| 冻结 replay 混合 proposal，在线 | base 与冻结历史后缀的 mixture | base suffix proposal；排除 cache build |
| warm cache 在线阶段 | 读取已经评分的 replay 记录 | fresh-only；排除历史生成与评分 |
| cache build + 首次 warm 查询 | 历史生成、评分和首次在线查询 | fresh-only |
| 历史树，始终草稿 | 所有 active batch 启用固定草稿长度 | 普通自回归解码 |
| 历史树，负载感知 | 只在小 active batch 启用草稿 | 普通自回归解码 |
| 固定 rollout 条件 IS | 每候选使用固定 evaluation rollout 数 | progressive/SMC 对照 |
| pilot/evaluation 分离 | pilot 估计预算，独立 evaluation 计算最终权重 | 固定 rollout 条件 IS |
| 流式奖励 + run-ahead | progressive IS 加完成回调和低优先级草稿 | pilot/evaluation 分离 |
| SMC forest，fresh-only | branch lookahead 全部新生成 | 条件后缀复用对照 |
| SMC forest，条件后缀复用 | 匹配时继承并一次性消费条件后缀 | 相同 SMC 的 fresh-only 路径 |

### 17.8 公平比较与复现

| 优化 | 固定分母 |
| --- | --- |
| 连续批处理 | 同方法逐 prompt 执行 |
| token tree | 相同 workload 的普通自回归解码 |
| broker | 丢弃 partial 后重生成 |
| streaming reward | 整批生成后提交相同 reward |
| MH prefetch | 同更新数普通 reward MH |
| delayed acceptance | 同 proposal 的普通精确 MH |
| warm replay | fresh-only；cache build 分列 |
| SMC reuse | 相同 SMC 的 fresh-only 路径 |
| vLLM | 同模型、dtype、GPU 数与 workload 的 Transformers |

执行层入口：

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python experiments\benchmark_rollout_infra.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers.json

.\.venv\Scripts\python experiments\benchmark_is_mh_reuse.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers_is_mh.json
```

<a id="alg-code-index"></a>
## 18. 代码与验证入口

| 内容 | 代码 | 主要测试 |
| --- | --- | --- |
| 幂分布与奖励 MH | [`mh.py`](../../src/inference_scaling/algorithms/mh.py) | `tests/test_mh.py` |
| 条件 IS 与 off-policy 修正 | [`conditional_energy.py`](../../src/inference_scaling/algorithms/conditional_energy.py) | `tests/test_conditional_energy.py` |
| replay 恒等式与 fresh/reserve | [`base_replay.py`](../../src/inference_scaling/algorithms/base_replay.py) | `tests/test_replay.py` |
| 动态候选和预算分配 | [`dynamic_is.py`](../../src/inference_scaling/algorithms/dynamic_is.py) | `tests/test_dynamic_is.py` |
| progressive pilot/evaluation | [`progressive_is.py`](../../src/inference_scaling/algorithms/progressive_is.py) | `tests/test_progressive_is.py` |
| frozen streaming estimator | [`streaming_is.py`](../../src/inference_scaling/algorithms/streaming_is.py) | `tests/test_streaming_is.py` |
| SMC rollout forest | [`smc_forest.py`](../../src/inference_scaling/algorithms/smc_forest.py) | `tests/test_smc_forest.py` |
| delayed/prefetch/replay MH | [`mh_acceleration.py`](../../src/inference_scaling/algorithms/mh_acceleration.py) | `tests/test_mh_acceleration.py` |
| 连续批处理与评分缓存 | [`batching.py`](../../src/inference_scaling/backends/batching.py)、[`cache.py`](../../src/inference_scaling/backends/cache.py) | `tests/test_batching_backend.py`、`tests/test_score_cache.py` |
| Transformers 后端 | [`transformers_backend.py`](../../src/inference_scaling/backends/transformers_backend.py) | `tests/test_transformers_backend.py` |
| vLLM 后端与动态 suffix | [`vllm_backend.py`](../../src/inference_scaling/backends/vllm_backend.py)、[`vllm_suffix_proposer.py`](../../src/inference_scaling/vllm_suffix_proposer.py) | `tests/test_vllm_backend.py`、`tests/test_vllm_suffix_proposer.py` |
| token tree、streaming、run-ahead | [`acceleration.py`](../../src/inference_scaling/acceleration.py) | `tests/test_acceleration.py` |
| 部分 rollout broker | [`rollout_broker.py`](../../src/inference_scaling/rollout_broker.py) | `tests/test_rollout_broker.py` |
| token/FLOPs 账本 | [`compute.py`](../../src/inference_scaling/compute.py) | `tests/test_compute_accounting.py` |

有限状态测试核对转移概率、权重恒等式、样本生命周期和批处理随机流；真实模型实验核对模型概率、token
轨迹、分模型 FLOPs 和墙钟。
