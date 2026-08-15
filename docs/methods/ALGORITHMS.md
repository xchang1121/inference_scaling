# 推理算法实现：目标分布、估计量与数据复用

本文档按目标分布、有限预算估计量、实现和误差来源说明已实现算法。执行机制见
[推理基础设施实现](INFRASTRUCTURE.md)。实验数值见
[GSM8K 方法质量与计算量实验](../reports/GSM8K_3090_ALIGNED_RESULTS.md)。

## 1. 统一记号与实现边界

给定 token 化提示 \(x\)，记基础模型的完整生成分布为

\[
p(y\mid x)=\prod_{t=1}^{|y|}p(y_t\mid x,y_{<t}).
\]

在已经生成前缀 \(g\) 时，下一段候选记为 \(z\)，候选后的补全记为 \(u\)。奖励写作
\(r(g,z,u)\)，奖励温度写作 \(\tau>0\)。仓库中最常用的显式奖励目标是

\[
\pi_r(y\mid x)
=\frac{p(y\mid x)\exp\{r(y)/\tau\}}
       {\sum_{y'}p(y'\mid x)\exp\{r(y')/\tau\}}.
\tag{1}
\]

另一类目标是幂分布

\[
\pi_\alpha(y\mid x)
=\frac{p(y\mid x)^\alpha}{\sum_{y'}p(y'\mid x)^\alpha},
\qquad \alpha>0.
\tag{2}
\]

本文使用三类性质：

- **平稳分布精确**：MH 转移核保持指定目标不变；有限更新轮次仍有链的收敛误差。
- **估计量无偏**：普通 IS 或 replay 恒等式对条件能量给出无偏估计；有限候选数下的归一化重采样仍是近似。
- **执行等价**：批处理、流式完成和预取保持随机请求与统计量固定。

下文中，MH 指 Metropolis--Hastings，IS 指 Importance Sampling（重要性采样），SIR 指
Sampling-Importance-Resampling（采样--重要性加权--重采样），SMC 指 Sequential Monte Carlo
（序贯蒙特卡洛），GRPO 指 Group Relative Policy Optimization（组相对策略优化）。

重要性修正要求 \(p(y)>0\Rightarrow q(y)>0\)。硬 top-k/top-p 可能破坏该条件；权重截断以偏差换取
有限权重范围。

<a id="alg-overview"></a>
## 2. 方法总览

| 方法 | 采样或估计对象 | 有限预算下的性质 | 主要实现 |
| --- | --- | --- | --- |
| Base / greedy / beam / Best-of-\(N\) | 基础模型采样或确定性搜索 | 基线分布或奖励最大化 | `experiments/gsm8k_reproduction.py` |
| 幂分布 MH | 式 (2) | 转移核精确；有限更新存在收敛误差 | `algorithms/mh.py::run_mh_chain` |
| 奖励目标 MH | 式 (1) | 转移核精确；每次 proposal 通常需完整奖励 | `algorithms/mh.py::run_reward_mh_chain` |
| 条件能量 IS | 式 (1) 的逐 block SIR | \(K,M\to\infty\) 时趋近目标 | `algorithms/conditional_energy.py` |
| off-policy 条件 IS | 同上，补全来自其他 proposal | 未截断普通 IS 对能量无偏 | `algorithms/conditional_energy.py` |
| proposal-energy | \(p(z)\,\mathbb E_q[e^{r/\tau}\mid z]\) | 有意改变目标的消融 | 同上，`apply_importance_correction=False` |
| base 候选 rollout replay | 式 (1) 的逐 block SIR | history + fresh-tail 能量估计无偏 | `algorithms/base_replay.py` |
| 动态候选 IS | 辅助候选、外层 IS、replay | 使用实际候选 proposal 的 \(p/q_c\) | `algorithms/dynamic_is.py` |
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

下文给出分块条件能量、fresh-tail replay、动态候选和冻结 evaluation 的公式与实现。

<a id="alg-report-labels"></a>
### 实验标签定义

标签按参数、目标、候选、rollout、概率修正和输出规则定义。`verifier` 表示 GSM8K 标准答案的确定性
数值检查器。

#### 直接生成与训练基线

| 报告名称 | 使用的参数 | 推理时的实际操作 |
| --- | --- | --- |
| Base | 原始 1.5B 基础模型 \(\theta_0\) | 每步按 \(p_{\theta_0}(\cdot\mid x,y_{<t})\) 随机抽取一个 token；主实验温度为 1 |
| Beam-8 | 原始 1.5B 基础模型 \(\theta_0\) | 保留累计 log-probability 最高的 8 条前缀，最终返回最高分序列 |
| 自一致性投票-8 | 原始 1.5B 基础模型 \(\theta_0\) | 独立随机生成 8 条序列，返回数值答案众数对应的序列 |
| GRPO 参数 + 随机采样 | GRPO 训练后的 1.5B 参数 \(\theta_{\mathrm{GRPO}}\) | \(y_t\sim p_{\theta_{\mathrm{GRPO}}}(\cdot\mid x,y_{<t})\)，温度为 1 |
| GRPO 参数 + 贪心解码 | GRPO 训练后的 1.5B 参数 \(\theta_{\mathrm{GRPO}}\) | \(y_t=\arg\max_v p_{\theta_{\mathrm{GRPO}}}(v\mid x,y_{<t})\) |

#### MH、条件 IS 与 replay 标签

| 报告名称 | 候选或状态 | rollout / proposal | 奖励与概率修正 |
| --- | --- | --- | --- |
| 幂分布 MH | 当前完整序列 | 1.5B 基础模型生成后缀 proposal | 目标为 \(p_{\theta_0}(y\mid x)^\alpha\) |
| verifier-MH | 当前完整序列 | 1.5B 基础模型生成后缀 proposal | 使用终局正确性奖励，目标为式 (1) |
| 标准条件 IS | 候选 block 来自 1.5B base | rollout 来自 1.5B base | 数值众数奖励；修正因子为 1 |
| 0.5B rollout proposal 条件 IS | 候选 block 来自 1.5B base | rollout 来自 0.5B 模型 \(q\) | 数值众数奖励；乘 \(p_{\theta_0}/q\)；默认截断 log-ratio |
| 0.5B rollout proposal IS（无截断） | 与上一行相同 | 与上一行相同 | 使用未经截断的 \(p_{\theta_0}/q\) |
| 0.5B rollout 奖励重加权 | 候选 block 来自 1.5B base | rollout 来自 0.5B 模型 | 数值众数奖励；目标为式 (12) |
| 标准 verifier-IS | 候选和 rollout 都来自 1.5B base | 1.5B on-policy rollout | 数值正确性奖励；估计式 (7) |
| 0.5B rollout proposal verifier-IS | 候选 block 来自 1.5B base | 0.5B rollout | 数值正确性奖励；乘 \(p_{\theta_0}/q\) |
| 0.5B rollout verifier 奖励重加权 | 候选 block 来自 1.5B base | 0.5B rollout | 数值正确性奖励；目标为式 (12) |
| fresh-only | 候选来自 1.5B base | 本轮 rollout 全部新生成 | history 数为 0 |
| warm replay | 候选来自 1.5B 基础模型 | 消费已建库 history，并生成独立 fresh tail | 按式 (14) 修正 history；`warm` 表示历史库已存在 |
| base 候选 + 固定 fresh | 候选来自 1.5B base | 每个候选使用固定数量 fresh rollout | 动态候选实验的基线 |
| 动态候选 + 固定 replay | 候选来自式 (15) 的 base/辅助模型 mixture | 命中时使用固定 history/fresh 数量 | 候选层乘式 (16)，rollout 层使用式 (14) |
| 动态候选 + 方差--成本分配 | 与上一行相同 | history/fresh 数量由独立 design 数据决定 | 最终 evaluation 预算按式 (19) 冻结 |

<a id="alg-baselines"></a>
## 3. 生成与训练基线

### 3.1 Base、greedy、beam 与 Best-of-\(N\)

`base` 按配置温度从基础模型抽样；`greedy` 逐 token 取最大概率项；beam search 保留累计对数概率最高的
若干前缀。

Best-of-\(N\) 先独立生成 \(y_1,\ldots,y_N\sim p\)，再按奖励或 self-consistency 规则选择一个序列：

\[
\widehat y=\arg\max_{1\le i\le N}\widehat r(y_i).
\tag{3}
\]

式 (3) 随 \(N\) 增大趋向奖励最大化。数值答案众数相同时，实验按模型 log-probability 确定性破同票。

### 3.2 GRPO 对照

本仓库中的 GRPO 使用同一基础模型和 GSM8K 数值正确性奖励进行参数训练。若忽略参数化限制，一个带 KL
正则的理想策略优化问题具有式 (1) 的形式；实际 GRPO 只通过有限 rollout、组内相对优势和有限梯度更新去近似
该目标。报告分别列出训练 FLOPs 与训练后采样 FLOPs；单次推理成本指训练完成后的生成成本。

训练得到固定策略 \(p_{\theta_{\mathrm{GRPO}}}\)。实验分别采用温度 1 随机采样和逐 token argmax 解码。

训练入口为 [`experiments/train_gsm8k_grpo.py`](../../experiments/train_gsm8k_grpo.py)，精确数值奖励实现为
[`evaluation/grpo_reward.py`](../../src/inference_scaling/evaluation/grpo_reward.py)。

<a id="alg-power-mh"></a>
## 4. 幂分布后缀 MH

固定生成长度为 \(L\)。当前状态为 \(y=(y_1,\ldots,y_L)\)，每次更新从所有后缀起点
\(s\in\{0,\ldots,L-1\}\) 中均匀抽取一个，保留 \(y_{<s}\)，再从 proposal
\(q_s(\cdot\mid x,y_{<s})\) 生成新后缀 \(v\)。接受概率为

\[
A(y\to y')=
\min\left\{1,
\exp\left[
\alpha\bigl(\log p(v\mid x,y_{<s})-\log p(y_{\ge s}\mid x,y_{<s})\bigr)
+\log q_s(y_{\ge s}\mid x,y_{<s})-\log q_s(v\mid x,y_{<s})
\right]\right\}.
\tag{4}
\]

候选前缀相同，切点选择概率正反相同，因此式 (4) 为完整 Hastings 比。温度 proposal 的逐前缀归一化
常数进入 \(q_s\) 的正反概率。

实现按 `block_size` 逐步扩展到 \(L\)，并在每个长度执行 `steps_per_block` 次后缀更新。最终长度上的有限更新
结果仍含 MCMC 误差。由于切点 \(s=0\) 能以正概率重生成整段，且未截断 softmax proposal 在有限词表、
固定长度空间上处处为正，转移矩阵任意两行都有正重叠。写

\[
\delta(K)=1-\min_{y,y'}\sum_v\min\{K(y,v),K(y',v)\}<1,
\]

则最终长度的核满足

\[
\left\|\mu K^n-\pi_\alpha\right\|_{\mathrm{TV}}
\le \delta(K)^n.
\tag{5}
\]

真实模型实验报告更新轮次、接受率和输出诊断；收缩常数需要显式转移矩阵 \(K\)。

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

\[
A_r(y\to y')=\min\left\{1,
\exp\left[
\log\frac{p(y'_{\ge s}\mid x,y_{<s})}{p(y_{\ge s}\mid x,y_{<s})}
+\frac{r(y')-r(y)}{\tau}
+\log\frac{q_s(y_{\ge s}\mid x,y_{<s})}{q_s(y'_{\ge s}\mid x,y_{<s})}
\right]\right\}.
\tag{6}
\]

当 \(q_s=p(\cdot\mid x,y_{<s})\) 时，基础模型与 proposal 项抵消，只剩
\(\min\{1,e^{(r(y')-r(y))/\tau}\}\)。代码仍保留展开后的四项，因而同样支持任意可精确评分、具有完整
support 的温度 proposal。与式 (5) 相同，整段重生成使有限状态链在通常条件下几何收敛到 \(\pi_r\)。

奖励在实现中是完整生成序列的函数。数值正确性、外部 verifier 等只能在完整 proposal 后得到时，每次普通
MH 更新都要完成整段后缀并调用奖励；降低这部分成本的方法见
[两阶段 MH](#alg-delayed-mh)与[执行层 proposal-tree 预取](INFRASTRUCTURE.md#infra-mh-prefetch)。

<a id="alg-conditional-is"></a>
## 6. 条件能量 IS

在已生成前缀 \(g\) 之后，式 (1) 对下一 block \(z\) 的条件分布可写为

\[
\pi_r(z\mid x,g)\propto p(z\mid x,g)h(g,z),
\qquad
h(g,z)=\mathbb E_{u\sim p(\cdot\mid x,g,z)}
\left[e^{r(g,z,u)/\tau}\right].
\tag{7}
\]

标准条件 IS 的一次 guidance step 为：

1. 生成 \(M\) 个候选 \(z_m\sim p(\cdot\mid x,g)\)；
2. 对每个候选生成 \(K\) 条 on-policy 补全 \(u_{mk}\sim p(\cdot\mid x,g,z_m)\)；
3. 计算
   \[
   \widehat h_m=\frac1K\sum_{k=1}^K e^{r(g,z_m,u_{mk})/\tau};
   \tag{8}
   \]
4. 以 \(\widehat h_m/\sum_j\widehat h_j\) 的概率选择候选并追加到 \(g\)，随后进入下一 block。

候选来自 \(p\)，因此 SIR 选择权重为条件能量 \(\widehat h_m\)。当
\(K\to\infty\) 时式 (8) 收敛到 \(h\)；当候选数 \(M\to\infty\) 时，sampling-importance-resampling
输出趋近式 (7)。有限 \(K,M\) 以及逐 block 重复选择共同构成实际近似误差。

关键实现直接在 log 域求均值并重采样：

```python
log_energy = logmeanexp(rollout.log_weight for rollout in evaluations)
probabilities = softmax([candidate.log_energy for candidate in candidates])
selected_index = rng.choice(len(candidates), p=probabilities)
```

实现入口为
[`run_conditional_is`](../../src/inference_scaling/algorithms/conditional_energy.py)，候选与所有 rollout 都按
异构请求展平为批次；执行细节见[重复前缀 KV 复用](INFRASTRUCTURE.md#infra-prefix-kv)。

<a id="alg-offpolicy-is"></a>
## 7. off-policy 补全与主模型重评分

若补全由 proposal \(q(u\mid x,g,z)\) 生成，则式 (7) 改写为

\[
h(g,z)=\mathbb E_{u\sim q}
\left[
e^{r(g,z,u)/\tau}
\frac{p(u\mid x,g,z)}{q(u\mid x,g,z)}
\right].
\tag{9}
\]

对应普通 IS 估计量为

\[
\widehat h_m=\frac1K\sum_{k=1}^K
\exp\left\{
\frac{r_{mk}}{\tau}
+\log p(u_{mk}\mid x,g,z_m)
-\log q(u_{mk}\mid x,g,z_m)
\right\}.
\tag{10}
\]

式 (10) 未截断时对 \(h(g,z_m)\) 无偏。实践中 proposal 可以是 0.5B 模型，候选 \(z_m\) 仍完全由
1.5B 基础模型生成；“1.5B 重评分”只是在小模型补全完成后，用基础模型一次批量前向计算式 (10) 中的
\(\log p(u_{mk}\mid x,g,z_m)\)。生成时已保存的 \(\log q\) 不需要再次计算。

```python
raw_log_ratio = base_logprob - proposal_logprob
applied_log_ratio = raw_log_ratio
if importance_log_ratio_clip is not None:
    applied_log_ratio = clip(raw_log_ratio, -clip_value, clip_value)
log_weight = reward / reward_temperature + applied_log_ratio
```

截断 \(\operatorname{clip}(\log p/q,-c,c)\) 将式 (9) 改为有偏估计。报告记录 raw ratio、applied ratio、
截断次数和 effective sample size（ESS）。

<a id="alg-proposal-energy"></a>
### 7.1 proposal-energy 目标

设置 `apply_importance_correction=False` 时，权重仅为 \(e^{r/\tau}\)：

\[
\widehat h^{(q)}(g,z)=\frac1K\sum_{k=1}^K e^{r(g,z,u_k)/\tau},
\qquad u_k\sim q(\cdot\mid x,g,z).
\tag{11}
\]

此时逐 block 目标为

\[
p(z\mid x,g)\,
\mathbb E_{u\sim q(\cdot\mid x,g,z)}[e^{r(g,z,u)/\tau}],
\tag{12}
\]

式 (12) 使用 1.5B 候选、0.5B 补全和奖励权重，主模型重评分成本为 0。该路径记为 proposal-energy
目标。两种路径的质量与
分模型 FLOPs 见[1.5B 重评分消融](../reports/GSM8K_3090_ALIGNED_RESULTS.md#15b-rescoring-ablation)。

<a id="alg-base-replay"></a>
## 8. base 候选上的 rollout replay

历史 completion 来自一个可精确评分的 behavior mixture \(b(u\mid x,g,z)\)。令

\[
w(u)=\frac{p(u\mid x,g,z)}{b(u\mid x,g,z)},
\qquad A(u)=e^{r(g,z,u)/\tau},
\]

并取截断常数 \(c>0\)。实现使用恒等式

\[
\mathbb E_b[\min\{c,w(u)\}A(u)]
+\mathbb E_p\left[\left(1-\frac{c}{w(u)}\right)_+A(u)\right]
=\mathbb E_p[A(u)].
\tag{13}
\]

逐点验证式 (13)：当 \(w\le c\) 时，左边第一项在同一离散样本空间上贡献 \(pA\)，第二项为 0；当
\(w>c\) 时，两项分别贡献 \(cbA\) 与 \((p-cb)A\)。因此，使用 \(H\) 条 history 与 \(F\) 条独立 fresh
base rollout 的估计量

\[
\widehat h=
\frac1H\sum_{i=1}^H\min\left\{c,\frac{p(u_i)}{b(u_i)}\right\}A(u_i)
+\frac1F\sum_{j=1}^F
\left(1-c\frac{b(v_j)}{p(v_j)}\right)_+A(v_j)
\tag{14}
\]

对式 (7) 的条件能量无偏。`corrected_replay_log_energy` 在 log 域分别计算两项均值，再执行 `logaddexp`：

```python
history_term = min(log(c), log_p - log_b) + reward / tau
if log_p - log_b <= log(c):
    fresh_term = float("-inf")
else:
    fresh_term = log1p(-exp(log(c) + log_b - log_p)) + reward / tau
log_energy = logaddexp(logmeanexp(history_terms), logmeanexp(fresh_terms))
```

当 \(H=0\) 时，算法使用 fresh base rollout 的式 (8)。历史中存在多个 behavior 版本时，\(b\)
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

\[
q_c(z\mid x,g)=(1-\lambda)p(z\mid x,g)+\lambda a(z\mid x,g),
\qquad 0\le\lambda<1,
\tag{15}
\]

其中 \(a\) 可以是辅助模型或依赖先前候选的 proposal。基础分量给出
\(p(z)>0\Rightarrow q_c(z)>0\)。每个候选使用其实际 proposal 计算外层比值

\[
\rho(z)=\frac{p(z\mid x,g)}{q_c(z\mid x,g)}.
\tag{16}
\]

候选最终 log weight 为

\[
\log W_m=\log\rho(z_m)+\log\widehat h(g,z_m),
\tag{17}
\]

其中 \(\widehat h\) 可由 fresh rollout 或式 (14) 的 replay 估计得到。静态辅助 proposal 会按真实 sampling
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

对候选 \(i\) 和来源 \(s\in\{\text{history},\text{fresh}\}\)，记单样本标准差为
\(\sigma_{i,s}\)、成本为 \(c_{i,s}\)、外层比值为 \(\rho_i\)、分配数量为 \(n_{i,s}\)。忽略整数与容量约束时，
实现近似最小化

\[
\sum_{i,s}\frac{\rho_i^2\sigma_{i,s}^2}{n_{i,s}}
\quad\text{s.t.}\quad
\sum_{i,s}c_{i,s}n_{i,s}\le C.
\tag{18}
\]

拉格朗日一阶条件给出

\[
n_{i,s}\propto \frac{\rho_i\sigma_{i,s}}{\sqrt{c_{i,s}}}.
\tag{19}
\]

代码先按式 (19) 求连续解，再施加逐候选 history 上限、相同 replay key 的共享容量、每个非终止候选的
最少 fresh 数量，最后执行确定性的 floor + largest-remainder 舍入。方差与成本只能来自 `design`；
`rollout_budget_provider` 的输入为候选、终止标记、库存数量和 design 统计量。

<a id="alg-progressive-is"></a>
## 10. progressive pilot / evaluation IS

当不同候选的补全长度、模型成本或权重方差差异较大时，固定 \(K\) 可能浪费预算。progressive 版本先为每个
候选生成少量 pilot rollout，估计

\[
\widehat\sigma_i=\operatorname{Std}
\left[\exp\{\ell_{ik}-\max_{j,k}\ell_{jk}\}\right],
\qquad
\ell_{ik}=r_{ik}/\tau+\log p(u_{ik})-\log q(u_{ik}),
\tag{20}
\]

并用生成 token 数乘 proposal/base 参数量估计相对成本。随后按式 (19) 冻结 evaluation 数量，再独立生成
新的 evaluation rollout。最终能量只使用 evaluation：

\[
\widehat h_i^{\mathrm{final}}
=\frac1{K_i^{\mathrm{eval}}}
\sum_{k=1}^{K_i^{\mathrm{eval}}}e^{\ell_{ik}^{\mathrm{eval}}}.
\tag{21}
\]

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
[流式奖励计算](INFRASTRUCTURE.md#infra-streaming-reward)。

<a id="alg-smc-forest"></a>
## 12. SMC rollout forest

Sequential Monte Carlo（SMC，序贯蒙特卡洛）版本维护 \(P\) 个前缀粒子。定义理想 lookahead

\[
h(s)=\mathbb E_{u\sim p(\cdot\mid x,s)}[e^{r(s,u)/\tau}].
\tag{22}
\]

从父粒子 \(s\) 按基础模型生成下一 block \(z\) 后，中间目标
\(p(s,z\mid x)h(s,z)\) 相对 proposal 的增量权重为

\[
\Delta(s\to sz)=\frac{h(sz)}{h(s)},
\qquad
\log\Delta=\log h(sz)-\log h(s).
\tag{23}
\]

实现用有限 rollout reservoir 的 `logmeanexp` 估计 \(h\)，按式 (23) 计算 branch 权重，再执行 systematic
resampling。增量在路径上望远镜相消；在精确 lookahead、足够粒子与完整长度下对应式 (1) 的序贯构造。

若父粒子的某条历史完整补全以新 block \(z\) 开头，删掉该 block 后的剩余后缀仍是 \(p(\cdot\mid x,s,z)\)
下的有效条件 rollout，可以继承到子 branch。一个 branch 对应多个粒子时，reservoir 在粒子间分桶，
随后用 fresh rollout 补足。

有限粒子数、有限 branch factor 和有限 rollout reservoir 产生 SMC 近似误差。实现同时报告 ESS、fresh
与 reused rollout 数。

<a id="alg-delayed-mh"></a>
## 13. 两阶段 delayed-acceptance MH

设便宜 surrogate 奖励为 \(\widetilde r(y)\)。第一阶段用
\(p(y)e^{\widetilde r(y)/\tau}\) 的完整 Hastings 比接受 proposal；只有通过时才计算精确奖励。第二阶段接受率为

\[
A_2(y\to y')=
\min\left\{1,
\exp\left[
\frac{r(y')-r(y)-\widetilde r(y')+\widetilde r(y)}{\tau}
\right]\right\}.
\tag{24}
\]

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

冻结历史后缀经验分布 \(h_{\mathrm{emp}}\)，并与基础模型组成混合 proposal

\[
q_s(v\mid x,y_{<s})=(1-\lambda)p(v\mid x,y_{<s})
+\lambda h_{\mathrm{emp}}(v\mid x,y_{<s}),
\qquad 0\le\lambda<1.
\tag{25}
\]

历史命中时可读取现成 suffix，并通过一次并行评分获得 \(p(v)\)；未命中时从基础模型生成。无论来源如何，
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
| 平均负熵 | \(\lvert y\rvert^{-1}\sum_t\sum_v p_t(v)\log p_t(v)\) | 置信度消融 |
| self-certainty | \(-\lvert y\rvert^{-1}\sum_t \lvert V\rvert^{-1}\sum_v[\log\lvert V\rvert+\log p_t(v)]\) | 置信度消融 |

后三类在每个 guidance step 内对全部 candidate rollout 做 min-max 归一化。熵类奖励需要全词表概率，
由精确 Transformers scoring backend 计算。self-consistency
实现见 [`evaluation/consensus.py`](../../src/inference_scaling/evaluation/consensus.py)。

<a id="alg-correctness-matrix"></a>
## 16. 正确性与近似来源

| 设置 | 统计性质 | 诊断 |
| --- | --- | --- |
| 增加 MH 更新轮次 | 目标固定；有限链误差下降 | 更新数、接受率、链间结果 |
| 增加条件 IS 的 \(M,K\) | 渐近目标固定；有限 SIR 误差下降 | 每候选 rollout、ESS、FLOPs |
| off-policy 补全 + 未截断 \(p/q\) | 式 (7) 的条件能量无偏 | 两侧 log-probability、ESS、support |
| 截断 log importance ratio | 有偏稳定化估计 | raw/applied ratio、截断次数 |
| proposal-energy | 目标为式 (12) | `score_calls=0`、分模型 FLOPs |
| replay 恒等式 + 独立 fresh tail | 式 (7) 的条件能量无偏 | behavior 版本、claim、fresh/history 数 |
| 动态候选 + 外层 \(p/q_c\) | 候选来源已校正；保留有限 SIR 误差 | 候选来源、outer ratio、共享容量 |
| pilot 决定 evaluation 数量 | 最终估计仅使用独立 evaluation | pilot/evaluation 分离、冻结预算 |
| 流式到达、连续批处理、预取 | 统计量固定，执行顺序变化 | 请求 id、seed、token/FLOPs、废弃工作 |

<a id="alg-code-index"></a>
## 17. 代码与验证入口

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
| 有限状态快速检查 | [`experiments/toy_mh.py`](../../experiments/toy_mh.py)、[`toy_conditional_is.py`](../../experiments/toy_conditional_is.py)、[`toy_base_replay.py`](../../experiments/toy_base_replay.py)、[`toy_dynamic_is.py`](../../experiments/toy_dynamic_is.py) | 全量 `pytest` |

有限状态测试核对转移概率、权重恒等式和批处理随机流；真实模型实验核对模型概率、token 轨迹、分模型
FLOPs 和墙钟。两份正式报告给出统计结论及适用范围。
