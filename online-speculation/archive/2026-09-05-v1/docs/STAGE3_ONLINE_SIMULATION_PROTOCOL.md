# Stage 3：Online Uno 非平稳仿真预注册协议

本协议在正式 20-seed 运行前冻结。此前只运行过 `1,200 tokens × 2 seeds` 的工程烟雾测试，用于发现
异常、验证 JSON 聚合和估算运行时间；没有据此改学习率、loss 权重、策略集合或成功阈值。Stage 3 的目的
是回答三个逐级问题：verifier feedback 能否学到、能否提高算法推进量、在一个明确但仍是合成的 update
成本模型下能否回本。它不声称已经测得真实 GPU backward 的净加速。

## 1. 可控环境

目标模型是 vocabulary size 8 的一阶 Markov LM。对状态 $s$，三个 regime 的转移矩阵分别为
$P^{(0)},P^{(1)},P^{(2)}$；每行包含三个结构化高概率 mode 和非零背景概率。deployment 按生成 token
位置切换：

| token 区间 | target |
| --- | --- |
| $[0,2000)$ | $P^{(0)}$，offline in-domain |
| $[2000,5000)$ | $P^{(1)}$，abrupt shift A |
| $[5000,7000)$ | $(1-\alpha)P^{(1)}+\alpha P^{(2)}$，gradual drift |
| $[7000,10000)$ | $P^{(2)}$，shift B |
| $[10000,12000)$ | $P^{(0)}$，return in-domain |

static Uno draft 只由 $P^{(0)}$ 构造。若进入一轮时最后状态是 $s$，Uno 先用 target 生成 free token，
第 $i$ 个 speculative 位置离 $s$ 有 $i+2$ 步，因此 offline row 为

$$
q_{0,i}(\cdot\mid s)
=0.98\,[P^{(0)}]^{i+2}_{s,:}+0.02/|\mathcal V|.
$$

正式配置固定 $B=8$，即一轮有 1 个 free AR token 和 7 个 speculative token。所有生成都调用 Stage 1
的精确 linear $\Psi$-Spec：首次拒绝从 $[p_i-q_i]_+$ correction，全接受时取 verifier lookahead。

## 2. 在线 fast weights 与更新时序

每个 request/seed 初始化 tabular logit correction 为零：

$$
q_{t,i}(\cdot\mid s)=
\operatorname{softmax}\bigl(\log q_{0,i}(\cdot\mid s)+\delta_{t,s,i}\bigr).
$$

第 $t$ 轮必须按以下顺序执行：

1. 复制并保存实际生成 proposal 的 $q_t$；
2. 用同一个旧 $q_t$ 做 acceptance denominator 和 residual correction；
3. verifier 完成后把 $(s,i,p_{t,i},q_{t,i})$ 写入本地 buffer；
4. 若 stride 到期，才更新 $\delta_t\to\delta_{t+1}$；新分布只供下一轮使用。

单条 feedback 的 loss 固定为

$$
\ell_{t,i}=m_{t,i}0.97^i\left[
D_{\mathrm{KL}}(p_{t,i}\Vert q_{\delta,i})
+0.5D_{\mathrm{TV}}(p_{t,i},q_{\delta,i})
+0.15D_{\mathrm{KL}}(q_{t,i}\Vert q_{\delta,i})
\right].
$$

学习率为 0.35，逐 item SGD，logit-gradient L2 clip 为 1.0。最后一项是 old-draft regularizer；这里
保存完整旧分布是为了先检验算法，不代表真实大词表实现会保留 full-vocab dense replay。

首次拒绝位置记为 $J$，三种 supervision 固定为：

- `full`：所有 verifier canvas row 的 $m_i=1$；
- `on_policy`：$i\le J$ 为 1，hypothetical tail 为 0；
- `discounted_tail`：$i\le J$ 为 1，$i>J$ 为 $0.25^{i-J}$；全接受时全部为 1。

## 3. 策略、随机化与配对

正式运行使用连续 20 个 seed：`20260905..20260924`；每个 seed 跑下列八条路径：

| label | update stride | supervision |
| --- | ---: | --- |
| `static` | 不更新 | — |
| `per_round_full` | 1 | full |
| `stride5_full` | 5 | full |
| `stride10_full` | 10 | full |
| `stride20_full` | 20 | full |
| `stride10_on_policy` | 10 | on-policy |
| `stride10_discounted` | 10 | discounted tail |
| `adaptive_discounted` | 初始 10 | discounted tail |

`stride10_discounted` 是唯一预注册主检验，因为 TTS 已给出 stride 10 的现实先验，而 tail discount 是
Uno verifier canvas 的架构特定假设。其余比较均标为 exploratory；不能在同一批 seeds 上选出最好策略后再把
其未校正区间当确认性证据。

seed 使初始化和随机数流可复现，但不同 proposer 会消耗不同数量/顺序的 proposal、acceptance 和 correction
随机数，因此不是逐 token 完全相同的 target trajectory。结果按 seed 做配对比值；这能消除 seed 层变化，
但不应解释为同一路径的反事实 replay。

## 4. Adaptive stride controller

每轮在相同 verifier canvas 上计算 shadow-static 和 current 的解析推进 proxy：

$$
\widehat\tau(q)=2+\sum_{j=1}^{B-1}
\prod_{i=1}^{j}\left(1-D_{\mathrm{TV}}(p_i,q_i)\right).
$$

proxy efficiency 为 $\widehat\tau/C$。controller 每 100 轮估计
$d=\widehat g_{\text{current}}-\widehat g_{\text{static}}$ 的均值和单侧 90% lower bound
$\bar d-1.645\,\mathrm{SE}(d)$：

- lower bound $>0.005$、mean TV $>0.04$ 且 update cost fraction $<0.30$：stride 向
  `20 -> 10 -> 5 -> 1` 方向移动一级；
- lower bound $<0$ 或 update cost fraction $\ge0.30$：向低频方向移动一级；
- 其余保持。

这只是 confidence-gated stride controller。动态 block size 留到后续真实模型阶段，因为本阶段先隔离
在线学习与 update cadence 的因果作用。

## 5. 成本模型与指标

Stage 2 的 $B=8$ HF fallback 得到 TPF 1.4006917、AR-relative decode speedup 1.3523271。若把一个
AR token forward 的成本设为 1，则两次 Uno forward 的等效成本校准为

$$
C_D+C_V=\frac{2\times1.4006917}{1.3523271}=2.07152795.
$$

online update 成本没有真机测量，中央合成场景固定

$$
C_U(n)=0.35+0.002n,
$$

其中 $n$ 为本次 update 的 feedback item 数；另报告 $0,0.5,1,2,4$ 倍 update cost 敏感性和每条策略的
break-even multiplier。由于前向校准来自真实 Stage 2、反向成本是假设，`tokens_per_cost` 只能称
**forward-equivalent proxy**，不能写成真实 tokens/s。

每条路径报告：

- TPF、tokens/round、spec acceptance；
- current TV 累计值及同一 canvas 上 static-shadow TV；
- update 次数、item 数、cost fraction、每次 update 前后 TV/KL；
- 分 segment efficiency、250-token trace、adaptive controller 事件；
- target NLL，作为 exact sampler 的诊断量而不是优化目标。

统计量是 20 个 seed 配对比值中位数的 30,000 次 percentile bootstrap 95% interval。

## 6. 冻结的判定规则

对主策略 `stride10_discounted` 分三级判定：

1. **学习成功**：`dynamic/static TV regret ratio` 的 95% CI 上界 $<1$；
2. **算法成功**：相对 static 的 paired TPF ratio 的 95% CI 下界 $>1$；
3. **合成成本成功**：中央成本下 paired `tokens_per_cost` ratio 的 95% CI 下界 $>1$。

其他策略也给出相同区间，但“任一策略通过”和“样本内最好策略”只作为探索性结果。无论第 3 条是否通过，
`real_gpu_online_speedup_tested=false`；只有 Stage 4/5 真实 backward 计时能升级为系统加速结论。

## 7. 已知边界

- tabular correction 比 rank-4/8 LoRA 容易优化，结果是可行性上界而非真实 adapter 效果；
- Markov vocabulary 很小，full-vocab TV、replay 内存和 backward 成本不代表 64K vocabulary；
- 每轮跨 regime 边界时按 round 起点归入 segment，最多影响一个短 block；
- 最后一轮按精确 token horizon 截断，但 acceptance/TV 诊断仍来自完整已执行 verifier canvas；
- 本实验验证 post-verification online update 与成本边界，不能替代 Uno-1B 上的真实 online 实现。
