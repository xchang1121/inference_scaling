# Recycling Uno：以端到端 TPS 为目标的在线设计与证明

状态：2026-09-05，R1 设计冻结，R2 HF 实现与小词表证明校验完成；
尚无该算法真机性能结论。

## 1. 问题重述

固定 target AR 模型 \(p_\theta\) 与离线 Uno adapter \(\phi_0\)。
允许随推理更新的是候选 token 状态、候选可靠性统计、耗时统计和路由策略。
在单张 RTX 3090 上，优化包含在线成本后的

\[
\rho = \frac{\text{实际输出 token 数}}
 {T_{\rm prefill}+T_{\rm draft}+T_{\rm verify}+T_{\rm sample}
  +T_{\rm sync}+T_{\rm online}+T_{\rm close}} .
\]

这里“在线”包含运行中更新 proposal state 和 policy，并不等价于每轮更新神经网络权重。
此前 residual 的不足是学习成本和未来轨迹泛化；长后缀 replay 的不足是命中率与索引成本。
本设计的主要状态直接来自刚刚执行的 verifier，因此首请求也能运行。

## 2. 从 verifier 取回被丢弃的信息

设 Uno draft 提出长度 \(B\) 的块
\(d=(d_0,\ldots,d_{B-1})\)。其中 \(d_0\) 来自 base seed 行，
其余来自 diffusion 路径。验证 forward 输出

\[
\ell_j=p_\theta(\cdot\mid h,d_0,\ldots,d_j),\quad 0\le j<B.
\]

本轮提交 \(C\) 个 token 后，最新 seed 是输出位置 \(C-1\)。
上一批 logits 中，对下一输出位置 \(C\) 的预测位于 **\(j=C-1\)**。
因此设置下一轮确定性候选

\[
r_{\rm next}=
 \operatorname{argmax}(\ell_{C-1}),\ldots,
 \operatorname{argmax}(\ell_{B-1}).
\]

若所有 \(B\) 个 proposal 和 lookahead 都已提交，\(C=B+1\)，候选为空。
这些尾部 logits 可能条件于已被拒绝的 token，故它们**不是正确 continuation 的标签**。
它们只是低成本的新 proposal，必须重新在当前真实 prefix 上验证。

对于 recycling forward 输入
\([s,r_0,\ldots,r_{K-1}]\)，第 \(j\) 行预测当前输出位置 \(j\)。
提交 \(C\) 个 token 后，下一轮候选是

\[
r_{\rm next}
=\operatorname{argmax}(\ell_C),\ldots,\operatorname{argmax}(\ell_K).
\]

两种切片相差一行，原因是 Uno 验证输入有一个已 target-sampled 的 free token，
recycling 输入则有一个上轮已经提交但尚未缓存的 seed。单测必须覆盖这一差别。

## 3. 执行循环

1. base prefill 产生第一个输出 token 与 KV。
2. 若无可用候选或 controller 选择 refill，运行标准两遍 Uno。
3. 否则执行一次 base forward：输入 uncached seed 与 \(K\) 个候选。
4. 在真实 target distribution 下验证，提交最长接受前缀与 correction/lookahead。
5. rollback KV 到最新序列长度减一；从本次 verifier logits 提取下一候选。
6. 用本轮实际提交数及完整耗时更新 controller；只影响下一轮选择。

尾部候选保存在 GPU token tensor，不复制全词表到 CPU。请求末尾无需建立全局 n-gram 索引。
第一版 horizon 限制在 \(K\le B-1\)；后续依据独立 validation 比较不同 refill 宽度及 horizon。

## 4. 定理一：任意确定性候选的单 token 校正是 exact 的

固定一轮开始前的全部状态 \(\mathcal F_t\)，候选 \(y\) 已确定。
其真实 proposal law 为 \(q(v)=\mathbf 1[v=y]\)。一般 speculative correction 为

\[
\alpha(y)=\min(1,p(y)/q(y))=p(y),\qquad
p_{\rm residual}(v)=
 \frac{p(v)\mathbf 1[v\ne y]}{1-p(y)} .
\]

当 \(p(y)=1\) 时不会拒绝，residual 无需定义。任意 \(v\) 的总输出概率为

\[
\Pr(X=v)=
p(y)\mathbf 1[v=y]+(1-p(y))p_{\rm residual}(v)=p(v).
\quad\square
\]

一个更省操作的等价实现：直接抽取 \(X\sim p\)；
若 \(X=y\) 则接受候选并继续，否则把 \(X\) 当 correction 并停止本轮。
接受率是 \(p(y)\)，在拒绝条件下 \(X\) 的分布正是 residual。
因此不需要显式构造 one-hot \(q\)、概率除法或第二次 correction 抽样。
greedy 时将 \(p\) 解释为 argmax 处的点质量即可。

## 5. 定理二：整块 verification 的联合分布不变

给定真实 prefix \(h\) 与已固定的候选 \(y_{0:K}\)，forward 计算
\(p_i(\cdot)=p_\theta(\cdot\mid h,y_{<i})\)，以及最后的 lookahead 分布 \(p_K\)。
使用相互独立的新随机变量从每行 \(p_i\) 抽样 \(X_i\)。
从左向右提交 \(X_i\)，直到第一个 \(X_i\ne y_i\)；若全部匹配则提交 \(X_K\)。

证明：第 0 行条件前缀是真实 \(h\)，故首 token 服从正确 target。
若前 \(i\) 个候选均被接受，已提交 token 恰好为 \(y_{<i}\)，于是第 \(i\) 行
条件前缀与真实历史相同，且其新随机变量独立于之前的接受事件，输出 law 为正确 \(p_i\)。
若出现 mismatch，提交的 correction 本身来自正确 \(p_i\)，之后停止；
后续不正确前缀下的 logits 与抽样值均不作为输出。对位置归纳即得联合 AR law。
全接受时 lookahead 的全部候选前缀都正确，也同样成立。\(\square\)

对 Uno refill 分支，沿用原本保存的实际 \(q_i\) 和
\(\min(1,p_i(y_i)/q_i(y_i))\) 及 \((p_i-q_i)_+\) 校正。
必须条件于 draft 的辅助噪声及已用采样状态，不能把相关 block proposal 错当边缘独立分布。
本设计不修改该分支的 \(q_i\) 或 verification。

## 6. 定理三：在线候选更新、路由和变长块不破坏 exactness

\(\mathcal F_t\) 包含过去所有 proposal、verifier logits、提交结果、计时、controller 状态，
也允许包含过去被拒绝分支的预测。动作 \(a_t\)、宽度 \(K_t\) 和候选是
\(\mathcal F_t\)-可测的；如路由自身随机，将其新随机变量先加入状态。

条件于该状态，所选择的下一解码核都是定理二或标准 speculative correction 的 exact 核。
对状态再取全期望，目标 law 仍为 \(p_\theta\)。逐轮复合及逐 token 归纳得到

\[
\Pr(x_{1:N}\mid h)=\prod_{i=1}^N p_\theta(x_i\mid h,x_{<i}).
\]

固定最大输出长度与第一次 EOS 的停止规则仅截断上述同一过程，结论继续成立。
严禁用本轮 verifier 结果回溯更改**本轮**保存的 proposal law；它只能影响下一轮。
尾部预测虽 off-policy，但作为下一轮已知的确定性候选完全符合上述条件。\(\square\)

这证明的是计算所得 target distributions 的不变性。不同 BF16 kernel shape 的 logits
可能略有不同，因此与另一个 AR kernel 的 bitwise 输出相等是额外数值要求，
不能由概率证明推出。真机结果要记录首个不一致位置和 backend。

实现验证：三元词表、3-token 完整联合分布、依赖完整历史的 target、两种候选长度及
off-policy tail 更新全部枚举，输出 law 与 AR 的误差门为 \(10^{-12}\)。
真实 KV 内容依赖的 fake model 覆盖 B=2/4/8 与 always/bounded/TPS 三种 policy，
逐 token 输出与单步 AR 相同；disabled 分支额外验证 stochastic RNG 行为与旧 Uno 一致。

## 7. 定理四：KV rollback 不变量

设循环开始时完整序列长度为 \(L\)，KV 长度为 \(L-1\)。

recycling 输入长度 \(K+1\)，forward 后 KV 长度为 \(L+K\)。
本轮提交 \(1\le C\le K+1\) 个 token，所需有效 KV 长度为 \(L+C-1\)，
因此裁掉 \(K+1-C\) 个位置。
此前被接受的输入恰好等于输出，correction/lookahead 本身尚未计算 KV，
故有效 KV 与真实输出前缀完全对应。

Uno draft 后保留 seed，verify 写入 \(B\) 个位置，最终 KV 长度为 \(L+B\)。
提交 \(C\le B+1\) 个 token，裁掉 \(B+1-C\) 即得相同边界 \(L+C-1\)。
EOS 或 token budget 提前截断只会减小 \(C\)，同一公式仍适用。\(\square\)

注意仅改变长度还不够：运行时必须确保被保留位置的 KV 未被 noise 或 rejected
位置覆盖。HF 使用 crop；官方 runtime 使用 paged KV 的 frontier 与调度器容量约束。

## 8. 接受长度与 TPS 的关系

设 \(A\) 为连续接受的候选数，\(C=A+1\)，忽略请求尾部截断。
无需独立同分布假设，尾和公式给出

\[
\mathbb E[C\mid\mathcal F_t]
=1+\sum_{i=1}^K
\Pr(A\ge i\mid\mathcal F_t).
\]

只有当逐位置条件接受率都等于常数 \(\alpha\) 时，才可化为
\((1-\alpha^{K+1})/(1-\alpha)\)；真实分析使用实测 survival curve。
off-policy logits 的 softmax peak 不是未来真实接受概率，只能作候选特征。

对动作 \(a\)，记完成 cycle 的提交量、总耗时为 \(C_a,T_a\)。
持续运行某个平稳动作且满足更新周期可再生/遍历及有限期望时，renewal reward 给出

\[
\rho_a=\frac{\mathbb E[C_a]}{\mathbb E[T_a]}.
\]

这不是 \(\mathbb E[C_a/T_a]\)。controller 应使用 tokens/time 的累计比值或
分子、分母分别平滑后的比值；汇总实验也报告总 tokens/总秒数。
比较 recycling 与 Uno 的局部条件是

\[
\mathbb E[C_R]>\rho_U\,\mathbb E[T_R].
\]

如果 recycling 一个 cycle 约花 Uno 一半时间，而 Uno 平均提交 3 token，
则 recycling 平均提交超过 1.5 token 就有局部收益空间；实际阈值由本机计时确定。

## 9. Controller：先做低成本、有限探索的可测策略

以候选长度区间为状态桶；两个动作是 refill 与 recycle。refill 的宽度先由
validation 选择，避免一个短请求同时探索过多动作。
每完成 cycle 更新该动作的 token/time EMA。候选足够长时允许少量 probe，
之后只有 recycle 的估计 TPS 超过 refill 的估计 TPS 乘 \(1+m\) 才继续；
失败后增加 cooldown，并定期低频重新探索。每个请求从空候选开始。

更新公式（\(\beta\in[0,1)\)）：

\[
u_{t+1}=\beta u_t+(1-\beta)C_t,\quad
v_{t+1}=\beta v_t+(1-\beta)T_t,\quad
\widehat\rho_{t+1}=u_{t+1}/v_{t+1}.
\]

计时区间覆盖路由、tensor 准备、forward、采样、host transfer、KV rollback 和候选更新。
无需额外每轮 GPU synchronize：提交 token 的现有 host transfer 完成同步；
请求级 timer 最后 synchronize，捕获剩余异步开销。计时模式须在两组保持一致。

**性能保证的范围。** 上述 controller 是吞吐启发式，不声称全局最优或保证不回退。
动作会改变下一轮候选状态，因此严格的最优问题是 semi-Markov average reward：

\[
0=\max_a\mathbb E[
C_a-\rho^\star T_a+V(S_{t+1})-V(S_t)\mid S_t=s].
\]

直接最大化单轮 ratio 忽略 \(V(S_{t+1})\)，可能忽略 refill 的未来候选价值。
所以必须比较 always-recycle、bounded-recycle-depth、TPS-gated、static Uno 消融。
若启发式效果不足，再做基于完整 refill→recycle segment 的 reward，
不能用单轮数学公式宣称全局最大 TPS。

对实测同 token budget 的配对请求，设 baseline 时间 \(T_0\)、新时间 \(T_1\)。
净节省的恒等式为
\(\Delta T=T_{\rm removed\ work}-T_{\rm added\ work}\)，包括所有同步/控制开销。
只有 \(\Delta T>0\) 才加速；任何 controller 更新前都无法凭接受率单独保证此条件。

## 10. 文献来源与设计归因

- [Leviathan et al., Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)：
  接受/残差校正与分布保持的基本来源。
- [Fu et al., Lookahead Decoding](https://arxiv.org/abs/2402.02057) 及
  [官方代码](https://github.com/hao-ai-lab/LookaheadDecoding)：
  使用过去 Jacobi 轨迹形成候选再验证；直接复用旧预测属于这一思想谱系。
- [Liu et al., Online Speculative Decoding](https://arxiv.org/abs/2310.07177)：
  请求流 verifier feedback 的在线适配背景；当前版本没有采用在线神经蒸馏。
- [Huang et al., SpecDec++](https://arxiv.org/abs/2405.19715)：
  draft 长度与耗时/接受 tradeoff；其阈值最优性有特定 MDP 假设，不能直接套用到本系统。
- [Uno](https://arxiv.org/abs/2609.04010)：
  frozen base 与 gated diffusion LoRA 的两遍式 draft/verify fallback。
  本项目使用锁定的官方 checkpoint 和 source revision，见 references。

本工作的可检验贡献候选是：Uno refill 与 verifier-tail recycling 的集成、
可测吞吐控制和单张消费 GPU 的实现/评估。尾部重用、Jacobi、exact rejection sampling
本身不是本项目首创。是否有论文级新颖性与泛化收益，需要后续证据决定。

## 11. R3B：verifier warm-start 的扩展假设（结果前记录）

R3 初始英文 pilot 显示 unconditional recycling 的 TPF 接近 1，低于 static Uno 约 1.4。
因此记录一个不同的 proposal-state 更新：不用尾部直接取代 Uno draft，
而是把它作为下一轮 diffusion forward 的噪声初始化。

设当前 seed 对应输出位置 \(-1\)，过去 verifier 提供的候选为
\(r_0,r_1,\ldots\)。将 draft 输入
\([s,z_1,\ldots,z_{B-1}]\) 的前 \(M=\min(K,B-1)\) 个 noise 行替换为
\([r_0,\ldots,r_{M-1}]\)，剩余 noise 仍按原始分布生成。
候选 \(r_0\) 对应下一输出 token，所以必须放在 **输入行 1**，该行 logits 预测输出 token 1；
base seed 行 0 仍产生 free token 0。

可另外把 noise 行的 LoRA 系数设为固定 \(g\in\{0,0.5,1\}\)，只用于 pilot 消融。
这些系数与候选在 forward 之前已确定，base seed/verify 的 LoRA 系数仍为零。
保存的新 proposal law 为

\[
q_{t,i}(\cdot)=
q_{\theta+g\phi_0,i}(\cdot\mid h_t,z_t^{\rm warm}),
\]

并用该实际 law 执行原始 \(\Psi\)-Spec 校正。因此定理三直接适用，
不要求 warm-start noise 服从训练时的 uniform noise。分布外输入可能降低 draft 质量，
这是性能问题，不是跳过验证的理由。

候选只来自上一轮反馈，权重完全不更新；这种在线适配属于 proposal-state adaptation。
它不是 online LoRA gradient descent。预定先比较固定 \(g=1,0.5,0\)，不预言哪个会更好。
若出现正方向，再冻结新 held-out 矩阵。默认保留 \(g=1\)；pilot 不覆盖原 R3 记录。
