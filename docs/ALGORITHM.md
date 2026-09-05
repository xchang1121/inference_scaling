# 当前实现：Uno 的在线块长控制器

对应 [native_online_policy.py](../scripts/native_online_policy.py)。本文仅描述当前维护的实现，不包含历史成绩或阶段记录。

## 1. 在 Uno 上的改动与明确边界

当前实现在线更新每个块长的收益/成本统计，**不更新 diffusion LoRA 或任何神经网络参数**。
它保留为原生推理基线和调度控制器，不能当作“verifier 反馈 → backward → drafter 参数更新”的在线蒸馏实现。

新增逻辑全部在 `NativeWidthPolicy` 和 `generate_online()` 中：choose 建立 pending action，
执行原官方 step，observe 接收真实反馈，下一轮才允许更换块长。请求结束或异常时恢复原 step 和原参数。
原版 FA2、CUDA graphs、融合验证器、紧凑 token 回传、模型、采样及 KV 管理均不修改。

只在 `engine.step()` 的整个 draft–verify–commit 之间修改 `SamplingParams.diffusion_block_size`。
它不是 KV page size，后者一直是 256，不能混淆。

batch=1、单 GPU、线性 Uno、预捕获 B∈{4,8,16}；默认锚点 B₀=8。
每请求新建状态，没有跨题训练或用将来的答案指导本轮选择。每个 epoch 连续执行 2 cycles：
先各探测一个 epoch，顺序 8→4→16；之后利用收益/成本 EMA，每 16 个自适应 epoch 刷新一个臂。
EMA retention=0.75，替代锚点至少需要 3% 的估计优势。这是控制器超参，不是用户认可的“接近”门槛。

当前控制器 不重用一个 B 的接受长度来声称另一个 B 的反事实收益：噪声输入长度、RNG 消耗和数值 kernel
shape 会随 B 改变，不能直接假定实际 q_B=q_B′。此 pinned 线性实现使用 causal draft attention，
并非双向 attention；若另行构造共享噪声前缀，需要重新证明/测试可观测反馈条件。
其他解码结构上的 side-feedback 条件不能直接搬到线性块长。

## 2. 在线更新方程及不变量

epoch e 的宽度 b，实际提交量 cᵢ∈{1,…,b+1}，观测耗时 dᵢ>0，m=2：

\[
 u_e=m^{-1}\sum_i c_i,\quad v_e=m^{-1}\sum_i d_i,
 \qquad
 \mu^c_b\leftarrow\beta\mu^c_b+(1-\beta)u_e,
 \quad\mu^d_b\leftarrow\beta\mu^d_b+(1-\beta)v_e.
\]

第一次观测用 β=0，之后 β=.75。其他臂完全不更新。评分 s_b=μᶜ_b/μᵈ_b；
除探索外取最大评分，若它未超过 (1+.03)s_B₀ 则选 B₀。请求尾部使用实际截断提交量；
不足一个 epoch 的统计只记录，不给下一请求带入状态，也不伪造一个完整 epoch。

**凸组合性质。** 初值为合法观测，0≤β<1。归纳可得 μᶜ_b∈[1,b+1]、μᵈ_b>0，
因此评分有限正数（实现同时拒绝 NaN/Inf/非正时间）。这不证明评分无偏或策略最优。

**探索上界。** E 个完整 epoch，K=3；前 K 个为初始化。之后刷新数恰为
⌊max(0,E−K)/16⌋。因此探索 epoch 数≤K+⌊max(0,E−K)/16⌋。
它仅约束探索次数，不限制每次探索的性能损失；不能由此推出 TPS 损失≤1/16。

每轮有 pending action，只有对应宽度的反馈可清除 pending。选择→完整验证→反馈之后
才能下一次选择；测试覆盖重复选择、错误归属、非法时间和 reset 边界。

## 3. 自适应块长的分布保持证明

令 F_t 包括过去已提交前缀、过往运行时间/接受反馈、policy 状态和过去随机性。
在新一轮随机数产生之前选择 B_t∈F_t。固定 base 参数 θ；adapter φ 在 当前控制器 也固定。
条件于 F_t、B_t 和本轮噪声 z，本轮 proposal q 是确定的分布；采样时保存的 q 在整轮
accept/reject 中不能改写。q 可以与过去历史任意相关，但不能偷看本轮未来 verifier 随机性。

对任一位置的目标 p、proposal q，定义 a(y)=min(1,p(y)/q(y))，q(y)=0 的分支不会被 q 采到。
拒绝总质量

\[
 Z=1-\sum_y\min\{p(y),q(y)\}
   =\sum_y[p(y)-q(y)]_+=D_{TV}(p,q).
\]

Z>0 时 correction r(x)=[p(x)−q(x)]₊/Z；则提交 x 的总概率

\[
 q(x)a(x)+Zr(x)=\min\{p(x),q(x)\}+[p(x)-q(x)]_+=p(x).
\]

Z=0 时 p=q、全部接受。按位置从左到右归纳，接受前缀后下一位置的 p 使用真正已提交的条件；
拒绝处输出 correction 并停止本轮，全部接受则由正确 target 条件采样 lookahead。
Uno causal seed 行和 verify 行 LoRA OFF，noise 行 LoRA ON；控制器不修改这个隔离关系。

固定任意条件历史时每轮的每个提交位置都服从 target kernel。跨轮对 F_t 再做条件期望，
自适应 B_t 的混合仍给出同一 target；停止于 EOS 或固定长度只是截取该过程。
若使用 top-k/top-p/temperature，p 指施加同样采样变换后的目标，而不是未过滤的原始 softmax。
这是一条条件于官方固定 B 实现正确的组合定理，不是对所有 GPU kernel 的形式化认证。

**KV 边界归纳。** 官方解码后 invariant 是 num_cached_tokens=len(seq)−1。
新 B 仅决定 scheduler scratch reservation 和已捕获 block graph；旧已提交 KV 不变。
draft 后回滚到 len(seq)，verify 后提交并再次回滚到新 len(seq)−1。
因为不在轮内改变 B、不动 position IDs、LoRA mask、噪声 KV 回滚或残差分布，下一轮保持相同 invariant。

## 4. 数值精度与“行为相近”

数学 exactness 不等于 BF16 bitwise 一致。不同 B 的矩阵/attention kernel 形状可能产生不同 logits。
若同一历史的数值 kernel 与理想 p 的 TV 距离至多 ε_j，则 maximal coupling 加 union bound 给出
长度 T 的序列 TV ≤min(1,Σ_j ε_j)。本项目没有测出这些 ε_j，因此该式不是数值保证。
greedy 情况：若最大 logit 与第二名间距大于 2δ，且各 logit 误差≤δ，argmax 不变；否则可能分叉。
真实测试须保存每个 token、首个差异和文本，不能从理论等式直接声称硬件输出逐 token 一样。

## 5. 吞吐、学习开销与不能证明的部分

官方每轮已经把紧凑 committed payload 从 GPU 同步到 CPU，当前控制器 不额外调用 CUDA synchronize。
观测 dᵢ 是选 B 开始到官方 step 返回的 exposed wall time，包含 choice 和官方回传，
不含随后 observe 与 trace bookkeeping；后两者以及 controller 构造/序列化、wrapper 安装恢复、
prefill、detokenization 全部包含在外层 E2E 定时。它不是精确分离的 GPU kernel 时间。

记固定长度 N，原版成本 T₀，在线节省 S、增加更新和探索成本 U，则

\[
 \frac{TPS_{online}}{TPS_0}=\frac{T_0}{T_0-S+U}.
\]

只有 S≥U 时净收益才非负；提高 TPF 不足以证明这一点。理想固定状态下更新后的真实比率
ρ₁、旧比率 ρ₀、未来剩余 H tokens、一次更新成本 U 的回本条件为
H(1/ρ₀−1/ρ₁)>U。非平稳请求中这些量未知，当前控制器 并未提供 regret 或全局最大 TPS 保证。

## 6. 文献归因

- [Uno 原文与官方实现](https://github.com/ifm-ai/uno/tree/ed2ee36bb7a3aea8732ebc635b3f09490a032ea3)：
  diffusion proposal、gated LoRA、Ψ-Spec、线性/树解码都属于原工作，不是 当前控制器 创新。
- [Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)：
  accept/reject 和正残差校正的分布保持基础。
- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)：用部署反馈在线适配 draft 的先例；
  当前控制器 的受限动作是宽度，不是它的神经 drafter 蒸馏。
- [Not a Bandit Problem / HedgeSpec](https://arxiv.org/abs/2510.20064)：
  反馈可观测性需要单独证明；当前控制器 没有借用完整反馈假设或引用其 regret 为自己的保证。
