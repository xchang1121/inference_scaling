# Online Uno：当前实现与证明

主实现为 `native_fast_weights.py`；`native_norm.py` 是独立的推理优化。
这里的在线算法**实际更新神经网络参数**，不再以块长控制器代替在线蒸馏。
仅支持锁定的 K2-Horizon / XLLM、单 GPU、batch=1、线性 Uno；默认 B=8。

## 1. 冻结 Uno，在线学习最后一层的低秩增量

冻结 AR 参数 θ 和离线 Uno adapter φ₀。最后一层 MLP down projection 新增
φₜ=(Aₜ,Cₜ)，A∈R^(r×5120)、C∈R^(1536×r)，r=8，共 53,248 个参数。
对某行，a 是 down projection 输入，u 是原 Uno 的输出，s 是最后一次 residual：

\[
 d_t=C_t A_t a,\qquad \widetilde u_t=u+m d_t,\qquad
 h_t=\operatorname{RMSNorm}(\widetilde u_t+s),\qquad
 q_t=\operatorname{softmax}(W_{head}h_t).
\]

m 直接使用官方 gated LoRA mask：draft noise 行为 1，causal seed 为 0；
prefill、verify 的 OFF 路径根本不执行新增分支。C₀=0，A₀ 为固定种子、
标准差 1/√5120 的随机矩阵。因此初始增量为零，且 C 的梯度不因双零初始化消失。
初始化不消耗生成使用的随机数流。

训练参数为 FP32，服务参数为独立 BF16 固定地址张量。每个完整 cycle 内服务版本不变；
commit 返回后才执行 Adam、梯度裁剪（范数上限 1）、copy_ 发布。
学习率默认 .003，每 8 个 cycles 更新一次，仅用该轮数据；最后一轮不再付训练成本。
每请求重置参数和 optimizer，不跨请求学习，不保存新的模型 checkpoint。

## 2. 为什么不需要重跑整个 teacher 或 student

**特征闭合命题。** 条件于本轮已经确定的 prefix、噪声和冻结权重，a、u、s 都不依赖 φₜ。
因为增量插在最后一个 MLP 的 down projection 输出处，其前面的计算没有可训练参数。
因此对于本轮损失 L，链式法则给出

\[
 \nabla_{\phi}L
 =\frac{\partial L}{\partial q}\frac{\partial q}{\partial h}
   \frac{\partial h}{\partial\widetilde u}
   \frac{\partial\widetilde u}{\partial\phi},
\]

右边只需缓存 a、u、s、重算低秩分支、最后 RMSNorm 和 LM head。
不需要对 28 层 transformer 反传，也不需要额外 teacher forward。
这是固定当前前缀的蒸馏梯度，不是对产生前缀的整个随机生成过程求导。

服务时在 CUDA graph 中把 a、u、s 写入小型静态缓冲区。OFF verify graph 不覆盖它们。
需要更新时保存 verifier logits，commit 后复制成普通（非 inference-mode）张量交给 autograd。
重放保留完整 B 行，先计算再取 loss mask；过早切成 k 行可能改变 BF16 GEMM kernel。
审计可比较重放与真实 draft logits；即使数学闭合，也不把有限精度差异默认为零。

**KV 隔离命题。** 每层 K/V 均在该层 attention 内生成；最后 MLP 之后没有 attention。
新增 φ 不参与任何本轮 KV 的计算。因此改变 φ 不要求刷新已提交 prefix 的 KV。
不同 φ 可以通过不同 proposal 改变将来的采样路径，但这不等于污染已有 KV。
官方 draft 噪声 KV 回滚、verify 提交、`num_cached_tokens=len(seq)−1` 均不改动。

## 3. Teacher 对齐和拒绝后的屏蔽

官方一轮 draft 输入为 [seed,z₁,…,z_(B−1)]，输出 [y₀,y₁,…,y_(B−1)]，
y₀ 由 seed 的 base 路径采样。verify 输入为 [y₀,y₁,…]。
**student 第 i 个 noise 行对应 verify 第 i−1 行**（i=1,…,B−1），不能同索引直接对齐。

若在 noise 位置 J 首次拒绝，只有 i≤J 的 teacher 条件使用真实将提交的历史；
i>J 的 teacher 看到了不会提交的 rejected token。当前算法屏蔽这部分，不把它当作 on-policy 监督。

设该轮实际提交 c 个 token。无截断时 c=2+K，K 是连续接受的 noise 数，
包含 clean root 和 correction / lookahead。取

\[
 k=\min(B-1,c-1),\quad w_i=\mathbf1(1\le i\le k).
\]

发生拒绝时 c=J+1，所以 k=J；全部接受时 k=B−1。
输出预算截断只减小 c，因此这个 mask 最多更保守，不引入拒绝后的历史。
只有一个 token 的尾轮 k=0，不训练。这个 mask 是可观测的实际反馈，不是反事实接受估计。

当前使用温度 τ=1 的全词表 forward KL：

\[
 L_t(\phi)=\frac1k\sum_{i=1}^{k}
 D_{KL}\left(\operatorname{softmax}(\ell^p_{i-1}/\tau)
 \middle\|\operatorname{softmax}(\ell^q_i(\phi)/\tau)\right).
\]

p stop-gradient，仅更新 A、C。对 logits 的梯度为 (q−p)/(kτ)，
令 gᵢ=∂L/∂ũᵢ，则实数模型下 ∇C=Σᵢgᵢ(Aaᵢ)ᵀ、∇A=ΣᵢCᵀgᵢaᵢᵀ。
混合精度训练使用 cast 的常规 autograd 梯度；不是对离散浮点舍入函数的真实导数。

Pinsker 给出 TV(p,q)≤√(KL(p‖q)/2)，而逐位置标准拒绝采样接受概率为 1−TV(p,q)。
这个联系只对同一对分布成立。当前 greedy 解码用 τ=0、训练用 τ=1，KL 是平滑代理，
不能把它降低直接等同于 greedy 接受率提高；更不能据此断言未来样本或整个 block 的收益。
mask 的数据分布也依赖当前 drafter，没有无偏全轨迹梯度或 regret 保证。

## 4. 在线更新仍可保持 speculative decoding 的目标分布

令 Fₜ 包括过去已提交前缀、所有过去反馈和当前 φₜ。本轮开始前 φₜ 已确定，
条件于 Fₜ 和本轮噪声，各 proposal 分布 q 固定。接受/拒绝使用**生成时的 q**，
不能使用更新后的 q。代码在整个官方 step 返回以后才发布 φₜ₊₁。

对任一位置的目标 p、proposal q，令 a(y)=min(1,p(y)/q(y))，
q(y)=0 的事件不会由 q 采到。拒绝概率

\[
 Z=1-\sum_y\min(p(y),q(y))=\sum_y[p(y)-q(y)]_+=TV(p,q).
\]

Z>0 时，correction r(x)=[p(x)−q(x)]₊/Z，因而

\[
 \Pr(X=x\mid F_t,\text{有效前缀})
 =q(x)a(x)+Zr(x)=\min(p(x),q(x))+[p(x)-q(x)]_+=p(x).
\]

Z=0 时 p=q、全部接受。按 block 内位置归纳，accepted prefix 后使用正确 target 条件；
首次拒绝处输出 correction 并结束该轮，全部接受则使用正确 lookahead。
再对 Fₜ 取条件期望、按轮归纳，φₜ 如何依赖**过去**反馈不会改变 target 分布。
训练数据包含当前拒绝信息也没问题，因为它只影响下一轮。
top-k/top-p/temperature 必须按官方实现对 p/q 一致变换后再验证。

这是条件于官方固定参数验证器正确的组合定理，不是所有 GPU kernel 的形式化认证。
greedy 的对应论证是只接受 target argmax 一致的 prefix，拒绝处输出 target argmax。
BF16 不同形状可能具有不同的数值 target，不能用实数证明宣称逐 token bitwise 相等。

## 5. 独立推理优化：融合 grouped RMSNorm

native_norm 将多次逐元素运算和 reduction 融合成一个 Triton kernel。
保留上游显式舍入位置：

1. FP32 residual 加法，转换回 BF16；
2. FP32 分组均方与 rsqrt，归一化后转换回 BF16；
3. 乘 norm weight，输出 BF16。

关闭 FP fusion 以避免合并表达式改变这些位置。仍可能因 reduction 顺序、rsqrt 实现而有差异，
所以测试残差精确相等、归一化误差，并在生成层报告实际 token 差异。
数学上每组为 w⊙v/√(mean(v²)+ε)，融合前后相同；有限精度相同则须实测。
融合同时用于 teacher/draft，是系统优化，不记作在线学习带来的增益。

## 6. TPS 的验收和收益边界

理想稳定区间下，静态每轮平均提交 g₀、成本 C₀，在线提交 g₁、额外推理成本 δ、
每 S 轮训练成本 U，则忽略边界时

\[
 \frac{TPS_{online}}{TPS_{static}}
 \approx\frac{g_1}{g_0}\frac{C_0}{C_0+\delta+U/S}.
\]

因此需要 g₁/g₀>1+(δ+U/S)/C₀ 才有净收益。实测使用整个 generate 的时间，
包括 reset、特征缓存、teacher 复制、backward、optimizer、同步发布、prefill 和 detokenization。
审计额外重放也计时；它不属于正常优化版本，比较时要注明是否启用。

默认比较原生基线、融合后的静态 Uno、同样融合路径上的 online LoRA。
fast-weights 引擎的静态 B=8 已包含零增量分支与特征缓存，另测无 fast-weights 的融合引擎
才可评估新增分支全部成本。权重字节 hash 在计时外比较，包含 base 和 packed/unpacked Uno，排除 KV。
四个旧 prompts 只用于开发，不称为 held-out，也不从训练 loss 或单次 TPS 宣称论文级收益。

## 7. 保留的块长控制器与归因

native_online_policy 保留作独立调度对照：B∈{4,8,16}，锚点 8，2 cycles/epoch，
提交量/耗时 EMA retention=.75，切换门槛 3%，每 16 个适应 epoch 刷新一个臂。
只用真实执行宽度反馈，不更新神经参数；它不是本页的 online LoRA，也不与其首轮评估混合。

- [Uno / 官方锁定源码](https://github.com/ifm-ai/uno/tree/ed2ee36bb7a3aea8732ebc635b3f09490a032ea3)：
  gated diffusion LoRA、two-pass proposal/verification、采样与 KV 管理由原工作提供。
- [Leviathan et al., Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)：
  accept/reject 与正残差校正的分布保持基础。
- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)：部署反馈在线适配 drafter 的先例。
- [Test-Time Speculation, Appendix C](https://arxiv.org/html/2605.09329v2)：
  verifier 监督、strided updates 和反传成本权衡。当前只重放最后 MLP 的闭合路径是本项目的受限改造，
  不是对 TTS 报告的收益或异步实现的复现。
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)：
  数学等价的批量/切片计算不保证浮点 bitwise 相同。
