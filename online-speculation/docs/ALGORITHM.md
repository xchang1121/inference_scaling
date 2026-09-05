# Online Uno：当前实现与证明

主实现为 `native_fast_weights.py`，`native_update_graph.py` 捕获完整更新路径；
`native_norm.py` 是独立的推理优化。
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
学习率默认 .001，每 S=16 个 cycles 更新一次，用该间隔最后 R=4 轮的有效数据；
R≤S 且更新后清空缓冲区，保证同一批样本都由当前参数版本生成。最后一轮不再付训练成本。
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

右边只需缓存 a、u、s、重算低秩分支和最后 RMSNorm，并通过冻结的 LM head 传回梯度。
不需要对 28 层 transformer 反传，也不需要额外 teacher forward。
这是固定当前前缀的蒸馏梯度，不是对产生前缀的整个随机生成过程求导。

**复用 draft logits 的梯度命题。** 令 M 为这次更新所有有效行数，τ=1。
保存实际服务 logits ℓᑫ，q=softmax(ℓᑫ)，p 为对应 verifier 分布。LM head 固定，
由 softmax 与 KL 的导数直接得到

\[
 \frac{\partial L}{\partial\ell_i^q}=\frac{q_i-p_i}{M},\qquad
 g_i^h=W_{head}^{\mathsf T}\frac{q_i-p_i}{M},\qquad
 \nabla_\phi L=\sum_i J_{h_i,\phi}^{\mathsf T}g_i^h.
\]

因此把 gʰ 当作外部上游梯度传给重放 hidden 的 backward，和完整 KL 反传的**一阶梯度**相同，
无需再做训练端 LM-head forward 或 softmax backward。这里 detach q 不会丢失一阶梯度：
q−p 正是已经求出的上游导数；我们不是要对这个导数再求二阶导。
实数模型的等式由链式法则给出，测试另与普通 autograd 对照；BF16 的梯度 cast 与 matmul
仍有有限精度边界。诊断中的 head 重放仅在 `--audit-fast` 打开时执行。

该捷径严格要求缓存 q 的参数版本等于当前版本。代码记录每个样本的 version，过期则报错；
不是把任意历史 q 都当成当前 q。a、u、s 虽然不依赖新 φ，历史 logits 却依赖，必须区分。

服务时在 CUDA graph 中把 a、u、s 写入小型静态缓冲区。OFF verify graph 不覆盖它们。
被选中的轮次保存 draft 和 verifier logits，commit 后复制特征成普通（非 inference-mode）张量。
缓冲区只保留最近 R 个 blocks，无全轨迹记录。重放保留每个 block 的 B 行，计算 hidden 后
再取 loss mask；合并 R 个 blocks 可能改变 GEMM shape，但梯度中的 q 使用实际缓存的服务 logits。
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

单轮使用温度 τ=1 的全词表 forward KL：

\[
 L_t(\phi)=\frac1k\sum_{i=1}^{k}
 D_{KL}\left(\operatorname{softmax}(\ell^p_{i-1}/\tau)
 \middle\|\operatorname{softmax}(\ell^q_i(\phi)/\tau)\right).
\]

当前批量更新把最近 R 轮所有有效项相加，再除以总有效行数 M=Σk，而非等权平均每轮 loss。
p stop-gradient，仅更新 A、C。单轮对 logits 的梯度为 (q−p)/(kτ)，
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
包括 reset、特征/分布缓存、梯度矩阵乘、backward、optimizer、同步发布、prefill 和 detokenization。
审计额外重放也计时；它不属于正常优化版本，比较时要注明是否启用。

默认比较原生基线、融合后的静态 Uno、同样融合路径上的 online LoRA。
在 fast-weights 引擎内，`8` 是包含零增量分支与特征缓存的控制组；`plain8` 才是
完全不执行新分支的静态 Uno；`fast8` 是实际在线学习。三者在同一进程内交错测量。
令 C 为无分支成本、δ 为分支成本。在理想稳定区间中，对零分支基线和真正静态基线的比值满足

\[
 R_{zero}=\frac{g_1}{g_0}\frac{C+\delta}{C+\delta+U/S},\qquad
 R_{plain}=R_{zero}\frac{C}{C+\delta}.
\]

因此只比较 `fast8/8` 不足以证明新增分支全部回本。主要净收益比较必须是 `fast8/plain8`，
保留 `8` 仅用于分离在线更新成本与新分支成本，不把 baseline overhead 当作算法收益。
每个 prompt 的相邻两次 repetition 使用互为反序的完整方法列表。若有 K 个方法，
某方法两次的位置索引之和总为 K−1，故每种方法在该配对内的平均位次均为 (K−1)/2。
这排除固定的早/晚出场偏差，不消除任意非线性热漂移、方法间 carryover 或测量噪声；仍报告置信区间。
审计验证实际记录的顺序确实配对；奇数 repetitions 的末次不配对，不宣称完整平衡。

**无分支对照构造。** 在同一冻结模型上额外捕获一组图；捕获时 Python 路径直接跳过
新增 matmul、addition 和特征缓存。图有独立 pool/output，模型权重、KV workspace 与原组相同。
只有 idle 请求边界才能同时切换 graph runner 和 eager 路径开关，`finally` 必须恢复二者。
仅把服务参数设零或改变 Python 布尔值不能删除已捕获图中的算子，所以那不构成无分支对照。
各图串行执行，无跨 pool 的中间值依赖；每轮输入由正常 staging 写入公共 workspace。
测试验证无分支图不读新增增量、不写特征缓存，在线图仍使用新增参数；生成测试另检查 token 一致性。
额外对照图的捕获时间计入 initialization，不计入任何方法的稳态 TPS，生产接入默认不捕获这组图。

权重字节 hash 在计时外比较，包含 base 和 packed/unpacked Uno，排除 KV。
四个旧 prompts 只用于开发，不称为 held-out，也不从训练 loss 或单次 TPS 宣称论文级收益。

## 7. 固定形状的训练 CUDA graph

更新使用固定 N=RB 行缓冲区，而有效行数随接受长度变化。令 vⱼ∈{0,1} 为有效行 mask，
seed、拒绝后的行和未填满的 slots 都为 0。M=Σvⱼ>0 时：

\[
 L=\frac1M\sum_{j=1}^{N}v_j KL(p_j\|q_j),\qquad
 \frac{\partial L}{\partial\ell^q_j}=\frac{v_j}{M}(q_j-p_j).
\]

去掉 v=0 的项便是原有效样本均值，因而 padding 不改变实数目标或梯度。
RMSNorm 和新增 MLP 分支都逐行计算，无效行的零梯度不会流入有效行。
实际训练始终要求至少一个有效样本；初始化 warmup 使用全零 mask、M 的分母 clamp 为 1，
其零梯度仅用于预热，随后把参数、Adam moments 和 step 全部清零/复位。

图中包含解析 head 梯度、末层反传、梯度裁剪、fused Adam 和 BF16 服务权重 copy。
特征与 logits 直接写入固定缓冲区，不在每次更新时重新 cat 或构造动态索引。
CUDA graph 不捕获 CPU 接受判断：仍先完成原生 commit，再检查样本版本，最后重放更新图。
使用独立 graph memory pool，不与 serving graph 并发重放；图外仅一次 metrics 回传屏障。
若 loss、梯度范数或新参数非有限值，重置并中止请求，不让后续 draft 使用坏参数。

**重置不变量。** 设捕获读写地址集合为 P。请求重置只执行 copy_/zero_，
不替换参数、grad、Adam moments 或 GPU step 张量，故地址集合仍为 P；其值恢复为
A=A₀、C=0、moments=0、step=0。按更新次数归纳，下一请求仍从同一初始优化状态开始。
测试同时检查地址不变、两次相同请求重放的参数逐 bit 一致，以及与 eager Adam 的数值接近。
eager 路径保留为梯度/更新对照，不是性能默认路径。浮点融合和 reduction 的差异仍适用第 4 节边界。

## 8. 保留的块长控制器与归因

native_online_policy 保留作独立调度对照：B∈{4,8,16}，锚点 8，2 cycles/epoch，
提交量/耗时 EMA retention=.75，切换门槛 3%，每 16 个适应 epoch 刷新一个臂。
只用真实执行宽度反馈，不更新神经参数；它不是本页的 online LoRA，也不与其首轮评估混合。

- [Uno / 官方锁定源码](https://github.com/ifm-ai/uno/tree/ed2ee36bb7a3aea8732ebc635b3f09490a032ea3)：
  gated diffusion LoRA、two-pass proposal/verification、采样与 KV 管理由原工作提供。
- [Leviathan et al., Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)：
  accept/reject 与正残差校正的分布保持基础。
- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)：部署反馈在线适配 drafter 的先例。
- [Test-Time Speculation, Appendix C](https://arxiv.org/html/2605.09329v2)：
  verifier 监督、strided updates 和反传成本权衡。当前末层闭合重放、同版本微批和解析 head 梯度是本项目的受限改造，
  不是对 TTS 报告的收益或异步实现的复现。
- [PyTorch numerical accuracy](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)：
  数学等价的批量/切片计算不保证浮点 bitwise 相同。
- [PyTorch CUDA graphs](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs) /
  [Adam capturable](https://docs.pytorch.org/docs/2.11/generated/torch.optim.Adam.html)：
  固定地址、side-stream warmup、禁止图内 CPU 同步和可捕获 optimizer 的实现约束。
