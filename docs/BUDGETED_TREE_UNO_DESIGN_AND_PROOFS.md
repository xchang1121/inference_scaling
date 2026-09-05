# Budgeted Online Tree Uno：设计、证明与预注册

2026-09-05，R3C。本文先于本设计的 GPU 实验提交。
目标：在固定 target/Uno 权重和单张 RTX 3090 下，提高包含树构建、同步、KV 整理、
在线学习成本的 TPS。树式 speculative decoding 与预算控制不是本项目首创。

## 1. 为什么改变主线

Direct recycling 的候选准确度低；FP32 warm-start 的 TPF 约等于 static，额外操作未回本。
新方案不依赖旧尾部恰好预测正确，而是利用一次 Uno draft 已产生的各位置 logits，
把有限验证计算分配给多个前缀。第一版采用 packed tree，不复制整份 prefix KV 到多个 batch。

在线学习对象是候选 rank 的覆盖统计、树预算的实际延迟和选择策略；没有 online LoRA SGD。
如果这一版本有效，再研究在线神经更新是否能在其净收益上额外回本，不混用两种“在线”。

## 2. 精确的执行定义

循环开始：序列长度 L，KV 长度 L-1，末 token s 尚未缓存。
标准 Uno draft 输入 [s,z_1,...,z_(B-1)]。seed 行 LoRA OFF，noise 行 ON。
保留 seed 的 base KV；从 seed 行 target distribution 抽样 free token f。
其余位置的 draft logits 给出 top-K token 候选。

树 T 的根是 f，深度 0；非根节点 u 的深度 d(u) 为 1...B-1，
标签从 draft 第 d(u) 行的 top-K 中选择。树必须 prefix-closed，同一父节点的子标签互异。
根与每个节点只存一次。根也计入节点预算 N。

一次 base-only forward 验证整树。节点 u 的 attention 仅可见：完整真实 KV prefix、
从根到 u 的祖先及 u 自身。RoPE position = L+d(u)，不是 packed-array index。
因此节点输出 law 是 p(. | h,s,f,path(u))。

随后从根出发，每个到达的节点仅抽一次新的 target token X_u：

1. 提交 X_u。
2. 若它与当前节点的某个子标签相同，进入该子节点。
3. 否则停止；刚提交的 X_u 是下轮 uncached seed。

free token f 在该遍历之前提交。叶子仍抽取一个 terminal/lookahead token。
为了减少同步，可预先独立抽取所有节点的 target draws，然后一次性传回小数组并遍历。
**不能在同一逻辑节点独立抽多次再选一条“最好路径”**；那会改变 target law。
本树核采用 target-draw exact matching，不把多候选误套进单候选 p/q 公式。
sampling 模式可能比标准 Uno 的 maximal-coupling 接受率低，必须单独测量。

## 3. 定理 A：任意已知树的联合输出 law 等于 AR

令 F 包含过去信息、本轮 draft 噪声、free token、树结构、全部策略统计。
条件于 F，树固定；对每个节点使用互相独立且独立于 F 的 target 抽样随机变量 U_u。

根的条件 prefix 是真实 prefix。若走到节点 u，先前已经提交的 token 恰为它的祖先路径。
到达 u 的事件只依赖祖先节点的随机变量，不依赖 U_u。因此在到达事件下，
X_u 仍服从该真实历史上的 p。若没有匹配的子节点，停止不会改变已经提交的 X_u。
若有子节点，则下一步重复同一论证。逐位置条件概率相乘即得 AR 联合分布。
free token 来自隔离的 base seed 行，所以也满足同一结论。对 F 积分并逐 cycle 归纳即完成证明。

推论：任意过去反馈驱动的树预算、拓扑、rank 校准、甚至错误的候选权重都不会改变输出 law，
前提是验证 attention/position 正确、抽样不被重复挑选、target 权重不变。
EOS 与固定 token budget 仅截断相同过程。

此结论对计算所得 target distributions 成立，不自动保证不同 BF16 kernel shape 的 bitwise 一致。
实现门首先用精确 oracle 和 FP32；BF16 的任何序列差异单独报告。

## 4. 定理 B：packed KV 的正确整理

draft 裁去 noise 后 KV 长度为 L；verify 后为 L+N。
若本 cycle 提交 C 个 token（包括 f 与 terminal），保留真实 prefix 的前 L 行，
再按路径顺序取根及已接受节点中前 C-1 行。最后一个输出 token 尚无有效 KV，不能保留。
整理后长度为 L+C-1 = (L-1)+C，恢复循环不变量。

被选择节点的每层 attention 只依赖真实前缀与其祖先，其 position 又等于逻辑位置，
所以每层归纳可证 KV 内容与对这条路径进行因果 forward 相同（数学运算意义）。
只 crop packed buffer 不足以成立：非连续路径必须 gather/scatter 后再 crop。
实现应只移动新写入的短尾部，不能每轮复制全部长 prefix。
提前 EOS/budget 截断时使用实际 C，包含 C=1（只保留旧 prefix）的边界。

## 5. 定理 C：真实覆盖期望与候选评分的区别

记非根节点 u 的 target 路径概率为

P_p(u) = product over edges (v -> w) on path(u) of p(label(w) | prefix(v)).

完整 cycle（不截断）总提交量恒等于 2 + 被进入的非根节点个数。
原因：一个 free token，加每次成功进入对应的一个 token，加最终一个 terminal token。
线性期望给出 E[C | F] = 2 + sum_(u != root) P_p(u)。
不要求位置独立。greedy target 时路径概率均为 0/1，该公式退化为覆盖深度。

目标未知时，以各深度的 rank 概率 r_(d,k) 构造替代权重
w(u)=product r_(d,k_d)。r 的和不大于 1，因此 child weight <= parent weight。
这是性能 surrogate，**不是已经观测到的 target 真概率**。

在固定 N、候选 lattice 和加性 surrogate 下，最大权重的 N-1 个非根节点形成最优前缀树：
若某节点在集合内，其权重不大于任何祖先；按权重递减且祖先优先打破平局即可得到 prefix closure。
因此 best-first frontier heap 精确求解这个 surrogate 优化问题。
这不证明真实 E[C] 最大，更不证明 E2E TPS 全局最优。

第一实现可选择强制包含 greedy top-1 spine，作为受约束版本。
在同一数学 target、greedy 输出和同一 draft 下，任何包含 spine 的树至少覆盖 static spine
所能接受的前缀，但耗时可能更大。不能从接受长度单调性推出加速。

## 6. 在线 rank 校准

只在实际遍历到的 verifier prefix 上观察下一 target token 在该深度 Uno top-K 中的 rank，
包括未覆盖/不在 top-K 的 missing 类；不把不可达分支当真实生成轨迹。
计数 n_(d,k) 与 n_d 可用衰减更新以适应请求内分布变化。
当前 draft 的原始 full-vocabulary softmax top-K 概率为 q_(d,k)，不对 top-K 重新归一化。

r_(d,k) = (n_(d,k) + a q_(d,k)) / (n_d + a), a > 0。

missing 质量保留在分母内；因此 sum_k r_(d,k) <= 1。
树构建只使用之前 cycle 的计数，本轮观察在验证结束后加入。
观察来自策略实际到达的条件分布，存在选择偏差；不声称无偏估计所有反事实分支。
该近似是否值得其成本，用 frozen-vs-online 消融检验。

## 7. 在线预算与严格的性能边界

允许预算来自有限集合，例如 N in {8,16,32,64}。每个动作实际 cycle 成本必须包括
draft、top-K、小数组同步、建树、mask、verify、采样、KV 整理、在线更新。
完成后分别平滑 tokens 与 seconds，不能平均逐 cycle 的 TPS。

第一版先测试固定预算，确认在本机存在净收益空间；再引入有限 warmup/probe 的在线 controller。
在线选择可用 (2+sum w(u)) / estimated_cost(N)；任何未测预算都需要显式探索和成本记账。
候选集很小，直接比较全部预算；不假设真实 CUDA 延迟连续或凸，也不在首个下降点提前停。
请求结束前学习/整理也进入 inclusive E2E。

对于同一状态下的小树 T 和扩展 T'，理想期望速率改善的充要条件是

Delta E[C] > (E[C_T] / E[time_T]) Delta E[time]。

仅在条件状态一致、耗时/奖励估计有效时可使用；真实在线决策还影响未来轨迹和计算边界。
该局部条件不是有限请求的必胜保证。统计确认仍需独立 prompts 与最优静态宽度基线。

## 8. 预注册实现门与 pilot

- 小词表枚举树采样的联合 law；不等深叶子、空子集、随机路由均覆盖。
- attention mask 不泄漏兄弟/后代；position 相同的兄弟彼此不可见。
- 内容依赖 fake model：packed KV 与逐 token AR 一致，包含 EOS/budget 截断。
- 固定权重检查；树预算、校准只能改变下一次 proposal，不能改 target。
- 真机先 FP32 B=8，K=4，固定 N=8/16/32；包含 static B=4/8/16 对照。
- 四个既有 pilot prompts 用于工程筛选，不视作新测试集。先 1 seed smoke，再至少 3 paired seeds。
- 通过后比较 frozen rank 与 online rank，再冻结新的 12-prompt held-out 配置。
- 所有方法至少 256-token 预热，交替顺序；安装下载背景明确标注。
- 主指标 ratio of total inclusive E2E seconds、绝对 TPS、prompt-cluster CI；同时报告 TPF。

## 9. 文献归因

- [Uno](https://arxiv.org/abs/2609.04010)：共享 base 与 gated diffusion LoRA。
- [Speculative Decoding](https://arxiv.org/abs/2211.17192)：分布保持的背景与标准单候选校正。
- [BASTION](https://arxiv.org/abs/2605.29727)，[官方代码](https://github.com/kaist-ai-osi-lab/BASTION)：
  block-diffusion logits 的预算树构建与硬件成本控制，是本轮最直接的既有工作。
  本项目的单机重实现与 Uno 集成不应写成树方法首创。
- [BlockPilot](https://arxiv.org/abs/2606.31315)：低开销块预算选择的相关方向。
- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)：verifier 反馈在线适配的背景。

潜在可研究差异：共享权重 Uno 的成本结构、请求内 rank 校准、低预算消费 GPU 的完整开销控制。
这些只是研究问题；是否有新颖性和论文级收益，尚待实验与更完整相关工作比较。
