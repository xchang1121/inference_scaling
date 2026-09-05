# 下一轮候选：用同一 verifier 的嵌套树反馈学习预算

2026-09-05。仅设计与 CPU 数学工具，未纳入 R3E 冻结 held-out。
当前 production-candidate controller 仍是 past-cost + draft-surrogate，不暗中更改它。

后续独立候选 R6A 的实现、额外残差证明和参数冻结见
[反馈校正协议](R6A_FEEDBACK_CORRECTED_TREE_PROTOCOL.md)。它不改变 R3E 的运行配置。

## 1. 为什么值得研究

当前 controller 需要比较不同轮的接受长度与成本，而每轮上下文不同。
一次较大的树验证其实包含较小前缀树所需的节点。因此可以在**同一上下文与同一组 target draws**
下计算多个预算的 hypothetical committed length，减少把任务难度变化误当预算作用的风险。
这不增加 teacher forward，但树遍历、统计与探索都有成本，不能称为零开销。

## 2. 嵌套树 counterfactual reward 定理

在 draft 后、verification 前，固定完整信息 F（包括 draft 噪声、free token 与最大树）。
候选树 T_(n1) subset T_(n2) subset ... subset T_(nm) 都 prefix-closed，
同一逻辑节点的标签、祖先、position 相同。
为最大树每个节点定义独立的 target 随机变量 U_u。定义 C(n) 为只在 T_n 内
沿这些 target draws 遍历所提交的 token 数（free + matched path + terminal）。

若实际验证预算 A>=n，则运行中已经获得计算 C(n) 所需的全部 logits/draws。
将大树的数组限制到前 n 个节点再遍历，即得到同一 F 和同一 U 下的小预算潜在结果。
无需额外 target forward。若两条遍历在某节点分开，必是小树缺少大树中的下一条边；
小树在正确 target token 上终止，大树可以继续。因此 C(n) 随 n 单调不减。

每个 C(n) 的边缘分布分别等于该预算的真实单次 target-draw verifier 分布，
因为限制数组没有改变任何实际访问节点的独立随机变量或正确条件 prefix。
不同预算之间的 C(n) 是有意耦合的，**不能当作独立样本**。

数值限制：不同实际 GPU batch/tree shape 的 logits 可能有舍入差异。
上述识别对同一数学 target 或同一大树计算所得的条件 law 精确成立；
不能未经检查就断言小形状 kernel 的 bitwise 输出也完全相同。

## 3. 选择偏差与 inverse-propensity 证明

只用“大树被选择的轮次”更新小预算 reward，会引入预算策略的选择偏差。
令实际动作 A 按已知分布 mu(a | F) 抽取，且独立于尚未抽取的 target U。
预算 n 的可见概率为

    pi_n(F) = sum_(a >= n) mu(a | F)。

若 pi_n(F)>0，定义

    R_hat(n) = 1{A>=n} C(n) / pi_n(F)。

条件于 F 与全部 U，C(n) 固定，且 A 独立于 U，故

    E_A[R_hat(n) | F,U] = C(n) sum_(a>=n) mu(a|F)/pi_n(F) = C(n)。

再对 U 取期望得到 E[R_hat(n)|F] = E[C(n)|F]。
这只是 reward 的无偏识别，不自动意味着 ratio estimator 无偏或最优策略一致收敛。
如果策略读取本轮 verifier 后才决定是否把这次算成“选了预算 n”，独立性条件被破坏，证明失效。

一种有 full support 的候选策略为 (1-epsilon) exploit + epsilon Uniform(all budgets)。
于是 pi_n >= epsilon/m，且 0<=C(n)<=B+1，故二阶矩有界：

    E[R_hat(n)^2 | F] <= (B+1)^2 / pi_n(F)。

预算越大，能覆盖它的动作越少，估计方差越高。不能忽略 exploration 成本或把大量加权
观察误算成大量独立 prompts。对当前 deterministic controller 不直接套用该无偏结论。

## 4. 成本仍需实测

大树耗时不是小树耗时；不能按节点比例伪造 T(n)。
需要单独的硬件成本模型/真实小预算探测，或对单动作耗时做相应 IPS（支持 mu(n)>0）。
对固定环境下的候选长期 ratio，可考虑 reward - lambda * cost 的分数优化：

    max_n [E C(n) - lambda E T(n)]。

若 lambda=max_n E C(n)/E T(n)，且所有 E T(n)>0，则最大值恰为 0。
证明是 E C(n)-lambda E T(n)<=0，最优 ratio 对应动作取等号。
这不包含跨轮候选/上下文状态价值；真实解码是 semi-Markov 问题，不能从这个恒等式
宣称全局 TPS 最优。后续应首先在 held-out 上验证实际净节省是否大于新增探索/统计成本。

## 5. 下一轮实施门

先验证嵌套 C(n) 的单调性、输出法则与 IPS 恒等式，再做单独的 preview 实验。
采样策略、epsilon、成本模型和 stopping rule 在结果前冻结；保留当前纯成本 controller 作为对照。
如果在线额外收益仍小于最优静态树，不将这些统计技巧包装成论文级突破。

本思路使用通用 counterfactual / importance-weighting 原理；此文未主张其新颖性，
落地前还需检索 speculative decoding 的 partial-feedback、tree-bandit 和预算学习相关工作。
