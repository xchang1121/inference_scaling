# R6A：反馈校正的 Online Tree Uno

2026-09-05。R3E 已结束，结果已公开；本文件在新控制器 GPU 运行之前提交。
Windows 尚未重启，不能启动官方 Linux runtime。本阶段先实现可移植的算法层，
允许在现有 Windows HF 上做 engineering pilot；官方运行时顺序和门槛不变。

## 1. 从观测到的不足出发

R3E 中在线成本预算未超过 fixed N=32。旧控制器记录真实 cycle 成本，但奖励仍用
draft 概率乘积 surrogate；真实提交量只记入 diagnostics，没有进入选择评分。
这里修正这一点，不把 preferred 改成 32 本身算作在线反馈算法的效果。

与旧版对照时，两者都使用 preferred=32、预算 {8,16,32}、每预算两次初始探测、
相同 cost EMA=0.8、switch margin=0.02、B=8、K=4、固定 top-1 spine。
R3E 数据不再作为未见测试；本阶段沿用旧四个 pilot prompts，仅用于算法筛选。

## 2. 同轮嵌套反馈的低开销实现

令实际验证树的已匹配路径索引为 j_0=0<j_1<...<j_k。对任何 n<=A，
其中 A 为实际验证预算，前 n 个 packed 节点形成前缀闭合树。它的提交量为

    C_t(n) = min(H_t, 2 + sum_(i=1..k) 1{j_i<n})。

H_t 是剩余输出预算与首次 EOS 位置给出的截断上限；无 EOS 时可用剩余输出预算。
大树在小树第一次缺失的边处停止后，小树不能越过缺失祖先，因此该公式等价于重新遍历。
所有小树输出都是同一次大树输出的前缀。可以复用一次 walk，不重新构造子树或重复遍历。
不同预算的反馈有意耦合，不能当成独立样本；n>A 的奖励不可见，不伪造为零。

EOS 的截断以同一组潜在 target draws 定义。对已验证较大树看见的首个 EOS，
小树若在它之前结束，不会受该 EOS 影响；若覆盖到它，恰在该 token 结束。
故按这个上限截断保持每个可见预算的潜在提交量。

## 3. 带选择概率的残差更新

每轮 draft 后、target verification 前，条件信息 F_t 包括过去状态、当前 draft、free token
和完整候选树。令 s_t(n) 为当前树 surrogate，截断到本轮剩余输出预算内。
对各预算保留残差 b_t(n)，初值为 0。实际选择评分为

    R_t(n) = clip(s_t(n)+b_t(n), 1, min(B+1, remaining_t)),
    S_t(n) = R_t(n) / estimated_cycle_seconds_t(n)。

先按 S 与 preferred margin 找到 exploit 动作。完成初始成本探测后，用

    mu_t = (1-epsilon) delta_exploit + epsilon Uniform({8,16,32})

选择实际动作；epsilon=0.15。策略随机数使用独立于模型抽样的 RNG，不能看本轮 verifier。
可见概率 pi_t(n)=sum_(a>=n) mu_t(a)。验证结束后，仅可见预算更新

    b_(t+1)(n) = b_t(n) + eta/pi_t(n) * [C_t(n)-s_t(n)-b_t(n)],

eta=0.05；不可见预算保持不变。初始探测的动作分布为已知点质量，
此时只对 pi=1 的可见预算更新，不宣称这几个轮次识别所有动作。
实际执行的 mu、surrogate 在 choose 时冻结，observe 不根据 verifier 改写它们。

### 定理：条件平均创新方向正确

在 full-support 阶段，A 与本轮未使用的 target 随机变量 U 条件独立。
定义 e_t(n)=C_t(n)-s_t(n)-b_t(n)，条件于 F_t,U 时它固定。则

    E_A[1{A>=n} e_t(n)/pi_t(n) | F_t,U] = e_t(n)。

再对 U 积分，得到

    E[b_(t+1)(n)-b_t(n) | F_t]
      = eta * [E(C_t(n)|F_t)-s_t(n)-b_t(n)]。

它证明的是条件平均残差更新方向，不是 b_t 对每轮条件奖励的无偏预测。
漂移上下文、有限样本、带噪成本和 clipping 都不产生 TPS 最优保证。

### 定理：更新不会因小 propensity 发生外插爆炸

full-support 时 pi>=epsilon/3=0.05，eta=0.05，因此 0<eta/pi<=1。
每次可见更新是旧残差与本次 C-s 的凸组合；初始点质量阶段同样成立。
由于 C 与 s 位于 [1,B+1]，初始化为 0，可归纳得到 b_t 始终在 [-B,B]。
配置验证必须拒绝 eta>epsilon/动作数，不用事后 clip 掩盖不稳定步长。

### 正确性与限制

残差和成本只影响下一轮树选择，不改变 target/LoRA 权重，也不复用已抽样随机变量。
故原树 joint-law 与 KV 不变量证明继续适用。这里只评估已验证的嵌套预算，
与“所有任意 drafter 都有完整信息”不是一回事。
大树的耗时不能当作小树的耗时；成本仍仅更新实际动作的实测完整 cycle。
探索、反馈、额外同步、初始化和结束成本均进入完整 generate-call TPS。

## 4. 实现门

- 枚举小词表，逐预算验证单遍公式等于重新 walk，并覆盖每种输出截断。
- 检查 residual innovation 的精确条件期望恒等式及有界更新。
- 未完成 observe 不能再次 choose；错误预算、缺失反馈、超出范围的反馈必须拒绝。
- 策略 RNG 与模型 RNG 分离；所有状态 request-local。
- 内容依赖 fake model 对比 AR 输出、KV 内容、EOS 与短预算边界；base/adapter 冻结。
- 保留旧 treebudget 路径与全部既有测试，不修改 R3E 的原始数据、协议或方法配置。

## 5. GPU pilot（不是新的 confirmatory study）

FP32 / Windows HF / batch=1 / HighQoS / 固定 256 输出 tokens。
四个既有 pilot prompts，每个三次重复；新 seed 起点 20269005。
方法：AR、linear B=8、fixed tree N=16、fixed tree N=32、
旧 cost-only treebudget preferred=32、新 treefeedback preferred=32。
共 72 次运行；每方法预热 256 tokens，顺序轮换并反向交替。

完整 E2E、所有配对、prompt-cluster 区间及 GPU 快照都保留。逐 token 比较 AR。
只用于筛选，不声称独立测试成功；频率异常继续标注，不移除不利运行。
若发生执行错误，保留 completed=false；修复必须记录，再用独立结果文件重跑。
不会在本组中按结果修改 epsilon、eta、preferred 或其他配置。

## 6. 更直接的文献归因

- [CaDDTree](https://arxiv.org/html/2606.01813v1)：联合考虑候选覆盖与运行成本；
  其停止规则有成本曲线形状假设。本机仍枚举少量离散预算，不假定 CUDA 延迟凸。
- [Not-a-Bandit / HedgeSpec](https://arxiv.org/html/2510.20064v2)：利用 verifier 信息评估
  多个 drafter，并讨论 censoring 和延迟反馈。本实现只有嵌套子树的可见反馈，
  不直接继承其 full-information/no-regret 定理。
- [BASTION](https://arxiv.org/abs/2605.29727)：预算树与在线延迟估计的直接相关工作。
- [BlockPilot](https://arxiv.org/abs/2606.31315)：prefill 后进行一次块预算决策，
  提醒我们将频繁在线控制的成本与更简单策略公平比较。

以上方向均已有研究。R6A 是面向本机 Uno 的实现与验证候选，不主张新的在线学习定理。

## 7. GPU 运行之前的实现审计

参考实现已经通过 191 项 CPU/回归测试及 Ruff 检查。新增测试包括精确枚举的单遍嵌套反馈、
propensity 校正创新方向、有界更新、缺失/重复反馈拒绝、独立 policy RNG、
真实内容依赖 fake model 的 AR/KV/权重隔离与 EOS/输出预算截断。

反馈更新放在 cycle 最后同步和成本标签时间戳之前；时间戳之后只有常数规模的成本记账，
全部过程仍包含在完整 generate-call 的 TPS 分母内。新结果同时记录解析后的全部方法配置，
并增加同 preferred=32 的旧 cost-only 控制器作为直接 secondary baseline。
