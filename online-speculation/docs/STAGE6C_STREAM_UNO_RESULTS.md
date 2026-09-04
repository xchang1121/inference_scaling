# Stage 6C：Stationary Stream-Uno 真机结果

## 结论

跨请求 persistent learner 和 validation-only checkpoint selection 均按协议工作，但选中的 hard residual
没有泛化到 10 个新 sampling seeds：

| test 主指标（10 pairs） | mean estimate | paired bootstrap 95% CI | 判定 |
|---|---:|---:|---|
| TPF ratio | 0.98048 | [0.94885, 1.01313] | 未来请求学习门失败 |
| decode TPS ratio | 0.98475 | [0.94301, 1.02419] | frozen serving 系统门失败 |
| acceptance-rate delta | -0.02233 | [-0.05629, 0.01197] | 跨 0 |
| serving seconds saved/request | -0.27819 | [-0.94055, 0.34756] | 平均更慢 |

选择门本身通过：snapshot 4 在 5 个 validation seeds 上的 mean TPF ratio 为 `1.02256`，高于固定
`1.002` threshold。但 test mean 变成 `0.98048`，validation-to-test optimism gap 为 **0.04208**。
所以 Stage 6C 总成功门失败，observed/instrumented 两种 break-even 都不存在。

## 1. 安全与可复算性

独立分析通过全部安全检查：

- 共检查 58 个 512-token 生成结果：4 个 training pairs、5 个 validation static、25 个 validation
  snapshot runs、10 个 test pairs；长度和数值均正常；
- checkpoint hash/revision、Uno clean/noise routing 均匹配；
- 39 份 persistent learner 隔离记录均为 trainable base tensors = 0、base optimizer overlap = 0、
  fast parameters = 526,336；
- 4 个 training requests 的 initial/final fast-weight L2 严格首尾相接；
- 所有 validation/test residual runs 的 feedback、buffer items、update attempts 均为 0；
- snapshot 0 在 5 个 validation seeds 上与 static 的 TPF ratio 精确为 1；
- 5×5 validation 网格、mean score 和 snapshot-4 选择可从 raw runs 完整重算；
- selected head 在整个 test 前后 SHA-256 均为
  `2658b259a63d4eefe473609addc0dba0a5dd91ddeeec65662a0537d5da4bc3e2`；
- method order、train/validation/test seed 分区和摊销数字均符合预注册协议。

因此 test 退化不是由模型偷偷继续训练、选错 checkpoint、zero 路径不同或 base 参数污染造成的。

## 2. Training 与 validation

training 流中 persistent TPF 四次都高于 paired static，但内部优化并不单调：

| request | initial L2 | final L2 | static TPF | persistent-train TPF |
|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 1.6668 | 1.2109 | 1.2586 |
| 1 | 1.6668 | 1.3871 | 1.2711 | 1.3036 |
| 2 | 1.3871 | 0.0000 | 1.2343 | 1.2778 |
| 3 | 0.0000 | 1.4736 | 1.2167 | 1.3036 |

17 次 update 中 1 次 rollback、3 次 same-buffer static reset；因此 snapshot 3 实际重新回到 zero。五个
validation score 为：

| snapshot | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| mean TPF ratio | 1.00000 | 1.00956 | 1.01241 | 1.00000 | **1.02256** |

但 validation 内部方差已经给出警告：snapshot 1 的范围是 `[0.95025, 1.09137]`，snapshot 2 是
`[0.90094, 1.12565]`，snapshot 4 是 `[0.96465, 1.04878]`。mean-only hard selection 没有约束下尾。

## 3. Test 尾部风险

10 个 test TPF ratios：

```text
1.0102, 0.9561, 0.8900, 1.0000, 1.0707,
0.9950, 1.0464, 0.9227, 0.9569, 0.9567
```

只有 3 wins、6 losses、1 tie；排除 tie 的 exact two-sided sign-test $p=0.50781$。TPS 是 4 wins、6
losses，$p=0.75391$。TPF 最好可达 +7.07%，最差却为 -11.00%；这说明 candidate 确实能改善部分
trajectory，但 full-weight activation 把错误选择的全部风险也带入 proposal。

median robustness 同样失败：TPF median ratio `0.97598 [0.93972, 1.02320]`，TPS median ratio
`0.97372 [0.93474, 1.05735]`。结果不是少数极端点单独造成的。

## 4. 摊销

4 个 training pairs 的 observed decode-time increment 为 `-2.752 s`，但显式 feedback/update/head 成本为
`+0.538 s`；负 observed increment 来自 base-forward/trajectory 时延波动，不能解释成训练免费。更关键的
是 frozen test 平均每请求不是节省、而是增加 `0.278 s`，分母收益为负，所以无论采用 observed 还是
instrumented training cost，break-even 都定义为不存在。

## 5. 为什么跨请求仍不够

Stage 6 排除了“单请求 horizon 太短”这一个因素，却暴露了另一个因素：同一 prompt 的不同 sampling seed
也会进入很不同的 continuation/context。rank-8 residual 在某些轨迹上学到的修正，可能在另一些轨迹上
反向改变 top-k/top-p support；一旦 hard activate，后续 proposal 又产生新的 policy-induced contexts。

这解释了三个表面矛盾同时成立：

1. training 四次 TPF 都提高；
2. validation mean 提高 2.26%；
3. held-out test mean 下降 1.95%。

它们分别测的是三组不同 trajectory，而不是同一分布表格中的独立、同方差标量。

## 6. 下一版：Static-anchored probability mixture

不再继续增加 validation seeds 或提高 selection threshold来补救 hard switch。下一版直接改变 proposal：

$$
q_w=(1-w)q_0+wq_\delta,\qquad 0\le w\le1,
$$

其中 $q_0$ 是 static Uno 经过相同 temperature/top-k/top-p 的分布，$q_\delta$ 是 residual candidate 的
filtered distribution。必须在**概率空间**合并两个 sparse support，而不是插值 logits 后再过滤。

对同一 context，TV 的凸性给出：

$$
D_{TV}(p,q_w)
\le(1-w)D_{TV}(p,q_0)+wD_{TV}(p,q_\delta).
$$

所以 candidate 很坏时，单 row 的最坏 TV 增量被 $w$ 线性限制；candidate 很好时仍可保留部分收益。多
token trajectory 会随 proposal 改变，因此这不是整条序列的风险证明，但比 full-weight hard switch 有明确
的局部保护。

mixture 采样后必须把完整 $q_w$ 保存给 verifier；这样 acceptance/rejection 仍严格 lossless。Stage 7 先
实现 sparse mixture、解析/Monte Carlo exactness 测试和 frozen persistent-head 支持，再用 validation 同时
约束 mean gain 与下尾风险。
