# Stage 3 结果：Online Uno 非平稳仿真

## 结论先行

本阶段在 exact $\Psi$-Spec、20 个随机 seed、每条路径 12,000 tokens 的非平稳 Markov workload 上，
把“在线学习有效”和“在线系统更快”分开检验。预注册主策略 `stride10_discounted` 的三个门均通过：

- **学习成功**：current/static TV-regret ratio 中位数为 0.9098，95% bootstrap CI
  $[0.9083,0.9106]$，即 verifier-feedback 使 proposal mismatch 降低约 9.0%；
- **算法成功**：TPF 从 static 的 1.4519 提高到 1.6959，相对比 1.1688，95% CI
  $[1.1635,1.1756]$；
- **合成成本成功**：在预注册 update-cost proxy 下，tokens/cost 相对比 1.1419，95% CI
  $[1.1367,1.1485]$。

第三条不是 GPU 加速结论。真实 backward 没有在此阶段计时，结果 JSON 明确保持
`real_gpu_online_speedup_tested=false`。准确表述是：

> Post-verification online distillation 在可控 distribution drift 下既降低 proposal regret，也提高
> exact speculative decoding 的推进量；在一组透明、可做敏感性分析的合成更新成本下存在净收益空间。
> 真实 Uno adapter 能否在 RTX 3090 上回本，仍须 Stage 4/5 实测。

## 正确性与完整性

正式运行包含 8 个策略 $\times$ 20 seeds = 160 条路径；每条恰好生成 12,000 个 target tokens，所有
summary 数值 finite。实现有三层防护：

1. `uno_linear_step` 在进入 verifier 前复制实际采样分布 $q_t$；
2. replay/update 只在 acceptance、residual correction 和 lookahead 全部结束后执行；
3. 单元测试在 update 后逐元素检查当前轮保存的 denominator 未改变。

Stage 1 已经对 adaptive proposal 做完整短序列分布枚举和 $10^5$ Monte Carlo 检验。本阶段额外比较 target
NLL：从 schedule 精确传播状态分布得到理论期望 1.22770，主策略 20 条路径的平均值为 1.22763；八种
方法的 NLL 中位数跨度仅 0.0077。有限样本 NLL 不是 lossless 的证明，但没有出现 proposer 改变 target
质量的诊断信号。

## 主结果

所有区间均为对 20 个 seed 的配对比值中位数做 30,000 次 percentile bootstrap。`static` 的
spec acceptance 中位数为 0.4764。

| 方法 | TV regret / static | TPF / static | cost efficiency / static | acceptance | update cost fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| per-round full | 0.8999 | 1.1793 | 1.0031 | 0.5900 | 14.94% |
| stride-5 full | 0.9021 | 1.1730 | 1.1274 | 0.5858 | 3.89% |
| stride-10 full | 0.9021 | 1.1701 | 1.1431 | 0.5849 | 2.31% |
| stride-20 full | 0.9059 | 1.1624 | 1.1451 | 0.5808 | 1.49% |
| stride-10 on-policy | 0.9111 | 1.1745 | **1.1524** | 0.5889 | 1.88% |
| **stride-10 discounted（主检验）** | **0.9098** | **1.1688** | **1.1419** | **0.5843** | **2.31%** |
| adaptive discounted | 0.9078 | 1.1739 | 1.0621 | 0.5857 | 9.48% |

主策略的完整区间为：

| 指标 | estimate | 95% CI |
| --- | ---: | ---: |
| absolute TPF | 1.6959 | [1.6882, 1.7060] |
| TV regret / static | 0.9098 | [0.9083, 0.9106] |
| TPF / static | 1.1688 | [1.1635, 1.1756] |
| proxy efficiency / static | 1.1419 | [1.1367, 1.1485] |

主策略每条路径 update 中位数为 353，update cost 占 2.31%。在保持其他成本不变时，预注册 update cost
可以放大到中央假设的 7.15 倍才到 break-even；在 $0,0.5,1,2,4$ 倍成本下，效率比分别为
1.1688、1.1552、1.1419、1.1162、1.0680。这说明仿真结论不是恰好依赖一个极窄的成本点，但仍不能把
合成 multiplier 换写成毫秒或 GPU tokens/s。

## 最重要的两个反例

### 1. 每轮更新几乎把算法收益全部吃掉

`per_round_full` 的 TV regret 最低、TPF 最高之一，但 3,506.5 次 median update 让成本占比达到
14.94%，最终效率只比 static 高 0.31%，break-even multiplier 仅 1.02。这直接验证了净收益不等式：

$$
\frac{\tau_1}{\tau_0}>
1+\frac{C_U}{S(C_D+C_V)}.
$$

只报告 acceptance 或 TPF 会把这个方案误判为最好；在线 Uno 的关键不是“能不能学”，而是“多久学一次”。

### 2. 当前 adaptive controller 过度更新

adaptive 策略能把 TPF 提高 17.39%，但效率只提高 6.21%，显著落后固定 stride。代表 seed 的轨迹是：

- in-domain 初期从 stride 10 退避到 20；
- abrupt shift 后依次变成 20、10、5、1；
- 在高 proposal gain 区间长时间保持每轮更新；
- return in-domain 后才逐步退避，最终回到 20。

该 controller 判断“online proposal 是否胜过 static”，却没有直接估计“下一次 update 是否比继续使用当前
fast weights 更值”。一旦已经学好，它仍把已有收益误归因于必须继续高频更新。这是策略定义的问题，不是
随机噪声：20 条 adaptive 路径的 update 中位数为 2,056，成本占 9.48%。下一版 controller 必须把
**fast-weight value** 和 **marginal update value** 分开估计。

## Regime 分解揭示了遗忘问题

主策略相对 static 的 segment efficiency：

| segment | efficiency ratio |
| --- | ---: |
| offline in-domain | 0.9783 |
| abrupt shift A | 1.2391 |
| gradual A→B | 1.1395 |
| shift B | 1.3268 |
| return in-domain | 0.8984 |

在线更新在两个真正的 shift 区间收益很大，却在初始 in-domain 仅产生开销；更重要的是，回到原分布后
stale fast weights 使效率低于从未更新的 static Uno。最终 fast-weight L2 仍约 47.7，说明单组累积
logit correction 没有及时恢复 offline prior。这给真实 Online Uno 一个清晰的设计约束：

$$
\boxed{\text{在线适配必须同时有 change detection、回退快照和可逆/衰减的 fast weights。}}
$$

仅调整 stride 不够，因为 stride 只能控制以后学多快，不能立即撤销已经过时的参数。

## Supervision 消融

`stride10_on_policy` 是同一批数据中的探索性最好方案，效率比 1.1524；它虽然 TV-regret 改善小于 full，
但跳过首次 rejection 后的 hypothetical tail，item 成本更低。预注册主策略 discounted-tail 为 1.1419，
full 为 1.1431。因 on-policy 是在同一 seeds 上被选出的最好者，没有独立确认集，也没有做在线策略之间的
多重比较校正，当前只能说：

- “discounted tail 必然优于 full canvas”的 H3 **没有得到支持**；
- on-policy masking 是 Stage 4 值得优先实测的低成本候选；
- 真模型上应按保存 logits、top-$K$ projection 和 backward 的实际时间比较，而不是沿用 tabular item cost。

## 从结果推导的 Online Uno v1 设计

下一阶段不直接在线修改完整 rank-128 adapter，而采用 frozen slow weights 加 request-local fast weights：

$$
q_t=q_{\theta_{\mathrm{AR}}+\phi_{\mathrm{Uno}}+\delta_t},
\qquad
\theta_{\mathrm{AR}},\phi_{\mathrm{Uno}}\text{ 永久冻结}.
$$

最小安全版本包含：

1. 每轮保存旧 filtered $q_t$，verification 后才更新；
2. 先实现 vocabulary logit residual / affine head，再实现顶部层 rank-4 LoRA；
3. 默认 stride 10、on-policy feedback、小 replay batch；
4. update 前保存 fast-weight snapshot，validation proxy 恶化即 rollback；
5. 对 $\delta$ 加 decay/elastic pull，change detector 检测 return shift 时可直接清零或恢复 offline snapshot；
6. controller 比较候选动作 `no-update` 与 `update` 的边际收益，包含 cooldown、minimum dwell 和 update
   budget，不再把当前 online-static gap 当成下一次 update 的收益；
7. 分别计时 draft、verify、feedback materialization、backward、optimizer、同步和 peak VRAM。

机器可读正式结果位于 `results/stage3_online_markov.json`。它保留全部 160 条 summary、所有 seed 的
segment 指标，以及一个代表 seed 的 250-token trace/controller events；不包含模型权重或逐 token 文本。

## 结论边界

- 这是 vocabulary 8 的 tabular fast weights；优化难度远低于 64K vocabulary 的神经网络 adapter；
- forward 成本锚定 Stage 2 实测，update 成本仍是假设；
- seed 配对可复现，但不同 proposer 消耗不同随机数，不是逐 token 相同 target trajectory；
- `adaptive_discounted` 的失败只否定当前 controller，不否定自适应策略本身；
- 本阶段通过的是学习、算法和合成成本三层门，不是 RTX 3090 上的 online wall-clock 门。
