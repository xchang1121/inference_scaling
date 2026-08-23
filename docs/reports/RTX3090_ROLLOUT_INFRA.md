# RTX 3090 推理执行与 rollout 复用实验

本报告只保留执行实验结果与统计解释。各机制的数学定义、执行流程、实现和计算量分母见
[推理扩展算法：基础、原理与实现](../methods/ALGORITHMS.md)；准确率和 pass@k 见
[GSM8K 方法质量与计算量实验](GSM8K_3090_ALIGNED_RESULTS.md)。
实验臂设置见[执行标签定义](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels)。

成对因子统一定义为“优化路径 / 对照路径”。小于 1 表示相应指标下降，大于 1 表示相应指标上升。
墙钟排除模型与数据加载；主模型 FLOPs 按 `2 × 参数量 × 实际 forward token slots` 估算。完整网格使用
32 题 FP32；隔离执行变量的 BF16 诊断使用 3 个 seed。详细固定项与未计项见实验设计。
本报告中的正式执行结果均来自 Qwen2.5-1.5B；dLLM 路径未运行。Qwen2.5-0.5B 只作为草稿模型或
off-policy rollout proposal 使用，其计算量与 1.5B 分列。

## IS 与 MH rollout 复用结果

图中绿色表示指标下降，红色表示指标上升。

![IS 与 MH rollout 复用消融](../assets/rtx3090_is_mh_reuse.svg)

| 优化路径 | 对照路径 | 墙钟因子 | 主模型 FLOPs 因子 | 直接观测 |
| --- | --- | ---: | ---: | --- |
| [部分 rollout 续跑](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 丢弃部分 token 后从头生成 | 0.793 ± 0.080× | 3.346 ± 0.000× | 有效生成 token 因子 0.769×；保存 96 token |
| [流式 IS，便宜 verifier](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 完整 batch 结束后提交 verifier | 1.027 ± 0.040× | 1.000 ± 0.000× | verifier 队列接近零成本 |
| [流式 IS，0.2 s verifier](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 完整 batch 结束后提交 verifier | 0.671 ± 0.008× | 1.000 ± 0.000× | 首次估计更新时间因子 0.367 ± 0.002× |
| [确定性历史草稿](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通自回归解码 | 0.981 ± 0.064× | 1.036 ± 0.006× | 草稿接受率 17.7% ± 13.0% |
| [精确随机历史草稿](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通自回归解码 | 0.982 ± 0.075× | 1.033 ± 0.004× | 草稿接受率 24.0% ± 9.0% |
| [MH proposal-tree 预取，便宜奖励](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通 MH | 1.050 ± 0.040× | 1.424 ± 0.004× | 额外分支缺少可重叠延迟 |
| [MH proposal-tree 预取，0.2 s 奖励](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通 MH | 0.817 ± 0.016× | 1.267 ± 0.007× | 每步预取两个分支并消费一个分支 |
| [delayed acceptance，0.2 s 精确奖励](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通 MH | 0.827 ± 0.111× | 1.000 ± 0.000× | 精确奖励调用因子 0.556 ± 0.294× |
| [冻结 replay 混合 proposal，在线](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | base suffix proposal | 0.534 ± 0.078× | 1.003 ± 0.001× | 32 次更新中历史 proposal 占 35.4% ± 9.5% |
| [多尺度后缀](QWEN15B_OPTIMIZATION_STUDY.md#qwen15b-mh-stack) | uniform 后缀、base proposal | 0.716 ± 0.084× | 1.000 ± 0.000× | 接受率由 61.5% 升至 69.8% |
| [多尺度后缀 + 冻结 replay](QWEN15B_OPTIMIZATION_STUDY.md#qwen15b-mh-stack) | uniform 后缀、base proposal | 0.357 ± 0.026× | 1.002 ± 0.001× | 接受率 80.2%；历史 proposal 占 30.2% |
| [IS replay 候选缓存](QWEN15B_OPTIMIZATION_STUDY.md#qwen15b-is-stack) | 顺序 warm replay | 0.697 ± 0.009× | 0.807 ± 0.004× | 复用 cache build 已产生的 16 个候选 draw |
| [IS replay 候选缓存 + 连续批处理](QWEN15B_OPTIMIZATION_STUDY.md#qwen15b-is-stack) | 连续批处理 fresh-only | 0.754 ± 0.093× | 0.744 ± 0.004× | 在线总 FLOPs 因子 0.907×；复用率 31.62% |

<a id="infra-report-broker"></a>
### rollout token 续跑

对照路径与续跑路径均产生 8 条完整 rollout，共包含 320 个有效 completion token。首轮过量提交批次中的
6 条轨迹各生成 16 token。丢弃路径共生成 416 token；续跑路径保存其中 96 token，总生成量降至 320。

续跑路径恢复 6 个不同前缀，prefill token 数为对照的 `7.700×`，主模型 FLOPs 因子为 `3.346×`；
墙钟因子为 `0.793×`。当前实现减少了重复 decode token，但恢复前缀的额外 prefill 增加了逻辑计算量。

<a id="infra-report-streaming"></a>
### 流式 IS 的 verifier 重叠

实验固定 12 条 rollout 和 2 个 verifier worker，两条路径包含相同数量的完整 IS 贡献。便宜 verifier
对应 `1.027×` 墙钟因子。单条 verifier 延迟为 0.2 s 时，墙钟因子降至 `0.671×`，首次估计更新时间
因子降至 `0.367×`；主模型 workload 与 FLOPs 保持不变。收益来自 verifier 队列与剩余 decode 的重叠。

<a id="infra-report-speculation"></a>
### 历史草稿的接受率

随机历史草稿相对确定性草稿将平均接受率由 17.7% 提高到 24.0%。两种草稿的墙钟区间均覆盖 1，
主模型 FLOPs 增加约 3%–4%；验证与 CPU 分布处理成本高于该规模下的接受率收益。

### 0.5B 草稿模型的精确 speculative decoding

Qwen2.5-0.5B 逐段提出 token，Qwen2.5-1.5B 批量验证并按精确接受与残差抽样规则校正。该算法保持
1.5B 的采样分布；实验只比较执行成本。8 题、2 个 draw、BF16、最长 128 token 的汇总如下：

| 路径 | 墙钟因子 | 输出吞吐因子 | 1.5B FLOPs 因子 | 0.5B PFLOPs | 合计 FLOPs 因子 | 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通 1.5B 生成 | 1.000× | 1.000× | 1.000× | 0 | 1.000× | — |
| 草稿长度 2 | 1.058× | 0.952× | 1.055× | 0.004362 | 1.439× | 86.50% |
| 草稿长度 4 | 1.154× | 0.869× | 1.089× | 0.004230 | 1.462× | 81.14% |
| 草稿长度 8 | 1.272× | 0.805× | 1.170× | 0.004364 | 1.555× | 72.66% |

最短草稿仍增加墙钟、主模型 FLOPs 与合计 FLOPs。Qwen 0.5B 与 1.5B 的高 token 一致性没有抵消单卡双模型
驻留、草稿生成、验证和残差分布处理成本。该路径默认关闭；普通 1.5B 连续批处理作为默认 rollout 后端。

<a id="infra-report-mh-prefetch"></a>
### MH proposal-tree 预取

4 次 MH 更新消费 4 个 proposal；预取路径生成 7 个 proposal，其中 3 个对应未选分支。便宜奖励下，
墙钟与 FLOPs 同时增加。加入 0.2 s 奖励延迟后，墙钟因子为 `0.817×`，主模型 FLOPs 因子为
`1.267×`。该结果显示预取只在存在可重叠奖励延迟时缩短关键路径，并以额外 proposal 为代价。

有限状态后端的预取路径与普通 MH 在相同随机流下逐字段一致。BF16 真实模型中，双分支与单分支 batch
可能形成不同 token trace；本组数字表示相同更新预算下的吞吐比较。

<a id="infra-report-delayed-acceptance"></a>
### Delayed acceptance

本组 surrogate 使用与精确奖励相同的 token 谓词，但移除 0.2 s 延迟。三个 seed 的精确奖励调用因子
均值为 0.556，墙钟因子为 0.827，主模型 FLOPs 因子为 1。收益方差来自各随机流中的第一阶段拒绝比例。

<a id="infra-report-replay-mh"></a>
### 冻结 replay 混合 proposal

本组采用 30% base proposal 与 70% 冻结历史后缀的混合 proposal。4 条独立链共享一个冻结历史库，
共执行 32 次更新。历史后缀占 35.4%；在线墙钟因子为 `0.534×`，主模型 FLOPs 因子为 `1.003×`。
计入 8 条历史序列的 cache build 后，墙钟因子为 `0.586 ± 0.078×`，FLOPs 因子为
`1.070 ± 0.001×`。本组收益主要来自用并行评分替代历史命中时的串行自回归生成。

### 冻结 replay 与多尺度后缀的组合

组合实验固定 4 条链、每链 8 次更新、长度 32，并使用三个成对 seed。四个实验臂分别控制后缀分布和
proposal 来源：

| proposal | 后缀分布 | 在线墙钟（秒） | 在线主模型 PFLOPs | 接受率 | 历史 proposal 比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| base | uniform | 12.555 ± 0.610 | 0.016783 | 61.46% | 0% |
| base | multiscale | 9.012 ± 1.469 | 0.016783 | 69.79% | 0% |
| frozen replay mixture | uniform | 6.550 ± 2.064 | 0.016826 | 70.83% | 34.38% |
| frozen replay mixture | multiscale | 4.484 ± 0.463 | 0.016822 | 80.21% | 30.21% |

组合路径相对 `base + uniform` 的在线墙钟因子为 `0.357×`；相对 `base + multiscale` 为 `0.503×`；
相对 `replay + uniform` 为 `0.754×`。replay cache build 平均为 0.657 秒与 0.001136 PFLOPs。三个 seed
的首次查询墙钟均已覆盖 cache build，但在线 FLOPs 略高，故没有 FLOPs 摊销交点。部署条件为 prompt、模型
版本和采样策略匹配的冻结历史库；没有匹配历史时使用 multiscale base proposal。

## GSM8K 完整网格的执行结果

<a id="infra-report-batching"></a>
### 连续批处理

8 个 prompt worker 共享一张 RTX 3090；分母为同一方法的逐 prompt 同步路径。

| 方法 | 同步墙钟 | 连续批处理墙钟 | 墙钟因子 | FLOPs 因子 | 数值答案一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Base](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 95.4 s | 19.7 s | 0.206× | 1.177× | 32/32 |
| [自一致性投票-8](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 136.9 s | 67.9 s | 0.496× | 1.021× | 30/32 |
| [标准条件 IS](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 427.1 s | 370.0 s | 0.866× | 1.008× | 32/32 |
| [0.5B rollout proposal 条件 IS](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 419.5 s | 399.5 s | 0.952× | 1.003× | 32/32 |

Base 请求最容易形成密集 batch，墙钟收益最大。IS 包含多个依赖阶段，可跨 prompt 合并的串行段较少。
padding 和 batch 分叉略微增加逻辑 slots；连续批处理的主要收益是 GPU 利用率。自一致性投票-8 有两道题
出现数值答案分叉，该行只用于相同配置 workload 的吞吐比较。

<a id="infra-report-warm-replay"></a>
### Warm rollout replay

本组固定 8 个 base 候选、每候选 3 条总 rollout。warm 路径最多读取 2 条已评分历史记录，并保留
1 条 fresh base rollout。

| 路径 | 推理 PFLOPs | 墙钟 | 相对 fresh-only FLOPs | 相对 fresh-only 墙钟 |
| --- | ---: | ---: | ---: | ---: |
| [fresh-only](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 1.3483 | 422.5 s | 1.000× | 1.000× |
| [warm cache 在线阶段](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 1.0326 | 362.9 s | 0.766× | 0.859× |
| [cache build + 首次 warm 查询](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 3.1563 | 763.4 s | 2.341× | 1.807× |

在线阶段的 FLOPs 降低 23.4%，墙钟降低 14.1%。旧版报告曾把完整构建成本除以在线节省量，得到同一
replay key 重复 7 次的算术交点；该数字不属于默认算法结论。默认生命周期将 evaluation 记录消费一次，
不能为摊销构建成本而重复使用同一记录。warm 行表示已有、匹配且未消费的 history；需要为当前请求新建
history 时，应比较完整冷启动成本。准确率差及配对区间见
[质量报告](GSM8K_3090_ALIGNED_RESULTS.md#quality-replay-dynamic)。

<a id="infra-report-is-stack"></a>
### IS replay 候选缓存与连续批处理

4 道固定题、3 个 seed 的 FP32 组合消融将候选缓存与跨 prompt 批处理分别打开。候选缓存省略 replay key
构建后对同一 base 候选的重复自回归生成；缓存前后输出逐 token 一致。连续批处理再合并不同 prompt 的候选、
rollout 和评分请求；对应顺序/批处理输出也逐 token 一致。

| 对比 | 在线墙钟因子 | 在线 1.5B FLOPs 因子 | 在线总 FLOPs 因子 |
| --- | ---: | ---: | ---: |
| 候选缓存 / 顺序 warm replay | 0.697 ± 0.009× | 0.807 ± 0.004× | 0.836× |
| 连续批处理 / 顺序候选缓存 | 0.537 ± 0.092× | 1.094× | 1.094× |
| 完整栈 / 连续批处理 fresh-only | 0.754 ± 0.093× | 0.744× | 0.907× |

连续批处理的局部 FLOPs 增量来自 padding；候选缓存与 replay 节省抵消该增量后，完整在线栈的合计 FLOPs
仍下降 9.3%。如果本次请求需要新建 history，冷启动相对连续批处理 fresh-only 为 `1.361×` 墙钟和
`1.941×` 总 FLOPs。部署时仅对匹配且未消费的 history 启用 warm replay；其余请求使用连续批处理
fresh-only。
标准 `replay` 复现入口现已复用建库候选，并把 Qwen2.5-1.5B 与 Qwen2.5-0.5B 的在线及建库 FLOPs 分列。
`async` 组件验证跨 prompt 连续批处理；两项组合的完整因子以上表最后一行为准。

<a id="infra-report-dynamic"></a>
### 动态候选与方差—成本预算

三组均使用 8 个候选、48-token block、最长 192 token 和每个非终止候选 3 条 evaluation rollout。

| 路径 | 实际复用率 | 稳态 PFLOPs | 稳态墙钟 | 一次性总 PFLOPs | 一次性总墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [base 候选 + 固定 fresh](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 0% | 2.0973 | 705.7 s | 2.0973 | 705.7 s |
| [动态候选 + 固定 replay](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 34.943% | 1.9312 | 704.0 s | 4.8764 | 1,401.5 s |
| [动态候选 + 方差—成本分配](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 5.707% | 2.3172 | 706.6 s | 7.2951 | 2,186.2 s |

动态固定组相对 base 固定组的稳态 FLOPs 因子为 `0.921×`，墙钟因子为 `0.998×`。方差—成本组相对
动态固定组为 `1.200×` FLOPs 和 `1.004×` 墙钟。每来源 2 条 design rollout 形成较高的方差估计噪声，
最终复用率为 5.707%；相对动态固定组的 FLOPs 因子为 `1.200×`。

### GRPO 训练与免训练推理的累计计算量

本地 GRPO 一次训练消耗 15.646 PFLOPs、5,007,660 个前向等价 token slots 和 9,545.2 秒。仅按累计
FLOPs 计算时，verifier-MH、标准 verifier-IS 和 0.5B rollout proposal verifier-IS 与
“GRPO 训练 + 参数随机采样”的
交点分别为 392、344 和 230 次查询。这些方法与 GRPO 的准确率绝对差超过预设阈值，因此交点只表示
计算账本，不表示质量匹配所需查询数。

## 历史树、progressive 与 SMC 结果

下图分别列出 cache build、在线路径和后台 drain。

![RTX 3090 rollout 基础设施消融](../assets/rtx3090_rollout_infra.svg)

<a id="infra-report-token-tree"></a>
### 历史 token tree 与负载门控

请求依次使用 active batch 4、2、1，模拟 rollout 尾部逐渐变稀。

| 路径 | 在线墙钟（s） | 输出 token/s | 主模型 PFLOPs | cache build（s） | 草稿接受率 | 墙钟因子 | FLOPs 因子 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通自回归解码 | 3.838 ± 0.059 | 116.8 ± 1.8 | 0.00247 ± 0.00000 | 0.000 ± 0.000 | 0.0% | 1.000× | 1.000× |
| [历史树，始终草稿](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 8.297 ± 0.075 | 54.0 ± 0.5 | 0.00411 ± 0.00001 | 1.280 ± 0.027 | 13.1% | 2.162× | 1.660× |
| [历史树，负载感知](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 3.783 ± 0.067 | 118.5 ± 2.1 | 0.00249 ± 0.00001 | 1.335 ± 0.058 | 37.5% | 0.986× | 1.006× |

始终草稿路径验证了较多随后被拒绝的 token。batch 1 长尾门控将墙钟因子恢复至 `0.986×`，其三 seed
波动范围覆盖 1；该门控在本组中的作用是限制低接受率草稿造成的吞吐退化。

<a id="infra-report-progressive-smc"></a>
### Progressive、run-ahead 与 SMC rollout forest

算法层使用 3 个候选、每候选 2 条总 rollout；SMC 使用 3 个粒子、每粒子 2 个分支。

| 路径 | 在线墙钟（s） | 在线主模型 PFLOPs | cache build（s） | 后台 drain（s） | fresh / reused rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| [固定 rollout 条件 IS](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 3.442 ± 0.162 | 0.00809 ± 0.00067 | 1.248 ± 0.057 | 0.000 ± 0.000 | 0.0 / 0.0 |
| [pilot/evaluation 分离](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 5.313 ± 0.231 | 0.01250 ± 0.00000 | 1.269 ± 0.100 | 0.000 ± 0.000 | 0.0 / 0.0 |
| [流式奖励 + run-ahead](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 5.605 ± 0.958 | 0.01261 ± 0.00192 | 1.256 ± 0.068 | 7.763 ± 1.927 | 0.0 / 0.0 |
| [SMC forest，fresh-only](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 2.985 ± 0.260 | 0.01271 ± 0.00038 | 1.250 ± 0.080 | 0.000 ± 0.000 | 35.3 / 0.0 |
| [SMC forest，条件后缀复用](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 2.571 ± 0.510 | 0.01226 ± 0.00206 | 1.263 ± 0.042 | 0.000 ± 0.000 | 24.7 / 13.3 |

pilot/evaluation 分离相对固定 rollout 的墙钟因子为 `1.544×`，FLOPs 因子为 `1.553×`。候选成本接近
同质时，额外 pilot 增加在线开销。run-ahead 组只使用低成本数值解析，在线墙钟为 progressive 组的
`1.051×`，并产生 `7.763 ± 1.927 s` 后台 drain；受控 0.2 s verifier 实验中的流式执行则达到
`0.671×` 墙钟因子。

SMC 条件后缀复用相对 fresh-only SMC 的墙钟因子为 `0.856×`，FLOPs 因子为 `0.963×`，fresh rollout
均值由 35.3 降至 24.7。

## 结果对应的适用范围

| 工作负载 | 当前结果支持的配置 | 需要同时记录的成本 |
| --- | --- | --- |
| 大量独立请求 | 连续批处理、重复前缀 KV 复用 | batch 形状、padding slots 与数值一致性 |
| 存在匹配且未消费的 replay 记录 | warm replay、候选缓存、连续批处理 | history 来源、cache build、1.5B/0.5B 分账 |
| 过量提交产生未完成 rollout | 部分 rollout broker | 保存 token、恢复 prefill 和完整轨迹数 |
| verifier 含 CPU 或远程延迟 | 流式 IS、delayed acceptance；高延迟下的 MH 预取 | verifier 延迟、精确调用数和未选分支 |
| 同一 prompt 存在冻结历史后缀 | multiscale replay-mixture MH | 历史 proposal 比例、接受率、评分长度和 cache build |
| 逐 block 粒子推理 | SMC 条件后缀 forest | fresh/reused rollout 与 ESS |
| 候选成本差异较大 | pilot/evaluation 分离和方差—成本分配 | design 样本量与 evaluation ESS |

## 结果总结

1. warm replay 与 SMC 条件后缀 forest 减少 fresh rollout；完整成本包含历史库构建和条件匹配。
2. 流式 IS、MH 预取和 delayed acceptance 可缩短 verifier 关键路径；收益随 verifier 延迟和拒绝比例变化。
3. 多尺度后缀与冻结 replay 在 Qwen 1.5B MH 中可以叠加，组合在线墙钟因子为 `0.357×`；FLOPs 因子为
   `1.002×`，其收益属于执行时间而非主模型计算量下降。
4. Qwen 0.5B 精确 speculative decoding、历史草稿和方差—成本分配未降低当前单卡设置的墙钟；这些路径
   保留为显式实验配置。
5. IS 候选缓存与连续批处理在 warm 在线阶段可叠加；相对连续批处理 fresh-only 的墙钟、1.5B FLOPs 和
   总 FLOPs 因子分别为 `0.754×`、`0.744×` 和 `0.907×`。新建 history 的冷启动不启用该路径。

<a id="infra-report-vllm"></a>
## 后端范围与机器可读结果

本报告的 RTX 3090 数值对应 Transformers 后端。vLLM 的实现、能力边界和成对复现入口见
[vLLM 后端](../methods/ALGORITHMS.md#infra-vllm)。

机器可读结果包括：

- [`results/gsm8k_3090/`](../../results/gsm8k_3090/) 中的 `compute`、`replay`、`dynamic_is` 和
  `async_grouped` 汇总；
- [`rtx3090_transformers_summary.json`](../../results/infra/rtx3090_transformers_summary.json)
  及其六份 rollout seed 文件；
- [`rtx3090_transformers_is_mh_summary.json`](../../results/infra/rtx3090_transformers_is_mh_summary.json)
  及三份 IS/MH seed 文件；
- [`mh_replay_multiscale_stack.json`](../../results/arllm/qwen15b_optimization/mh_replay_multiscale_stack.json)
  、[`is_replay_batching_stack.json`](../../results/arllm/qwen15b_optimization/is_replay_batching_stack.json)
  与 [`draft_model_speculation_screen.json`](../../results/arllm/qwen15b_optimization/draft_model_speculation_screen.json)。
