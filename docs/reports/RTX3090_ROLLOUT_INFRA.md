# RTX 3090 推理执行与 rollout 复用实验

本报告只保留执行实验结果与统计解释。各机制的数学定义、执行流程、实现、计算量统计方法和比较基准见
[推理扩展算法：基础、原理与实现](../methods/ALGORITHMS.md)；准确率和 pass@k 见
[GSM8K 方法质量与计算量实验](GSM8K_3090_ALIGNED_RESULTS.md)。
实验组设置见[执行标签定义](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels)。

成对因子统一定义为“优化路径 / 对照路径”。小于 1 表示相应指标下降，大于 1 表示相应指标上升。
墙钟排除模型与数据加载；主模型 FLOPs 按 `2 × 参数量 × 实际参与前向计算的 token 位置数` 估算。32 题完整实验使用
FP32；隔离执行变量的 BF16 诊断使用 3 个随机种子。详细固定项与未计项见实验设计。
本报告中的正式执行结果均来自 Qwen2.5-1.5B；dLLM 路径未运行。Qwen2.5-0.5B 只作为草稿模型或
off-policy rollout proposal 使用，其计算量与 1.5B 分列。
ESS 表示有效样本量（effective sample size）。

## IS 与 MH rollout 复用结果

图中绿色表示指标下降，红色表示指标上升。

![IS 与 MH rollout 复用消融](../assets/rtx3090_is_mh_reuse.svg)

| 优化路径 | 对照路径 | 墙钟因子 | 主模型 FLOPs 因子 | 直接观测 |
| --- | --- | ---: | ---: | --- |
| [部分 rollout 续跑](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 丢弃部分 token 后从头生成 | 0.793 ± 0.080× | 3.346 ± 0.000× | 有效生成 token 因子 0.769×；保存 96 token |
| [流式 IS，近零延迟 verifier](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 整批生成结束后提交 verifier | 1.027 ± 0.040× | 1.000 ± 0.000× | verifier 不包含额外等待时间 |
| [流式 IS，0.2 s verifier](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 整批生成结束后提交 verifier | 0.671 ± 0.008× | 1.000 ± 0.000× | 首次估计更新时间因子 0.367 ± 0.002× |
| [确定性历史草稿](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通自回归解码 | 0.981 ± 0.064× | 1.036 ± 0.006× | 草稿接受率 17.7% ± 13.0% |
| [精确随机历史草稿](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通自回归解码 | 0.982 ± 0.075× | 1.033 ± 0.004× | 草稿接受率 24.0% ± 9.0% |
| [MH 候选分支预取，近零延迟奖励](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通 MH | 1.050 ± 0.040× | 1.424 ± 0.004× | 没有可与额外分支生成重叠的等待时间 |
| [MH 候选分支预取，0.2 s 奖励](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通 MH | 0.817 ± 0.016× | 1.267 ± 0.007× | 每步预取两个分支，随后使用其中一个 |
| [两阶段延迟接受，0.2 s 精确奖励](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 普通 MH | 0.827 ± 0.111× | 1.000 ± 0.000× | 精确奖励调用因子 0.556 ± 0.294× |
| [冻结历史混合 proposal，在线阶段](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 基础模型后缀 proposal | 0.534 ± 0.078× | 1.003 ± 0.001× | 32 次更新中历史 proposal 占 35.4% ± 9.5% |
| [多尺度后缀](QWEN15B_OPTIMIZATION_STUDY.md#qwen15b-mh-stack) | 均匀后缀、基础模型 proposal | 0.716 ± 0.084× | 1.000 ± 0.000× | 接受率由 61.5% 升至 69.8% |
| [多尺度后缀 + 冻结历史](QWEN15B_OPTIMIZATION_STUDY.md#qwen15b-mh-stack) | 均匀后缀、基础模型 proposal | 0.357 ± 0.026× | 1.002 ± 0.001× | 接受率 80.2%；历史 proposal 占 30.2% |
| [IS replay 候选缓存](QWEN15B_OPTIMIZATION_STUDY.md#qwen15b-is-stack) | 顺序执行已有历史 replay | 0.697 ± 0.009× | 0.807 ± 0.004× | 复用缓存构建时已产生的 16 组候选 |
| [IS replay 候选缓存 + 连续批处理](QWEN15B_OPTIMIZATION_STUDY.md#qwen15b-is-stack) | 连续批处理纯新生成 | 0.754 ± 0.093× | 0.744 ± 0.004× | 在线总 FLOPs 因子 0.907×；复用率 31.62% |

<a id="infra-report-broker"></a>
### rollout token 续跑

对照路径与续跑路径均产生 8 条完整 rollout，共包含 320 个有效补全 token。首轮过量提交批次中的
6 条轨迹各生成 16 token。丢弃路径共生成 416 token；续跑路径保存其中 96 token，总生成量降至 320。

续跑路径恢复 6 个不同前缀，前缀预填充 token 数为对照的 `7.700×`，主模型 FLOPs 因子为 `3.346×`；
墙钟因子为 `0.793×`。当前实现减少了重复生成的 token，但恢复前缀时额外执行的预填充增加了计算量。

<a id="infra-report-streaming"></a>
### 流式 IS 的 verifier 重叠

实验固定 12 条 rollout 和 2 个 verifier 工作线程，两条路径包含相同数量的完整 IS 贡献。近零延迟 verifier
对应 `1.027×` 墙钟因子。单条 verifier 延迟为 0.2 s 时，墙钟因子降至 `0.671×`，首次估计更新时间
因子降至 `0.367×`；主模型请求量与 FLOPs 保持不变。收益来自 verifier 队列处理与剩余生成的并行执行。

<a id="infra-report-speculation"></a>
### 历史草稿的接受率

随机历史草稿相对确定性草稿将平均接受率由 17.7% 提高到 24.0%。两种草稿的墙钟区间均覆盖 1，
主模型 FLOPs 增加约 3%–4%；验证与 CPU 分布处理成本高于该规模下的接受率收益。

### 0.5B 草稿模型的精确推测解码

Qwen2.5-0.5B 逐段提出 token，Qwen2.5-1.5B 批量验证并按精确接受与残差抽样规则校正。该算法保持
1.5B 的采样分布；实验只比较执行成本。8 题、2 次独立重复、BF16、最长 128 token 的汇总如下：

| 路径 | 墙钟因子 | 输出吞吐因子 | 1.5B FLOPs 因子 | 0.5B PFLOPs | 合计 FLOPs 因子 | 接受率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通 1.5B 生成 | 1.000× | 1.000× | 1.000× | 0 | 1.000× | — |
| 草稿长度 2 | 1.058× | 0.952× | 1.055× | 0.004362 | 1.439× | 86.50% |
| 草稿长度 4 | 1.154× | 0.869× | 1.089× | 0.004230 | 1.462× | 81.14% |
| 草稿长度 8 | 1.272× | 0.805× | 1.170× | 0.004364 | 1.555× | 72.66% |

最短草稿仍增加墙钟、主模型 FLOPs 与合计 FLOPs。Qwen 0.5B 与 1.5B 较高的 token 预测一致率没有抵消单卡双模型
同时占用显存、草稿生成、验证和残差分布处理成本。该路径默认关闭；普通 1.5B 连续批处理作为默认 rollout 后端。

<a id="infra-report-mh-prefetch"></a>
### MH 候选分支预取

4 次 MH 更新使用 4 个 proposal；预取路径生成 7 个 proposal，其中 3 个对应未选分支。近零延迟奖励下，
墙钟与 FLOPs 同时增加。加入 0.2 s 奖励延迟后，墙钟因子为 `0.817×`，主模型 FLOPs 因子为
`1.267×`。存在可与候选生成并行执行的奖励等待时，预取缩短从更新开始到奖励计算完成的时间，但会生成额外
proposal。

有限状态后端的预取路径与普通 MH 在相同随机数序列下逐字段一致。BF16 真实模型中，双分支与单分支批量
可能生成不同 token 序列；本组数字表示相同更新预算下的吞吐比较。

<a id="infra-report-delayed-acceptance"></a>
### 两阶段延迟接受

本组近似奖励使用与精确奖励相同的 token 判定条件，但移除 0.2 s 延迟。三个随机种子的精确奖励调用因子
均值为 0.556，墙钟因子为 0.827，主模型 FLOPs 因子为 1。结果差异来自各随机数序列中的第一阶段拒绝比例。

<a id="infra-report-replay-mh"></a>
### 冻结历史混合 proposal

本组采用 30% 基础模型 proposal 与 70% 冻结历史后缀的混合 proposal。4 条独立链共享一个冻结历史库，
共执行 32 次更新。历史后缀占 35.4%；在线墙钟因子为 `0.534×`，主模型 FLOPs 因子为 `1.003×`。
计入 8 条历史序列的缓存构建后，墙钟因子为 `0.586 ± 0.078×`，FLOPs 因子为
`1.070 ± 0.001×`。本组收益主要来自用并行评分替代历史命中时的串行自回归生成。

### 冻结历史与多尺度后缀的组合

组合实验固定 4 条链、每链 8 次更新、长度 32，并使用三个随机种子；每组配置用相同的随机种子配对比较。
四个实验组分别控制后缀分布和 proposal 来源：

| proposal | 后缀分布 | 在线墙钟（秒） | 在线主模型 PFLOPs | 接受率 | 历史 proposal 比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| 基础模型 | 均匀 | 12.555 ± 0.610 | 0.016783 | 61.46% | 0% |
| 基础模型 | 多尺度 | 9.012 ± 1.469 | 0.016783 | 69.79% | 0% |
| 冻结历史混合分布 | 均匀 | 6.550 ± 2.064 | 0.016826 | 70.83% | 34.38% |
| 冻结历史混合分布 | 多尺度 | 4.484 ± 0.463 | 0.016822 | 80.21% | 30.21% |

组合路径相对“基础模型 + 均匀后缀”的在线墙钟因子为 `0.357×`；相对“基础模型 + 多尺度后缀”为
`0.503×`；相对“冻结历史 + 均匀后缀”为 `0.754×`。replay 缓存构建平均消耗 0.657 秒和
0.001136 PFLOPs。三个随机种子下，首次查询节省的墙钟均超过缓存构建时间；在线 FLOPs 略高，因此累计
FLOPs 不会随查询次数增加而低于对照。部署要求提示、模型版本和采样策略均与冻结历史库匹配；没有匹配历史
时使用多尺度基础模型 proposal。

## GSM8K 32 题全部方法的执行结果

<a id="infra-report-batching"></a>
### 连续批处理

8 个并发提示任务共享一张 RTX 3090；比较基准为同一方法逐条处理提示的同步路径。

| 方法 | 同步墙钟 | 连续批处理墙钟 | 墙钟因子 | FLOPs 因子 | 数值答案一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Base](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 95.4 s | 19.7 s | 0.206× | 1.177× | 32/32 |
| [自一致性投票-8](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 136.9 s | 67.9 s | 0.496× | 1.021× | 30/32 |
| [标准条件 IS](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 427.1 s | 370.0 s | 0.866× | 1.008× | 32/32 |
| [0.5B rollout proposal 条件 IS](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 419.5 s | 399.5 s | 0.952× | 1.003× | 32/32 |

Base 请求的执行阶段相同，最容易合并成较大的批量，因此墙钟收益最大。IS 包含多个依赖阶段，可跨提示
合并的串行段较少。
填充和批量执行路径差异略微增加参与前向计算的 token 位置数；连续批处理的主要收益是提高 GPU 利用率。自一致性投票-8 有两道题
出现数值结果差异，该行只用于相同请求配置下的吞吐比较。

<a id="infra-report-warm-replay"></a>
### 已有历史 rollout replay

本组固定 8 个基础模型候选、每候选 3 条总 rollout。已有历史路径最多读取 2 条已评分历史记录，并保留
1 条新生成的基础模型 rollout。

| 路径 | 推理 PFLOPs | 墙钟 | 相对纯新生成的 FLOPs | 相对纯新生成的墙钟 |
| --- | ---: | ---: | ---: | ---: |
| [纯新生成](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 1.3483 | 422.5 s | 1.000× | 1.000× |
| [已有缓存的在线阶段](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 1.0326 | 362.9 s | 0.766× | 0.859× |
| [缓存构建 + 首次查询](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 3.1563 | 763.4 s | 2.341× | 1.807× |

在线阶段的 FLOPs 降低 23.4%，墙钟降低 14.1%。默认记录管理规则中，每条最终估计记录使用一次后即标记为
已使用，缓存构建成本不通过重复使用同一记录分摊。已有缓存行表示历史记录已经存在、条件匹配且尚未使用；
需要为当前请求新建历史记录时，应比较包含构建阶段的完整成本。准确率差及配对区间见
[质量报告](GSM8K_3090_ALIGNED_RESULTS.md#quality-replay-dynamic)。

<a id="infra-report-is-stack"></a>
### IS replay 候选缓存与连续批处理

4 道固定题、3 个随机种子的 FP32 组合消融分别启用候选缓存与跨提示批处理。候选缓存省略匹配键
构建后对同一基础模型候选的重复自回归生成；缓存前后输出逐 token 一致。连续批处理再合并不同提示的候选、
rollout 和评分请求；对应顺序/批处理输出也逐 token 一致。

| 对比 | 在线墙钟因子 | 在线 1.5B FLOPs 因子 | 在线总 FLOPs 因子 |
| --- | ---: | ---: | ---: |
| 候选缓存 / 顺序执行已有历史 replay | 0.697 ± 0.009× | 0.807 ± 0.004× | 0.836× |
| 连续批处理 / 顺序候选缓存 | 0.537 ± 0.092× | 1.094× | 1.094× |
| 候选缓存与连续批处理的组合 / 连续批处理纯新生成 | 0.754 ± 0.093× | 0.744× | 0.907× |

连续批处理的局部 FLOPs 增量来自填充；候选缓存与 replay 节省抵消该增量后，组合路径的合计 FLOPs
仍下降 9.3%。如果本次请求需要新建历史记录，包含构建阶段的路径相对连续批处理纯新生成路径为 `1.361×`
墙钟和 `1.941×` 总 FLOPs。部署时仅对匹配且尚未使用的历史记录启用 replay；其余请求使用连续批处理纯新
生成路径。
标准 `replay` 复现入口现已复用建库候选，并把 Qwen2.5-1.5B 与 Qwen2.5-0.5B 的在线及建库 FLOPs 分列。
`async` 组件验证跨提示连续批处理；两项组合的结果以上表最后一行为准。

<a id="infra-report-dynamic"></a>
### 动态候选与方差—成本预算

三组均使用 8 个候选、48-token 生成块、最长 192 token 和每个非终止候选 3 条最终估计 rollout。

| 路径 | 实际复用率 | 在线阶段 PFLOPs | 在线阶段墙钟 | 构建加首次查询 PFLOPs | 构建加首次查询墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [基础模型候选 + 固定新样本](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 0% | 2.0973 | 705.7 s | 2.0973 | 705.7 s |
| [动态候选 + 固定 replay](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 34.943% | 1.9312 | 704.0 s | 4.8764 | 1,401.5 s |
| [动态候选 + 方差—成本分配](../experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels) | 5.707% | 2.3172 | 706.6 s | 7.2951 | 2,186.2 s |

动态固定组相对基础模型固定组的在线 FLOPs 因子为 `0.921×`，墙钟因子为 `0.998×`。方差—成本组相对
动态固定组为 `1.200×` FLOPs 和 `1.004×` 墙钟。每个来源只有 2 条设计阶段 rollout，方差估计误差较大，
最终复用率为 5.707%；相对动态固定组的 FLOPs 因子为 `1.200×`。

### GRPO 训练与免训练推理的累计计算量

本地 GRPO 一次训练消耗 15.646 PFLOPs、5,007,660 个前向等价 token 位置和 9,545.2 秒。仅按累计 FLOPs
计算时，verifier-MH、标准 verifier-IS 和 0.5B rollout proposal verifier-IS 与“GRPO 训练 + 参数随机采样”
累计 FLOPs 相等时的查询次数分别为 392、344 和 230。这些方法与 GRPO 的准确率绝对差超过预设阈值，
因此这些查询次数只比较累计计算量，不表示达到相同准确率所需的查询数。

## 历史树、分阶段估计与 SMC 结果

下图分别列出缓存构建、在线路径和等待后台队列完成的时间。

![RTX 3090 rollout 基础设施消融](../assets/rtx3090_rollout_infra.svg)

<a id="infra-report-token-tree"></a>
### 历史 token 树的启用条件

请求依次使用当前批量大小 4、2、1，模拟 rollout 接近结束时可同时处理的序列逐渐减少。

| 路径 | 在线墙钟（s） | 输出 token/s | 主模型 PFLOPs | 缓存构建（s） | 草稿接受率 | 墙钟因子 | FLOPs 因子 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通自回归解码 | 3.838 ± 0.059 | 116.8 ± 1.8 | 0.00247 ± 0.00000 | 0.000 ± 0.000 | 0.0% | 1.000× | 1.000× |
| [历史树，始终草稿](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 8.297 ± 0.075 | 54.0 ± 0.5 | 0.00411 ± 0.00001 | 1.280 ± 0.027 | 13.1% | 2.162× | 1.660× |
| [历史树，仅在批量大小为 1 时启用](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 3.783 ± 0.067 | 118.5 ± 2.1 | 0.00249 ± 0.00001 | 1.335 ± 0.058 | 37.5% | 0.986× | 1.006× |

始终使用历史树的路径验证了较多随后被拒绝的 token。只在批量大小为 1 时启用历史树后，墙钟因子恢复至
`0.986×`；三个随机种子的波动范围覆盖 1。这一启用条件限制了低接受率草稿造成的吞吐下降。

<a id="infra-report-progressive-smc"></a>
### 初始估计与最终估计分离、空闲时预生成和 SMC 多树搜索

算法层使用 3 个候选、每候选 2 条总 rollout；SMC 使用 3 个粒子、每粒子 2 个分支。

| 路径 | 在线墙钟（s） | 在线主模型 PFLOPs | 缓存构建（s） | 后台队列清空（s） | 新生成 / 复用 rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| [固定 rollout 条件 IS](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 3.442 ± 0.162 | 0.00809 ± 0.00067 | 1.248 ± 0.057 | 0.000 ± 0.000 | 0.0 / 0.0 |
| [初始估计与最终估计分离](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 5.313 ± 0.231 | 0.01250 ± 0.00000 | 1.269 ± 0.100 | 0.000 ± 0.000 | 0.0 / 0.0 |
| [流式奖励 + 空闲时预生成](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 5.605 ± 0.958 | 0.01261 ± 0.00192 | 1.256 ± 0.068 | 7.763 ± 1.927 | 0.0 / 0.0 |
| [SMC 多树搜索，纯新生成](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 2.985 ± 0.260 | 0.01271 ± 0.00038 | 1.250 ± 0.080 | 0.000 ± 0.000 | 35.3 / 0.0 |
| [SMC 多树搜索，条件后缀复用](../experiments/GSM8K_EXPERIMENT_DESIGN.md#infra-labels) | 2.571 ± 0.510 | 0.01226 ± 0.00206 | 1.263 ± 0.042 | 0.000 ± 0.000 | 24.7 / 13.3 |

初始估计与最终估计分离相对固定 rollout 的墙钟因子为 `1.544×`，FLOPs 因子为 `1.553×`。各候选成本
接近时，额外的初始估计增加在线开销。空闲时预生成组只使用低成本数值解析，在线墙钟为分阶段估计组的
`1.051×`，并需要 `7.763 ± 1.927 s` 清空后台队列；受控 0.2 s verifier 实验中的流式执行则达到
`0.671×` 墙钟因子。

SMC 条件后缀复用相对纯新生成 SMC 的墙钟因子为 `0.856×`，FLOPs 因子为 `0.963×`，新生成 rollout
均值由 35.3 降至 24.7。

## 结果对应的适用范围

| 使用场景 | 当前结果支持的配置 | 需要同时记录的成本 |
| --- | --- | --- |
| 大量独立请求 | 连续批处理、重复前缀 KV 复用 | 批量形状、填充 token 位置数与数值一致性 |
| 存在匹配且尚未使用的 replay 记录 | 已有历史 replay、候选缓存、连续批处理 | 历史来源、缓存构建、1.5B 与 0.5B 计算量分别统计 |
| 并发提交数超过最终所需的完整 rollout 数 | 部分 rollout 调度器 | 保存 token、恢复时的前缀预填充和完整轨迹数 |
| verifier 含 CPU 或远程延迟 | 流式 IS、两阶段延迟接受；高延迟下的 MH 预取 | verifier 延迟、精确调用数和未选分支 |
| 同一提示存在冻结历史后缀 | 多尺度后缀与冻结历史混合 MH | 历史 proposal 比例、接受率、评分长度和缓存构建 |
| 分块粒子推理 | SMC 条件后缀多树搜索 | 新生成/复用 rollout 与 ESS |
| 候选成本差异较大 | 初始估计与最终估计分离、方差—成本分配 | 设计样本量与最终估计 ESS |

## 结果总结

1. 已有历史 replay 与 SMC 条件后缀多树搜索减少新生成 rollout；完整成本包含历史库构建和条件匹配。
2. 流式 IS、MH 预取和两阶段延迟接受可缩短从请求开始到 verifier 完成的时间；收益随 verifier 延迟和
   拒绝比例变化。
3. 多尺度后缀与冻结 replay 在 Qwen 1.5B MH 中可以叠加，组合在线墙钟因子为 `0.357×`；FLOPs 因子为
   `1.002×`，墙钟下降未伴随主模型计算量下降。
4. Qwen 0.5B 精确推测解码、历史草稿和方差—成本分配未降低当前单卡设置的墙钟；这些路径
   保留为显式实验配置。
5. IS 候选缓存与连续批处理在已有缓存的在线阶段可以组合；相对连续批处理纯新生成路径的墙钟、1.5B FLOPs
   和总 FLOPs 因子分别为 `0.754×`、`0.744×` 和 `0.907×`。需要新建历史记录的请求不启用该路径。

<a id="infra-report-vllm"></a>
## 后端范围与机器可读结果

本报告的 RTX 3090 数值对应 Transformers 后端。vLLM 的实现、能力边界和成对复现入口见
[vLLM 后端](../methods/ALGORITHMS.md#infra-vllm)。

机器可读结果包括：

- [`results/gsm8k_3090/`](../../results/gsm8k_3090/) 中的 `compute`、`replay`、`dynamic_is` 和
  `async_grouped` 汇总；
- [`rtx3090_transformers_summary.json`](../../results/infra/rtx3090_transformers_summary.json)
  及其六份 rollout 随机种子结果文件；
- [`rtx3090_transformers_is_mh_summary.json`](../../results/infra/rtx3090_transformers_is_mh_summary.json)
  及三份 IS/MH 随机种子结果文件；
- [`mh_replay_multiscale_stack.json`](../../results/arllm/qwen15b_optimization/mh_replay_multiscale_stack.json)、
  [`is_replay_batching_stack.json`](../../results/arllm/qwen15b_optimization/is_replay_batching_stack.json)
  与 [`draft_model_speculation_screen.json`](../../results/arllm/qwen15b_optimization/draft_model_speculation_screen.json)。
