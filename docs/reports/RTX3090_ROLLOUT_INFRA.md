# RTX 3090 推理执行与 rollout 复用实验

这份报告讨论同一个 IS 或 MH 算法怎样跑得更快、少浪费多少 rollout，以及相应代价。方法准确率、
pass@k 和共享奖励目标比较单独见
[GSM8K 方法效果与准确率](GSM8K_3090_ALIGNED_RESULTS.md)。这里不把单题诊断奖励当成质量排名，也不把
墙钟下降自动解释成主模型计算量下降。

## 先理解推理时间花在哪里

一次带 rollout 的推理大致经过下面几段：

```text
请求队列 ──> 主模型 prefill / decode ──> 奖励或 verifier ──> IS 权重或 MH 接受判断
   ↑                   ↑                         │
   │                   │                         │
历史 rollout ──> 统计复用 / 草稿 / proposal <────┘
```

因此，加速来源只有几类：让 GPU 同时处理更多请求；避免重复 prefill 或 decode；让历史 rollout 替代
一部分 fresh 工作；把 GPU 生成和 CPU verifier 重叠；或者在调用昂贵 verifier 之前先排除明显不会接受
的 proposal。不同机制节省的对象并不相同，必须同时报告墙钟、主模型 FLOPs、缓存构建和后台工作。

## 每项优化的直观原理

| 优化 | 实际做法 | 直观收益来源 | 不会自动省掉什么 |
| --- | --- | --- | --- |
| 连续批处理 | 汇合不同 prompt 在相近时刻提交的生成/评分请求 | 用更满的 batch 提高 GPU 利用率 | 通常不减少逻辑 token 或 FLOPs |
| 重复前缀 KV 复用 | 同一 batch 只 prefill 一次相同候选前缀，再复制 KV 状态 | 避免每条 rollout 重算相同 prompt 和候选 | 后续每条 completion 的 decode 仍要执行 |
| warm rollout replay | 读取已保存的 completion、奖励和真实 behavior 概率，只补少量 fresh rollout | 历史样本直接替代在线生成与评分 | 首次建库成本不会消失 |
| 部分 rollout broker | 长短请求一起执行；达到完成目标后保存未完成 token，下次从该前缀续跑 | 不再丢掉过量提交批次中已经生成的 token | 只保存 token 时，恢复仍需重新 prefill；保存 KV 才能消除此项 |
| 历史 token tree | 历史序列提出多个草稿 token，主模型一次验证一段 | 命中时减少串行 decode 轮次 | 错误草稿仍占 target verification slots |
| active-batch 草稿门控 | 大 batch 保持普通解码，只在稀疏长尾启用草稿 | 避免草稿破坏本来已经高效的大 batch | 它首先是防退化策略，不保证净加速 |
| 流式 frozen-design IS | 先冻结 fresh request id；每条完成后立即送入 verifier 和 IS 估计器 | 提前消化有限 CPU worker 的 verifier 队列 | 若 verifier 很便宜或最终仍等最长一条，收益很小 |
| 低优先级 run-ahead | 在 verifier、通信或调度空泡里预生成未来草稿 | 用本来闲置的 GPU 时间换取后续草稿 | 无空泡时会争抢 GPU，并留下后台 drain |
| MH proposal-tree 预取 | 当前 proposal 评分时，同时为“接受”和“拒绝”两个下一状态各生成 proposal | 隐藏奖励延迟 | 每步必有一个分支作废，因此增加主模型 FLOPs |
| delayed acceptance | 先用便宜 surrogate 做第一阶段接受判断；通过后才计算精确奖励 | 提前拒绝，减少昂贵 verifier 调用 | proposal 生成量和主模型 FLOPs 不变 |
| replay-aware MH proposal | 从“base proposal + 冻结历史后缀”的混合分布抽样，并显式计算正反向概率 | 命中历史时，把串行自回归生成改成并行评分 | 历史库构建、评分和 Hastings 修正仍要计入 |
| pilot / evaluation 分离 | pilot 只估计成本和方差，随后冻结独立 evaluation 数量 | 将预算移向更需要或更便宜的候选，同时避免边看结果边停 | 候选成本相同时，额外 pilot 往往只有开销 |
| SMC rollout forest | 所选 block 与旧 rollout 前缀匹配时，去掉该 block 并继承剩余后缀 | 跨 block 复用仍满足当前条件的 rollout | 不匹配的后缀不能复用，库存不足仍需 fresh top-up |

### 两种“复用”不能混在一起

历史 rollout 可以作为两种完全不同的资源：

- 统计复用：completion、奖励和 behavior 概率进入 IS 估计。这时必须去重、记录真实概率，并遵守冻结的
  evaluation 设计。
- 执行复用：历史 token 只作为 speculative draft。它不增加 IS 样本数，也不进入最终权重；主模型仍
  验证每个接受 token。

同一历史序列可以同时保存在两套数据结构中，但不能因为它既是 replay 又是 draft，就在估计量中计算
两次。broker 也只在完整 trajectory 完成后才产生 replay record；部分 token 只是可续跑状态。

### 保持目标分布不变的约束

- 流式 IS 在看到 fresh 数值前冻结全部 request id 和候选归属；完成顺序只影响何时更新，不影响最终
  样本集合。
- 随机历史草稿记录完整经验 proposal 概率。草稿 token 按标准 speculative acceptance 验证，拒绝时
  从主模型与草稿分布的正残差中抽样；不是把历史 token 直接当成 base 输出。
- proposal-tree 预取只改变下一步 proposal 的生成时间；真正使用的仍是普通 MH 选中的一条分支。
- delayed acceptance 的第二阶段补回 surrogate 与精确奖励之差，因此第一阶段早拒绝不会改变最终目标。
- replay-aware MH 对新后缀和旧后缀都计算同一个冻结混合 proposal 的概率，并保留 base 防御分量以
  覆盖完整支持集。这里没有裁剪 Hastings ratio。

这些性质由有限状态分布测试和逐随机流一致性测试覆盖；真实 BF16 GPU 仍可能因 batch 形状改变而出现
不同 token trace，因此固定 trace 不是分布正确性的必要条件。

## 实验与计量口径

所有成对因子均为“优化路径 / 对照路径”：小于 1 表示减少，大于 1 表示增加。墙钟排除模型和数据
加载。主模型 FLOPs 按
`2 × 参数量 × 实际 forward token slots` 估算，覆盖 prefill、decode、完整评分和 speculative target
verification；不包含 attention 的长度二次项、逐元素 kernel、CPU token tree、tokenization、奖励解析
和调度开销，所以不能用 FLOPs 代替墙钟。

| 实验组 | 用途 | setting | 重复 |
| --- | --- | --- | --- |
| GSM8K 完整网格 | 连续批处理、warm replay、动态候选和累计训练成本 | 32 道固定 test 题，Qwen2.5-1.5B-Instruct，FP32，最长 192 token | 固定请求级随机数 |
| rollout 加速栈 | 历史树、负载门控、progressive、run-ahead 和 SMC forest | 固定公开 test 第 1311 题，同一 1.5B 模型，BF16，最长 64 token | 3 个独立 seed |
| IS / MH 复用诊断 | 部分续跑、流式 IS、随机草稿、预取、delayed acceptance、replay proposal | 同一公开题与模型，16-token chunk；流式/奖励诊断使用明确标注的 0.2 s verifier | 3 个独立 seed；replay 每 seed 4 链 |

后两组刻意缩小任务来隔离 infra 因果关系，不使用单题奖励给方法质量排序。0.2 s 延迟是受控诊断条件，
不计入主模型 FLOPs，也不代表 GSM8K verifier 的真实固定耗时。

## IS / MH 复用消融

图中每个因子的分母都写在下表。绿色只表示该项指标下降；例如部分续跑的墙钟为绿色，但 FLOPs 为红色，
不能概括成“全面更省计算”。

![IS 与 MH rollout 复用消融](../assets/rtx3090_is_mh_reuse.svg)

| 优化路径 | 对照路径 | 墙钟因子 | 主模型 FLOPs 因子 | 直接观测 |
| --- | --- | ---: | ---: | --- |
| 部分 rollout 续跑 | 丢弃部分 token 后从头生成 | 0.793 ± 0.080× | 3.346 ± 0.000× | 有效生成 token 因子 0.769×；保存 96 token |
| 流式 IS，便宜 verifier | 等完整 batch 后再提交 verifier | 1.027 ± 0.040× | 1.000 ± 0.000× | 无可隐藏的 verifier 队列 |
| 流式 IS，0.2 s verifier | 等完整 batch 后再提交 verifier | 0.671 ± 0.008× | 1.000 ± 0.000× | 首次估计更新时间因子 0.367 ± 0.002× |
| 确定性历史草稿 | 无草稿 | 0.981 ± 0.064× | 1.036 ± 0.006× | 草稿接受率 17.7% ± 13.0% |
| 精确随机历史草稿 | 无草稿 | 0.982 ± 0.075× | 1.033 ± 0.004× | 草稿接受率 24.0% ± 9.0% |
| MH proposal-tree 预取，便宜奖励 | 普通 MH | 1.050 ± 0.040× | 1.424 ± 0.004× | 额外分支没有延迟可隐藏 |
| MH proposal-tree 预取，0.2 s 奖励 | 普通 MH | 0.817 ± 0.016× | 1.267 ± 0.007× | 每步预取两分支，只消费一分支 |
| delayed acceptance，0.2 s 精确奖励 | 普通 MH | 0.827 ± 0.111× | 1.000 ± 0.000× | 精确奖励调用因子 0.556 ± 0.294× |
| 冻结 replay 混合 proposal，在线 | base suffix proposal | 0.534 ± 0.078× | 1.003 ± 0.001× | 32 次更新中历史 proposal 占 35.4% ± 9.5% |

### 部分 rollout：少 decode 不等于少 FLOPs

对照和续跑都最终产生 8 条完整 rollout、共 320 个有效 completion token。过量提交的首轮批次中有
6 条各完成 16 token：丢弃路径浪费这 96 token，续跑路径把生成 token 从 416 降到 320。可是当前
跨后端 broker 保存的是 token，不是引擎内部 KV；恢复 6 个不同前缀使 prefill token 变为对照的
`7.700×`，逻辑 FLOPs 因而升到 `3.346×`。

墙钟仍降到 `0.793×`，原因是本卡上批量 prefill 的并行效率高于重复串行 decode，而不是主模型少算。
这项结果说明 token 级续跑可以消除“已生成内容被直接扔掉”的浪费，但若目标是同时降低 FLOPs，下一步
必须让 vLLM APC 命中恢复前缀，或为 Transformers 暴露可安全复用的 KV handle。

### 流式 IS：收益来自提前消化 verifier 队列

实验固定 12 条 rollout 和 2 个 verifier worker。最终 IS request id 在生成前已冻结，两条路径得到相同
数量的完整贡献。verifier 近乎零成本时，流式回调没有净收益；每条 verifier 为 0.2 s 时，短序列完成后
立刻占用 CPU worker，使大部分队列与剩余 GPU decode 重叠，墙钟降到 `0.671×`。主模型 workload 完全
相同，因此 FLOPs 因子严格为 1。

### 随机历史草稿：接受率提高，但尚未形成稳定加速

随机 proposal 不只选择历史树中最高频 token，而是按完整经验分布抽样并做残差校正。平均接受率从
确定性草稿的 17.7% 提高到 24.0%，但两种草稿相对无草稿的墙钟标准差都跨过 1，且主模型 FLOPs 增加
约 3%–4%。当前 8 条历史、4 条单请求解码还不足以让更高接受率稳定覆盖验证与 CPU 分布处理开销；这项
实现首先扩展了可用 proposal，而不是已经确认的默认加速项。

### MH proposal-tree 预取：用额外 FLOPs 换奖励延迟

4 次 MH 更新共使用 4 个 proposal；预取版实际生成 7 个，其中 3 个属于最终未选择的分支。奖励便宜时，
这些额外工作使墙钟和 FLOPs 都上升。每次奖励增加 0.2 s 受控延迟后，两个下一状态的 proposal batch
与当前奖励重叠，墙钟降到 `0.817×`，代价是 `1.267×` 主模型 FLOPs。因此它适合外部 verifier、远程
工具或 CPU 奖励明显慢于 proposal 的场景，不适合普通廉价解析器。

有限状态后端上，预取路径与普通 MH 在相同随机流下逐字段一致。BF16 真实模型中，两分支 batch 与单分支
batch 可能使用不同数值 kernel，三个 seed 的固定 token trace 并非全部相同；这不改变两条路径各自定义
的 MH 分布，但意味着墙钟应解释为相同预算的吞吐比较，而不是固定 token 的逐条计时。

### Delayed acceptance：少调用 verifier，不少生成 proposal

受控 surrogate 是与精确奖励相同的 token 谓词，但不带 0.2 s 延迟。第一阶段拒绝后不调用精确奖励；
通过时第二阶段做完整校正。三个 seed 的精确调用平均降到普通 MH 的 55.6%，墙钟降到 82.7%，主模型
FLOPs 完全不变。调用节省的方差较大，说明实际收益取决于 surrogate 能否以足够低的成本拒绝 proposal。
若 surrogate 几乎全部放行，它会退化为额外一次便宜评分，而不会加速。

### Replay proposal：并行评分替代一部分串行生成

4 条独立链共享一个冻结历史库，共进行 32 次更新。混合 proposal 保留 30% base 防御分量，并为历史
后缀和新后缀都显式计算混合概率。平均 35.4% 的更新使用历史后缀；这些命中不再逐 token 自回归生成，
而是通过一次 teacher-forced forward 得到 base 概率，所以在线墙钟降到 `0.534×`，主模型逻辑 FLOPs
基本不变。

若把 8 条历史序列的 cache build 完整计入，墙钟因子仍为 `0.586 ± 0.078×`，但 FLOPs 因子变为
`1.070 ± 0.001×`。这是“相近逻辑计算量通过更并行的 kernel 更快完成”的受控结果，不表示任意历史库
都能加速：当前历史与链前缀匹配率较高，真实部署应同时监控前缀命中率、接受率、评分长度和建库摊销。

## 完整 GSM8K 网格中的执行优化

### 连续批处理

8 个 prompt worker 共享一张 RTX 3090，并保留一次算法调用中的重复前缀组。分母是同一方法的逐 prompt
同步路径。

| 方法 | 同步墙钟 | 连续批处理墙钟 | 墙钟因子 | FLOPs 因子 | 数值答案一致 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 95.4 s | 19.7 s | 0.206× | 1.177× | 32/32 |
| Best-of-8 | 136.9 s | 67.9 s | 0.496× | 1.021× | 30/32 |
| 标准条件 IS | 427.1 s | 370.0 s | 0.866× | 1.008× | 32/32 |
| 0.5B proposal 条件 IS | 419.5 s | 399.5 s | 0.952× | 1.003× | 32/32 |

Base 最容易填满物理 batch，因此收益最大；IS 含多个依赖阶段，可跨 prompt 合并的串行段更少。padding
和 batch 分叉甚至会增加逻辑 slots，所以连续批处理的收益应写成 GPU 利用率提高，而不是算法计算减少。
Best-of-8 有两条数值答案分叉，该行只支持相同配置 workload 的吞吐比较。

### Warm rollout replay

对照固定为 8 个 base 候选、每候选 3 条总 rollout。warm 路径最多读取 2 条已评分历史记录，并始终
保留 1 条 fresh base rollout。

| 路径 | 推理 PFLOPs | 墙钟 | 相对 fresh-only FLOPs | 相对 fresh-only 墙钟 |
| --- | ---: | ---: | ---: | ---: |
| fresh-only | 1.3483 | 422.5 s | 1.000× | 1.000× |
| warm cache 在线阶段 | 1.0326 | 362.9 s | 0.766× | 0.859× |
| cache build + 首次 warm 查询 | 3.1563 | 763.4 s | 2.341× | 1.807× |

在线阶段少 23.4% FLOPs、少 14.1% 墙钟，但完整计入建库后，需要同一 replay-key 到第 7 次重复查询才
同时在两项指标上回本。warm 与 fresh-only 的准确率差及配对区间见准确率报告。

### 动态候选、缓存与方差—成本预算

固定 8 个候选、48-token block、最长 192 token 和每个非终止候选 3 条 evaluation rollout。

| 路径 | 实际复用率 | 稳态 PFLOPs | 稳态墙钟 | 一次性总 PFLOPs | 一次性总墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| base 候选 + 固定 fresh | 0% | 2.0973 | 705.7 s | 2.0973 | 705.7 s |
| 动态候选 + 固定 replay | 34.943% | 1.9312 | 704.0 s | 4.8764 | 1,401.5 s |
| 动态候选 + 方差—成本分配 | 5.707% | 2.3172 | 706.6 s | 7.2951 | 2,186.2 s |

动态固定组相对 base 固定组的稳态 FLOPs 为 `0.921×`、墙钟为 `0.998×`；小模型承担了一部分工作，
但没有形成实际墙钟加速。方差—成本组相对动态固定组为 `1.200×` FLOPs、`1.004×` 墙钟。当前每来源
2 条 design rollout 的统计噪声较大，分配器只复用 5.707% rollout，平均最终 ESS 还略低；它验证了
正确的数据分离流程，但没有验证效率收益。

### 训练与免训练推理的累计账本

GRPO 一次训练为 15.646 PFLOPs、5,007,660 个前向等价 token slots 和 9,545.2 秒。只比较累计 FLOPs
时，verifier-MH、标准 verifier-IS 和 0.5B proposal verifier-IS 与“训练 + GRPO 推理”的交点分别为
392、344 和 230 次查询。由于这些方法与 GRPO 的准确率绝对差超过预设阈值，它们只是计算账本交点，
不是“达到相同效果所需查询数”。

## 历史树、progressive 与 SMC 消融

下图对应较早的 rollout 加速栈实验。cache build、在线路径和后台 drain 分列。

![RTX 3090 rollout 基础设施消融](../assets/rtx3090_rollout_infra.svg)

### 历史 token tree 与负载门控

请求依次使用 active batch 4、2、1，模拟 rollout 尾部逐渐变稀。

| 路径 | 在线墙钟（s） | 输出 token/s | 主模型 PFLOPs | cache build（s） | 草稿接受率 | 墙钟因子 | FLOPs 因子 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 无草稿 | 3.838 ± 0.059 | 116.8 ± 1.8 | 0.00247 ± 0.00000 | 0.000 ± 0.000 | 0.0% | 1.000× | 1.000× |
| 历史树，始终草稿 | 8.297 ± 0.075 | 54.0 ± 0.5 | 0.00411 ± 0.00001 | 1.280 ± 0.027 | 13.1% | 2.162× | 1.660× |
| 历史树，负载感知 | 3.783 ± 0.067 | 118.5 ± 2.1 | 0.00249 ± 0.00001 | 1.335 ± 0.058 | 37.5% | 0.986× | 1.006× |

始终草稿会让 target 验证大量随后丢弃的 token；只在 batch 1 长尾启用草稿可以避免严重退化。
`0.986×` 相对三 seed 波动不能视为稳定加速，因此 active-batch 调度的已验证价值首先是保护吞吐。

### Progressive、run-ahead 与 SMC rollout forest

算法层使用 3 个候选、每候选 2 条总 rollout；SMC 使用 3 个粒子、每粒子 2 个分支。

| 路径 | 在线墙钟（s） | 在线主模型 PFLOPs | cache build（s） | 后台 drain（s） | fresh / reused rollout |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 rollout 条件 IS | 3.442 ± 0.162 | 0.00809 ± 0.00067 | 1.248 ± 0.057 | 0.000 ± 0.000 | 0.0 / 0.0 |
| pilot/evaluation 分离 | 5.313 ± 0.231 | 0.01250 ± 0.00000 | 1.269 ± 0.100 | 0.000 ± 0.000 | 0.0 / 0.0 |
| 流式奖励 + run-ahead | 5.605 ± 0.958 | 0.01261 ± 0.00192 | 1.256 ± 0.068 | 7.763 ± 1.927 | 0.0 / 0.0 |
| SMC forest，不复用 | 2.985 ± 0.260 | 0.01271 ± 0.00038 | 1.250 ± 0.080 | 0.000 ± 0.000 | 35.3 / 0.0 |
| SMC forest，复用 | 2.571 ± 0.510 | 0.01226 ± 0.00206 | 1.263 ± 0.042 | 0.000 ± 0.000 | 24.7 / 13.3 |

pilot/evaluation 分离相对固定 rollout 为 `1.544×` 墙钟、`1.553×` FLOPs；成本同质时，额外 pilot
没有可换取的预算收益。旧 run-ahead 实验的 verifier 几乎没有 CPU 尾部，因此在线为 progressive 的
`1.051×`，还产生 `7.763 ± 1.927 s` drain。它与上面的流式 IS 结果并不矛盾：新的受控实验明确加入
了两 worker verifier 队列，只有存在空泡时重叠才有效。

SMC forest 复用相对同一 SMC 不复用版为 `0.856×` 墙钟、`0.963×` FLOPs，fresh rollout 均值从
35.3 降到 24.7。它只继承与所选 block 匹配的条件后缀，同一有限 rollout 也不会复制成多个独立观测。

## 部署选择

| 场景 | 建议优先项 | 暂不默认开启的项 |
| --- | --- | --- |
| 大量独立请求 | 连续批处理、重复前缀 KV 复用 | 高并发时的激进 token draft |
| replay key 会重复 | warm replay；同时单列 cache build | 冷启动只有一次的历史库 |
| 过量提交造成长尾 rollout 被丢弃 | 部分 rollout broker；优先结合 APC/KV handle | 仅保存 token 却忽略重复 prefill 的 FLOPs |
| verifier 有明显 CPU/远程延迟 | 流式 IS、delayed acceptance；延迟足够大时 MH 预取 | verifier 便宜时的分支预取和 run-ahead |
| 历史前缀命中率高 | replay-aware MH proposal、带接受率门控的历史草稿 | 低命中静态草稿 |
| 逐 block 粒子推理 | SMC 条件后缀 forest | 复制同一 rollout 充当多条独立样本 |
| 候选 rollout 成本差异大 | pilot/evaluation 分离后再评估预算分配 | 当前小 design pool 的方差—成本分配器 |

当前最清晰的结论不是“某一个开关普遍最快”，而是：

1. replay/forest 可以真正减少 fresh 工作，但必须把建库和条件匹配写清楚；
2. streaming、prefetch 和 delayed acceptance 主要优化 verifier 关键路径，收益随 verifier 延迟变化；
3. speculative draft 和部分续跑可能降低墙钟却增加逻辑 FLOPs，二者必须并列报告；
4. 精确 proposal 修正使历史数据可以服务 IS 和 MH，而不必把目标分布改成历史策略的分布。

## vLLM 状态与复现

broker、流式 frozen-design IS、proposal-tree 调度、delayed acceptance 和 replay-aware MH 都建立在公共
后端接口上，可由 Transformers 或 vLLM 执行。vLLM 使用常驻 `AsyncLLM` 的完成回调和 Automatic Prefix
Caching；历史 suffix draft 使用 vLLM 原生 target verifier。任意外部经验分布的随机 token-tree 与残差
校正目前是 Transformers 专用消融，不能写成已经注入 vLLM 原生 suffix cache。

当前 RTX 3090 是 Windows 主机且没有 WSL，vLLM 不原生支持 Windows，因此本报告没有填造 vLLM 数值。
在 Linux/WSL2 上可按相同 schema 运行后端无关部分；原生 suffix 的成对入口见
[vLLM 推理运行时](../methods/VLLM_RUNTIME.md)。

复现新增实验：

```powershell
$env:PYTHONPATH = "src;."
foreach ($seed in 20260812, 20260813, 20260814) {
  .\.venv\Scripts\python experiments\benchmark_is_mh_reuse.py `
    --backend transformers --dtype bfloat16 --section all --seed $seed `
    --output "results\infra\rtx3090_transformers_is_mh_seed$seed.json"
}

.\.venv\Scripts\python experiments\summarize_is_mh_reuse.py `
  --inputs results\infra\rtx3090_transformers_is_mh_seed20260812.json `
           results\infra\rtx3090_transformers_is_mh_seed20260813.json `
           results\infra\rtx3090_transformers_is_mh_seed20260814.json `
  --output results\infra\rtx3090_transformers_is_mh_summary.json `
  --svg docs\assets\rtx3090_is_mh_reuse.svg
```

机器可读来源包括：

- [`results/gsm8k_3090/`](../../results/gsm8k_3090/) 中的 `compute`、`replay`、`dynamic_is` 和
  `async_grouped` 汇总；
- [`results/infra/rtx3090_transformers_summary.json`](../../results/infra/rtx3090_transformers_summary.json)
  及其六份旧 rollout seed 文件；
- [`results/infra/rtx3090_transformers_is_mh_summary.json`](../../results/infra/rtx3090_transformers_is_mh_summary.json)
  及三份新增 IS/MH seed 文件。
