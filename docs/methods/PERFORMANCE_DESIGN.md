# 推理性能设计

本文汇总批处理、评分、缓存和计算账本的设计选择。实现细节见
[推理基础设施实现](INFRASTRUCTURE.md)，统计目标见[推理算法实现](ALGORITHMS.md)。

## 请求与批处理

| 设计 | 实现 | 记录量 |
| --- | --- | --- |
| 跨 prompt 连续批处理 | 兼容的 `sample_batch` / `score_batch` 调用在等待窗口内合并 | batch 形状、padding、墙钟 |
| rollout 组切分 | 优先沿相同 prefix、policy 和长度的完整组切分 | sequence 数、token 上限 |
| 请求级随机流 | 每个请求保存 seed 和固定 uniform stream | token、共同前缀、数值结果 |
| 结果还原 | 物理 batch 结束后按调用方索引拆分 | 请求顺序 |

例如，15 个候选各 3 条 rollout 在 32 行上限下切分为 30+15，使同一候选的三条 rollout 保持相邻。
Transformers 使用 FP64 累积概率执行 inverse-CDF；CUDA batch 形状引起的 logits 差异通过 token
匹配率、共同前缀和最终数值结果记录。

vLLM 路径使用常驻 `AsyncLLM` 的原生调度器。仓库适配层负责请求顺序、生命周期和计量；后端比较采用
同模型、dtype、GPU 数、数据和 workload。

## 概率与评分

| 路径 | 评分策略 | 计算影响 |
| --- | --- | --- |
| on-policy rollout | 复用生成时保存的 base-policy log-probability | 省去整段重评分 |
| MH 温度 proposal | 同一 logits 同时计算 proposal 与温度 1 base 概率 | 省去 proposal 重评分 |
| off-policy rollout | proposal 生成概率 + base 批量评分 | 提供精确 `p/q` |
| 长序列评分 | 有界 microbatch + `logits_to_keep` | 限制 logits 显存 |
| 确定性评分复用 | `ScoreCachingBackend` 按模型、policy、prefix、continuation 缓存 | 将重复评分移至首次 miss |

评分 cache 与模型实例绑定；sampling policy、prefix 和 continuation 共同构成 key。显式候选缓存保存同一
request id 对应的 draw，用于固定随机性的消融。

## KV 与向量化

Transformers 对一个 batch 内的重复前缀执行一次 prefill，再复制 KV、末位置 logits 和 attention
状态。若第 $`i`$ 个前缀长度为 $`L_i`$、重复 $`K_i`$ 次，节省的非 padding prefill slots 为

$$
\sum_i (K_i-1)L_i.
$$

条件 IS、replay 和动态预算将跨候选 rollout 展平为异构 batch。多条 MH 链按 stage 和 update 锁步，
每条链独立抽取 cut、proposal seed 与 acceptance uniform。账本分别记录物理 batch、forward slots
和墙钟。

## replay 与动态预算

历史 completion 的 base/behavior 概率在 cache build 阶段验证并缓存。在线阶段包含 fresh rollout 的
生成、奖励和概率计算；端到端成本包含历史生成与全部预评分。

动态预算流程为：

1. 生成并冻结全部候选；
2. 跨候选批量生成独立 design rollout；
3. 按模型批量计算两侧概率；
4. 从 design 数据估计方差与单样本成本；
5. 冻结 evaluation 配额；
6. 生成或领取 evaluation rollout。

账本将 `cache_build`、`design`、`online` 和 `background_drain` 分列。

## 计量

主模型计算量按

$$
\widehat F=2\sum_j N_jS_j
$$

估算，其中 $`N_j`$ 为模型参数量，$`S_j`$ 为实际 forward token slots。prefill、decode、完整评分和
speculative verification 分别计数。墙钟、显存和吞吐描述硬件执行；FLOPs 描述逻辑主干计算。

各优化的精确定义、分母和 RTX 3090 结果见
[推理执行与 rollout 复用实验](../reports/RTX3090_ROLLOUT_INFRA.md)。
