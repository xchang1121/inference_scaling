# 推理基础设施实现：调度、缓存、异步执行与计算量

本文档集中说明仓库中已经实现的执行层优化。算法目标、重要性权重、replay 恒等式和 MH 接受率见
[推理算法实现](ALGORITHMS.md)；本文件说明每项优化复用了什么、节省了什么、增加了什么，以及如何验证
它没有暗中改变算法。RTX 3090 消融见
[推理执行与 rollout 复用实验](../reports/RTX3090_ROLLOUT_INFRA.md)。

## 1. 分层原则与计量口径

一次推理任务可分为四层：

1. **统计设计**：确定候选、rollout 数、replay claim 和 MH 更新；
2. **逻辑请求**：带有固定 prefix、sampling policy、seed 和 request id 的生成或评分请求；
3. **物理执行**：合批、prefill、KV decode、speculative verification、CPU reward 和异步调度；
4. **硬件结果**：墙钟、显存、吞吐和能耗。

执行优化只能重排或复用第二、三层的工作。若它改变进入估计器的样本集合、漏掉 proposal 概率，或把一条历史
轨迹重复当成多个独立观测，就不再是基础设施等价变换。

下文的 FLOPs 指 floating-point operations（浮点运算次数）。仓库的主模型计算量采用稠密矩阵主导项

\[
\widehat F_{\mathrm{forward}}=2N\,S,
\tag{1}
\]

其中 \(N\) 为模型参数量，\(S\) 为实际 forward token slots。多个模型分别计算后求和：

\[
\widehat F_{\mathrm{total}}=\sum_j2N_jS_j.
\tag{2}
\]

该估计覆盖仓库计数的 prefill、decode、完整序列评分和 target speculative verification；不覆盖 attention
长度二次项、逐元素 kernel、tokenization、CPU 调度与通信。墙钟衡量硬件执行效率，式 (1) 衡量逻辑主干
工作量；二者必须并列报告。连续批处理可能明显降低墙钟而几乎不改变式 (1)，预取和 speculative decoding
也可能降低墙钟但增加 token slots。

<a id="infra-overview"></a>
## 2. 已实现机制总览

| 机制 | 主要收益来源 | 可能增加的成本 | 主要实现 |
| --- | --- | --- | --- |
| 跨 prompt 连续批处理 | 提高有效 batch 与 GPU 利用率 | 等待窗口、padding | `backends/batching.py` |
| 重复前缀 key-value（KV）复用 | 每组相同前缀只 prefill 一次 | KV 复制与 batch padding | `backends/transformers_backend.py` |
| vLLM AsyncLLM + APC | 引擎级连续调度与跨调用 prefix block 复用 | 常驻显存、调度开销 | `backends/vllm_backend.py` |
| 生成概率直返 | 避免 on-policy 与 MH proposal 的重复评分 | 额外 log-prob 处理 | 两套 backend |
| 评分 cache / microbatch | 复用确定性分数并限制显存峰值 | cache build、查表 | `backends/cache.py` |
| 展平 rollout / 向量化 MH | 删除逐候选、逐链同步点 | 变长 padding | 算法模块与 backend |
| 历史 token tree | 用多 token 草稿减少串行 decode 轮次 | 被拒绝草稿的验证 slots | `acceleration.py` |
| active-batch 草稿长度 | 只在长尾启用 speculation | 阈值标定 | `ActiveBatchSpeculationConfig` |
| 部分 rollout broker | 保存过量提交产生的已生成 token | 恢复 prefix 的 prefill | `rollout_broker.py` |
| 流式奖励回调 | GPU 生成与 CPU/verifier 工作重叠 | 线程池与尾部排队 | `StreamingRewardEvaluator` |
| 低优先级 run-ahead | 用空泡生成未来 draft 材料 | 后台 FLOPs、前台等待 | `LowPriorityRunAheadBackend` |
| MH proposal-tree 预取 | 用额外分支隐藏奖励延迟 | 未使用 proposal | `algorithms/mh_acceleration.py` |
| delayed acceptance | 早拒绝减少精确 verifier 调用 | surrogate 计算 | 同上 |
| replay-mixture proposal | 历史命中把串行生成变为批量评分 | cache build、正反向评分 | 同上 |
| SMC reservoir 继承 | 减少条件 lookahead 的 fresh rollout | reservoir 管理 | `algorithms/smc_forest.py` |

下文相对源码路径均位于 [`src/inference_scaling`](../../src/inference_scaling/)。

<a id="infra-sources"></a>
### 设计来源

| 机制族 | 主要文献 | 本仓库中的关系 |
| --- | --- | --- |
| 迭代级连续调度 | [Orca，Yu et al. (2022)](https://www.usenix.org/conference/osdi22/presentation/yu) | 跨 prompt 汇合生成与评分请求 |
| KV block 与前缀复用 | [PagedAttention，Kwon et al. (2023)](https://doi.org/10.1145/3600006.3613165)、[SGLang，Zheng et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/724be4472168f31ba1c9ac630f15dec8-Paper-Conference.pdf) | 重复前缀 prefill、APC 与恢复前缀复用 |
| speculative decoding | [Leviathan, Kalman, and Matias (2023)](https://proceedings.mlr.press/v202/leviathan23a.html) | target 接受率与拒绝后的残差校正 |
| 检索式 token tree | [REST，He et al. (2024)](https://aclanthology.org/2024.naacl-long.88/)、[SpecInfer，Miao et al. (2024)](https://doi.org/10.1145/3620666.3651335) | 历史 rollout 构成多 token 草稿 |
| 异步生成与消费 | [IMPALA，Espeholt et al. (2018)](https://proceedings.mlr.press/v80/espeholt18a.html)、[SAO，Hou et al. (2026)](https://arxiv.org/abs/2607.07508) | completion callback、部分 rollout 和低优先级 run-ahead 的调度来源 |
| MCMC prefetch | [Brockwell (2006)](https://doi.org/10.1198/106186006X100579) | 奖励等待期间预取接受/拒绝两条 proposal 分支 |
| delayed-acceptance MCMC | [Christen and Fox (2005)](https://doi.org/10.1198/106186005X76983) | surrogate 早拒绝与精确第二阶段 |

replay 的统计校正、动态候选和 SMC 数学来源列在[算法文档的方法来源](ALGORITHMS.md#alg-sources)；
本文件只扩展它们的执行路径与成本账本。

<a id="infra-report-labels"></a>
### 报告中的实验臂名称

infra 报告中的逗号前是被启用的机制，逗号后是 workload 条件或成本口径。例如，“流式 IS，0.2 s
verifier”表示启用流式完成回调，并将每条 verifier 的受控耗时设为 0.2 s；它不是一种新的 IS
权重。“在线”表示只计已经建库后的调用，不含 cache build。

| 报告名称 | 启用的机制 | 与相邻对照的唯一主要差异 |
| --- | --- | --- |
| 部分 rollout 续跑 | `AsyncRolloutBroker` 保存未完成 token 并从保存前缀继续 | 对照丢弃部分轨迹并从原始前缀重生成 |
| 流式 IS，便宜 verifier | 完成一条 rollout 后立即提交奖励 worker | verifier 几乎没有受控延迟 |
| 流式 IS，0.2 s verifier | 与上一行相同 | 每条相同 verifier 额外加入 0.2 s 延迟 |
| 确定性历史草稿 | token tree 每步提出最高频 token | target 仍逐位置验证草稿 |
| 精确随机历史草稿 | 从完整经验分布 \(q_t\) 随机提出 token，并执行式 (5)--(6) | “精确”表示输出仍服从 target，而非草稿一定正确 |
| MH proposal-tree 预取，便宜奖励 | 为接受与拒绝两种下一状态各预取一个 proposal | reward 几乎没有受控延迟 |
| MH proposal-tree 预取，0.2 s 奖励 | 与上一行相同 | 每次相同 reward 额外加入 0.2 s 延迟 |
| delayed acceptance，0.2 s 精确奖励 | surrogate 第一阶段早拒绝，精确奖励执行第二阶段 | 对照对每个 proposal 直接调用带 0.2 s 延迟的精确奖励 |
| 冻结 replay 混合 proposal，在线 | MH proposal 为 base 与冻结历史后缀的 mixture | “在线”排除历史库构建；含 build 的成本另列 |
| warm cache 在线阶段 | IS 读取已经评分的 replay 记录 | 排除历史生成和评分成本 |
| cache build + 首次 warm 查询 | 与上一行相同 | 将历史生成、评分和第一次在线查询全部计入 |
| 历史树，始终草稿 | 所有 active batch 大小都启用固定草稿长度 | 无负载门控 |
| 历史树，负载感知 | 只在 active batch 足够小时启用草稿 | 当前实验在长尾 batch 1 启用 |
| 固定 rollout 条件 IS | 每个候选直接使用固定 evaluation rollout 数 | progressive/SMC 对照 |
| pilot/evaluation 分离 | pilot 只估计方差和成本，另生成 evaluation | pilot 不进入最终 IS 权重 |
| 流式奖励 + run-ahead | progressive IS 加完成回调和低优先级后台草稿 | 报告同时计在线路径与最终 background drain |
| SMC forest，fresh-only | SMC 每个 branch 的 lookahead 全部重新生成 | 禁用条件后缀 reservoir 继承 |
| SMC forest，条件后缀复用 | 所选 block 匹配时继承并一次性消费条件后缀 | 其余粒子使用 fresh top-up |

算法层的 `Base`、`GRPO 贪心`、`0.5B proposal 条件 IS`、`verifier-MH` 和动态候选等名称见
[报告组合名称](ALGORITHMS.md#alg-report-labels)。

<a id="infra-request-contract"></a>
## 3. 后端协议、随机数与数值一致性

算法只依赖 `sample_batch` 和 `score_batch`：

```python
GenerationRequest(prefix, max_new_tokens, sampling, seed, request_id)
ScoreRequest(prefix, continuations, sampling)
```

每个生成请求拥有独立 seed。连续批处理、链向量化或异步完成不会改用 batch 级随机流；输出还会按调用方原始
顺序拆回。Transformers 后端预先为每个请求生成固定 uniform stream，并用 FP64 cumulative probability
做 inverse-CDF：

```python
uniforms = np.random.default_rng(request.seed).random(request.max_new_tokens)
cumulative = probabilities.to(torch.float64).cumsum(dim=-1)
token = (cumulative < uniform).sum(dim=-1)
```

这使请求换到另一个物理 batch 后仍使用同一随机阈值。不同 CUDA batch 形状可能产生微小 logits 差异，导致
阈值恰好跨过 token 边界；因此真实硬件验证同时报告 exact-token match、共同前缀和最终数值结果，而不是只凭
seed 宣称逐 token 必然相同。

生成样本保存实际 sampling policy 下的逐 token log-probability。温度、top-p、top-k 或模型版本均属于概率
定义；replay 与 MH 的概率记录包含这些字段。

<a id="infra-continuous-batching"></a>
## 4. 跨 prompt 连续批处理

`ContinuousBatchingBackend` 允许多个同步算法 worker 共享一个 dispatcher。时间上相邻且兼容的调用组在
`batch_wait_seconds` 内合并，兼容键包含操作类型、sampling policy、生成长度或评分长度 bucket。dispatcher
同时约束 sequence 数和估计 token 数：

```python
fits = (
    sequence_count + candidate.sequence_count <= max_batch_size
    and token_cost + candidate.token_cost <= max_batch_tokens
)
```

一次调用方提交的 rollout 组优先保持完整。若超出上限，生成请求先沿“相同 prefix + policy + 长度”的连续组
切分，使下层仍能识别重复前缀。评分结果与生成结果按原组顺序返回，调度顺序不进入统计设计。

该机制的分母是“同一方法逐 prompt 同步运行”。它通常减少 wall time、增加 samples/s，但算法 token 数不变；
padding 形状变化还可能令式 (1) 略增。异步 vLLM 已有原生 continuous scheduler，wrapper 在该模式只统计并
直接透传，避免增加一层 Python 串行队列。

<a id="infra-flattening"></a>
## 5. rollout 展平与独立 MH 链向量化

条件 IS、replay 和动态预算为不同候选分配的 rollout 数可以不同。实现先构造所有异构
`GenerationRequest`，记录每个请求所属候选，再发出一个物理 batch；奖励与分数完成后按索引还原。reserve
rollout 使用同一路径。这删除了“候选 0 完成后才生成候选 1”的同步点。

多条 MH 链按 stage 和 update 锁步推进。每条链独立抽 cut、proposal seed 和 acceptance uniform，只把该步
所有链的变长后缀请求放入一个 batch：

```python
cuts = tuple(stream.generator(..., "cut").integers(0, stage_length) for stream in streams)
suffix_lengths = tuple(stage_length - cut for cut in cuts)
proposals = backend.sample_batch(requests_for_all_chains)
```

向量化不共享链状态，也不减少逻辑 proposal 数；收益来自并行度。后缀长度差异会形成 padding，因此报告分别
列出物理 batch 数、forward slots 与墙钟。

<a id="infra-prefix-kv"></a>
## 6. Transformers 重复前缀 KV 复用

若 \(M\) 个候选各有 \(K\) 条 rollout，同一候选的 \(K\) 条请求共享 `prompt + generated + candidate`
前缀。朴素实现对每条请求重复 prefill；当前实现先对 \(M\) 个唯一前缀执行一次 prefill，再把每组 KV、最后
位置 logits 和 attention mask 复制 \(K\) 次，继续执行一个 \(MK\) 行 decode batch。

若第 \(i\) 个唯一前缀长度为 \(L_i\)，理想情况下保存的非 padding prefill slots 为

\[
S_{\mathrm{saved}}=\sum_{i=1}^{M}(K_i-1)L_i.
\tag{3}
\]

```python
unique_outputs = model(unique_input_ids, use_cache=True, logits_to_keep=1)
cache = repeat_cache(unique_outputs.past_key_values, prefix_repeat_count)
logits = unique_outputs.logits[:, -1, :].repeat_interleave(prefix_repeat_count, dim=0)
```

实现支持一个 batch 内的多组重复前缀；只有各组 repeat count 一致时才进入该快速路径，否则使用普通 padded
batch，避免错误重排。`shared_prefill_tokens_saved` 按式 (3) 记录实际避免的前缀处理量。

<a id="infra-score-path"></a>
## 7. 概率评分的避免、合并与缓存

### 7.1 生成时同时返回 proposal 与基础概率

on-policy rollout 的生成 log-probability 就是[算法式 (10)](ALGORITHMS.md#alg-offpolicy-is)中的基础概率，
因此不再执行整段重评分。温度 proposal
用于 MH 时，Transformers 在同一份 logits 上同时计算实际 proposal policy 和温度 1 基础 policy 的选中
token 概率，并随样本返回：

```python
SequenceSample(
    token_logprobs=proposal_values,
    reference_token_logprobs=base_values,
    reference_policy_id=base_policy_id,
)
```

算法仅在 reference 缺失时调用 `score_batch`。这减少一次模型前向，不删减 Hastings 比中的任何项。

### 7.2 精确评分 cache

`ScoreCachingBackend` 只缓存确定性的 continuation 分数，key 为

\[
(\text{完整 SamplingConfig},\ \text{prefix},\ \text{continuation}).
\tag{4}
\]

cache 与单个模型实例绑定；不同温度、截断策略或模型版本使用不同 cache key。miss 会按
`(policy, prefix)` 重新分组为批量评分，完成后写入有界 LRU。随机生成从不被透明缓存。

`CachedCandidateBackend` 是另一种显式算法缓存：只有 request id、prefix、policy、model 和最大长度全部匹配
时才重放已经冻结的 candidate draw，概率评分仍委托真实 backend。它用于固定公平消融中的候选随机性，
语义是重放同一次 draw，而非产生新的独立样本。

### 7.3 microbatch 与 `logits_to_keep`

长序列评分按 `max_score_batch_size` 分块，防止 `[batch, length, vocabulary]` logits 占满 24 GB 显存。
支持 Qwen `logits_to_keep` 时，生成 prefill 只保留最后一行 vocabulary logits；评分只保留覆盖 continuation
预测位置的尾部 logits：

```python
logits_to_keep = max(len(continuation) for continuation in chunk) + 1
outputs = model(input_ids, use_cache=False, logits_to_keep=logits_to_keep)
```

Transformer body 仍处理完整上下文，所以这主要减少输出投影与 logits 显存，不应写成完整 prompt FLOPs 被删除。

<a id="infra-speculation"></a>
## 8. 历史 token tree 与精确 speculative verification

`RolloutTokenTree` 在 CPU 上保存有界的“后缀 context → 下一 token 计数”。查询从最长匹配 context 开始，
最多提出 \(K\) 个 draft token。历史轨迹只作为执行草稿，不直接进入 IS 权重。

确定性模式沿最高频 token 前进。target verifier 对每个位置按真实目标 \(p_t\) 使用请求自己的 uniform 抽样；
若抽样 token 等于 draft 就继续验证，否则保留 target 抽样 token 并停止草稿路径。因此输出仍是逐 token target
抽样。

随机模式从经验 proposal \(q_t\) 抽 draft \(a\)，接受概率为

\[
\Pr(\text{accept }a)=\min\left\{1,\frac{p_t(a)}{q_t(a)}\right\}.
\tag{5}
\]

拒绝后从归一化残差

\[
\frac{(p_t(v)-q_t(v))_+}{\sum_w(p_t(w)-q_t(w))_+}
\tag{6}
\]

抽取替代 token。某 token 由接受路径产生的概率为 \(\min(p_t,q_t)\)，由拒绝路径产生的概率为
\(p_t-\min(p_t,q_t)\)，两者相加恰为 \(p_t\)。这就是随机历史 proposal 不改变目标分布的逐步证明。

Transformers 一次前向验证 `prefix + K drafts`。若中途拒绝，会把 `DynamicCache` 裁到已接受 draft 末端，
再把 target 抽出的 mismatch token 写入并继续 decode，避免重新 prefill：

```python
reusable_cache = crop_cache(outputs.past_key_values, len(prefix) + accepted)
continue_verified_cache(request, cache=reusable_cache, consumed=consumed)
```

被拒绝的 draft 仍占 target verification slots。需要同时记录 tree hit rate、draft acceptance、验证 slots 和
wall time；高命中但低接受率仍可能减速。

<a id="infra-active-batch"></a>
## 9. active-batch 草稿长度

大 batch 下普通解码已有较高算力利用率，错误草稿会扩大无效验证工作；长尾只剩少量请求时，减少串行 decode
轮次更有价值。实现令草稿长度为分段函数 \(K(b)\)，其中 \(b\) 是当前 active batch：

```toml
[acceleration.speculation]
enabled = true
tiers = [[1, 8], [4, 0], [512, 0]]
min_context_tokens = 2
min_token_probability = 0.10
tree_max_context_tokens = 24
tree_max_contexts = 100000
dynamic_vllm = false
stochastic_tree = false
```

每个 tier 表示 `[最大 active batch, K]`。Transformers 每次调用直接查询该表。vLLM 可将其转换为
`[起始 batch, 结束 batch, K]`，并在 `dynamic_vllm=true` 时加载
[`DynamicSuffixDecodingProposer`](../../src/inference_scaling/vllm_suffix_proposer.py)。该适配器仍委托官方 suffix
cache 和 target verifier，只在一次加锁的 `propose` 调用内临时采用 scheduler 选定的 \(K\)。动态 vLLM
路径必须单独测量；固定最大 \(K\) 与动态 \(K(b)\) 不是同一实验臂。

<a id="infra-rollout-broker"></a>
## 10. 部分 rollout broker

`AsyncRolloutBroker` 把长生成拆成 `chunk_tokens` 大小的片段，并可过量提交请求以维持 batch。当完整轨迹数达到
目标时，尚未完成的轨迹保存：原请求、已生成 token、实际 behavior/reference log-probability、分段计数和
优先级。下一次调度从 `original prefix + saved tokens` 继续：

```python
GenerationRequest(
    prefix=original.prefix + partial.token_ids,
    max_new_tokens=min(chunk_tokens, partial.remaining_tokens),
    sampling=original.sampling,
    seed=continuation_seed(original.seed, partial.segments),
)
```

只有 EOS 或长度预算结束后才触发 completion callback 并生成 `ReplayRecord`。部分 token 保持为调度状态，
IS 估计器仅消费完整轨迹。当前 broker 保存可序列化 token，不保存跨调用通用 KV handle。Transformers 恢复时会重新 prefill；
vLLM 若 APC 尚持有相同 prefix block 则可命中。因此“少丢弃生成 token”不等于“主模型 FLOPs 一定减少”，
报告会同时列出 saved tokens、resumed prefill、wall time 与式 (1)。

<a id="infra-streaming-reward"></a>
## 11. 流式奖励计算

支持完成回调的 backend 在每条序列结束时立即触发 `on_complete(index, sample)`。
`StreamingRewardEvaluator` 将该序列提交给 CPU/verifier 线程池，使短序列的奖励与仍在 GPU 上生成的长序列
重叠：

```python
def completed(index, sample):
    futures[index] = executor.submit(
        reward, prompt[index], generated_prefix[index] + sample.token_ids
    )

samples = sample_batch_with_callback(backend, requests, completed)
rewards = tuple(future.result() for future in futures)
```

它记录 `generation_seconds` 和生成结束后的 `reward_tail_seconds`。与
[frozen-design streaming IS](ALGORITHMS.md#alg-streaming-is) 配合时，样本 id 在开始前冻结，完成顺序不会改变
最终估计量。该机制减少的是等待和队列尾部，不减少生成 FLOPs；若 reward 极便宜，线程开销可能占主导。

<a id="infra-runahead"></a>
## 12. 低优先级 run-ahead

`LowPriorityRunAheadBackend` 利用 reward 等待空泡生成未来可能有用的 base rollout，并只把输出写入历史 draft
tree。它具有三项限制：

- 后台请求按有界 token chunk 执行；
- 前台请求到达时最多等待当前 chunk，下一 chunk 必须让路；
- 后台输出不进入当前 evaluation estimator，不增加统计样本数。

队列容量、completed tokens、失败数、前台等待和最终 drain 单独记账。run-ahead 把空闲时间转换为未来可能
接受的 draft，后台工作计入总成本；若 tree 命中率低或没有 reward 空泡，它会增加总 FLOPs 和资源争用。

<a id="infra-mh-prefetch"></a>
## 13. MH proposal-tree 预取

普通 reward MH 必须等待当前 proposal 的精确奖励，才能知道下一步从接受状态还是拒绝状态出发。预取版在奖励
线程运行时，一次 batch 生成两条下一步 proposal：

```text
当前状态 y ── 当前 proposal y' 的奖励 ──┬── 接受：使用从 y' 预取的 proposal
                                       └── 拒绝：使用从 y  预取的 proposal
```

Hastings 判断仍只消费被选分支，未选分支计为 `unused_prefetched_proposals`。cut、proposal seed 与 acceptance
uniform 沿用普通链命名，有限状态测试可逐步核对所消费分支。除最后一步外，它大约用两条 proposal 的生成量换取
一条 proposal 的关键路径；只有奖励延迟足够大且双分支能有效合批时才可能降低墙钟。

这项优化不减少主模型工作量。实验中的“加速”分母是同更新数的普通 reward MH，且必须同时报告预取 FLOPs 因子。

<a id="infra-delayed-reward"></a>
## 14. 精确奖励调用的削减

[Delayed-acceptance MH](ALGORITHMS.md#alg-delayed-mh) 用便宜 surrogate 早拒绝 proposal，只对第一阶段通过项
调用精确 verifier。基础设施收益应写成

\[
\text{精确奖励调用因子}
=\frac{\text{delayed 路径精确调用数}}
       {\text{普通 MH 精确调用数}},
\tag{7}
\]

而不是写成“主模型 FLOPs 加速”：proposal 生成与基础模型评分通常不变。第二阶段校正保持目标的数学理由见算法
文档；执行报告同时列出 surrogate calls、exact calls、early rejection、wall time 和主模型 FLOPs。

<a id="infra-replay-execution"></a>
## 15. replay 的在线成本与冷启动成本

rollout replay 包含两类复用：

- **统计复用**：历史 completion 通过 behavior probability 和 fresh-tail 恒等式进入能量估计；
- **执行复用**：同一 completion 的 base/behavior 分数、prefix KV 或 token 路径被缓存。

历史库构建需要生成 completion，并在所有相关 policy 下验证概率。固定模型与 policy 下，这些确定性分数进入
`ScoreCachingBackend`；后续在线查询只读缓存。fresh base rollout 的新生成、奖励与必要评分仍计入在线成本。

报告区分：

\[
C_{\mathrm{first}}=C_{\mathrm{build}}+C_{\mathrm{online}},
\qquad
C_{R}=C_{\mathrm{build}}+R\,C_{\mathrm{online}}.
\tag{8}
\]

只有当同一兼容 replay key 被复用足够多次，累计成本才可能低于每次 fresh-only。成本比较采用式 (8) 的实际
token/FLOPs，cache hit rate 仅作为复用诊断。

replay-mixture MH 的历史命中则把自回归 suffix 生成改为 teacher-forced 批量评分。它可能显著降低墙钟，因为评分
并行度更高，但逻辑 token slots 未必下降；混合 proposal 的正反概率仍按
[式 (25)](ALGORITHMS.md#alg-replay-mh) 校正。

<a id="infra-smc-reuse"></a>
## 16. SMC rollout reservoir 的条件复用

SMC 中一条父粒子 rollout 若以新 block 开头，其余后缀可直接作为子前缀下的条件 rollout。实现只继承前缀匹配
的样本，并在同一 branch 被 resample 多次时把 reservoir 分桶，不复制样本；不足部分再批量 fresh top-up。

该机制同时可能减少 fresh rollout 数、主模型 FLOPs 与墙钟，但收益取决于 branch 命中和粒子退化。对照路径是
相同粒子数、branch factor、lookahead 数与随机种子，仅设置 `reuse_rollout_forest=false`；报告必须列出
fresh/reused 数和 ESS。

<a id="infra-transformers"></a>
## 17. Transformers 后端

Transformers 后端是概率与 KV 行为的显式参考实现，包含：

- request-local FP64 inverse-CDF；
- 同一 logits 上返回实际 policy 与基础 policy 的选中 token 概率；
- 多组重复 prefix 的一次 prefill + KV repeat；
- bounded score microbatch 与 `logits_to_keep`；
- 确定性和随机 token-tree verifier；
- 拒绝后的 `DynamicCache` crop 与续算；
- generation/score/speculative slot、cache saving 与 FLOPs 快照。

完整生成和评分由同一模型锁保护，防止线程同时改写 cache/model 状态；跨 prompt 并行通过外层 batching 汇合为更大
物理调用。`AbsorbingEOSBackend` 是 MH 固定长度适配层，不改变普通变长生成路径。

<a id="infra-vllm"></a>
## 18. vLLM 后端

### 18.1 常驻 AsyncLLM 与完成回调

`runtime.backend="vllm"` 为每个模型保留一个常驻 `AsyncLLM` 和事件循环线程。同步算法调用被提交到同一引擎；
引擎负责跨调用 continuous scheduling。多个请求使用 `asyncio.as_completed` 逐条触发完成回调，最终数组仍按原
索引返回：

```python
for completed in asyncio.as_completed(tasks):
    index, output = await completed
    parsed[index] = parse(output)
    on_complete(index, parsed[index].sample)
```

`vllm-sync` 使用离线 `LLM`，适合调试或同步 beam；它不是默认吞吐路径。

### 18.2 Automatic Prefix Caching

Automatic Prefix Caching（APC）默认开启。vLLM 返回 `num_cached_tokens`，实现从 prefill slots 中扣除命中部分：

```python
cached = min(prompt_length, int(output.num_cached_tokens or 0))
forward_slots = prompt_length - cached + max(0, generated_tokens - 1)
```

APC 可跨调用复用共同 prompt、候选前缀和恢复后的 broker prefix；它只减少重复 prefill，不减少第一个请求或长
decode。APC 的 KV block cache 与 suffix speculative cache 是两个独立机制。

### 18.3 生成与精确评分边界

引擎固定设置 `generation_config="vllm"` 和 `logprobs_mode="processed_logprobs"`，确保生成概率对应实际温度/
top-p/top-k policy，而不是模型目录中的隐式 generation config。

vLLM 原生 `prompt_logprobs` 只用于温度 1、无硬截断的 continuation 评分。非单位温度 behavior 的重评分、
top-k/top-p policy 和完整词表 entropy/self-certainty 会委托配置的 exact Transformers backend：

```python
if supports_native_score(request.sampling):
    native.append(item)
else:
    delegated.append(item)
```

没有 exact backend 时，不支持的评分直接报错，不用 raw prompt probability 生成错误 importance ratio。快照单独
记录 native/delegated sequences、slots 与 FLOPs，fallback 成本不会从总账本中消失。

### 18.4 原生 suffix proposer

统一 speculation 配置可转换为 vLLM 原生 suffix config。每个 draft 由 target verifier 验证，未接受 draft slots
根据 vLLM metrics 加回 generation slots。vLLM 0.25 的公共接口不提供任意外部 token tree 注入；同一常驻引擎
处理过的历史请求会自然进入 global suffix cache。任意外部经验 proposal 的显式随机残差校正当前由
Transformers 路径实现；二者属于不同实验臂。

<a id="infra-vllm-config"></a>
### 18.5 单卡角色划分与配置

24 GB 单卡同时驻留 1.5B base 与 0.5B proposal 时，可为不同角色设置独立显存比例和 batch 上限：

```toml
[vllm]
asynchronous = true
enable_prefix_caching = true
exact_scoring_backend = "none"

[vllm.base]
gpu_memory_utilization = 0.62
max_num_seqs = 48
max_num_batched_tokens = 12288

[vllm.proposal]
gpu_memory_utilization = 0.28
max_num_seqs = 24
max_num_batched_tokens = 6144

[vllm.engine_kwargs]
enable_chunked_prefill = true
```

公共设置先应用，再由 `base`、`proposal`、`rl` 角色覆盖。支持 TP/DP、量化、LoRA、最大模型长度、最大 batch
token 与 eager mode。多卡、量化或不同 dtype 构成不同硬件设置，与单卡 FP32 结果分开报告。

安装、Linux/WSL2 限制与完整参数见 [vLLM 专题说明](VLLM_RUNTIME.md)。当前正式 RTX 3090 数字来自
Transformers；vLLM 加速结论需要同一模型、dtype、数据哈希和 GPU 数下的成对 benchmark。

<a id="infra-accounting"></a>
## 19. 快照与公平比较

两套 backend 的快照至少记录：

- sampled sequences、generated tokens、prefill tokens；
- shared/cached prefix tokens；
- score calls、scored tokens；
- generation 与 score forward token slots；
- speculative proposed/accepted/rejected slots；
- native/delegated score workload；
- 式 (1) 的 estimated dense forward FLOPs。

GRPO 训练另按 rollout generation、reference scoring、policy forward/backward 等价 slots 和 AdamW adapter 更新估算。
梯度 checkpointing 的 policy 路径计一次 forward、一次 backward 等价和一次重算；详细实现见
[`compute.py`](../../src/inference_scaling/compute.py)。

每个加速数字必须明确分母：

| 优化 | 正确分母 | 不充分的说法 |
| --- | --- | --- |
| 连续批处理 | 同一方法逐 prompt 执行 | “比 Base 快” |
| token tree | 同 workload 无草稿 | “接受率为 30%，所以加速 30%” |
| broker | 丢弃 partial 后重生成 | “保存了 96 token” |
| streaming reward | 整批生成后再提交相同 reward | “异步更快” |
| MH prefetch | 同更新数普通 reward MH | “proposal 数减少” |
| delayed acceptance | 同 proposal 的普通精确 MH | “主模型 FLOPs 加速” |
| warm replay | fresh-only；另列 build | “cache hit 即免费” |
| SMC reuse | 相同 SMC 且关闭 reservoir reuse | “比条件 IS 快” |
| vLLM | 同模型、dtype、GPU 数与 workload 的 Transformers | “vLLM 理论吞吐” |

<a id="infra-benchmark-entry"></a>
## 20. 复现入口

Transformers rollout/算法栈：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\benchmark_rollout_infra.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers.json

.\.venv\Scripts\python experiments\benchmark_is_mh_reuse.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers_is_mh.json
```

正式三 seed IS/MH 消融与汇总：

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

vLLM 对应入口：

```bash
export PYTHONPATH=src
python experiments/benchmark_rollout_infra.py \
  --backend vllm --dtype bfloat16 --section all \
  --output results/infra/rtx3090_vllm.json

python experiments/run_vllm_backend_benchmark.py \
  --config configs/gsm8k_3090_aligned.toml \
  --limit 32 --workers 8 --tag rtx3090
```

汇总脚本会检查 schema、workload 完整性和关键配置。结果目录与报告对应关系见
[`results/README.md`](../../results/README.md)。

<a id="infra-code-index"></a>
## 21. 代码与验证入口

| 内容 | 代码 | 主要测试 |
| --- | --- | --- |
| 连续批处理 | [`batching.py`](../../src/inference_scaling/backends/batching.py) | `tests/test_batching_backend.py` |
| 精确评分缓存 | [`cache.py`](../../src/inference_scaling/backends/cache.py) | `tests/test_score_cache.py` |
| 显式候选缓存 | [`candidate_cache.py`](../../src/inference_scaling/backends/candidate_cache.py) | `tests/test_cached_candidate_backend.py` |
| Transformers KV/评分/speculation | [`transformers_backend.py`](../../src/inference_scaling/backends/transformers_backend.py) | `tests/test_transformers_backend.py` |
| vLLM Async/APC/评分 fallback | [`vllm_backend.py`](../../src/inference_scaling/backends/vllm_backend.py) | `tests/test_vllm_backend.py` |
| vLLM 动态 suffix 适配 | [`vllm_suffix_proposer.py`](../../src/inference_scaling/vllm_suffix_proposer.py) | `tests/test_vllm_suffix_proposer.py` |
| token tree、streaming、run-ahead | [`acceleration.py`](../../src/inference_scaling/acceleration.py) | `tests/test_acceleration.py` |
| 部分 rollout broker | [`rollout_broker.py`](../../src/inference_scaling/rollout_broker.py) | `tests/test_rollout_broker.py` |
| 配置驱动 backend 加载 | [`loader.py`](../../src/inference_scaling/backends/loader.py) | `tests/test_backend_loader.py` |
| token/FLOPs 账本 | [`compute.py`](../../src/inference_scaling/compute.py) | `tests/test_compute_accounting.py` |

工程测试验证请求顺序、概率 shape、seed 隔离、cache key、重复 id、防止部分样本进入估计器以及计数器增量；真实
RTX 3090 报告验证相同 workload 下的 wall time、slots、FLOPs、复用率和冷启动成本。两类证据分别用于判断
实现正确性与硬件收益。
