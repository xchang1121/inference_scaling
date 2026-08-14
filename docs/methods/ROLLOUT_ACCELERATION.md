# rollout 生成、复用与验证优化

这套实现把算法层和执行层分开：算法决定哪些样本可以进入估计量，执行层只负责更快地产生和验证这些
样本。两者必须分开记账，否则很容易把同一条历史轨迹既当作新的统计样本，又当作推测解码草稿，造成
没有被 importance ratio 修正的重复计数。

## 整体数据流

```text
历史 / off-policy rollout ──┬──> replay store ──> 带行为概率的 IS 估计
                            │
                            └──> draft token tree ──> base 模型逐 token 验证

过量提交的 rollout ──> completion broker ──> 完整样本 / 可续跑部分前缀

base 候选 ──> pilot rollout ──> 冻结 evaluation 预算 ──> 独立 evaluation rollout
                                                       │
                                                       ├──> 条件 IS 选择
                                                       └──> SMC block 权重与后缀森林

当前 MH 状态 ──┬──> 普通 / replay 混合后缀 proposal
               ├──> accept/reject 下一状态 proposal 预取
               └──> surrogate 早拒绝 ──> 精确奖励校正
```

draft tree 中的数据不进入最终权重；因此它可以来自已经消费过的 replay、旧策略或后台预生成。进入
replay estimator 的记录则必须保存真实 behavior probability，并遵守原有的生命周期和去重规则。

## 历史 rollout token tree

`RolloutTokenTree` 在 CPU 上保存“最近若干 token → 下一个 token”的有界计数表。生成时从最长匹配
后缀开始查找，并给出最多 $K$ 个草稿 token。确定性模式沿最高频分支前进；随机模式从完整经验条件
分布抽样，并把每一步的 proposal 概率一并交给 verifier。

确定性模式通过“target 抽样 token 是否与草稿相同”决定接受。随机模式使用
`min(1, p(token)/q(token))`，拒绝后从归一化的 `(p-q)_+` 残差抽样，因此同样保持请求所定义的 target
分布，不把历史经验分布冒充 base。Transformers 会把验证后的 `DynamicCache` 裁到已接受位置，从该
KV 状态继续生成，避免拒绝后重新 prefill。vLLM 使用原生 global suffix proposer 和 target verifier，
并直接读取 drafted / accepted token 计数器；任意外部经验分布的随机残差校正目前不注入 vLLM 原生
suffix cache。

在 BF16 下，不同 batch 形状可能经过不同数值 kernel，因而不能要求两条运行逐 token 完全相等；应把
正确性表述为草稿不改变定义的抽样规则，并用 FP32 有限状态测试检查固定随机流。真实硬件报告同时保留
共同前缀比例，避免把数值路径分叉误报成算法错误。

## 按 active batch 调整草稿长度

令 $b$ 为当前 active batch，草稿长度写成分段函数 $K(b)$。大 batch 时普通批处理已有较高算力
利用率，错误草稿会把一次 target 验证扩大为多个无效 token slot；长尾只剩少量请求时，减少串行
decode 轮次才更可能有收益。因此调度器允许例如

```toml
[acceleration.speculation]
enabled = true
tiers = [[1, 8], [4, 0], [512, 0]]
min_context_tokens = 2
min_token_probability = 0.10
tree_max_context_tokens = 24
tree_max_contexts = 100000
vllm_max_cached_requests = 10000
dynamic_vllm = false
stochastic_tree = false
```

其中每一项为 `[最大 active batch, K]`。Transformers 在每批请求进入 verifier 前查询该表；设置
`dynamic_vllm = true` 后，vLLM 才把同一张表转成 `num_speculative_tokens_per_batch_size`。vLLM 当前把
非 EAGLE proposer 的动态表视为较新的路径，且 0.25 上已有并发吞吐与 CUDA graph 兼容性报告，因此
默认关闭动态表，只使用固定最大 $K$；它仍保留为单独消融，不与算法收益混报。vLLM 0.25 原生 suffix
proposer 还要求运行时 $K$ 固定；动态模式因此加载仓库内的
`DynamicSuffixDecodingProposer`，它继续使用官方 suffix cache，只在一次串行 proposer 调用期间传入
调度器选定的 $K$，调用后恢复启动值。

配置中的通用默认值不是硬件最优值。3090 消融发现历史树接受率偏低时，只在 `batch=1` 开启草稿能够
保护吞吐，而始终使用 $K=8$ 明显变慢；部署前应在自己的模型、prompt 分布和 batch 曲线上重新标定。

## 部分 rollout broker

`AsyncRolloutBroker` 把一个长请求拆成有界 token chunk。当过量提交的 batch 已达到所需 completion 数时，
未完成请求不会被丢弃，而是保存原请求、已生成 token、逐 token 行为概率、参考概率、优先级和分段数。
下一轮先调度这些部分轨迹，并从“原 prefix + 已生成 token”继续。

broker 只在 EOS 或完整 token 预算完成后触发 completion callback；部分状态不能保存成 `ReplayRecord`，
也不能进入 IS 估计。它在 Transformers 与 vLLM 上都可用，但保存的是可序列化 token 状态，不承诺跨
权重版本保留引擎 KV。Transformers 恢复时需要重新 prefill；vLLM 若 Automatic Prefix Caching 仍持有
相同 block，则可命中引擎缓存。报告因此同时记录保存/丢弃 token、恢复 prefill、墙钟和主模型 FLOPs。

## pilot 与 evaluation 分离的预算分配

对第 $i$ 个 base 候选前缀 $s_i$，目标条件量为

\[
h(s_i)=\mathbb E_{z\sim p_{\rm base}(\cdot\mid s_i)}
\left[\exp\!\left(\frac{r(s_i,z)}{\tau}\right)\right].
\]

先为每个候选生成少量 pilot rollout，估计单条成本 $c_i$ 和方差尺度；随后在总成本预算内冻结
evaluation 条数 $m_i$，再从 base 条件分布独立生成 evaluation rollout：

\[
\widehat h_i=\frac{1}{m_i}\sum_{j=1}^{m_i}
\exp\!\left(\frac{r(s_i,z_{ij})}{\tau}\right).
\]

pilot reward 只决定 $m_i$，不进入上式。这样可以根据 EOS、序列长度或 replay 成本差异重新分配预算，
同时避免“先看到某条 evaluation 的 reward，再决定是否继续抽该候选”带来的可选停止偏差。实现先用
pilot 的真实 token cost 将每候选 evaluation 条数换成冻结成本，再由独立请求执行。
`evaluation_reference_rollouts_per_candidate` 仅定义与“每候选固定若干条”对照相同的总估计成本，实际
条数仍可在候选之间移动；它只用于受控消融。

若每个保留候选的 $m_i\to\infty$，大数定律给出

\[
\widehat h_i\xrightarrow{p}h(s_i).
\]

候选数有限时，归一化权重和最终选择概率由连续映射定理同时收敛。这里的论证依赖 pilot 与 evaluation
独立；若把 pilot 值再次并入最终均值，就需要额外处理数据依赖，当前实现明确不这样做。

## 流式 frozen-design IS 与低优先级 run-ahead

异步后端在每条序列完成时触发 callback，`StreamingRewardEvaluator` 立即把该序列交给 CPU reward
线程池，不必等待同批最长序列完成。GPU 生成结束后若 reward future 尚未完成，空出的时间可提交少量
run-ahead 请求。

`FrozenStreamingISEstimator` 在 fresh 生成前冻结每个候选允许进入估计量的 request id。历史贡献可以
在冻结前立即加入；冻结后，fresh completion 无论以何种顺序完成，都只按固定 id 进入一次。最终 log
energy、ESS 和选择概率只由这组固定贡献决定，因此流式完成改变的是等待与排队，不是统计样本集合。
估计器内部使用线程安全更新，可直接接收并行 verifier 的完成回调。

`LowPriorityRunAheadBackend` 遵守三条规则：

- 后台结果只写入 draft tree，不进入当前 evaluation estimator；
- 请求按小 token chunk 执行，前台到达后最多等待当前 chunk，下一 chunk 必须让路；
- 队列有界，后台生成量、前台等待和最终 drain 都单独记账。

因此 run-ahead 不是免费计算。若 reward 几乎没有 CPU 尾部，后台工作反而会增加争用；默认值
`run_ahead_rollouts_per_candidate = 0`，只有测到稳定的 reward / KV 空泡后才开启。

## 奖励目标 MH 的三种执行优化

### Proposal-tree 预取

普通 reward MH 先生成本步 proposal，再等待精确奖励，然后才知道下一状态。预取版在等待奖励期间，
分别以“本步拒绝”和“本步接受”为下一状态，一次 batch 生成下一步两个 proposal。普通 Hastings 判断
完成后只消费对应分支；另一分支明确计为 unused prefetch。proposal seed、cut 和接受随机数沿用普通链
的命名，因此有限状态后端上可以逐字段核对两条路径。

它没有减少算法更新数，并且除最后一步外每步多生成一个最终不用的 proposal。只有精确奖励足够慢、
额外 proposal batch 能被该延迟覆盖时，墙钟才可能下降。

### Delayed acceptance

第一阶段用固定的便宜 surrogate reward 构造完整 proposal 接受比。若第一阶段拒绝，就不调用精确奖励；
若通过，再用“精确奖励差减 surrogate 奖励差”做第二阶段判断。两阶段相乘恢复普通精确目标的接受概率，
因此 surrogate 只影响计算发生的位置，不改变最终 target。实现不裁剪任何接受比，并分别记录 surrogate、
精确奖励和 early rejection 数量。

### 冻结 replay 混合 proposal

`FrozenReplaySuffixProposal` 对每个“保留前缀、后缀长度”使用

```text
(1 - beta) * base suffix probability + beta * frozen empirical probability.
```

`beta < 1` 保留 base 的完整支持集。抽到历史后缀时仍由 base 模型精确评分；每次 MH 更新对新后缀和旧
后缀都计算同一个混合 proposal 概率，把正反向概率放入 Hastings ratio。历史库在链开始前冻结，链内
不能边看接受结果边改变 proposal。历史命中可以把自回归生成变成 teacher-forced 评分，但不保证减少
逻辑 FLOPs。

## SMC rollout forest

SMC 版不在每个 block 后丢掉所有 lookahead。粒子 $s$ 先从 base 产生候选 block $a$，并用 rollout
估计 $h(sa)$。该分支的增量权重为

\[
G(s,a)=\frac{\widehat h(sa)}{\widehat h(s)}.
\]

沿一条祖先链相乘时，中间项相消：

\[
\prod_t G(s_{t-1},a_t)=\widehat h(s_T),
\]

所以粒子滤波是在 base 序列概率上逐 block 加入最终奖励的条件期望，而不是每层重复乘完整奖励。
系统重采样后，与被选 block 前缀匹配的旧 rollout 去掉该 block，余下后缀仍是对应子前缀下的 base
条件 rollout，可以带到下一层；数量不足时再 fresh top-up。

一次有限 rollout 不能被复制成多个独立观测。若同一分支在重采样中出现多次，实现会把 reservoir
轮流分给这些副本，而不是把整份 reservoir 复制多份。终止粒子直接计算终局 reward。报告分别记录
`fresh_rollouts` 和 `reused_rollouts`，可以用 `reuse_rollout_forest=false` 得到同算法、同粒子数的直接
消融分母。

有限 rollout 下，
\(\widehat h(sa)/\widehat h(s)\) 是随机比值，因此这是渐近一致的 particle approximation，不宣称有限
样本无偏。随着每个有效分支的 rollout 数和粒子数增长，条件均值一致收敛，增量权重收敛到
\(h(sa)/h(s)\)，标准有限层 SMC 递推便收敛到相应的 reward-tilted 序列分布。工程上应同时监控 ESS、
reservoir 命中率和 fresh top-up；只看复用率可能掩盖粒子退化。

## 两套后端的能力对应

| 优化 | Transformers | vLLM |
| --- | --- | --- |
| 历史 token tree | 确定性/随机经验 proposal + 显式 target 验证 | 原生 global suffix proposer |
| 拒绝后的 KV 续算 | 裁剪 `DynamicCache` 后继续 | 引擎内部 verifier / cache 管理 |
| active-batch 调度 | 每次 `sample_batch` 查询 $K(b)$ | 动态 batch table + suffix 兼容适配层 |
| 序列完成回调 | batch fallback；按返回顺序触发 | 常驻 `AsyncLLM` 请求完成即回调 |
| 部分 rollout 续跑 | token 状态恢复并重新 prefill | token 状态恢复；前缀可由 APC 命中 |
| streaming IS / progressive / SMC | 后端无关算法层 | 后端无关算法层 |
| MH 预取 / delayed acceptance | 后端无关调度与接受判断 | 后端无关调度与接受判断 |
| replay 混合 proposal | 生成或 teacher-forced 精确评分 | 需要配置可精确评分的后端 |
| 原生 draft 指标 | 仓库计数 | vLLM metrics drafted / accepted counters |

vLLM 的 global suffix 与动态调度由引擎内部管理，因此 Python 侧不会默认再构建一份
`RolloutTokenTree`；历史请求通过常驻引擎自然进入原生 cache。Transformers 路径显式暴露 token tree，
便于验证概率、KV 与 FLOPs 账本。vLLM 0.25 没有把任意外部 token 序列直接插入原生 suffix cache 的
公开接口：同一 base 引擎已经完成的历史请求可以用于草稿，来自另一模型且从未经过 base 引擎的离线
off-policy 序列仍可进入带概率修正的 replay estimator 或冻结 MH 混合 proposal，但不会被误写成已有
的 vLLM 原生草稿加速。

## 复现实测

RTX 3090 的三随机种子结果、图表、计算量口径和限制见
[推理基础设施优化汇总](../reports/RTX3090_ROLLOUT_INFRA.md)。同一入口支持两套后端：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\benchmark_rollout_infra.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers.json
```

新增的 IS/MH 复用消融使用独立入口：

```powershell
$env:PYTHONPATH = "src;."
.\.venv\Scripts\python experiments\benchmark_is_mh_reuse.py `
  --backend transformers --dtype bfloat16 --section all `
  --output results\infra\rtx3090_transformers_is_mh.json
```

```bash
export PYTHONPATH=src
python experiments/benchmark_rollout_infra.py \
  --backend vllm --dtype bfloat16 --section all \
  --output results/infra/rtx3090_vllm.json
```

两侧必须使用同一模型、dtype、数据哈希、长度和预算。墙钟比较要明确分母；主模型逻辑 FLOPs 按实际
target forward token slot 估算，并把 vLLM 被拒绝草稿所占的验证 slot 加回。cache build、在线路径和
后台 drain 始终分列，不能用其中一项代替端到端成本。
