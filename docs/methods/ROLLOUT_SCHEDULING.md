# 条件 IS 的 rollout 调度

请求准入只调整已确定采样树的执行顺序。M/K/B、请求编号与种子、长度上限、奖励和选择规则由算法决定。
异步 vLLM 与 CPU 模拟共用准入队列。默认两阶段探索和提前选择会改变有限预算结果，其统计条件与调度等价性分别说明。

## 请求释放与完成

1. 保留候选的前缀已知，其额外 K−1 条补全在本轮开始时提交；保留补全的奖励直接复用。
2. 新候选使用一条完整原生请求。K>1 时，引擎增量首次覆盖 B 个 token 后提交 K−1 条分支；原请求继续产生第零条补全，随机流不中断。K=1 无额外分支。
3. B 以内结束的候选只有一条空补全；`block_first=true` 沿用先生成块、再续写的请求与种子规则。
4. 完成的叶子按 `(candidate, rollout)` 槽位记录奖励。关闭 `selection_ready` 时，全部规定叶子齐全后才按槽位顺序计算权重并选择。

`selection_ready=true` 以缓存前缀估计未完成权重，选出已完整候选后取消其余请求。首轮至少需要两组全部完成；
末轮有完整参考序列，可在新结果完成前选择。完成时间和前缀估计会引入偏差，详见[提前选择](ALGORITHMS.md#is-selection-ready)。

`two_stage` 把首轮之后的探索集中为一个完整后缀池，用未选 rollout 的权重差异决定执行末池或复用已有序列，
并在提交前确定候选数与辅助长度区间。区间外权重为零，截停前缀无需评分；所有已执行工作照常计费。
公式见[两阶段探索](ALGORITHMS.md#is-two-stage)与[长度区间](ALGORITHMS.md#is-length-window)。

## 资源限制与长尾

`ar.engine.rollout_scheduler` 由同一引擎的全部调用者共用：

| 机制 | 准入规则 |
| --- | --- |
| 并发与预填充 | 原生请求数受 `max_in_flight` 限制；尚未产生首 token 的请求累计完整提示长度，受 `max_prefill_tokens` 限制。超长提示允许独占预填充预算 |
| KV 水位 | 达到水位后暂停新准入，已准入请求继续解码；空闲时允许一条进入。vLLM 负责实际 KV 分配和抢占 |
| 公平与长度排序 | `fair_lpt` 先平衡各题活动请求数，再推进候选前缀，同类请求按预计长度从长到短排列；预测只用长度上限与已完成请求历史 |
| 等待老化 | 等待过久的任务优先；老任务需要较大预填充预算时停止准入小任务，等待额度释放 |
| 尾部保护 | 一轮分支已全部释放、队列已空且仅余至多两条活动请求时，暂缓其他题的长预填充；等待老化可解除此保护 |

`fifo` 用作排序对照。准入队列不抢占样本；收益来自提前释放分支和协调资源，单条自回归长尾仍需顺序完成。
默认 K=1 没有额外分支，收益主要来自输出处理、跨题准入与资源保护。单并发且工作量固定时，排序本身没有并行收益。

<a id="compact-output"></a>
## 输出交付与概率处理

vLLM 使用 DELTA 输出，`stream_interval=32`：首 token 立即交付并释放预填充额度，其余增量按间隔交付，完成时交付余量。
普通批量调用也在内部消费这些增量，外层一次返回完整结果。

| 路径 | 交付给算法的事件 |
| --- | --- |
| 完整池、K=1 | 完成结果 |
| 完整池、K>1 | 新候选增加一次 B-token 前缀事件，随后只交付完成结果 |
| 提前选择 | 另交付中间增量，用于前缀权重估计与取消 |
| 多 token 思考边界 | 保留中间事件以检测边界；单 token 标记使用引擎原生停止 |

前缀事件最多因分批多等待 `stream_interval−1` 个 token（排队另计），切点仍取前 B 个 token。
K=1 且 `block_first=false` 时，候选提交完即可声明本轮不再产生分支，及时启用尾部保护。

`flat_logprobs=true` 直接读取扁平概率数组，省去逐 token 字典与对象；Consilience 保留原 top-K 集合及 `fsum` 归约。
Self-Certainty 的 Ascend 路径读取随生成返回的全词表 KL 标量，详见[概率统计](ALGORITHMS.md#infra-fused-logprobs)。
活跃序列使用增量数组和前缀视图，结束后保留紧凑副本，避免每次查询都转换整条序列。

同一请求尚未消费的事件合并为一份通知；奖励可向同一引擎提交评分请求。
选择完成后，并发取消其余请求并等待资源释放。语义停止只取消相应请求；错误路径完成取消和额度清理后报错。
越过多 token 边界的额外解码计入成本，逻辑输出截在原边界。

`stream_outputs` 计引擎增量，`stream_events` 计外层事件。`engine_execution` 保存生效的输出间隔、内部异步调度和执行图配置；
`asynchronous=true` 仅表示异步接口，执行图配置也仅表示可用设置，实际命中需单独观测。

GPU `fused_logprobs=true` 在请求结束后提供统计量，使用批次算法和共享准入，须关闭提前选择。
同步 vLLM、Transformers 使用各自批次路径。运行记录的 `rollout_scheduler=streaming/batch` 表示实际路径。

实现索引：[候选树](../../src/inference_scaling/arllm/algorithms/candidates.py)、
[准入队列](../../src/inference_scaling/arllm/backends/scheduling.py)、
[事件合并](../../src/inference_scaling/arllm/backends/streaming.py)、
[vLLM 接口](../../src/inference_scaling/arllm/backends/vllm_backend.py)、
[统计缓存](../../src/inference_scaling/arllm/backends/statistics.py)。
公共接口：[DELTA](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/sampling_params.py)、
[generate/abort](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/engine/async_llm.py)、
[FlatLogprobs](https://docs.vllm.ai/en/v0.26.0/api/vllm/logprobs/)。

<a id="默认调度的执行等价性"></a>
## 完整候选池下的执行等价性

以下论证要求 `selection_ready=false`、无墙钟截止，且奖励是序列的固定函数。
为每个逻辑请求固定独立随机流，为选择固定独立随机数。给定同一轮状态：

1. 请求的条件生成分布、随机流与停止规则相同；第零条请求在 B 处继续运行，分支只在前缀确定后提交。
2. 调度可以依赖长度、完成情况和资源指标，但全部规定请求最终参与计算。
3. 每个叶子的序列和奖励相同，候选的平均权重、候选及补全的归一化概率相同。
4. 按固定槽位归约并使用同一选择随机数，得到相同保留序列；逐轮归纳得到相同输出。

因此，固定工作量下的调度保持原有限步算法的输出分布，原条件核的平稳目标也保留。
从基础模型初始化仍有有限步误差；默认提前选择及显式墙钟限制不满足上述全部条件。
CPU 精确耦合验证的是相同底层采样律，真实设备的不同批形状仍可能通过浮点舍入改变轨迹。

## 验证

[CPU 报告](../reports/ROLLOUT_SCHEDULING_CPU.md)核对实际算法路径、有限状态转移矩阵与虚拟时延；
[昇腾报告](../reports/ASCEND_IS_INFRA.md)记录单卡、双卡的真实成本及适用范围。
