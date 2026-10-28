# 设置文件

推理与训练各有一个固定的设置文件，从仓库根目录读取：

| 文件 | 读取者 |
| --- | --- |
| `settings/inference.json` | `python -m inference_scaling`（命令行只选择算法、模型族、奖励和数据集） |
| `settings/training.json` | `python -m training`（没有命令行参数） |
| `settings/serve.json` | `python -m inference_scaling.serve`（另读 `settings/inference.json` 的 `ar` 与 `rewards.consilience`） |

两个文件都按代码中的 schema 严格校验（[`app/settings.py`](../src/inference_scaling/app/settings.py)、
[`training/settings.py`](../training/settings.py)）：缺少字段、多出字段或类型不符都会在加载模型前报错。
所有行为参数都写在文件里，代码中不再有隐藏的默认值；标注为“对象”的字段原样传给对应库（如 vLLM、TRL）。
下表中的数值是仓库提供的文件中的取值。

一次推理只读取与所选组合相关的部分：`run.seed`、`datasets.<数据集>`、`rewards.<奖励>`（仅使用奖励的算法）、
`<模型族>` 中除 `algorithms` 外的全部字段，以及 `<模型族>.algorithms.<算法>`。这些“有效设置”连同模型与数据的
哈希、包源码哈希一起决定结果目录的指纹；修改无关部分（例如另一个模型族）不会使已有结果失效，
`run.draws` 与 `run.hash_cache_dir` 也不参与指纹。

## `settings/inference.json`

### `run`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `seed` | 整数 | 所有随机流的根种子；每题每次抽取的种子由它、抽取序号、算法名和题目 ID 派生 |
| `draws` | 整数 | 每题独立重复次数；大于 1 时汇总报告 pass@k。增加它会在同一目录续跑 |
| `hash_cache_dir` | 字符串 | 大文件 SHA-256 缓存目录（按路径、大小和修改时间失效） |

### `datasets.gsm8k`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `path` | 字符串 | 本地 JSONL 路径 |
| `download` | 布尔 | 文件缺失或校验不符时从 `source.url` 下载 |
| `source.url` / `source.sha256` / `source.rows` | 字符串 / 字符串 / 整数 | 官方拆分的地址、SHA-256 与行数；三者都会校验 |
| `selection.count` | 整数或 `null` | 在推理前按种子抽取的题数（保持数据集顺序）；`null` 为全部 |
| `selection.seed` | 整数 | 抽题种子 |
| `prompt_template` | 字符串 | 用户消息模板，`{question}` 恰好出现一次；其余字符（包括 LaTeX 花括号）原样保留 |
| `max_new_tokens` | 整数 | 每条输出的生成上限（AR 还受上下文长度约束；dLLM 还受 `dllm.max_new_tokens` 约束并取块长整数倍） |

评测取最终答案文本中的数值（`####`、`\boxed{}`、“answer is”，否则最后一个数），与参考值按分数比较。

### `datasets.math500`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `path` / `download` | 字符串 / 布尔 | 本地文件；缺失时从固定版本下载 |
| `source.repository` / `source.revision` / `source.filename` | 字符串 | Hugging Face 数据集、提交与文件名 |
| `selection.seed` | 整数 | 分层顺序的种子：按学科×难度分组，组内与组间打乱后轮流取题，与答案无关 |
| `selection.minimum_level` | 整数 | 只保留不低于该难度的题 |
| `selection.excluded_ids` | 字符串数组 | 事先排除的题目 ID（如开发时看过的题） |
| `selection.skip` / `selection.count` | 整数 | 跳过顺序中的前 `skip` 题（留作开发），评测随后的 `count` 题 |
| `prompt_template` / `max_new_tokens` | 字符串 / 整数 | 同 GSM8K |
| `judge_timeout_seconds` | 数 | Math-Verify 工作进程单次判定的超时 |

载入时会检查每道题的参考答案能被判定器识别。

### `rewards`

奖励 $`r(x,y)`$ 是逐序列的固定分数；温度 $`\tau`$ 定义目标 $`p(y\mid x)\exp\{r/\tau\}`$。
`best_of_n` 只用奖励排序，因此不读取温度。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `verifier.temperature` | 数 | verifier 的温度，两种来源共用 |
| `verifier.source` | `dataset` \| `vote` | `dataset`：数据集判定器对照参考答案（oracle），正确为 1，否则为 0；数据集没有参考答案时改为 `vote`。`vote`：与模型自身答案一致的比例，不读参考答案 |
| `verifier.pool_size` | 整数 | 投票时 `is`/`mh` 的冻结样本池大小；奖励为池中与该答案一致的比例。`best_of_n` 不用池：候选之间投票，得票最多的答案胜出，平票在最高票候选中按种子随机选一个 |
| `self_certainty.temperature` / `score_temperature` / `scope` | 数 / 数 / `thinking` \| `full` | 奖励温度；计算分布所用的温度（1 为模型原始分布）；评分范围（原始定义为 `full`）。奖励为评分段上各位置的分布相对全词表均匀分布的 KL 散度的均值；要用整个词表的概率，vLLM 引擎需设 `ar.engine.vllm.fused_logprobs = true`（评分温度 1）或 `exact_scoring = "transformers"` |
| `consilience.temperature` / `score_temperature` | 数 | 奖励温度；计算 top-$`K`$ 置信度所用的温度 |
| `consilience.scope` | `thinking` \| `full` | 只评思考段（缺少完整思考段时回退到全序列并记录原因）或评全序列 |
| `consilience.top_k` / `window_fraction` / `window_tokens` / `skip_fraction` / `initial_penalty` | 整数 / 数 / 整数或 `null` / 数 / 数 | 置信度窗口：跳过开头 `skip_fraction`，首段与末段各取 `window_fraction`（或固定 `window_tokens`），分数为末段均值减 `initial_penalty` 倍首段均值 |

`self_certainty` 与 `consilience` 读取自回归模型的概率，只用于 `--model ar`。

### `ar.model`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `path` | 字符串 | 本地目录或 Hub 模型 ID |
| `revision` | 字符串或 `null` | 固定的 Hub 提交 |
| `weight_sha256` | 对象或 `null` | 权重文件名到 SHA-256 的对象；非空时必须恰好列出检查点的全部权重文件并逐个校验（下载、训练与 `dllm.model` 用同一格式） |
| `adapter` | `null` 或 `{path, revision}` | 叠加在基础模型上的 PEFT 适配器（如 GRPO 训练结果） |
| `tokenizer` / `tokenizer_revision` / `tokenizer_kwargs` | 字符串或 `null` / 字符串或 `null` / 对象 | 独立的 tokenizer 及其参数 |
| `cache_dir` / `local_files_only` / `trust_remote_code` | 字符串或 `null` / 布尔 / 布尔 | Hub 缓存与加载选项 |
| `token_penalty` | `null` 或 `{words, strength}` | 每个词前有空格的小写与首字母大写形式、以及行首的首字母大写形式（只取单个 token 的形式），其 logit 在温度之前减去 `strength`；惩罚后的分布就是模型本身，采样、参考概率与评分都用它（见[算法说明](methods/ALGORITHMS.md#token-penalty)）。vLLM 的原生评分与 `fused_logprobs` 读未惩罚的 logit，所以评分需要 `exact_scoring = "transformers"`，同步引擎也不能做 beam search；`null` 关闭 |

### `ar.engine`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `backend` | `transformers` \| `vllm` | 执行引擎 |
| `device` / `dtype` | 字符串 | Transformers 设备与精度；vLLM 使用 `dtype`，精确评分后端同时使用两者 |
| `context_window` | 整数或 `null` | 额外的上下文上限；实际生成上限取它、模型上下文与 `max_new_tokens` 的最小值 |
| `transformers.attn_implementation` / `device_map` / `model_kwargs` | 字符串或 `null` / 字符串、对象或 `null` / 对象 | Transformers 加载选项 |
| `transformers.max_score_batch_size` | 整数 | 一个评分批次的最大序列数；批次的填充位置数也不超过它乘以 `score_chunk_size` |
| `transformers.score_chunk_size` | 整数 | 长序列评分与前缀预填充的分块长度 |
| `transformers.prefix_cache_mib` | 整数 | 前缀 KV 存储的容量（MiB，`0` 关闭）：结束的生成行保存 KV，后续请求从最长的已存前缀接着预填充；结果不变 |
| `transformers.in_place_kv` | 布尔 | 解码时把新位置写入预留缓冲区，而不是每步复制整层 KV；结果不变 |
| `transformers.cuda_graphs` | 布尔 | 解码步在固定形状上执行：KV 放入常驻缓冲区，行数与关注的位置数取整到 2 的幂，其余部分由掩码屏蔽；CUDA 上每种形状的第一步捕获为 CUDA graph，之后重放（CPU 上走同一路径但不捕获）。结果与动态解码相同，至多差浮点舍入；缓冲区保留到见过的最大批。需要模型在单一设备上、注意力为 `sdpa` 或 `eager` |
| `vllm.asynchronous` | 布尔 | 异步引擎（原生连续批处理）或同步引擎 |
| `vllm.async_scheduling` | 布尔或 `null` | 引擎内部的 CPU/GPU 异步调度，与 `asynchronous` 接口选项分开；默认 `null`，由 vLLM 按模型和设备决定。融合统计模式要求 `false` 或 `null`，实际生效值记录在 `engine_execution` |
| `vllm.stream_interval` | 正整数 | 默认 `32`；vLLM 保留首 token 与完成通知，其余增量按此 token 间隔交付，减少 Python 输出开销。K>1 的前缀通知因分批最多额外等待 `stream_interval−1` 个 token（队列等待另计）；较短的块可设为 `1`。多 token 停止标记仍由外层增量检测，可能产生边界后的额外解码 |
| `vllm.flat_logprobs` | 布尔 | 默认 `true`；直接读取 vLLM 的扁平概率数组，避开逐 token 概率对象，Consilience 的 top-K 归约公式不变；`false` 使用原字典格式作对照 |
| `vllm.tensor_parallel_size` / `data_parallel_size` / `gpu_memory_utilization` / `max_model_len` / `max_num_seqs` / `max_num_batched_tokens` / `quantization` / `enforce_eager` / `max_lora_rank` | — | 对应 vLLM 引擎参数 |
| `vllm.enable_prefix_caching` | 布尔 | 前缀缓存 |
| `vllm.fused_logprobs` | 布尔 | 在采样的同一步 logits 上另读出温度 1 的模型下选中 token 的对数概率与分布相对均匀分布的 KL 散度（见[算法说明](methods/ALGORITHMS.md#infra-fused-logprobs)）：`mh_power` 省去基础模型重评分，`self_certainty` 在生成时记录。支持 NVIDIA 的同步和异步接口；需要 vLLM 0.26.x，关闭 vLLM 的内部异步调度，不能与 `token_penalty` 同用；Ascend 不支持此融合 worker |
| `vllm.exact_scoring` | `none` \| `transformers` | 用同一份权重的 Transformers 副本精确评分（生成时读不到的 Consilience 统计量与其他策略的对数概率需要） |
| `vllm.parameter_count` | 整数或 `null` | 计算量统计用的参数量；`null` 时从本地 safetensors 读取，跳过纯文本生成不运行的视觉编码器与多 token 预测（MTP）头 |
| `vllm.engine_kwargs` | 对象 | 其他引擎参数；不能覆盖上述字段，也不能开启 speculative decoding 或关闭引擎统计（`disable_log_stats`，抢占计数依赖它） |
| `continuous_batching.workers` | 整数 | 同时求解的题数；大于 1 时 `is` 的后端按请求归属前向 token 开销，逐题成本仍然保留，其余算法的逐题成本为 `null` |
| `continuous_batching.max_batch_size` / `max_batch_tokens` / `batch_wait_seconds` | 整数 / 整数 / 数 | 合并批的上限与等待时间 |
| `rollout_scheduler.enabled` | 布尔 | 默认 `true`；异步 vLLM 共用一个准入队列，IS 在候选前缀就绪时释放分支。`fused_logprobs = true` 时保留批次生成、仍限制准入；同步 vLLM 与 Transformers 保留原路径。实际路径记录为 `rollout_scheduler = streaming/batch`；详见[调度说明](methods/ROLLOUT_SCHEDULING.md) |
| `rollout_scheduler.max_in_flight` | 正整数 | 默认 `32`；所有调用者共用的原生请求上限，加载时再取它与非空 `vllm.max_num_seqs` 的较小值 |
| `rollout_scheduler.max_prefill_tokens` | 正整数 | 默认 `16384`；尚未产生首 token 的准入请求，其完整提示长度之和的保守预算；超大提示可单独预填充，避免永久饥饿 |
| `rollout_scheduler.kv_high_watermark` | `(0, 1]` | 默认 `0.9`；KV 使用率达到水位后暂缓新请求。它是软限，正在解码的请求继续；空闲时允许一条请求进入，避免陈旧指标阻止进展 |
| `rollout_scheduler.policy` | `fair_lpt` \| `fifo` | 默认 `fair_lpt`：优先等待过久的任务，否则先平衡各题活动请求数、再推进候选前缀，再按预计剩余长度从长到短准入；预测只使用长度上限与已完成请求的长度历史 |
| `rollout_scheduler.aging_seconds` | 正数 | 默认 `2.0`；等待超过此时间后按等待先后准入，并允许突破长尾预填充保护；不突破并发上限或丢弃样本 |
| `rollout_scheduler.tail_prefill_tokens` | 正整数 | 默认 `4096`；一轮已释放所有分支、没有待准入请求且仅余至多两条活动请求时，暂缓其他题超过此提示长度的新预填充，直到尾部完成或待办老化 |

生成概率固定使用 `processed_logprobs`，包含采样温度、词表惩罚及 top-k/top-p 的处理。
Ascend 后端在 V1 worker 初始化时同步两级采样器的概率模式，保留原生生成路径和自定义 `worker_extension_cls`。
该适配要求 V1 `AscendSampler` 且 `enable_reduce_sample=false`；不兼容的配置显式报错。
提示词概率仍为原始模型概率，其他策略的精确重评分沿用 `exact_scoring` 设置。

### `ar.prompt`、`ar.output`、`ar.sampling`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `prompt.system` | 字符串或 `null` | 系统消息 |
| `prompt.format` | `auto` \| `chat` \| `plain` | 使用 chat template、强制使用，或直接用用户文本 |
| `prompt.chat_template_kwargs` | 对象 | 传给 chat template 的参数（如 Qwen3.8 的 `reasoning_effort`：`low`、`medium`、`xhigh`）；`thinking_mode` 不为 `auto` 时模板的 `enable_thinking` 随之设置 |
| `output.thinking_mode` | `auto` \| `enabled` \| `disabled` | 思考模式；`enabled` 时未完成的思考没有最终答案（评测文本为空） |
| `output.thinking_start_text` / `thinking_end_text` / `starts_in_thinking` | 字符串或 `null` / 字符串或 `null` / 布尔或 `null` | 显式的思考段标记；为 `null` 时从 tokenizer 词表与 chat template 识别 |
| `output.sampling_scope` | `full` \| `thinking` | `mh`、`mh_power`、`is` 在完整输出或思考段上采样；思考段结束后由基础模型生成最终内容。读取答案文本的 `verifier` 会回退到 `full` 并记录原因 |
| `sampling.temperature` / `top_p` / `top_k` | 数 / 数 / 整数或 `null` | 基础策略。`mh`、`mh_power`、`is` 的目标需要完整支持集（`top_p = 1`、`top_k = null`） |

### `ar.algorithms`

| 字段 | 含义 |
| --- | --- |
| `sample`、`greedy` | 无参数：单次采样；贪心解码 |
| `beam.num_beams` | beam 宽度 |
| `best_of_n.samples` | 候选数 |
| `mh_power.alpha` | 幂目标 $`p^\alpha`$ 的指数 |
| `mh_power.proposal_temperature` | 后缀 proposal 相对基础策略的温度（常用 $`1/\alpha`$） |
| `mh_power.block_size` / `steps_per_block` / `iterations` | 分段延长的块长与每段更新数；`iterations` 非空时在完整长度上做固定次数更新 |
| `mh_power.suffix_schedule` | `uniform` \| `inverse_length` \| `multiscale` \| `log_uniform`；分别为均匀、长度倒数、偏向 2 的幂长度、按 2 的幂区间等量分配概率 |
| `mh.block_size` / `steps_per_block` / `iterations` / `suffix_schedule` | 同上，目标为 $`p\exp\{r/\tau\}`$ |
| `mh.ensemble_size` | 默认 `1`，保持单链 MH。大于 `1` 时保留该数量的完整序列，每轮更新全部候选的后缀，最后按权重抽样；要求显式设置 `iterations`、`proposal = "base"`、`suffix_replay = false`。`iterations` 是整池更新轮数，总生成阶段数为 `1 + iterations`，此时 `block_size` 与 `steps_per_block` 不分配更新预算。见[完整候选池](methods/ALGORITHMS.md#ensemble-mh) |
| `mh.execution` | 默认 `batch`，逐轮批量生成。`pipeline` 在流式后端中重叠已确定源状态的跨轮请求，保持相同逻辑更新；要求 `ensemble_size > 1`、`initialization = "iid"`，关闭 `stratified_updates` 和 `nonself_updates`。非流式后端继续使用 `batch`，实际方式写入 trace；[调度条件与收益来源](methods/ALGORITHMS.md#ensemble-pipeline) |
| `mh.initialization` | 默认 `iid`，首轮候选独立采样。`stratified` 将序列累计概率区间等分，每层独立生成一条，再随机排列槽位；后续 MH 不变。要求 `ensemble_size > 1` 和 vLLM Ascend 后端，其他后端显式拒绝；样本均值方差不增、候选覆盖概率不降，不保证每例输出误差或耗时更低。见[分层初始化](methods/ALGORITHMS.md#ensemble-initial-strata) |
| `mh.stratified_updates` | 默认 `false`。要求 `initialization = "stratified"` 和 `ensemble_size > 1`；每层固定一个槽位，后续从同层整条重生成，或从区间约束已解除的前缀开始局部更新，使用修正后的切点概率。最终输出的平稳目标不变，有限预算误差与耗时可能增减。见[持续分层](methods/ALGORITHMS.md#ensemble-persistent-strata) |
| `mh.nonself_updates` | 默认 `false`。要求 `ensemble_size > 1`、`stratified_updates = false` 及 vLLM Ascend 采样适配；后缀 proposal 排除原序列，直接抽取第一次分歧，使用正反条件概率修正接受率。可与首轮分层结合，保持同一平稳目标；[算法及有限预算边界](methods/ALGORITHMS.md#ensemble-nonself) |
| `mh.proposal` | `base`（基础策略后缀）或 `frozen_history`（冻结历史混合 proposal） |
| `mh.frozen_history.samples` / `mixture` | 历史样本数；从历史后缀提议的概率 |
| `mh_power.early_rejection` | 先抽接受用的均匀数，proposal 一旦不可能被接受就停止生成（见[算法说明](methods/ALGORITHMS.md#mh-early-rejection)）：链与关闭时逐步相同，被拒 proposal 的尾部不再生成；要求 `proposal_temperature` 在 $`[1/\alpha,1]`$ 内，需要 `ar.engine.backend = transformers` |
| `mh.suffix_replay` / `mh_power.suffix_replay` | 把当前后缀作为草稿交给后端重放（见[算法说明](methods/ALGORITHMS.md#mh-suffix-replay)）：proposal 与关闭时逐 token 相同，与当前后缀相同的开头部分不调用模型；需要 `ar.engine.backend = transformers` |
| `is.candidate_count` / `rollout_count` / `block_size` | 初始候选数 M、每候选补全数 K 与块长 B；默认全程固定，块长在输出上限处截短 |
| `is.exploration_schedule` | 默认 `fixed`。`single_pool` 完成首轮 M/K/B 候选池，作一次 Metropolized 索引选择后直接提交完整序列；`two_stage` 按首轮未选 rollout 的长度与权重分配末轮探索，并使用容量限制及长度区间。后两种模式与同块补全复用、数量配额、超发互斥；单轮可另启用权重配额并使用普通 SIR。减少探索可能增大有限预算误差。见[单轮候选池](methods/ALGORITHMS.md#is-single-pool)与[两阶段探索](methods/ALGORITHMS.md#is-two-stage) |
| `is.metropolized_blocks` | 默认 `false`。在完整池中合并 token 块相同的候选，先更新块索引，再按原权重抽取补全。支持 `fixed`、`two_stage`、`single_pool`，与兄弟复用、数量/权重配额、提前选择、末条筛选和超发互斥。不增加当前候选池的生成或评分，保持原平稳目标；后续生成量和有限预算误差可能增减。见[按候选块选择](methods/ALGORITHMS.md#is-block-selection)；基准入口对应 `--metropolized-blocks` |
| `is.weighted_completion_quota` | 默认 `false`。单轮中按完整奖励和缓存前缀分数估计已完成权重比例，达到 `1-1/M` 后取消剩余请求；至少两个组有完整结果，M=1 时为一个。完整叶子按原定补全数加权作普通 SIR。需要流式后端和缓存前缀分数，与先块生成、末条筛选及其他配额/复用选项互斥。存在完成时间及前缀分数相关偏差；见[权重完成配额](methods/ALGORITHMS.md#is-weighted-completion-quota) |
| `is.selection_ready` | 默认 `false`。`single_pool` 固定一次抽样随机数，在所选叶子完整时结束；K>1 分叉由首条根请求结束触发。`two_stage` 首轮保持预设 B，以组权重除以固定独立均匀随机数确定优先级，只允许收齐原定补全的组入选；末轮沿用原 SIR 随机数和长度区间。未完成权重使用缓存前缀估计，存在选择偏差。需要流式后端，与其他早停、先块生成及复用选项互斥。完成条件及回退见[算法说明](methods/ALGORITHMS.md#is-selection-ready) |
| `is.sibling_replay` | 默认 `false`。复用所选块的其他完整补全及奖励，替代下一轮部分新候选根；K=1 无额外可复用样本。见[同块补全复用](methods/ALGORITHMS.md#is-sibling-replay) |
| `is.completion_fraction` | 默认 `1.0`，每个候选收齐所需补全。设为 `(0,1)` 后，各候选的平均完成比例达到该值即取消未完成请求；至少取得一条新结果，未完成补全权重为零，保留原定 K 分母。需要流式后端，会引入长度相关偏差。见[完成配额](methods/ALGORITHMS.md#is-completion-quota) |
| `is.rollout_race` | 默认 `false`。每候选至多发出双份缺失补全，收齐 K 条后取消该组其余请求；已有完整补全计入 K。需要流式后端，会偏向较快完成的轨迹，额外生成量可能增加。见[候选内超发](methods/ALGORITHMS.md#is-rollout-race) |
| `is.block_first` | 新候选先只生成一块，再发第一条续写请求（见[算法说明](methods/ALGORITHMS.md#alg-conditional-is)）。默认 `false` 配合流式调度即可在 B 处释放其他分支，原请求继续为第零条补全；调度器不会更改此设置或重置原请求随机流 |

<a id="ar-defaults"></a>
### 默认 AR 配置

仓库提供的 `ar` 设置面向 [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)（Gated DeltaNet 与门控注意力 3:1 混合的
64 层模型）在单张 H100 级（80 GB 及以上）GPU 上运行 `is` + `consilience`。显存更大时同一比例自动得到更多缓存；
多卡时调 `tensor_parallel_size` 或 `data_parallel_size`。下表中 `ar` 下的字段省略前缀 `ar.`。

| 设置 | 取值 | 依据 |
| --- | --- | --- |
| `model.path` / `revision` / `local_files_only` | Hub ID / 固定提交 / `false` | 首次运行下载 BF16 权重（约 55.6 GB）；提交固定文件内容，清单另记每个权重文件的哈希 |
| `model.token_penalty` | 8 个词，`strength = 1.0` | wait、hmm、alternatively、actually、however、but、maybe、perhaps；覆盖 thinking 与正文，改变基础分布；该模型上的质量与速度收益尚未验证，设为 `null` 可关闭 |
| `engine.backend` / `dtype` | `vllm` / `bfloat16` | 权重的原生精度；候选与补全经前缀缓存共享预填充。线性注意力层的前缀状态缓存（vLLM 的 `align` 模式）仍属实验功能，出现异常时关闭 `enable_prefix_caching` |
| `vllm.gpu_memory_utilization` / `max_num_seqs` | 0.92 / 32 | 纯文本部分的权重约 50 GiB，其余用作缓存（全注意力层的 KV 每 token 约 64 KiB，同一前缀的请求共享前缀块）；限制并发以减少抢占（抢占使计算量记录变为下界） |
| `engine.continuous_batching.workers` | 1 | 逐题执行才有逐题计算量；只看吞吐时可调大 |
| `vllm.max_model_len` / `max_num_batched_tokens` | 133120 / 16384 | 131,072 token 的输出上限加 2,048 token 的提示余量，一条满长序列的 KV 约 8 GiB；预填充批取 vLLM 在 H100 级 GPU 上的默认值（A100 上宜取 8192） |
| `vllm.engine_kwargs.language_model_only` / `parameter_count` | `true` / `null` | 不加载视觉编码器，MTP 头也不加载；FLOPs 按实际运行的 26,895,998,464 个参数估算 |
| `prompt.chat_template_kwargs` | `{"reasoning_effort": "xhigh"}` | 模板默认值，思考最充分；IS 的每个候选与补全都要生成到思考结束，要降成本可改 `medium` 或 `low` |
| `output.thinking_mode` / `sampling_scope` | `enabled` / `thinking` | 未结束的思考没有答案；IS 与 MH 只重采样思考段，Consilience 也只读思考段 |
| `sampling` | 温度 1，完整支持集 | 模型推荐温度 1；IS 与 MH 要求完整支持集，因而不用推荐的 top-p 0.95、top-k 20 |
| `rewards.consilience.score_temperature` | 1 | 等于采样温度，vLLM 生成时返回 top-5 对数概率，奖励不需要评分前向，`exact_scoring` 保持 `none` |
| `rewards.self_certainty` | 温度 0.25，`scope = "full"` | 原始定义对整条输出取平均；温度未经校准，应在留出题上按 IS 权重的有效样本量调整。它要整个词表的概率：vLLM 上设 `fused_logprobs = true`（默认关闭，因为它关掉 vLLM 的异步调度） |
| `datasets.<name>.max_new_tokens` | 131072 | 思考段与答案共用，给 `xhigh` 的长思考留足长度 |
| `algorithms.is` | M=4、K=1、B=1024；`block_first = false` | 每一步使用相同的候选数、补全数与块长；M=4、K=1 同 [Qwen3 报告](reports/QWEN3_MATH500_REASONING.md)的高预算档；K=1 时候选连同第一条补全由一次请求生成，先生成块反而多一轮请求 |
| `algorithms.mh` / `mh_power` | 块长 1024，每块 3 次更新；`suffix_replay` 与 `early_rejection` 关闭 | 块长按思考长度取而不随上限变：切点落在输出之后的更新直接跳过，奖励目标 MH 的有效更新约为 3 × 思考长度 / 1024；两个开关需要 Transformers 的请求级均匀数流 |

### `dllm`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `model.path` / `revision` | 字符串 / 字符串或 `null` | LLaDA 目录与记录用的提交 |
| `model.weight_sha256` | 对象或 `null` | 同 `ar.model.weight_sha256` |
| `model.mask_token_id` / `trust_remote_code` | 整数 / 布尔 | 掩码 token 与自定义代码加载 |
| `model.adapter` | `null` 或 `{path}` | 叠加的 LoRA 适配器（如 VRPO 训练结果） |
| `engine.device` / `dtype` / `attn_implementation` / `max_batch_size` | — | 加载与批处理选项 |
| `engine.block_logits_only` | 布尔 | 输出投影只计算当前块的位置（模型仍读取整张画布）；logits 与轨迹不变，省去其余位置的 LM head 计算，计算量按实际投影的位置统计 |
| `prompt.system` | 字符串或 `null` | 系统消息 |
| `max_new_tokens` | 整数 | dLLM 输出上限，与数据集的 `max_new_tokens` 取较小值 |
| `sampling` | 对象 | 普通采样与 IS 候选、补全的策略：`block_length`、`steps_per_block`、`temperature`、`top_k`、`top_p`、`remasking`（`low_confidence` \| `random`） |
| `exact_sampling` | 对象 | 字段同上；随机重掩码使轨迹概率可计算，用于块 beam、轨迹幂 MH 与冻结历史 MH |
| `algorithms.beam.decision_block_size` / `width` / `branching_factor` | 整数 | 分块 beam |
| `algorithms.best_of_n.samples` | 整数 | 候选数 |
| `algorithms.mh_power.alpha` / `decision_block_size` / `updates_per_stage` | 数 / 整数 / 整数 | 轨迹幂 MH |
| `algorithms.mh.updates` | 整数 | 独立 MH 的更新数 |
| `algorithms.mh.proposal` / `frozen_history.samples` / `frozen_history.mixture` | — | 同 AR；`frozen_history` 使用 `exact_sampling` 的冻结轨迹 |
| `algorithms.is.candidate_count` / `rollout_count` / `decision_block_size` | 整数 | 逐块扩散 IS 的 M、K 与决策块长 |
| `algorithms.is.candidate_canvas` | `block` \| `full` | `block`：候选单独请求，画布止于该块；`full`：从到输出上限的完整输出切出，其余部分作为第一条补全，与补全同一策略（见[算法说明](methods/ALGORITHMS.md#alg-dllm-is)） |
| `algorithms.is.kept_sequence` | 布尔 | 保留一条完整序列的条件 IS（同 AR）；需要 `candidate_canvas = "full"` |

## `settings/serve.json`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `server.host` / `port` / `served_model_name` | 字符串 / 整数 / 字符串 | 监听地址与 `/v1/models` 报告的模型名（请求里的模型名不做校验） |
| `server.max_concurrent_requests` | 整数 | 同时推理的请求数；各请求共用一个引擎的连续批处理，超出的请求排队 |
| `server.keepalive_seconds` | 数 | 流式请求在搜索期间发送 keep-alive 的间隔 |
| `server.seed` | 整数 | 各请求随机流的根种子，按请求序号派生 |
| `default_effort` / `effort_aliases` | 字符串 / 对象 | 请求未给推理强度时的档位；其他强度名到档位名的映射（如 `minimal` → `low`） |
| `efforts.<名>.reasoning_effort` | `low` \| `medium` \| `xhigh` | chat template 的推理强度 |
| `efforts.<名>.max_new_tokens` | 整数 | 单条序列（思考与答案共用）的输出上限；再受上下文（`ar.engine.vllm.max_model_len` 减提示）限制，请求的 `max_tokens` 只能调低它 |
| `efforts.<名>.max_seconds` | 正数或 `null` | 默认 `null`，只按逻辑长度、EOS/思考边界完成，保持执行调度不影响输出分布。设为数值后，超时不再开始新的 IS 步，至少完成第一步，进行中的一步与答案仍完成；这不是请求总时长硬限，且不再保证调度前后的有限步输出分布相同 |

## `settings/training.json`

GRPO 的奖励与 VRPO 的偏好对都取数据集判定的正确性（正确为 1，否则为 0），即推理 verifier 的 oracle。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `stages` | 数组 | 依次运行的阶段：`download`、`grpo`、`vrpo_preferences`、`vrpo`；各阶段可续跑 |
| `hash_cache_dir` | 字符串 | 同推理 |
| `gsm8k.train` / `gsm8k.test` | 对象 | 字段同 `datasets.gsm8k`；训练集供 GRPO 与 VRPO，测试集只用于检查训练/测试题目重叠 |
| `download.retries` / `retry_wait_seconds` | 整数 / 数 | 下载重试 |
| `download.huggingface.endpoint` / `max_workers` | 字符串或 `null` / 整数 | Hugging Face 镜像与并发 |
| `download.models[]` | 数组 | 每项：`path`、`repository`、`revision`、`allow_patterns`（数组或 `null`）、`weight_sha256`（文件名到哈希的对象，或 `null` 表示不校验） |
| `grpo.model` | 对象 | `path`、`revision`、`weight_sha256`、`tokenizer`、`tokenizer_revision`、`tokenizer_kwargs`、`cache_dir`、`local_files_only`、`trust_remote_code`，以及加载模型用的 `model_kwargs` |
| `grpo.output` / `grpo.resume` | 字符串 / 布尔 | LoRA 输出目录；从最新检查点续跑 |
| `grpo.lora` | 对象 | `r`、`lora_alpha`、`lora_dropout`、`bias`、`target_modules` |
| `grpo.trainer` | 对象 | 原样传给 `trl.GRPOConfig`（步数、批大小、生成数、学习率、KL 系数 `beta`、精度与检查点等） |
| `grpo.power_sample_seconds` | 数 | `nvidia-smi` 功率采样间隔 |
| `vrpo.model` / `engine` / `prompt` / `sampling` | 对象 | 字段同 `dllm` 的对应部分（模型不含适配器） |
| `vrpo.max_new_tokens` | 整数 | 候选生成长度与训练补全的截断长度 |
| `vrpo.preferences.data` / `manifest` | 字符串 | 偏好对 JSONL 与清单 |
| `vrpo.preferences.selection.count` / `seed` | 整数 | 从训练集抽取的候选题数 |
| `vrpo.preferences.pairs` / `num_generations` / `include_reference_completion` / `seed` | 整数 / 整数 / 布尔 / 整数 | 目标偏好对数；每题生成数；是否把参考解答作为一个同样评分的候选；生成种子 |
| `vrpo.training.*` | — | `output`、`resume`、`max_steps`、`gradient_accumulation_steps`、ELBO 估计的 `timestep_samples` / `masks_per_timestep` / `antithetic`、`learning_rate`、`beta`、`max_grad_norm`、`save_steps`、`seed`、`gradient_checkpointing` |
| `vrpo.lora` | 对象 | 同 `grpo.lora` |
