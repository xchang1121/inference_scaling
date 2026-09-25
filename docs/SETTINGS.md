# 设置文件

推理与训练各有一个固定的设置文件，从仓库根目录读取：

| 文件 | 读取者 |
| --- | --- |
| `settings/inference.json` | `python -m inference_scaling`（命令行只选择算法、模型族、奖励和数据集） |
| `settings/training.json` | `python -m training`（没有命令行参数） |

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
| `verifier.temperature` | 数 | 外部奖励的温度 |
| `verifier.source` | `dataset` \| `python` \| `constant` | 外部奖励来源 |
| `verifier.dataset.correct` / `incorrect` / `unparseable` | 数 | 数据集判定器对正确、错误、无答案的奖励值 |
| `verifier.python.factory` | 字符串或 `null` | `package.module:function`，以 `factory(context=..., **options)` 调用，返回 `(prompt, completion) -> 分数` 或带 `score`（可选 `score_batch(prompt, completions)`）的对象，例如外部评分模型 |
| `verifier.python.options` | 对象 | 传给工厂的参数（不能含 `context`） |
| `verifier.python.requires_reference` | 布尔 | 为真时 `context.reference` 才包含参考答案 |
| `verifier.constant.value` | 数 | 常数奖励（对照与测试） |
| `vote.temperature` | 数 | 投票奖励的温度 |
| `vote.pool_size` | 整数 | `is`/`mh` 的冻结样本池大小；奖励为池中与该答案一致的比例。`best_of_n` 不用池：候选互相投票，得票最多的答案胜出，平票在最高票候选中按种子随机选一个 |
| `logprob.temperature` / `logprob.score_temperature` | 数 | 奖励温度；评分策略的温度（1 为模型原始分布）。奖励为有效输出 token 的平均对数概率 |
| `consilience.temperature` / `score_temperature` | 数 | 奖励温度；计算 top-$`K`$ 置信度所用的温度 |
| `consilience.scope` | `thinking` \| `full` | 只评思考段（缺少完整思考段时回退到全序列并记录原因）或评全序列 |
| `consilience.top_k` / `window_fraction` / `window_tokens` / `skip_fraction` / `initial_penalty` | 整数 / 数 / 整数或 `null` / 数 / 数 | 置信度窗口：跳过开头 `skip_fraction`，首段与末段各取 `window_fraction`（或固定 `window_tokens`），分数为末段均值减 `initial_penalty` 倍首段均值 |

`logprob` 与 `consilience` 读取自回归模型的概率，只用于 `--model ar`。

### `ar.model`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `path` | 字符串 | 本地目录或 Hub 模型 ID |
| `revision` | 字符串或 `null` | 固定的 Hub 提交 |
| `weight_sha256` | 字符串或 `null` | 权重摘要（单文件为其 SHA-256，分片为各分片哈希映射的哈希）；非空时强制校验 |
| `adapter` | `null` 或 `{path, revision}` | 叠加在基础模型上的 PEFT 适配器（如 GRPO 训练结果） |
| `tokenizer` / `tokenizer_revision` / `tokenizer_kwargs` | 字符串或 `null` / 字符串或 `null` / 对象 | 独立的 tokenizer 及其参数 |
| `cache_dir` / `local_files_only` / `trust_remote_code` | 字符串或 `null` / 布尔 / 布尔 | Hub 缓存与加载选项 |

### `ar.engine`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `backend` | `transformers` \| `vllm` | 执行引擎 |
| `device` / `dtype` | 字符串 | Transformers 设备与精度；vLLM 使用 `dtype`，精确评分后端同时使用两者 |
| `context_window` | 整数或 `null` | 额外的上下文上限；实际生成上限取它、模型上下文与 `max_new_tokens` 的最小值 |
| `transformers.attn_implementation` / `device_map` / `model_kwargs` | 字符串或 `null` / 字符串、对象或 `null` / 对象 | Transformers 加载选项 |
| `transformers.max_score_batch_size` | 整数 | 一个评分批次的最大序列数；批次的填充位置数也不超过它乘以 `score_chunk_size` |
| `transformers.score_chunk_size` | 整数 | 长序列评分与前缀预填充的分块长度 |
| `vllm.asynchronous` | 布尔 | 异步引擎（原生连续批处理）或同步引擎 |
| `vllm.tensor_parallel_size` / `data_parallel_size` / `gpu_memory_utilization` / `max_model_len` / `max_num_seqs` / `max_num_batched_tokens` / `quantization` / `enforce_eager` / `max_lora_rank` | — | 对应 vLLM 引擎参数 |
| `vllm.enable_prefix_caching` | 布尔 | 前缀缓存 |
| `vllm.mh_fused_logprobs` | 布尔 | 幂目标 MH 在同一次解码中取得 proposal 与基础模型概率；需要 `asynchronous = false`，只影响 `mh_power` |
| `vllm.exact_scoring` | `none` \| `transformers` | 用同一份权重的 Transformers 副本精确评分（Consilience 与精确对数概率需要） |
| `vllm.parameter_count` | 整数或 `null` | 计算量统计用的参数量；`null` 时从权重读取 |
| `vllm.engine_kwargs` | 对象 | 其他引擎参数；不能覆盖上述字段，也不能开启 speculative decoding |
| `continuous_batching.workers` | 整数 | 同时求解的题数；大于 1 时各题请求经连续批处理合并，记录中的逐题成本为 `null` |
| `continuous_batching.max_batch_size` / `max_batch_tokens` / `batch_wait_seconds` | 整数 / 整数 / 数 | 合并批的上限与等待时间 |

### `ar.prompt`、`ar.output`、`ar.sampling`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `prompt.system` | 字符串或 `null` | 系统消息 |
| `prompt.format` | `auto` \| `chat` \| `plain` | 使用 chat template、强制使用，或直接用用户文本 |
| `prompt.chat_template_kwargs` | 对象 | 传给 chat template 的参数（如 `enable_thinking`） |
| `output.thinking_mode` | `auto` \| `enabled` \| `disabled` | 思考模式；`enabled` 时未完成的思考没有最终答案（评测文本为空） |
| `output.thinking_start_text` / `thinking_end_text` / `starts_in_thinking` | 字符串或 `null` / 字符串或 `null` / 布尔或 `null` | 显式的思考段标记；为 `null` 时从 tokenizer 词表与 chat template 识别 |
| `output.sampling_scope` | `full` \| `thinking` | `mh`、`mh_power`、`is` 在完整输出或思考段上采样；思考段结束后由基础模型生成最终内容。读取答案文本的奖励（`vote`、`verifier`）会回退到 `full` 并记录原因 |
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
| `mh_power.suffix_schedule` | `uniform` \| `inverse_length` \| `multiscale`（后缀长度分布） |
| `mh.block_size` / `steps_per_block` / `iterations` / `suffix_schedule` | 同上，目标为 $`p\exp\{r/\tau\}`$ |
| `mh.proposal` | `base`（基础策略后缀）或 `frozen_history`（冻结历史混合 proposal） |
| `mh.frozen_history.samples` / `mixture` | 历史样本数；从历史后缀提议的概率 |
| `is.planning` | `fixed`：固定候选数、补全数与块长；`full_horizon` / `chunk_adaptive`：在前向 token 预算内逐块重新规划 |
| `is.fixed.candidate_count` / `rollout_count` / `block_size` | 固定规划的 M、K、B |
| `is.joint.forward_token_budget` | 每题的前向 token 位置预算 |
| `is.joint.block_sizes` / `candidate_counts` / `rollout_counts` | B、M、K 的候选网格 |
| `is.joint.pilot_candidates` / `pilot_rollouts` / `pilot_fraction` | 试点共享的完整输出数、每候选补全数与最多占用的预算比例 |
| `is.joint.relative_variance_floor` | 规划时相对方差的下限 |
| `is.joint.expected_output_tokens` | 初始期望输出长度；`null` 时先生成一条完整输出测量 |
| `is.chunk_adaptive.initial_block_size` / `initial_candidate_count` / `initial_rollout_count` / `adjustment_min_improvement` | `chunk_adaptive` 的起始配置与调整所需的最小改进比例 |

规划细节见 [BUDGET.md](methods/BUDGET.md)。

### `dllm`

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `model.path` / `revision` | 字符串 / 字符串或 `null` | LLaDA 目录与记录用的提交 |
| `model.weight_files` / `weight_bytes` / `weight_sha256` | 数组 | 逐分片的文件名、字节数与 SHA-256，全部强制校验 |
| `model.mask_token_id` / `trust_remote_code` | 整数 / 布尔 | 掩码 token 与自定义代码加载 |
| `model.adapter` | `null` 或 `{path}` | 叠加的 LoRA 适配器（如 VRPO 训练结果） |
| `engine.device` / `dtype` / `attn_implementation` / `max_batch_size` | — | 加载与批处理选项 |
| `prompt.system` | 字符串或 `null` | 系统消息 |
| `max_new_tokens` | 整数 | dLLM 输出上限，与数据集的 `max_new_tokens` 取较小值 |
| `sampling` | 对象 | 普通采样与 IS 候选、补全的策略：`block_length`、`steps_per_block`、`temperature`、`top_k`、`top_p`、`cfg_scale`、`remasking`（`low_confidence` \| `random`） |
| `exact_sampling` | 对象 | 字段同上；随机重掩码使轨迹概率可计算，用于块 beam、轨迹幂 MH 与冻结历史 MH |
| `algorithms.beam.decision_block_size` / `width` / `branching_factor` | 整数 | 分块 beam |
| `algorithms.best_of_n.samples` | 整数 | 候选数 |
| `algorithms.mh_power.alpha` / `decision_block_size` / `updates_per_stage` | 数 / 整数 / 整数 | 轨迹幂 MH |
| `algorithms.mh.updates` | 整数 | 独立 MH 的更新数 |
| `algorithms.mh.proposal` / `frozen_history.samples` / `frozen_history.mixture` | — | 同 AR；`frozen_history` 使用 `exact_sampling` 的冻结轨迹 |
| `algorithms.is.candidate_count` / `rollout_count` / `decision_block_size` | 整数 | 逐块扩散 IS 的 M、K 与决策块长 |

## `settings/training.json`

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
| `grpo.verifier` | 对象 | 字段同 `rewards.verifier`（无温度）；`dataset` 来源按每行参考答案评分 |
| `grpo.power_sample_seconds` | 数 | `nvidia-smi` 功率采样间隔 |
| `vrpo.model` / `engine` / `prompt` / `sampling` | 对象 | 字段同 `dllm` 的对应部分（模型不含适配器） |
| `vrpo.max_new_tokens` | 整数 | 候选生成长度与训练补全的截断长度 |
| `vrpo.preferences.data` / `manifest` | 字符串 | 偏好对 JSONL 与清单 |
| `vrpo.preferences.selection.count` / `seed` | 整数 | 从训练集抽取的候选题数 |
| `vrpo.preferences.pairs` / `num_generations` / `include_reference_completion` / `seed` | 整数 / 整数 / 布尔 / 整数 | 目标偏好对数；每题生成数；是否把参考解答作为一个同样评分的候选；生成种子 |
| `vrpo.verifier` | 对象 | 同 `grpo.verifier` |
| `vrpo.training.*` | — | `output`、`resume`、`max_steps`、`gradient_accumulation_steps`、ELBO 估计的 `timestep_samples` / `masks_per_timestep` / `antithetic`、`learning_rate`、`beta`、`max_grad_norm`、`save_steps`、`seed`、`gradient_checkpointing` |
| `vrpo.lora` | 对象 | 同 `grpo.lora` |
