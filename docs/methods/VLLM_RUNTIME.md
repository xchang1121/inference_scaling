# vLLM 推理运行时

仓库现在可以在不改动 MH、条件 IS、rollout replay、渐进预算和 SMC forest 算法代码的情况下，把模型
执行层从 Transformers 切换为 vLLM。默认的 `vllm` 模式使用一个常驻 `AsyncLLM`：不同题目、候选和
rollout 产生的请求都进入同一个连续调度器；算法仍通过同步的 `sample_batch` / `score_batch` 接口取回
完整结果。请求级 seed、实际采样分布的 token log-probability、原始请求顺序和现有 token/FLOPs 账本
都保留。

这里的支持是运行时实现，不是新的实验结论。当前仓库中的正式 3090 数字仍来自 Transformers 后端；
在 Linux 或 WSL2 上完成下面的成对 benchmark 前，不能把既有批处理加速数字写成 vLLM 加速数字。

## 三种后端

| `runtime.backend` / `--backend` | 执行方式 | 适用场景 |
| --- | --- | --- |
| `transformers` | 仓库自带的显式 KV、批处理和精确评分实现 | 参考实现、概率诊断、完整词表奖励 |
| `vllm` | 常驻 `AsyncLLM`，跨调用原生连续批处理 | 推荐的 vLLM 吞吐路径 |
| `vllm-sync` | 离线 `LLM`；并发时仍可由仓库 dispatcher 合批 | 调试同步接口或使用 vLLM 原生 beam search |

`run_gsm8k_suite.py`、主方法、replay、动态 IS、分布审计、pass@k 和异步吞吐入口都接受
`--backend`。GRPO 训练入口不使用 vLLM；`rtx3090_reproduction.py` 专门检查 Transformers 的 KV
内部行为，也仍保持 Transformers-only。

## 安装

本项目固定 `vllm>=0.25,<0.26`，因为历史 suffix proposer、动态 speculative batch table 和原生
speculation metrics 在这一版本线内使用统一接口。vLLM 的官方 GPU 安装要求是 Linux，且明确不原生支持
Windows；Windows 机器应使用 WSL2 的 Linux 环境。不要把 Windows 下创建的 `.venv` 直接拿到 WSL
中使用。为了避免 `/mnt/c` 的文件系统开销影响模型加载与 benchmark，长实验最好把仓库和模型放在
WSL 的 Linux 文件系统内。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,vllm]"
nvidia-smi
python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"
```

安装版本必须与驱动、PyTorch 和 vLLM wheel 相容。官方安装说明见
[vLLM GPU installation](https://docs.vllm.ai/en/v0.25.1/getting_started/installation/gpu/)。

## 运行

单个方法可以直接覆盖 TOML 中的默认后端：

```bash
export PYTHONPATH=src
python experiments/gsm8k_reproduction.py \
  --config configs/gsm8k_3090_aligned.toml \
  --backend vllm \
  --method conditional_is \
  --tag vllm-smoke \
  --limit 8
```

完整套件同样只需增加一个参数；该参数会传递给主网格、replay、动态 IS、异步比较、pass@k 和全部
消融任务，并在实验指纹中留下记录：

```bash
python experiments/run_gsm8k_suite.py \
  --config configs/gsm8k_3090_aligned.toml \
  --backend vllm \
  --tag vllm \
  --with-matched-target \
  --with-replay \
  --with-dynamic-is \
  --with-async
```

每个运行器正常结束时会显式关闭 vLLM 引擎、子进程和事件循环；分阶段加载不同方法的 pass@k 与
分布审计也会在下一模型进入显存前释放上一模型。

## 配置

四个 GSM8K profile 已包含可直接覆盖的 vLLM 配置。下面是 24 GB 单卡、同时驻留 1.5B base 与
0.5B proposal 的结构；`0.62 + 0.28 = 0.90` 是两个引擎各自可使用的显存比例，不是算法预算：

```toml
[runtime]
backend = "transformers"
device = "cuda"
dtype = "float32"

[vllm]
asynchronous = true
enable_prefix_caching = true
exact_scoring_backend = "none"
tensor_parallel_size = 1
data_parallel_size = 1

[vllm.base]
gpu_memory_utilization = 0.62
max_num_seqs = 48
max_num_batched_tokens = 12288

[vllm.proposal]
gpu_memory_utilization = 0.28
max_num_seqs = 24
max_num_batched_tokens = 6144

[vllm.rl]
gpu_memory_utilization = 0.62

[vllm.engine_kwargs]
enable_chunked_prefill = true

[acceleration.speculation]
enabled = false
tiers = [[1, 8], [4, 0], [512, 0]]
min_context_tokens = 2
min_token_probability = 0.10
tree_max_context_tokens = 24
vllm_max_cached_requests = 10000
dynamic_vllm = false
```

公共 `[vllm]` 值会先应用，再由 `[vllm.base]`、`[vllm.proposal]` 或 `[vllm.rl]` 覆盖。支持的显式
设置包括 dtype、TP/DP、显存比例、最大序列长度、最大并发序列与 token、量化、eager mode、LoRA
最大 rank、revision、下载目录和 prefix caching。未识别的键会直接报错，避免拼写错误被静默忽略。
`[vllm.engine_kwargs]` 以及每个角色自己的 `engine_kwargs` 会传给 vLLM；已有显式配置入口的 model、
dtype、并行度、显存、量化、长度、batch、LoRA、generation config、logprob mode 和 prefix-caching
设置不能从这里被第二次覆盖。beam 所需的 `max_logprobs` 会自动提升到至少
`max(20, 2 * num_beams)`。

多卡可以设置 `tensor_parallel_size` / `data_parallel_size`，但这会改变 GPU 数量。它不能与单卡
Transformers 的耗时直接写成“同硬件加速”；应单独报告 GPU 数、总 token、总 FLOPs 和墙钟。若模型
能放入单卡，vLLM 官方也建议先使用单卡；模型放不下时再采用 tensor parallel。

LoRA adapter 会自动以 `LoRARequest` 加载。量化会改变数值表示；如要与 FP32 主表比较，必须把它当成
单独设置。

`[acceleration.speculation]` 是两套后端共用的历史草稿入口。vLLM 路径把它转换为原生 global suffix
proposer；`tiers` 进一步转换为 `[起始 batch, 结束 batch, K]` 的动态表。每个草稿 token 都由 target
模型验证，历史数据不直接进入算法权重。`dynamic_vllm=false` 会保留 suffix proposer，但固定使用表中
最大的 $K$，也是当前默认值。只有显式设置 `dynamic_vllm=true` 才启用动态表；vLLM 0.25 已有并发
阈值附近的吞吐下降与部分 speculator CUDA graph 兼容性报告，因此它必须作为独立消融测量。原生
suffix proposer 本身还断言 $K$ 固定；动态模式自动改用仓库的
`inference_scaling.vllm_suffix_proposer.DynamicSuffixDecodingProposer`。该类委托给官方 suffix proposer，
只解除这一处运行时 $K$ 约束，不替换 suffix cache 或 target verification。对应限制可直接核对
[vLLM 0.25 suffix proposer 源码](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/v1/spec_decode/suffix_decoding.py)；
动态调度的并发风险见上游
[吞吐问题 #49548](https://github.com/vllm-project/vllm/issues/49548)。

如果只想试验 vLLM 自带的其他 proposer，仍可通过 `engine_kwargs` 显式传入；它不能与上面的统一配置
同时出现：

```toml
[vllm.engine_kwargs]
enable_chunked_prefill = true
speculative_config = { method = "ngram", num_speculative_tokens = 4, prompt_lookup_min = 2, prompt_lookup_max = 4 }
```

原生 metrics 提供 draft 次数、draft token 和 accepted token。仓库把未接受的 draft slot 计入 target
verification slot，再估算主模型 FLOPs；这仍是逻辑 FLOPs，不等于 fused verifier kernel 的硬件指令数。
更完整的实现与正确性边界见 [rollout 生成与复用](ROLLOUT_ACCELERATION.md)。

## 哪些概率量由 vLLM 原生精确提供

vLLM 引擎固定使用 `generation_config="vllm"`，防止模型目录中的 generation config 暗中改写采样；
生成 token 使用 `processed_logprobs`，因此返回的是温度、top-p 和 top-k 处理之后，算法实际抽样分布的
log-probability。每条请求单独保留 seed，不能为了少建几个参数对象而折叠成一个 `n > 1` 请求。

| 操作 | 原生路径 | 说明 |
| --- | --- | --- |
| 任意已支持温度/top-p/top-k 的生成及其行为概率 | 是 | 使用生成 token 的 processed log-probability |
| on-policy 条件 rollout | 是 | rollout 自带实际 base-policy 概率，不重复评分 |
| 温度 1、无 top-p/top-k 截断的 continuation 评分 | 是 | 从 prompt log-probability 读取所选 token |
| MH 的温度 proposal 与温度 1 target | 是 | proposal 概率来自生成；target 由原生 base 评分得到 |
| 历史数据在非单位温度或截断 behavior 下重新评分 | 否 | prompt log-probability 不是处理后分布，需精确评分后端 |
| entropy、self-certainty 等完整词表统计 | 否 | 所选 token 的概率不足以恢复完整词表统计 |
| greedy / beam baseline | 是 | greedy 原生生成；同步 beam 用 vLLM API，异步 beam 按官方逐 token 扩展方式进入同一调度器 |

需要后两类操作时，显式配置 Transformers fallback：

```toml
[vllm]
exact_scoring_backend = "transformers"
exact_scoring_device = "cpu"
exact_scoring_dtype = "float32"
```

fallback 只接收 vLLM 无法精确给出的评分，不参与普通生成。放在 CPU 上节省显存但可能很慢；放在 GPU
上则必须相应降低各 vLLM 引擎的 `gpu_memory_utilization`。报告会分别记录 delegated sequence、token
slot 与 FLOPs，不能把 fallback 成本算到 vLLM 之外。没有配置 fallback 时，遇到不受支持的评分策略会
报错，不会用 raw prompt 概率产生错误 importance ratio。

## 连续批处理与 prefix 复用

`vllm` 模式为每个模型保留一个事件循环线程和一个常驻 AsyncLLM。多个算法 worker 可以同时调用同步
接口；适配层把请求提交给同一个引擎并等待各自结果。由于 vLLM 已经拥有动态 scheduler，仓库的
`ContinuousBatchingBackend` 在该模式下只做统计与生命周期管理，不再增加一个会把请求串行化的 Python
dispatcher。

Automatic Prefix Caching 默认开启。共同题目 prompt、候选前缀、重复 rollout 和 replay 评分可以复用
已有 KV block；报告读取 vLLM 返回的 `num_cached_tokens`，从实际 prefill token slot 中扣除并写入
`shared_prefill_tokens_saved`。APC 只减少共同前缀的 prefill，不能加快第一个请求，也不能减少长输出的
decode。相关语义见
[vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/v0.25.1/features/automatic_prefix_caching/)。

启用 suffix proposer 后，常驻引擎还会把已完成请求维护在 global suffix cache 中，用历史 token 路径
提出草稿；这与 APC 的 KV block 复用是两件事。APC 减少共同前缀 prefill，suffix proposer 减少接受
草稿时所需的串行 decode 轮次。两者都不能保证加速：低 suffix 命中或低接受率会增加验证工作，因此
必须同时报告墙钟、draft 接受率、target slot 和 cache build。

global suffix cache 的公开路径缓存同一引擎已经处理的请求；vLLM 0.25 没有直接插入任意外部 token
轨迹的公共 API。因此，来自其他模型的离线 off-policy 数据可以按原算法进入 replay estimator，但只有
经过该 base 引擎的历史请求才会进入原生 suffix 草稿路径。仓库不会维护一份无法影响 vLLM proposer 的
Python 副本来制造“已支持外部注入”的假象。

运行时快照还记录 engine request 数、原生/委托评分数和 `maximum_in_flight_requests`。后者是并发请求
高水位，不是可相加的计算量；吞吐报告按高水位本身展示，不对两次 snapshot 做差。

## 成对测量 vLLM 加速

下面的入口在两个独立进程中依次运行完全相同的 Transformers 与异步 vLLM workload，避免前一个引擎
占用后一个进程的显存；随后由汇总器核对数据集哈希与行号、模型权重、算法参数、dtype、worker 数、
软件环境和实现文件哈希：

```bash
export PYTHONPATH=src
python experiments/run_vllm_backend_benchmark.py \
  --config configs/gsm8k_3090_aligned.toml \
  --limit 32 \
  --workers 8 \
  --tag rtx3090
```

若某一侧已完成，可加 `--reuse-existing`。三个文件默认写入 `results/validation/`：Transformers 原始
报告、vLLM 原始报告和成对 comparison。汇总器只接受 `AsyncVLLMBackend`，并拒绝 dtype、量化、额外
精确评分模型或 TP/DP GPU 数不一致的“加速”比较。

主要指标
`transformers_over_vllm_concurrent_wall_time_factor = Transformers 并发 workload 秒数 / vLLM 并发 workload 秒数`；
大于 1 才表示 vLLM 相对 Transformers 更快。`sequential` 因子只比较逐 prompt 执行，用来区分后端
kernel 收益与跨 prompt 调度收益。逻辑 FLOP 因子使用双方记录的 forward token slot，不能代替 fused
kernel、padding、通信或 speculative verification 的硬件 FLOPs。

每个原始报告会比较该后端的逐 prompt 与并发输出，包括精确 token、数值答案、共同前缀和准确率；成对
报告目前不保存两后端逐题 token trace，因此不会声称 cross-backend 输出逐 token 相等。一次成对运行也
不是耗时置信区间；要报告稳定吞吐，应重复整个 pair 并给出离散程度。

## rollout 基础设施测量入口

`benchmark_rollout_infra.py` 在同一 schema 下比较无草稿、固定草稿、active-batch 草稿，以及固定条件
IS、渐进预算、流式奖励/run-ahead 和 SMC forest。vLLM 模式使用常驻异步引擎并读取原生 draft 指标：

```bash
export PYTHONPATH=src
python experiments/benchmark_rollout_infra.py \
  --backend vllm --dtype bfloat16 --section all \
  --output results/infra/rtx3090_vllm.json
```

当前仓库中的 Transformers 三随机种子实测见
[RTX 3090 推理基础设施优化汇总](../reports/RTX3090_ROLLOUT_INFRA.md)。这台 3090 是 Windows 主机且
未安装 WSL，因此报告明确留空 vLLM 数值；不能把 Transformers 的负载感知收益冒充成 vLLM 收益。
