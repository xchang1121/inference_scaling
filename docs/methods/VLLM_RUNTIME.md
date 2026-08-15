# vLLM 推理运行时

vLLM 后端实现见[推理基础设施实现](INFRASTRUCTURE.md#infra-vllm)。本页给出安装、配置、概率能力与
成对测量命令。

## 后端

| `runtime.backend` / `--backend` | 执行方式 | 用途 |
| --- | --- | --- |
| `transformers` | 显式 KV、批处理和完整概率评分 | 参考实现、概率诊断、全词表奖励 |
| `vllm` | 常驻 `AsyncLLM` 与原生连续调度 | vLLM 吞吐实验 |
| `vllm-sync` | 离线 `LLM` | 同步接口与原生 beam |

`run_gsm8k_suite.py`、主方法、replay、动态 IS、分布审计、pass@k 和执行层 benchmark 均接受
`--backend`。GRPO 训练使用 Transformers/TRL。

## 安装

项目依赖范围为 `vllm>=0.25,<0.26`。GPU wheel 的运行平台为 Linux；Windows 主机使用 WSL2，并在
Linux 文件系统中创建独立虚拟环境。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,vllm]"
nvidia-smi
python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"
```

版本兼容说明见
[vLLM GPU installation](https://docs.vllm.ai/en/v0.25.1/getting_started/installation/gpu/)。

## 运行

单方法：

```bash
export PYTHONPATH=src
python experiments/gsm8k_reproduction.py \
  --config configs/gsm8k_3090_aligned.toml \
  --backend vllm \
  --method conditional_is \
  --tag vllm-smoke \
  --limit 8
```

完整套件：

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

后端名称写入 manifest fingerprint。运行结束时，runner 关闭引擎、子进程和事件循环。

## 单卡配置

以下配置面向一张 24 GiB GPU，同时驻留 1.5B base 与 0.5B rollout proposal：

```toml
[runtime]
backend = "vllm"
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

`gpu_memory_utilization` 是各引擎的显存上限。公共 `[vllm]` 设置先应用，随后由 `base`、
`proposal` 和 `rl` 角色覆盖。显式配置覆盖 dtype、TP/DP、显存比例、模型长度、并发序列与 token、
量化、eager mode、LoRA、revision、下载目录和 prefix caching；其他 vLLM 参数放入
`engine_kwargs`。重复配置和未知键触发配置错误。

多卡、量化和 dtype 分别定义独立硬件设置。LoRA adapter 通过 `LoRARequest` 加载。

## 概率与评分

引擎固定使用 `generation_config="vllm"` 和 `logprobs_mode="processed_logprobs"`。生成概率对应实际
温度、top-p 和 top-k policy；每个请求使用独立 seed。

| 操作 | vLLM 原生路径 | 数据来源 |
| --- | --- | --- |
| 温度/top-p/top-k 生成 | 支持 | processed token log-probability |
| on-policy 条件 rollout | 支持 | 生成时保存的 base-policy 概率 |
| 温度 1、无截断 continuation 评分 | 支持 | prompt log-probability |
| MH 温度 proposal 与温度 1 target | 支持 | 生成概率 + base 评分 |
| 非单位温度或截断 behavior 重评分 | Transformers 委托 | exact scoring backend |
| entropy、self-certainty | Transformers 委托 | 完整词表 logits |
| greedy / beam | 支持 | vLLM generation / beam API |

委托评分配置：

```toml
[vllm]
exact_scoring_backend = "transformers"
exact_scoring_device = "cpu"
exact_scoring_dtype = "float32"
```

CPU 委托节省 GPU 显存；GPU 委托需要相应降低 vLLM 引擎显存比例。快照分别记录 native/delegated
sequence、token slot 与 FLOPs。缺少 exact backend 时，委托类请求返回明确错误。

## prefix 与 suffix 复用

Automatic Prefix Caching（APC）复用共同 prompt、候选前缀、重复 rollout 和 broker 恢复前缀。vLLM
返回 `num_cached_tokens`；账本据此扣除命中的 prefill slots。APC 的收益范围为重复 prefill。

global suffix proposer 用同一常驻引擎处理过的请求提出草稿，每个草稿 token 由 target verifier 验证。
drafted、accepted 和 rejected slots 来自 vLLM metrics，并进入主模型逻辑 FLOPs。

`tiers` 定义 active batch 对应的草稿长度。`dynamic_vllm=false` 使用表中最大 \(K\)；
`dynamic_vllm=true` 使用 `DynamicSuffixDecodingProposer` 将调度器选择的 \(K\) 传给官方 suffix
proposer。固定 \(K\) 与动态 \(K\) 分别测量。相关上游接口见
[suffix proposer 源码](https://github.com/vllm-project/vllm/blob/v0.25.0/vllm/v1/spec_decode/suffix_decoding.py)
和[吞吐问题 #49548](https://github.com/vllm-project/vllm/issues/49548)。

其他原生 proposer 可通过 `engine_kwargs` 配置：

```toml
[vllm.engine_kwargs]
enable_chunked_prefill = true
speculative_config = { method = "ngram", num_speculative_tokens = 4, prompt_lookup_min = 2, prompt_lookup_max = 4 }
```

外部 off-policy 数据进入带概率修正的 replay estimator 或冻结 MH 混合 proposal；vLLM 原生 suffix
cache 的数据来源为该引擎已处理请求。

## 成对测速

以下入口在独立进程中运行相同的 Transformers 与 vLLM workload，并核对数据、模型、算法、dtype、
worker、软件环境和代码哈希：

```bash
export PYTHONPATH=src
python experiments/run_vllm_backend_benchmark.py \
  --config configs/gsm8k_3090_aligned.toml \
  --limit 32 \
  --workers 8 \
  --tag rtx3090
```

主要指标为

`transformers_over_vllm_concurrent_wall_time_factor = Transformers 并发墙钟 / vLLM 并发墙钟`。

原始报告同时保存逐 prompt 与并发墙钟、forward slots、精确 token 匹配、数值答案匹配和共同前缀。
稳定吞吐由多个独立 pair 的均值与离散程度描述。

rollout 执行层入口：

```bash
python experiments/benchmark_rollout_infra.py \
  --backend vllm --dtype bfloat16 --section all \
  --output results/infra/rtx3090_vllm.json
```

正式 RTX 3090 报告使用 Transformers 后端。
