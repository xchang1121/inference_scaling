# inference_scaling

自回归语言模型的 training-free 推理扩展。当前主线为 **Qwen3.8-27B、跨 NPU/GPU vLLM 与远程 OpenAI 引擎、思考段自适应条件 IS**。
算法侧（`arllm/pipeline.py`）与服务侧（`serve/reasoner.py`、`serve/server.py`）完全收敛分层；CLI、HTTP 服务和性能评测共用 [`settings/inference.json`](settings/inference.json) 或 `settings/presets/*.json` 硬件预设。

## 快速开始

在已安装兼容 vLLM（昇腾 `vllm-ascend` 或 NVIDIA CUDA `vllm`）的环境中：

```bash
python -m pip install -e ".[serve,evaluation]"
python -m inference_scaling.serve                                                     # 读取默认 settings/inference.json
python -m inference_scaling.serve --settings settings/presets/ascend_910b3_tp2.json   # 昇腾 910B3 TP=2 BF16 开箱预设
python -m inference_scaling.serve --settings settings/presets/cuda_gpu_bf16.json      # 80–96GB NVIDIA GPU BF16 开箱预设
python -m inference_scaling.serve --settings settings/presets/cuda_gpu_fp8.json       # 40–96GB NVIDIA GPU 在线 FP8 预设
python -m inference_scaling.serve --settings settings/presets/cuda_gpu_24g_mxfp4.json # 24GB 单卡 L4/4090 在线 mxfp4 预设
python -m inference_scaling.serve --settings settings/presets/openai_remote.json      # 远程 OpenAI /v1/completions 引擎预设
INFERENCE_SCALING_SETTINGS=settings/presets/ascend_910b3_tp2.json python -m inference_scaling --dataset math500
```

启动前设置权重路径和 `ar.engine.vllm.tensor_parallel_size`（或直接使用 `settings/presets/` 中对应硬件预设）。卡数、上下文与并发需满足设备内存容量，
部署步骤见[服务文档](src/inference_scaling/serve/README.md)，完整 NPU/GPU 四基准与多维度 Scaling Law 实测见[实测报告](docs/reports/CONVERGED_IS_NPU_GPU_RESULTS.md)（完整历史消融分支保留于 `feat/converged-scaling-serve`）。

| 通用默认 | 设置 |
| --- | --- |
| 奖励 | BF16 / FP8 / 远程服务默认 `run.reward = "consilience"`（`initial_penalty=1.5, incomplete_penalty=2.0`）；4-bit `mxfp4` 量化预设使用 `self_certainty`（温度归一化全词表 KL 均值 + `incomplete_penalty=1.5`） |
| 奖励温度 / 评分温度 | `consilience`: `2.0 / 0.7`；`self_certainty`: `0.25 / 0.7`（评分温度与全局采样温度一致） |
| 外部 IS 规模参数 `(M, K, S)` | `candidate_count = 8`、`rollout_count = 2`、`stages = 4`（可在 HTTP 请求 `scaling` 字段按次覆盖，例如低延迟两阶段传 `{"candidate_count": 4, "stages": 2}`） |
| 内部自动调度 | 单一收敛多阶段 Block-SMC 实现：硬件 KV 页自动对齐首段检查点（昇腾 `1536` / GPU `1024` / L4 `832`）+ `B_0 // 2` 试探门控 + 实测长度自适应几何阶段切分 + 衰减选择性前瞻 + 全阶段长尾停滞剪枝 + 稀释校准 ESS 温控 + `S>=3, M>=4` 自动 `P=2` 多粒子系统重采样与精确父质量守恒 + `block_metropolized` 选择核 |
| 概率统计 | 跨平台预掩码标量统计 `scalar_self_certainty = true`（昇腾与 NVIDIA vLLM 0.25–0.34+ 均无额外评分前向） |
| 全局采样 | `temperature = 0.7`、`top_p = 0.95`、`top_k = 20`、`token_penalty = null` |
| 引擎默认 | `gpu_memory_utilization = 0.88`、`stream_interval = 8`、`enforce_eager = false`、`enable_prefix_caching = true`、`max_concurrent_requests = 2` |

其他对比算法（`sample`、`greedy`、`beam`、`best_of_n`、`mh`、`mh_power`）保持可用。[算法与适用条件](docs/methods/ALGORITHMS.md#alg-conditional-is)

### 长度与采样配置

| 字段 | 含义与默认值 |
| --- | --- |
| `ar.max_new_tokens` | 单条输出总长，含思考与最终内容；`null` 使用剩余上下文 |
| `ar.engine.context_window` | 输入与输出合计容量；`null` 读取模型配置，当前模型原生为 262144 |
| `ar.thinking` | 未指定档位或预算时，`xhigh`、131072 thinking tokens |
| `ar.output.sampling_scope` | `thinking`：IS/MH 选定思考段后，由基模型续写最终内容 |

显式档位映射为 `low → 4096`、`medium → 16384`、`xhigh → 262144`；`none` 关闭思考。
也可单独指定 `thinking_budget`。思考预算耗尽后关闭思考段，继续生成最终内容；客户端 `max_tokens` /
`max_completion_tokens` 进一步限制生成总长。这些上限针对单条序列，全部候选的计算量另行累计。
[完整预算规则](docs/SETTINGS.md#arthinking)

## 目标分布

给定提示 $`x`$、基础分布 $`p(y\mid x)`$、奖励 $`r(x,y)`$ 和温度 $`\tau>0`$，考虑 KL 正则化目标：

```math
\max_{\pi(\cdot\mid x)}
\left\{\sum_y\pi(y\mid x)r(x,y)
-\tau D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\,\|\,p(\cdot\mid x)\right)\right\},
\qquad \sum_y\pi(y\mid x)=1.
```

加入归一化约束后，一阶条件给出闭式解：

```math
\pi_r(y\mid x)=
\frac{p(y\mid x)\exp\{r(x,y)/\tau\}}
{\sum_{y'}p(y'\mid x)\exp\{r(x,y')/\tau\}}.
```

IS 与奖励 MH 以此为目标；有限候选数、有限更新轮次和提前选择分别产生近似误差。
奖励只参与选择，模型参数保持固定。

<a id="算法"></a>
<a id="奖励"></a>
## 算法与奖励

| `--algorithm` | 行为 |
| --- | --- |
| `sample` | 按共用采样策略生成一次，作为 Base |
| `greedy` / `beam` | 逐 token 贪心 / beam search |
| `best_of_n` | 生成 N 条完整序列，选择奖励最高者 |
| `is` | 条件 IS，保留完整 rollout；默认使用两阶段探索与提前选择 |
| `mh` | 奖励目标的后缀 MH，可选完整候选池和冻结历史 proposal |
| `mh_power` | 以幂分布 $`p(y\mid x)^\alpha`$ 为目标的后缀 MH |

| `--reward` | 序列评分 |
| --- | --- |
| `self_certainty` | 评分段上全词表 $`D_{\mathrm{KL}}(U\Vert p_t)`$ 的均值，默认评思考段 |
| `consilience` | 末段 top-K 置信度均值减去加权首段均值，默认评思考段 |
| `verifier` | `source=dataset` 对照参考答案；`source=vote` 使用答案一致比例。IS/MH 的投票参照池在搜索前冻结 |

CLI 还接受 `--dataset gsm8k|math500`、`--model ar` 和 `--output`。默认算法为 `is`、数据集为 `gsm8k`，
未指定奖励时读取 `run.reward`。`sample`、`greedy`、`beam` 和 `mh_power` 不使用奖励。
参数定义见[设置说明](docs/SETTINGS.md)；公式、算法步骤、关键代码与限制集中在[算法文档](docs/methods/ALGORITHMS.md)。

<a id="推理服务"></a>
<a id="执行优化"></a>
## 推理服务与执行优化

[推理服务](src/inference_scaling/serve/README.md)提供 OpenAI Chat Completions 与 Anthropic Messages 接口，
支持思考与内容流式交付、工具调用和请求级预算。服务读取同一默认奖励与 IS 配置。

执行层提供连续批处理、前缀 KV 复用、流式分支释放、共享准入和长尾保护；
Self-Certainty 直接复用生成时的概率统计。具体实现见[rollout 调度](docs/methods/ROLLOUT_SCHEDULING.md)
及[后端与计量](docs/methods/ALGORITHMS.md#11-共同执行实现)。

## 质量与性能结果

| 报告 | 主要结果 |
| --- | --- |
| [收敛版 Adaptive IS 与多阶段 Block-SMC：跨基准帕累托与 NPU/GPU 实测](docs/reports/CONVERGED_IS_NPU_GPU_RESULTS.md) | Wave B1（880 组）锁定 `token_penalty=null, T=0.7, p=0.95, k=20`（89.90% vs 86.15%）；Wave B2 13 臂 × 88 题 Adaptive IS v2 达 **88.02%**；多阶段多维度 Block-SMC 8 臂 × 48 难题中 `M=8, K=2, S=4` (`P=2` + `consilience`) 达 **85.42%（41/48，+2/-0 零退化，LCB 79.2% / MATH 91.7%）**；覆盖昇腾 910B3 与 Blackwell 96GB / A100-40GB / L4 24GB 三档 GPU 及 5 套收敛开箱预设 |
| [代码评测：正确率、性能与思维链风格](experiments/qwen38-20261106/reports/coding/REPORT.md) | LiveCodeBench 100 题：按块 SC K=2 为 67%，Base 为 63%，配对区间含 0；HumanEval+ 164 题未超过 Base。含十二方法性能表和跨题风格分析 |
| [100 题选择策略与奖励对照](docs/reports/QWEN38_IS_SELECTION.md) | Base / 快速 Consilience IS / 默认 Self-Certainty IS：92% / 93% / 94%。按块 SC K=2 为 95%，比快速 SC 耗时约多 20%、FLOPs 多 75%；准确率差区间包含 0 |
| [IS 服务延迟与吞吐](docs/reports/QWEN38_IS_SERVING.md) | 同环境的 TP2/TP4、IS/Base 时延比、思考与内容 TTFT，以及输出吞吐 |
| [昇腾调度实验](docs/reports/ASCEND_IS_INFRA.md) | 历史调度与探索消融，区分固定工作量加速和减少探索计算 |

[完整分析材料](experiments/qwen38-20261106/README.md)包含代码评测 3168 份输出及判题、数学评测 1200 份输出，
以及报告、完整思维链、审阅标签、冻结判题输入、配置和分析脚本。普通 Git 归档附逐文件 SHA-256 与离线展开工具。

默认配置依据质量与成本对照选定，通用长度预算保持独立。性能入口为
[`scripts/benchmark_is_gpu.py`](scripts/benchmark_is_gpu.py)，默认比较同题 Base 与配置中的 IS；
`--study selection`、`rewards`、`selection-self-certainty` 显式复现对应历史方法组。
[指标与命令](docs/SETTINGS.md#性能评测)

## 输出

结果位于 `<output>/<dataset>/ar/<algorithm>[-<reward>]/<fingerprint>/`：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 生效设置、源码/模型/数据身份与运行环境 |
| `records.jsonl` | 逐题输出、答案判分、奖励、算法摘要、分阶段成本与耗时 |
| `executions.json` | 各次执行的状态、运行级计算量与墙钟时间 |
| `summary.json` | 准确率及区间、pass@k、长度、失败和成本汇总 |

同一指纹续跑缺失记录；增加 `run.draws` 可追加重复次数。FLOPs 以 `2 × 参数量 × 前向 token 位置数` 估算，
包括全部候选、补全、评分与最终内容生成。取消或抢占后的不完整计量显式标记，输出 TPS 与内部生成 TPS 分开报告。

<a id="安装"></a>
<a id="测试与目录"></a>
## 安装与测试

默认昇腾环境沿用已安装的 PyTorch、vLLM 和 vllm-ascend，安装本仓库时使用 `.[serve,evaluation]`。
NVIDIA vLLM 使用 `.[vllm,serve,evaluation]`，Transformers 使用 `.[gpu,evaluation]`；
替换后端时需按[概率统计与流式能力](docs/SETTINGS.md#arengine)调整相应设置。

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

源码分为 `app/`（入口与记录）、`datasets/`（提示和判分）、`arllm/`（算法与后端）、`shared/`（公共概率核与模型工具）。
`online-speculation` 已完整移至[独立分支 `codex/online-speculation`](https://github.com/xchang1121/inference_scaling/tree/codex/online-speculation)，保留原项目文件及提交历史。
日常运行数据保存在 Git 忽略的 `results/` 中，已归档分析材料位于 `experiments/`。

<a id="历史实现"></a>
<a id="历史结果"></a>
## 历史实现与结果

dLLM、训练及旧 IS 组合保存在[归档分支](https://github.com/xchang1121/inference_scaling/tree/archive/legacy-dllm-training-is)。
历史报告保留原配置与图片：[GSM8K 准确率](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md)、
[Qwen3 思考模式](docs/reports/QWEN3_MATH500_REASONING.md)、[RTX 3090 执行成本](docs/reports/RTX3090_ROLLOUT_INFRA.md)、
[加速研究总结](docs/reports/ASCEND_RESEARCH_SUMMARY.md)。
