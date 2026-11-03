# inference_scaling

本仓库实现自回归语言模型的 training-free 推理扩展，当前主线为 Qwen3.8-27B 的 IS 推理与昇腾执行。
批量评测与 HTTP 服务共用 [`settings/inference.json`](settings/inference.json)，包括模型、生成预算、采样、奖励与执行设置。
性能评测以同题 Base 思考采样为对照，报告正确率、IS/Base 端到端时延比、生成量与计算量；质量对照同时评估增加计算换取正确率的效果。

## 目标分布

给定提示 $`x`$、基础模型分布 $`p(y\mid x)`$、序列奖励 $`r(x,y)`$ 和奖励温度 $`\tau`$，考虑在完整序列分布上的
KL 正则化目标：

```math
\max_{\pi(\cdot\mid x)}
\left\{
\sum_y \pi(y\mid x)r(x,y)
-\tau D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\,\|\,p(\cdot\mid x)\right)
\right\},
\qquad \sum_y\pi(y\mid x)=1.
```

加入归一化约束的拉格朗日乘子后，一阶条件给出

```math
\pi_r(y\mid x)
=\frac{p(y\mid x)\exp\{r(x,y)/\tau\}}
       {\sum_{y'}p(y'\mid x)\exp\{r(x,y')/\tau\}}.
```

`is` 与 `mh` 以这一分布为目标，有限候选数或有限更新轮数产生近似误差；`mh_power` 的目标为幂分布 $`p(y\mid x)^\alpha`$。
IS 的可选提前选择还会引入前缀估计与完成时间相关的偏差。其余算法是生成与选择基线；奖励只用于打分，不回传给模型。

## 快速开始

```bash
python -m inference_scaling                                   # 默认：--algorithm is --model ar --reward consilience --dataset gsm8k
python -m inference_scaling --algorithm best_of_n --reward verifier
python -m inference_scaling --algorithm mh_power --dataset math500
```

| 参数 | 取值 | 默认 |
| --- | --- | --- |
| `--algorithm` | `sample`、`greedy`、`beam`、`best_of_n`、`mh`、`mh_power`、`is` | `is` |
| `--model` | `ar`（兼容既有命令；具体权重由 `ar.model` 指定） | `ar` |
| `--reward` | `verifier`、`self_certainty`、`consilience`；只用于 `best_of_n`、`mh`、`is` | `consilience` |
| `--dataset` | `gsm8k`、`math500` | `gsm8k` |
| `--output` | 结果根目录 | `results` |

没有其他命令行参数。块长、候选数、温度、引擎与优化选项都在设置文件中，缺少、多出或类型不符的字段在加载模型前
报错。每个字段的含义见 [SETTINGS.md](docs/SETTINGS.md)。默认 AR 设置使用 vLLM、BF16 和
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) 的思考模式；卡数须按权重、缓存及上下文容量验证。配置依据见
[SETTINGS.md](docs/SETTINGS.md#ar-defaults)。

### 长度与采样配置

| 配置位置 | 含义 | 当前值 |
| --- | --- | --- |
| `ar.max_new_tokens` | 单条生成总长（thinking + 最终内容）；`null` 按实际剩余上下文计算 | `null` |
| `ar.engine.context_window` | 输入与输出合计的容量；`null` 读取模型/引擎配置，Qwen3.8-27B 原生为 262144 | `null` |
| `ar.thinking` | 共用的思考档位、预算和别名；两项选择均为 `null` 时用官方默认 | `xhigh`，thinking 上限 131072 |
| `ar.sampling` / `ar.model.token_penalty` | 共用采样策略与词表惩罚 | 温度 1、完整词表；8 词、强度 1 |

数据集只配置题目、提示与判定，不设置生成长度。模型实际上下文不足时，剩余空间会进一步限制输出；
评测的 `trace.generation_budget` 和服务的 `scaling.generation_budget` 记录请求上限、生效上限及限制原因。
服务请求的 `max_tokens` / `max_completion_tokens` 限制生成总长；`reasoning_effort` 同时选择模板强度和 thinking 预算：
`low` 为 4096，`medium` 为 16384，显式 `xhigh` 为 262144，`none` 关闭思考。
`thinking_budget` 可单独指定思考上限，与 `reasoning_effort` 互斥。思考预算耗尽后关闭思考段并继续最终内容，生成总长仍优先。
完整映射和实现边界见 [思考预算](docs/SETTINGS.md#arthinking)。
上述上限均不代表 IS 全部候选的总 token 预算。性能测量脚本也继承同一配置，显式参数和生效配置随结果保存。
服务的 TTFT、TPS、多副本吞吐和正确率对照使用[服务性能入口](src/inference_scaling/serve/README.md#性能测量)。

## 算法

| 算法 | 实现 |
| --- | --- |
| `sample` | 按共用采样策略正常生成一次，作为 Base |
| `greedy` / `beam` | 贪心解码 / token 级 beam search |
| `best_of_n` | 独立生成 N 个样本后按奖励选择；verifier 投票时为多数投票 |
| `mh` | 奖励目标的后缀 MH，可选完整候选池和冻结历史 proposal |
| `mh_power` | 幂目标的后缀 MH |
| `is` | 保留完整 rollout 的条件 IS；主推加速方向为[两阶段探索](docs/methods/ALGORITHMS.md#is-two-stage)和[所选补全完成后结束](docs/methods/ALGORITHMS.md#is-selection-ready) |

原理、步骤与实现见[算法文档](docs/methods/ALGORITHMS.md)。算法层只接收 `reward(prompt_tokens, completion_tokens)`，
不接触数据集或文本解析；共享的 SIR 选择、IS 权重和 MH 接受核位于 [`shared/sampling/`](src/inference_scaling/shared/sampling/)。

## 奖励

| 奖励 | 定义 | 实现 |
| --- | --- | --- |
| `verifier` | 按最终答案评分。默认对照参考答案（oracle：正确为 1，否则为 0）；`rewards.verifier.source = "vote"` 或数据集没有参考答案时改为投票：与模型自身答案一致的比例（`best_of_n` 在候选之间投票，`is`/`mh` 与冻结的 `pool_size` 个独立样本比较） | [`app/rewards.py`](src/inference_scaling/app/rewards.py)、[`shared/rewards/vote.py`](src/inference_scaling/shared/rewards/vote.py) |
| `self_certainty` | [Self-Certainty](https://arxiv.org/abs/2502.18581)：对评分段取全词表 $`D_{\mathrm{KL}}(U\Vert p_t)`$ 的均值；`scope` 选 `thinking` 或 `full`。Ascend 用 `scalar_self_certainty` 随生成返回标量，GPU 用 `fused_logprobs`，约束见[设置](docs/SETTINGS.md) | [`arllm/rewards/intrinsic.py`](src/inference_scaling/arllm/rewards/intrinsic.py) |
| `consilience` | [Consilience](https://arxiv.org/abs/2608.09898) 置信度轨迹：末段 top-$`K`$ 置信度均值减去若干倍首段均值，默认只评思考段（AR） | 同上及 [`shared/rewards/consilience.py`](src/inference_scaling/shared/rewards/consilience.py) |

`is` 与 `mh` 所用的奖励都是逐序列的固定分数（投票用冻结样本池），因此条件 IS 可以复用保留序列的奖励，MH 的
接受率只含奖励差。温度写在各奖励的设置中。答案文本取思考段之后的内容；`thinking_mode = "enabled"` 时未完成的思考没有最终答案。

## 数据集

[`datasets/`](src/inference_scaling/datasets/) 为每个数据集提供题目、提示模板、答案规则（投票比较答案用）与判定器，
评测总是用数据集判定器。`gsm8k` 读取 OpenAI 官方拆分并校验 SHA-256，按最终数值判定；`math500` 读取固定提交的
MATH-500，按学科×难度分层抽题，用 [Math-Verify](https://github.com/huggingface/Math-Verify) 在独立进程中判定。

## 输出

每次运行写入 `<output>/<数据集>/<模型族>/<算法>[-<奖励>]/<指纹>/`：

| 文件 | 内容 |
| --- | --- |
| `manifest.json` | 命令行选择、完整设置及其哈希、有效设置、git 提交与是否有未提交改动、依赖版本与硬件、模型权重与元数据哈希、数据文件哈希与所选题目、创建时间 |
| `records.jsonl` | 每题每次抽取一行：问题、参考答案、输出（全文、思考段、内容、token 数、是否以 EOS 结束）、判定出的答案、是否可解析与正确、输出的奖励、算法轨迹、按阶段（奖励池、搜索、收尾）与模型的计算量、回退原因、耗时 |
| `executions.json` | 每次实际执行的状态、已保存的题目/抽取编号、运行级前向 token 与 FLOPs、墙钟时间；续跑追加，已完成结果直接读取时不新增 |
| `summary.json` | 准确率与 Wilson 95% 区间、`draws > 1` 时的 pass@k、奖励统计、计算量、单题耗时统计、输出长度、失败计数及运行级汇总 |

`execution_totals` 累计该结果目录所有执行的计算量与墙钟时间，包含失败尝试的已测成本；时间包含推理、判定和结果写入，
排除模型加载。吞吐量为已保存输出的 token 数（或记录数）除以累计墙钟时间，候选与 rollout 的计算计入成本。
逐题成本统一统计奖励池、候选与补全生成、奖励重评分及正文续写。串行运行使用后端计数器差值；并发运行要求每次调用均有请求级计数，
任一调用缺少计数时，该题的 `cost` 为 `null`；
只要有一条记录缺少成本，`cost_total_scope` 就是 `all_executions`，`cost_total` 改用运行级总量，否则为
`selected_records`，汇总当前选定记录。`cost_attributed_records` 给出有逐题成本的记录数。
减少 `draws` 只改变质量统计范围，运行级汇总仍覆盖全部历史执行。进程异常退出留下 `running` 记录，或历史记录缺少成本时，
`complete_accounting = false`，已知成本仅为部分统计。

指纹由有效设置、模型与数据哈希和包源码哈希决定。相同指纹的目录会续跑缺少的记录；增加 `run.draws` 也在原目录继续。
IS 历史步骤仅保存长度、权重、奖励与选择索引，完整 token 保留在当前序列及本轮候选中。MH 当前状态、草稿重放和显式冻结 proposal 池仍按算法需要保留。
计算量以前向 token 位置计：FLOPs 约为 `2 × 参数量 × 前向 token 位置数`。

## 执行优化

| 优化 | 设置 |
| --- | --- |
| 跨题连续批处理 | `ar.engine.continuous_batching.workers > 1` |
| IS 前缀就绪即释放分支、共享并发上限与长尾调度 | `ar.engine.rollout_scheduler`（默认启用；异步 vLLM，非融合统计模式；[实现与分布保证](docs/methods/ROLLOUT_SCHEDULING.md)、[INFRA 结果](docs/reports/RTX3090_ROLLOUT_INFRA.md#infra-report-rollout-scheduling)） |
| IS 必要事件交付、紧凑概率输出 | K=1 向算法交付完成结果，K>1 增加分支所需的前缀事件；`ar.engine.vllm.stream_interval = 32`、`flat_logprobs = true`（[CPU 输出层测试](docs/reports/RTX3090_ROLLOUT_INFRA.md#infra-cpu-output)） |
| vLLM 引擎、前缀缓存、融合概率（MH 的基础概率与 Self-Certainty 在采样同一步读出） | `ar.engine.backend = "vllm"`、`ar.engine.vllm.enable_prefix_caching`、`ar.engine.vllm.fused_logprobs` |
| 长序列分块评分 | `ar.engine.transformers.score_chunk_size` |
| 多尺度 MH 后缀 | `ar.algorithms.mh_power.suffix_schedule = "multiscale"` |
| 完整候选池后缀更新 | `ar.algorithms.mh.ensemble_size = 4`、`iterations = 1`、`suffix_schedule = "log_uniform"`；使用基础 proposal，关闭 `suffix_replay`。初始生成加一轮更新，共两阶段，最后按权重抽样 |
| 候选池流水线 | 完整候选池设置 `ar.algorithms.mh.execution = "pipeline"`；完整奖励足以确定接受结果时，提前开始该槽位的下一轮。保持原请求和逻辑更新顺序，默认 `batch`；要求独立初始化和普通基础 proposal，详见[跨轮执行](docs/methods/ALGORITHMS.md#ensemble-pipeline) |
| 分层候选池 | 上述完整候选池可设 `ar.algorithms.mh.initialization = "stratified"`，在 vLLM Ascend 上分层生成首轮候选；再设 `stratified_updates = true` 可在后续更新中保留各层。分别改善初始化覆盖、维持探索范围，默认均关闭；[初始化](docs/methods/ALGORITHMS.md#ensemble-initial-strata)、[持续分层](docs/methods/ALGORITHMS.md#ensemble-persistent-strata) |
| 排除原轨迹的后缀更新 | 完整候选池可设 `ar.algorithms.mh.nonself_updates = true`，避免提议相同序列；按第一次分歧复用前缀，以接受率修正保持目标。默认关闭，可与首轮分层结合，与持续分层互斥；[算法与适用范围](docs/methods/ALGORITHMS.md#ensemble-nonself) |
| 冻结历史 MH proposal | `ar.algorithms.mh.proposal = "frozen_history"` |

互相冲突的组合（如融合概率与词表惩罚）在加载模型前报错；与所选算法无关的优化不生效。

## 历史实现

已移除的实现保存在独立归档分支
[`archive/legacy-dllm-training-is`](https://github.com/xchang1121/inference_scaling/tree/archive/legacy-dllm-training-is)。
归档以主线精简前的 `62af76c` 为基线，包含完整源码、依赖、设置、测试和原文档：

- dLLM 推理与 LLaDA 后端；
- GRPO、VRPO 训练及数据准备；
- IS 的同块补全复用、数量与权重配额、超发、末条随机筛选、单池及 Metropolized 块选择。

归档独立于主线的安装与测试。历史实验报告与图片仍保留在主线；当前快速路径集中在两阶段探索与可选提前选择。
完整池的[Metropolized 选择核](docs/methods/ALGORITHMS.md#is-metropolized)已单独恢复为非默认选项，用于质量与成本对照；其余历史组合保留在归档。
在单独目录中获取归档：

```bash
git clone --branch archive/legacy-dllm-training-is --single-branch \
  https://github.com/xchang1121/inference_scaling.git inference_scaling-archive
```

## 推理服务

```bash
python -m pip install -e ".[serve]"   # 另按硬件安装 vLLM（NVIDIA）或 vLLM + vllm-ascend（昇腾）
python -m inference_scaling.serve
```

OpenAI 与 Anthropic 兼容的接口；每个请求按配置在思考段上做条件 IS（Consilience 加权）后作答。安装、配置、启动、
接入网关与限制见 [serve/README.md](src/inference_scaling/serve/README.md)。

## 安装

当前 Qwen3.8-27B 推理环境：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,gpu,evaluation]" # Transformers 后端
python -m pip install -e ".[dev,vllm,serve,evaluation]" # vLLM（Linux 或 WSL2）
```

## 测试与目录

```bash
python -m pytest
```

| 路径 | 内容 |
| --- | --- |
| `src/inference_scaling/app/` | 推理入口：命令行、设置 schema、运行与续跑、结果记录、算法组装、奖励绑定 |
| `src/inference_scaling/datasets/` | 数据集：题目、提示、答案规则与判定器 |
| `src/inference_scaling/arllm/` | AR-LLM：`algorithms/`（MH、条件 IS）、`backends/`（Transformers、vLLM、连续批处理与包装器）、`rewards/`（Self-Certainty、Consilience） |
| `src/inference_scaling/shared/` | 公共组件：`sampling/`（SIR、IS 权重、MH 接受核）、`model/`（加载、提示、生成上限、思考段解析）、`rewards/`（投票与 Consilience 算术） |
| `settings/` | 推理设置 |
| `tests/` | 分布、实现一致性、端到端运行与结果处理测试 |
| `docs/` | [设置说明](docs/SETTINGS.md)、[算法](docs/methods/ALGORITHMS.md)与实验报告 |
| `results/` | 运行结果，Git 忽略 |
| `online-speculation/` | 独立的在线推测解码项目 |

测试覆盖权重与选择、思考边界、异步资源清理、概率接口和计量；相同公共流程使用代表配置验证。

## 质量与性能结果

[配对评测入口](docs/SETTINGS.md)对每题运行 Base 与配置中的快速两阶段 IS，交替顺序并直接判题，报告
`IS 总时延 / Base 总时延`、正确率、生成量与计算量。固定块 IS 退出真机性能对照；其小型正确性回归保留。

[Qwen3.8-27B IS 服务延迟与吞吐](docs/reports/QWEN38_IS_SERVING.md)记录当前模型的卡数容量、思考与内容 TTFT、
输出吞吐和实际生成量；区分已测结果、交付延迟改善与尚未验证的吞吐收益。

[100 题的选择策略与奖励对照](docs/reports/QWEN38_IS_SELECTION.md)比较 Base、快速 IS、完整池 SIR、
逐条与按块 Metropolized、Best-of-4 + Consilience，以及快速 IS 和三种 Metropolized 的全词表 Self-Certainty 版本，
共 12 种方法。Base / 快速 Consilience IS / 快速 Self-Certainty IS 为 92% / 93% / 94%，按块 Self-Certainty K=2 为 95%；
后者比快速 Self-Certainty IS 净多对 1 题，耗时约 +20%、FLOPs 约 +75%，准确率差区间包含 0。
报告统一汇总 100 题的答案、论证审核、配对区间和成本；默认设置保持不变。

## 历史结果

[IS 加速研究总结](docs/reports/ASCEND_RESEARCH_SUMMARY.md)汇总2026年10月18日至28日的尝试、已合入收益、负收益方案及证据范围。

[GSM8K 算法与准确率](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md)、
[Qwen3 / MATH-500 思考模式](docs/reports/QWEN3_MATH500_REASONING.md)与
[推理成本与执行效率](docs/reports/RTX3090_ROLLOUT_INFRA.md)三份报告保留统一入口之前实验得到的结论与图；
已删除方法只保留说明，其余方法给出在当前实现上复现的命令与设置。

[昇腾调度验证](docs/reports/ASCEND_IS_INFRA.md)记录探索轮数、末条补全筛选、同块补全复用、完成配额与超发的单题对照，包含耗时回退、计算量与有限预算限制。
其中旧参数扫描与逐轨迹比较入口可从历史版本恢复；当前 `scripts/benchmark_is_gpu.py` 默认进行 Base/快速 IS 配对评测，
`--study selection` 增加上述完整池质量与成本对照。
