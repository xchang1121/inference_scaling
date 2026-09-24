# inference_scaling

本仓库实现自回归语言模型（AR-LLM）与掩码扩散语言模型（dLLM）的推理扩展与训练对照。推理只有一个入口：
命令行选择算法、模型族、奖励和数据集，其余参数全部写在 [`settings/inference.json`](settings/inference.json)。
训练（GRPO、VRPO、数据与权重下载）是独立的一套入口，读取 [`settings/training.json`](settings/training.json)。

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

`is` 与 `reward_mh` 直接对这一分布采样；`mh` 采样幂分布 $`p(y\mid x)^\alpha`$；其余算法是生成与选择基线。
奖励只用于打分，不回传给模型。

## 快速开始

```bash
python -m inference_scaling                                   # 默认：--algorithm is --model ar --reward vote --dataset gsm8k
python -m inference_scaling --algorithm best_of_n --reward verifier
python -m inference_scaling --algorithm mh --dataset math500
python -m inference_scaling --algorithm is --model dllm --reward verifier --output results
```

| 参数 | 取值 | 默认 |
| --- | --- | --- |
| `--algorithm` | `sample`、`greedy`、`beam`、`best_of_n`、`mh`、`reward_mh`、`is` | `is` |
| `--model` | `ar`、`dllm`（模型族；具体模型在 `settings/inference.json` 的 `ar.model` / `dllm.model`） | `ar` |
| `--reward` | `verifier`、`vote`、`logprob`、`consilience`；只用于 `best_of_n`、`reward_mh`、`is` | `vote` |
| `--dataset` | `gsm8k`、`math500` | `gsm8k` |
| `--output` | 结果根目录 | `results` |

没有其他命令行参数。块长、候选数、温度、引擎与优化选项都在设置文件中，缺少、多出或类型不符的字段在加载模型前
报错。每个字段的含义见 [SETTINGS.md](docs/SETTINGS.md)。例如对齐模型的采样只需在 `ar.model.adapter` 填入
GRPO 适配器，再运行 `--algorithm sample` 或 `--algorithm greedy`。

## 算法

| 算法 | AR-LLM | dLLM |
| --- | --- | --- |
| `sample` | 基础策略采样一次 | 按 `dllm.sampling` 分块解码一次 |
| `greedy` | 贪心解码 | 温度 0 的分块解码 |
| `beam` | token 级 beam search | 按轨迹概率保留的分块 beam |
| `best_of_n` | $`N`$ 个样本中按奖励选一个；`vote` 时取得票最多的答案 | 同左 |
| `mh` | [幂目标后缀 MH](docs/methods/ALGORITHMS.md)，可选 `multiscale` 后缀长度分布 | 反向轨迹幂 MH |
| `reward_mh` | 目标 $`p\exp\{r/\tau\}`$ 的后缀 MH；proposal 为基础策略或冻结历史混合 | 独立 MH；proposal 为基础策略或冻结历史轨迹混合 |
| `is` | 保留完整序列的条件 IS：`fixed` 固定候选数 M、补全数 K、块长 B，或在前向 token 预算内逐块重新规划（[BUDGET.md](docs/methods/BUDGET.md)） | 条件扩散 IS；补全来自主模型或早退 proposal（可做轨迹概率校正与截断） |

原理、步骤与实现见[算法文档](docs/methods/ALGORITHMS.md)。算法层只接收 `reward(prompt_tokens, completion_tokens)`，
不接触数据集或文本解析；共享的 SIR 选择、IS 权重和 MH 接受核位于 [`shared/sampling/`](src/inference_scaling/shared/sampling/)。

## 奖励

| 奖励 | 定义 | 实现 |
| --- | --- | --- |
| `verifier` | 外部奖励来源，即只有模型自身时拿不到的信息：数据集判定器对照参考答案（正确/错误/无答案三个取值）、Python 工厂 $`r=f(x,y)`$（如外部评分模型）或常数 | [`shared/rewards/verifier.py`](src/inference_scaling/shared/rewards/verifier.py) |
| `vote` | `best_of_n`：候选互相投票，平票在最高票中按种子随机选；`is`/`reward_mh`：与冻结的 `pool_size` 个独立样本答案一致的比例 | [`shared/rewards/vote.py`](src/inference_scaling/shared/rewards/vote.py) |
| `logprob` | 有效输出 token 的平均对数概率（AR） | [`arllm/rewards/intrinsic.py`](src/inference_scaling/arllm/rewards/intrinsic.py) |
| `consilience` | [Consilience](https://arxiv.org/abs/2608.09898) 置信度轨迹：末段 top-$`K`$ 置信度均值减去若干倍首段均值，默认只评思考段（AR） | 同上及 [`shared/rewards/consilience.py`](src/inference_scaling/shared/rewards/consilience.py) |

四种奖励都是逐序列的固定分数，因此条件 IS 可以复用保留序列的奖励，MH 的接受率只含奖励差。温度写在各奖励的
设置中。答案文本取思考段之后的内容；`thinking_mode = "enabled"` 时未完成的思考没有最终答案。

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
| `summary.json` | 准确率与 Wilson 95% 区间、`draws > 1` 时的 pass@k、奖励统计、计算量与时间合计、输出长度、失败计数 |

指纹由有效设置、模型与数据哈希和包源码哈希决定。相同指纹的目录会续跑缺少的记录；增加 `run.draws` 也在原目录继续。
计算量以前向 token 位置计：FLOPs 约为 `2 × 参数量 × 前向 token 位置数`（LLaDA 用激活参数量）。

## 执行优化

| 优化 | 设置 |
| --- | --- |
| 跨题连续批处理 | `ar.engine.continuous_batching.workers > 1` |
| vLLM 引擎、前缀缓存、同步引擎上 MH 的融合概率 | `ar.engine.backend = "vllm"`、`ar.engine.vllm.enable_prefix_caching`、`ar.engine.vllm.mh_fused_logprobs` |
| 长序列分块评分 | `ar.engine.transformers.score_chunk_size` |
| 多尺度 MH 后缀 | `ar.algorithms.mh.suffix_schedule = "multiscale"` |
| 冻结历史 MH proposal | `ar.algorithms.reward_mh.proposal = "frozen_history"`（dLLM 同名字段） |

互相冲突的组合（如异步引擎上的融合概率、预算规划的 IS 配思考段采样）在加载模型前报错；与所选算法无关的优化不生效。

## 训练

```bash
python -m training
```

按 `settings/training.json` 的 `stages` 依次运行，没有命令行参数，各阶段可续跑：

| 阶段 | 内容 | 代码 |
| --- | --- | --- |
| `download` | 下载并校验 GSM8K 训练/测试拆分与固定版本的模型权重（Hugging Face 或 ModelScope） | [`training/download.py`](training/download.py) |
| `grpo` | 在 GSM8K 训练集上训练 GRPO LoRA，奖励为配置的 verifier；记录墙钟、生成 token、显存与 GPU 功率积分 | [`training/grpo.py`](training/grpo.py) |
| `vrpo_preferences` | 用 LLaDA 生成候选并按 verifier 选出偏好对 | [`training/vrpo.py`](training/vrpo.py) |
| `vrpo` | 方差缩减偏好优化（[VRPO](https://arxiv.org/abs/2505.19223)）：以掩码扩散 ELBO 代替序列对数似然 | 同上及 [`dllm/training/`](src/inference_scaling/dllm/training/) |

训练得到的适配器填入推理设置的 `ar.model.adapter` 或 `dllm.model.adapter` 即可评测。

## 安装

AR-LLM 与官方 LLaDA-MoE 需要不同的 Transformers 版本，两个模型族分别安装到各自的 Python 环境：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,gpu,training,evaluation]"      # AR-LLM
python -m pip install -e ".[dev,dllm,dllm-training,evaluation]" # LLaDA-MoE 与 VRPO（另一个环境）
python -m pip install -e ".[dev,vllm]"                          # vLLM（Linux 或 WSL2 的独立环境）
```

## 测试与目录

```bash
python -m pytest
```

| 路径 | 内容 |
| --- | --- |
| `src/inference_scaling/app/` | 推理入口：命令行、设置 schema、运行与续跑、结果记录、两个模型族的算法组装、奖励绑定 |
| `src/inference_scaling/datasets/` | 数据集：题目、提示、答案规则与判定器 |
| `src/inference_scaling/arllm/` | AR-LLM：`algorithms/`（MH、条件 IS、预算 IS）、`backends/`（Transformers、vLLM、连续批处理与包装器）、`rewards/`（logprob、Consilience） |
| `src/inference_scaling/dllm/` | LLaDA：`algorithms/`（条件扩散 IS、MH、分块 beam）、`backends/`、`training/`（VRPO） |
| `src/inference_scaling/shared/` | 两侧共用：`sampling/`（SIR、IS 权重、MH 接受核）、`budget/`（预算规划）、`model/`（加载、提示、生成上限、思考段解析）、`rewards/`（verifier、投票、Consilience 算术） |
| `settings/` | 推理与训练设置 |
| `training/` | 训练入口与各阶段 |
| `tests/` | 分布、实现一致性、端到端运行与结果处理测试 |
| `docs/` | [设置说明](docs/SETTINGS.md)、[算法](docs/methods/ALGORITHMS.md)、[预算](docs/methods/BUDGET.md)与实验报告 |
| `results/` | 运行结果，Git 忽略 |
| `online-speculation/` | 独立的在线推测解码项目 |

[`tests/test_repository_layout.py`](tests/test_repository_layout.py) 检查分层：底层不依赖上层，两个模型族互不依赖。

## 历史结果

[GSM8K 算法与准确率](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md)、
[Qwen3 / MATH-500 思考模式](docs/reports/QWEN3_MATH500_REASONING.md)与
[推理成本与执行效率](docs/reports/RTX3090_ROLLOUT_INFRA.md)三份报告由统一入口之前的实验脚本产生，其中部分方法和优化
已删除。报告中的数字与命令对应 git 标签 `pre-unified-cli`，从该标签可完整复现。
