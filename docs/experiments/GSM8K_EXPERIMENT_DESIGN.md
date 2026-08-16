# GSM8K 统一实验设计

本文件固定数据、模型、方法、预算、统计量、成本分母和复现流程。算法原理、实现、执行优化与计量口径见
[推理扩展算法：基础、原理与实现](../methods/ALGORITHMS.md)。

## 数据与配置

实验使用公开 [GSM8K](https://arxiv.org/abs/2110.14168)。训练集含 7,473 题，仅供 GRPO 训练；测试集含
1,319 题，用于准确率和分布评测。数据文件固定字节级校验和，训练入口检查 train/test 问题重合。

| profile | 样本 | 用途 |
| --- | ---: | --- |
| `quick` | 8 | 集成检查 |
| `gsm8k_3090_aligned` | 32 | 单卡正式实验 |
| `standard` | 128 | 较大样本实验 |
| `full` | 1,319 | 完整测试集 |

RTX 3090 对齐配置：

| 项目 | 固定值 |
| --- | --- |
| 基础模型 | [`Qwen/Qwen2.5-1.5B-Instruct`](https://arxiv.org/abs/2412.15115)，revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| rollout proposal | `Qwen/Qwen2.5-0.5B-Instruct`，revision `7ae557604adf67be50417f59c2c2f167def9a775` |
| GRPO | 同一 1.5B checkpoint 的 [LoRA](https://openreview.net/pdf?id=nZeVKeeFYf9)；205 步；每个 prompt 4 条 rollout |
| 硬件 | 单张 RTX 3090 24 GiB |
| 质量网格 dtype | FP32 |
| 最大生成长度 | 192 token |
| 条件 IS | 8 个候选；每候选 3 条 rollout；4 个引导阶段 |
| 幂分布 MH | $`\alpha=4`$；16 个长度阶段；每阶段 3 次更新 |
| pass@k | 每题 8 个独立 draw |

`standard` 使用 256 token、20 beams、Best-of-20、$`M=15,K=3,I=4`$；`full` 使用 512 token、
20 beams、Best-of-30、$`M=15,K=3,I=4`$。两者的 MH 每阶段更新 10 次。

## 方法与目标

| 标识 | 参数或状态 | 候选 | rollout / proposal | 奖励或目标 | 概率修正 |
| --- | --- | --- | --- | --- | --- |
| `base` | 1.5B base | — | — | base 分布 | 温度 1 |
| `beam` | 1.5B base | beam 前缀 | — | 累计 log-probability | beam search |
| `best_of_n` | 1.5B base | 独立完整生成 | — | 数值众数 | 选择 |
| `mh` | 完整序列 | — | 1.5B 后缀 | $`p_{\mathrm{base}}^4`$ | Hastings 比 |
| `conditional_is` | 1.5B base | 1.5B block | 1.5B completion | cumulative self-consistency | on-policy |
| `conditional_is_small_proposal` | 1.5B base | 1.5B block | 0.5B completion | cumulative self-consistency | 1.5B/0.5B 后缀比 |
| `conditional_is_small_proposal_uncorrected` | 1.5B base | 1.5B block | 0.5B completion | proposal-energy | [式 (12)](../methods/ALGORITHMS.md#alg-proposal-energy) |
| `rl_sample` | GRPO 参数 | — | — | 训练后策略 | 温度 1 |
| `rl_greedy` | GRPO 参数 | — | — | 训练后策略 | 逐 token argmax |
| `verifier_mh` | 完整序列 | — | 1.5B 后缀 | 数值正确性 | Hastings 比 |
| `verifier_conditional_is` | 1.5B base | 1.5B block | 1.5B completion | 数值正确性 | on-policy |
| `verifier_conditional_is_small_proposal` | 1.5B base | 1.5B block | 0.5B completion | 数值正确性 | 1.5B/0.5B 后缀比 |

主要比较：

| 比较 | 方法 | 统计范围 |
| --- | --- | --- |
| 最终任务质量 | Base、搜索、自一致性、幂分布 MH、条件 IS、GRPO | 准确率与计算量 |
| 共享奖励 | verifier-MH、verifier-IS、GRPO | 准确率与经验答案分布 |
| off-policy | 标准 IS、0.5B rollout proposal IS、proposal-energy | 准确率、ESS、分模型 FLOPs |
| replay 与动态候选 | fresh、warm、动态 proposal、方差—成本分配 | 准确率、ESS、复用率、冷启动/在线成本 |

共享奖励和动态候选使用 test gold 数值，标记为 oracle 诊断。部署质量实验使用 cumulative
self-consistency 或模型置信度。

## 奖励

| 奖励 | 定义 | gold access |
| --- | --- | --- |
| 数值正确性 | 解析最终数值，与标准答案比较，取 0/1 | 是 |
| cumulative self-consistency | 按已评估数值累计众数，匹配取 1 | 无 |
| 平均 token log-probability | 完整生成的平均选中 token log-probability | 无 |
| 平均负熵 | 完整生成的逐 token 负熵均值 | 无 |
| self-certainty | 逐 token $`D_{\mathrm{KL}}(U\|p_{\mathrm{base}})`$ 均值 | 无 |

后三种置信度奖励在每次候选决策内执行 min-max 归一化；常数信号统一置零。完整词表评分计入
token-slot/FLOPs。

## 概率设置

候选和 on-policy rollout 使用同一参考 sampling policy。非单位温度时，温度缩放后的完整支持策略定义
本轮参考分布；off-policy 后缀比在相同温度下计算。

0.5B rollout proposal 的默认 log 比值截断区间为 `[-10,10]`。原始比值、应用比值、截断次数和 ESS
进入结果。`importance_log_ratio_clip = null` 使用普通未截断重要性权重。

proposal-energy 设置 `apply_importance_correction=false`，候选权重为

```math
w_m=\frac1K\sum_{k=1}^K
\exp\!\left(\frac{r(z_m,u_{mk})}{\tau}\right),
\qquad
z_m\sim p_{\mathrm{1.5B}},\quad
u_{mk}\sim q_{\mathrm{0.5B}}(\cdot\mid z_m).
```

该路径的 base `score_calls`、`scored_tokens` 和评分 slots 为 0。

## 统计量

- pass@k 使用 [Chen et al. (2021)](https://arxiv.org/abs/2107.03374) 的无偏估计式。
- 单方法准确率区间使用 [Wilson (1927)](https://doi.org/10.1080/01621459.1927.10502953) 区间。
- 方法差异使用题目级配对 [bootstrap](https://doi.org/10.1214/aos/1176344552)。
- 经验答案分布使用 total variation（TV）和
  [Jensen--Shannon 散度](https://doi.org/10.1109/18.61115)，JS 单位为 bit。
- 每个 pass@k draw 使用独立候选、rollout 和 replay 状态。

## 计算量

模型 $`j`$ 的推理主干 FLOPs 估计为

```math
\widehat F_j=2N_jS_j,
```

其中 $`N_j`$ 为参数量，$`S_j`$ 为实际 forward token slots。1.5B 与 0.5B 分别计算后求和。计数覆盖
prefill、decode、完整序列评分和 target speculative verification；墙钟排除模型与数据加载。

GRPO 成本分为 rollout generation、reference scoring、policy forward/backward 和 AdamW adapter
update。gradient checkpointing 的 policy 路径按 forward、backward 与重算三个前向等价过程计量。
训练 manifest 保存样本、权重、版本、LoRA 参数量、completion、token、显存、墙钟和功率积分。

共享奖励目标为

```math
\max_\pi\ \mathbb E_\pi[R]-\beta D_{\mathrm{KL}}(\pi\|p_{\mathrm{base}}),
```

其无参数限制闭式解正比于 $`p_{\mathrm{base}}\exp(R/\beta)`$。累计成本比较为

```math
F_{\mathrm{GRPO\ train}}+QF_{\mathrm{GRPO\ infer}}
\quad\text{与}\quad
QF_{\mathrm{training\text{-}free}}.
```

准确率匹配的临界查询数要求配对准确率差落入预设容差；联合匹配还要求答案分布 TV/JS 通过阈值。

## 成本分母

| 指标 | 分子 / 分母 | 固定项 |
| --- | --- | --- |
| `compute_multiple_vs_base` | 方法 FLOPs / Base FLOPs | 样本、长度 |
| 小 proposal FLOPs 因子 | 标准条件 IS / 0.5B rollout proposal IS | 候选、rollout、block、seed、长度 |
| `runtime_multiple_vs_base` | 方法墙钟 / Base 墙钟 | 样本、硬件 |
| 连续批处理加速 | 逐 prompt 墙钟 / 批处理墙钟 | 方法、请求、seed |
| repeated-prefix KV | 逐 rollout prefill / 唯一前缀 prefill | 同一生成 batch |
| warm replay 在线因子 | fresh-only / warm online | 候选、$`H+F`$、block |
| warm replay 首次查询 | fresh-only / (cache build + warm online) | 同上 |
| 动态候选在线因子 | base candidate fixed / replay-aware fixed | evaluation 成本预算 |
| 最优预算在线因子 | replay-aware fixed / replay-aware optimal | candidate proposal、成本预算 |
| vLLM 加速 | Transformers / vLLM | 模型、dtype、GPU、数据、workload |

连续批处理结果同时保存 token 匹配、数值答案匹配、共同前缀和分叉题号。cache build、design、online
与 background drain 分列。

## replay 与动态候选

每条 evaluation 历史记录原子消费一次。benchmark 使用重复公开 prompt 与候选 seed 形成可控 replay key。

| 实验臂 | 候选 | history | fresh | design |
| --- | --- | --- | --- | --- |
| `base_candidate_fixed` | 1.5B base | 0 | 每个非终止候选 3 条 | 0 |
| `replay_aware_fixed` | `0.5 × base + 0.5 × proposal` | 命中时最多 2 条 | 补足至 3 条 | 0 |
| `replay_aware_optimal` | 同上 | 方差—成本配额 | 方差—成本配额 | 每来源 2 条 |

重复候选共享同一 replay key 的 evaluation 库存。预算代理将一条历史样本记为 1 个 base 重评分等价，
一条 fresh 样本记为

```math
1+\frac{P_{\mathrm{0.5B}}}{P_{\mathrm{1.5B}}}=1.3200.
```

最终成本采用实际 forward token slots 与参数量。配额冻结使用候选、策略版本、库存数量和 design
统计量；evaluation reward 在领取后进入最终估计。

## 复现

以下命令从仓库根目录运行。原始逐题记录位于 `results/gsm8k/<profile>/`；正式汇总位于
`results/gsm8k_3090/`。

### 准备与训练

```powershell
$env:PYTHONPATH = "src"
python experiments\prepare_gsm8k.py `
  --config configs\gsm8k_3090_aligned.toml

python experiments\train_gsm8k_grpo.py --resume auto
```

已有匹配 `configs/gsm8k_grpo.toml` 与 base revision 的 adapter 时，可直接进入推理实验。

### 主网格

```powershell
$env:PYTHONPATH = "src"
python experiments\run_gsm8k_suite.py `
  --config configs\gsm8k_3090_aligned.toml `
  --tag validated `
  --summary-root results\gsm8k_3090 `
  --with-matched-target `
  --with-replay `
  --with-dynamic-is `
  --with-async `
  --with-ablations `
  --with-budget-curve `
  --with-length-ablation `
  --ablation-limit 8
```

动态候选设置位于 `configs/gsm8k_3090_dynamic_is.toml`。vLLM 套件在 Linux/WSL2 上增加
`--backend vllm`。

### vLLM 成对测速

```bash
export PYTHONPATH=src
python experiments/run_vllm_backend_benchmark.py \
  --config configs/gsm8k_3090_aligned.toml \
  --limit 32 \
  --workers 8 \
  --tag rtx3090
```

汇总器核对数据、题号、权重、算法参数、dtype、worker、环境、代码哈希与 GPU 数。

### 汇总与重评分消融

```powershell
$env:PYTHONPATH = "src"
python experiments\summarize_gsm8k.py `
  --config configs\gsm8k_3090_aligned.toml `
  --tag validated `
  --output results\gsm8k_3090\gsm8k_3090_aligned_comparison_validated.json

python experiments\gsm8k_distribution_audit.py `
  --config configs\gsm8k_3090_aligned.toml `
  --problem-count 4 --draws 8 `
  --output results\gsm8k_3090\gsm8k_3090_aligned_distribution_audit_validated.json

python experiments\summarize_gsm8k_compute.py `
  --config configs\gsm8k_3090_aligned.toml `
  --tag validated `
  --training-cost models\Qwen2.5-1.5B-Instruct-GRPO-GSM8K\training_cost.json `
  --distribution-audit results\gsm8k_3090\gsm8k_3090_aligned_distribution_audit_validated.json `
  --output results\gsm8k_3090\gsm8k_3090_aligned_compute_validated.json

python experiments\summarize_gsm8k_ablations.py `
  --config configs\gsm8k_3090_aligned.toml `
  --output results\gsm8k_3090\gsm8k_3090_aligned_ablations_validated.json

$env:PYTHONPATH = "src;."
python experiments\gsm8k_reproduction.py `
  --config configs\gsm8k_3090_aligned.toml `
  --method verifier_conditional_is_small_proposal `
  --tag with-rescore-paired-validated --limit 32

python experiments\gsm8k_reproduction.py `
  --config configs\gsm8k_3090_aligned.toml `
  --method verifier_conditional_is_small_proposal `
  --tag no-rescore-validated --limit 32 `
  --disable-importance-correction

python experiments\summarize_gsm8k_verifier_rescoring.py
```

### pass@k

```powershell
$env:PYTHONPATH = "src"
python experiments\gsm8k_passk.py `
  --config configs\gsm8k_3090_aligned.toml `
  --limit 32 --draws 8 --workers 8 --tag validated `
  --output results\gsm8k_3090\gsm8k_3090_aligned_passk_validated.json

python experiments\gsm8k_is_passk.py `
  --config configs\gsm8k_3090_aligned.toml `
  --limit 32 --draws 8 --workers 8 --tag validated `
  --output results\gsm8k_3090\gsm8k_3090_aligned_is_passk_validated.json

python experiments\gsm8k_is_passk.py `
  --config configs\gsm8k_3090_aligned.toml `
  --limit 32 --draws 8 --workers 8 `
  --methods conditional_is_small_proposal_uncorrected `
  --tag is-uncorrected-validated `
  --output results\gsm8k_3090\gsm8k_3090_aligned_is_uncorrected_validated.json

python experiments\summarize_gsm8k_is_rescoring.py

python experiments\summarize_gsm8k_passk.py `
  results\gsm8k_3090\gsm8k_3090_aligned_passk_validated.json `
  results\gsm8k_3090\gsm8k_3090_aligned_is_passk_validated.json `
  --is-raw-chunks results\gsm8k_3090\gsm8k_3090_aligned_is_passk_validated.chunks.jsonl `
  --output results\gsm8k_3090\gsm8k_3090_aligned_passk_comparison_validated.json
```

### 图表

```powershell
$env:PYTHONPATH = "src"
python experiments\plot_gsm8k_quality_compute.py
python experiments\plot_gsm8k_passk.py
python experiments\plot_gsm8k_ablations.py
```

## 消融矩阵

- MH：$`\alpha\in\{1,2,4,8\}`$，每 block 更新数 $`\{1,2,5,10\}`$。
- 条件 IS：候选数 $`M`$、rollout 数 $`K`$、引导阶段数 $`I`$。
- 搜索：Beam、Best-of-$`N`$ 与条件 IS 的质量—计算曲线。
- 奖励：平均 token log-probability、平均负熵、self-certainty、self-consistency、oracle correctness。
- off-policy：截断、未截断与 proposal-energy。
- 生成：温度 $`\{0.7,1.0,1.5\}`$，最大长度 $`\{128,256,512\}`$。
- 执行：逐 prompt、连续批处理、fresh-only、warm replay。
- 动态候选：base fixed、replay-aware fixed、variance-cost allocation。
- 多次采样：Base、MH、GRPO 与三种条件 IS 的 8 draw pass@k。

## 完整性

原始 JSONL 按 manifest fingerprint 追加。fingerprint 包含有效配置、GSM8K 行号、模型权重和关键实现
文件 SHA-256。后处理器核对题目网格、manifest 和输入哈希后生成 `validated` 汇总。代码或配置变更使用
新 tag。
