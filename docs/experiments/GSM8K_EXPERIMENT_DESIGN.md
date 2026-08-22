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
| 迭代条件 IS 筛选 | 9 个不同候选-rollout 状态；比较 `(pool, updates)=(9,1),(5,2),(3,4)` |
| 幂分布 MH | $`\alpha=4`$；16 个长度阶段；每阶段 3 次更新 |
| pass@k | 每题 8 个独立 draw |

`standard` 使用 256 token、20 beams、Best-of-20、$`M=15,K=3,I=4`$；`full` 使用 512 token、
20 beams、Best-of-30、$`M=15,K=3,I=4`$。两者的 MH 每阶段更新 10 次。

<a id="method-labels"></a>
## 方法与目标

| 报告名称 | AR-LLM 标识 | dLLM 标识 | 候选或 proposal | 奖励、目标与修正 |
| --- | --- | --- | --- | --- |
| Base | `base` | `base` | 基础模型直接采样 | 基础分布，温度 1 |
| Beam-8 | `beam` | `block_beam` | 累计概率最高的前缀或 block | 确定性 beam search |
| 自一致性投票-8 | `best_of_n` | `best_of_n` | 8 条独立完整生成 | 返回数值众数对应的序列 |
| 幂分布 MH | `mh` | `trajectory_power_mh` | AR 后缀或 dLLM 反向轨迹 proposal | 目标为 $`p_{\mathrm{base}}^4`$；使用完整 Hastings 比 |
| 标准条件 IS | `conditional_is` | `conditional_is` | 主模型候选；主模型 rollout | cumulative self-consistency；on-policy |
| 迭代条件 IS | `iterated_conditional_is` | 本轮不运行 | 主模型候选；主模型 rollout | 独立 pilot 冻结数值众数；有限池 i-SIR |
| 低成本 proposal 条件 IS | `conditional_is_small_proposal` | `conditional_is_reduced_layer_proposal` | 主模型候选；0.5B 或低层 rollout | 用主模型概率除以实际 rollout proposal 概率 |
| 未校正 rollout 加权 | `conditional_is_small_proposal_uncorrected` | `conditional_is_reduced_layer_proposal_uncorrected` | 与上一行相同 | 省略 $`p/q`$；目标为[式 (12)](../methods/ALGORITHMS.md#alg-uncorrected-rollout) |
| RL 参数随机采样 | `rl_sample` | `vrpo_sample` | GRPO 或 VRPO 训练后的参数 | 温度 1 |
| RL 参数贪心解码 | `rl_greedy` | `vrpo_greedy` | GRPO 或 VRPO 训练后的参数 | 每一步取最大概率项 |
| verifier-MH | `verifier_mh` | `verifier_mh` | 完整序列 MH proposal | 数值正确性奖励；完整 Hastings 比 |
| 标准 verifier-IS | `verifier_conditional_is` | `verifier_conditional_is` | 主模型候选与 rollout | 数值正确性奖励；on-policy |
| 低成本 proposal verifier-IS | `verifier_conditional_is_small_proposal` | `verifier_conditional_is_reduced_layer_proposal` | 主模型候选；低成本 rollout | 数值正确性奖励；乘 $`p/q`$ |

AR 的“低成本 proposal”指 Qwen2.5-0.5B；dLLM 对应 LLaDA 共享前缀层的低层 proposal。`unclipped`
后缀表示不截断 log importance ratio；`uncorrected` 后缀表示省略主模型轨迹重评分。replay 与动态候选的
成对标识列在[对应实验设置](#replay-与动态候选)。

主要比较：

| 比较 | 方法 | 统计范围 |
| --- | --- | --- |
| 最终任务质量 | Base、搜索、自一致性、幂分布 MH、条件 IS、迭代条件 IS、GRPO | 准确率与计算量 |
| 共享奖励 | verifier-MH、verifier-IS、GRPO | 准确率与经验答案分布 |
| off-policy | 标准 IS、0.5B rollout proposal IS、未校正 rollout 加权 | 准确率、ESS、分模型 FLOPs |
| replay 与动态候选 | fresh、warm、动态 proposal、方差—成本分配 | 准确率、ESS、复用率、冷启动/在线成本 |

共享奖励和动态候选使用 test gold 数值，标记为 oracle 诊断。部署质量实验使用 cumulative
self-consistency 或模型置信度。

## 奖励

| 奖励 | 定义 | gold access |
| --- | --- | --- |
| 数值正确性 | 解析最终数值，与标准答案比较，取 0/1 | 是 |
| cumulative self-consistency | 按已评估数值累计众数，匹配取 1 | 无 |
| frozen consensus | 从独立 base pilot completion 取数值众数，随后固定“是否匹配众数”的 0/1 奖励 | 无 |
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

未校正 rollout 加权设置 `apply_importance_correction=false`，候选权重为

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
prefill、decode、完整序列评分和 target speculative verification；墙钟排除模型与数据加载。该主干估算不计
attention 的长度二次项、逐元素 kernel、tokenization、CPU 奖励解析与调度；墙钟仍包含后四项的执行影响。

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

<a id="infra-labels"></a>
## 执行实验标签

| 机制 | 优化路径 | 对照路径 | 固定口径 |
| --- | --- | --- | --- |
| 批处理 | AR 连续批处理 / dLLM block 批处理 | 同方法逐 prompt | 请求、seed、模型与 token 上限 |
| 部分续跑 | 保存未完成 token / 已提交 block 后继续 | 从原始前缀重新生成 | 相同最终请求集合 |
| 流式奖励 | completion 完成后立即提交 verifier | 整批生成后提交 | 相同 verifier 延迟与样本 |
| 历史草稿 | AR token tree / dLLM 轨迹 cache | 普通生成 | target 校正后的相同生成分布 |
| MH 预取 | 并行预取接受与拒绝分支 | 普通 MH | 相同更新数与实际消费分支 |
| delayed acceptance | surrogate 早拒绝后再调用精确奖励 | 每个 proposal 调用精确奖励 | 相同 proposal 和精确目标 |
| replay-mixture MH | base 与冻结历史 proposal 的 mixture | base proposal | 正反 mixture 概率均进入 Hastings 比；cache build 分列 |
| warm replay | 已建库 history + fresh tail | fresh-only | 候选、总 rollout 数与在线请求固定 |
| progressive IS | pilot 冻结配额，独立 evaluation 计算权重 | 每候选固定 rollout 数 | pilot 与 evaluation 分列 |
| SMC forest | 条件后缀或轨迹 reservoir 复用 | 同一 SMC 的 fresh-only 路径 | 粒子数、lookahead 与 resampling 固定 |
| 执行后端 | AR vLLM / dLLM 批量 Transformers | AR Transformers / dLLM 逐请求 | 模型、dtype、设备、请求与统计设计固定 |

报告中的“便宜奖励”表示额外奖励延迟为 0；“0.2 s 奖励”表示每次调用加入 0.2 秒固定延迟；“在线”排除
cache build。所有因子均在表中明确分子与分母。

<a id="replay-labels"></a>
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

以下命令均从仓库根目录运行。统一入口负责准备数据与模型、续跑 RL 训练、执行所选实验组件、写入命令清单
并记录每个已完成子任务。AR 与 dLLM 可以使用不同 Python 解释器。

### 成对完整复现

```powershell
$env:AR_PYTHON = "C:\path\to\ar-python.exe"
$env:DLLM_PYTHON = "C:\path\to\dllm-python.exe"

python experiments\run_reproduction.py `
  --family both --stage all --profile full --tag full-reproduction
```

`full` 默认运行两侧全部公共组件；AR 额外支持 `vllm`。`smoke` 使用一题、缩短生成预算、一次 GRPO 更新和
CPU VRPO 反向传播预检，用途限于实现检查。

### 单侧与组件选择

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage inference --profile full --tag ar-full `
  --components quality replay dynamic_is async passk infra vllm

python experiments\run_reproduction.py `
  --family dllm --stage inference --profile full --tag dllm-full `
  --components quality replay dynamic_is async passk infra
```

| 组件 | 统计对象 |
| --- | --- |
| `quality`、`matched_target` | 主质量网格与共享奖励诊断 |
| `replay`、`dynamic_is` | fresh/warm replay、动态候选与预算分配 |
| `async`、`infra` | 批处理、部分续跑、流式奖励、MH 预取与 SMC |
| `passk`、`distribution` | 独立 draw 的 pass@$`k`$ 与答案分布 |
| `ablations`、`budget_curve`、`length_ablation` | 算法参数、计算预算与长度消融 |
| `vllm` | AR-LLM 的 Transformers/vLLM 成对执行检查 |

`--ar-methods` 与 `--dllm-methods` 选择具体方法；成对标识由
[`gsm8k_llada_moe_3090.toml`](../../configs/gsm8k_llada_moe_3090.toml)固定。`--dry-run` 只写命令清单，
适合在目标机器上检查解释器、路径和实验范围。

### 运行产物

| 模型族 | 原始记录与清单 | 正式汇总 |
| --- | --- | --- |
| AR-LLM | `results/gsm8k/<profile>/` 与所选 `summary-root` | `results/gsm8k_3090/` |
| dLLM | `results/reproduction/dllm/<tag>/` | 各组件目录中的 `summary.json` 或聚合 JSON |
| 成对调度 | `results/reproduction/<tag>/manifest.json` | 两侧子清单及已完成命令数 |

逐题 JSONL 按 fingerprint 续写；pass@$`k`$ 使用独立 chunks；汇总器核对数据行号、模型 revision、有效配置、
实现哈希、dtype、worker 与 GPU 数。图表由已验证汇总生成，结果文件索引见
[`results/README.md`](../../results/README.md)。

### 直接调用底层脚本

单实验脚本保留用于调试与定点重跑，其参数由两侧 suite 入口生成。正式复现优先使用统一入口，避免手工遗漏
汇总、fingerprint 或配对参数。具体下一层命令可通过 `--dry-run` 查看，无需在文档中维护第二份命令清单。

## 消融矩阵

- MH：$`\alpha\in\{1,2,4,8\}`$，每 block 更新数 $`\{1,2,5,10\}`$。
- 条件 IS：候选数 $`M`$、rollout 数 $`K`$、引导阶段数 $`I`$。
- 迭代条件 IS：固定 9 个不同候选-rollout 状态，比较一次性大池与多轮有限池复用。
- 搜索：Beam、Best-of-$`N`$ 与条件 IS 的质量—计算曲线。
- 奖励：平均 token log-probability、平均负熵、self-certainty、self-consistency、oracle correctness。
- off-policy：截断、未截断与未校正 rollout 加权。
- 生成：温度 $`\{0.7,1.0,1.5\}`$，最大长度 $`\{128,256,512\}`$。
- 执行：逐 prompt、连续批处理、fresh-only、warm replay。
- 动态候选：base fixed、replay-aware fixed、variance-cost allocation。
- 多次采样：Base、MH、GRPO 与三种条件 IS 的 8 draw pass@k。

## 完整性

原始 JSONL 按 manifest fingerprint 追加。fingerprint 包含有效配置、GSM8K 行号、模型权重和关键实现
文件 SHA-256。后处理器核对题目网格、manifest 和输入哈希后生成 `validated` 汇总。代码或配置变更使用
新 tag。
