# GSM8K 统一实验设计

本文件固定数据、模型、方法、预算、统计量、各项成本比的比较基准和复现流程。算法原理、实现、执行优化与计算量统计定义见
[推理扩展算法：基础、原理与实现](../methods/ALGORITHMS.md)。

当前正式实验范围为 Qwen2.5-1.5B 自回归路线。dLLM 只保留实现入口和轻量接口检查，不生成质量或性能结果。
Qwen2.5-0.5B 仅作为辅助提议模型、rollout 模型或推测解码草稿模型，其计算量单独报告。

## 数据与配置

实验使用公开 [GSM8K](https://arxiv.org/abs/2110.14168)。训练集含 7,473 题，仅供 GRPO 训练；测试集含
1,319 题，用于准确率和分布评测。数据文件固定字节级校验和，训练入口检查 train/test 问题重合。

| 配置档 | 样本 | 用途 |
| --- | ---: | --- |
| `quick` | 8 | 集成检查 |
| `gsm8k_3090_aligned` | 32 | 单卡正式实验 |
| `standard` | 128 | 较大样本实验 |
| `full` | 1,319 | 完整测试集 |

RTX 3090 统一配置：

| 项目 | 固定值 |
| --- | --- |
| 基础模型 | [`Qwen/Qwen2.5-1.5B-Instruct`](https://arxiv.org/abs/2412.15115)，模型版本（revision）`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| rollout 提议模型 | `Qwen/Qwen2.5-0.5B-Instruct`，模型版本（revision）`7ae557604adf67be50417f59c2c2f167def9a775` |
| GRPO | 同一 1.5B 模型检查点的 [LoRA](https://openreview.net/pdf?id=nZeVKeeFYf9)；205 步；每个提示 4 条 rollout |
| 硬件 | 单张 RTX 3090 24 GiB |
| 质量实验数值类型（`dtype`） | FP32 |
| 最大生成长度 | 192 token |
| 条件 IS | 8 个候选；每候选 3 条 rollout；4 个引导阶段 |
| 迭代条件 IS 筛选 | 9 个不同候选-rollout 状态；比较“候选池大小、更新数”为 `(9,1),(5,2),(3,4)` |
| 幂分布 MH | $`\alpha=4`$；16 个长度阶段；每阶段 3 次更新 |
| pass@k | 每题 8 次独立重复采样 |

`standard` 使用 256 token、20 beams、Best-of-20、$`M=15,K=3,I=4`$；`full` 使用 512 token、
20 beams、Best-of-30、$`M=15,K=3,I=4`$。两者的 MH 每阶段更新 10 次。

<a id="method-labels"></a>
## 方法与目标

| 报告名称 | AR-LLM 标识 | dLLM 标识 | 候选或 proposal | 奖励、目标与修正 |
| --- | --- | --- | --- | --- |
| Base | `base` | `base` | 基础模型直接采样 | 基础分布，温度 1 |
| Beam-8 | `beam` | `block_beam` | 累计概率最高的前缀或生成块 | 确定性 beam search |
| 自一致性投票-8 | `best_of_n` | `best_of_n` | 8 条独立完整生成 | 返回数值众数对应的序列 |
| 幂分布 MH | `mh` | `trajectory_power_mh` | AR 后缀或 dLLM 反向轨迹 proposal | 目标为 $`p_{\mathrm{base}}^4`$；使用完整 Hastings 比 |
| 标准条件 IS | `conditional_is` | `conditional_is` | 主模型候选；主模型 rollout | 累计自一致性；on-policy |
| 迭代条件 IS | `iterated_conditional_is` | 本轮不运行 | 主模型候选；主模型 rollout | 独立的初始估计补全冻结数值众数；有限池 i-SIR |
| 低成本 proposal 条件 IS | `conditional_is_small_proposal` | `conditional_is_reduced_layer_proposal` | 主模型候选；0.5B 或低层 rollout | 用主模型概率除以实际 rollout proposal 概率 |
| 未校正 rollout 加权 | `conditional_is_small_proposal_uncorrected` | `conditional_is_reduced_layer_proposal_uncorrected` | 与上一行相同 | 省略 $`p/q`$；目标为[式 (12)](../methods/ALGORITHMS.md#alg-uncorrected-rollout) |
| RL 参数随机采样 | `rl_sample` | `vrpo_sample` | GRPO 或 VRPO 训练后的参数 | 温度 1 |
| RL 参数贪心解码 | `rl_greedy` | `vrpo_greedy` | GRPO 或 VRPO 训练后的参数 | 每一步取最大概率项 |
| verifier-MH | `verifier_mh` | `verifier_mh` | 完整序列 MH proposal | 配置型 verifier；完整 Hastings 比 |
| 标准 verifier-IS | `verifier_conditional_is` | `verifier_conditional_is` | 主模型候选与 rollout | 配置型 verifier；on-policy |
| 低成本 proposal verifier-IS | `verifier_conditional_is_small_proposal` | `verifier_conditional_is_reduced_layer_proposal` | 主模型候选；低成本 rollout | 配置型 verifier；乘 $`p/q`$ |

AR 的“低成本 proposal”指 Qwen2.5-0.5B；dLLM 对应 LLaDA 共享前缀层的低层 proposal。`unclipped`
后缀表示不截断对数重要性概率比；`uncorrected` 后缀表示省略主模型轨迹重评分。replay 与动态候选的
成对标识列在[对应实验设置](#replay-与动态候选)。

主要比较：

| 比较 | 方法 | 统计范围 |
| --- | --- | --- |
| 最终任务质量 | Base、搜索、自一致性、幂分布 MH、条件 IS、迭代条件 IS、GRPO | 准确率与计算量 |
| 共享奖励 | verifier-MH、verifier-IS、GRPO | 准确率与经验答案分布 |
| off-policy | 标准 IS、0.5B rollout proposal IS、未校正 rollout 加权 | 准确率、ESS、分模型 FLOPs |
| replay 与动态候选 | 新生成、已有历史、动态 proposal、方差—成本分配 | 准确率、ESS、复用率、历史库构建/在线成本 |

已归档的共享奖励和动态候选结果采用默认数值参考值 verifier，会读取测试集标准答案，因此只作为算法关系
诊断。新实验可通过 `--verifier-config` 替换奖励来源；配置声明 `requires_reference = false` 时不会向
verifier 传入标准答案。部署质量实验使用累计自一致性或模型置信度。

## 奖励

| 奖励 | 定义 | 是否读取标准答案 |
| --- | --- | --- |
| 数值正确性 | 解析最终数值，与标准答案比较，取 0/1 | 是 |
| 累计自一致性（cumulative self-consistency） | 按已评估数值累计众数，匹配取 1 | 否 |
| 固定众数奖励（代码名 `frozen consensus`） | 从独立的基础模型初始估计补全取数值众数，随后固定“是否匹配众数”的 0/1 奖励 | 否 |
| 完整序列对数概率（`sequence_log_probability`） | $`c\log p(y\mid x)`$；与奖励温度共同给出 $`p^{1+c/\tau}`$ 目标 | 否 |
| Consilience 轨迹（`consilience`） | top-5 token 置信度的末段均值减去 3 倍首段均值；首段跳过前 5%，窗口各占 20% | 否 |
| token 平均对数概率（`log_probability`） | 完整生成中各选中 token 对数概率的平均值，并在候选组内归一化 | 否 |
| 平均负熵 | 完整生成的逐 token 负熵均值 | 否 |
| 自确定度（`self-certainty`） | 逐 token $`D_{\mathrm{KL}}(U\|p_{\mathrm{base}})`$ 均值 | 否 |

token 平均对数概率、平均负熵与自确定度在每次候选决策内按最小值和最大值线性归一化；常数信号统一置零。
完整序列对数概率与 Consilience 保持原始值，不做候选组归一化。完整词表评分计入
参与前向计算的 token 位置数和 FLOPs。

## 概率设置

候选和 on-policy rollout 使用同一参考采样分布。非单位温度时，温度缩放后的完整支持策略定义
本轮参考分布；off-policy 后缀比在相同温度下计算。

0.5B rollout proposal 的默认对数比值截断区间为 `[-10,10]`。原始比值、实际使用的比值、截断次数和
有效样本量（ESS）进入结果。`importance_log_ratio_clip = null` 使用普通未截断重要性权重。

未校正 rollout 加权设置 `apply_importance_correction=false`，候选权重为

```math
w_m=\frac1K\sum_{k=1}^K
\exp\!\left(\frac{r(z_m,u_{mk})}{\tau}\right),
\qquad
z_m\sim p_{\mathrm{1.5B}},\quad
u_{mk}\sim q_{\mathrm{0.5B}}(\cdot\mid z_m).
```

该路径中，基础模型的 `score_calls`、`scored_tokens` 和参与评分的 token 位置数均为 0。

## 统计量

- pass@k 使用 [Chen et al. (2021)](https://arxiv.org/abs/2107.03374) 的无偏估计式。
- 单方法准确率区间使用 [Wilson (1927)](https://doi.org/10.1080/01621459.1927.10502953) 区间。
- 方法差异使用题目级配对[自助法（bootstrap）](https://doi.org/10.1214/aos/1176344552)。
- 经验答案分布使用总变差距离（TV）和
  [Jensen--Shannon 散度](https://doi.org/10.1109/18.61115)，JS 单位为 bit。
- pass@k 的每次重复使用独立候选、rollout 和 replay 状态。

## 计算量

模型 $`j`$ 基于参数量和 token 数的前向 FLOPs 估计为

```math
\widehat F_j=2N_jS_j,
```

其中 $`N_j`$ 为参数量，$`S_j`$ 为实际参与前向计算的 token 位置数；对应代码字段以 `forward_token_slots`
结尾。1.5B 与 0.5B 分别计算后求和。计数覆盖
前缀预填充、逐 token 生成、完整序列评分和目标模型草稿验证；墙钟排除模型与数据加载。该估算不计
注意力计算中随长度平方增长的项、逐元素 CUDA 算子、分词、CPU 奖励解析与调度；墙钟仍包含后四项的执行影响。

GRPO 成本分为 rollout 生成、参考模型评分、当前策略前向/反向计算和 AdamW 对 LoRA 适配器的更新。梯度检查点对应的
当前策略路径按前向、反向与重算三个前向等价过程计量。
训练运行清单保存样本、权重、版本、LoRA 参数量、补全、token、显存、墙钟和功率积分；对应代码文件名使用
`manifest`。

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

只有配对准确率差落入预设容差时，才报告两条路径累计 FLOPs 相等所需的查询数；联合匹配还要求答案分布
TV/JS 通过阈值。

## 成本比与比较基准

| 指标 | 分子 / 分母 | 固定项 |
| --- | --- | --- |
| `compute_multiple_vs_base` | 方法 FLOPs / Base FLOPs | 样本、长度 |
| 小 proposal FLOPs 节省因子 | 标准条件 IS / 0.5B rollout proposal IS；大于 1 表示小 proposal 节省 FLOPs | 候选、rollout、生成块、随机种子、长度 |
| `runtime_multiple_vs_base` | 方法墙钟 / Base 墙钟 | 样本、硬件 |
| 连续批处理加速因子 | 逐提示墙钟 / 批处理墙钟；大于 1 表示批处理更快 | 方法、请求、随机种子 |
| 重复前缀 KV 节省因子 | 逐 rollout 执行前缀预填充 / 每个唯一前缀只执行一次；大于 1 表示省去更多重复计算 | 同一生成批次 |
| 已有历史 replay 在线加速因子 | 纯新生成 / 已有历史在线阶段；大于 1 表示 replay 更快或计算量更低 | 候选、$`H+F`$、生成块 |
| replay 首次查询加速因子 | 纯新生成 /（缓存构建 + 已有历史在线阶段）；小于 1 表示首次查询成本更高 | 同上 |
| 动态候选在线节省因子 | 固定基础模型候选 / 固定 replay 感知候选；大于 1 表示动态候选成本更低 | 最终估计成本预算 |
| 最优预算在线节省因子 | 固定 replay 感知分配 / 方差—成本分配；大于 1 表示方差—成本分配成本更低 | 候选 proposal、成本预算 |
| 多尺度 replay MH 墙钟因子 | 多尺度后缀 + 冻结历史 / 均匀后缀 + 基础模型；小于 1 表示优化路径更快 | 提示、链数、更新数、长度、奖励 |
| 小模型推测解码成本因子 | 0.5B 草稿 + 1.5B 验证 / 1.5B 普通生成；小于 1 表示推测解码成本更低 | 采样分布、请求、数值类型、token 上限 |
| vLLM 加速因子 | Transformers / vLLM；大于 1 表示 vLLM 更快 | 模型、数值类型、GPU、数据、请求集合 |

连续批处理结果同时保存 token 匹配、数值答案匹配、共同前缀和出现差异的题号。缓存构建、设计样本、在线阶段
与后台任务收尾时间分列。

<a id="infra-labels"></a>
## 执行实验标签

| 机制 | 优化路径 | 对照路径 | 保持不变的设置 |
| --- | --- | --- | --- |
| 批处理 | AR 连续批处理 / dLLM 分块批处理 | 同方法逐提示执行 | 请求、随机种子、模型与 token 上限 |
| 部分续跑 | 保存未完成 token / 已提交生成块后继续 | 从原始前缀重新生成 | 相同最终请求集合 |
| 流式奖励 | 补全结束后立即提交 verifier | 整批生成后提交 | 相同 verifier 延迟与样本 |
| 历史草稿 | AR token 树 / dLLM 轨迹缓存 | 普通生成 | 目标模型校正后的相同生成分布 |
| 小模型草稿 | 0.5B 草稿 + 1.5B 精确验证 | 1.5B 普通生成 | 相同目标采样分布；两模型 FLOPs 分列 |
| MH 预取 | 并行预取接受与拒绝分支 | 普通 MH | 相同更新数与最终采用的分支 |
| 两阶段延迟接受 | 近似奖励提前拒绝后再调用精确奖励 | 每个 proposal 调用精确奖励 | 相同 proposal 和精确目标 |
| 冻结历史混合 proposal 的 MH | 基础模型与冻结历史 proposal 的混合分布 | 基础模型 proposal | 正反混合概率均进入 Hastings 比；缓存构建成本分列 |
| 多尺度后缀与冻结历史混合 proposal 的 MH | 多尺度后缀 + 冻结历史混合分布 | 均匀后缀 + 基础模型 proposal | 提示、链数、更新数与奖励固定；缓存构建成本分列 |
| 已有历史 replay | 已建库历史记录 + 新样本校正项 | 纯新生成 | 候选、总 rollout 数与在线请求固定 |
| 分阶段 IS | 初始样本冻结配额，独立的最终估计样本计算权重 | 每候选固定 rollout 数 | 初始估计与最终估计成本分列 |
| SMC 多树搜索 | 复用条件后缀或轨迹样本池 | 同一 SMC 的纯新生成路径 | 粒子数、后续权重估计与重采样方式固定 |
| 执行后端 | AR vLLM / dLLM 批量 Transformers | AR Transformers / dLLM 逐请求 | 模型、数值类型、设备、请求与统计设计固定 |

报告中的“近零延迟奖励”表示额外奖励延迟为 0；“0.2 s 奖励”表示每次调用加入 0.2 秒固定延迟；“在线阶段”
排除缓存构建。所有因子均在表中明确分子与分母。

<a id="replay-labels"></a>
## replay 与动态候选

每条最终估计历史记录通过一次不可分割的存储操作预留，并且只使用一次。基准实验使用重复公开提示与候选
随机种子形成可控的 replay 匹配键。

| 实验组 | 候选 | 历史记录 | 新生成样本 | 设计样本 |
| --- | --- | --- | --- | --- |
| `base_candidate_fixed` | 1.5B 基础模型 | 0 | 每个非终止候选 3 条 | 0 |
| `replay_aware_fixed` | 0.5 × 基础模型 + 0.5 × proposal | 命中时最多 2 条 | 补足至 3 条 | 0 |
| `replay_aware_optimal` | 同上 | 方差—成本配额 | 方差—成本配额 | 每来源 2 条 |

重复候选共享同一 replay 匹配键的最终估计样本库存。分配预算时，一条历史样本按 1 次基础模型重评分计算，
一条新生成样本记为

```math
1+\frac{P_{\mathrm{0.5B}}}{P_{\mathrm{1.5B}}}=1.3200.
```

最终成本采用实际参与前向计算的 token 位置数与参数量。配额冻结使用候选、策略版本、库存数量和设计样本统计量；
预留记录被读取后，其奖励进入最终估计。

## 复现

以下命令均从仓库根目录运行。统一入口负责准备数据与模型、续跑 RL 训练、执行所选实验组件、写入命令清单
并记录每个已完成子任务。AR 与 dLLM 可以使用不同 Python 解释器。

### 成对完整复现

```powershell
$env:AR_PYTHON = "C:\path\to\ar-python.exe"
$env:DLLM_PYTHON = "C:\path\to\dllm-python.exe"

python experiments\run_reproduction.py `
  --family arllm --stage all --profile full --tag qwen15b-full
```

`full` 默认运行已经进入正式复现的质量、共享目标、replay、异步批处理、pass@$`k`$ 和分布诊断组件。
研究消融与后端专项测试通过 `--components` 显式选择。低成本功能检查（`smoke`）使用一题和缩短生成预算；选择 AR 训练时执行
一次 GRPO 更新，选择 dLLM 时执行 CPU VRPO 反向传播预检。本轮正式实验只运行 Qwen2.5-1.5B；dLLM 入口
保留但不执行。

### 单侧与组件选择

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage inference --profile full --tag ar-full `
  --components quality replay dynamic_is async passk infra vllm

python experiments\run_reproduction.py `
  --family dllm --stage inference --profile full --tag dllm-full `
  --components quality replay dynamic_is async passk infra
```

当前 Qwen2.5-1.5B 完整流程使用第一条命令；dLLM 命令只保留为后续大显存实验入口，本轮不执行。
新增优化消融的机器可读状态与直接复现命令见
[Qwen2.5-1.5B 优化研究](../reports/QWEN15B_OPTIMIZATION_STUDY.md)。

| 组件 | 统计对象 |
| --- | --- |
| `quality`、`matched_target` | 主要质量实验与共享奖励诊断 |
| `replay`、`dynamic_is` | 新生成/已有历史 replay、建库候选复用、动态候选与预算分配 |
| `async`、`infra` | 批处理、部分续跑、流式奖励、MH 预取与 SMC |
| `passk`、`distribution` | 独立重复采样的 pass@$`k`$ 与答案分布 |
| `ablations`、`budget_curve`、`length_ablation` | 算法参数、计算预算与长度消融 |
| `vllm` | AR-LLM 的 Transformers/vLLM 成对执行检查 |

`--ar-methods` 与 `--dllm-methods` 选择具体方法；成对标识由
[`gsm8k_llada_moe_3090.toml`](../../configs/gsm8k_llada_moe_3090.toml)固定。`--dry-run` 只写命令清单，
适合在目标机器上检查解释器、路径和实验范围。

AR 统一入口的 MH 后缀分布默认为 `multiscale`，并将同一设置传给质量与 pass@$`k`$ 子任务。使用
`--ar-mh-suffix-schedule uniform` 可复现均匀后缀基线。配置文件本身保留基线参数，因此独立调用底层
`gsm8k_reproduction.py` 时只有显式传入 `--mh-suffix-schedule multiscale` 才启用默认设置。

### 运行结果文件

| 模型族 | 原始记录与清单 | 正式汇总 |
| --- | --- | --- |
| AR-LLM | `results/gsm8k/<profile>/` 与所选 `summary-root` | `results/gsm8k_3090/` |
| dLLM | `results/reproduction/dllm/<tag>/` | 各组件目录中的 `summary.json` 或聚合 JSON |
| 成对调度 | `results/reproduction/<tag>/manifest.json` | 两侧子清单及已完成命令数 |

逐题 JSONL 按配置标识续写；pass@$`k`$ 使用独立任务分块；汇总器核对数据行号、模型 revision、有效配置、
实现哈希、dtype、工作线程数与 GPU 数。图表由已验证汇总生成，结果文件索引见
[`results/README.md`](../../results/README.md)。

### 直接调用底层脚本

单实验脚本保留用于调试与指定配置重跑，其参数由两侧实验套件入口生成。正式复现优先使用统一入口，避免
手工遗漏汇总、配置标识或配对参数。具体下一层命令可通过 `--dry-run` 查看，无需在文档中维护第二份命令清单。

## 消融矩阵

- MH：$`\alpha\in\{1,2,4,8\}`$，每个生成块的更新数 $`\{1,2,5,10\}`$。
- 条件 IS：候选数 $`M`$、rollout 数 $`K`$、引导阶段数 $`I`$。
- 迭代条件 IS：固定 9 个不同候选-rollout 状态，比较一次性大池与多轮有限池复用。
- 搜索：Beam、Best-of-$`N`$ 与条件 IS 的质量—计算曲线。
- 奖励：完整序列对数概率、Consilience 轨迹、token 平均对数概率、平均负熵、自确定度、自一致性、配置型 verifier。
- off-policy：截断、未截断与未校正 rollout 加权。
- 生成：温度 $`\{0.7,1.0,1.5\}`$，最大长度 $`\{128,256,512\}`$。
- 执行：逐提示、连续批处理、纯新生成、已有历史 replay。
- 动态候选：基础模型固定分配、replay 感知固定分配、方差—成本分配。
- 多次采样：Base、MH、GRPO 与三种条件 IS 的 8 次独立重复 pass@k。

## 完整性

原始 JSONL 按运行清单中的配置标识追加。配置标识包含有效配置、GSM8K 行号、模型权重和关键实现
文件 SHA-256。后处理器核对题目集合、运行清单和输入哈希后生成 `validated` 汇总。代码或配置变更使用
新的运行标签（`tag`）。
