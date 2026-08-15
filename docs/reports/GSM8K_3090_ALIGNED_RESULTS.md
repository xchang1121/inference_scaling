# GSM8K 方法质量与计算量实验

本报告仅汇总实验设置、结果与结果解读。算法定义、数学性质和关键实现见
[推理算法实现](../methods/ALGORITHMS.md)；完整数据版本、模型 revision、运行命令和统计定义见
[GSM8K 统一实验设计](../experiments/GSM8K_EXPERIMENT_DESIGN.md)；执行层消融见
[RTX 3090 推理执行与 rollout 复用实验](RTX3090_ROLLOUT_INFRA.md)。
表中由多个限定词组成的方法名称，均可在[报告中的组合名称](../methods/ALGORITHMS.md#alg-report-labels)
中查到各部分的独立含义。

## 报告范围与固定设置

| 项目 | 本报告采用的设置 |
| --- | --- |
| 数据 | 主结果固定 32 道 GSM8K test 题；预算消融使用另一组 8 题；答案分布审计使用 4 题 × 8 draw |
| 模型 | 1.5B 基础模型；0.5B rollout proposal；同一 1.5B checkpoint 上训练的 GRPO LoRA |
| 硬件 | 单张 RTX 3090 24 GiB；主质量网格为 FP32 |
| 生成预算 | 最长 192 token；条件 IS 为 8 个候选、每候选 3 条 rollout、4 个引导阶段 |
| MH 预算 | 幂次 4；16 个递增长度阶段；每阶段 3 次更新 |
| pass@k | 每题 8 个独立 draw；draw 之间不共享候选、rollout 或 replay |
| 统计 | 准确率使用 Wilson 95% 区间；方法差异使用题目级配对 bootstrap；FLOPs 按实际 forward token slots 估算 |

主质量比较允许不同方法采用不同目标，只比较最终 GSM8K 质量与预算。共享奖励实验统一使用显式正确性奖励；
该实验与动态候选实验会读取 test split 标准答案，因此属于 oracle 诊断。

## 单次生成结果

| 方法 | 正确数 / 32 | 准确率 | 推理 PFLOPs | 相对 Base FLOPs | 墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Base](../methods/ALGORITHMS.md#alg-report-labels) | 13 | 40.625% | 0.0279 | 1.00× | 91.1 s |
| [Beam-8](../methods/ALGORITHMS.md#alg-report-labels) | 12 | 37.500% | 0.2306 | 8.27× | 130.6 s |
| [自一致性投票-8](../methods/ALGORITHMS.md#alg-report-labels) | 14 | 43.750% | 0.1621 | 5.81× | 136.8 s |
| [幂分布 MH](../methods/ALGORITHMS.md#alg-report-labels) | 12 | 37.500% | 1.3077 | 46.88× | 1485.3 s |
| [标准条件 IS](../methods/ALGORITHMS.md#alg-report-labels) | 21 | 65.625% | 1.3706 | 49.13× | 422.1 s |
| [0.5B proposal 条件 IS](../methods/ALGORITHMS.md#alg-report-labels) | 15 | 46.875% | 2.4724 | 88.63× | 422.8 s |
| [GRPO 随机采样](../methods/ALGORITHMS.md#alg-report-labels) | 22 | 68.750% | 0.0254 | 0.91× | 124.9 s |
| [GRPO 贪心](../methods/ALGORITHMS.md#alg-report-labels) | 18 | 56.250% | 0.0263 | 0.94× | 127.2 s |

![GSM8K 单次生成准确率与推理计算量](../assets/gsm8k_3090_aligned_quality_compute.svg)

标准条件 IS 与 GRPO 随机采样的准确率差为 -3.125 个百分点，逐题配对 bootstrap 95% 区间为
[-12.500, 6.250]；标准条件 IS 相对 Base 提高 25 个百分点。两者采用不同奖励，因此该结果只支持单次
任务准确率接近；分布关系由共享奖励实验单独评估。标准条件 IS 的推理 FLOPs 约为 GRPO 随机采样的
54 倍，训练完成后的 GRPO 具有更低的单次推理成本。

0.5B proposal 条件 IS 相对标准版本低 18.75 个百分点，区间为 [-34.375, -6.250]；其 FLOPs 是标准
版本的 `1.804×`，墙钟因子为 `1.002×`。当前 1.5B 精确重评分成本抵消了小 proposal 的生成成本优势。

幂分布 MH 得到 12/32，与 Base 的点估计接近；本组基模概率锐化未改善正确率。正确性奖励下的 MH
结果见共享奖励实验。

## 多次采样结果

每种方法在相同 32 道题上独立采样 8 次，共生成 256 条序列。

![GSM8K 既有六种方法的 pass@k 与题目级不确定性](../assets/gsm8k_3090_aligned_passk.svg)

| 方法 | pass@1 | pass@2 | pass@4 | pass@8 | 8 draw 推理 PFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Base](../methods/ALGORITHMS.md#alg-report-labels) | 39.844% | 52.009% | 61.518% | 68.750% | 0.1613 |
| [幂分布 MH](../methods/ALGORITHMS.md#alg-report-labels) | 38.281% | 47.098% | 53.571% | 59.375% | 13.2872 |
| [GRPO 随机采样](../methods/ALGORITHMS.md#alg-report-labels) | 58.984% | 68.638% | 75.536% | 81.250% | 0.1516 |
| [标准条件 IS](../methods/ALGORITHMS.md#alg-report-labels) | 58.203% | 63.728% | 68.929% | 75.000% | 11.0284 |
| [0.5B proposal IS（截断）](../methods/ALGORITHMS.md#alg-report-labels) | 46.484% | 53.125% | 58.705% | 62.500% | 19.4781 |
| [0.5B proposal IS（无截断）](../methods/ALGORITHMS.md#alg-report-labels) | 46.484% | 53.348% | 59.509% | 65.625% | 19.2690 |
| [0.5B rollout（无重评分）](../methods/ALGORITHMS.md#alg-report-labels) | 45.313% | 52.009% | 58.839% | 65.625% | 4.6542 |

标准条件 IS 相对 GRPO 随机采样的 pass@1 差为 -0.781 个百分点，区间为 [-6.250, 4.297]；pass@8 差为
-6.250 个百分点，区间为 [-15.625, 0]。两者的单次成功率接近，GRPO 在较大采样预算下得到更高覆盖率。
两个 0.5B proposal 版本的 pass@1 均比标准 IS 低 11.719 个百分点。移除权重截断后，pass@8 由
62.500% 升至 65.625%，说明当前 proposal 与 base 的重叠仍造成较高的重要性权重方差。

幂分布 MH 相对 Base 的 pass@1 差异较小，每题不同数值答案数由 4.56 降至 3.25；其主要观测效应是
生成多样性收缩。

<a id="15b-rescoring-ablation"></a>
### 1.5B 重评分消融

该消融固定 1.5B 候选和 0.5B rollout，只比较保留 `p_1.5B/q_0.5B` 修正与完全删除主模型重评分。
两条路径对应的统计目标见[off-policy 补全与无重评分目标](../methods/ALGORITHMS.md#alg-offpolicy-is)。

相对截断重评分版本，无重评分版本的 pass@1 低 1.172 个百分点，题目级配对 bootstrap 95% 区间为
[-3.906, 1.172]；当前 32 题网格没有分辨出稳定的 pass@1 下降。pass@2 差为 -1.116 个百分点，
pass@4 差为 +0.134 个百分点，pass@8 差为 +3.125 个百分点；相应区间均包含 0。相对标准 1.5B
rollout IS，无重评分版本的 pass@1 低 12.891 个百分点，配对区间为 [-20.313, -6.250]。

无重评分版本总计 4.6542 PFLOPs，其中 1.5B 候选生成 1.5749 PFLOPs、0.5B rollout 生成 3.0794
PFLOPs，两个后端的 `score_calls` 与 `scored_tokens` 均为 0。截断重评分版本需要 19.4781 PFLOPs，
即其 FLOPs 为无重评分版本的 4.185 倍；标准 IS 的 FLOPs 为无重评分版本的 2.370 倍。

完整网格的无重评分运行耗时 4064.3 秒，已有截断重评分运行耗时 2842.9 秒；两次运行日期不同，因此
不构成受控墙钟比。同一会话的 2 题 × 1 draw 检查分别耗时 36.5 秒和 30.3 秒。无重评分路径生成了
更多自回归 token，而批量序列评分具有较高并行度；稠密 FLOPs 的减少未直接转化为墙钟收益。

## 共享奖励目标

本组统一采用 `p_target(y|x) ∝ p_base(y|x) × exp(exact_reward(y) / 0.04)`。精确奖励读取 test split
标准答案。

| 方法 | 正确数 / 32 | 准确率 | 推理 PFLOPs | 墙钟 |
| --- | ---: | ---: | ---: | ---: |
| [verifier-MH](../methods/ALGORITHMS.md#alg-report-labels) | 25 | 78.125% | 1.3028 | 2319.7 s |
| [标准 verifier-IS](../methods/ALGORITHMS.md#alg-report-labels) | 24 | 75.000% | 1.4839 | 439.7 s |
| [0.5B proposal verifier-IS（1.5B 重评分）](../methods/ALGORITHMS.md#alg-report-labels) | 20 | 62.500% | 2.2077 | 476.5 s |
| [0.5B rollout verifier-energy（无重评分）](../methods/ALGORITHMS.md#alg-report-labels) | 20 | 62.500% | 0.5740 | 556.2 s |
| [GRPO 随机采样](../methods/ALGORITHMS.md#alg-report-labels) | 22 | 68.750% | 0.0254 | 124.9 s |

verifier-MH 与标准 verifier-IS 的准确率差为 3.125 个百分点，区间为 [-9.375, 15.625]；二者相对
GRPO 随机采样分别为 +9.375 和 +6.250 个百分点。共享奖励条件下，两种直接采样方法达到与本地
GRPO 随机采样接近或更高的点估计；32 题样本不足以判断完整序列分布等价。

经过 1.5B 重评分的 0.5B proposal verifier-IS 相对标准版本低 12.5 个百分点，配对 bootstrap 95%
区间为 [-28.125, 0]，FLOPs 为标准版本的 `1.488×`。标准版本与该行来自不同运行批次，墙钟不作
受控比较。

<a id="verifier-rescoring-ablation"></a>
### 精确奖励下的小模型补全消融

两条路径在相同代码版本、硬件、32 道题、随机种子、候选数、rollout 数和长度预算下连续运行。候选均
由 1.5B 模型生成，补全均由 0.5B 模型生成；差别仅为是否执行 1.5B 后缀重评分。算法定义见
[无重评分补全目标](../methods/ALGORITHMS.md#alg-proposal-energy)。

无重评分与重评分版本均为 20/32，逐题准确率差为 0，配对 bootstrap 95% 区间为 [-9.375, 9.375]
个百分点。两者各自多答对 1 题、30 题正确性相同；解析后的数值结果有 26/32 相同，完整生成序列只有
6/32 相同。当前样本没有观察到准确率下降，但区间不足以证明两种目标或算法等价。无重评分版本相对
标准 verifier-IS 低 12.5 个百分点，配对区间为 [-28.125, 0]。

无重评分版本的 0.5740 PFLOPs 由 1.5B 候选生成 0.2014 PFLOPs 和 0.5B 补全 0.3726 PFLOPs 组成，
1.5B 评分成本为 0。重评分版本包含 1.5B 候选生成 0.1730 PFLOPs、1.5B 后缀评分 1.6882 PFLOPs
和 0.5B 补全 0.3465 PFLOPs，共 2.2077 PFLOPs。删除重评分使估算 FLOPs 降低 74.0%，重评分版本
的计算量是无重评分版本的 `3.846×`。

无重评分版本耗时 556.2 秒，重评分版本耗时 476.5 秒，前者增加 16.7%。无重评分路径的 1.5B 候选
生成 token 增加 16.0%，0.5B 补全 token 增加 12.2%，最终序列平均长度增加 15.9%。批量重评分的并行度
高于自回归补全，因此评分 FLOPs 降低 74.0% 后仍未获得墙钟加速。

4 道固定题、每题 8 次采样的答案分布审计以 GRPO 为参考：

| 方法 | 平均 TV | 平均 JS（bit） |
| --- | ---: | ---: |
| [Base](../methods/ALGORITHMS.md#alg-report-labels) | 0.4375 | 0.3267 |
| [verifier-MH](../methods/ALGORITHMS.md#alg-report-labels) | 0.2500 | 0.1423 |
| [标准 verifier-IS](../methods/ALGORITHMS.md#alg-report-labels) | 0.2500 | 0.1423 |
| [0.5B proposal verifier-IS](../methods/ALGORITHMS.md#alg-report-labels) | 0.2500 | 0.1423 |

三种直接采样方法的答案级距离点估计均低于 Base。TV bootstrap 区间上界为 0.4063；该样本规模只提供
趋势性证据。

<a id="quality-replay-dynamic"></a>
## rollout replay 与动态候选

固定 replay 实验使用 8 个 base 候选和每候选 3 条总 rollout。warm 路径最多读取 2 条已评分历史
rollout，并保留 1 条 fresh base rollout；统计修正见
[base 候选 rollout replay](../methods/ALGORITHMS.md#alg-base-replay)。

| 路径 | 正确数 / 32 | 准确率 |
| --- | ---: | ---: |
| [fresh-only](../methods/ALGORITHMS.md#alg-report-labels) | 23 | 71.875% |
| [warm replay](../methods/ALGORITHMS.md#alg-report-labels) | 22 | 68.750% |

warm replay 相对 fresh-only 的准确率差为 -3.125 个百分点，逐题配对 bootstrap 95% 区间为
[-12.500, 6.250]。当前样本对稳定质量差异与质量等价均缺乏充分分辨率。

<a id="quality-dynamic-is"></a>
### 动态候选与方差—成本分配

本组为 oracle 诊断。三组均使用 8 个候选、48-token block、最长 192 token 和每个非终止候选 3 条
evaluation rollout；主模型与 proposal 温度为 1.0，奖励温度为 0.1，历史权重截断阈值为 8.0。

| 实验臂 | 候选与 rollout 设置 |
| --- | --- |
| [base 候选 + 固定 fresh](../methods/ALGORITHMS.md#alg-report-labels) | 候选来自 1.5B base；每候选 3 条 fresh rollout |
| [动态候选 + 固定 replay](../methods/ALGORITHMS.md#alg-report-labels) | 候选来自 `0.5 × base + 0.5 × 0.5B proposal`；库存充足时使用 2 条历史与 1 条 fresh |
| [动态候选 + 方差—成本分配](../methods/ALGORITHMS.md#alg-report-labels) | 沿用相同候选 proposal；独立 design 样本决定冻结后的 evaluation 配额 |

算法定义见[动态候选与外层 IS](../methods/ALGORITHMS.md#alg-dynamic-is)和
[方差—成本预算分配](../methods/ALGORITHMS.md#alg-budget-allocation)。

| 方法 | 正确数 / 32 | 准确率 | 辅助候选占比 | rollout 复用率 | 平均外层 ESS | 平均最终 ESS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [base 候选 + 固定 fresh](../methods/ALGORITHMS.md#alg-report-labels) | 23 | 71.875% | 0% | 0% | 8.000 | 6.323 |
| [动态候选 + 固定 replay](../methods/ALGORITHMS.md#alg-report-labels) | 21 | 65.625% | 51.282% | 34.943% | 4.365 | 3.424 |
| [动态候选 + 方差—成本分配](../methods/ALGORITHMS.md#alg-report-labels) | 23 | 71.875% | 51.282% | 5.707% | 4.382 | 3.315 |

动态固定组相对 base 固定组的准确率差为 -6.25 个百分点，逐题配对 bootstrap 95% 区间为
[-21.875, 9.375]；逐题赢 2、输 4、平 26。方差—成本版本相对动态固定组为 +6.25 个百分点，区间为
[-6.250, 18.750]；逐题赢 3、输 1、平 28。两个区间均覆盖 0。完整版本与 base 固定组的点估计同为
23/32，平均最终 ESS 为 3.315；当前 design 样本量尚未形成更稳定权重的证据。

在线与冷启动成本见[基础设施报告](RTX3090_ROLLOUT_INFRA.md#infra-report-dynamic)。

## 质量与预算消融

各消融点使用同一组 8 道题；准确率最小变化为 12.5%，误差线为 Wilson 95% 区间。

![GSM8K 候选数、引导阶段、MH 更新次数和生成长度消融](../assets/gsm8k_3090_aligned_ablations.svg)

| 维度 | 设置与正确数 | 8 题合计 PFLOPs | 观测结果 |
| --- | --- | --- | --- |
| [标准 IS](../methods/ALGORITHMS.md#alg-report-labels) 候选数 `M`，`K=3` | `3/5/8/10 → 6/6/6/5` | `0.1281/0.2064/0.3068/0.3861` | `M=3` 达到本组最高点估计 |
| [0.5B proposal IS](../methods/ALGORITHMS.md#alg-report-labels) 候选数 `M` | `3/5/8/10 → 3/5/4/5` | `0.2299/0.3602/0.5730/0.6755` | 各点位于标准 IS 的质量—FLOPs 前沿下方 |
| [标准 IS](../methods/ALGORITHMS.md#alg-report-labels) rollout 数 `K`，`M=10` | `1/3/5 → 5/5/6` | `0.2319/0.3861/0.6063` | `K=5` 增加一题正确，同时增加计算成本 |
| 引导阶段数 `S` | `2/4/8/16 → 5/6/6/6` | `0.1293/0.3068/0.7426/1.8104` | `S=4` 后点估计保持 6/8 |
| [MH](../methods/ALGORITHMS.md#alg-report-labels) 幂次 `α` | `1/2/4/8 → 3/4/6/3` | `0.2895/0.3067/0.3108/0.3148` | `α=4` 在本组得到最高点估计 |
| [MH](../methods/ALGORITHMS.md#alg-report-labels) 每阶段更新数 `U` | `1/2/5/10 → 3/5/6/7` | `0.1526/0.2266/0.4726/0.8569` | 更新数增加改善有限链结果，边际成本同步上升 |
| [0.5B proposal](../methods/ALGORITHMS.md#alg-report-labels) 权重 | `截断/无截断 → 4/5` | `0.5730/0.5253` | 单次消融相差一题；32×8 网格的 pass@1 相同 |
| 最大生成长度 | 标准 IS：`128/256/512 → 4/7/6` | `0.2669/0.3151/0.2381` | 256 token 在本组取得最高点估计 |

奖励消融中，标准 IS 的 self-consistency 得到 6/8；平均 token 对数概率、平均负熵和 self-certainty
分别得到 4/8、5/8、5/8，各使用约 0.9 PFLOPs，高于 self-consistency 的 0.3068 PFLOPs。使用 test
split 标准答案作为选择信号时，oracle Best-of-8 与 oracle 标准 IS 均得到 8/8，说明多数题目的候选池
已经包含正确数值答案，主要误差来自部署可用的选择信号。

sampling temperature 为 0.7、1.0、1.5 时，标准 IS 分别得到 4/8、6/8、1/8。temperature 1.5 下，
无法解析或达到长度上限的生成序列比例上升。

## 结果总结与适用范围

1. 标准条件 IS 的单次生成准确率与本地 GRPO 接近，并高于 Base；其推理 FLOPs 约为已训练 GRPO 的
   54 倍。
2. 当前 0.5B off-policy proposal 相对标准 IS 同时降低准确率并增加 FLOPs。删除主模型重评分后，
   pass@1 相对截断版本变化 -1.172 个百分点且配对区间覆盖 0，FLOPs 降至 23.9%；该版本对应不同目标。
3. warm replay、动态候选固定组和方差—成本组的配对区间均覆盖 0；当前样本规模不足以判断质量等价。
4. 方差—成本分配复用 5.707% evaluation rollout，平均最终 ESS 为 3.315，低于动态固定组的 3.424；
   本组尚未观测到权重稳定性收益。
5. 幂分布 MH 的主要效应是多样性收缩；共享正确性奖励下，verifier-MH 与标准 verifier-IS 的准确率
   点估计接近。

结果范围为固定 32 道 test 题和单张 RTX 3090。FLOPs 估算排除 attention 的长度二次项、逐元素 kernel、
tokenization 和主机调度。机器可读结果位于
`results/gsm8k_3090/gsm8k_3090_aligned_*_validated.json`，逐文件索引见
[`results/README.md`](../../results/README.md)。
