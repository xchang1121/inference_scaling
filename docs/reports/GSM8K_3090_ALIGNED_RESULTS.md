# GSM8K 方法质量与计算量实验

本报告比较训练式强化学习、Metropolis–Hastings 采样、重要性采样和 rollout 复用方法在 GSM8K 上的
生成质量与计算成本。质量指标包括单次生成准确率、pass@k、共享目标下的准确率与经验答案分布距离。
墙钟、批处理、缓存和 rollout 执行优化见
[RTX 3090 推理执行与 rollout 复用实验](RTX3090_ROLLOUT_INFRA.md)。

## 术语与统计量

| 术语 | 全称或定义 | 本报告中的用途 |
| --- | --- | --- |
| IS | Importance Sampling，重要性采样 | 用目标概率与行为概率之比修正异分布 rollout |
| MH | Metropolis–Hastings | 通过提议与接受步骤产生目标分布样本 |
| GRPO | Group Relative Policy Optimization，组相对策略优化 | 使用组内相对优势更新模型参数的训练基线 |
| self-consistency | 自一致性 | 独立采样多条推理路径，并按最终数值答案的众数选择 |
| off-policy | 异策略 | rollout 的生成分布与目标分布存在差异 |
| replay | 经验回放 | 保存历史 rollout，并在后续估计中按真实行为概率复用 |
| ESS | Effective Sample Size，有效样本量 | 衡量重要性权重集中程度；数值越高表示权重越均匀 |
| pass@k | `k` 次采样至少一次成功的概率 | 衡量多次独立采样的覆盖率 |
| oracle | 预言机诊断 | 读取测试集标准答案构造奖励，仅用于检验算法关系 |
| TV / JS | Total Variation / Jensen–Shannon | 比较经验答案分布的总变差距离与 Jensen–Shannon 散度 |
| defensive mixture | 防御混合分布 | proposal 中保留 base 分量，以覆盖目标分布的支持集 |

## 实验设置

### 数据、模型与训练

- **数据集**：[GSM8K](https://arxiv.org/abs/2110.14168) 的 train split 包含 7,473 条样本，test split
  包含 1,319 条样本。GRPO 训练使用完整 train split。主实验固定 32 道 test 题，消融使用另一组
  8 道题，答案分布审计使用 4 道题并为每种方法独立采样 8 次。
- **主模型**：[`Qwen/Qwen2.5-1.5B-Instruct`](https://arxiv.org/abs/2412.15115)，revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`。
- **rollout proposal**：`Qwen/Qwen2.5-0.5B-Instruct`，revision
  `7ae557604adf67be50417f59c2c2f167def9a775`。
- **训练基线**：从同一 1.5B checkpoint 出发执行 GRPO；参数更新采用
  [Low-Rank Adaptation（LoRA）](https://openreview.net/pdf?id=nZeVKeeFYf9)，共 205 个优化步。每个
  prompt 生成 4 条序列，奖励为最终数值答案的正确性。
- **硬件与精度**：单张 RTX 3090 24 GiB。主结果使用 FP32，以控制低精度及 batch 形状变化导致的
  生成概率与重评分偏差。

### 方法、目标与文献依据

文献列给出母方法或设计原则。标准条件 IS、分块引导、历史 rollout 修正和动态候选属于本仓库的组合
实现，其效果由后续成对实验直接评估。

| 方法 | 候选与决策规则 | 目标或比较对象 | 文献依据 |
| --- | --- | --- | --- |
| Base | 从 `p_base(y∣x)` 采样一次 | 单次随机采样基线 | [Qwen2.5，Yang et al., 2024](https://arxiv.org/abs/2412.15115) |
| Beam-8 | 保留累计对数概率最高的 8 条部分序列并确定性扩展 | 近似最大概率搜索基线 | [Beam Search Strategies，Freitag and Al-Onaizan, 2017](https://aclanthology.org/W17-3207/) |
| 自一致性投票-8 | 独立采样 8 条推理路径，按数值答案众数选择 | 并行采样与答案边缘化基线 | [Self-Consistency，Wang et al., 2023](https://openreview.net/pdf?id=1PL1NIMMrw) |
| 幂分布 MH | 对完整序列执行后缀提议与接受，目标为 `p_base(y∣x)^4` | 基模概率锐化 | [Metropolis–Hastings，Hastings, 1970](https://doi.org/10.1093/biomet/57.1.97) |
| 标准条件 IS | Base 候选与 Base rollout；累积 self-consistency 奖励形成候选权重 | 免训练条件重加权 | [重要性采样，Hesterberg, 1995](https://doi.org/10.1080/00401706.1995.10484303)；[Self-Consistency，Wang et al., 2023](https://openreview.net/pdf?id=1PL1NIMMrw) |
| 0.5B proposal 条件 IS | Base 候选与 0.5B rollout；后缀权重乘精确 `p_base/q` | off-policy rollout | [Off-policy IS，Precup et al., 2000](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf) |
| 0.5B rollout 无重评分 | Base 候选与 0.5B rollout；候选权重不乘 `p_base/q` | 重评分消融；估计 0.5B 而非 Base 的后续能量，不属于 Base 目标的 off-policy IS | 与上一行直接配对的有偏消融 |
| GRPO | 正确性奖励与 KL 正则训练后的策略 | 训练式强化学习基线 | [DeepSeekMath，Shao et al., 2024](https://arxiv.org/abs/2402.03300) |
| verifier-MH / verifier-IS | 统一目标 `p_base(y∣x) exp(exact_reward(y)/0.04)` | 相同奖励下的算法比较 | [GSM8K verifier，Cobbe et al., 2021](https://arxiv.org/abs/2110.14168)；[Hastings, 1970](https://doi.org/10.1093/biomet/57.1.97)；[Hesterberg, 1995](https://doi.org/10.1080/00401706.1995.10484303) |

主表中的标准条件 IS、幂分布 MH 与 GRPO 分别对应 self-consistency、基模概率幂和正确性奖励。主表比较
最终任务质量与成本；共享目标实验进一步控制奖励差异，并报告经验答案分布距离。

### 推理预算与公平性约束

| 项目 | 主实验设置 |
| --- | --- |
| 最大新生成长度 | 192 token |
| Base 采样 | temperature 1.0；无 top-k/top-p 硬截断 |
| Beam / 自一致性投票 | 8 beams / 8 samples |
| 条件 IS | `M=8` 个候选；每候选 `K=3` 条 rollout；`S=4` 个引导阶段 |
| 幂分布 MH | `α=4`；16 个递增长度阶段；每阶段 3 次更新；共 48 次后缀更新 |
| 0.5B proposal 权重 | 后缀 log 重要性比默认截到 `[-10,10]`；另设无截断对照 |
| 无重评分消融 | 0.5B rollout 权重只保留 `exp(reward/τ)`；不提交 1.5B rollout 评分请求 |
| pass@k | 每题 8 个独立 draw；draw 之间无候选、rollout 或 replay 共享 |

各主要方法使用相同题目、prompt、最大长度和请求级随机种子。主质量表中的候选均由 1.5B Base 生成；
0.5B proposal 仅替换 rollout 生成器。动态候选实验单独采用 base/0.5B defensive mixture，并以精确
候选层概率比校正。replay 实验固定候选数和每候选 rollout 总预算。连续批处理仅改变物理执行调度。

### 指标与统计方法

主指标为单次生成的最终数值答案准确率。pass@k 使用
[Chen et al. (2021)](https://arxiv.org/abs/2107.03374) 的无偏估计形式。单方法准确率区间采用
[Wilson (1927)](https://doi.org/10.1080/01621459.1927.10502953) 区间；方法差异采用题目级配对
[bootstrap](https://doi.org/10.1214/aos/1176344552)。JS 散度采用
[Lin (1991)](https://doi.org/10.1109/18.61115) 的定义，以 bit 为单位。

计算量采用 `2 × 参数量 × 实际 forward token slots` 的稠密矩阵主导项估算，分别计入主模型、
proposal、重复 prompt 和重评分。缓存、连续批处理、吞吐与冷启动成本在基础设施报告中单列。

## 实验目标与比较关系

| 实验目标 | 比较方法 | 结论范围 |
| --- | --- | --- |
| 单次与多次采样质量 | Base、Beam-8、自一致性投票-8、幂分布 MH、条件 IS、GRPO | 比较最终 GSM8K 准确率与推理预算；各方法目标允许不同 |
| 共享奖励目标 | verifier-MH、verifier-IS、GRPO | 比较相同显式奖励下的准确率和经验答案分布 |
| off-policy 与 rollout 复用 | 0.5B proposal、无重评分消融、动态候选、warm replay、方差—成本分配 | 比较准确率、ESS、复用率与选择稳定性 |

oracle 实验读取 test split 的标准答案，结论限于算法诊断。部署方法对应 self-consistency、模型置信度或
测试时可用 verifier。

## 单次生成结果

| 方法 | 正确数 / 32 | 准确率 | 推理 PFLOPs | 相对 Base FLOPs | 墙钟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 13 | 40.625% | 0.0279 | 1.00× | 91.1 s |
| Beam-8 | 12 | 37.500% | 0.2306 | 8.27× | 130.6 s |
| 自一致性投票-8 | 14 | 43.750% | 0.1621 | 5.81× | 136.8 s |
| 幂分布 MH | 12 | 37.500% | 1.3077 | 46.88× | 1485.3 s |
| 标准条件 IS | 21 | 65.625% | 1.3706 | 49.13× | 422.1 s |
| 0.5B proposal 条件 IS | 15 | 46.875% | 2.4724 | 88.63× | 422.8 s |
| GRPO 随机采样 | 22 | 68.750% | 0.0254 | 0.91× | 124.9 s |
| GRPO 贪心 | 18 | 56.250% | 0.0263 | 0.94× | 127.2 s |

![GSM8K 单次生成准确率与推理计算量](../assets/gsm8k_3090_aligned_quality_compute.svg)

标准条件 IS 与 GRPO 随机采样的准确率差为 -3.125 个百分点，逐题配对 bootstrap 95% 区间为
[-12.500, 6.250]；标准条件 IS 相对 Base 提高 25 个百分点。两者采用不同奖励，因此该结果支持单次
任务准确率接近，分布关系由共享目标实验单独评估。标准条件 IS 的推理 FLOPs 约为 GRPO 随机采样的
54 倍；完成训练后，GRPO 具有更低的单次推理成本。

0.5B proposal 条件 IS 相对标准版本低 18.75 个百分点，区间为 [-34.375, -6.250]；FLOPs 因子为
`1.804×`，墙钟因子为 `1.002×`。当前路径仍由 1.5B 模型精确重评分 off-policy rollout，小 proposal
的生成成本优势尚未转化为端到端计算收益。

幂分布 MH 的目标为 `p_base(y|x)^4`。该方法得到 12/32，与 Base 的点估计接近；本组结果表明基模概率
锐化未改善当前 GSM8K 子集的正确率。正确性奖励下的 MH 结果见共享目标实验。

## 多次采样结果

每种方法在相同 32 道题上独立采样 8 次，共生成 256 条序列。draw 之间保持统计独立，连续批处理仅改变
物理执行方式。

![GSM8K 既有六种方法的 pass@k 与题目级不确定性](../assets/gsm8k_3090_aligned_passk.svg)

图中保留既有六种方法的统一比较；下表另加入本次无重评分消融。

| 方法 | pass@1 | pass@2 | pass@4 | pass@8 | 8 draw 推理 PFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 39.844% | 52.009% | 61.518% | 68.750% | 0.1613 |
| 幂分布 MH | 38.281% | 47.098% | 53.571% | 59.375% | 13.2872 |
| GRPO 随机采样 | 58.984% | 68.638% | 75.536% | 81.250% | 0.1516 |
| 标准条件 IS | 58.203% | 63.728% | 68.929% | 75.000% | 11.0284 |
| 0.5B proposal IS（截断） | 46.484% | 53.125% | 58.705% | 62.500% | 19.4781 |
| 0.5B proposal IS（无截断） | 46.484% | 53.348% | 59.509% | 65.625% | 19.2690 |
| 0.5B rollout（无重评分） | 45.313% | 52.009% | 58.839% | 65.625% | 4.6542 |

标准条件 IS 相对 GRPO 的 pass@1 差为 -0.781 个百分点，区间为 [-6.250, 4.297]；pass@8 差为
-6.250 个百分点，区间为 [-15.625, 0]。两者的单次成功率接近，GRPO 在较大采样预算下得到更高覆盖率。
两个 0.5B proposal 版本的 pass@1 均比标准 IS 低 11.719 个百分点。移除权重截断后，pass@8 由
62.500% 升至 65.625%；proposal 与 base 的有限重叠仍形成较高的重要性权重方差。

### 1.5B 重评分消融

经过重评分的 0.5B rollout 估计

`E_{u~q_0.5B}[p_1.5B(u|s) / q_0.5B(u|s) × exp(r(s,u)/τ)]`。

消融版本删除概率比与全部 1.5B rollout 评分请求，实际估计变为

`E_{u~q_0.5B}[exp(r(s,u)/τ)]`。

因此该消融改变了候选选择目标，而不是对同一 off-policy IS 估计器进行低成本近似。候选仍全部来自
1.5B Base，只有后续能量改由 0.5B rollout 分布定义。

相对截断重评分版本，无重评分版本的 pass@1 低 1.172 个百分点，题目级配对 bootstrap 95% 区间为
[-3.906, 1.172]；当前 32 题网格没有分辨出稳定的 pass@1 下降。pass@2 差为 -1.116 个百分点，
pass@4 差为 +0.134 个百分点，pass@8 差为 +3.125 个百分点；相应区间均包含 0。相对标准 1.5B
rollout IS，无重评分版本的 pass@1 低 12.891 个百分点，配对区间为 [-20.313, -6.250]，下降明确。

无重评分版本总计 4.6542 PFLOPs，其中 1.5B 候选生成 1.5749 PFLOPs、0.5B rollout 生成 3.0794
PFLOPs，两个后端的 `score_calls` 与 `scored_tokens` 均为 0。截断重评分版本需要 19.4781 PFLOPs，
即其 FLOPs 为无重评分版本的 4.185 倍；标准 IS 的 FLOPs 为无重评分版本的 2.370 倍。

FLOPs 减少没有转化为本组 Transformers 运行的墙钟加速。完整网格的无重评分运行耗时 4064.3 秒，
已有截断重评分运行耗时 2842.9 秒；二者在不同日期执行，不能作为受控加速比。同一会话的 2 题 × 1
draw 检查分别耗时 36.5 秒和 30.3 秒；无重评分路径在该小样本中生成了更多自回归 token。批量序列
评分具有较高并行度，而候选和 rollout 解码受最长序列约束，因此减少稠密 FLOPs 不保证降低墙钟。

幂分布 MH 相对 Base 的 pass@1 差异较小，每题不同数值答案数由 4.56 降至 3.25。其主要观测效应为
生成多样性收缩。

## 共享奖励目标

受控实验统一采用

`p_target(y|x) ∝ p_base(y|x) × exp(exact_reward(y) / 0.04)`。

精确奖励读取 test split 标准答案，属于 oracle 诊断。verifier-MH 的状态和 proposal 均为完整序列；
每次保留前缀后，proposal 后缀生成至 EOS，再计算终局奖励与接受率。

| 方法 | 正确数 / 32 | 准确率 | 推理 PFLOPs | 墙钟 |
| --- | ---: | ---: | ---: | ---: |
| verifier-MH | 25 | 78.125% | 1.3028 | 2319.7 s |
| 标准 verifier-IS | 24 | 75.000% | 1.4839 | 439.7 s |
| 0.5B proposal verifier-IS | 20 | 62.500% | 2.2077 | 374.5 s |
| GRPO 随机采样 | 22 | 68.750% | 0.0254 | 124.9 s |

verifier-MH 与标准 verifier-IS 的准确率差为 3.125 个百分点，区间为 [-9.375, 15.625]；二者相对
GRPO 分别为 +9.375 和 +6.250 个百分点。共享奖励条件下，两种直接采样方法达到与本地 GRPO 接近或
更高的点估计。32 题样本对完整序列分布一致性的统计分辨率有限。

0.5B proposal verifier-IS 相对标准版本低 12.5 个百分点。其墙钟因子为 `0.852×`，即相对标准版本
具有 `1.174×` 吞吐提升；FLOPs 因子为 `1.488×`。该路径同时改变质量与计算量，因而归类为质量—成本
权衡，而非质量匹配条件下的加速。

4 道固定题、每题 8 次采样的答案分布审计以 GRPO 为参考：

| 方法 | 平均 TV | 平均 JS（bit） |
| --- | ---: | ---: |
| Base | 0.4375 | 0.3267 |
| verifier-MH | 0.2500 | 0.1423 |
| 标准 verifier-IS | 0.2500 | 0.1423 |
| 0.5B proposal verifier-IS | 0.2500 | 0.1423 |

三种直接采样方法的答案级距离点估计均低于 Base。TV bootstrap 区间上界为 0.4063；该样本规模提供
趋势性结果，分布等价检验需要更大的题目与采样网格。

## rollout replay 与动态候选

经验回放的母方法见 [Lin (1992)](https://doi.org/10.1007/BF00992699)，off-policy 修正采用真实行为
概率的重要性比。固定 replay 实验使用 8 个 base 候选和每候选 3 条总 rollout。warm 路径最多读取
2 条已评分历史 rollout，并保留 1 条 fresh base rollout。

| 路径 | 正确数 / 32 | 准确率 |
| --- | ---: | ---: |
| fresh-only | 23 | 71.875% |
| warm replay | 22 | 68.750% |

warm replay 相对 fresh-only 的准确率差为 -3.125 个百分点，逐题配对 bootstrap 95% 区间为
[-12.500, 6.250]。区间覆盖 0；当前样本对稳定质量差异与质量等价均缺乏充分分辨率。

### 动态候选与方差—成本分配

该实验检验动态 proposal、外层 IS、rollout replay 和预算分配。奖励直接读取 GSM8K 标准答案，因此
属于 oracle 诊断。主模型与 proposal 的采样温度均为 1.0，奖励温度为 0.1，历史权重截断阈值为 8.0。
三组均使用 8 个候选、48-token block、最长 192 token 和每个非终止候选 3 条 evaluation rollout。

1. `base_candidate_fixed`：候选来自 1.5B base；无历史库；每候选使用 3 条 fresh rollout。
2. `replay_aware_fixed`：候选来自 `0.5 × base + 0.5 × 0.5B proposal`，并乘精确外层 `p_base/q`；
   辅助候选预存 2 条 evaluation rollout；库存充足时使用 2 条历史与 1 条 fresh rollout。
3. `replay_aware_optimal`：沿用相同 defensive proposal 与缓存规则；读取 evaluation 值前，为每个候选
   生成 2 条 base design rollout，并为命中候选生成 2 条 proposal design rollout；随后根据方差与
   单样本成本冻结 evaluation 预算。分配原则来自
   [Neyman (1934)](https://doi.org/10.1111/j.2397-2335.1934.tb04184.x) 与
   [Étoré and Jourdain (2010)](https://doi.org/10.1007/s11009-008-9108-0)。

重复候选共享一次性 evaluation 库存。库存缺口由 fresh rollout 补齐，固定组的每个非终止候选保持
3 条 rollout，历史记录按一次观测消费。预算代理将历史重评分成本记为 1，将 fresh 生成与 0.5B
重评分成本记为 1.3200；最终计算量仍按实际 forward token slots 与各模型参数量统计。

| 方法 | 正确数 / 32 | 准确率 | 辅助候选占比 | rollout 复用率 | 平均外层 ESS | 平均最终 ESS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base 候选 + 固定 fresh | 23 | 71.875% | 0% | 0% | 8.000 | 6.323 |
| 动态候选 + 固定 replay | 21 | 65.625% | 51.282% | 34.943% | 4.365 | 3.424 |
| 动态候选 + 方差—成本分配 | 23 | 71.875% | 51.282% | 5.707% | 4.382 | 3.315 |

动态固定组相对 base 固定组的准确率差为 -6.25 个百分点，逐题配对 bootstrap 95% 区间为
[-21.875, 9.375]；逐题赢 2、输 4、平 26。方差—成本版本相对动态固定组为 +6.25 个百分点，区间为
[-6.250, 18.750]；逐题赢 3、输 1、平 28。两个区间均覆盖 0。完整版本与 base 固定组的点估计同为
23/32，平均最终 ESS 为 3.315；当前 design 样本量尚未形成更稳定权重的证据。

replay 在线与冷启动成本、动态候选的分阶段 FLOPs、缓存利用率及连续批处理结果见基础设施报告。

## 质量与预算消融

各消融点使用同一组 8 道题；单题贡献为 0 或 1，准确率最小变化为 12.5%。图中误差线为 Wilson 95%
区间。

![GSM8K 候选数、引导阶段、MH 更新次数和生成长度消融](../assets/gsm8k_3090_aligned_ablations.svg)

| 维度 | 设置与正确数 | 8 题合计 PFLOPs | 观测结果 |
| --- | --- | --- | --- |
| 标准 IS 候选数 `M`，`K=3` | `3/5/8/10 → 6/6/6/5` | `0.1281/0.2064/0.3068/0.3861` | `M=3` 达到本组最高点估计 |
| 0.5B proposal IS 候选数 `M` | `3/5/8/10 → 3/5/4/5` | `0.2299/0.3602/0.5730/0.6755` | 各点位于标准 IS 的质量—FLOPs 前沿下方 |
| 标准 IS rollout 数 `K`，`M=10` | `1/3/5 → 5/5/6` | `0.2319/0.3861/0.6063` | `K=5` 增加一题正确，同时增加计算成本 |
| 引导阶段数 `S` | `2/4/8/16 → 5/6/6/6` | `0.1293/0.3068/0.7426/1.8104` | `S=4` 后点估计保持 6/8 |
| MH 幂次 `α` | `1/2/4/8 → 3/4/6/3` | `0.2895/0.3067/0.3108/0.3148` | `α=4` 在本组得到最高点估计 |
| MH 每阶段更新数 `U` | `1/2/5/10 → 3/5/6/7` | `0.1526/0.2266/0.4726/0.8569` | 更新数增加改善有限链结果，边际成本同步上升 |
| 0.5B proposal 权重 | `截断/无截断 → 4/5` | `0.5730/0.5253` | 单次消融相差一题；32×8 网格的 pass@1 相同 |
| 最大生成长度 | 标准 IS：`128/256/512 → 4/7/6` | `0.2669/0.3151/0.2381` | 256 token 在本组取得最高点估计 |

奖励消融中，标准 IS 的 self-consistency 得到 6/8；平均 token 对数概率、平均负熵和自确定性分别得到
4/8、5/8、5/8，各使用约 0.9 PFLOPs，高于 self-consistency 的 0.3068 PFLOPs。使用 test split 标准
答案作为选择信号时，oracle Best-of-8 与 oracle 标准 IS 均得到 8/8，表明多数题目的候选池已经包含
正确数值答案，主要误差来源为部署可用的选择信号。

sampling temperature 为 0.7、1.0、1.5 时，标准 IS 分别得到 4/8、6/8、1/8。temperature 1.5 条件下，
无法解析或达到长度上限的生成序列比例上升。

## 结论与适用范围

1. 标准条件 IS 的单次生成准确率与本地 GRPO 接近，并高于 Base；其推理 FLOPs 约为已训练 GRPO 的
   54 倍。该方法适用于训练成本需要规避或查询量较小的场景。
2. 当前 0.5B off-policy proposal 相对标准 IS 同时降低准确率并增加 FLOPs。删除主模型重评分后，
   pass@1 相对截断版本变化 -1.172 个百分点且配对区间覆盖 0，FLOPs 降至 23.9%；该版本改为估计
   0.5B continuation energy，不再保持 Base 目标的 off-policy IS 解释。
3. warm replay、动态候选固定组和方差—成本组的配对区间均覆盖 0；当前样本规模对稳定质量差异和质量
   等价的统计分辨率有限。
4. 方差—成本分配复用 5.707% evaluation rollout，平均最终 ESS 为 3.315，低于动态固定组的 3.424。
   本组实验尚未观测到权重稳定性收益。
5. 幂分布 MH 的主要效应为基模概率锐化和多样性收缩。共享正确性奖励下，verifier-MH 与标准
   verifier-IS 得到接近的准确率点估计。

结果范围为固定 32 道 test 题和单张 RTX 3090。完整 1,319 题评测可进一步缩小准确率与分布距离区间。
FLOPs 估算范围排除二次 attention、逐元素 kernel、tokenization 和主机调度。机器可读结果位于
`results/gsm8k_3090/gsm8k_3090_aligned_*_validated.json`，图表由相应汇总 JSON 确定性生成；文件索引见
[`results/README.md`](../../results/README.md)。

## 参考文献

1. Cobbe, K., et al. (2021). [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168). arXiv:2110.14168.
2. Yang, A., et al. (2024). [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115). arXiv:2412.15115.
3. Hu, E. J., et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://openreview.net/pdf?id=nZeVKeeFYf9). ICLR 2022.
4. Freitag, M., and Al-Onaizan, Y. (2017). [Beam Search Strategies for Neural Machine Translation](https://aclanthology.org/W17-3207/). NGT 2017, 56–60.
5. Wang, X., et al. (2023). [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://openreview.net/pdf?id=1PL1NIMMrw). ICLR 2023.
6. Hastings, W. K. (1970). [Monte Carlo Sampling Methods Using Markov Chains and Their Applications](https://doi.org/10.1093/biomet/57.1.97). Biometrika, 57(1), 97–109.
7. Hesterberg, T. (1995). [Weighted Average Importance Sampling and Defensive Mixture Distributions](https://doi.org/10.1080/00401706.1995.10484303). Technometrics, 37(2), 185–194.
8. Precup, D., Sutton, R. S., and Singh, S. (2000). [Eligibility Traces for Off-Policy Policy Evaluation](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf). ICML 2000, 759–766.
9. Shao, Z., et al. (2024). [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). arXiv:2402.03300.
10. Lin, L.-J. (1992). [Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching](https://doi.org/10.1007/BF00992699). Machine Learning, 8, 293–321.
11. Neyman, J. (1934). [On the Two Different Aspects of the Representative Method](https://doi.org/10.1111/j.2397-2335.1934.tb04184.x). Journal of the Royal Statistical Society, 97, 558–606.
12. Étoré, P., and Jourdain, B. (2010). [Adaptive Optimal Allocation in Stratified Sampling Methods](https://doi.org/10.1007/s11009-008-9108-0). Methodology and Computing in Applied Probability, 12, 335–360.
13. Chen, M., et al. (2021). [Evaluating Large Language Models Trained on Code](https://arxiv.org/abs/2107.03374). arXiv:2107.03374.
14. Wilson, E. B. (1927). [Probable Inference, the Law of Succession, and Statistical Inference](https://doi.org/10.1080/01621459.1927.10502953). Journal of the American Statistical Association, 22(158), 209–212.
15. Efron, B. (1979). [Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/aos/1176344552). The Annals of Statistics, 7(1), 1–26.
16. Lin, J. (1991). [Divergence Measures Based on the Shannon Entropy](https://doi.org/10.1109/18.61115). IEEE Transactions on Information Theory, 37(1), 145–151.
