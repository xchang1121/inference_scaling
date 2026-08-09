# GSM8K quick 集成检查

> 这是工程验证记录，不是当前主实验报告。正式比较见
> [GSM8K 单卡对齐实验](../reports/GSM8K_3090_ALIGNED_RESULTS.md)。

## 范围

本记录使用 `configs/gsm8k_quick.toml` 和 tag `validated`。主方法、共同目标诊断与 replay 使用官方
GSM8K 测试集中的 8 条固定样本；异步调度使用 32 条固定样本和 8 个并发 worker。模型为固定 revision
的 Qwen2.5-1.5B-Instruct，off-policy proposal 为同系列 0.5B 模型，RL 对照为本机训练的 GRPO LoRA。

这是一轮集成检查，目的是验证方法路径、概率、缓存、计算账本和结果汇总能够一起工作。8 条样本的
Wilson 区间很宽，不能据此确认稳定的质量排序。可追溯结果为：

- `results/validation/gsm8k_quick_comparison_validated.json`；
- `results/validation/gsm8k_quick_replay_validated.json`；
- `results/validation/gsm8k_quick_async_validated.json`；
- `results/validation/gsm8k_quick_compute_validated.json`；
- `results/training/gsm8k_grpo_training_summary.json`。

这些 JSON 保存模型权重、关键实现文件和方法 manifest 的 SHA-256；原始逐题记录保留在本机并被
Git 忽略。

## 单次回答质量与计算量

| 方法 | 正确数 / 8 | 准确率 | 相对单次 Base 的估算 FLOPs | 相对 Base 的墙钟时间 |
|---|---:|---:|---:|---:|
| Base | 2 | 25.0% | 1.00× | 1.00× |
| Beam-4 | 2 | 25.0% | 4.00× | 2.37× |
| Best-of-4 | 3 | 37.5% | 2.59× | 2.15× |
| 幂分布 MH | 2 | 25.0% | 18.20× | 10.65× |
| 条件 IS | 3 | 37.5% | 21.27× | 5.12× |
| 0.5B proposal 条件 IS | 2 | 25.0% | 33.54× | 4.38× |
| GRPO 随机采样 | 5 | 62.5% | 0.93× | 3.18× |
| GRPO 贪心 | 4 | 50.0% | 0.97× | 3.48× |

这里的墙钟时间排除模型加载，但包含实际采样；GRPO 推理首次加载 adapter 的固定开销不在逐题计时
中。普通幂分布 MH 的目标是基模概率的四次幂，并不等于 GRPO 的 exact-verifier 目标，因此这一行与
GRPO 的准确率差异不能解释为 MH 无法近似同一个目标。

0.5B proposal 条件 IS 相对标准条件 IS 的墙钟时间快 `1.169×`，但估算 FLOPs 反而是标准版本的
`1.577×`。分母是相同问题、候选数、rollout 数、block、seed 与最大长度下的标准 on-policy 条件
IS；小模型减少了主模型生成，却需要主模型重评分和精确 importance correction。因此这轮只能称为
墙钟加速，不能称为 FLOPs 缩减。

## 共同目标诊断

共同目标固定为

`base probability × exp(exact numeric verifier reward / 0.04)`。

| 方法 | 正确数 / 8 | 准确率 | 估算 FLOPs | 墙钟时间 |
|---|---:|---:|---:|---:|
| verifier MH | 5 | 62.5% | 0.0969 PFLOPs | 278.9 s |
| verifier 条件 IS | 5 | 62.5% | 0.1210 PFLOPs | 115.5 s |
| 0.5B proposal verifier 条件 IS | 4 | 50.0% | 0.1751 PFLOPs | 76.0 s |
| GRPO 随机采样 | 5 | 62.5% | 0.0054 PFLOPs/这 8 次推理 | 61.3 s |

verifier MH 与 GRPO 在这 8 条样本上的逐题正确向量完全相同；verifier 条件 IS 的总正确数相同，但
答对的具体题目不完全相同。这个结果与“直接采样和训练可以近似同一输出目标”的预期一致，但样本太少，
还不能确认统计等价。GRPO 行的推理 FLOPs 不包含一次性的 15.646 PFLOPs 训练成本；重复查询的盈亏平衡
必须在正式计算报告中另算。

quick 账本把这次相同准确率暂时视为“准确率匹配”，得到以下探索性盈亏平衡：verifier MH 相对 GRPO
随机采样约为 1,369 次查询（FLOPs）或 352 次查询（本机墙钟）；verifier 条件 IS 约为 1,083 次
查询（FLOPs）或 1,410 次查询（墙钟）。这些数字的分母都是“无训练方法每题成本减去 GRPO 每题推理
成本”，并包含 GRPO 的一次性 15.646 PFLOPs / 9,545 s 训练成本。0.5B proposal verifier 条件 IS
比 GRPO 低 12.5 个百分点，因此只报告原始盈亏平衡，不把它标成质量匹配。该 quick 产物没有运行
答案分布审计，所以不能给出“准确率与输出分布同时匹配”的结论；正式实验已单独补充该诊断。

## rollout replay

replay 的分母是相同候选、相同 `H+F` 总 rollout 预算的 fresh-only 决策。本配置使用一条历史 rollout
和一条 fresh rollout，平均实际复用率为 `42.86%`。

| 比较 | FLOPs 因子 | 墙钟因子 | 解释 |
|---|---:|---:|---|
| fresh-only / warm replay 在线阶段 | 1.020× | 1.024× | 大于 1 才表示 warm replay 更省；本轮收益约 2% |
| fresh-only / 首次 cold-start replay | 0.464× | 0.527× | 小于 1；包含缓存构建时，首次 replay 更贵 |

换一种方向表达，首次 cold-start replay 使用约 `2.156×` FLOPs、耗时约 `1.898×`。历史数据只有在被
后续多次决策复用时才可能摊薄构建成本；quick 配置没有展示出显著的在线收益。

## 连续批处理

异步分母均为同一方法的逐 prompt 同步执行。请求级 seed 固定随机流，inverse-CDF 使用 FP64 累加；
不同 CUDA batch 形状仍可能产生轻微 logits 差异并在长条件生成中放大，因此同时报告 token 与数值
答案匹配，而不是假定逐 token 相同。

| 方法 | 同步 / 异步墙钟 | 异步 / 同步 FLOPs | token 完全匹配 | 数值答案匹配 | 平均共同前缀比例 |
|---|---:|---:|---:|---:|---:|
| Base | 5.420× | 1.149× | 32/32 | 32/32 | 100.00% |
| Best-of-4 | 2.827× | 1.042× | 32/32 | 32/32 | 100.00% |
| 条件 IS | 1.413× | 1.619× | 26/32 | 30/32 | 88.26% |
| 0.5B proposal 条件 IS | 1.554× | 1.091× | 30/32 | 30/32 | 93.77% |

四种方法的同步与异步准确率在本轮都相同。Base 与 Best-of-N 是固定 trace 的严格加速结果；两种条件
方法存在 live sampling path 分叉，其墙钟因子只表示相同配置与 seed 下的真实 workload 对比。连续
批处理提高硬件利用率，但 padding 和分叉路径使模型处理的 token slots 增加，因此它不是 FLOPs 优化。

## 与正式结果的关系

后续 32 题单卡对齐实验沿用相同方法和计量口径，并补充独立 draw 的 pass@k、共享目标答案分布审计、
更完整的 replay 成本及消融。quick 阶段观察到的小 proposal 墙钟优势没有在正式实验中转化为质量匹配
或 FLOPs 缩减；replay 的在线收益则在更完整的缓存实验中变得可测。后续结论统一以
[主实验报告](../reports/GSM8K_3090_ALIGNED_RESULTS.md)为准。
