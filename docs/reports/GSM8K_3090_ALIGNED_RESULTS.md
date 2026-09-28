# 算法设计与准确率：Qwen2.5-1.5B / GSM8K

Qwen2.5-1.5B-Instruct 在 GSM8K 测试集固定抽取的 32 题上（子集种子 20260808）评测，最长 192 token，采样温度 1，
FP32，单张 RTX 3090。以下结论来自统一入口之前的实验，数字按当时的实现测得；复现命令对应当前实现。
执行成本见[执行成本报告](RTX3090_ROLLOUT_INFRA.md)，方法见[算法文档](../methods/ALGORITHMS.md)。

## 结论

- **条件 IS 的单次准确率接近 GRPO，推理计算量高得多。** 以自一致性为奖励（8 个候选、每候选 3 条补全、块长 48），
  条件 IS 为 65.6%（21/32），比基础采样高 25 个百分点；GRPO 训练后随机采样为 68.8%，差值区间为 [−12.5, 6.25]
  个百分点。条件 IS 的单次推理 FLOPs 约为 GRPO 采样的 54 倍。多次采样时 GRPO 覆盖更好（pass@8：81.3% 对 75.0%）。
- **多数投票与 Beam 的收益有限。** 8 路多数投票为 43.8%，Beam-8 为 37.5%，基础采样为 40.6%。
- **幂目标 MH 没有提高准确率，主要降低多样性。** α=4 时为 37.5%；每题不同数值答案数从 4.56 降至 3.25。
- **读取正确答案的奖励给出上限诊断。** 以答案正确性为奖励时，MH 为 78.1%，条件 IS 为 75.0%，均高于 GRPO；
  它们在选择时读取测试答案，不是可部署的方法。
- **预算敏感性只作趋势参考。** 8 题上，MH 每阶段更新从 1 次增至 10 次，正确数从 3 增至 7；IS 的候选数与块数
  增加没有带来明确收益；生成长度同样影响结果。样本小，区间较宽。
- **后缀长度分布主要改变成本。** 多尺度后缀的准确率点估计较高（18.0% 对均匀后缀 12.5%），区间覆盖 0；
  明确的收益是生成 token 与墙钟下降。
- **奖励选择。** 8 题筛选中，自一致性优于 token 平均对数概率等模型自身信号。

## 复现

设置文件的默认值就是本报告的配置：模型、32 题、192 token、温度 1、FP32，Beam 8 路，Best-of-N 8 个样本，
`mh_power` 的 α=4、块长 12、每阶段 3 次更新、均匀后缀，条件 IS 的 8 个候选、3 条补全、块长 48。自一致性奖励设
`rewards.verifier.source = "vote"`，正确性奖励保持 `"dataset"`。

| 结论中的方法 | 命令 | 设置改动 |
| --- | --- | --- |
| 基础采样 | `python -m inference_scaling --algorithm sample` | — |
| Beam-8 | `python -m inference_scaling --algorithm beam` | — |
| 多数投票-8 | `python -m inference_scaling --algorithm best_of_n` | `rewards.verifier.source = "vote"` |
| 幂目标 MH | `python -m inference_scaling --algorithm mh_power` | — |
| 条件 IS，自一致性 | `python -m inference_scaling --algorithm is` | `rewards.verifier.source = "vote"`，`ar.algorithms.is.planning = "fixed"` |
| 正确性奖励的 MH 与 IS | `python -m inference_scaling --algorithm mh`、`--algorithm is` | IS 设 `planning = "fixed"` |
| GRPO | `python -m training`，再用 `--algorithm sample`、`--algorithm greedy` 评测 | 训练时 `stages = ["grpo"]`；评测时 `ar.model.adapter` 指向 `grpo.output` |
| pass@k | 同上各命令 | `run.draws = 8` |
| 预算敏感性 | `--algorithm mh_power`、`--algorithm is` | `datasets.gsm8k.selection.count = 8`；`steps_per_block`、`fixed.candidate_count`、`datasets.gsm8k.max_new_tokens` |
| 后缀长度分布 | `--algorithm mh_power` | `suffix_schedule`；`max_new_tokens = 128`、`block_size = 16`、`steps_per_block = 2`、`run.draws = 4` |
| 奖励选择 | `--algorithm is --reward verifier`、`--reward logprob` | `selection.count = 8`，`planning = "fixed"`，verifier 设 `source = "vote"` |

与当时实现的差异：自一致性奖励现在与冻结的 `pool_size` 个独立样本比较（当时为已评估补全的累计众数）；
正确性奖励的温度为 0.1（当时为 0.04）。
