# Qwen3-1.7B：思考模式、IS 与 MH 的质量和计算量

Qwen3-1.7B 在 MATH-500 Level 5 的 30 题上评测（子集与生成种子 20260911，开发题目排除），温度 0.6，完整词表采样，
BF16，单张 RTX 3090，Math-Verify 评定最终答案。每个方法与预算组合每题评测一次；两档预算为每题 32,768 /
131,072 个前向 token 位置，多条生成的方法把单条的提示加生成限制在 8,192 / 16,384 token。以下结论来自统一入口
之前的实验，数字按当时的实现测得；复现命令对应当前实现。

## 结论

- **思考模式提高了准确率。** 单次采样从 14/30 提高到 21/30，平均前向 token 位置数从 1,333 增至 10,024。
- **高预算下 log-probability IS 的正确数最高。** 24/30，比思考模式的单次采样多 3 题（+10.0 个百分点，区间
  [−3.3, 23.3]），计算量约为其 3.6 倍；与同一候选池的多数投票计算量相同，多答对 1 题。每题一次输出，1 题约
  3.3 个百分点，区间仍包含 0。
- **单条长度分配影响低预算结果。** 把思考模式的单次采样限制到 8,192 / 16,384 token 时，准确率为 46.7% / 66.7%，
  完整预算为 70.0%；两档预算同时改变候选数与单条长度。
- **Consilience 与短链 MH 的收益有限。** 高预算 Consilience IS 与思考模式单次采样同为 70.0%，开销更高；三种奖励的
  MH（每题 3 次更新）为 60.0%–70.0%。思考段未闭合时 Consilience 回退到全序列评分，末段窗口反映的是中途状态的
  置信度，本组结果混合了完整思考段与截断轨迹两种评分。

## 复现

模型设为 `ar.model.path = "Qwen/Qwen3-1.7B"`（`revision` 固定版本，`weight_sha256` 固定权重），并设
`ar.engine.dtype = "bfloat16"`、`ar.sampling.temperature = 0.6`、`datasets.math500.selection.count = 30`。思考模式由
`ar.prompt.chat_template_kwargs.enable_thinking` 与 `ar.output.thinking_mode = "enabled"` 打开，普通采样取
`false` 与 `"disabled"`。

| 结论中的方法 | 命令 | 设置改动 |
| --- | --- | --- |
| 普通采样、思考模式单次采样 | `python -m inference_scaling --algorithm sample --dataset math500` | 见上；单条长度改 `datasets.math500.max_new_tokens` |
| 多数投票 | `--algorithm best_of_n --dataset math500` | `rewards.verifier.source = "vote"`，`best_of_n.samples = 2 / 4` |
| IS，三种奖励 | `--algorithm is --dataset math500 --reward verifier`（或 `logprob`、`consilience`） | `planning = "fixed"`，`fixed.candidate_count = 2 / 4`，`rollout_count = 1`，`block_size` 不小于单条长度（对完整序列一次重采样）；`max_new_tokens = 8192 / 16384` |
| MH，三种奖励 | `--algorithm mh --dataset math500 --reward …` | `iterations = 1 / 3`，`suffix_schedule = "uniform"` |

自一致性奖励设 `source = "vote"`、`pool_size = 2`、`temperature = 0.25`；log-probability 奖励 `temperature = 10`；
Consilience 使用默认设置。与当时实现的差异：log-probability 奖励当时是对数概率之和（目标为 $`p^{1.1}`$），当前
`logprob` 为长度平均；两档预算当时由统一的前向 token 预算约束，当前以候选数与单条长度复现。
