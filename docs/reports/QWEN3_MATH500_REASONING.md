# Qwen3-1.7B：思考模式、IS 与 MH 的质量和计算量

使用 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 的固定权重，在
[MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) 的 Level 5 分层子集中按固定顺序取前 30 题（子集与
生成种子 20260911，开发题目排除），单张 RTX 3090，BF16、SDPA；最终答案由 Math-Verify 评定，标准答案只用于评测。
每个方法与预算组合，每题评测一次随机输出。结论来自统一入口之前的实验，数字按当时的实现测得；复现命令对应当前实现。

| 项目 | 当时的设置 |
| --- | --- |
| 生成 | 温度 0.6，完整词表采样，最大生成长度 32,768 token，同时受模型上下文与预算约束 |
| 普通采样与思考模式单次采样 | 分别关闭、开启模型的思考模式；单条生成可使用整档预算 |
| 两档预算 | 每题 32,768 / 131,072 个前向 token 位置，包含提示、生成、独立样本与奖励评分；按 `2 × 1,720,574,976 × 前向 token 位置数` 估算 FLOPs |
| 多数投票、IS、MH | 投票与 IS 使用 2 / 4 条候选，投票平票时取最早的可解析候选；IS 对完整序列做一次加权重采样；MH 均匀选择后缀起点，更新 1 / 3 次 |
| 多条生成的单条长度 | 提示加生成限制在 8,192 / 16,384 token，为生成与评分预留预算 |
| 奖励 | 自一致性：与两个独立生成并固定的样本在最终答案上的一致比例；log-probability：实际采样策略的逐 token 对数概率之和；Consilience：思考段末期置信度均值减去 3 倍初期均值 |
| 奖励参数 | 温度依次为 0.25 / 10 / 2；Consilience 取 top-5、20% 窗口、跳过初始 5%、初始窗口系数 3、置信度评分温度 1 |

## 结论

- **思考模式提高了准确率。** 单次采样从 14/30 提高到 21/30，平均前向 token 位置数从 1,333 增至 10,024。
- **高预算下 log-probability IS 的正确数最高。** 24/30，比思考模式单次采样多 3 题（+10.0 个百分点，题目级配对
  自助法 95% 区间 [−3.3, 23.3]，仍包含 0），平均前向 token 位置数约为其 3.6 倍；与同一候选池的多数投票计算量相同，
  多答对 1 题。
- **单条长度分配影响低预算结果。** 把思考模式单次采样限制到 8,192 / 16,384 token 时，准确率为 46.7% / 66.7%，完整
  预算为 70.0%；两档预算同时改变候选数与单条长度，预算增益包含两者的作用。
- **Consilience 与短链 MH 的收益有限。** 高预算 Consilience IS 与思考模式单次采样同为 70.0%，开销更高；三种奖励的
  MH 为 60.0%–70.0%。每种奖励在 30 题上各进行 90 次更新，其中 46、45、47 次产生了不同的候选序列，结论对应每题
  3 次更新的有限步设置。

### Consilience 的思考段切分与截断

- **切分正确。** 逐 token 核查 210 条原始生成与两档预算下 Consilience IS 的 180 次候选使用，切分均与原始标签一致；
  完整输出只对 `<think>` 与 `</think>` 之间的内容评分，开标签保留为上下文。
- **截断使评分混合了两种情形。** 低、高预算下分别有 40% 与 20% 的 IS 候选因达到单条长度上限、尚未生成 `</think>`
  而回退到全序列评分，其末期窗口反映的是中途状态的置信度；两档预算分别有 1、2 题在候选池存在正确输出时选中了未闭合
  候选。本组结果评价的是“完整思考段评分与截断轨迹全序列评分”的混合设置，对完整思考轨迹的奖励效果仍需单独验证。

## 复现

模型设为 `ar.model.path = "Qwen/Qwen3-1.7B"`（`revision` 固定版本，`weight_sha256` 固定权重），并设
`ar.engine.backend = "transformers"`、`ar.sampling.temperature = 0.6`、`ar.prompt.chat_template_kwargs = {}`、
`ar.output.sampling_scope = "full"`、`datasets.math500.selection.count = 30`、`max_new_tokens = 32768`。BF16 与思考模式
（`ar.output.thinking_mode = "enabled"`）同默认设置，普通采样取 `"disabled"`。

| 方法 | 命令 | 设置改动 |
| --- | --- | --- |
| 普通采样、思考模式单次采样 | `python -m inference_scaling --algorithm sample --dataset math500` | 见上；单条长度改 `datasets.math500.max_new_tokens` |
| 多数投票 | `--algorithm best_of_n --dataset math500 --reward verifier` | `rewards.verifier.source = "vote"`，`best_of_n.samples = 2 / 4` |
| IS，三种奖励 | `--algorithm is --dataset math500 --reward verifier`（或 `logprob`、`consilience`） | `planning = "fixed"`，`fixed.candidate_count = 2 / 4`，`rollout_count = 1`，`block_size` 不小于单条长度（对完整序列一次重采样）；`max_new_tokens = 8192 / 16384` |
| MH，三种奖励 | `--algorithm mh --dataset math500 --reward …` | `iterations = 1 / 3`，`suffix_schedule = "uniform"` |

自一致性奖励设 `source = "vote"`、`pool_size = 2`、`temperature = 0.25`；log-probability 奖励设 `temperature = 10`；
Consilience 使用默认设置。与当时实现的差异：log-probability 奖励当时是对数概率之和（目标为 $`p^{1.1}`$），当前
`logprob` 为长度平均；两档预算当时由统一的前向 token 预算约束，当前以候选数与单条长度复现。
