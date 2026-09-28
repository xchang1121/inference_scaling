# 推理成本与执行效率：RTX 3090

Qwen2.5-1.5B-Instruct，单张 RTX 3090 24 GiB，Transformers 后端。FLOPs 按“2 × 参数量 × 实际前向 token 位置数”估算，
不含注意力的长度平方项。以下结论来自统一入口之前的实验，数字按当时的实现测得；复现命令对应当前实现。
准确率见[算法质量报告](GSM8K_3090_ALIGNED_RESULTS.md)，机制见[算法文档](../methods/ALGORITHMS.md#alg-runtime)，
成本口径见 [BUDGET.md](../methods/BUDGET.md#budget-accounting)。

## 结论

- **连续批处理主要提高设备利用率。** 8 个题目并发共享后端时，32 题的基础采样墙钟下降 79%，8 路投票下降 50%，
  条件 IS 只下降 13%，受候选、补全与选择之间的阶段依赖限制；填充使前向 token 位置数略增。
- **短后缀减少 MH 的串行生成。** 相对均匀后缀，长度倒数与多尺度后缀分别减少约 50% 与 32% 的生成 token，墙钟
  降为 0.54 与 0.59 倍。当时每次提议都重新预填充保留的前缀，FLOPs 基本不变；当前的前缀 KV 存储复用这部分计算。
- **冻结历史 proposal 以并行评分替代部分串行生成。** 在线墙钟降为 0.53 倍（计入建库为 0.59 倍），FLOPs 约增 7%；
  与多尺度后缀组合时墙钟降为 0.36 倍。
- **训练与推理成本需要一并比较。** 该 GRPO 检查点训练消耗 15.6 PFLOPs、9,545 s；32 题上条件 IS 的单次推理 FLOPs
  约为 GRPO 训练后采样的 54 倍（1.37 对 0.025 PFLOPs），两者的单次准确率点估计接近。
- 选择执行方案时应同时比较墙钟、FLOPs 以及当前请求实际承担的建库成本。

## 复现

每条记录的 `cost` 按阶段给出前向 token 位置数与 FLOPs，结果目录的 `summary.json` 汇总墙钟与成本。

| 结论 | 命令 | 设置改动 |
| --- | --- | --- |
| 连续批处理 | `python -m inference_scaling --algorithm sample`（或 `best_of_n`、`is`） | `ar.engine.continuous_batching.workers = 8`，对照为 1；投票设 `rewards.verifier.source = "vote"`，IS 设 `planning = "fixed"` |
| 后缀长度分布 | `--algorithm mh_power` | `suffix_schedule` 三选一；`datasets.gsm8k.max_new_tokens = 128`、`block_size = 16`、`steps_per_block = 2`、`run.draws = 4` |
| 冻结历史 proposal | `--algorithm mh` | `ar.algorithms.mh.proposal = "frozen_history"`，对照为 `"base"`；组合时再设 `suffix_schedule = "multiscale"` |
| 训练成本 | `python -m training` | `stages = ["grpo"]`；训练成本写入 `grpo.output` 下的 `training_cost.json` |

与当时实现的差异：冻结历史与后缀对照当时用人工奖励（末 token 奇偶）在单题上测成本，当前命令用正式奖励在 GSM8K 上
测量同一机制。
