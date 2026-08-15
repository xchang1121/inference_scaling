# GSM8K quick 集成检查

本记录验证 `configs/gsm8k_quick.toml` 的端到端执行。正式统计结论见
[GSM8K 方法质量与计算量实验](../reports/GSM8K_3090_ALIGNED_RESULTS.md)。

## 设置

| 项目 | 设置 |
| --- | --- |
| 数据 | 主方法、共享目标与 replay：8 道固定 GSM8K test 题；批处理：32 道固定题 |
| 模型 | Qwen2.5-1.5B-Instruct；0.5B rollout proposal；GRPO LoRA |
| 并发 | 批处理实验使用 8 个 worker |
| 统计范围 | 集成检查；8 题准确率的 Wilson 区间较宽 |

产物：

- `results/validation/gsm8k_quick_comparison_validated.json`
- `results/validation/gsm8k_quick_replay_validated.json`
- `results/validation/gsm8k_quick_async_validated.json`
- `results/validation/gsm8k_quick_compute_validated.json`
- `results/training/gsm8k_grpo_training_summary.json`

## 单次生成

| 方法 | 正确数 / 8 | 准确率 | 相对 Base FLOPs | 相对 Base 墙钟 |
| --- | ---: | ---: | ---: | ---: |
| Base | 2 | 25.0% | 1.00× | 1.00× |
| Beam-4 | 2 | 25.0% | 4.00× | 2.37× |
| Best-of-4 | 3 | 37.5% | 2.59× | 2.15× |
| 幂分布 MH | 2 | 25.0% | 18.20× | 10.65× |
| 条件 IS | 3 | 37.5% | 21.27× | 5.12× |
| 0.5B rollout proposal 条件 IS | 2 | 25.0% | 33.54× | 4.38× |
| GRPO 参数 + 随机采样 | 5 | 62.5% | 0.93× | 3.18× |
| GRPO 参数 + 贪心解码 | 4 | 50.0% | 0.97× | 3.48× |

墙钟排除模型加载。0.5B rollout proposal 路径相对标准条件 IS 的墙钟因子为 `0.855×`，FLOPs 因子为
`1.577×`；主模型后缀评分增加了逻辑计算量。

## 共享奖励

目标为 `p_base(y|x) × exp(r_exact(y) / 0.04)`。

| 方法 | 正确数 / 8 | 准确率 | 推理 FLOPs | 墙钟 |
| --- | ---: | ---: | ---: | ---: |
| verifier-MH | 5 | 62.5% | 0.0969 PFLOPs | 278.9 s |
| verifier 条件 IS | 5 | 62.5% | 0.1210 PFLOPs | 115.5 s |
| 0.5B rollout proposal verifier-IS | 4 | 50.0% | 0.1751 PFLOPs | 76.0 s |
| GRPO 参数 + 随机采样 | 5 | 62.5% | 0.0054 PFLOPs | 61.3 s |

verifier-MH 与 GRPO 的逐题正确向量相同；verifier-IS 的总正确数相同，题目集合不同。GRPO 推理成本
排除一次性 15.646 PFLOPs 训练成本。正式实验使用 32 题、独立 draw 和答案分布审计。

## rollout replay

每个候选使用一条历史 rollout 和一条 fresh rollout，平均复用率为 42.86%。

| 路径 / 对照 | FLOPs 因子 | 墙钟因子 |
| --- | ---: | ---: |
| warm replay 在线 / fresh-only | 0.980× | 0.977× |
| cache build + 首次 warm / fresh-only | 2.156× | 1.898× |

## 连续批处理

分母为同一方法的逐 prompt 同步执行。

| 方法 | 同步 / 批处理墙钟 | 批处理 / 同步 FLOPs | token 匹配 | 数值答案匹配 | 平均共同前缀 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 5.420× | 1.149× | 32/32 | 32/32 | 100.00% |
| Best-of-4 | 2.827× | 1.042× | 32/32 | 32/32 | 100.00% |
| 条件 IS | 1.413× | 1.619× | 26/32 | 30/32 | 88.26% |
| 0.5B rollout proposal 条件 IS | 1.554× | 1.091× | 30/32 | 30/32 | 93.77% |

请求级随机流保持固定；CUDA batch 形状引起的数值分叉由 token 匹配、数值答案匹配和共同前缀共同记录。
