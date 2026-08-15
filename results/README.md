# 结果索引

本目录保存可由脚本读取的汇总结果。逐题 JSONL、pass@k chunks、checkpoint 和运行日志由
`.gitignore` 管理。

## GSM8K 质量结果

目录：[`gsm8k_3090/`](gsm8k_3090/)

| 文件 | 内容 |
| --- | --- |
| `gsm8k_3090_aligned_comparison_validated.json` | 单次生成质量、成本与配对区间 |
| `gsm8k_3090_aligned_distribution_audit_validated.json` | 共享奖励下的答案分布 TV/JS |
| `gsm8k_3090_aligned_passk_validated.json` | Base、MH 与 GRPO 的独立 draw |
| `gsm8k_3090_aligned_is_passk_validated.json` | 标准、截断和未截断 off-policy IS |
| `gsm8k_3090_aligned_is_uncorrected_validated.json` | 0.5B rollout、零主模型重评分消融 |
| `gsm8k_3090_aligned_is_rescoring_ablation_validated.json` | IS 重评分配对区间与成本比 |
| `gsm8k_3090_aligned_verifier_rescoring_ablation_validated.json` | 精确奖励下的重评分消融 |
| `gsm8k_3090_aligned_passk_comparison_validated.json` | 六种方法的统一 pass@k 与权重诊断 |
| `gsm8k_3090_aligned_ablations_validated.json` | 候选、rollout、阶段、奖励、温度与长度消融 |

算法定义见[推理算法实现](../docs/methods/ALGORITHMS.md)，文字结果见
[GSM8K 方法质量与计算量实验](../docs/reports/GSM8K_3090_ALIGNED_RESULTS.md)。

## 推理执行结果

| 目录或文件 | 内容 |
| --- | --- |
| `gsm8k_3090/gsm8k_3090_aligned_compute_validated.json` | GRPO 训练成本与累计 FLOPs 交点 |
| `gsm8k_3090/gsm8k_3090_aligned_replay_validated.json` | fresh、warm、cache build 与摊销次数 |
| `gsm8k_3090/gsm8k_3090_aligned_dynamic_is_validated.json` | 动态候选、design、复用率与成本 |
| `gsm8k_3090/gsm8k_3090_aligned_async_grouped_validated.json` | 逐 prompt 与连续批处理 |
| `infra/rtx3090_transformers_summary.json` | 历史树、progressive、run-ahead 与 SMC 的三 seed 聚合 |
| `infra/rtx3090_transformers_is_mh_summary.json` | broker、流式 IS、草稿和 MH 复用的三 seed 聚合 |
| `infra/rtx3090_transformers_decode_*.json` | rollout 解码实验的单 seed 记录 |
| `infra/rtx3090_transformers_algorithms_*.json` | progressive 与 SMC 的单 seed 记录 |
| `infra/rtx3090_transformers_is_mh_seed*.json` | IS/MH 复用实验的单 seed 记录 |

执行机制和成本分母见[推理基础设施实现](../docs/methods/INFRASTRUCTURE.md)，文字结果见
[RTX 3090 推理执行与 rollout 复用实验](../docs/reports/RTX3090_ROLLOUT_INFRA.md)。

## 训练与验证

| 路径 | 内容 | 用途 |
| --- | --- | --- |
| `training/gsm8k_grpo_training_summary.json` | 205 步 GRPO 的 token、FLOPs、显存、墙钟与 adapter 哈希 | 训练成本 |
| `validation/gsm8k_quick_*_validated.json` | 主比较、replay、批处理和计算路径 | 集成检查 |
| `validation/rtx3090_reproduction.json` | FP32 真实模型算法与后端结果 | 工程检查 |
| `validation/rtx3090_backend_bfloat16.json` | BF16 概率与显存结果 | 精度检查 |

对应记录见 [GSM8K quick 集成检查](../docs/validation/GSM8K_QUICK_VALIDATION.md) 和
[RTX 3090 复现记录](../docs/validation/RTX3090_REPRODUCTION.md)。

## 产物状态

- 原始记录位于 `results/gsm8k/<profile>/`，按 manifest fingerprint 追加并恢复。
- `*.chunks.jsonl` 保存 pass@k 任务块；正式汇总记录其 SHA-256。
- `validated` 后缀要求题目网格完整、manifest 一致、输入哈希一致且后处理成功。
- 统计适用范围由对应报告中的样本、模型和硬件设置确定。
