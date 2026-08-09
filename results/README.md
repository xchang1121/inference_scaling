# 结果索引

此目录只提交体积较小、可由脚本读取的汇总结果。逐题 JSONL、pass@k raw chunks、模型权重、checkpoint
和运行日志属于可恢复的中间产物，由 `.gitignore` 排除。

## GSM8K 单卡主实验

目录：[`gsm8k_3090/`](gsm8k_3090/)

| 文件 | 内容 |
| --- | --- |
| `gsm8k_3090_aligned_comparison_validated.json` | Base、搜索、MH、条件 IS 与 GRPO 的 pass@1、成本和配对区间 |
| `gsm8k_3090_aligned_compute_validated.json` | GRPO 训练成本、推理 FLOPs 和摊销临界查询数 |
| `gsm8k_3090_aligned_replay_validated.json` | fresh-only、warm replay、缓存构建与回本次数 |
| `gsm8k_3090_aligned_dynamic_is_validated.json` | base 固定候选、动态候选外层 IS 与方差—成本预算的质量、复用和分阶段成本 |
| `gsm8k_3090_aligned_async_grouped_validated.json` | 各方法同步/连续批处理墙钟、FLOPs 与输出一致性 |
| `gsm8k_3090_aligned_distribution_audit_validated.json` | 共享目标下的答案分布 TV/JS 诊断 |
| `gsm8k_3090_aligned_passk_validated.json` | Base、MH 与 GRPO 的独立 draw 汇总 |
| `gsm8k_3090_aligned_is_passk_validated.json` | 标准、截断 off-policy 与非截断 off-policy IS 的独立 draw 汇总 |
| `gsm8k_3090_aligned_passk_comparison_validated.json` | 六种方法的统一 pass@k、配对区间与权重诊断 |
| `gsm8k_3090_aligned_ablations_validated.json` | 候选、rollout、引导阶段、奖励、温度和长度消融 |

对应的可读报告是
[`docs/reports/GSM8K_3090_ALIGNED_RESULTS.md`](../docs/reports/GSM8K_3090_ALIGNED_RESULTS.md)。

## 训练摘要

目录：[`training/`](training/)

`gsm8k_grpo_training_summary.json` 记录 205 步 GRPO 训练的 rollout、token/FLOPs、硬件诊断、adapter
哈希和端到端加载检查。它是训练成本输入，不是测试集方法比较结果。

## 工程验证

目录：[`validation/`](validation/)

| 文件组 | 用途 |
| --- | --- |
| `gsm8k_quick_*_validated.json` | 检查主比较、replay、连续批处理与计算汇总路径能够贯通 |
| `*_runtime_*_{transformers,vllm,comparison}.json` | 同 setting 的 Transformers/vLLM 原始吞吐报告与成对汇总 |
| `rtx3090_reproduction.json` | FP32 真实模型算法与后端 smoke |
| `rtx3090_backend_bfloat16.json` | BF16 概率一致性和显存诊断 |

这些文件不用于最终方法排序。文字解释分别位于
[`docs/validation/GSM8K_QUICK_VALIDATION.md`](../docs/validation/GSM8K_QUICK_VALIDATION.md) 和
[`docs/validation/RTX3090_REPRODUCTION.md`](../docs/validation/RTX3090_REPRODUCTION.md)。

## 中间产物与晋级规则

- 默认原始记录位于 `results/gsm8k/<profile>/`，按 manifest fingerprint 追加写入并支持恢复。
- `*.chunks.jsonl` 保存 pass@k 的逐任务块输出；正式汇总会记录其 SHA-256，但 raw 文件不提交。
- 只有在题目网格完整、manifest 一致、输入哈希通过且后处理成功后，汇总才可复制到上述正式目录并使用
  `validated` 后缀。
- `validated` 表示产物完整性已检查，不表示统计结论可以外推到其他模型、硬件或样本规模。
