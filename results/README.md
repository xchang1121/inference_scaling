# 结果索引

此目录只提交体积较小、可由脚本读取的汇总结果。逐题 JSONL、pass@k raw chunks、模型权重、checkpoint
和运行日志属于可恢复的中间产物，由 `.gitignore` 排除。

## GSM8K 方法效果结果

目录：[`gsm8k_3090/`](gsm8k_3090/)

机器可读结果使用的组合标签逐项定义在
[报告中的组合名称](../docs/methods/ALGORITHMS.md#alg-report-labels)。

| 结果族 | 方法定义 |
| --- | --- |
| Base、搜索、GRPO | [生成与训练基线](../docs/methods/ALGORITHMS.md#alg-baselines) |
| 幂分布与 verifier MH | [幂分布 MH](../docs/methods/ALGORITHMS.md#alg-power-mh)、[奖励目标 MH](../docs/methods/ALGORITHMS.md#alg-reward-mh) |
| 标准与 off-policy 条件 IS | [条件能量 IS](../docs/methods/ALGORITHMS.md#alg-conditional-is)、[主模型重评分](../docs/methods/ALGORITHMS.md#alg-offpolicy-is) |
| 无重评分小模型补全 | [proposal-energy 目标](../docs/methods/ALGORITHMS.md#alg-proposal-energy) |
| warm replay、动态候选和预算分配 | [rollout replay](../docs/methods/ALGORITHMS.md#alg-base-replay)、[动态候选](../docs/methods/ALGORITHMS.md#alg-dynamic-is)、[方差—成本分配](../docs/methods/ALGORITHMS.md#alg-budget-allocation) |

| 文件 | 内容 |
| --- | --- |
| `gsm8k_3090_aligned_comparison_validated.json` | Base、搜索、MH、条件 IS 与 GRPO 的 pass@1、成本背景和配对区间 |
| `gsm8k_3090_aligned_distribution_audit_validated.json` | 共享目标下的答案分布 TV/JS 诊断 |
| `gsm8k_3090_aligned_passk_validated.json` | Base、MH 与 GRPO 的独立 draw 汇总 |
| `gsm8k_3090_aligned_is_passk_validated.json` | 标准、截断 off-policy 与非截断 off-policy IS 的独立 draw 汇总 |
| `gsm8k_3090_aligned_is_uncorrected_validated.json` | 0.5B rollout 无重评分消融的独立 draw、分模型 token 与 FLOPs 汇总 |
| `gsm8k_3090_aligned_is_rescoring_ablation_validated.json` | 无重评分与三种 IS 的题目级配对区间及成本比 |
| `gsm8k_3090_aligned_verifier_rescoring_ablation_validated.json` | 精确 verifier 奖励下，0.5B 补全有无 1.5B 重评分的 32 题配对质量、分模型 FLOPs 与受控墙钟比较 |
| `gsm8k_3090_aligned_passk_comparison_validated.json` | 六种方法的统一 pass@k、配对区间与权重诊断 |
| `gsm8k_3090_aligned_ablations_validated.json` | 候选、rollout、引导阶段、奖励、温度和长度消融 |

这些文件对应的可读报告是
[`GSM8K 方法效果与准确率`](../docs/reports/GSM8K_3090_ALIGNED_RESULTS.md)。

## 推理基础设施优化结果

实验臂名称中的执行机制、workload 后缀与成本口径见
[报告中的实验臂名称](../docs/methods/INFRASTRUCTURE.md#infra-report-labels)。

早期完整 GSM8K 网格中可分离的 infra 汇总仍位于 [`gsm8k_3090/`](gsm8k_3090/)：

| 结果族 | 方法定义与计量分母 |
| --- | --- |
| 连续批处理与重复前缀 | [连续批处理](../docs/methods/INFRASTRUCTURE.md#infra-continuous-batching)、[KV 复用](../docs/methods/INFRASTRUCTURE.md#infra-prefix-kv) |
| warm replay 与动态执行 | [replay 在线/冷启动成本](../docs/methods/INFRASTRUCTURE.md#infra-replay-execution)、[rollout 展平](../docs/methods/INFRASTRUCTURE.md#infra-flattening) |
| 部分 rollout 与流式 verifier | [rollout broker](../docs/methods/INFRASTRUCTURE.md#infra-rollout-broker)、[流式奖励](../docs/methods/INFRASTRUCTURE.md#infra-streaming-reward) |
| 历史草稿与负载门控 | [精确 speculation](../docs/methods/INFRASTRUCTURE.md#infra-speculation)、[active-batch 调度](../docs/methods/INFRASTRUCTURE.md#infra-active-batch) |
| MH 执行复用 | [proposal-tree 预取](../docs/methods/INFRASTRUCTURE.md#infra-mh-prefetch)、[精确奖励削减](../docs/methods/INFRASTRUCTURE.md#infra-delayed-reward)、[replay 执行成本](../docs/methods/INFRASTRUCTURE.md#infra-replay-execution) |
| progressive、run-ahead 与 SMC | [progressive IS](../docs/methods/ALGORITHMS.md#alg-progressive-is)、[run-ahead](../docs/methods/INFRASTRUCTURE.md#infra-runahead)、[SMC 复用](../docs/methods/INFRASTRUCTURE.md#infra-smc-reuse) |

| 文件 | Infra 内容 |
| --- | --- |
| `gsm8k_3090_aligned_compute_validated.json` | GRPO 训练成本、推理 FLOPs 和累计成本交点 |
| `gsm8k_3090_aligned_replay_validated.json` | fresh-only、warm replay、cache build 与回本次数 |
| `gsm8k_3090_aligned_dynamic_is_validated.json` | 动态候选、缓存、design 阶段、复用率与稳态/一次性成本 |
| `gsm8k_3090_aligned_async_grouped_validated.json` | 各方法逐 prompt/连续批处理墙钟、FLOPs 与输出一致性 |

新增 rollout 加速栈的结果目录为 [`infra/`](infra/)。

三份 `rtx3090_transformers_decode_*.json` 和三份
`rtx3090_transformers_algorithms_*.json` 是 RTX 3090 上的独立随机种子原始报告；
`rtx3090_transformers_summary.json` 保存均值、样本标准差和成对因子。它们只比较基础设施成本，不用
单题 reward 对方法质量排序。对应文字与图表见
[`RTX 3090 推理基础设施优化汇总`](../docs/reports/RTX3090_ROLLOUT_INFRA.md)。

三份 `rtx3090_transformers_is_mh_seed*.json` 记录部分 rollout broker、流式 frozen-design IS、随机
历史草稿、MH proposal-tree 预取、delayed acceptance 和 replay 混合 proposal；
`rtx3090_transformers_is_mh_summary.json` 保存经过 workload 完整性检查的三 seed 聚合。受控 0.2 s
verifier 只用于验证延迟隐藏与早拒绝，不是方法质量结果。

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
