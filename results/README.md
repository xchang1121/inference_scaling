# 结果索引

本目录保存可由脚本读取的汇总结果。逐题 JSONL、pass@k chunks、checkpoint 和运行日志由
`.gitignore` 管理。

## 成对复现产物

统一入口将调度清单写入 `results/reproduction/<tag>/manifest.json`。AR 原始记录沿用
`results/gsm8k/<profile>/`；dLLM 的质量、replay、动态候选、pass@$`k`$、分布、消融和执行结果写入
`results/reproduction/dllm/<tag>/components/`。大显存机器完成正式 LLaDA 运行后，可按本页现有
`validated` 规则选择需要纳入版本控制的聚合 JSON；正式结果区只收录完整运行的聚合结果。

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
| `gsm8k_3090/gsm8k_3090_aligned_replay_validated.json` | fresh、warm、cache build 与历史算术交点；默认记录仍单次消费 |
| `gsm8k_3090/gsm8k_3090_aligned_dynamic_is_validated.json` | 动态候选、design、复用率与成本 |
| `gsm8k_3090/gsm8k_3090_aligned_async_grouped_validated.json` | 逐 prompt 与连续批处理 |
| `infra/rtx3090_transformers_summary.json` | 历史树、progressive、run-ahead 与 SMC 的三 seed 聚合 |
| `infra/rtx3090_transformers_is_mh_summary.json` | broker、流式 IS、草稿和 MH 复用的三 seed 聚合 |
| `infra/rtx3090_transformers_decode_*.json` | rollout 解码实验的单 seed 记录 |
| `infra/rtx3090_transformers_algorithms_*.json` | progressive 与 SMC 的单 seed 记录 |
| `infra/rtx3090_transformers_is_mh_seed*.json` | IS/MH 复用实验的单 seed 记录 |

执行机制和成本分母见[推理扩展算法：基础、原理与实现](../docs/methods/ALGORITHMS.md)，文字结果见
[RTX 3090 推理执行与 rollout 复用实验](../docs/reports/RTX3090_ROLLOUT_INFRA.md)。

## Qwen2.5-1.5B 优化研究

[`arllm/qwen15b_optimization/attempt_registry.json`](arllm/qwen15b_optimization/attempt_registry.json)
登记新增与已有消融的状态、比较对象和决定依据。只有 `accepted` 或 `accepted_existing` 方法可以标为
`active_execution=true`；dLLM 实验在该登记表中固定关闭。文字设置和结论见
[Qwen2.5-1.5B 推理扩展优化研究](../docs/reports/QWEN15B_OPTIMIZATION_STUDY.md)。

[`arllm/qwen15b_optimization/isir_screen.json`](arllm/qwen15b_optimization/isir_screen.json) 保存 8 题、
2 draw 的 i-SIR 同预算筛选、题目聚类区间、分支成本和未进入确认阶段的决定；对应 suite manifest 与汇总
同目录保存。

| 文件 | 内容 | 决定 |
| --- | --- | --- |
| `mh_suffix_screen.json`、`mh_suffix_confirmation.json` | uniform、inverse-length 与 multiscale 后缀调度的筛选和 32 题确认 | `multiscale` 进入默认 MH 组合 |
| `mh_replay_multiscale_stack.json` | 后缀调度 × 冻结 replay proposal 的三 seed、四臂组合消融 | 墙钟默认组合；cache build 与在线成本分列 |
| `is_replay_batching_stack.json` | warm replay × 候选缓存 × 连续批处理的三 seed 组合消融；1.5B/0.5B 分账 | 匹配且未消费的 history 存在时启用完整在线栈 |
| `rqmc_screen.json` | IID 与 scrambled Sobol rollout 的成对筛选 | 默认继续使用 IID |
| `bounded_stop_screen.json` | 有界精确提前停止与完整 rollout 评估 | 默认关闭提前停止 |
| `draft_model_speculation_screen.json` | 1.5B 普通生成与 0.5B 草稿长度 2/4/8 的执行比较 | 默认使用 1.5B target-only batching |

上述文件只包含 Qwen2.5-1.5B 正式实验。0.5B 辅助模型的 forward slots 和 FLOPs 单列；未运行 dLLM
质量或性能实验。

## 训练与验证

| 路径 | 内容 | 用途 |
| --- | --- | --- |
| `training/gsm8k_grpo_training_summary.json` | 205 步 GRPO 的 token、FLOPs、显存、墙钟与 adapter 哈希 | 训练成本 |
| `validation/gsm8k_quick_*_validated.json` | 主比较、replay、批处理和计算路径 | 集成检查 |
| `validation/rtx3090_reproduction.json` | FP32 真实模型算法与后端结果 | 工程检查 |
| `validation/rtx3090_backend_bfloat16.json` | BF16 概率与显存结果 | 精度检查 |
| `validation/arllm-real-20260816/` | GRPO smoke、全部 AR 推理组件与修复后真机检查 | 全链路可用性 |
| `validation/qwen-positive-default-smoke-20260823/` | multiscale MH、候选复用 replay、1.5B/0.5B 分账与 async 输出一致性 | 生产默认路径回归 |

对应记录见 [GSM8K quick 集成检查](../docs/validation/GSM8K_QUICK_VALIDATION.md) 和
[RTX 3090 复现记录](../docs/validation/RTX3090_REPRODUCTION.md)。完整 AR 入口的训练、推理与 infra 覆盖见
[AR-LLM 全链路真机验证](../docs/validation/ARLLM_FULL_ROUTE.md)。

## 产物状态

- 原始记录位于 `results/gsm8k/<profile>/`，按 manifest fingerprint 追加并恢复。
- `*.chunks.jsonl` 保存 pass@k 任务块；正式汇总记录其 SHA-256。
- `validated` 后缀要求题目网格完整、manifest 一致、输入哈希一致且后处理成功。
- 统计适用范围由对应报告中的样本、模型和硬件设置确定。
