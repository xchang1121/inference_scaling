# 文档索引

| 文档 | 唯一职责 |
| --- | --- |
| [算法基础、原理与实现](methods/ALGORITHMS.md) | 默认 Qwen MH/IS 完整流程、目标分布、参数、直观收敛说明、公共接口与模型适配；含尚未实现的 logit-adjustment 理论参考 |
| [GSM8K 实验设计](experiments/GSM8K_EXPERIMENT_DESIGN.md) | 方法标签、模型、预算、指标、成本比较基准与复现入口 |
| [方法质量与计算量](reports/GSM8K_3090_ALIGNED_RESULTS.md) | 准确率、pass@$`k`$、off-policy、replay 与消融结果 |
| [推理执行与 rollout 复用](reports/RTX3090_ROLLOUT_INFRA.md) | 墙钟、FLOPs、吞吐、缓存与复用率结果 |
| [Qwen2.5-1.5B 优化研究](reports/QWEN15B_OPTIMIZATION_STUDY.md) | 新增算法与执行候选的筛选协议、消融结果和默认组合决定 |
| [GSM8K 集成检查](validation/GSM8K_QUICK_VALIDATION.md) | 小规模端到端工程检查 |
| [RTX 3090 复现记录](validation/RTX3090_REPRODUCTION.md) | CUDA、概率评分与后端工程检查 |
| [AR-LLM 完整流程真机验证](validation/ARLLM_FULL_ROUTE.md) | GRPO、全部推理组件、执行优化路径与修复后回归 |
| [机器可读结果](../results/README.md) | 已纳入版本控制的汇总文件索引 |

图表位于 [`assets/`](assets/)；原理只在算法文档定义，实验设置只在实验设计定义，报告只记录结果与解读。
