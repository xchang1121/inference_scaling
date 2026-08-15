# 文档导航

## 方法

[算法基础、原理与实现](methods/ALGORITHMS.md)是唯一的方法说明文档，依次覆盖统一记号、目标分布、MH、IS、
off-policy 修正、rollout replay、动态候选、SMC、异步执行、KV/评分复用、vLLM 和计算量口径。每个算法的
原理、步骤、关键代码与误差来源位于同一节。

## 实验

| 文档 | 内容 |
| --- | --- |
| [GSM8K 实验设计](experiments/GSM8K_EXPERIMENT_DESIGN.md) | 数据、模型、预算、指标、成本分母、命令和产物 |
| [方法质量与计算量](reports/GSM8K_3090_ALIGNED_RESULTS.md) | 准确率、pass@k、共享奖励、off-policy、replay 与消融 |
| [推理执行与 rollout 复用](reports/RTX3090_ROLLOUT_INFRA.md) | 墙钟、FLOPs、吞吐、缓存成本和复用率 |

## 验证与结果

| 文档 | 内容 | 统计范围 |
| --- | --- | --- |
| [GSM8K 集成检查](validation/GSM8K_QUICK_VALIDATION.md) | 8 题端到端路径和 32 题批处理检查 | 工程验证 |
| [RTX 3090 复现记录](validation/RTX3090_REPRODUCTION.md) | CUDA、概率评分、KV、MH、IS 与 replay 检查 | 工程验证 |
| [机器可读结果](../results/README.md) | 正式汇总、训练摘要和验证产物 | 实验数据 |

图表位于 [`assets/`](assets/)。
