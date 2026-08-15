# 文档导航

文档按用途分为方法、实验协议、正式报告和工程验证四类。建议先看正式报告，再按需要进入协议或实现
细节；`validation/` 中的记录只证明代码路径和硬件行为可运行，不用于替代主实验结论。

## 推荐阅读顺序

1. [GSM8K 方法质量与计算量实验](reports/GSM8K_3090_ALIGNED_RESULTS.md)：实验设置、准确率、pass@k、
   共享奖励目标与质量消融。
2. [RTX 3090 推理基础设施优化汇总](reports/RTX3090_ROLLOUT_INFRA.md)：连续批处理、rollout/replay
   复用、流式 IS 和 MH 执行优化的墙钟、FLOPs 与适用条件。
3. [推理算法实现](methods/ALGORITHMS.md)：统一说明所有已实现算法的目标分布、估计量、数学性质、
   数据生命周期和关键代码。
4. [推理基础设施实现](methods/INFRASTRUCTURE.md)：统一说明调度、KV、评分、speculation、异步执行、
   Transformers/vLLM 后端和计算量分母。
5. [GSM8K 统一实验设计](experiments/GSM8K_EXPERIMENT_DESIGN.md)：数据版本、公平性约束、计算量
   口径、复现命令和消融矩阵。
6. [算法映射](methods/ALGORITHM_MAP.md)、[rollout 专题](methods/ROLLOUT_ACCELERATION.md)和
   [vLLM 专题](methods/VLLM_RUNTIME.md)：分别补充论文标签、rollout 数据流与运行时安装配置。

## 正式报告

| 文档 | 用途 |
| --- | --- |
| [GSM8K 方法质量与计算量实验](reports/GSM8K_3090_ALIGNED_RESULTS.md) | 32 题主比较、共享目标、pass@k、off-policy/replay 质量与消融 |
| [RTX 3090 推理基础设施优化汇总](reports/RTX3090_ROLLOUT_INFRA.md) | 术语定义、论文依据、成对墙钟/FLOPs、复用和冷启动成本 |

正式报告引用的机器可读 JSON 位于 [`results/gsm8k_3090/`](../results/gsm8k_3090/) 与
[`results/infra/`](../results/infra/)，图表位于 [`docs/assets/`](assets/)。

## 方法与实现

| 文档 | 用途 |
| --- | --- |
| [推理算法实现](methods/ALGORITHMS.md) | 所有算法的统一入口；含数学细节、关键代码、精确性与近似来源 |
| [推理基础设施实现](methods/INFRASTRUCTURE.md) | 所有 infra 机制的统一入口；含收益来源、额外成本、后端与计量 |
| [算法映射](methods/ALGORITHM_MAP.md) | 论文标签与核心代码标识的简表 |
| [推理性能设计](methods/PERFORMANCE_DESIGN.md) | 批处理、KV、评分与账本的设计补充 |
| [rollout 生成与复用](methods/ROLLOUT_ACCELERATION.md) | rollout、验证、MH 和 SMC 数据流的专题补充 |
| [vLLM 推理运行时](methods/VLLM_RUNTIME.md) | vLLM 安装、配置、评分 fallback 与成对测速说明 |

## 实验协议

| 文档 | 用途 |
| --- | --- |
| [GSM8K 统一实验设计](experiments/GSM8K_EXPERIMENT_DESIGN.md) | 固定模型、数据、指标、预算、运行顺序与结果生成方式 |

## 工程验证

| 文档 | 范围 | 不应用于 |
| --- | --- | --- |
| [GSM8K quick 集成检查](validation/GSM8K_QUICK_VALIDATION.md) | 8 题端到端路径及 32 题批处理 smoke | 最终方法排序 |
| [RTX 3090 复现记录](validation/RTX3090_REPRODUCTION.md) | 0.5B 模型的 CUDA、概率评分、KV 与 replay 行为 | benchmark 级准确率结论 |

旧的 quick 结果和硬件诊断已归入 [`results/validation/`](../results/validation/)。训练摘要单独位于
[`results/training/`](../results/training/)，不会与推理主表混合。
