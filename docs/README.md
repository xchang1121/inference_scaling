# 文档导航

文档按用途分为方法、实验协议、正式报告和工程验证四类。建议先看正式报告，再按需要进入协议或实现
细节；`validation/` 中的记录只证明代码路径和硬件行为可运行，不用于替代主实验结论。

## 推荐阅读顺序

1. [GSM8K 单卡对齐实验](reports/GSM8K_3090_ALIGNED_RESULTS.md)：完整 setting、比较目标、数据、图表与
   结论。
2. [GSM8K 统一实验设计](experiments/GSM8K_EXPERIMENT_DESIGN.md)：数据版本、公平性约束、计算量
   口径、复现命令和消融矩阵。
3. [算法映射](methods/ALGORITHM_MAP.md)：数学对象、实现标识与必须保持的概率性质。
4. [推理性能设计](methods/PERFORMANCE_DESIGN.md)：连续批处理、KV 复用、评分缓存和 token/FLOPs
   计量。
5. [rollout 生成与复用](methods/ROLLOUT_ACCELERATION.md)：token tree、负载感知草稿、渐进预算、
   流式奖励与 SMC forest。
6. [vLLM 推理运行时](methods/VLLM_RUNTIME.md)：异步调度、概率精确性边界、配置与成对 benchmark。

## 正式报告

| 文档 | 用途 |
| --- | --- |
| [GSM8K 单卡对齐实验](reports/GSM8K_3090_ALIGNED_RESULTS.md) | 32 题主比较、共享目标、pass@k、replay、批处理和消融 |
| [rollout 基础设施消融](reports/RTX3090_ROLLOUT_INFRA.md) | 3090 上五层优化的三随机种子墙钟、FLOPs、复用与适用边界 |

正式报告引用的机器可读 JSON 统一位于 [`results/gsm8k_3090/`](../results/gsm8k_3090/)，图表位于
[`docs/assets/`](assets/)。

## 方法与实现

| 文档 | 用途 |
| --- | --- |
| [算法映射](methods/ALGORITHM_MAP.md) | 将 MH、条件 IS、base replay 和 dynamic IS 对应到代码入口 |
| [推理性能设计](methods/PERFORMANCE_DESIGN.md) | 说明哪些优化保持算法不变，以及每种加速的分母 |
| [rollout 生成与复用](methods/ROLLOUT_ACCELERATION.md) | 说明五层优化、两套后端实现和渐近正确性边界 |
| [vLLM 推理运行时](methods/VLLM_RUNTIME.md) | 说明 vLLM 的安装、调度、精确评分 fallback 与公平测速方式 |

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
