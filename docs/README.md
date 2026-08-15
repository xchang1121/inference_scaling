# 文档导航

## 报告

| 文档 | 内容 |
| --- | --- |
| [GSM8K 方法质量与计算量实验](reports/GSM8K_3090_ALIGNED_RESULTS.md) | 准确率、pass@k、共享奖励、off-policy、replay 与消融 |
| [RTX 3090 推理执行与 rollout 复用实验](reports/RTX3090_ROLLOUT_INFRA.md) | 墙钟、FLOPs、吞吐、缓存成本与复用率 |

## 方法与实现

| 文档 | 内容 |
| --- | --- |
| [推理算法实现](methods/ALGORITHMS.md) | 目标分布、估计量、收敛性质、replay 生命周期与代码 |
| [推理基础设施实现](methods/INFRASTRUCTURE.md) | 调度、KV、评分、speculation、异步执行、后端与计量 |
| [算法映射](methods/ALGORITHM_MAP.md) | 论文标签、实现标识与核心不变量 |
| [推理性能设计](methods/PERFORMANCE_DESIGN.md) | 批处理、评分、缓存与计算账本的设计摘要 |
| [rollout 生成与复用](methods/ROLLOUT_ACCELERATION.md) | rollout、MH 与 SMC 的执行数据流 |
| [vLLM 推理运行时](methods/VLLM_RUNTIME.md) | 安装、配置、原生能力、评分委托与成对测速 |

## 实验与验证

| 文档 | 内容 | 统计范围 |
| --- | --- | --- |
| [GSM8K 统一实验设计](experiments/GSM8K_EXPERIMENT_DESIGN.md) | 数据、模型、预算、指标、命令与产物 | 正式实验协议 |
| [GSM8K quick 集成检查](validation/GSM8K_QUICK_VALIDATION.md) | 8 题端到端路径与 32 题批处理检查 | 工程验证 |
| [RTX 3090 复现记录](validation/RTX3090_REPRODUCTION.md) | CUDA、概率评分、KV 与 replay 检查 | 工程验证 |

机器可读结果见[结果索引](../results/README.md)。图表位于 [`assets/`](assets/)。
