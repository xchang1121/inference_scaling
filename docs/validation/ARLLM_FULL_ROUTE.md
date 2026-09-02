# AR-LLM 完整流程真机验证

## 验证结论

2026 年 8 月 16 日在 NVIDIA GeForce RTX 3090 24 GB 上完成 AR-LLM 低成本功能检查（`smoke`）。公开 GSM8K
数据准备、一次真实 GRPO 更新、LoRA 保存与重新加载、11 种推理方法、rollout replay、动态候选、异步
批处理、两组 pass@$`k`$、消融、预算曲线、长度实验、分布诊断和两组执行优化基准均正常结束。训练与推理
总清单的 `status` 均为 `complete`。

该配置用于验证实现、调度、模型加载、概率评分、计算量记录和结果文件恢复。质量部分只取 1 道测试题，
pass@$`k`$ 只取 2 次独立重复，因此本页不据此比较方法准确率。正式质量结论仍以
[GSM8K 方法质量与计算量](../reports/GSM8K_3090_ALIGNED_RESULTS.md)为准。

## 环境

| 项目 | 实际值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090，24,576 MiB |
| 驱动 | 596.49 |
| 系统 | Windows 11 |
| Python | 3.12.7 |
| PyTorch / CUDA 运行时 | 2.13.0+cu130 / 13.0 |
| Transformers | 5.15.0 |
| Accelerate / PEFT / TRL | 1.14.0 / 0.20.0 / 1.9.2 |
| NumPy | 1.26.4 |
| 主模型 | Qwen2.5-1.5B-Instruct，模型版本（revision）`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` |
| 辅助 proposal 模型 | Qwen2.5-0.5B-Instruct |

AR 环境使用 Conda Python，并设置 `PYTHONNOUSERSITE=1`，使编译型科学计算包统一来自同一环境。
`pip check` 与 CUDA FP16 矩阵乘法均通过。

## 执行入口

训练阶段：

```powershell
$env:PYTHONNOUSERSITE = "1"
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_arllm_suite `
  --stage train --profile smoke --tag ar-real-20260816 `
  --summary-root results\validation\arllm-real-20260816 `
  --training-output models\Qwen2.5-1.5B-Instruct-GRPO-GSM8K-smoke-ar-real-20260816 `
  --restart
```

推理阶段：

```powershell
$env:PYTHONNOUSERSITE = "1"
C:\Users\singm\anaconda3\python.exe -m experiments.arllm.run_arllm_suite `
  --stage inference --profile smoke --tag ar-real-inference-20260816 `
  --summary-root results\validation\arllm-real-20260816 `
  --training-output models\Qwen2.5-1.5B-Instruct-GRPO-GSM8K-smoke-ar-real-20260816 `
  --backend transformers `
  --components quality matched_target replay dynamic_is async passk ablations `
               budget_curve length_ablation distribution infra
```

推理入口的续跑清单记录 4 个完成的子命令。第一次执行发现 LoRA 适配器参数被传给 replay 和 dynamic IS
脚本；修正参数作用域后，以同一运行标签续跑，已完成的质量方法直接复用原结果文件。

## GRPO 训练路径

| 指标 | 结果 |
| --- | ---: |
| 优化器更新 | 1 |
| 训练样本 | 4 |
| 生成补全 | 8 |
| 生成补全 token | 768 |
| 训练墙钟 | 18.751 s |
| 估计总计算量 | 0.026751 PFLOPs |
| CUDA 峰值已分配显存 | 3,766,979,584 bytes |
| `nvidia-smi` 峰值显存 | 5,872 MiB |

LoRA 适配器已写入模型检查点，并由 `rl_sample`、`rl_greedy`、pass@$`k`$ 和分布诊断重新加载。
适配器权重 SHA-256 为
`699482a16f3592f463d216cfb497c7a78a58a86070b501beb164515f6725e362`。

## 推理覆盖

| 组件 | 真机执行内容 | 结果 |
| --- | --- | --- |
| 质量方法 | Base、Beam、Best-of-N、MH、标准条件 IS、小 proposal 条件 IS、三种精确奖励方法、GRPO 随机与贪心生成 | 11 种方法均生成非空序列并写入计量 |
| 共享目标 | verifier-MH、verifier-IS、0.5B proposal verifier-IS | 完成 |
| rollout replay | 新生成、缓存构建、已有历史在线执行、概率重评分与复现检查 | 候选批次全部复现，rollout 复用率 42.86% |
| 动态 IS | 基础模型候选、固定 replay 感知分配、方差—成本分配 | 三个分支完成；replay 候选命中率 50% |
| 异步批处理 | Base、Best-of-N、标准 IS、小 proposal IS 的同步/异步成对运行 | 4 组输出逐 token 一致 |
| 通用 pass@$`k`$ | Base、MH、GRPO 随机采样 | 3 种方法、每题 2 次独立重复完成 |
| IS pass@$`k`$ | 标准 IS、截断 off-policy、未截断 off-policy、无主模型重评分 | 4 种方法、每题 2 次独立重复完成 |
| 消融 | MH 更新数与幂指数、候选数、rollout 数、候选选择阶段数、奖励来源、温度 | 所有 `smoke` 分支完成 |
| 预算与长度 | Beam、Best-of-N、两类 IS 的预算点；32-token 短输出路径 | 完成 |
| 分布诊断 | Base、verifier-MH、两类 verifier-IS、GRPO，共 10 次生成 | 完成并记录模型、适配器和实现哈希 |
| rollout 执行优化 | 历史树、根据批量大小启用历史草稿、分阶段 IS、空闲时预生成、SMC | 3 个解码分支和 5 个算法分支完成 |
| IS/MH 执行优化 | 部分 rollout、流式 IS、随机草稿、MH 预取、延迟接受、历史 proposal | 17 个执行分支完成 |

## 关键工程检查

以下数值用于确认复用、异步和筛选机制确实进入执行路径。

| 检查 | 观测值 |
| --- | ---: |
| replay 纯新生成 / 已有历史在线执行的墙钟比 | 1.0167 |
| replay 纯新生成 / 已有历史在线执行的 FLOPs 比 | 1.1157 |
| replay 基础模型 / proposal 概率缓存命中率 | 50% / 50% |
| 流式 IS 首次估计更新时间：整批等待 | 3.036 s |
| 流式 IS 首次估计更新时间：完成即提交 | 0.820 s |
| 带 0.2 s verifier 延迟时的首次更新时间 | 3.324 s → 1.008 s |
| MH 候选分支预取与普通 MH 的路径校验 | 两组设置均一致 |
| 两阶段延迟接受的精确奖励调用 | 9 → 8，最终 token 哈希一致 |
| 历史混合 proposal 与基础 proposal 的平均最终奖励 | 0.75 / 0.75 |
| 历史混合 proposal 与基础 proposal 的墙钟 | 9.535 s / 35.380 s |

单次短基准的运行时间会波动，表中墙钟只用于确认机制和计量路径。稳定的加速结论见三个随机种子的
[推理执行报告](../reports/RTX3090_ROLLOUT_INFRA.md)。

## 运行中发现并修正的问题

| 问题 | 修正 | 验证 |
| --- | --- | --- |
| NumPy 版本下界与既有科学计算依赖冲突 | 支持 NumPy 1.26 及以上 | `pip check` 通过；AR 测试解释器可导入全部训练依赖 |
| replay / dynamic IS 收到无效的 `--rl-adapter` | 适配器参数只传给质量、pass@$`k`$ 和分布脚本 | 参数路由回归测试通过；统一入口续跑完成 |
| 通用 pass@$`k`$ 未写入 `summary-root` | 显式传入聚合输出路径 | 修复后重新执行 3 种方法，聚合文件位于本次验证目录 |
| 嵌套验证目录中的原始记录/任务块可能进入 Git | 使用递归忽略规则 | Git 只收录聚合 JSON 与实验套件清单 |
| Transformers 5.x `Cache.crop` 参数语义变更 | 按接口签名选择绝对长度或 token 移除语义 | 两种接口测试通过；1.5B 真实 KV 路径复跑 6 条序列且无弃用警告 |

## 回归测试与资源释放

| 测试 | 结果 |
| --- | ---: |
| 实际 Conda 解释器全仓测试 | 318 passed |
| AR 实际解释器测试 | 234 passed |
| 最终两项修复的定向测试 | 34 passed |
| dLLM 运行入口隔离测试 | 5 passed |

Ruff 全仓检查和共享层 24 个文件的 mypy 检查均通过。所有真机子进程结束后没有残留 Python 进程；GPU
占用回到桌面基线 1,862 MiB，可用显存 22,465 MiB。

## Qwen 默认路径回归（2026-08-23）

在结构隔离和默认组合接入后，使用 Qwen2.5-1.5B 对 `base`、MH、条件 IS、replay 和 async 组件执行一题
真实模型回归。Qwen2.5-0.5B 只在 replay 与小 proposal 调度检查中作为辅助模型；本轮未启动 dLLM。调度
清单确认统一入口向质量路径传入 `--mh-suffix-schedule multiscale`。

| 检查 | 结果 |
| --- | ---: |
| 完成的统一入口子命令 | 1 / 1 |
| base / MH / 条件 IS | 三条真实生成路径均完成 |
| replay 建库候选复用 | 16 个候选样本 |
| 纯新生成 / 已有历史在线阶段墙钟比 | 1.641× |
| 纯新生成 / 已有历史在线阶段 FLOPs 比 | 1.452× |
| 已有历史在线阶段 1.5B / 0.5B FLOPs | 0.009151 / 0.002806 PFLOPs |
| 缓存构建 1.5B / 0.5B FLOPs | 0.012186 / 0.005674 PFLOPs |
| replay 纯新生成/历史复用数值输出一致率 | 100% |
| async 四种方法的顺序/批处理输出 | 全部逐 token 一致 |

异步子任务只使用一个问题和一个工作线程，用于验证请求路由、随机数序列与输出一致性，不据此估计吞吐收益。
正式的三个随机种子组合结果见[优化研究](../reports/QWEN15B_OPTIMIZATION_STUDY.md#qwen15b-is-stack)。全部子进程结束
后无模型进程残留，GPU 占用为 1,218 MiB、可用显存 23,109 MiB。

## vLLM 验证边界

本机没有 WSL Linux 发行版，AR 解释器也未安装 vLLM。vLLM 后端、同步/异步适配器、请求映射、
对数概率解析、分组和错误处理由单元测试与模拟引擎覆盖；本次真实模型运行使用 Transformers。
真实 vLLM 引擎验证需要 Linux 或 WSL2 CUDA 环境，入口和参数保持一致。

## 输出文件

- [机器可读总摘要](../../results/validation/arllm-real-20260816/arllm_full_route_summary.json)
- [训练调度清单](../../results/validation/arllm-real-20260816/ar-real-20260816/arllm_suite_manifest.json)
- [推理调度清单](../../results/validation/arllm-real-20260816/ar-real-inference-20260816/arllm_suite_manifest.json)
- [通用 pass@$`k`$](../../results/validation/arllm-real-20260816/gsm8k_quick_passk_ar-real-postfix-20260816.json)
- [IS pass@$`k`$](../../results/validation/arllm-real-20260816/gsm8k_quick_is_passk_ar-real-inference-20260816.json)
- [rollout replay](../../results/validation/arllm-real-20260816/gsm8k_quick_replay_ar-real-inference-20260816.json)
- [动态 IS](../../results/validation/arllm-real-20260816/gsm8k_quick_dynamic_is_ar-real-inference-20260816.json)
- [异步批处理](../../results/validation/arllm-real-20260816/gsm8k_quick_async_grouped_ar-real-inference-20260816.json)
- [分布诊断](../../results/validation/arllm-real-20260816/arllm_distribution_ar-real-inference-20260816.json)
- [rollout infra](../../results/validation/arllm-real-20260816/arllm_rollout_infra_ar-real-inference-20260816.json)
- [IS/MH infra](../../results/validation/arllm-real-20260816/arllm_is_mh_infra_ar-real-inference-20260816.json)
- [Transformers 5.x KV 修复后复跑](../../results/validation/arllm-real-20260816/cache_api_postfix.json)
- [默认 replay 回归](../../results/validation/qwen-positive-default-smoke-20260823/gsm8k_quick_replay_qwen-positive-default-smoke-20260823.json)
- [默认异步批处理回归](../../results/validation/qwen-positive-default-smoke-20260823/gsm8k_quick_async_grouped_qwen-positive-default-smoke-20260823.json)
- [默认入口调度清单](../../results/validation/qwen-positive-default-smoke-20260823/qwen-positive-default-smoke-20260823/arllm_suite_manifest.json)
