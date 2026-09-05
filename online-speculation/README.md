# Online Uno

当前实现是在官方 Uno 原生推理引擎外增加一个**请求内在线块长控制器**：根据刚完成的
draft–verify cycle 的实际提交 token 数和耗时，更新各块长的收益/成本估计，再选择下一轮块长。

在线更新的是策略统计量，不是 diffusion LoRA 权重。base 模型、Uno adapter、proposal 采样、
验证和 KV 回滚继续使用锁定的官方实现。版本管理内容不保留历史实验结果、阶段日志、旧原型或未启用补丁。

## 代码与说明

| 文件 | 职责 |
| --- | --- |
| [native_online_policy.py](scripts/native_online_policy.py) | 全部在线改动：策略状态、EMA 更新、块长选择、原生引擎 wrapper |
| [benchmark_native_uno.py](scripts/benchmark_native_uno.py) | 同引擎比较 AR、固定块长 Uno、在线 Uno 和固定宽度 shadow |
| [analyze_native_uno.py](scripts/analyze_native_uno.py) | 完整矩阵与计时审计、TPF/TPS 计算、输出差异检查 |
| [ALGORITHM.md](docs/ALGORITHM.md) | 相对 Uno 的改动、更新公式、分布保持证明、开销与适用边界 |
| [RUNNING.md](docs/RUNNING.md) | 当前运行环境、运行命令、API 用法与测试方法 |
| [tests/](tests/) | 不依赖历史实验文件的控制器与审计单元测试 |
| [bootstrap_uno_runtime.sh](scripts/bootstrap_uno_runtime.sh) / [wsl_runtime_smoke.py](scripts/wsl_runtime_smoke.py) | 可重复的依赖安装入口与运行时自检 |
| [config/](config/) / [upstream.lock.json](references/upstream.lock.json) | 依赖下载校验与官方源码、模型版本锁定 |

## 当前范围

- RTX 3090，WSL2 / Ubuntu 22.04，单 GPU、batch=1，Uno 1B，BF16 / FlashAttention-2 / CUDA graphs。
- 线性 Uno，候选块长 `{4, 8, 16}`，默认锚点 `8`；每个请求重新初始化学习状态。
- 只在完整验证并提交后更新策略；轮内不修改 proposal、模型参数、attention mask 或 KV 管理。
- 没有在线 LoRA SGD、跨请求训练、树解码改动，也不声称已证明在线策略一定提高净 TPS。
- 数学上的分布保持以固定块长 Uno 正确为前提；BF16 不同执行形状的逐 token 一致性仍需单独检验。

先读 [运行说明](docs/RUNNING.md)，再按需查看 [算法和证明](docs/ALGORITHM.md)。
新运行的输出放在被 Git 忽略的 `results/` 下，由调用者显式指定文件名；不会再把阶段结果和归档堆入源码目录。
