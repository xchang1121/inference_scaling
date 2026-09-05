# Online Uno

当前主线是**真正更新 draft 参数的 Online Uno**，同时优化原生推理。
冻结 base 和原 Uno adapter，在最后一层 MLP 加 rank-8 LoRA，仅作用于 draft noise 行；
复用 verifier 分布和 draft 特征，在 commit 后完成 backward、Adam 和下一轮参数发布。
另有独立的 Triton grouped RMSNorm 融合优化。原在线块长控制器只保留作对照。

目标：在 RTX 3090 / WSL2 上改进计入全部在线开销的净 TPS，检查 teacher/KV 隔离和行为接近，
明确区分系统优化收益与在线学习增益。只维护当前实现、算法证明、运行说明和本页简短进展。
失败尝试只记出处/结论后移除；不建立阶段文档、失败代码或结果档案。

## 代码与说明

| 文件 | 职责 |
| --- | --- |
| [native_fast_weights.py](scripts/native_fast_weights.py) | 新增 LoRA、特征重放、teacher 对齐、梯度更新、轮间发布、请求重置 |
| [native_norm.py](scripts/native_norm.py) | 保留 BF16 舍入位置的融合 grouped RMSNorm |
| [native_online_policy.py](scripts/native_online_policy.py) | 独立块长控制器对照，不训练模型 |
| [benchmark_native_uno.py](scripts/benchmark_native_uno.py) | 同引擎比较 AR、固定块长 Uno、在线 Uno 和固定宽度 shadow |
| [analyze_native_uno.py](scripts/analyze_native_uno.py) | 完整矩阵与计时审计、TPF/TPS 计算、输出差异检查 |
| [ALGORITHM.md](docs/ALGORITHM.md) | 相对 Uno 的改动、更新公式、分布保持证明、开销与适用边界 |
| [RUNNING.md](docs/RUNNING.md) | 当前运行环境、运行命令、API 用法与测试方法 |
| [tests/](tests/) | 不依赖历史实验文件的控制器与审计单元测试 |
| [bootstrap_uno_runtime.sh](scripts/bootstrap_uno_runtime.sh) / [wsl_runtime_smoke.py](scripts/wsl_runtime_smoke.py) | 可重复的依赖安装入口与运行时自检 |
| [config/](config/) / [upstream.lock.json](references/upstream.lock.json) | 依赖下载校验与官方源码、模型版本锁定 |

## 当前范围

- RTX 3090，WSL2 / Ubuntu 22.04，单 GPU、batch=1，Uno 1B，BF16 / FlashAttention-2 / CUDA graphs。
- 线性 Uno，在线 LoRA 首版固定 `B=8`；每个请求重置参数和 optimizer。
- 只在完整验证并提交后更新 53,248 个新增参数；轮内不修改生成时的 proposal 分布。
- 默认全词表 KL、学习率 .003、每 8 cycles 更新一次；不是重新训练全部原始 Uno adapter。
- 没有跨请求训练或树解码改动，不声称已证明在线学习一定提高净 TPS。
- 数学上的分布保持以固定块长 Uno 正确为前提；BF16 不同执行形状的逐 token 一致性仍需单独检验。

先读 [运行说明](docs/RUNNING.md)，再按需查看 [算法和证明](docs/ALGORITHM.md)。
新运行的输出放在被 Git 忽略的 `results/` 下，由调用者显式指定文件名；不会再把阶段结果和归档堆入源码目录。

## 简短进展

- 2026-09-05：末层在线 LoRA 已跑通，WSL 全部 35 项测试通过。4 个开发 prompts × 3 次 ×
  512 tokens：原生静态 B8 **225.47 TPS**；融合后无新增分支 **258.14 TPS**；
  含零增量分支的静态对照 **255.59 TPS**；在线更新 **244.20 TPS**（相对配对静态 −4.46%）。
  系统优化与原生基线分进程测量，受时间/负载漂移影响，不把 14.49% 当作确认性收益。
  在线 12/12 与同引擎静态 B8 的 512 个 tokens 一致，272 次真实更新，冻结权重 hash 不变。
  在线 TPF 1.363 < 静态 1.379，说明当前 .003 / stride=8 的最新一轮 KL 更新尚未回本；
  这是可运行的在线算法，**不是已获得稳定在线额外加速的结果**。后续优化复用这些模块，不新增阶段档案。
