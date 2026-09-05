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
| [native_update_graph.py](scripts/native_update_graph.py) | 固定地址的完整更新 CUDA graph、可捕获 Adam、有界反馈缓冲区 |
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
- 默认全词表 KL、学习率 .001、每 16 cycles 用最后 4 轮反馈更新一次；不是重新训练全部原始 Uno adapter。
- 复用实际 draft logits，直接算冻结 LM head 的梯度，不再为训练多做一次词表投影。
- 默认把梯度、反传与 Adam 一起用 CUDA graph 重放；eager 更新仅保留作对照。
- 没有跨请求训练或树解码改动，不声称已证明在线学习一定提高净 TPS。
- 数学上的分布保持以固定块长 Uno 正确为前提；BF16 不同执行形状的逐 token 一致性仍需单独检验。

先读 [运行说明](docs/RUNNING.md)，再按需查看 [算法和证明](docs/ALGORITHM.md)。
新运行的输出放在被 Git 忽略的 `results/` 下，由调用者显式指定文件名；不会再把阶段结果和归档堆入源码目录。

## 简短进展

- 2026-09-05：末层在线 LoRA、融合 norm、解析 head 梯度、同版本微批与完整更新图已实现，WSL 全部 46 项测试通过。
  系统优化开发测试（4 prompts × 3 次 × 512 tokens）：原生静态 B8 **225.47 TPS**，融合后 **258.14 TPS**。
  两配置分进程测量，受时间/负载漂移影响，不把 14.49% 当作确认性收益。
- 当前 CUDA-graph 在线配置在预先固定的 12 个新 prompts × 2 次 × 1,024 tokens 上：
  静态 **286.40 TPS / TPF 1.5480**，在线 **287.31 TPS / TPF 1.5587**。
  净 TPS **+0.32%**，prompt-cluster bootstrap 95% 区间约 **[−0.77%, +1.42%]**，
  尚不能宣布稳定额外收益。24/24 次与同引擎静态 B8 的 1,024 tokens 相同；477 次真实更新，
  teacher / 原 Uno 字节 hash 不变、无抢占或 backbone graph miss。
  更新平均约 **1.00 ms**（原 eager 开发检查约 3.3 ms）；完整 reset/训练/同步均计入 TPS。
  新输入见 `config/evaluation_prompts.json`，是独立于四道开发题的人工测试，不冒充公开 benchmark。
- 已弃用的开发配置：`cbb49c8` 的 R1/S8/lr=.003，在线 −4.46%；同模块 R4/S8/lr=.001，
  完整 head 重放和解析梯度两次检查均未回本。只保留此处结论与可重跑的参数，不保留旧分支或中间结果档案。
- `fa4d7c1` 的 eager R4/S16 更新在四道开发题上 −2.38%；本轮 profile 确认一次更新有 109 次
  kernel launch，GPU 算子约 0.51 ms，推动了完整更新图的实现。Profiler 总耗时不当作普通延迟。
