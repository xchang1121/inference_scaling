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

- 2026-09-05：末层在线 LoRA、融合 norm、解析 head 梯度、同版本微批、完整更新图、无分支对照及顺序审计已实现，
  WSL 全部 **55 项测试**通过。
  系统优化开发测试（4 prompts × 3 次 × 512 tokens）：原生静态 B8 **225.47 TPS**，融合后 **258.14 TPS**。
  两配置分进程测量，受时间/负载漂移影响，不把 14.49% 当作确认性收益。
- 当前完整成本对照：固定 12 个 prompts × 2 次 × 1,024 tokens，seed=20270909，四组共 96 次，
  每个 prompt 的两次方法顺序互为反序。无分支静态 `plain8` **310.06 TPS / TPF 1.5480**；
  零分支控制 `8` **311.60 TPS / TPF 1.5480**；在线 `fast8` **312.86 TPS / TPF 1.5587**。
  在线 / 无分支静态净 TPS **+0.90%**，描述性 prompt-cluster bootstrap 95% 区间
  **[+0.10%, +1.68%]**。这是小幅正收益观察，不是论文级或跨环境稳定收益证明；
  零分支控制反而略快于无分支组，也说明桌面测时噪声尚不可忽略。
  24/24 次在线与两组静态 B8 的全部 1,024 tokens 相同；477 次真实更新，
  teacher / 原 Uno 字节 hash 不变、无抢占或 backbone graph miss。
  更新平均约 **0.91 ms**（原 eager 开发检查约 3.3 ms）；完整 reset/训练/同步均计入 TPS。
  输入见 `config/evaluation_prompts.json`，不重复四道开发题，但该 suite 已被多次测量，不能称为新 held-out。
  固定预算 / ignore_eos=True 也不等同于真实服务到 EOS 的时延评估。
  本轮绝对 TPS 高于此前运行，不把跨时间段差值当作新优化收益。
- `7e10bbd` 的旧 +0.32%（区间跨零）只比较了零分支静态组，不是完整新增分支成本的净收益证明；
  已由上述完整对照取代。四组扩展初次试跑因出场位次不平衡中止，修正并通过顺序审计后重测。
- 已弃用的开发配置：`cbb49c8` 的 R1/S8/lr=.003，在线 −4.46%；同模块 R4/S8/lr=.001，
  完整 head 重放和解析梯度两次检查均未回本。只保留此处结论与可重跑的参数，不保留旧分支或中间结果档案。
- `fa4d7c1` 的 eager R4/S16 更新在四道开发题上 −2.38%；本轮 profile 确认一次更新有 109 次
  kernel launch，GPU 算子约 0.51 ms，推动了完整更新图的实现。Profiler 总耗时不当作普通延迟。
- 已回滚的初始化候选（思路参考 [PiSSA](https://arxiv.org/abs/2404.02948)，并非其直接复现）：
  取冻结 Uno 末层增量的 rank-8 主右奇异子空间、C=0，其余沿用 `7e10bbd`；
  四开发题 × 3 × 1,024 tokens，seed=20270912。TPF 1.44051，随机初始化 1.44085；
  未改善接受效率，弃用候选实现及中间结果，不新增设计文档。
