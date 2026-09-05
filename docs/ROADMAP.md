# 当前研究路线图

2026-09-05 重新规划。上一轮研究的完整路线图与记录移至
[归档](../archive/2026-09-05-v1/README.md)。

| 阶段 | 交付 | 判定 |
| --- | --- | --- |
| R1 | 归档旧记录；Recycling Uno 推导、文献归因、实验协议 | 先提交设计，再读取新实验结果 |
| R2 | 一遍 tail-recycling verifier、在线 TPS controller、KV/采样测试 | 分布正确；greedy 数学 oracle 一致；旧路径回归 |
| R3A/B | Recycling / warm-start pilot | 已结束：没有可靠净收益，负结果保留 |
| R3C/D | Packed tree + rank/cost budget 在线消融 | 已实现；HighQoS pilot 对线性约 +18%，对固定树仅约 +1.84% |
| R3E | 12 新 prompts × 5 repeats 独立 held-out | 360/360 完成、token 一致；在线 49.58 TPS < fixed N=32 的 50.60，且计时频率门未过，仅作描述性测量 |
| R4 | WSL2 + cu128/FA2/Triton + pinned Nano-vLLM | Windows 已重启，Ubuntu/RTX 3090/Python 3.10 已验证；wheel 下载恢复中，kernel smoke 和官方基线未完成 |
| R5 | FA2 树路径候选、固定 CUDA graph 形状与减少同步 | 补丁仅默认关闭、apply-check 通过；必须先跑官方未修改基线 |
| R6A | 嵌套树 counterfactual feedback | 72/72 pilot 完成；正确性通过，46.85 TPS < fixed N=32 49.63，性能未过，不提升为默认 |
| R7 | 原生 Uno 请求内块长学习 | 设计/证明/实现、13 项 CPU 单测已完成；GPU pilot 协议已冻结，待运行时安装结束 |

R2/R3 在 Windows 可用环境进行；R4 并行准备系统依赖。重启若为必要外部条件，
记录明确状态并继续不依赖重启的算法工作。

当前优化次序：低预算树覆盖 → 实测在线预算与静态树比较 → 官方 CUDA graphs/FA2 →
低开销同轮 counterfactual reward → 有明确回本空间后才考虑梯度更新。
没有收益的候选退出默认主线，保留可复核证据。

R3E 的[完整结果](STAGE11_TREE_HELDOUT_RESULTS.md)不支持在线额外收益已成功。
Windows HF 当前最快的已测配置是固定 N=32 树；不能与不同 backend/dtype 的原生 Linux TPS 直接做算法归因。
优先用嵌套树反馈校准实际提交 token 收益，减少只靠 draft 概率 surrogate 的偏差，
但必须在官方 Linux 基线恢复后另外冻结协议，不重复使用 R3E 作为未见测试集。

用户最新接受与原 Uno 接近的行为/吞吐，故 R7 的第一步是同 backend 的原生线性 Uno 对照，
不强制要求打败 Windows 最强静态树。保留独立记录、更新开销和负结果，见
[R7 协议和证明](R7_NATIVE_ONLINE_DESIGN_AND_PROOFS.md)。
