# 当前研究路线图

2026-09-05 重新规划。上一轮研究的完整路线图与记录移至
[归档](../archive/2026-09-05-v1/README.md)。

| 阶段 | 交付 | 判定 |
| --- | --- | --- |
| R1 | 归档旧记录；Recycling Uno 推导、文献归因、实验协议 | 先提交设计，再读取新实验结果 |
| R2 | 一遍 tail-recycling verifier、在线 TPS controller、KV/采样测试 | 分布正确；greedy 数学 oracle 一致；旧路径回归 |
| R3A/B | Recycling / warm-start pilot | 已结束：没有可靠净收益，负结果保留 |
| R3C/D | Packed tree + rank/cost budget 在线消融 | 已实现；HighQoS pilot 对线性约 +18%，对固定树仅约 +1.84% |
| R3E | 12 新 prompts × 5 repeats 独立 held-out | 冻结参数；分别比较 AR、线性与固定树，不将树收益等同在线收益 |
| R4 | WSL2 + cu128/FA2/Triton + pinned Nano-vLLM | WSL 本体/组件已装，等待重启；Linux kernel smoke 和官方基线未完成 |
| R5 | FA2 树路径候选、固定 CUDA graph 形状与减少同步 | 补丁仅默认关闭、apply-check 通过；必须先跑官方未修改基线 |
| R6 | 嵌套树 counterfactual feedback | 数学与 CPU 工具已准备，尚未参与 GPU 控制；需独立预注册实验 |

R2/R3 在 Windows 可用环境进行；R4 并行准备系统依赖。重启若为必要外部条件，
记录明确状态并继续不依赖重启的算法工作。

当前优化次序：低预算树覆盖 → 实测在线预算与静态树比较 → 官方 CUDA graphs/FA2 →
低开销同轮 counterfactual reward → 有明确回本空间后才考虑梯度更新。
没有收益的候选退出默认主线，保留可复核证据。
