# 当前研究路线图

2026-09-05 重新规划。上一轮研究的完整路线图与记录移至
[归档](../archive/2026-09-05-v1/README.md)。

| 阶段 | 交付 | 判定 |
| --- | --- | --- |
| R1 | 归档旧记录；Recycling Uno 推导、文献归因、实验协议 | 先提交设计，再读取新实验结果 |
| R2 | 一遍 tail-recycling verifier、在线 TPS controller、KV/采样测试 | 分布正确；greedy 数学 oracle 一致；旧路径回归 |
| R3 | 真实 Uno 0.9B pilot，冻结参数，独立多域 held-out | 包括全部在线成本后的 E2E/decode TPS，成对报告 |
| R4 | WSL2 + cu128/FA2/Triton + pinned Nano-vLLM | GPU kernel smoke；同 runtime AR/Uno 基线 |
| R5 | 将胜出设计接入官方运行时，固定图形状/减少同步 | 相对最优静态宽度的 held-out TPS CI 和绝对 TPS |

R2/R3 在 Windows 可用环境进行；R4 并行准备系统依赖。重启若为必要外部条件，
记录明确状态并继续不依赖重启的算法工作。

优化次序：省 forward → 降低每轮同步/采样成本 → 实测选择 refill 宽度与 recycling horizon →
官方 CUDA graphs/FA2 → 才考虑新梯度更新。没有收益的候选进入归档，保留可复核证据。
