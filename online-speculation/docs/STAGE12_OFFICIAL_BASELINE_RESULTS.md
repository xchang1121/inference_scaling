# Stage 12：官方 Uno 原生 FA2 基线完成

2026-09-05，本机 RTX 3090 / WSL2 / Ubuntu 22.04；32/32 runs 完成，preemptions=0。
按[冻结协议](STAGE12_OFFICIAL_BASELINE_PROTOCOL.md)，未经修改的 pinned Uno，BF16、FA2、CUDA graphs，
四个旧 pilot prompts × 2 repeats × 4 方法，每请求 128 tokens，完整 generate E2E 计时。

| 方法 | 聚合 E2E TPS | 官方 decode-only TPF | 与同 seed AR 完全一致 |
| --- | ---: | ---: | ---: |
| AR / B=1 | 184.37 | 0.9922 | 8/8 |
| 原 Uno / B=4 | 223.33 | 1.4150 | 0/8 |
| 原 Uno / B=8 | 225.07 | 1.4598 | 4/8 |
| 原 Uno / B=16 | 211.02 | 1.4514 | 2/8 |

B=8 相对同运行时 AR 的吞吐约 +22.08%。这是本机小模型上的原版工程复现，不是论文全规模复现，
也不是 online 改进；不把它和 Windows HF FP32 的绝对 TPS 差异归因到某个算法。
模型加载/graph 初始捕获 7.454 s 单独排除；完整生成计时包含 prefill、调度、decode、detokenization。

所有 32 个 post-run GPU 快照 SM=1950 MHz、memory=9501 MHz；温度 50–56℃。
所有 decode graph misses=0，target/adapter parameters 只冻结、不修改。

## 行为差异不能隐瞒

BF16 官方原版不同 B 并不都产生逐 token 相同的 greedy 输出：24 个 speculative runs 中只有 6 个
与 AR 完全一致，首次差异在生成位置 25–94。这个差异在 R7 开启前已存在，不能归咎于在线学习。
抽查文本仍在讨论对应哈希冲突、ACID、合并区间、求和任务，但不是质量基准或等价性证明。

不同 kernel/GEMM shape 下的低精度差异是一个待验证解释，**本轮没有完成其因果诊断**；
不能仅靠 BF16 标签把所有差异认定为无害误差，更不能宣称实机 bitwise exactness 已通过。
R7 必须同时对照同一官方原版的行为，以及通过固定宽度 shadow 检查外围 wrapper 本身。

## 证据

- [完整 token、时间、GPU 状态与环境](../results/stage12_official_fa2_baseline.json)
- [完整矩阵审计与首差异位置](../results/stage12_official_fa2_baseline_audit.json)
- 原始 JSON SHA-256：`c10f062b909e90ad96be802ee5221933e098578e0a1ee3ee0be2cb22ab785518`。
- [Linux 运行时安装与 kernel 证据](STAGE12_LINUX_RUNTIME_PROGRESS.md)

下一步为已经冻结的 60-run R7 请求内在线块长 pilot；不根据本基线结果修改其 B₀=8 或超参。
