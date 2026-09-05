# R3C packed-tree FP32 smoke

2026-09-05。数据：[stage11_tree_fp32_smoke.json](../results/stage11_tree_fp32_smoke.json)。
已通过 148 项测试；本组为 4 prompts × 1 seed × 4 methods 的工程 smoke，非独立测试集。
所有方法预热 256 tokens；每次固定输出 256 tokens；B=8，top-K=4，包含 top-1 spine。

| 方法 | 总 E2E TPS | 相对 static TPS | 平均 TPF 比值 | 全部 token IDs 与 static 相同 |
| --- | ---: | ---: | ---: | --- |
| static B=8 | 39.2887 | 1.0000 | 1.0000 | 是 |
| tree N=8 | 40.6292 | 1.0341 | 1.0000 | 是 |
| tree N=16 | 45.5379 | 1.1591 | 1.1588 | 是 |
| tree N=32 | 43.4762 | 1.1066 | 1.2110 | 是 |

N=16 在四个配对 prompt 上的 E2E ratio 分别为 1.1548、1.1859、1.1381、1.1552。
这是新主线首次同时表现出接受长度和实际速度的正方向。
但每个任务只有一次、运行仍有背景下载、静态最优 B 尚未比较，不能据此声称稳定泛化收益。
N=32 在代码任务为 0.9618×，表明更大树的 TPF 增益不保证 TPS 增益。

下一组（读取结果前确定）：FP32、同四个 pilot prompts、全新 seeds、三次重复，
static B=4/8/16，frozen-rank N=16，online-rank N=16/32。
在线 rank 更新为请求内 past-only，prior strength=8，decay=0.98；不修改任何网络权重。

计时改进：从 schema v2 起用完整 generation call 的外层墙钟计时，额外覆盖初始化、
请求结束统计与 text decode；共享 prompt encoding、模型加载和 benchmark JSON I/O 不在分母内。
保留 prefill_plus_decode_seconds 供与旧计时口径核对。不追改本组 schema v1 原始数据。
