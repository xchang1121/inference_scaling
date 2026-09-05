# Stage 11 / R3：候选复用与 warm-start 的完整 pilot

2026-09-05。工程筛选结果，不是 held-out confirmatory 结论。
模型为锁定 K2-Horizon-0.9B + Uno adapter，Windows HF / RTX 3090，B=8。
每组包含英文、中文、代码、数学四个 prompt，各三次配对、固定输出 256 tokens。

## 1. Direct recycling：未达到成功门

原始记录：[BF16 pilot](../results/stage11_recycling_pilot_b8.json)。

| 方法 | 总 E2E TPS 比值 | paired mean speedup / 95% CI | paired TPF 比值 |
| --- | ---: | --- | ---: |
| always recycle | 0.8344 | 0.8380 [0.7189, 1.0245] | 0.7451 |
| bounded depth=2 | 1.0036 | 1.0082 [0.8799, 1.2268] | 0.8962 |
| TPS gated | 1.0605 | 1.0467 [0.9513, 1.1926] | 0.9677 |

所有新方法的配对 BF16 token 序列均出现差异，未通过逐 token 一致性门。
工程首组静态运行经历明显 GPU 时钟/吞吐变化，背景有 WSL 下载。
因此 TPS-gated 的 1.0605 **不能当成可靠加速结果**：CI 跨 1，且早期静态分母受污染。
TPF 下降表明尾部候选准确度不足，节省一个 forward 未稳定转化为收益。

数学 exactness 证明不依赖候选准确度，但也不能证明不同 BF16 kernel 计算的 logits 相同。
当前证据不足以将全部差异归因于浮点计算；数值原因与实现错误需要分别排查。

## 2. Warm-start：FP32 通过 token 门，未改善 TPS

原始记录：[FP32 pilot](../results/stage11_warmstart_fp32_pilot.json)。
使用新 seeds，所有方法预热 256 tokens；保存每次前后 GPU 温度、频率、功耗。
这组是数值控制实验，不能与前一 BF16 组跨组比较绝对 TPS。

| 方法 | 总 E2E TPS | 相对静态总 TPS | paired mean speedup / 95% CI | paired TPF 比值 |
| --- | ---: | ---: | --- | ---: |
| static B=8 | 41.1323 | 1.0000 | 1.0000 | 1.0000 |
| warm-start g=1 | 40.8299 | 0.9926 | 0.9933 [0.9748, 1.0110] | 1.0001 |
| warm-start g=0.5 | 38.2063 | 0.9289 | 0.9294 [0.9050, 0.9543] | 0.9421 |
| warm-start g=0 | 30.3765 | 0.7385 | 0.7407 [0.7029, 0.7748] | 0.7400 |

36 个新方法运行全部与配对 static 的 token IDs 相同。
g=1 的 prompt-cluster mean CI 为 [0.9668, 1.0167]，仍跨 1。
只有四个 prompt，不能利用重复 seeds 将样本量解释成 12 个独立任务。

## 3. 决策

不将上述方案推广为默认加速器，不发布论文级收益声明。
下一版改为预算受限的多分支 tree verification，利用 Uno 已计算的各位置 logits
覆盖更多可能的正确前缀。在线学习对象为 rank 覆盖统计与实际预算耗时，不更新 target。
树验证、多候选和成本控制有既有文献；当前工作是 Uno/3090 集成和可复现评估。

原始失败记录保留，可追溯到对应实现提交。清理只调整活跃目录与归档，不删除负证据。
