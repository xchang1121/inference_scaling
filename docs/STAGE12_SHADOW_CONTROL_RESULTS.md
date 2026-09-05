# Stage 12：固定 B=8 的在线框架行为对照

2026-09-05，按[事先记录的协议](R7_SHADOW_CONTROL_PROTOCOL.md)完成 24/24 runs。
同一 BF16/FA2/CUDA-graph 引擎，四个旧 prompts × 2 repeats × AR/B8/shadow8。

**8/8 个配对中，shadow8 和官方 B=8 的完整 token IDs、文本以及 official stats 完全一致。**
所有在线 trace 的动作均为 8，policy 仍然真实更新统计；没有改写模型或 decoder。
这直接验证了该控制实验中：仅接入外围学习框架，不改变块长时，保持原版行为。

| 方法 | 聚合 E2E TPS | decode-only TPF |
| --- | ---: | ---: |
| AR | 184.38 | 0.9922 |
| 官方 B=8 | 218.60 | 1.4391 |
| shadow8 | 216.92 | 1.4391 |

shadow8/原版 TPS=0.99234，本组约 -0.77%；包含在线更新和整个 wrapper 的完整计时。
只作描述性工程开销对照，不由 8 对小样本推断精确固定 overhead。
所有 post-run memory clocks=9501 MHz、SM=1950 MHz，preemptions=0、graph misses=0。

## 结论范围

本测试支持外围 wrapper 在同动作时保持原版输出；它没有证明动态 B 的所有输出都逐 token 一致。
本组官方 B=8 和 shadow8 各有 4/8 与 AR 不同，且差异完全相同；因此这部分不是 wrapper 引入的。
原生 R7 动态实验的跨 B 数值/行为差异仍需单独诊断，不能用本控制测试将其消除。

## 证据和回归

- [24 次完整结果](../results/stage12_native_shadow8.json)
- [完整审计](../results/stage12_native_shadow8_audit.json)
- 原始 SHA-256：`ad7f41beb8b4155c68b640186b04bedbc169a66dc8b42ba266debf850ef3d95f`。
- 真实 32/60/24-run 矩阵及 shadow 完整输出一致性已加入 CPU 回归测试。
- 清除一次性下载器测试并增加原始数据 SHA 回归后，最终全套 **218 passed**，Ruff 全通过，Linux pip check 通过。
- `.gitattributes` 固定本轮原始结果为 LF，避免 Windows checkout 的 CRLF 转换破坏已记录的字节摘要。
