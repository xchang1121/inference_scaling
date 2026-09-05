# Stage 12：R7 原生在线 Uno 的 60-run 实测

2026-09-05。按[R7 事先冻结的设计/证明/协议](R7_NATIVE_ONLINE_DESIGN_AND_PROOFS.md)，
RTX 3090、WSL2、BF16、FA2、CUDA graphs，四个旧 pilot prompts × 3 repeats × 5 方法，
每请求 256 tokens；60/60 runs 完成，固定目标/adapter，preemptions=0，decode graph misses=0。

## 结论先行

R7 已在官方原生引擎上真正在线学习块长，端到端 TPS 与原 Uno B=8 接近。
本组约高 1.11%，但置信区间跨过 1，**没有证明稳定的在线额外提速**。
这符合用户认可“接近原 Uno 也有价值”的方向，但不是已证明的行为等价性或论文级优化收益。

| 方法 | 聚合 E2E TPS | 官方 decode-only TPF | R7 / 该方法 TPS |
| --- | ---: | ---: | ---: |
| AR / B=1 | 185.76 | 0.9961 | 1.1376 |
| 原 Uno / B=4 | 207.80 | 1.3516 | 1.0169 |
| 原 Uno / B=8 | 208.98 | 1.3846 | 1.0111 |
| 原 Uno / B=16 | 198.38 | 1.3796 | 1.0652 |
| R7 online | 211.31 | 1.3909 | 1.0000 |

R7 相对 B=8 的 prompt-cluster bootstrap 95% 区间为 [0.97025, 1.07206]；
相对 B=4 为 [0.97477, 1.06488]。只有 4 个旧题目 cluster，区间不稳定，不是 confirmatory study。
相对 AR +13.76% 是包含 Uno 自身加速的总效果，不能称作 online learning 独有收益。
不要混用上一组 128-token 基线的 225.07 TPS 和本组 256-token 结果做配对比较。

## 在线状态确实更新，不是静态配置重命名

- 1,100 个真实 decode cycles：B=8 使用 687 次，B=4 使用 267 次，B=16 使用 146 次。
- 完成 548 个 epoch 更新；12 个请求各自从空统计开始，没有跨题 warm-start。
- 初始化探测 72 cycles，guarded exploit 980 cycles，刷新探测 48 cycles。
- `pending` 全部清除，逐轮 committed token 总数与官方 accepts 对账；没有多算 prefill。
- target/base 和原 diffusion adapter 均固定；optimizer_steps=0、model_weight_updates=0。
  **这是在线策略学习，不是在线训练 LoRA。**

仪器化记录的 choice+observe 时间共 0.014460845 s / 1,100 cycles，约 13.15 µs/cycle。
该数不是总 wrapper 开销：trace append、构造/快照等仍有成本；所有这些成本均已包含在
R7 的总 E2E 时间 14.538139923 s，不能把仅约 0.1% 的 choice+observe 比例当作完整开销比例。
没有为在线策略添加 CUDA synchronization，也没有将更新移出计时区间。

所有 60 个 post-run memory clocks 为 9501 MHz；SM=1950 MHz 有 18 次，1935 MHz 有 42 次，
温度由 54℃ 增至 64℃。这些快照不证明全程频率固定，仍保留系统噪声的限制。
峰值 allocated GPU memory=11,373,778,944 bytes（包含预分配 KV/cache，并非模型权重大小）。

| Workload | 原 B=8 TPS | R7 TPS |
| --- | ---: | ---: |
| English | 213.12 | 212.24 |
| Chinese | 209.65 | 207.05 |
| Code | 195.21 | 214.99 |
| Math | 219.50 | 211.10 |

收益并非每题都为正；聚合正差主要来自 code。不同配置生成内容存在差异，不能把这个差异
毫无保留地归因到在线预算策略本身。后续需固定条件历史的数值诊断与更丰富任务评估。

## 输出边界

本组 256-token BF16 输出里，所有 48 个 speculative runs（包括固定宽度和 R7）都与 B=1 AR
存在 token 差异；这不是 R7 独有现象，见[未修改官方基线](STAGE12_OFFICIAL_BASELINE_RESULTS.md)。
R7 对 AR 首差异为 83–148；原版 B=16 的 code 在 25 即分叉。
所有 token 和文本都保留，不把“任务主题看起来相同”当作质量等价或 exactness 已认证。

数学证明提供的是：固定 B 解码正确、target 不变、轮内 proposal law 不变时，自适应 B 保持
同一条件采样法则。它没有界定该 GPU/BF16 实现的数值 ε，也没有诊断官方跨 B 差异的根因。
另用[固定宽度 shadow 协议](R7_SHADOW_CONTROL_PROTOCOL.md)检验 wrapper 在动作相同情况下是否改变输出。
该[控制实验现已完成](STAGE12_SHADOW_CONTROL_RESULTS.md)：8/8 对输出及 stats 与官方 B=8 完全一致，
shadow8 216.92 TPS vs 官方 218.60 TPS（约 -0.77%）。动态 R7 与同组 B=8 的 12 对完整输出则均有差异；
不能把固定动作的控制通过外推为所有自适应动作都 bitwise exact。

## 可复核证据

- [60 次完整原始结果](../results/stage12_native_online_r7_pilot.json)
- [矩阵、时序反馈、graph 与输出审计](../results/stage12_native_online_r7_pilot_audit.json)
- 原始 SHA-256：`8906f1909046fe2efbc72e5914e730b2b77d36b2c1eccd994650d06597a1968b`。
- 固定设计、CPU 单测、真实 kernel smoke、原版结果及负结果均保留在版本历史中。

当前建议：把 R7 作为可运行、低开销的原生在线候选，继续独立评估，而不是据 +1.11% 宣称
最优或稳定加速。进一步优化首先区分 acceptance 改善、shape 数值差异和实际系统耗时。
