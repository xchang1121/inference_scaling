# R7 固定宽度 shadow 对照

2026-09-05，在原生 R7 60-run pilot 完成后、shadow 结果产生前添加。
目的不是调优或未见集测试，而是定位“外围在线 wrapper 是否在动作相同时改变输出”。
官方 BF16 基线已出现不同 B 的 greedy 分叉，因此不能以所有方法等于 B=1 作为这个控制实验的唯一标准。

使用相同模型/未修改官方源、FA2、CUDA graphs、同一个持久引擎，
B=1、B=8、shadow8 三方法；shadow8 使用完全相同的 `generate_online` 路径和在线统计更新，
但动作集合固定为 {8}。每题两 repeats，128 tokens，四个旧 prompts，seed=20270105，
按同一轮转/反转调度，24 runs。逐对要求 shadow8 与 B=8 的完整 token IDs 相同；
同时保留 AR 对照差异，不据此证明自适应 B 的数值误差原因。

计时包括所有 wrapper/更新开销；8 个配对、小样本噪声，TPS 只作描述。
失败保留新记录，不能覆盖已完成的官方基线或 R7 pilot。
