# R3D：稳定 QoS 下的在线树 pilot

2026-09-05。完整记录：[stage11_tree_highqos_fp32_pilot.json](../results/stage11_tree_highqos_fp32_pilot.json)。
4 prompts × 3 seeds × 6 methods，固定输出 256 tokens，完整生成调用计时，FP32 / RTX 3090。
72 个运行全部结束；全部方法的 token IDs 与同 prompt/seed 的 static B=8 一致。
144 次前/后 GPU snapshot 的 memory clock 均为 9501 MHz，没有上组的大范围频率切换。
这支持该组作为工程筛选依据，但不是因果证明“此前所有变化都是 EcoQoS 导致”。

| 方法 | 总 E2E TPS | 相对 linear B=8 | paired TPF 比值 |
| --- | ---: | ---: | ---: |
| linear B=8 | 41.2743 | 1.0000 | 1.0000 |
| linear B=4 | 40.8372 | 0.9894 | 0.9979 |
| fixed tree N=16 | 47.8604 | 1.1596 | 1.1737 |
| online rank, N=16 | 47.8194 | 1.1586 | 1.1701 |
| online budget, frozen ranks | 48.7403 | 1.1809 | 1.1816 |
| online rank + budget | 48.1241 | 1.1660 | 1.1734 |

## 不能混淆的两个效果

budget-only 对线性 Uno 的点估计为 +18.09%，paired mean 1.1841，
prompt-cluster mean CI [1.1465,1.2217]。但大部分收益来自树候选覆盖。

budget-only 对固定 N=16 树的总 TPS 仅 +1.84%。paired mean 1.0192，
seed-pair bootstrap CI [0.9979,1.0426]，prompt-mean cluster CI [1.0052,1.0327]。
不同估计单位给出的 CI 不应混用；这里仅有 4 个 pilot prompts，且据它们筛选了方法。
即使某个 CI 下界大于 1，也未满足预定至少 +5% 的在线额外收益门，更没有独立验证。

rank-only 没有稳定超过固定树；rank+budget 也不优于 budget-only。
因此不把“在线更新越多越好”写成结论。当前候选默认只更新成本统计和预算。
没有任何 target 或离线 adapter 权重更新，不属于 online LoRA SGD。

## 下一步

独立 held-out 协议已冻结，见 [TREE_HELDOUT_PROTOCOL_20260905.md](TREE_HELDOUT_PROTOCOL_20260905.md)。
包括 AR、linear B=16、fixed tree N=32，避免只拿弱静态参照对比。
官方 WSL/Nano-vLLM 运行时仍需用户重启后才能验证；此处所有数值均来自 Windows HF。
