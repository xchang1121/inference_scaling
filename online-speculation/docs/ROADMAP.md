# 实现与实验路线图

每个阶段只有在代码、测试、机器可读结果和结论文档同时达到门槛后才算完成，并独立 commit + push。

## Stage 0：审计与版本锁定

交付：硬件 manifest、上游 commit/checkpoint revision、文献矩阵、复现边界。

通过条件：

- preflight 可在无模型下载时重复运行；
- 明确区分算法正确性、Uno 1B 性能趋势和论文 H200 数字；
- 发现的论文/公开 recipe 差异进入文档而非静默覆盖。

## Stage 1：可枚举 $\Psi$-Spec 核心

实现纯 NumPy/PyTorch categorical linear sampler，输入每个位置的 $p_i,q_i$ 和 proposal，输出 longest
accepted prefix、residual correction 和 lookahead token。覆盖 greedy 与 stochastic 两条路径。

实验：

- 二元/五元 vocabulary，长度 1--8；
- fixed、历史依赖和每轮变化的 $q_t$；
- 至少 $10^5$ Monte Carlo samples，比输出 token/短序列频率与 target；
- 错用更新后 $q_{t+1}$ 的负面对照必须显著失败。

通过条件：解析单步概率误差 $<10^{-12}$；Monte Carlo 误差落在预注册置信区间；tests 全通过。

## Stage 2：Uno 1B 静态真机基线

固定官方 source/model revisions，优先 FA2 linear sampler。比较：

- base AR，temperature 0 与 1；
- Uno $B\in\{2,4,8,16\}$；
- 相同 prompt/output token budget，batch 1，随后测试可容纳的 batch；
- warmup 后至少 10 次重复，报告 median、IQR 和 bootstrap CI。

指标：output tokens、forward 次数、TPF、accepted prefix、TPS、TTFT、峰值显存、GPU 功耗/利用率。

通过条件：至少一个非退化 workload 上 TPF>1；只有 wall-clock CI 显示 speedup 才称为“复现加速”。

实际阶段门结果（2026-09-05）：公开 Uno-1B checkpoint 的 HF KV-cache 回退 backend 通过。
$B=8$ 的 TPF median 1.401（paired bootstrap 95% CI [1.341, 1.432]），相对 AR 的 decode
speedup median 1.352×（[1.250, 1.386]），10/10 配对运行胜出。官方 Nano-vLLM 因当前 Windows
没有 Triton/FlashAttention 而未执行；该项保留为独立的 Linux 系统复现任务，不冒充已完成。

## Stage 3：在线学习仿真与成本 controller

先在 tabular/小神经 proposer 上引入 distribution drift，比较 static、per-round、fixed-stride、adaptive-stride，
并实现 full/on-policy/discounted supervision。

预注册 controller：

```math
\widehat g_t(S,B)=
\frac{\operatorname{EMA}(\text{committed tokens}|S,B)}
{\operatorname{EMA}(C_D+C_V+C_U/S|S,B)}.
```

仅当收益下置信界高于 static baseline 上置信界时增加更新频率；低于时退避。block size 使用相同 reward
或轻量 bandit，不用 acceptance 单指标决策。

实际阶段门结果（2026-09-05）：完成。预注册 `stride10_discounted` 的 dynamic/static TV-regret ratio
为 0.9098（95% CI [0.9083, 0.9106]），paired TPF ratio 为 1.1688
（[1.1635, 1.1756]），中央合成成本下 efficiency ratio 为 1.1419
（[1.1367, 1.1485]）。逐轮更新因 14.94% update cost 几乎不回本；第一版 adaptive controller 因把
当前 fast-weight 收益误当成下一次 update 的边际收益而过度更新。return-to-domain 区间的效率降至 static
的 0.8984，暴露出必须加入 change detection、fast-weight decay/snapshot rollback。GPU backward 尚未
测试，不能把本阶段 proxy 写成真实系统加速。

## Stage 4：Online Uno fast weights

最小实现次序：

1. 保存本轮旧 draft logits/probabilities；
2. verification 后构造可微 replay item；
3. request-local logit correction；
4. top-layer rank-4/8 fast LoRA；
5. strided mini-batch update；
6. 若 3090 有余量，再做 CUDA stream overlap。

安全不变量：base AR 和 offline Uno adapter 永不被 optimizer 持有；optimizer parameter IDs 在测试中白名单；
任一 NaN、KL/TV 激增或吞吐下界恶化触发 rollback/disable-update。

Stage 4A 状态（2026-09-05）：完成独立 low-rank logit residual learner。zero-init 保持 static logits，
detached hidden + draft/target top-K union 构造稀疏 replay，transactional AdamW update 支持 static-shadow
reset、held-out rollback、向 offline snapshot 衰减；optimizer/base parameter ID 白名单已由合成测试覆盖。
下一门是接入真实 K2-Horizon-0.9B-Uno draft/verify loop，并计入 feedback、backward、optimizer 与同步时间。

## Stage 5：消融与最终判定

核心矩阵：

| 维度 | 候选值 |
| --- | --- |
| 模型 | tabular、Qwen 0.5B/1.5B 原型、Uno 1B |
| loss | TV、forward KL、TV+old-q KL、top-K approximate TV |
| supervision | full、on-policy、discounted tail |
| fast weights | logits、gate、rank 4、rank 8、full adapter（若可行） |
| stride | 1、5、10、20、adaptive |
| block | 2、4、8、16、adaptive |
| workload | stationary、domain shift、long continuation、mixed requests |

成功分三级报告：

- **正确性成功**：生成分布与 AR target 无统计可辨偏差；
- **学习成功**：配对实验中 acceptance/TPF 显著改善；
- **系统成功**：计入 update 后的 tokens/s 显著超过 static Uno。

若只达到前两级，结论必须写“online adaptation 有效但尚未净加速”，不能用接受率代替系统效果。
