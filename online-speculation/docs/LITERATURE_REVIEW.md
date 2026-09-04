# Online Uno 文献矩阵与设计启示

本页只把论文或作者官方代码作为算法事实来源；二手解读仅用于发现关键词，不作为实现依据。

## 1. 直接基础

| 工作 | 已验证的关键结论 | 对 Online Uno 的作用 |
| --- | --- | --- |
| [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) | proposal $q$ 经逐 token rejection sampling 和 residual correction 后可严格得到 target $p$ | 正确性底座；在线更新不能改变本轮实际 proposal 的分母 |
| [Uno: Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010) | frozen AR + conditional diffusion LoRA；TV loss 直接针对接受概率；linear/tree $\Psi$-Spec lossless | 在线版必须保留 frozen verifier、conditional routing 与 exact correction |
| [Uno 官方实现](https://github.com/ifm-ai/uno) | Nano-vLLM 两遍 draft/verify、shared KV cache、conditional LoRA、full-vocab chunked TV | 后续性能实现的上游基线与 patch 点 |
| [DistillSpec](https://research.google/pubs/distillspec-improving-speculative-decoding-via-knowledge-distillation/) | draft/target 对齐的损失、数据来源和采样分布会改变接受率 | 为 KL/TV、on-policy/off-policy 数据消融提供基线 |

Uno linear sampler 每轮先由 base AR 精确产生一个“免费”token，再由 diffusion pathway 并行提出
$B-1$ 个 token，最后 AR verifier 一次检查。若第一个 speculative token 就拒绝，仍推进免费 token 和
residual replacement；若全接受，再从 verifier lookahead 采一个 token。因此一轮推进长度在
$[2,B+1]$，而 draft 和 verify 共两次 forward，论文定义

```math
1 \leq \mathrm{TPF} \leq \frac{B+1}{2}.
```

## 2. 已有在线推测解码

### Online Speculative Decoding（OSD，ICML 2024）

[论文](https://arxiv.org/abs/2310.07177)；[官方代码](https://github.com/LiuXiaoxuanPKU/OSD)。

OSD 把 target correction 和 logits 放入 replay buffer，每隔 $I$ 个请求更新小 draft。它比较 forward
KL、reverse KL 和 JSD，并提出在服务低负载、有 spare FLOPs 时机会式更新。可迁移部分：

- verification feedback 是零额外 teacher-forward 的蒸馏数据；
- buffer 允许批量更新、重放困难状态和把更新移出请求关键路径；
- 多个 domain-specific drafts 可由 routing 选择。

PMLR 正式版本报告在其模型/服务设定中 acceptance rate 增加 0.1--0.65、latency reduction 为
1.42--2.17 倍；这些数字依赖独立小 draft、请求流和可用 spare FLOPs，不能直接移植成 Uno/3090 的预期值。
本项目只采用其“跨 observed query stream 持久化与 replay”的实验单位。

限制是其 draft 与 target 是两套模型，且主要按错误 token 收集；Uno 的共享 backbone 反传成本和状态管理不同。

### Test-Time Speculation（TTS，2026）

[论文](https://arxiv.org/abs/2605.09329)。TTS 直接在长序列生成中交错 draft update，目标为位置加权
forward KL 加“新旧 draft”KL 正则。它发现逐轮 backward 虽然接受长度最高，却可能抵消速度收益；strided
update 的最佳点依 workload 而变，并用独立 CUDA stream 与后续轮次重叠。论文在 LiveCodeBench/Qwen3-8B
上报告 stride 10 相对 stride 1 的 wall-clock 因子最佳，同时指出固定 stride 应被在线 controller 取代。

可迁移部分：

- update stride 是系统参数而非只属于优化器；
- old-draft regularization 可防单条 trajectory 过拟合；
- inference fused kernels 往往没有正确 backward，更新轮需切换到可微 kernel；
- 对 3090 这类单 GPU，CUDA stream 重叠仍会争抢 SM/带宽，必须实测净收益。

### OnlineSPEC（ICML 2026）

[论文](https://arxiv.org/abs/2603.12617)；[官方代码](https://github.com/ZinYY/OnlineSPEC)。

OnlineSPEC 将每轮 target feedback 写成 online loss，以 dynamic regret 衡量随时间变化的最优 draft。
它明确证明单 token 的期望接受率

```math
\operatorname{Acc}_t
= \sum_x \min\{p_t(x),q_t(x)\}
= 1-D_{\mathrm{TV}}(p_t,q_t),
```

并把 regret、draft/target 成本比和候选长度联系到 speedup。其 optimistic update 利用历史梯度作为 hint；
ensemble 以多学习率 learner 对抗非平稳 workload。对本项目最重要的不是直接照搬多份大 LoRA，而是：

- 用 EMA gradient/low-rank optimizer state 近似 optimism；
- 用少量学习率或 update-policy arms 做 controller ensemble；
- 报告 static regret、dynamic regret proxy 和实际 tokens/s，检验理论 proxy 是否预测系统收益。

官方仓库的 Ens-EAGLE/EAGLE-3 复现命令显式维护三份不同学习率 draft，并以 `chunk-size=40` 消费流数据；
Opt-Hydra 则复用历史梯度作为 optimistic hint。这进一步说明 ensemble/chunk 的状态跨样本存在，而不是
每个请求从 zero optimizer 重启。

## 3. 并行 drafter 与动态控制

| 工作 | 可迁移点 |
| --- | --- |
| [DFlash](https://arxiv.org/abs/2602.06036) / [代码](https://github.com/z-lab/dflash) | block diffusion draft、target feature/KV 注入；是 TTS 已验证可在线更新的最接近架构对照 |
| [Draft Model Knows When to Stop / SVIP](https://arxiv.org/abs/2411.18462) | 用 draft entropy 估计难度并动态停止；可作为 Uno 动态 $B_t$ 的无训练 controller 对照 |
| [SpecDec++](https://arxiv.org/abs/2405.19715) | 从 token-level acceptance proxy 决定候选长度；用于比较 EMA-TV controller |
| [Learning to Draft](https://arxiv.org/abs/2603.01639) | 直接优化 draft+verify cycle throughput，而非 acceptance-length proxy | 支持本项目把 wall-clock reward 作为最终 controller 目标 |

## 4. 本项目相对已有工作的新增问题

### 4.1 本轮旧 proposal 不变量

令第 $t$ 轮开始前的 fast weights 为 $\delta_t$，proposal 为 $q_t=q_{\phi+\delta_t}$。
先保存 proposal token 的 $q_t$ 概率或 logits，再 verification、accept/reject，最后才执行
$\delta_t\rightarrow\delta_{t+1}$。本轮接受率永远使用旧 $q_t$：

```math
a_{t,i}=\min\left(1,\frac{p_{t,i}(y_{t,i})}{q_{t,i}(y_{t,i})}\right).
```

只要每轮条件于 history 和旧 fast weights 的 correction 精确，proposal 如何依赖过去反馈都不改变 AR
联合分布。实现测试必须故意构造“先更新再验”的错误版本，确保分布检验能抓到偏差。

### 4.2 first rejection 后的 supervision

若 proposal $y_{1:B}$ 在位置 $J$ 首次拒绝，verifier 对 $i>J$ 的 logits 条件于已经不会进入真实 history 的
draft prefix。这些数据不是无效，但属于 hypothetical/off-policy states。已有 TTS 使用整个 canvas；本项目比较：

1. `full_canvas`：全部位置，复现 TTS 风格；
2. `on_policy_prefix`：只到 $J$（包含 rejection state）；
3. `discounted_tail`：$i\le J$ 权重 1，尾部按 $\rho^{i-J}$ 衰减；
4. `replay_corrected`：下一轮真实 replacement prefix 的 logits 进入 replay。

这是一项实验假设，不预先声称 masking 一定更好。

### 4.3 TV、KL 与可计算近似

Uno 离线目标使用 full-vocabulary L1（论文记作 TV loss，但公开实现未乘 $1/2$，常数不影响最优点）。
在线候选目标：

```math
\mathcal L_t=
\sum_{i=1}^{B-1}m_{t,i}w_i
\left[
\gamma D_{\mathrm{TV}}(p_{t,i},q_{\delta,t,i})
+\beta D_{\mathrm{KL}}(p_{t,i}\Vert q_{\delta,t,i})
+\lambda D_{\mathrm{KL}}(q_{\delta_t,t,i}\Vert q_{\delta,t,i})
\right].
```

- TV 与期望接受率直接对应，但 full-vocab backward 贵且在相等概率处不光滑；
- forward KL 梯度平滑，TTS/OSD 有先例，但并不直接最小化 rejection probability；
- top-$K$+tail bucket TV 可减少保存/传输，必须测其对 acceptance 的偏差；
- 只对 sampled token 做 loss 很便宜，却可能忽略 residual distribution，作为低成本消融而非默认。

### 4.4 slow weights + fast weights

默认结构不是在线更新完整 rank-128、0.35B Uno adapter，而是

```math
q_t=q_{\theta_{\mathrm{AR}}+\phi_{\mathrm{offline}}+\delta_t},
```

其中 $\theta_{\mathrm{AR}}$ 和 $\phi_{\mathrm{offline}}$ 冻结，$\delta_t$ 是 request/domain 级 fast weights。
候选从低到高成本依次为：logit affine correction、per-layer gate、顶部若干层 rank-4/8 LoRA、完整 adapter。
request 结束可丢弃 $\delta_t$，domain 版本则经过 replay 验证后再持久化。

### 4.5 净收益条件

设 static Uno 每轮 draft/verify 成本为 $C_D,C_V$，平均推进 $\tau_0$；在线更新后为 $\tau_1$，
一次 update 成本 $C_U$，每 $S$ 轮一次，则在线方案更快的必要且充分比较为

```math
\frac{C_D+C_V+C_U/S}{\tau_1}
<\frac{C_D+C_V}{\tau_0}
\quad\Longleftrightarrow\quad
\frac{\tau_1}{\tau_0}>
1+\frac{C_U}{S(C_D+C_V)}.
```

因此 controller 的 reward 使用“推进 token / 实际耗时”，而不是单独最大化 TV 或接受率。

## 5. 待验证假设

- H1：对当前真实 trajectory 的在线蒸馏能在 domain shift/长输出下提高 Uno acceptance length。
- H2：小 fast weights 能取得完整 adapter 更新的大部分收益，而 update wall time 显著更低。
- H3：on-policy/discounted-tail supervision 在同等 update budget 下优于 full-canvas。
- H4：自适应 stride 和 block size 比任一固定组合有更高端到端 throughput。
- H5：纯 online cold start 在单请求内很难回本，但跨请求/domain replay 可以摊销；需要明确 break-even 请求数。

Stage 4B/5B 已实证支持 H5 的前半句：request-local immediate 与 deferred 均未通过真机门。Stage 6 将把
OSD 式 request stream 与 TTS 式 within-request update 分开报告，避免用一个含糊的“online”混合两种设定。
