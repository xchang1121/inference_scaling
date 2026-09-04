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

Stage 4B 结果（2026-09-05）：真实 checkpoint 接入和安全门完成，但预注册性能门失败。`online_s10`
paired TPF ratio 为 0.9796 [0.9470, 1.0000]，decode TPS ratio 为 0.9624
[0.9533, 0.9819]；`online_s20` 分别为 1.0068 [0.9574, 1.0286] 和
0.9860 [0.9243, 1.0075]，均不确定。45/45 固定长度，30/30 online parameter-isolation 记录通过。
下一阶段将 same-buffer transactional update 改成跨未来窗口验证的 shadow candidate，再决定 promote/reset。

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

Stage 5A 状态（2026-09-05）：已实现 future-validated shadow candidate。过去窗口只训练候选，下一窗口
用实际 filtered $1-TV$ 比较 active/candidate/zero 后才 promote/keep/reset；candidate clone 保留真实
zero anchor 与 optimizer state。支持每 $K$ 轮采一批训练 feedback，以降低 Stage 4 的 1.2% 每轮
materialization 成本。假模型端到端和 promote/reset/keep 单测通过后进入真实 checkpoint Stage 5B。

Stage 5B 协议状态（2026-09-05）：2-seed 英文工程 pilot 后冻结 `stride=40`、`feedback_interval=4`、
`candidate_evaluation_interval=4`、promotion margin `0.0005`。实现会在 active 仍为 zero residual 时跳过
无效词表投影，并对 shadow future evaluation 降频。正式实验固定为 3 prompts × 5 paired repetitions ×
512 tokens；判定规则已写入 `STAGE5B_DEFERRED_ONLINE_PROTOCOL.md`，正式数据不得反向修改协议。

Stage 5B 结果（2026-09-05）：安全门通过；TPF mean ratio `0.99503 [0.98713, 1.00285]`，TPS mean
ratio `1.02194 [0.99665, 1.04774]`，学习门与系统门均失败。中文 prompt 的 TPF 区间完全低于 1，且它
反而获得最多 promotion，证明一窗口后才激活的 active-trajectory shadow TV 不能可靠预测下一窗口收益。
下一阶段不再扫描 margin，而研究渐进 mixture 或跨请求 amortization。详见
`STAGE5B_DEFERRED_ONLINE_RESULTS.md`。

Stage 6A 状态（2026-09-05）：优先实现 OSD 式跨请求 persistent learner。runner 已增加显式外部 learner
接口，并验证 config/shape/device/optimizer ownership；diagnostics 审计请求首尾 fast-weight L2。下一门是
实现 train/validation/test stream harness，在 held-out future requests 上比较 frozen residual 与 static，
并计算 observed/instrumented 两种 break-even。设计见 `STAGE6_STREAM_ONLINE_DESIGN.md`。

Stage 6B 实现状态（2026-09-05）：stream harness 已完成，保存 zero 与逐请求 snapshot，只凭 validation
mean TPF 和固定 margin 选择，随后释放未选状态并在独立 seed 区间测试；training static pair 与显式 kernel
计时共同给出 observed/instrumented break-even。先做单域工程 pilot，再冻结多模板正式协议。

Stage 6C 协议状态（2026-09-05）：2-test-seed stationary pilot 出现 TPF/TPS 正方向后，冻结一个全新 seed
的 repeated-query case study：4 train、5 validation、10 test；只按 validation mean TPF 选 0--4 快照，
低于 0.2% gain 自动回退 zero。正式成功要求 nonzero selection、test TPF 和 frozen TPS 的 mean bootstrap
下界都大于 1。范围明确限于同一 prompt 分布。

Stage 6C 结果（2026-09-05）：安全与 nonzero selection 门通过，但 10 个新 seed 的 TPF mean ratio 为
`0.98048 [0.94885, 1.01313]`，TPS 为 `0.98475 [0.94301, 1.02419]`；validation-to-test optimism gap
达 4.21 个百分点，未来学习/系统门失败且无 break-even。下一阶段实现概率空间 static mixture，直接限制
hard residual 的下尾风险，而不是继续扫描 validation threshold。

Stage 7A 状态（2026-09-05）：已实现 `q_w=(1-w)q_static+wq_candidate` 的 sparse 概率 mixture；重复 support
id 的概率由 sampler/verifier 求和，保留 exactness。非单位权重暂只允许 frozen persistent 请求，防止现有
old-q surrogate 与真实 mixture proposal 不一致。固定 `w=0.25` pilot 的 validation TPF +5.94%，但 5 个
新 test seeds 的 mean TPF 为 `0.99493`，不进入正式正结论。

Stage 7B 实现状态（2026-09-05）：增加 per-request verifier-feedback EMA scalar gate。controller 从 static
开始，每 4 cycles 在已经验证的 on-policy rows 上比较 pure static/candidate filtered TV；warmup 后只有正证据
才令**下一轮**使用 capped `w=0.25` mixture，负证据则退回 static。head 在整个 frozen evaluation 请求中
不更新；zero snapshot 保持逐 token/forward static 等价，实际每轮 $q_t$ 仍原样交给 exact verifier。

Stage 7B pilot 结果（2026-09-05）：4 个非零 snapshots 的 validation mean TPF ratios 均低于 1，规则正确
回退 zero；5 个 test TPF ratios 因而全部严格为 1。非零 snapshot advantage 的 lag-1 correlation 仅在
`[-0.170, 0.231]`，不支持继续扫描 EMA/margin。下一阶段用 greedy target 固定 AR trajectory，单独检验
跨请求 residual 能否对新的 Uno noise seeds 泛化。
