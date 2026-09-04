# Stage 8：Greedy repeated-query Stream-Uno 预注册协议

## 1. 冻结问题

本协议在读取 base seed `20262405` 的任何 train/validation/test 输出前提交。它只回答：

> 当 target AR 使用 greedy decoding、同一个请求重复到达而 Uno noise seed 变化时，前 4 个请求中边推理边
> 更新的 persistent rank-8 residual，能否经 5 个 validation noise seeds 选出非零快照，并在 20 个未见
> noise seeds 上稳定提高 frozen TPF；计入推理 residual 后是否也提高 decode TPS？

这是一个刻意受限的 repeated-query case study，不代表 stochastic sampling、新 prompt、多 domain、官方
Nano-vLLM runtime 或在线更新完整 diffusion LoRA。它的价值在于隔离 Stage 6/7 暴露的 stochastic target
trajectory shift：greedy target token 轨迹固定，seed 只改变 diffusion draft 的初始 noise 和 proposal。

## 2. Pilot 只用于选择协议

两个工程 pilots 已使用相同 prompt/config，因此只有正式 seed 区间是 held out：

| pilot | train/val/test | validation 选择 | test mean TPF | test mean TPS |
| --- | --- | --- | ---: | ---: |
| `20262205` | 4/2/5 | snapshot 2，`1.00773` | `1.00105` | `1.01896` |
| `20262305` | 8/1/2 | snapshot 2，`1.02577` | `1.03125` | `0.94907` |

第二个 pilot 的 snapshot 3--8 没有单调优于 snapshot 2，因此正式设计恢复 4 个 training requests，并保留
snapshot 0--4 的 validation-only 选择。pilot 一致提示 TPF 可能有小正信号、TPS 尚不稳定，所以两个门必须
独立报告；不能以 TPF 代替净加速。

## 3. 固定模型、后端和解码

| 项目 | 固定值 |
| --- | --- |
| base | `IFM/K2-Horizon-0.9B@ee770e713760cf6350e4322cdbbff91a163b7d70` |
| offline Uno | `IFM/K2-Horizon-0.9B-Uno@b0d8896a301a2f4bc755538b1234a35100da50d0` |
| checkpoint hashes | base `6392cc…c365`；adapter `5a4992…fe4e` |
| backend | RTX 3090 BF16，Windows HF/PyTorch KV-cache fallback |
| prompt | `Explain in three concise paragraphs why speculative decoding can be lossless.` |
| output | 512 tokens，`--ignore-stop` |
| target/proposal filter | temperature 1.0，top-k 1，top-p 1.0 |
| Uno block | 8（1 free + 7 speculative） |

`top-k=1` 令 filtered $p_i,q_i$ 都是 delta distribution。验收等价于 greedy longest-prefix match；拒绝时
residual correction 必定提交 target token。因此同一 prompt 的最终 target tokens 应与 noise seed 和 residual
状态无关，这在正式分析中作为额外 exactness 审计。

## 4. 固定在线训练和选择

| 参数 | 值 |
| --- | ---: |
| training requests | 4 |
| update stride | 40 cycles |
| feedback interval | 4 cycles |
| supervision | on-policy，position discount 0.97 |
| feedback support | raw verifier/draft top-50 union |
| residual | rank 8，alpha 8 |
| optimizer | AdamW，learning rate 0.005 |
| validation repetitions | 5 |
| test repetitions | 20 |
| selection minimum mean TPF gain | 0.002 |

训练期间每个 request 的 verifier feedback 在 exact verification **之后**进入 buffer；每 40 cycles 只更新
residual head，base 和 offline Uno adapter 永不进入 optimizer。保存 zero snapshot 以及每次 training request
后的 snapshots 1--4。

每个 snapshot 在同一 5 个 validation seeds 上冻结运行：

$$
s_j=\frac{1}{5}\sum_{r=1}^{5}
\frac{TPF_{j,r}}{TPF_{0,r}}.
$$

取最大 $s_j$；只有最好快照非零且 $s_j\ge1.002$ 才部署，否则回退 zero。并列时选更早 snapshot。test
不能改变快照、threshold、学习率、rank、stride 或 token budget。

## 5. Seed 分区和配对顺序

正式 base seed 为 `20262405`，与 stochastic 和 greedy pilots 均不重叠：

- train：`20262405 + request_index`；
- validation：`20362405 + 1000 * repetition`；
- test：`20462405 + 1000 * repetition`。

training/test 内 static 与 persistent 的先后次序交替；validation snapshot 顺序循环旋转。所有运行在同一进程
复用同一模型，正式计时前各做一次不计入统计的 static/online warmup。

## 6. 安全和 exactness 门

全部通过后才讨论效果：

1. 数值均有限，所有输出长度都是 512；
2. checkpoint revision/hash、clean/noisy conditional routing 通过；
3. 所有 online 记录 `base_optimizer_overlap=0`、trainable base tensors 为 0、fast params 为 526,336；
4. training request $r+1$ 的 initial L2 等于 request $r$ 的 final L2；
5. validation zero snapshot 对每个 seed 的 TPF ratio 精确等于 1；
6. 所有 validation/test 请求无 feedback、无 update，initial/final L2 相同；
7. selected head 在 test 前后 SHA-256 相同；
8. 每个 train、validation 和 test 配对的 `output_token_ids` 完全相同；所有 seeds 的 greedy target 输出也相同；
9. 运行数、seed 分区、顺序、选择规则与本协议一致。

第 8 条不是用 token equality 替代分布证明；它是 top-k=1 特例中对 exact verifier 集成的额外强审计。

## 7. 主统计和阶段门

对 20 个 test pairs 计算 arithmetic mean of paired ratios，并做 50,000 次 paired percentile bootstrap：

$$
R_{TPF}=\operatorname{mean}_r\frac{TPF_{selected,r}}{TPF_{static,r}},\qquad
R_{TPS}=\operatorname{mean}_r\frac{TPS_{selected,r}}{TPS_{static,r}}.
$$

同时报告 median bootstrap、逐 pair 结果和排除 exact ties 的双侧 sign test。

1. **选择门**：选择非零 snapshot；
2. **学习统计门**：$R_{TPF}$ 的 95% CI 下界严格大于 1；
3. **学习实际幅度门**：$R_{TPF}$ 点估计至少为 1.005；
4. **系统门**：$R_{TPS}$ 的 95% CI 下界严格大于 1；
5. **总成功**：安全、选择、两个学习门和系统门全部通过；
6. 学习门通过而系统门失败时，只称“在线 residual 提高 greedy speculation 效率，但当前 HF 实现没有净加速”。

训练成本另报 observed paired time increment、显式 feedback/update/head 时间和按 test mean seconds saved 计算的
break-even；它不能移动 frozen serving 系统门。

## 8. 冻结命令

```powershell
.\.venv\Scripts\python -m online_speculation.hf_stream_uno `
  --model-path ..\.tmp_k2_horizon_09b `
  --adapter-path ..\.tmp_k2_horizon_09b_uno `
  --training-requests 4 --validation-repetitions 5 --test-repetitions 20 `
  --max-new-tokens 512 --warmup-tokens 16 --block-size 8 `
  --update-stride 40 --feedback-interval 4 --feedback-top-k 50 `
  --rank 8 --alpha 8 --learning-rate 0.005 `
  --selection-minimum-gain 0.002 --temperature 1.0 --top-k 1 --top-p 1.0 `
  --seed 20262405 --ignore-stop `
  --output .\online-speculation\results\stage8_greedy_stream_uno1b_rtx3090_hf.json
```

正式 raw 写完前不得修改本协议；若运行因外部错误中断，只允许用同一命令重跑并记录原因。
