# Stage 6C：Stationary Stream-Uno 真机预注册协议

## 1. 冻结问题与结论边界

本协议在读取 Stage 6C 的新 seed 流之前冻结，只回答一个窄但可证伪的问题：

> 对重复到达的同一个英文 speculative-decoding 请求，先在 4 个历史请求上持久化训练 rank-8 residual，
> 再只凭 5 个 validation 请求选择快照，能否在 10 个未见 seed 的未来请求上提高 frozen TPF 和 decode TPS？

这是 stationary repeated-query case study。prompt 文本和超参数已在 2-validation/2-test 工程 pilot 中用过，
所以**不能**声称对新 prompt、任意 domain 或完整 Uno diffusion LoRA 泛化。Stage 6C 的独立性只来自全新的
train/validation/test seeds 和在 test 前冻结的选择规则。

若本阶段成功，正确表述是“同一请求分布上的跨请求 fast residual 可以积累并摊销”；若失败，不能更换
seed、snapshot 或门限，后续 mixture/多域实验必须另编号。

## 2. Pilot 仅用于冻结配置

Stage 6B pilot 使用 base seed `20260905`：4 train、2 validation、2 test requests。validation 只看 TPF，
从 zero 和 4 个历史快照中选中 snapshot 3：validation mean TPF ratio `1.02304`。两个 pilot test ratios 为
`0.99507/1.02604`，均值 `1.01056`；TPS ratios 为 `1.02056/1.02020`。样本太少，不作为正式证据。

pilot 同时显示 snapshot 1/2/4 的 validation mean ratios 分别为 `0.98290/0.97605/0.98792`，说明“保留
最后状态”不可靠，validation-only checkpoint selection 和 zero fallback 是必要算法组件，而不是事后筛选。

## 3. 固定模型、后端和采样

| 项目 | 固定值 |
|---|---|
| base | `IFM/K2-Horizon-0.9B@ee770e713760cf6350e4322cdbbff91a163b7d70` |
| offline Uno | `IFM/K2-Horizon-0.9B-Uno@b0d8896a301a2f4bc755538b1234a35100da50d0` |
| device/backend | RTX 3090 BF16，Windows Transformers/PyTorch KV-cache fallback |
| prompt | `Explain in three concise paragraphs why speculative decoding can be lossless.` |
| output | 每请求 512 tokens，`--ignore-stop` |
| sampling | temperature 1.0, top-k 50, top-p 0.95 |
| Uno block | 8（7 speculative + 1 free） |

checkpoint SHA-256 和 conditional LoRA routing 必须由程序核验。base AR 与 offline Uno 永远冻结。

## 4. 固定 Stream-Uno 配置

| 参数 | 值 |
|---|---:|
| persistent training requests | 4 |
| update stride | 40 cycles |
| feedback interval | 4 cycles |
| supervision | on-policy，position discount 0.97 |
| feedback support | draft/target top-50 union |
| residual | rank 8, alpha 8 |
| optimizer | AdamW, lr 0.005 |
| validation repetitions | 5 |
| test repetitions | 10 |
| selection minimum mean TPF gain | 0.002 |

保存 snapshot 0（zero）和每个 training request 后的 snapshot 1--4。每个 snapshot 在完全相同的 5 个
validation seeds 上 frozen 运行；令

$$
s_j=\frac{1}{5}\sum_{r=1}^{5}\frac{TPF_{j,r}}{TPF_{static,r}}.
$$

取最大 $s_j$；若最好的是非零快照且 $s_j\ge1.002$，选择它，否则强制选 snapshot 0。相同分数选择更早
snapshot。test 不能参与这一步。validation 重跑只是本地 benchmark 实现；其重复运行 wall time 不计作
生产训练成本，但必须报告这一边界。

## 5. Seed 防泄漏与执行次序

正式 base seed 固定为 `20261005`，与 pilot 不重叠：

- train：`20261005 + request_index`；
- validation：`20361005 + 1000 * repetition`；
- test：`20461005 + 1000 * repetition`。

training 和 test 的 static/persistent 次序交替；validation 的 snapshot 顺序循环旋转。所有方法在一个进程中
复用同一模型，正式运行前做不计入统计的 static 与 fresh-online warmup。

## 6. 安全门

以下全部通过才讨论性能：

1. 所有 training、validation、test 输出长度为 512，数值有限；
2. hash/revision/routing 通过；
3. training request $r+1$ 的 initial fast L2 等于 $r$ 的 final L2；
4. 所有 persistent optimizer 的 base overlap = 0、trainable base tensors = 0、fast params = 526,336；
5. validation snapshot 0 对每个 seed 的 TPF ratio 精确为 1；
6. validation/test 的 feedback cycles、items、update attempts 全为 0；
7. selected head 在整个 test 前后的 SHA-256 完全相同，initial/final L2 也相同；
8. raw run order、seed partition、selection threshold 与本协议一致。

exactness 仍要求每 cycle 使用 sampling 时保存的 $q_t$ 做 acceptance denominator；跨请求 optimizer 状态只
依赖过去 feedback。frozen test 绝不更新参数。

## 7. 主统计与阶段门

对 10 个 test pairs 计算 arithmetic mean of paired ratios，并做 50,000 次 paired percentile bootstrap：

$$
R_{TPF}=\operatorname{mean}_r\frac{TPF_{selected,r}}{TPF_{static,r}},\qquad
R_{TPS}=\operatorname{mean}_r\frac{TPS_{selected,r}}{TPS_{static,r}}.
$$

同时报告 median robustness、逐 pair 数据和排除 exact ties 的 two-sided sign test。

1. **选择门**：必须选择非零 snapshot；否则学习流水线没有产生可部署状态；
2. **未来请求学习门**：$R_{TPF}$ 95% CI 下界严格大于 1；
3. **frozen serving 系统门**：$R_{TPS}$ 95% CI 下界严格大于 1；
4. **总成功**：安全门、选择门、学习门、系统门全部通过；
5. 点估计 $>1$ 但 CI 跨 1，只称方向有益、证据不足；CI 上界 $<1$ 才称明确退化。

训练摊销另报告，不作为移动系统门的替代：

- observed training increment：4 个 persistent training requests 与 paired static 的总 decode time 差；
- instrumented cost：training 中 feedback/update/head 显式计时总和；
- 以 test mean seconds saved/request 分别计算 break-even；若 mean saving $\le0$，写“不存在”。

## 8. 正式命令

```powershell
.\.venv\Scripts\python -m online_speculation.hf_stream_uno `
  --model-path ..\.tmp_k2_horizon_09b `
  --adapter-path ..\.tmp_k2_horizon_09b_uno `
  --training-requests 4 --validation-repetitions 5 --test-repetitions 10 `
  --max-new-tokens 512 --warmup-tokens 16 --block-size 8 `
  --update-stride 40 --feedback-interval 4 --feedback-top-k 50 `
  --rank 8 --alpha 8 --learning-rate 0.005 `
  --selection-minimum-gain 0.002 --seed 20261005 --ignore-stop `
  --output .\online-speculation\results\stage6c_stream_uno1b_rtx3090_hf.json
```
