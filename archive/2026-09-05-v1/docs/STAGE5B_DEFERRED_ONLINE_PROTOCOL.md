# Stage 5B：Deferred Online-Uno 真机预注册协议

## 1. 要回答的问题

Stage 4B 的 immediate fast residual 在真实 `K2-Horizon-0.9B-Uno` 上造成了显著减速。Stage 5B 只检验
一个在查看正式数据前冻结的问题：

> 用过去窗口训练、经下一窗口真实 verifier rows 批准后才上线的 request-local residual，能否在 RTX 3090
> Hugging Face fallback 上提高 Uno 的 tokens-per-forward（TPF），并在计入全部在线成本后提高 decode TPS？

这是 post-verification rank-8 logit residual，不是在线更新整份 diffusion LoRA；结果也不能外推到论文的
H200、Nano-vLLM 或 Qwen3-8B 配置。

## 2. 冻结时间与 pilot 隔离

正式协议在 2026-09-05、读取正式三 prompt 结果前冻结。此前只在论文解释英文 prompt 上做了 2-seed
工程 pilot，用于确认控制器会发生 promotion 并选择成本参数。代码和中文 prompt 没有用于 Stage 5B
超参数搜索；英文正式结果因与 pilot prompt 重合，必须标注为 partially tuned workload。

不在正式运行后更改 stride、门限、rank、学习率或采样参数。若主门失败，后续任何新算法必须成为新的
编号阶段和新协议，不能替换本阶段结果。

## 3. 固定 checkpoint 与执行边界

| 项目 | 固定值 |
|---|---|
| base | `IFM/K2-Horizon-0.9B@ee770e713760cf6350e4322cdbbff91a163b7d70` |
| offline Uno | `IFM/K2-Horizon-0.9B-Uno@b0d8896a301a2f4bc755538b1234a35100da50d0` |
| device | 本机 RTX 3090 24 GiB，BF16 |
| backend | Transformers/PyTorch KV-cache fallback |
| block | 8（7 speculative + 1 free token） |
| decoding | temperature 1.0, top-k 50, top-p 0.95 |
| stop | `--ignore-stop`，每个 run 恰好 512 output tokens |

权重 SHA-256 由 runner 强制核验。base 和 offline Uno 全程冻结且不得进入 optimizer；online optimizer
只能持有 526,336 个 fast-head 参数。

## 4. 冻结算法配置

只比较两个方法：

- `static`：Stage 2 的 exact filtered linear $\Psi$-Spec；
- `deferred_s40`：相同 verifier 与 sampling，再加 future-validated candidate。

主方法固定为：

| 参数 | 值 |
|---|---:|
| update stride $S$ | 40 cycles |
| feedback interval $K$ | 4 cycles |
| candidate evaluation interval $J$ | 4 cycles |
| promotion margin | 0.0005 filtered TV |
| future reset margin | 0.005 filtered TV |
| supervision | on-policy，position discount 0.97 |
| residual | rank 8, alpha 8, AdamW lr 0.005 |
| feedback support | draft/target top-50 union |

候选在第 $S$ 轮后才由过去 feedback 训练；至少到第 $2S$ 轮，经过独立的 future window 后，才可能首次
promote。任意 cycle 的 accept/reject 必须使用该 cycle sampling 时保存的旧 $q_t$；本轮 verify 后发生的
训练或 promotion 只能影响未来 cycle。

## 5. 工作负载、配对与运行次序

固定三个 prompt：

1. 英文论文解释：lossless speculative decoding；
2. Python 工程代码：production-quality LRU cache、复杂度与边界测试；
3. 中文数学推导：Metropolis-Hastings 的 detailed balance、不可约性、非周期性与离散例子。

每个 prompt 做 5 个 paired repetitions，共 15 对、30 runs。每一对 static/deferred 使用相同 prompt、seed
和最大 token 数。方法顺序按 `(repetition + prompt_index) mod 2` 循环旋转，抵消单向热漂移。正式运行前
仅各做一次不计入统计的 static 和 immediate warmup。

## 6. 指标与统计

主算法指标：

$$
R_{TPF}=\frac{TPF_{deferred}}{TPF_{static}}.
$$

主系统指标：

$$
R_{TPS}=\frac{decode\ TPS_{deferred}}{decode\ TPS_{static}}.
$$

报告 15 个 paired ratios 的均值和 30,000 次 percentile bootstrap 95% CI，并同时报告：

- 每个 prompt 的 paired ratio；
- acceptance-rate delta、peak-memory delta；
- exact two-sided sign test（正收益 pair 数）；
- promotion/reject/reset 次数和三方 future filtered TV；
- feedback、update、active head、candidate head 的显式 wall-clock 占比；
- 未被细分计时覆盖的 residual overhead。

15 对只覆盖 3 个固定 workload family，不能当作独立任务总体的广泛统计样本；因此必须展示 prompt 分层，
不能只给 pooled 数字。

## 7. 预先定义的判定门

1. **安全门**：30/30 长度为 512；无 NaN；hash、routing、cache frontier 全通过；每个 deferred run
   满足 base optimizer overlap = 0、trainable base = 0、fast trainable parameters = 526,336；active-head
   evaluation cycles + static-skip cycles = total cycles；promotion 计数守恒。
2. **算法门**：$R_{TPF}$ 的 pooled paired-bootstrap 95% CI 下界严格大于 1。
3. **系统门**：$R_{TPS}$ 的 pooled paired-bootstrap 95% CI 下界严格大于 1。
4. 点估计大于 1 但区间跨 1，只写“方向有益但证据不足”；区间上界低于 1，写“该门明确失败”。
5. 英文单项不能覆盖代码/中文的负结果；若 prompt 方向不一致，必须明确报告异质性。

只有安全门、算法门和系统门都通过，才能称本阶段在 RTX 3090 HF fallback 上实现净 online 加速。

## 8. 正式命令

```powershell
.\.venv\Scripts\python -m online_speculation.hf_online_uno `
  --model-path ..\.tmp_k2_horizon_09b `
  --adapter-path ..\.tmp_k2_horizon_09b_uno `
  --block-size 8 --update-strides 40 `
  --max-new-tokens 512 --warmup-tokens 16 --repetitions 5 `
  --prompt "Explain in three concise paragraphs why speculative decoding can be lossless." `
  --prompt "Implement a production-quality Python LRU cache from scratch. Explain the invariants, analyze complexity, and include tests for edge cases." `
  --prompt "请严格推导为什么 Metropolis-Hastings 算法以目标分布为平稳分布，说明 detailed balance、不可约性与非周期性的作用，并给出一个离散状态空间例子。" `
  --feedback-top-k 50 --supervision on_policy --rank 8 --alpha 8 `
  --learning-rate 0.005 --activation-mode deferred `
  --feedback-interval 4 --candidate-evaluation-interval 4 `
  --promotion-margin 0.0005 --future-reset-margin 0.005 `
  --bootstrap-samples 30000 --ignore-stop `
  --output .\online-speculation\results\stage5b_deferred_online_uno1b_rtx3090_hf.json
```
