# Stage 7B：Adaptive mixture 工程 pilot 结果

## 1. 结论

Stage 7 的两个 stochastic 工程 pilots 均未产生可推广的 TPF 收益：

| proposal | validation 选择 | 5-test mean TPF ratio | 解释 |
| --- | ---: | ---: | --- |
| fixed `w=0.25` | snapshot 3，`1.05936` | `0.99493` | validation 收益未推广 |
| verifier EMA adaptive | zero snapshot，`1.00000` | `1.00000` | 安全回退，但无学习收益 |

adaptive 方案达成了 fail-safe 目标：5 个 test requests 的 token/forward 指标与 static 完全相同，head hash
前后相同；但因为 validation 正确拒绝了所有非零快照，它不能被表述为“online adaptation 提速”。

## 2. 固定权重 pilot

固定权重使用 seed `20262005`，4 train / 2 validation / 5 test，512 tokens，`B=8`。validation 选择
snapshot 3，两个 validation seeds 的 mean TPF ratio 为 `1.05936`。新的 5 个 test ratios 为：

```text
0.9899, 0.9118, 0.9712, 1.0547, 1.0471
```

mean 为 `0.99493`，范围 `[0.91176, 1.05473]`。TPS mean `1.03453`，但范围同样跨 1；工程样本太小，
而算法 TPF 没有改善，所以不宣称系统成功。

## 3. Adaptive EMA pilot

adaptive pilot 使用全新 seed `20262105` 和相同请求矩阵。参数在运行前固定为：

```text
w_max=0.25, evaluation_interval=4, warmup=2,
ema_decay=0.75, activation_margin=deactivation_margin=0.0005
```

validation 的 mean TPF ratios 为：

| snapshot | 0 | 1 | 2 | 3 | 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| ratio | 1.00000 | 0.96885 | 0.99003 | 0.98994 | 0.99016 |

因此 validation-only 规则选择 zero snapshot。5 个 test 的 TPF ratios 全部严格等于 `1.0`；TPS ratio
mean `1.01817`、范围 `[0.99448, 1.03480]`，只是没有算法工作差异时的短配对计时波动，不能作为加速证据。

安全审计全部通过：zero snapshot L2 为 0，test 前后 head SHA-256 都是
`9c32ac6ea134604a1a373c5fc0d6ac83e3623daf8e0702d15695c36f320e1d1d`。

## 4. 为什么不继续调 EMA

非零 snapshots 的 verifier advantage 平均值均为正，但它对**下一次评价时刻**的预测很弱：

- lag-1 correlation 范围为 `[-0.170, 0.231]`；
- advantage 符号持续率范围为 `[0.464, 0.577]`；
- 每个请求内多次 activate/deactivate，仍得到低于 static 的 sequence TPF。

这说明失败不只是 margin 太小或 EMA 太快。本轮访问 context 上更低的 one-step TV 并不可靠预测下一轮
context，更不能抵消 mixture 改变 token 后的 trajectory shift。继续用同一批 test 扫 `beta/margin/w_max`
会转化为事后调参，因此本分支停止。

## 5. 复现命令与范围

```powershell
.\.venv\Scripts\python -m online_speculation.hf_stream_uno `
  --model-path ..\.tmp_k2_horizon_09b `
  --adapter-path ..\.tmp_k2_horizon_09b_uno `
  --training-requests 4 --validation-repetitions 2 --test-repetitions 5 `
  --max-new-tokens 512 --block-size 8 --update-stride 40 `
  --feedback-interval 4 --feedback-top-k 50 --rank 8 --alpha 8 `
  --learning-rate 0.005 --selection-minimum-gain 0.002 `
  --adaptive-mixture --mixture-max-weight 0.25 `
  --mixture-evaluation-interval 4 --mixture-warmup-observations 2 `
  --mixture-ema-decay 0.75 --mixture-activation-margin 0.0005 `
  --mixture-deactivation-margin 0.0005 --seed 20262105 --ignore-stop `
  --output <temporary-pilot-output.json>
```

机器可读摘要在 `results/stage7_engineering_pilots_summary.json`。原始 JSON 包含生成文本，保留在本机临时
实验目录而不提交；本页只报告聚合统计。两次运行都是工程 hypothesis-generation，不是预注册正式检验。

## 6. 下一步

Stage 8 把问题缩窄到 greedy target 的 repeated-query stream。`top-k=1` 时 target AR trajectory 固定，
不同 seeds 只改变 Uno 的 noise/proposal；若跨请求 residual 仍不能在 held-out noise seeds 上提高 TPF，便可排除
“主要失败来自 stochastic target trajectory shift”这一假设。TPF 学习门和包含 head 开销的 TPS 系统门仍分开。
