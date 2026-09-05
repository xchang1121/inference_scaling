# Stage 8：Greedy repeated-query Stream-Uno 正式结果

## 1. 一句话结论

在严格限定的 greedy repeated-query workload 上，跨请求 persistent residual **通过了在线学习门**：20 个新
Uno noise seeds 的 mean TPF ratio 为 `1.00950`，95% CI `[1.00268, 1.01621]`，且点估计超过预注册
`+0.5%` 实际幅度门。它**没有通过系统门**：mean decode TPS ratio 为 `1.00428`，95% CI
`[0.98711, 1.02319]`。

准确表述是：

> 4 个历史 greedy 请求中的 verifier feedback 训练出的 rank-8 logit residual，在 20 个未见 Uno noise seeds
> 上显著减少了相同 512-token target 所需的 decoder forwards；当前 Windows HF fallback 尚未证明净
> wall-clock 加速。

这不是完整 Uno diffusion LoRA 在线训练，也不能外推到 stochastic sampling、新 prompt 或官方 runtime。

## 2. 预注册门

| 门 | 判据 | 结果 |
| --- | --- | --- |
| 安全/exactness | 全部审计通过 | **通过** |
| 非零选择 | validation 选择 snapshot > 0 | snapshot 2，**通过** |
| 学习统计 | mean TPF 95% CI low > 1 | 1.00268，**通过** |
| 学习幅度 | mean TPF ratio >= 1.005 | 1.00950，**通过** |
| frozen system | mean TPS 95% CI low > 1 | 0.98711，**失败** |
| 总成功 | 上述全部通过 | **失败** |

因此 `greedy_online_learning_success=true`，`all_stage8_gates_pass=false`。没有把不确定的 TPS 点估计包装成
系统加速。

## 3. Validation-only 选择

正式 base seed 为 `20262405`。4 个 training requests 后保留 snapshot 0--4；5 个 validation seeds 的 mean
TPF ratios 为：

| snapshot | 0 | 1 | 2 | 3 | 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| mean ratio | 1.00000 | 0.98811 | **1.00818** | 0.99302 | 1.00008 |

选择规则因此部署 snapshot 2。正式 test mean `1.00950` 反而略高于 validation，optimism gap 为
`-0.00132`；没有出现 Stage 6C 的大幅 validation-to-test 反转。

## 4. Held-out test 效果

### 4.1 算法效率

| 指标 | mean（95% CI） | median（95% CI） |
| --- | --- | --- |
| TPF ratio | `1.00950 [1.00268, 1.01621]` | `1.00760 [1.00000, 1.02046]` |
| acceptance-rate delta | `+0.00977 [+0.00297, +0.01652]` | `+0.01046 [0, +0.01917]` |

20 对中 TPF 有 12 wins、3 losses、5 exact ties；排除 ties 的双侧 exact sign-test 为 `p=0.03516`。
static 平均 198.80 cycles，online 平均 196.95 cycles，即每个 512-token request 平均少 1.85 个 speculative
cycles。效果不大，但同时通过 bootstrap 下界和预注册实际幅度门。

### 4.2 Wall clock

| 指标 | mean（95% CI） | median（95% CI） |
| --- | --- | --- |
| decode TPS ratio | `1.00428 [0.98711, 1.02319]` | `0.99891 [0.98259, 1.00601]` |
| seconds saved/request | `+0.0542 [-0.2103, +0.3477]` | `-0.0176 [-0.2651, +0.0944]` |

TPS 为 9 wins / 11 losses，sign-test `p=0.82380`。平均 static/online decode time 分别为 15.641 s 和
15.587 s，但区间明显跨 0/1，因此不能称净提速。residual head 的 CUDA-event 时间平均只占 online decode
约 0.109%，新增峰值显存约 3.34 MB；目前无法仅凭这些 coarse timers 把 TPS 方差归因于 head matmul，
还需 profiler 或官方 fused runtime。

## 5. Exactness 与冻结审计

分析脚本检查了 78 个生成结果、49 个 persistent isolation records：

- 所有输出长度都是 512，所有数值有限；
- base/adapter revision、SHA-256 和 clean/noisy conditional routing 通过；
- `base_optimizer_overlap=0`、trainable base tensors 为 0、fast trainable params 为 526,336；
- 4 个 training requests 的 initial/final L2 连续；
- 25 个 validation frozen runs 和 20 个 test frozen runs 均无 feedback/update，initial/final L2 相同；
- zero snapshot 的 5 个 validation TPF ratios 全部精确为 1；
- selected head test 前后 SHA-256 都是
  `dbf347a49547354ac8d64b14d62adce12cf79183c05c2b317b54f82d60292456`；
- 所有 static/online 配对的 `output_token_ids` 逐 token 相同；
- 全部 train/validation/test noise seeds 最终都得到同一条 greedy target sequence；
- seed 分区、交替顺序、validation rotation 和 snapshot 选择均由分析脚本重算通过。

最后两项说明 fast head 只改变 speculative 工作量，没有改变 greedy AR 结果。一般 stochastic exactness 仍由
保存实际旧 $q_t$ 的 $\Psi$-Spec 证明和 Stage 1 Monte Carlo 检验承担。

## 6. 在线训练成本

4 个 training requests 共进行 17 次 update attempts：16 次应用、1 次 rollback，并发生 3 次
same-buffer static reset。训练阶段：

- observed paired decode increment：`2.2188 s`；
- instrumented feedback/update/head cost：`0.5699 s`；
- 以不显著的 test mean `0.0542 s/request` 点估计算，break-even 分别为 41 和 11 个 future requests。

由于 frozen TPS 系统门失败，这些 break-even 只作为成本描述，不是已证实的生产回本承诺。

## 7. 为什么这个结果与 stochastic 失败不矛盾

Stage 6/7 的 stochastic requests 每个 seed 都访问不同 AR target trajectory；本轮 one-step verifier advantage
对下一轮的相关性很弱。Stage 8 的 top-k 1 target trajectory 固定，不同 seed 只改变 Uno noise，过去请求的
hidden/verifier pairs 因而更可复用。正式结果支持一个窄推断：

$$
\text{trajectory repeatability}
\quad\text{是当前 fast residual 能否跨请求泛化的重要条件。}
$$

它没有证明 rank-8 logit residual 是最终结构。相反，+0.95% TPF 仍不足以在当前 backend 上给出稳定 TPS；
更有价值的下一步是把已确认的算法增益迁移到低开销 gate/top-layer LoRA 或 fused kernel，并用新 seed 重新测
系统门，而不是继续在本结果上扫描学习率。

## 8. 可复现文件

- raw：`results/stage8_greedy_stream_uno1b_rtx3090_hf.json`；
- independent analysis：`results/stage8_greedy_stream_uno1b_rtx3090_hf_analysis.json`；
- 冻结协议：`docs/STAGE8_GREEDY_STREAM_PROTOCOL.md`；
- 分析入口：`python -m online_speculation.stage8_analysis`。

分析命令：

```powershell
.\.venv\Scripts\python -m online_speculation.stage8_analysis `
  --input .\online-speculation\results\stage8_greedy_stream_uno1b_rtx3090_hf.json `
  --output .\online-speculation\results\stage8_greedy_stream_uno1b_rtx3090_hf_analysis.json `
  --bootstrap-samples 50000 --seed 20262405
```
