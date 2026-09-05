# Stage 10C：首请求 Causal Replay 工程 Pilot

运行日期：2026-09-05。原始结果：

- `results/stage10c_causal_replay_uno1b_rtx3090_hf_pilot.json`：`min_suffix=8`、宽 reliability bucket；
- `results/stage10c_causal_replay_uno1b_rtx3090_hf_min16_pilot.json`：保守 `min_suffix=16` 消融。

## 问题和结论

本阶段问的是：不依赖旧请求，在**第一个请求内部**只使用已经由 target verifier committed 的过去 token，
能否让 Uno 少做 forward 并获得净吞吐收益？

答案分两层：

- 算法路径成立。自然回答中出现的长 repetition 可被 request-local overlay 捕获；`min_suffix=16` 时 mean
  TPF 提高 4.75%，且三个 workload 上没有 TPF 退化。
- 系统收益未成立。保守方案的 aggregate decode TPS ratio 为 1.0027，95% 区间
  `[0.9687, 1.0396]`；自然 repetition 自身也只有 0.9986 `[0.9922, 1.0051]`。这仍是“有时少 forward，
  但当前短请求/HF backend 未确认 wall-clock 加速”。

因此 causal overlay 保留为 exact-repeat/global replay 的补充，而不替代高收益的跨请求路径。正式协议采用
`min_suffix>=16` 的保守下界，并要求 controller 可以零损回退。

## 设计

三个 workload、每个 3 个 paired Uno noise seeds，每对都重新创建空 global cache：

1. `natural_answer`：普通 speculative-decoding 解释；该 0.9B 模型自然产生了长段重复；
2. `explicit_repetition`：要求逐行重复一句话，但模型在 128-token budget 内主要复述指令，并未形成可命中段；
3. `code_template`：六个相同 body 的函数；`min_suffix=8` 出现短上下文错误命中。

共同设置：greedy、$B=8$、128 output tokens、future/global cache 为空、request-local overlay 在结束后 discard、
static/causal 交替顺序、10,000 bootstrap samples。增量 indexing 位于 decode critical path；discard close 不再
无意义索引 tail。

## Router 改进

第一版按精确 suffix length 建 bucket。自然回答的命中长度是 9/10、17/18、25/26、32；代码错误命中是
8/10，导致每个相邻长度都被当成新 bucket 重复探索。

新增 `match_length_bucket_width` 后，`[8,39]` 或 `[16,47]` 共享一个 replay TPF EMA：

- 自然 workload：第一次高收益 replay 后，后续命中由 `exploit` 执行；
- code workload：第一次 immediate-rejection 后，第二个相邻长度命中改为 `below-margin` fallback；
- route state 仍只由已完成 cycle 更新，当前 outcome 不能回溯改变本轮 proposal。

这把 `min_suffix=8` 的 code replay 从约 1.67 cycles/request 降至 1；但一次探索仍会令 code TPF 约
0.9869。进一步把最小 suffix 提到 16 后，code 的全部短错误命中被拒绝，TPF 精确回到 1。

## 结果

### Aggregate

| 配置 | TPF ratio mean [95% CI] | decode TPS ratio | inclusive E2E TPS ratio |
| --- | ---: | ---: | ---: |
| min suffix 8 | 1.02534 [0.99711, 1.05774] | 0.97963 [0.96297, 0.99788] | 0.97989 [0.96266, 0.99793] |
| min suffix 16 | 1.01584 [1.00000, 1.03744] | 1.00267 [0.96873, 1.03957] | 1.00311 [0.97067, 1.04059] |

### `min_suffix=16` 分 workload

| Workload | mean replay cycles | TPF ratio | decode TPS ratio |
| --- | ---: | ---: | ---: |
| natural answer | 3.0 | 1.04751 [1.01266, 1.09091] | 0.99863 [0.99224, 1.00514] |
| explicit repetition/no hit | 0 | 1.00000 | 1.04797 [0.97076, 1.11602] |
| code template/no hit | 0 | 1.00000 | 0.96141 [0.92172, 1.01227] |

两个 no-hit workload 在 TPF 上与 static 完全相同，说明执行路径正确回退；TPS 的相反摆动远大于约数毫秒的
lookup/index 成本，反映 3-pair 短测量的系统噪声，不能解释为算法加速或稳定退化。WSL 下载进程也在后台
进行，因此这批 wall-clock 只用于工程筛选。

## Exactness 审计与 BF16 边界

两份 pilot 的 9/9 causal outputs 都与各自 paired static Uno 128 token IDs 完全相同，新 overlay 没有改变
既有 verifier target path。`explicit_repetition` 的 static Uno 与单-token AR 在第 3 个 token 出现固定差异：
AR 选 `4824`（“says”），block Uno 选 `10076`（“wants”），之后轨迹相应变化。causal 与 static 都选择后者。

这是 static Uno 已存在的 BF16 kernel-shape 数值边界，而不是 replay 更新错误：一个 token 形状的 AR forward
与八 token block verification 可因 reduction/attention 舍入差异翻转接近的 greedy argmax。数学上的
$\Psi$-Spec 仍保持 target distribution；但“bitwise 等于另一个 kernel shape 的 greedy AR”比论文的分布
lossless 主张更强，实际系统不能把二者混作一个门。

所以结果文件明确拆成：

- 主实现门：`all_causal_static_token_ids_equal=true`；
- 数值可移植性诊断：`all_static_ar_token_ids_equal=false`；
- 若未来需要 bitwise AR，必须统一 target kernel/precision，不能用隐藏容差把差异抹掉。

## 下一步

1. 首请求路径固定 `min_suffix=16`、宽 reliability bucket，不再从这 9 个 pairs 调参。
2. 正式 test 增加更多 repetitions，且等 WSL 安装/后台下载结束后再测 wall clock。
3. 跨请求方向进入 near-repeat 与 mixed-domain：cache 用 temporal train requests 填充，validation 冻结
   suffix/controller，test 不再更新。
4. WSL 官方 runtime 接入 CUDA-event utility；TPF-only router 仅作为安全先验。

