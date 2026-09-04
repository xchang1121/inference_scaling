# Stage 6A：跨请求 Stream-Uno 设计与持久状态接口

## 1. 为什么从 request-local 转向 request stream

Stage 4B/5B 已经排除了一个重要但过强的假设：在单个 384--512 token 请求里从 zero fast weights 冷启动，
不一定有足够 horizon 学习、验证、上线并摊薄 backward。Stage 5B 中首次 promotion 最早在 cycle 80，15 个
请求总计只有 34.87% cycles 真正使用非零 active head，最终 TPF 主统计仍为负方向。

已有工作的在线单位并不相同：

- [Online Speculative Decoding](https://proceedings.mlr.press/v235/liu24y.html) 持续利用观察到的用户 query
  更新一个或多个 draft，并用 buffer threshold 批量训练；其适应目标是 query distribution，而不是每个
  query 都重新冷启动；
- [OnlineSPEC 官方实现](https://github.com/ZinYY/OnlineSPEC) 的 Ens-EAGLE 示例按 chunk 处理流数据，维护
  多个学习率 learner；这支持把 learner state 当成 serving-domain 状态；
- [Test-Time Speculation](https://arxiv.org/abs/2605.09329) 报告收益随生成长度增加，说明短 horizon 与
  update amortization 必须单独测量。

所以 Stage 6 的新假设是：

> 同一 domain/session 的过去请求提供的 verifier feedback，能否改善未来、未参与训练的请求；训练成本能否
> 跨多个未来请求摊销？

这不是把 Stage 5 的失败隐藏到离线训练。每次 training request 的 feedback 都在真实 serving 顺序中产生，
learner 只允许使用过去数据；正式评价会明确报告训练请求数和 break-even future requests。

## 2. 状态与时间索引

对 domain $d$ 维护一份小状态：

$$
S_d^{(r)}=(\delta_d^{(r)},m_d^{(r)},v_d^{(r)},n_d^{(r)}),
$$

其中 $\delta$ 是 rank-8 residual，$m,v$ 是 AdamW moments，$n$ 是已消费 feedback 数。第 $r$ 个请求开始
时只读 $S_d^{(r)}$；该请求第 $t$ 个 speculative cycle 实际使用并保存：

$$
q_{r,t}=q_{\phi+\delta_{r,t}}(\cdot\mid h_{r,t}).
$$

verify/accept/reject 完成后才允许更新 $\delta_{r,t+1}$。请求结束得到 $S_d^{(r+1)}$，供未来请求使用。
因此跨请求持久化不改变 Stage 1 的条件正确性证明：只要每轮 verification 分母仍是该轮采样时保存的
$q_{r,t}$，$\delta$ 可以是任意过去 history/feedback 的函数。

## 3. Stage 6A 实现边界

`HfOnlineUnoRunner.generate(..., persistent_learner=...)` 现在允许调用者显式传入并保留 learner：

- 不传时保持 Stage 4/5 的 fresh request-local 行为；
- 传入时验证 fast config、hidden size、vocabulary、device 和 optimizer ownership；
- 只允许 `immediate` mode，避免 Stage 5 deferred 的局部 candidate rebind 被误认为已写回外部状态；
- diagnostics 记录是否复用、请求开始的 fast-weight L2 和请求结束 L2；
- 初始 learner 非零时不能错误跳过 active head；base/Uno 仍全冻结。

这一步只提供可审计的持久状态 plumbing，不预先声称跨请求会提升 TPF。

## 4. Pilot 与正式实验必须分开

工程 pilot 可以回答：训练几次后 residual 是否仍持续非零、held-out seed 上 TPF 是否有可见方向、哪一类
domain 值得进入正式协议。正式实验则必须在查看 test split 前冻结：

```text
train stream       validation stream       test stream
past requests  ->  选择/拒绝 checkpoint ->  只评估一次冻结策略
```

- train/validation/test 使用互不重叠的 seeds；
- 若探索 prompt 模板，test 至少包含未逐字出现在 train 中的同域模板；
- checkpoint 只能由 validation TPF/filtered-TV 选择；
- test 同时跑相同 seed 的 static Uno 与 frozen persistent residual；
- 训练过程、validation 选择和失败 checkpoint 都保留，不能只保存最好 test run；
- Stage 4/5 的 request-local 方法作为诊断对照，但不能反向进入 checkpoint 选择。

## 5. 两个系统指标

### 冻结 serving 收益

先看训练完成后未来请求的 paired serving time：

$$
\Delta T_{serve}=T_{static}-T_{persistent,frozen}.
$$

frozen 评价不做 feedback materialization/backward，只支付 active residual head；这是 domain adapter 已在线
学成后的 steady-state serving 成本。

### 在线训练摊销

令所有 training requests 相对 static 的增量成本为 $C_{train}$，每个未来请求平均节省为
$\overline{\Delta T}_{serve}>0$，则保守 break-even 请求数：

$$
N_{BE}=\left\lceil
\frac{\max(0,C_{train})}{\overline{\Delta T}_{serve}}
\right\rceil.
$$

分别用 observed paired wall-clock 增量和 instrumented feedback/update/head 时间计算两个版本。若 serving
没有正节省，则 break-even 定义为不存在，而不是报负数或无穷大伪装成功。

## 6. 风险与下一分支

- domain 混流会导致 stale/catastrophic adaptation，因此状态必须按 domain route，或保留 zero anchor；
- 同一 prompt 不同 seed 的泛化弱于同域不同 prompt，必须分层报告；
- 直接持久化 immediate learner 仍可能把有害 update 带入未来请求；若 validation 无法筛掉，Stage 6B
  将改成 active/static/candidate 的小权重 mixture；
- 单张 3090 上 training 与 serving 串行，不能借用 OSD 的 spare-FLOPs 假设；所有成本仍实测；
- rank-8 logit residual 成功不等价于完整 diffusion LoRA 在线训练成功。

Stage 6A 的阶段门只有：状态确实跨请求连续、第二个请求的 initial L2 等于第一个请求 final L2、optimizer
仍只拥有 fast head、fresh 默认路径回归不变。通过后才写 stream benchmark harness。

## 7. Stage 6B harness

`hf_stream_uno.py` 已把上述切分变成一个不可混用的执行流程：

1. training requests 中 persistent learner 边生成边更新，并与同 seed static 配对计时；
2. 保存 zero snapshot 和每个 training request 后的 learner/optimizer snapshot；
3. validation seeds 上只做 frozen proposal，按 mean TPF ratio 选 checkpoint，低于固定 gain threshold 则回退
   zero snapshot；
4. 释放未选快照后，在独立 test seed 区间只评估 selected snapshot 一次；
5. 输出 frozen-serving TPF/TPS、observed training increment、instrumented online cost 和两个 break-even。

训练、验证、测试 seed 分别使用 `seed+[0, 100000, 200000]` 区间。zero snapshot 在 validation 上必须与
static TPF 精确相等，否则说明两个执行路径或 RNG 语义不一致，结果不得继续解释。snapshot selection 与
break-even 的纯函数回归测试已经覆盖 threshold、tie、fallback 和无正 savings 情形。

单域工程 pilot 选中 snapshot 3，两个未见 test seed 的 TPF mean ratio 为 `1.01056`、TPS mean ratio 为
`1.02038`；这只是进入正式实验的信号。新的 stationary stream 配置、全新 seed 分区和严格判定门已在
[`STAGE6C_STREAM_UNO_PROTOCOL.md`](STAGE6C_STREAM_UNO_PROTOCOL.md) 中冻结。
