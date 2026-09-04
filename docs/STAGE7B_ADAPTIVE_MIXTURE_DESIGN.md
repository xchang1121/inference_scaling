# Stage 7B：Verifier-feedback adaptive mixture

## 1. 为什么固定小权重仍不够

Stage 7A 之后先做了一个**不进入正式结论**的固定 `w=0.25` 工程 pilot：同一英文 prompt，4 个 train
requests、2 个 validation seeds、5 个 test seeds、每次 512 tokens，base seed 为 `20262005`。validation
选择 snapshot 3，mean TPF ratio 为 `1.05936`；但 5 个新 test ratios 是：

```text
0.9899, 0.9118, 0.9712, 1.0547, 1.0471
```

mean TPF ratio 只有 `0.99493`。TPS mean ratio 是 `1.03453`，但 5 次的范围为
`[0.96597, 1.08748]`，不足以支持稳定系统收益。这个 pilot 说明小权重确实压缩了 hard activation 的风险，
却仍会在 candidate 对当前 trajectory 有害时机械地使用它。因此 Stage 7B 不扫描更多固定权重，而让已经计算的
verifier distribution 决定**下一轮**是否启用一个有上限的 candidate mixture。

## 2. 因果顺序与 proposal

令 frozen Uno proposal 和跨请求训练后但在当前请求中冻结的 residual proposal 分别为

$$
q_0(\cdot\mid h_t),\qquad q_\delta(\cdot\mid h_t).
$$

控制器只选两个权重：

$$
w_t\in\{0,w_{\max}\},\qquad
q_t=(1-w_t)q_0+w_tq_\delta.
$$

其中 $w_t$ 必须是历史 $\mathcal F_{t-1}$ 的函数。第 $t$ 轮严格按以下顺序执行：

1. 读取进入本轮前已经确定的 $w_t$；
2. 从完整的 sparse $q_t$ 采 proposal，并保留这个对象；
3. verifier 计算 target $p_t$，exact $\Psi$-Spec 用保存的同一 $q_t$ 验收/修正；
4. 只在验收完成后更新控制器，得到 $w_{t+1}$。

因此任何本轮 verifier 信息都不能反过来改变本轮 acceptance denominator。即使 controller 在第 $t$ 轮决定
activate/deactivate，动作也只影响下一轮；生成分布仍严格等于 target AR 分布。

## 3. Verifier evidence

每隔 $K$ 个 speculative cycles，同时评价两个 frozen experts。对本轮第 $i$ 个 speculative row，定义

$$
\ell_{0,t,i}=D_{TV}(p_{t,i},q_{0,t,i}),\qquad
\ell_{\delta,t,i}=D_{TV}(p_{t,i},q_{\delta,t,i}).
$$

只使用真实 on-policy prefix：若在 row $r$ 首次拒绝，则 $a_{t,i}=1$ 当且仅当 $i\le r$；若全部接受，
所有 row 权重均为 1。一次观测是

$$
\Delta_n=
\frac{\sum_i a_{t,i}(\ell_{0,t,i}-\ell_{\delta,t,i})}
     {\sum_i a_{t,i}}.
$$

$\Delta_n>0$ 表示在**刚刚实际访问并验证**的 contexts 上 candidate 比 static 更接近 target。控制器保存

$$
\bar\Delta_n=
\begin{cases}
\Delta_1,&n=1,\\
\beta\bar\Delta_{n-1}+(1-\beta)\Delta_n,&n>1.
\end{cases}
$$

在至少 $M$ 次观测后采用带滞回的门：

$$
w_{t+1}=
\begin{cases}
w_{\max},&w_t=0\ \land\ \bar\Delta_n>m_{\rm on},\\
0,&w_t>0\ \land\ \bar\Delta_n<-m_{\rm off},\\
w_t,&\text{otherwise}.
\end{cases}
$$

首版固定 `w_max=0.25`、`K=4`、`M=2`、`beta=0.75`、`m_on=m_off=0.0005`。这些是工程 pilot
参数，不是读取正式 test 后选择的超参数。

## 4. 计算路径

- 请求开始时 controller 总是 reset 到 `w=0`，不会把某个 sampling seed 的短期门状态带到下个请求；
- `w=0` 且不在评价 cycle 时，完全跳过 residual head 和第二次 top-k/top-p；
- `w=0` 的评价 cycle 只 shadow-evaluate candidate，本轮仍从 static proposal 采样；
- `w=w_max` 时每轮计算 static/candidate 两个独立过滤分布，再在概率空间混合；
- diagnostics 保存每次 `instantaneous_advantage`、EMA、前后权重、动作、非零 cycle 数、平均权重和 controller
  wall time；
- stream test 前后的 residual-head SHA-256 必须相同，证明 controller 只更新标量状态。

## 5. 当前训练边界

Stage 7B 与 Stage 7A 一样只开放 frozen persistent learner。当前请求的 `feedback_interval` 和
`update_stride` 必须都大于最大输出长度，且 activation mode 必须是 immediate。原因是 fast-residual replay
中的 old-$q$ regularizer 仍以 pure candidate logits 表示；在 mixture proposal 下直接 backward 会把
$q_\delta$ 错当成实际 $q_t$。在实现 mixture-aware replay 前，runner 对这种组合 fail closed。

## 6. 已知局限与可证伪点

这个 controller 只知道当前访问 contexts 上的 one-step filtered TV，仍没有 sequence-level oracle。inactive
时的 shadow candidate 是在 static trajectory 上评价的；activate 后 trajectory 会变化，所以 $w_{\max}$ 的
上限仍不可省略。EMA 也可能因短 horizon、高方差或非平稳 domain 滞后。Stage 7B 的问题不是“是否出现过一次
activate”，而是：

- zero snapshot 是否逐 token/forward 与 static 完全一致；
- 非零 snapshot 是否在独立 test seeds 上提高 mean TPF；
- controller/head/optimizer 安全审计是否全部通过；
- 计入 shadow head、双过滤和控制器开销后，TPS 是否仍提高。

## 7. 实现阶段门与下一实验

实现提交前必须通过：

- controller 的 warmup、EMA、activate/deactivate 和非法输入单测；
- fake runtime 中 controller 只在预定 cycle 观察，head 参数逐 tensor 不变；
- zero snapshot 的 adaptive 输出 token、forward 数与 static 完全相同；
- adaptive 与固定非单位 mixture 同时启用时拒绝运行；
- 全部既有 request-local、deferred、stream 和 exact sampler 测试回归通过。

实现提交后使用新 seed 做 4-train/2-validation/5-test 工程 pilot。只有出现足够稳定的正方向，才另写并先提交
Stage 7C 的正式预注册协议；pilot 数据不能直接升级为正式结论。
