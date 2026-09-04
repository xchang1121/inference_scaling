# Stage 5A：Future-validated shadow candidate controller

Stage 4B 的主要失败不是 fast head 完全学不动，而是 same-buffer validation 不能预测下一段 context；s10
在 212 次 update 中即使多数局部 validation loss 下降，正式 TPF 仍低于 static。Stage 5A 把 parameter
update 和 serving activation 拆成两个事件。

## 三份状态

任意时刻维护：

- `active`：唯一参与 proposal sampling 的 fast head $\delta_t$；
- `candidate`：用过去窗口 feedback 从 active 克隆并训练的 $\tilde\delta_t$；
- `static`：始终为 zero residual 的 offline Uno shadow。

时间线为：

```text
window k:     active 生成并 exact verify
              -> 用 feedback 训练 candidate（不上线）
window k+1:   active 继续生成
              -> 在真正未来 verifier rows 上评估 active/candidate/static
              -> promote / keep / reset
              -> 再训练下一 candidate
```

candidate 从不进入产生当前 proposal 的 $q_t$，也不进入当前 acceptance denominator。因此无论 promote
规则是否有偏，只要 action 在本轮 verify 后执行，输出仍严格服从 AR target。

## 用实际 filtered overlap 批准

Stage 4 的训练 loss 是 raw-logit top-K union surrogate。Stage 5 不用它批准上线，而在未来 canvas 上计算
与采样完全相同的 temperature/top-k/top-p 分布：

$$
O_i(p,q)=\sum_x\min(p_i(x),q_i(x))=1-D_{TV}(p_i,q_i).
$$

只累计真实 accepted prefix 到首次 rejection row 的 on-policy 权重 $0.97^i$。窗口均值记为
$\overline{TV}_{active},\overline{TV}_{candidate},\overline{TV}_{static}$。默认动作：

$$
\begin{aligned}
\text{promote},&\quad TV_c+0.002<TV_a\ \land\ TV_c\le TV_0,\\
\text{reset},&\quad TV_0+0.005<TV_a\ \land\ TV_0<TV_c,\\
\text{keep},&\quad\text{otherwise}.
\end{aligned}
$$

promotion margin 抑制微小噪声，reset margin 防止在 active 与 static 几乎相同时抖动。每次决策保存未来
row weight、三方 TV 和 action，可审计是否真的在 out-of-time 数据上批准。

这仍是 active proposal 路径上的 shadow evaluation；candidate 若真的上线会采不同 token，故不是完整
counterfactual rollout。它比 same-buffer validation 更强，但正式文档不能称为无偏策略评估。

## 成本控制

训练窗口仍由 update stride 控制，但 feedback 不必每轮 materialize。新参数 `feedback_interval=K` 只在
每 $K$ 轮保存一次 hidden/top-K union；candidate 的未来 overlap 则利用 verification 已有 logits 每轮
累计。由此：

$$
C_{online}\approx C_{active\ head}+C_{candidate\ shadow}
+\frac{C_{feedback}}{K}+\frac{C_{clone+update}}{S}.
$$

Stage 5B pilot 将优先检查 `S=20/40, K=4`。第二个 head 的 full-logit matmul在 Stage 4 仅约 0.1% decode，
而 feedback 每轮约 1.2%；稀疏采集应比盲目减少 rank 更有效。

## 状态与 optimizer

candidate clone 同时复制：

- active head 参数；
- AdamW moment/step；
- 真正的 zero/offline anchor。

最后一点很重要：clone 的当前 active 参数不能被误当作“offline”。`reset_to_offline()` 必须仍回到全零
residual。promotion 后 candidate 的 optimizer state 一并成为 active，避免参数与 moment 不一致；reject
则整份 candidate 丢弃。

## Stage 5A 阶段门

- learner clone 与 active 内存独立，但共享同一个真正 zero anchor；
- sparse filtered overlap 对不同 support 给出精确 $1-TV$；
- synthetic future evidence 覆盖 promote/reset/keep 三个动作；
- 假模型端到端至少经历一次 candidate future validation，且决策计数守恒；
- immediate mode 的 Stage 4 路径仍通过原回归测试；
- base/Uno parameter isolation 与旧 $q_t$ 时序不变。

Stage 5B 才在真实 checkpoint 上比较 deferred 与 static；若只减少伤害而没有净加速，也必须照实报告。
