# Stage 7A：Static-anchored probability mixture

## 1. 目标

Stage 6C 的 hard residual 在 10 个 test seeds 上从 `+7.07%` 到 `-11.00%` TPF，说明 candidate 不是始终
无用，而是 trajectory 风险太大。Stage 7 不再把 candidate 当成全有或全无的模型，而令实际 proposal 为：

$$
q_w(x\mid h)=(1-w)q_0(x\mid h)+wq_\delta(x\mid h),\qquad w\in[0,1].
$$

$q_0$ 是 frozen Uno logits 独立经过 temperature/top-k/top-p 后的 sparse distribution，$q_\delta$ 是同一
hidden state 加 residual 后独立过滤的分布。**先过滤再混概率**与“插值 logits 后过滤”不是同一个算法；
后者会非线性改变 top-k/top-p support，不能使用本页的凸性结论。

## 2. 单 context 的风险界

由于 total variation 对第二个变量凸：

$$
\begin{aligned}
D_{TV}(p,q_w)
&\le (1-w)D_{TV}(p,q_0)+wD_{TV}(p,q_\delta),\\
\operatorname{Overlap}(p,q_w)
&\ge (1-w)\operatorname{Overlap}(p,q_0)
 +w\operatorname{Overlap}(p,q_\delta).
\end{aligned}
$$

若 candidate 在当前 row 比 static 差 $\Delta$，mixture 的 TV 增量至多 $w\Delta$；若更好则保留至少线性
插值下界。这个结论只针对相同 $h$ 的单 row。mixture 会改变采样 token 和未来 context，因此不能把它
误写成整条生成 trajectory 的 worst-case guarantee；Stage 7 仍必须做 held-out sequence 实验。

## 3. Sparse 实现

两个 top-50 distribution 的 support 可能不同。实现直接连接两组 `(token_id, probability)`：

```text
ids   = [ids_static, ids_candidate]
probs = [(1-w) p_static, w p_candidate]
```

相同 token 可出现两次。categorical sampling 把两项采成同一 token，`probability_of`、filtered overlap 和
residual correction 都会对匹配 id 求和，所以其总质量与显式 union/coalesce 完全相同；避免了每 row
变长的 unique/scatter kernel。`w=0/1` 直接返回原 distribution。

实际 `spec_tokens` 从 $q_w$ 采样，verifier 的 acceptance denominator 和 rejection residual 都持有同一份
$q_w$ 对象。因此 mixture 不引入近似 sampling，仍严格服从 target AR 分布。

## 4. 当前训练边界

非单位 mixture 暂时只允许 frozen persistent learner：

- `persistent_learner` 必须显式提供；
- activation 必须是 immediate 路径；
- `feedback_interval` 和 `update_stride` 必须大于最大输出长度，确保本请求不创建 replay、不 backward；
- diagnostics 保存实际 `proposal_mixture_weight`；
- test 前后 head hash/L2 仍必须不变。

限制的原因不是 exact verifier，而是现有 fast loss 的 old-$q$ regularizer 从 candidate logits 构造；当实际
proposal 是 mixture 时，它不再等于旧 $q_w$。在实现 mixture-aware replay 前禁止一边 mixture sampling
一边沿用错误 old-$q$ loss。

## 5. Stage 7A 测试门

- 不同 sparse support 的 mixture token mass 精确等于手算值；
- mixture TV 不超过凸组合上界；
- frozen fake-model 请求无 feedback/update，head L2 不变；
- 非单位 mixture + 同请求学习必须 fail closed；
- `w=1` 的旧 request-local/deferred/stream 路径全部回归通过；
- 真机 pilot 只用于选择一个后续固定 $w$，不能作为正式 test 结论。

Stage 7B 先在全新工程 seed 上检查固定小权重（优先 $w=0.25$）。若它只把退化缩小到 0 而没有正信号，
下一步是根据过去 verifier overlap 更新**标量** $w_t$，而不是改回 hard switch。
