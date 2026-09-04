# Stage 4A：真实模型 Online Uno fast residual 设计

Stage 3 证明 verifier feedback 在 drift 下有用，也证明“每轮更新完整 adapter”很可能不回本。Stage 4A
因此先实现一个与模型解耦、可严格审计的 request-local low-rank logit residual，而不是直接对公开
rank-128 Uno adapter 反向传播。

## 结构

Uno draft forward 的第 $i$ 个 noisy row 给出 frozen hidden state $h_{t,i}$ 和 offline logit
$\ell^0_{t,i}$。fast head 为

$$
\Delta\ell_{t,i}
=\frac{\alpha}{r}B_tA_t h_{t,i},
\qquad
q_{t,i}=\operatorname{FilterSoftmax}
(\ell^0_{t,i}+\Delta\ell_{t,i}).
$$

默认 $r=8,\alpha=8$。$A\in\mathbb R^{8\times1536}$、
$B\in\mathbb R^{64256\times8}$，约 0.53M 参数；$B$ 从零初始化，所以启用 fast head 的第一个 token
与 static Uno 完全相同。hidden state 在 `torch.inference_mode()` 的模型 forward 后 detach，梯度图只包含
这个小 head。

优化器必须满足集合恒等式：

$$
\operatorname{IDs}(\mathrm{optimizer})
=\operatorname{IDs}(A,B),
\qquad
\operatorname{IDs}(\mathrm{optimizer})
\cap\operatorname{IDs}(\theta_{AR},\phi_{Uno})=\varnothing.
$$

运行前同时要求所有 base/Uno parameter 的 `requires_grad=False`。任一条件不满足直接报错。

## Top-K union surrogate

真实 sampling 使用 top-k/top-p，直接对离散 top-k 集求导不稳定。每个 verifier row 固定一个 support：

$$
U_{t,i}=\operatorname{TopK}(\ell^0+\Delta\ell)
\cup\operatorname{TopK}(\ell^{V}).
$$

在同一 $U_{t,i}$ 上分别归一化 verifier、旧 draft 和当前 draft。replay item 只保存 detached hidden、
token IDs、这些 token 的 offline logits、target probabilities 和旧 draft probabilities；默认每行至多 100
个 token，不保存 64K dense probability。训练目标沿用预注册形式：

$$
\mathcal L=
D_{KL}(p_U\Vert q_U)
+0.5D_{TV}(p_U,q_U)
+0.15D_{KL}(q^{old}_U\Vert q_U)
+10^{-6}\|B\|_2^2.
$$

这不是 full-vocabulary TV 的无偏估计，因此正式结果必须同时报告 surrogate loss 与真正 filtered
acceptance/TPF；后者才决定算法收益。

## Transactional update

每个 stride buffer 以固定 index 拆成 train/validation：每五条的第一条只做 validation，其余训练。
一次 update 的顺序为：

1. 在 validation 上比较 current fast head 与 zero/static shadow；current 恶化超过 5% 时先清零 fast
   weights 和 optimizer state；
2. 同时 snapshot fast head 与 AdamW state；
3. 一次 update，global gradient norm clip 1.0；
4. validation objective 若非有限或比 update 前恶化超过 1%，恢复参数和 optimizer snapshot；
5. 只在 verification 完成后执行，当前轮仍永久使用保存的旧 filtered $q_t$。

这一 rollback 是安全护栏，不等同于无偏的在线 model selection。真实 benchmark 还要报告 reset/rollback
频率；若多数 update 被 rollback，说明 support、学习率或 validation split 不适合。

## 与 Stage 3 stale-weight 失败的对应

fast learner 提供两种恢复操作：

- `decay_toward_offline(f)`：$\delta\leftarrow f\delta$；
- static-shadow reset：如果近期 verifier feedback 表明 zero residual 更好，直接回到 offline snapshot。

它们只能在下一轮生效，不会改变当前轮的 lossless verifier。后续 controller 将比较 `no-update`、`update`
和 `reset` 三个动作的边际价值，而不再把已有 fast-weight 收益错误解释为继续高频更新的理由。

## Stage 4A 阶段门

- zero-init head 对 logits 逐元素无影响；
- feedback 必须 detached，draft/target top-k union 完整；
- 合成数据上 held-out loss 降低；冲突 train/validation 必须触发完整 rollback；
- stale head 必须能由 static shadow reset；
- decay 数学精确；
- optimizer/base 参数集合白名单测试通过。

通过后再接入 K2-Horizon-0.9B-Uno 的 HF KV-cache fallback。Stage 4B 才会测真实 feedback
materialization、head forward、backward/optimizer、TPF、tokens/s 和峰值显存。
