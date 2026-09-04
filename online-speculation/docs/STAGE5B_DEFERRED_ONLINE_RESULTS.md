# Stage 5B：Deferred Online-Uno 真机结果

## 结论先行

future-validated candidate controller 保住了 exact sampling 和参数隔离，但**没有通过预注册的学习门或
系统门**。在 RTX 3090、K2-Horizon-0.9B-Uno、Transformers/PyTorch KV-cache fallback 上：

| 主指标（15 paired runs） | estimate | paired bootstrap 95% CI | 判定 |
|---|---:|---:|---|
| TPF ratio，预注册 mean | 0.99503 | [0.98713, 1.00285] | 学习门失败 |
| decode TPS ratio，预注册 mean | 1.02194 | [0.99665, 1.04774] | 系统门失败，方向为正但证据不足 |
| acceptance-rate delta，mean | -0.00397 | [-0.01169, 0.00423] | 跨 0 |
| peak-memory delta，mean | +24.21 MiB | [+21.91, +26.33] MiB | 增加 |

通用 runner 先输出的是中位数 bootstrap，而预注册协议写的是配对均值。独立分析忠实地以 mean 为主统计，
并保留 median 作为稳健性检查：TPF median ratio 为 `1.00000 [0.98446, 1.00000]`，TPS median ratio 为
`1.02982 [0.98173, 1.05611]`。两种统计量得出相同阶段判定，未因选择统计量改变结论。

因此，本阶段只能称为：

> **实现并验证了 lossless、future-gated 的在线 residual 原型；没有证据表明它提升真实模型 TPF，也没有
> 证据表明计入在线成本后获得稳定 wall-clock 加速。**

## 1. 完整性与 exactness 边界

全部预注册安全检查通过：

- 30/30 runs 均输出恰好 512 tokens，所有数值有限；
- checkpoint revision 与 SHA-256 完全匹配；
- clean rows 的 LoRA-off logits 完全一致、noise rows 确实受 Uno adapter 影响；
- 15/15 deferred runs 的 trainable base tensors = 0、base optimizer overlap = 0；
- 每个 optimizer 只持有 526,336 个 rank-8 residual 参数；
- active-head evaluation + zero-head skip 恰好等于每个 run 的 cycle 数；
- 49/49 candidate decision 均有正的未来 row 权重，并严格复现冻结的 promote/keep/reset 规则；
- 方法顺序和 paired seed 均符合循环轮换协议。

当前 cycle 的 proposal 总是由 action 前保存的 $q_t$ 采样，verification 也使用同一份旧 $q_t$ 作分母；
candidate 的训练或 promotion 发生在 verify 后，只影响未来 cycle。因此参数在线变化不破坏 $\Psi$-Spec
的 exactness。这里的“lossless”来自算法不变量和 Stage 1 的完整分布测试，不等于本 15 对性能样本重新
估计了任意长序列的全分布。

## 2. Prompt 分层结果

| workload | TPF mean ratio [95% CI] | TPS mean ratio [95% CI] | TPF wins/losses/ties |
|---|---:|---:|---:|
| 英文解释（pilot 用过） | 1.00604 [0.99615, 1.01880] | 1.04298 [0.98117, 1.09657] | 2 / 2 / 1 |
| Python LRU（未调参） | 0.99349 [0.97730, 1.00317] | 1.01560 [0.98047, 1.05268] | 1 / 1 / 3 |
| 中文 MH 推导（未调参） | 0.98557 [0.97674, 0.99378] | 1.00725 [0.98519, 1.03150] | 0 / 4 / 1 |

英文点估计最好，但它是工程 pilot 用过的 prompt，不能当独立验证。代码基本持平；中文的 TPF 和
acceptance 明确向坏方向移动（acceptance mean delta `-0.01405`，区间
`[-0.02240, -0.00574]`）。pooled TPF 是 3 wins、7 losses、5 ties；排除 ties 的 exact two-sided sign-test
$p=0.34375$。TPS 是 9 wins、6 losses，$p=0.60724$。都不支持稳定正收益。

## 3. Controller 到底做了什么

15 个 deferred runs 共 2,891 cycles：

- 1,883 cycles（65.13%）仍是 zero residual，成功跳过 active residual 词表投影；
- candidate shadow 在 548 cycles（18.96%）上被评估；
- 64 次 candidate update 中 3 次 same-buffer rollback；
- 49 次 future decision：22 promote、26 keep、1 reset；
- 11/15 runs 至少 promote 一次；
- 被 promote 的 candidate 在批准窗口上的 filtered-TV advantage 中位数为 `0.00215`。

最重要的反例来自中文：它获得 10 次 promote，是三个 prompt 中最多的，但最终 TPF 4 次下降、1 次持平。
因此，“批准窗口上的 filtered TV 更小”没有转化成“上线后的下一窗口更高 TPF”。这不是实现没有更新，
而是当前 decision signal 的时序与 counterfactual 含义不够强。

## 4. 成本与 TPS 为什么不能宣称成功

显式计时的在线成本占 deferred decode 的均值如下：

| 组件 | 平均占比 |
|---|---:|
| clone + backward + optimizer | 0.569% |
| feedback materialization | 0.309% |
| active residual head | 0.038% |
| candidate residual head | 0.021% |
| 合计 | 0.937% |

TPS 的 +2.19% mean point estimate 与 TPF 的 -0.50% 方向相反，而且系统 CI 跨 1。用 paired static
每-forward 时间按 forward 数缩放后，未归因 residual 甚至平均为 online decode 的 `-3.64%`；这只能说明
Windows/PyTorch 运行间的 base-forward 时延波动大于被测在线 kernel 成本，不能解释为 residual 让同一个
base forward 变快。正式结论因此依赖预注册区间，而不是挑选有利的 TPS 点估计。

额外约 24.21 MiB 显存来自 active/candidate head、optimizer moments 和 feedback；对 24 GiB 3090 很小，
但仍应计入系统代价。

## 5. 失败机制

Stage 5A 的 timeline 是：

```text
window k       用旧 active 生成，训练 candidate
window k+1     继续用旧 active 生成，在其轨迹上 shadow-validate candidate
边界           promote candidate
window k+2     candidate 第一次真正参与 proposal
```

这留下两个 gap：

1. **一窗口陈旧性**：批准证据来自 $k+1$，真正收益要到 $k+2$；长回答内局部 token/context 分布仍在变化；
2. **policy-induced context**：$k+1$ 的 target rows 来自 active 采出的 proposal prefix。candidate 真上线后
   会采不同 token，所以 shadow TV 不是 candidate policy 的无偏 counterfactual rollout。

此外，`0.0005` 门只要求非常小的平均 TV 差就整头 hard-switch；收益量级可能小于跨窗口漂移和 sampling
方差。request-local 512-token run 约 180--210 cycles，首次 promotion 又不能早于 cycle 80，留给收益
摊薄训练成本的 horizon 很短。

## 6. 下一版设计约束

结果不支持继续扫描同一个 margin。下一阶段应改变机制而不是移动门槛：

- 把 hard promote 改成 static/candidate 的**渐进 mixture**，把每次错误批准的最大影响限制在小权重；
- 用 verify 后的累计 expert loss 更新下一轮 mixture weight，始终保留 static anchor；
- 或把学习状态提升到 session/request stream，在同域的多个请求间摊薄训练和验证成本；
- 对任何 mixture 仍保存实际采样的完整 $q_t$，保持 exact verification；
- 单独预注册“request-local 学习收益”和“跨请求 amortized 系统收益”，不能混在一个 TPS 数字里。

原始 30-run JSON 与独立分析 JSON 均保存在 `results/`，包含所有 paired observation、promotion event、
组件计时和安全检查，可复算本报告中的每个数字。
