# Lossless $\Psi$-Spec 推导、在线不变量与 Stage 1 验证

## 1. 单 token correction

在固定 history $h$ 下，target 为 $p(x\mid h)$，本轮 proposal 为 $q_t(x\mid h)$。先采
$Y\sim q_t$，以

```math
a_t(Y)=\min\left(1,\frac{p(Y\mid h)}{q_t(Y\mid h)}\right)
```

接受；拒绝后从

```math
r_t(x\mid h)=
\frac{[p(x\mid h)-q_t(x\mid h)]_+}
{\sum_z[p(z\mid h)-q_t(z\mid h)]_+}
```

采 correction。接受路径给 token $x$ 的概率质量为
$q_t(x)a_t(x)=\min\{p(x),q_t(x)\}$。总拒绝概率为

```math
1-\sum_x\min\{p(x),q_t(x)\}
=D_{\mathrm{TV}}(p,q_t)
=\sum_x[p(x)-q_t(x)]_+.
```

因此 correction 路径给 $x$ 的质量是 $[p(x)-q_t(x)]_+$，两条路径相加为

```math
\min\{p(x),q_t(x)\}+[p(x)-q_t(x)]_+=p(x).
```

这同时证明期望单 token 接受率是 $1-D_{\mathrm{TV}}(p,q_t)$，也是 Uno 使用 TV 蒸馏的直接动机。

## 2. block 与 Uno 的免费 AR token

对 proposal block $y_{1:k}$ 从左到右应用上一节 correction。条件于前 $i-1$ 个已接受 token，
第 $i$ 个输出仍精确来自对应 AR conditional $p_i$。首次拒绝后提交 residual token 并结束本轮；若全部接受，
再从 AR verifier 的 lookahead row $p_{k+1}$ 采一个 token。由条件概率链式法则，任意已提交 prefix 都服从
target AR 联合分布。

Uno 与普通 speculative decoding 的差异是 draft forward 还免费产生一个只走 $\theta_{\rm AR}$ 的 token：

```math
x_{L+1}\sim p_{\theta_{\rm AR}}(\cdot\mid x_{\le L}).
```

其后 $B-1$ 个位置由同一 forward 的 diffusion pathway 并行提出，再由第二次 AR forward 验证。因此即使
第一个 diffusion token 拒绝，一轮也推进“免费 AR token + residual token”两个 token。

参考实现位于 `src/online_speculation/psi_spec.py`：

- `uno_linear_step` 固定 draft matrix 后才采 free token，模拟并行 forward；
- `verify_speculative_block` 逐位置接受并从 $[p-q]_+$ correction；
- `one_token_output_distribution` 精确枚举单步输出，供负面对照使用。

## 3. 为什么在线变化的 proposal 仍然 lossless

令 $\mathcal F_t$ 包含第 $t$ 轮前的 history、过去 proposal、verification feedback、optimizer state 和
fast weights。在线算法允许

```math
q_t=F(\mathcal F_t)
```

任意依赖过去。条件于 $\mathcal F_t$，$q_t$ 已固定，而本轮 rejection sampling 输出仍为 $p_t$：

```math
\Pr(X_{t+1}=x\mid\mathcal F_t)=p_t(x\mid h_t).
```

对 $\mathcal F_t$ 取全期望后右侧不依赖 proposal 参数，因此边缘条件分布仍是 $p_t$。逐轮归纳得到整个输出
联合分布等于原 AR model。在线学习只改变成本与接受率，不改变答案分布。

## 4. 必须保存旧 $q_t$

正确时序是：

```text
sample from q_t -> save q_t -> verify/correct with q_t -> update -> use q_(t+1)
```

若从旧 $q_t$ 采 proposal，却在接受分母和 residual 中代入更新后的 $q_{t+1}$，上面的质量分解不再成立。
Stage 1 使用：

```math
p=(0.65,0.25,0.10),\quad
q_t=(0.05,0.15,0.80),\quad
q_{t+1}=p.
```

错误实现会认为所有 token 的 $p/q_{t+1}=1$ 并全部接受，于是输出仍是旧 $q_t$，与 target 的 TV 距离为
0.70。这个反例说明“更新得更准”也不能挽救错误的 acceptance denominator。

## 5. 数值与接口约束

- 每行概率先以 float64 验证并归一化；NaN、负值和零总质量立即失败。
- proposal token 在其保存的 $q_t$ 下必须有正概率。
- 只有真实发生 rejection 时才构造 residual；若此时 $[p-q]_+$ 质量为零，说明实现或输入矛盾。
- GPU 实现可以只保存 sampled-token $q_t(y_i)$ 供接受率，但 residual correction 仍需能得到
  $[p_i-q_{t,i}]_+$；top-k 稀疏化必须保持与实际 sampling filter 完全一致。
- temperature、top-p、top-k 的顺序属于分布定义。保存 raw logits 却用不同 filter 重算 $q_t$ 同样会破坏
  lossless 性。

## 6. Stage 1 实验设计

目标是验证完整输出 prefix，而非只看 acceptance：

1. 定义 3-token vocabulary 的非平稳 AR conditional；精确枚举长度 4 的 81 条 completion 概率。
2. 每种模式独立生成 100,000 条 completion，block size 4。
3. `static_draft` 在整条 completion 中固定错误 proposal。
4. `post_round_adaptive_draft` 每轮 verification 完成后才更新内部 proposal state。
5. 比较 empirical/target 的 TV、最大标准化误差和 Pearson $\chi^2$/dof。
6. 同时运行旧 $q_t$/新 $q_{t+1}$ 负面对照。

预注册通过线：TV 不超过
$\max(0.02,\frac32\sqrt{(|\Omega|-1)/N})$，最大标准化误差不超过 6，
$\chi^2/\mathrm{dof}\le2$；负面对照 TV 必须大于 0.25。

正式数值写入 `results/stage1_lossless_validation.json`，不从运行结果反向调整阈值。

## 7. Stage 1 正式结果

2026-09-05 在 Windows/RTX 3090 主机上按上述配置运行；该实验是 CPU categorical simulation，GPU
不参与数值。结果如下：

| 模式 | joint TV | max standardized error | $\chi^2$/dof | speculative accepts/round | 通过 |
| --- | ---: | ---: | ---: | ---: | --- |
| static draft | 0.009061 | 2.324 | 1.010 | 1.4771 | 是 |
| post-round adaptive draft | 0.009758 | 2.368 | 0.949 | 1.5578 | 是 |

两组 joint TV 都远低于预注册上限 0.04243，且 goodness-of-fit 指标符合 multinomial sampling noise。
在线更新组的 proposal state 每条 completion 平均更新 1.544 次，接受 speculative tokens/round 相对
static 增加约 5.46%，但完整输出仍与 AR target 一致。

解析负面对照中，正确旧分母的 TV 为 $7.63\times10^{-17}$，错误新分母的 TV 为 0.70。Stage 1
因此同时验证了两件事：

1. 每轮 verification 完成后再改变 proposal，不破坏 lossless 联合分布；
2. 对本轮 proposal 使用错误版本的 denominator 会造成巨大而可检测的分布偏差。

这里的 toy TPF 只用于检查计数逻辑，不能外推为神经网络或 GPU throughput。真实加速结论留给固定上游
checkpoint 的 Stage 2。
