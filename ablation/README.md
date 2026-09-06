# 消融与对照实现

本目录集中保存当前主线以外的起草结构、在线修正及相关验证代码。
主线为共享骨干的双向注意力起草、整块概率校正和起草注意力后段续训，见
[主报告](../docs/ALGORITHM.md)与[性能记录](../docs/RESULTS.md)。

## 1. 设计与保留理由

| 设计 | 当前证据与取舍 | 实现 |
|---|---|---|
| 因果噪声块上的条件低秩起草 | 所测小模型固定起草接近 AR；保留结构与训练对照 | `model.py`、`diffusion.py`、`distillation.py` |
| 低秩参数全量或后段在线续训 | 多组所测配置中学习成本抵消候选改善 | `online.py`、`replay_execution.py` |
| 半自回归条件小头与候选准入 | 学习已抽出候选的条件修正；当前实测净收益有限 | `relay.py`、`relay_execution.py` |
| 稀疏温度混合 | 参数量小、反馈梯度便宜；所测在线吞吐接近或低于固定起点 | `calibration.py`、`scripts/dual_online.py` |
| 历史接续与拷贝混合 | 复用近期序列后继，在线更新少量系数；所测收益接近测量波动 | `continuation.py`、`scripts/overlap_mix.py` |
| 候选树与旧骨干执行路径 | 用于结构、数值和执行布局对照 | `tree.py`、`execution.py`、`hf_execution.py` |

本表描述相应配置下的本地实验。不同起点和提示设置的数值用于各自配对对照。

条件低秩分支采用约 0.9B 基座、秩 128、BF16、块长 8、温度 1、top-k=50、top-p=0.95；
17 个问题各输入／输出 256 token，重复两次：

| 方法 | TPS | 对照 |
|---|---:|---|
| AR | 125.106 | 固定起草的同配置基线 |
| 固定起草 | 132.252 | 相对 AR 为 1.0571× |
| 原起点在线接续混合 | 131.178 | 参数仅为接续系数 |
| 独立学习后固定 | 131.548 | 与继续学习共用学习起点 |
| 继续在线接续混合 | 131.303 | 相对原固定 0.9928×，95% 区间 [0.9699, 1.0170] |

双向分支的稀疏温度混合更新 155 个系数。在 8 个问题、每题 256 token、重复两次的随机采样对照中，
固定／在线为 86.465／85.918 TPS，在线／固定的 95% 配对区间为 [0.9614, 1.0275]。
这些结果把后续重点转向对实际采样质量的直接监督与起草注意力后段更新。

## 2. 数学摘要

### 2.1 因果噪声与条件低秩

一层线性映射使用 $Wh+a(\lambda/r)BAh$。$A\in\mathbb R^{r\times d}$、$B\in\mathbb R^{d'\times r}$，
开关 $a$ 对 AR 行取零、起草行取一，$\lambda/r$ 为适配器缩放。起草输入为真实锚点及随机 token 块。
锚点行走 AR 参数产生精确根 token；其余行走低秩分支提出候选。
带噪行读取真实历史和本块内此前的带噪位置，使用因果掩码。

离线训练把干净序列和污染序列拼接。干净行提供教师概率，
带噪行只读取块起点之前的干净历史及本块带噪前缀。蒸馏使用完整分布的 KL 或 TV。
`parallel/branches.py` 将这一布局接到主线的提交与缓存循环。

### 2.2 半自回归条件修正

设并行骨干的第 $i$ 行 logits 为 $u_i$，上一枚实际候选为 $y_{i-1}$。
低秩条件头给出

$$
\widetilde q_i
=\operatorname{softmax}\left(u_i+E[y_{i-1}]W\right).
$$

小头利用块内已经抽样的结果，把独立行分布改成条件分布。
可学习置信度头估计当前位置的平均接受概率，并在采样下一候选之前决定是否继续扩展。
该决策基于当时已知的信息；每个被提出的候选保留其实际采样分布，
随后沿用主线的正残差校正。

### 2.3 稀疏概率混合

以若干温度变换或历史接续分布作为专家 $r_k$，混合权重满足单纯形约束：

$$
q_w(v)=\sum_{k=1}^{K}w_k r_k(v),\qquad
w_k\ge0,\quad\sum_k w_k=1.
$$

在固定专家和教师下，损失
$\ell_t(w)=\frac12\sum_v|q_w(v)-p_t(v)|$
对 $w$ 是凸函数，其一个次梯度为

$$
g_{t,k}=\frac12\sum_v
\operatorname{sign}(q_w(v)-p_t(v))r_k(v).
$$

投影更新 $w_{t+1}=\Pi_\Delta(w_t-\eta g_t)$。
若单纯形上使用的距离直径至多 $D$，且 $\|g_t\|\le G$，投影的非扩张性给出

$$
\|w_{t+1}-w^\ast\|^2
\le\|w_t-w^\ast\|^2
-2\eta g_t^\top(w_t-w^\ast)+\eta^2G^2.
$$

结合凸性
$\ell_t(w_t)-\ell_t(w^\ast)\le g_t^\top(w_t-w^\ast)$，
求和并取 $\eta=D/(G\sqrt T)$，得到对固定比较权重的累计遗憾界：

$$
\sum_{t=1}^{T}[\ell_t(w_t)-\ell_t(w^\ast)]
\le DG\sqrt T.
$$

这个界衡量给定专家与反馈序列上的混合选择，实际加速还取决于专家覆盖、连续接受长度及执行时间。

### 2.4 离散逆向转移

给定干净类别 $x$ 与噪声先验 $\pi$，定义
$P(z_t=v\mid x)=\alpha_t\mathbf1[v=x]+(1-\alpha_t)\pi(v)$。
对 $s<t$ 且 $\alpha_s>0$，令 $\beta=\alpha_t/\alpha_s$，
前向转移为

$$
P(z_t=v\mid z_s=u)=\beta\mathbf1[v=u]+(1-\beta)\pi(v).
$$

Bayes 公式给出

$$
P(z_s=u\mid z_t=v,x)=
\frac{
[\alpha_s\mathbf1[u=x]+(1-\alpha_s)\pi(u)]
[\beta\mathbf1[v=u]+(1-\beta)\pi(v)]
}{
\alpha_t\mathbf1[v=x]+(1-\alpha_t)\pi(v)
}.
$$

在终点直接预测干净 token 的一步起草使用学生对 $x$ 的估计分布，
之后由 AR 进行接受与残差校正。`test_diffusion_math.py` 枚举前向联合概率验证逆向公式。

## 3. 独立运行与测试

在项目根目录安装主线和可选消融包：

~~~bash
python -m pip install -e '.[dev,hf,text,data]'
python -m pip install -e ./ablation --no-deps
python -m pytest -c ablation/pyproject.toml ablation/tests -q
~~~

`blockspec_ablation` 单向导入主线的通用概率、缓存和数据工具。
默认的 `python -m pytest -q` 仅运行主线测试。
消融目录中的脚本可从项目根目录运行：

~~~bash
python -m blockspec_ablation --help
python ablation/scripts/benchmark_offline.py --help
python ablation/scripts/prefix_relay.py --help
python ablation/scripts/overlap_mix.py --help
python ablation/scripts/dual_online.py --help
~~~

低秩模型、适配器、数据、外部参考和输出位置都由调用参数提供。
数据准备和旧训练入口归于 `blockspec_ablation`。
`scripts/audit_pipeline.py` 可选择本地 Git 版本进行旧管线数值对照。
具体参数由各入口的 `--help` 给出。

已有失败尝试保留上述取舍与相关代码；新增原始结果和临时产物保存在版本控制之外。
主线选择以匹配基线下的净吞吐和正确性验证为依据。

## 参考

- [离散扩散起草与条件低秩分支](https://arxiv.org/abs/2609.04010)及[参考源码](https://github.com/ifm-ai/uno)。
- [半自回归修正与置信度调度](https://arxiv.org/abs/2607.05147)。
- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)、[Test-Time Speculation](https://arxiv.org/abs/2605.09329)。
- [历史接续候选](https://github.com/apoorvumang/prompt-lookup-decoding)、[SuffixDecoding](https://arxiv.org/abs/2411.04975)。
