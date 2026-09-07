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
| 前缀重叠长度目标 | 同配置确认中冻结续训后有改善，持续在线的净收益较弱；保留精确估计与梯度对照 | `prefix_objective.py`、`scripts/prefix_overlap.py` |
| 冷启动完整块在线蒸馏 | 从 AR 注意力初始化，完整序列 KL 与离线共用更新核；预算式服务与独立学习曲线验证 | `cold_start.py`、`scripts/cold_start.py` |

本表描述相应配置下的本地实验。不同起点和提示设置的数值用于各自配对对照。

冷启动完整块训练的服务成本与学习曲线集中于 [性能记录](../docs/RESULTS.md#7-短块冷启动与同预算离线对照)。
首轮长块训练后，仅缩短推理块的改善有限；当前短块方案同时调整训练跨度与随机锚点数。
它以约定的时间预算积累实际回答，提供候选质量、在线净 TPS 和同更新数离线训练三组测量。
更新执行筛选中，融合 AdamW 的单窗口耗时较低；`foreach` 的额外显存占用与六窗口的教师计算成本，
使后两者留作筛选结论。当前多窗口入口支持对上下文覆盖与更新耗时进行配对比较。
三窗口冷启动确认流的门控与训练预算延后了发布，整段净吞吐为 0.9902× AR，详见
[性能记录第 8 节](../docs/RESULTS.md#8-冷启动更新的执行与上下文覆盖)。
另一次相同重放记录的短程筛选中，将峰值学习率提高到 0.0002／0.0004，32 步后的 KL
分别为 1.9861／2.2508，高于原学习率 0.0001 的 1.7739；后续沿用原学习率。
在相同已交付记录上的四题、64 步筛选中，三 token 块的早期学习弱于四 token 块；
只在回答部分选锚点改善了后期候选质量，早期上线优势仍有限，当前保留完整窗口抽样。
复用已交付 AR 前缀减少约一半的额外验证时间；独立冷启动流的两次单前缀门控分别为
1.0570×／1.0946×，低于 1.10 发布门槛，全流为 0.9901× AR。
81 步在线／同预算离线候选分别为 1.2512×／1.2331× AR，具体对照见
[性能记录第 9 节](../docs/RESULTS.md#9-复用已交付前缀的发布检查)。

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

较早的起草注意力 KL 续训使用 thinking、温度 1、top-k=20、top-p=0.8，
16 题、每题 512 token、重复两次。固定／在线为 90.141／92.398 TPS，
比值 1.0250×，95% 配对区间 [0.9918, 1.0603]。
同前缀审计的原始 KL 下降 3.80%，实际采样 TV 接近持平。
这一配置的增量接近测量波动，后续对照转向训练与采样概率一致的完整词表设置。

前缀重叠目标采用主线 TV 实验相同的 16 题、3 条打乱顺序的学习流、完整词表及更新配置。
固定／在线为 174.014／170.788 TPS，在线／固定为 0.9815×，95% 配对区间 [0.9455, 1.0105]，
三条流分别为 0.9625×、1.0267×、0.9533×。在线更新 137 次、累计 1.0784 秒，平均 7.87 毫秒。
预学习后固定为 185.210 TPS，相对原固定 1.0643×；继续在线为 179.296 TPS，
相对学习后固定 0.9681×，区间 [0.9408, 0.9955]。
预学习耗时 20.206 秒、更新 32 次，同前缀 TV 从 0.168953 降至 0.161841。
此前 8 题试测的在线点估计为 +4.32%，确认实验转为 −1.85%；当前主线采用 TV 在线续训。

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

### 2.5 前缀重叠目标

这个对照直接优化每轮期望提交数，仍使用一次并行起草与一次 AR 验证。
给定轮次开始时的真实历史，生成候选的各行分布记为 $q_i^0$，
更新中的分布记为 $q_{\phi,i}$。教师 $p_i(\cdot\mid y_{<i})$ 来自完整候选的验证前向。
候选总数为 $m$，剩余输出预算为 $R$，取 $h=\min(m,R-1)$。

把 EOS 记为 $e$，定义该位置的接受且继续生成的概率，以及前缀的重要性权重：

$$
c_i(\phi)=\sum_{v\ne e}\min(p_i(v),q_{\phi,i}(v)),\qquad
r_i(\phi)=\frac{\min(p_i(y_i),q_{\phi,i}(y_i))}{q_i^0(y_i)}\mathbf1[y_i\ne e].
$$

一份完整候选反馈给出估计量

$$
\widehat J_\phi(y_{1:h})
=\sum_{i=1}^{h}\left(\prod_{j<i}r_j(\phi)\right)c_i(\phi).
$$

每一项把最后一个位置的抽样对词表求和，仅用实际候选估计它之前的前缀。
对生成候选的分布 $\prod_jq_j^0(y_j)$ 取期望时，前缀权重中的分母逐项抵消，故

$$
\mathbb E_{q^0}[\widehat J_\phi]
=\sum_{i=1}^{h}\sum_{y_{1:i}:y_j\ne e}
  \prod_{j=1}^{i}\min(p_j(y_j\mid y_{<j}),q_{\phi,j}(y_j))
=\mathbb E_{q_\phi}[N]-1.
$$

$N$ 为包含替代或额外 token 的实际提交数。恒等式依据
$N=1+\sum_{i=1}^{h}\mathbf1[\text{前 }i\text{ 枚候选均接受且均非 EOS}]$。
末尾预算通过 $h$ 截断，EOS 通过词表求和与前缀指示函数处理。

有限词表、完整支持且各项可微时，求导与有限求和可交换：
$\mathbb E_{q^0}\nabla_\phi\widehat J_\phi=\nabla_\phi\mathbb E_{q_\phi}[N]$。
分母、候选编号与教师视为固定快照，梯度同时通过 $c_i$ 和此前全部 $r_j$。
实际更新使用当前参数版本产生的新反馈，重放完整块，再选取目标所需的行。
更新间隔到达且请求仍在生成时执行学习；请求结束释放重放数据，沿用主线的调度约定。
上面的无偏等式针对给定历史下的完整候选抽样，具体更新轨迹还包含这项调度选择。
这是固定轮次历史下的长度梯度；端到端吞吐另包含更新成本和后续轮次的上下文变化。

实现位于 `prefix_objective.py`，入口为 `scripts/prefix_overlap.py`。
小词表检查枚举接受、拒绝、EOS 与预算分支，并核对估计值、梯度、完整前向与后段重放。

### 2.6 多窗口监督与梯度方差

一条回答上的多个锚点共享文本背景。固定一次更新的监督位置数时，
可以将锚点分配到多条回答，使梯度同时覆盖不同上下文。
设当前重放区已经给定，从中有放回抽取 $`M`$ 条记录，分别裁剪得到窗口 $`X_1,\ldots,X_M`$。
每个窗口独立均匀抽取 $`K`$ 个锚点。每个锚点贡献一个完整块，监督位置数为 $`MK(B-1)`$。

记 $`\ell(\phi;X,a)`$ 为该块的平均 KL，$`g(X,a)=\nabla_\phi\ell(\phi;X,a)`$。
裁剪之前的平均梯度为

```math
\widehat g=\frac1M\sum_{j=1}^M\frac1K\sum_{k=1}^K g(X_j,a_{jk}),
\qquad \mathbb E[\widehat g]=\nabla_\phi F_{\mathcal R}(\phi).
```

$`F_{\mathcal R}`$ 是[主报告第 6.6 节](../docs/ALGORITHM.md#66-从-ar-初始化的完整序列学习)在当前重放区上的窗口目标。
令 $`\mu(X)=\mathbb E_a[g(X,a)\mid X]`$，将梯度波动分为两部分：

```math
V_{\rm between}=\mathbb E_X\|\mu(X)-\nabla_\phi F_{\mathcal R}\|^2,
\qquad
V_{\rm within}=\mathbb E_X\mathbb E_a\|g(X,a)-\mu(X)\|^2.
```

第一项描述窗口之间的差别，第二项描述同一窗口内锚点之间的差别。
将 $`\widehat g-\nabla_\phi F_{\mathcal R}`$ 展开为窗口均值项与窗口内残差项。
残差在给定窗口后均值为零，独立样本的交叉项在期望中消失，因而

```math
\mathbb E\|\widehat g-\nabla_\phi F_{\mathcal R}\|^2
=\frac{V_{\rm between}}M+\frac{V_{\rm within}}{MK}.
```

保持 $`MK`$ 固定并增加 $`M`$，第二项保持原值，第一项随窗口数减小。
该关系描述固定参数、固定重放区上的原始梯度估计；梯度裁剪和 AdamW 随后作用于这一估计。
每个新增窗口也需要一次对应的 AR 教师计算，因此配置选择同时考虑学习质量与更新耗时。

实现将较短窗口在右侧补齐到同批最长窗口的长度，锚点在各窗口原始长度内抽取。
教师有效行通过因果掩码只读取自身及此前位置；起草块只读取锚点之前的干净 KV 和本块内容。
补齐位置因此被排除在所有受监督行的信息路径之外。
由层数上的归纳，打包后的有效隐藏状态与逐窗口计算相同；各窗口锚点数一致时，
打包损失及其梯度正好对应各窗口损失及其梯度的平均值。数值测试按所用浮点精度检查这一等式。

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
python ablation/scripts/prefix_overlap.py --help
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
