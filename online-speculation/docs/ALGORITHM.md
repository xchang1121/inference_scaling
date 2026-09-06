# 共享骨干的并行扩散起草与自回归校正

本文讨论一条共同的生成管线：**共享 AR 骨干和历史 KV，以并行扩散分支提出候选，由 AR 分支校正并提交结果。**
目标模型决定输出分布；起草分支决定一次验证能够推进的长度。
条件低秩投影与独立双向注意力是这条管线中的两种起草实现。

## 1. 自回归模型与块验证

### 1.1 文本与下一项概率

语言模型处理的基本单位称为 token，它可以表示一个字、词的一部分、标点或特殊符号。
分词器把文本转换为整数序列，每个整数对应词表 $\mathcal V$ 中的一项，词表大小记为 $V$。
把一句话简化为四个 token：

$$
[\text{I},\text{love},\text{New},\text{York}].
$$

给定前缀“I love”，模型为下一项提供一张概率表，例如 New 为 0.6、you 为 0.3，
其余项合计 0.1。按这张表抽样，得到 New 的概率就是 0.6。

记完整序列为 $x_1,\ldots,x_T$，前附起始符 $x_0$，
$x_{<t}$ 表示 $x_0,\ldots,x_{t-1}$。下一项概率写成

$$
p_\theta(x_t=v\mid x_{<t}).
$$

竖线右侧是已给定的前缀，左侧是被预测的 token；$\theta$ 汇总模型参数。
固定前缀后，对词表中的全部 $v$ 求和，概率之和等于 1。

模型先输出各个词表项的分数，称为 logit。设分数为 $\ell_1,\ldots,\ell_V$，
softmax 将其转换为概率：

$$
p_\theta(v\mid x_{<t})
=\frac{\exp(\ell_v)}{\sum_{u\in\mathcal V}\exp(\ell_u)}.
$$

指数函数把分数转换为正数，分母将这些正数归一化。
较高的 logit 对应较大的概率。

### 1.2 NTP 训练中的并行预测

下一 token 预测简称 NTP，英文为 next-token prediction。
训练数据提供完整句子，各位置的输入和监督关系为：

| 输入位置 | 该位置可读的文本 | 预测的下一项 |
|---|---|---|
| I | I | love |
| love | I love | New |
| New | I love New | York |

输入经过各层网络、得到 logits 的过程称为一次前向计算。
表中三个位置的输入同时给定，一次前向可以同时算出三张概率表。
因果注意力将每一行的可见内容限定为自身及左侧输入，
使三张表分别具有表中对应的前缀条件。
这种使用真实前缀提供训练输入的方式称为 teacher forcing。

训练最小化平均负对数概率：

$$
\mathcal L_{\rm NTP}(\theta)
=-\frac1T\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

$\log$ 是自然对数。例如真实下一项的概率从 0.1 提高到 0.8，
该项损失从约 2.303 降到 0.223。
训练通过梯度更新参数，使真实下一项获得较高概率。

### 1.3 AR 生成中的串行依赖

生成时，输入可能只有“I love”。后续条件随抽样结果逐项形成：

$$
x_3\sim p_\theta(\cdot\mid\text{I love}),\qquad
x_3=\text{New}\ \Longrightarrow\
x_4\sim p_\theta(\cdot\mid\text{I love New}).
$$

$\sim$ 表示按右侧分布抽样，$\cdot$ 表示整张词表概率表。
抽到 New 后，New 成为预测 York 的输入条件。
这形成逐 token 的数据依赖。

自回归的英文为 autoregressive，简称 AR。
整段序列的概率由条件概率相乘得到：

$$
p_\theta(x_{1:T}\mid x_0)
=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

$\prod$ 表示连乘。以三项序列为例，
$p(x_1,x_2,x_3)=p(x_1)p(x_2\mid x_1)p(x_3\mid x_1,x_2)$，省略了共同的起始符条件。
NTP 训练同时提供全部真实前缀；AR 生成则沿实际抽样路径扩展前缀。

### 1.4 候选与块验证

投机解码引入一个较便宜或适合并行计算的起草分布 $q$。
它在当前文本 $s$ 后提出候选 $y_1,\ldots,y_m$。
候选一旦给定，就形成一段完整输入，目标模型可以像 teacher forcing 一样，
在一次因果前向中同时计算

$$
p_i(v)=p_\theta(v\mid s,y_{<i}),\qquad i=1,\ldots,m+1.
$$

例如 $s=$“I love”，候选为 New / York / City。
验证得到以“I love”“I love New”“I love New York”为条件的概率表，
还得到整块之后的下一项概率。
候选的提出可以并行，目标对这条已给定路径的评价也可以并行。

验证器按这些目标概率保留连续前缀，在首次分歧处补齐目标概率质量，
由此形成一轮包含多枚输出的计算。第 5 节给出精确校正规则与分布保持证明。
起草质量决定一轮通常推进多少 token，执行耗时决定最终每秒产出。

### 1.5 生成时的概率变换

推理常使用温度、top-k 和 top-p。
正温度 $\tau$ 将 logits 替换为 $\ell/\tau$；较小温度使概率集中。
top-k 保留分数最高的 $k$ 项；top-p 按概率排序，保留累计概率首次达到阈值的前缀，
随后归一化。贪心生成直接取最大 logit 对应的 token。

将完整变换记为 $\mathcal T$，实际校正使用

$$
p=\mathcal T(\ell_{\rm AR}),\qquad q=\text{起草时实际使用的概率表}.
$$

AR 基线和投机验证采用同一目标变换。
这使输出分布的比较同时固定模型权重与采样规则。

## 2. 起草结构的分类

| 结构 | 起草输入与参数 | 块内关系 |
|---|---|---|
| 独立小模型 | 文本前缀、另一套较小模型 | 逐 token 自回归 |
| MTP | 目标表示、多预测头或串联预测模块 | 多头可并行，串联模块可保留深度间依赖 |
| EAGLE-3 | 多层目标特征、轻量起草网络 | 起草网络逐项预测 |
| DFlash | 目标特征注入独立轻量扩散网络的 KV | 一次前向并行预测一块 |
| DSpark | 并行表示、轻量 Markov / RNN 修正、置信度 | 小模块读取已采样的块内前缀 |
| 本文共享骨干管线 | 原 AR 历史 KV、共享骨干及起草专用参数 | 并行扩散起草，可外接条件修正 |

MTP 是多 token 预测的训练与架构范畴。简单多头的损失为
$\mathcal L_{\rm MTP}=-\sum_{t,j}\log q_j(x_{t+j}\mid x_{\le t})$；
具体模型也可采用串联模块。各类起草器与校正规则组合，形成完整的投机解码算法。

DFlash 复用目标特征来构造自己的起草 KV；本文两条分支直接读取各层原始 AR KV。
这一差别影响参数存储、历史状态和起草成本。DSpark 的条件修正与调度思想可以作用于多种并行起草器。

## 3. 共享骨干与历史状态

### 3.1 两种视图

设共享参数为 $\theta$，起草专用参数为 $\phi$。AR 视图定义目标，
扩散视图将占位符块映射为未来 token 的分布：

$$
p_\theta(\cdot\mid s),\qquad
q_{\theta,\phi}(\cdot\mid C_\theta(s),z).
$$

$C_\theta(s)$ 是各层历史键和值的集合；$z$ 包含锚点与随机或掩码占位符。
共享嵌入、前馈层、归一化等参数的范围由具体分支确定。
两种视图每次都执行完整的层序列。

设一层的当前隐藏状态为 $H$，历史 KV 为 $K_c,V_c$。
起草分支产生本块的 $Q_d,K_d,V_d$，注意力为

$$
O_d=
\operatorname{softmax}\!\left(
\frac{Q_d[K_c\Vert K_d]^\top}{\sqrt{d_h}}+M_d
\right)[V_c\Vert V_d].
$$

$\Vert$ 表示沿 token 轴拼接；$d_h$ 是注意力头宽度。
掩码 $M_d$ 在允许访问的位置取 $0$，其余位置取 $-\infty$。
输出投影和共享前馈层继续变换 $O_d$。
分支的差别集中在投影参数、占位符、掩码与输出位置对齐。

### 3.2 历史缓存不变量

记 $C_\theta(x_{<t})$ 为原 AR 模型在已提交前缀上计算的 KV。
每轮入口保持这一状态，末枚已提交 token 可以作为待处理锚点单独保存。

因果层的第 $j$ 行只由前缀 $x_{\le j}$ 决定。
对网络层数归纳：第一层输入与逐项 AR 相同；若某层之前的前缀表示相同，
该层的查询、键、值、可见位置和前馈计算也相同。
因此，整块因果验证在一个已接受前缀上得到的 KV，与逐项 AR 的对应 KV 相同。

首次分歧后，仅提交验证前缀的 KV。替代 token 作为下一轮锚点，
在下一次 AR 计算中取得自身的 KV。起草占位符产生的临时状态在轮次结束时释放。

若层数为 $n_\ell$、KV 头数为 $n_{\rm kv}$、每个元素占 $w$ 字节，
历史长度为 $t$，历史 KV 占用为

$$
M_{\rm history}=2n_\ell n_{\rm kv}d_h\,t\,w.
$$

固定块长 $K$ 的临时起草 KV 为 $O(n_\ell n_{\rm kv}d_hK)$，
随历史长度增长的持久 KV 仍只有一份。
总显存还包括起草参数、注意力工作区、logits 和图执行存储；
历史张量拼接的实现也可能产生随 $t$ 增长的临时副本。

## 4. 并行扩散起草的两条分支

### 4.1 条件低秩、因果噪声分支

设当前前缀末项为 $c$，块长为 $B$，起草输入为

$$
[c,z_1,\ldots,z_{B-1}],\qquad z_i\sim\pi.
$$

原线性投影 $Wh$ 加上逐位置开启的低秩增量：

$$
f(h,m)=Wh+m\frac{\alpha}{r}BAh,\qquad
A\in\mathbb R^{r\times d_{\rm in}},\
B\in\mathbb R^{d_{\rm out}\times r}.
$$

根行 $c$ 的开关为 $m=0$，噪声行取 $m=1$。
因果掩码使根行读取干净历史，其输出恰为 $p_\theta(\cdot\mid s)$。
一次起草提供

$$
y_0\sim p_\theta(\cdot\mid s),\qquad
y_i\sim q_i(\cdot\mid s,z_{\le i}),\quad 1\le i<B.
$$

所有行沿用 NTP 的一位偏移：输入 $c$ 预测 $y_0$，输入 $z_1$ 的行预测 $y_1$。
条件于噪声，本块后续候选的联合分布为
$Q(y_{1:B-1}\mid s,y_0,z)=\prod_iq_i(y_i\mid s,z_{\le i})$。

根行的 KV 由原 AR 参数产生，可以加入历史。
验证输入为 $[y_0,y_1,\ldots,y_{B-1}]$，其第 $i$ 行为候选 $y_{i+1}$ 提供目标分布，
末行提供全部接受后的额外 token。
完整轮次的新产出为

$$
G_{\rm causal}=2+A,\qquad 0\le A\le B-1.
$$

其中两项分别来自精确根 token，以及替代或额外 token。

### 4.2 独立注意力、双向掩码分支

设 $a=x_t$ 是已由 AR 分布产生的锚点，历史为 $C_\theta(x_{<t})$。
长度 $K$ 的输入为

$$
[a,\underbrace{\text{MASK},\ldots,\text{MASK}}_{K-1}].
$$

每层起草使用独立的 $W_Q^d,W_K^d,W_V^d,W_O^d$ 和 Q/K 归一化，
共享原嵌入、层归一化、前馈层与语言输出头。
这些专用投影在初始化时可复制 AR 对应参数；公开权重已经完成蒸馏。

本块各行读取全部历史以及本块全部位置，故第 3.1 节中的块内 $M_d=0$。
双向交互发生在锚点与掩码表示之间。
第 $j$ 个输入位置仍预测下一 token；前 $K-1$ 行产生

$$
y_i\sim q_i(\cdot\mid C_\theta(x_{<t}),a,\mathrm{MASK}^{K-1}),
\qquad 1\le i<K.
$$

验证输入为 $[a,y_1,\ldots,y_{K-1}]$。
验证第 $0$ 行提供 $p_\theta(\cdot\mid x_{\le t})$，对应 $y_1$；
第 $i$ 行提供以 $a,y_{1:i}$ 为前缀的下一项分布。
完整轮次的新产出为

$$
G_{\rm bidirectional}=1+A,\qquad 0\le A\le K-1.
$$

锚点已在上一轮产出。当前轮新产生连续接受的 $A$ 枚候选和一枚替代或额外 token。
请求的初始 prefill 单独产生第一枚锚点。

两种分支的统一计数为 $G=b+A$：
条件低秩分支的 $b=2$，独立注意力分支的 $b=1$。
EOS 与输出预算在提交阶段截断这一计数。

### 4.3 占位符与离散扩散

离散替换过程以保留率 $\alpha_t$ 混合干净 token 与噪声分布：

$$
q_t(z\mid x)=\alpha_t\mathbf1[z=x]+(1-\alpha_t)\pi(z),
\qquad \alpha_0=1,\quad\alpha_1=0.
$$

均匀 $\pi$ 对应随机 token；集中在 MASK 的 $\pi$ 对应掩码替换。
锚点作为条件保留。一步起草从完全污染的块直接预测干净输出。

对 $s<t$，令 $\rho=\alpha_t/\alpha_s$。已知干净 token $x$ 和观测 $z_t=u$，
逆向条件概率由贝叶斯公式给出：

$$
q(z_s=v\mid z_t=u,x)\propto
[\rho\mathbf1[v=u]+(1-\rho)\pi(u)]
[\alpha_s\mathbf1[v=x]+(1-\alpha_s)\pi(v)].
$$

比例常数由对 $v$ 求和得到。在完全噪声到干净的端点，
$\alpha_t=0,\alpha_s=1$，用模型概率替代干净 token 的指示向量后，
归一化的逆向分布就是起草模型的输出 $q_\phi(v)$。
因此一步起草直接使用模型 softmax；多步去噪对应额外中间状态与前向计算。

### 4.4 蒸馏的位置与掩码

条件低秩分支采用干净序列与带噪序列配对。
设干净序列长为 $L$，位置 $j$ 的块起点为 $g(j)=B\lfloor j/B\rfloor$。
干净行 $j$ 读取干净位置 $0,\ldots,j$；
带噪行 $j$ 读取干净位置 $0,\ldots,g(j)-1$ 和本块噪声位置 $g(j),\ldots,j$。
两路位置编号相同，两路第 $j$ 行都预测 $x_{j+1}$。

双向分支在干净序列上选择锚点 $a_b$，组成多个
$[x_{a_b},\mathrm{MASK}^{K-1}]$。第 $b$ 块第 $j$ 行的位置编号为 $a_b+j$。
允许的注意力边为

$$
\mathrm{allow}((b,j),k)=
\begin{cases}
k<a_b,&k\text{ 属于干净序列},\\
b'=b,&k\text{ 属于起草块 }b'.
\end{cases}
$$

不同锚点块相互隔离。起草第 $j$ 行的教师为干净 AR 第 $a_b+j$ 行，
对应标签 $x_{a_b+j+1}$。与公开推理的前 $K-1$ 行对齐时，取 $j=0,\ldots,K-2$。
这一索引约定同时规定训练目标、验证输入和公开权重的数值对照。

教师整段因果前向得到干净 KV 与隐藏状态；起草前向读取其停止梯度的副本。
未来干净 KV 虽然已计算，但在起草掩码中权重为零。

### 4.5 分布蒸馏目标

记同一位置的教师概率为 $P$，起草概率为 $Q$。
双向分支采用前向 KL：

$$
D_{\rm KL}(P\Vert Q)=\sum_vP(v)\log\frac{P(v)}{Q(v)},\qquad
\mathcal L_d=\mathbb E\frac1{|\mathcal I|}
\sum_{i\in\mathcal I}D_{\rm KL}(P_i\Vert Q_i).
$$

$P$ 固定时，与起草参数有关的部分为交叉熵 $-\sum_vP(v)\log Q(v)$。
若 $Q=\operatorname{softmax}(u)$，则

$$
\frac{\partial D_{\rm KL}(P\Vert Q)}{\partial u_v}=Q(v)-P(v).
$$

每一项直接比较学生和教师分配的概率质量。
条件低秩分支的原配方采用反向 KL 加 L1 热身，随后使用 L1：
$\sum_vQ(v)\log[Q(v)/P(v)]+\sum_v|P(v)-Q(v)|$；
L1 项等于 $2D_{\rm TV}(P,Q)$。训练接口明确记录目标方向、监督位置、块长和随机种子。

## 5. 统一的概率校正

### 5.1 单位置的质量守恒

给定实际前缀，候选 $Y\sim q$ 的接受概率为

$$
a(Y)=\min\left(1,\frac{p(Y)}{q(Y)}\right).
$$

直接接受 token $v$ 的质量为 $q(v)a(v)=\min(p(v),q(v))$。
定义正残差及其总质量

$$
r(v)=\frac{[p(v)-q(v)]_+}{Z},\qquad
Z=\sum_v[p(v)-q(v)]_+.
$$

$p,q$ 均归一化，所以
$\sum_v[q(v)-p(v)]_+=Z$，恰等于总拒绝概率。
拒绝时按 $r$ 抽样，得到

$$
\Pr(X=v)=\min(p(v),q(v))+Zr(v)=p(v).
$$

例如 $p=(0.5,0.3,0.2)$、$q=(0.4,0.4,0.2)$，
接受分支保留 $(0.4,0.3,0.2)$，拒绝分支向第一项补入 $0.1$。
$q(v)=0$ 的目标质量由残差承接；$p=q$ 时接受概率为 1。

贪心目标是最大项上的点质量。验证因而简化为最长相同前缀，
分歧处取目标最大项。

### 5.2 整块与自适应起草

令 $\mathcal H$ 包含已提交历史、已完成的在线更新与当前起草随机条件。
在提出第 $i$ 枚候选时，使用条件概率

$$
q_i(v\mid\mathcal H,y_{<i}),\qquad
p_i(v)=p_\theta(v\mid s,y_{<i}).
$$

独立并行分布和半自回归分布都属于这一形式。
每个已到达位置都满足第 5.1 节的等式，沿实际输出前缀归纳，
整段输出的条件概率乘积等于目标 AR 分布。

首次拒绝后，后续验证行对应原候选前缀；下一轮从已校正前缀继续。
全部接受时，额外一行目标概率提供下一 token。
每轮保存采样时的 $q_i$；参数更新在本轮校正与提交之后生效。

变长起草的准入决定在抽取当前候选之前作出。
例如块长由过去轮次的耗时与接受统计确定，便满足这一条件。
读取未来候选后追溯选择前缀，会改变已选候选的条件分布，需要另行推导选择校正。

### 5.3 接受率、前缀存活与轮次产出

总变差距离为 $D_{\rm TV}(p,q)=\frac12\sum_v|p(v)-q(v)|$。
由 $\min(a,b)=(a+b-|a-b|)/2$，

$$
\Pr(\text{接受})=\sum_v\min(p(v),q(v))=1-D_{\rm TV}(p,q).
$$

设 $A$ 为连续接受数，$S_i=\Pr(A\ge i)$，$S_0=1$。
在实际到达第 $i$ 项的条件下定义

$$
e_i=\mathbb E[D_{\rm TV}(p_i,q_i)\mid A\ge i-1].
$$

条件概率乘法法则给出

$$
S_i=S_{i-1}(1-e_i),\qquad
\mathbb E[A]=\sum_{i=1}^{m}S_i,\qquad
\mathbb E[G]=b+\sum_{i=1}^{m}\prod_{j=1}^{i}(1-e_j).
$$

这里 $e_i$ 对已到达的随机前缀取平均。
逐条路径上的 TV 乘积是另一种统计量；上述公式由条件期望定义保证精确性。

若 $e_i\le\epsilon$，则
$\mathbb E[G]\ge b+\sum_{i=1}^{m}(1-\epsilon)^i$。
相同到达分布上的平均 KL 为 $\delta_i$ 时，
Pinsker 不等式与 Jensen 不等式给出 $e_i\le\sqrt{\delta_i/2}$。
离线平均损失到该到达分布的联系，需要训练覆盖与分布偏移的控制。

## 6. 在线与半自回归扩展

### 6.1 实际前缀的信息

设目标只产生“喝／茶”和“吃／饭”，各占一半。
独立并行分布即使准确拟合每个位置的边缘概率，仍会为四个组合各分配 $1/4$。
已抽中的“喝”则直接揭示下一项应偏向“茶”。

双向掩码让位置表示交换信息；实际采样前缀提供另一类条件。
冻结并行表示后，可加入小型修正：

$$
\widetilde q_i(v)\propto
q_i^0(v)\exp g_\phi(h_i,y_{<i},v).
$$

$g_\phi=0$ 对应原起草分布。
Markov 形式取 $g_\phi=E_\phi[y_{i-1}]W_\phi$；
上下文形式还读取当前位置的 $h_i$。
候选集形式令集合外的修正为零，并对整个概率表统一归一化。

训练反馈来自同一候选前缀的验证分布。
缓存的 $h_i$ 与教师概率作为常量，小头更新单独计时，
接受与残差校正使用实际的 $\widetilde q_i$。

若 $h=F_\theta(x)$ 的参数保持冻结，缓存 $h$ 后计算的小头梯度为
$\nabla_\phi\mathcal L=J_\phi g_\phi(h)^\top\nabla_g\mathcal L$，
与重新计算同一 $F_\theta(x)$ 得到的梯度相同。
末层子集续训同理在冻结层与可训练层之间缓存边界表示；缓存版本绑定冻结参数和历史状态。

### 6.2 小型概率混合

现有扩展提供温度专家与历史接续专家。
温度专家 $e_j$ 由同一分布变换而来，混合为
$q_w=\sum_jw_je_j$，$w_j\ge0,\sum_jw_j=1$。
前缀门控的接续专家在实际候选仍匹配历史接续时使用

$$
q_\lambda=(1-\lambda)q_0+\lambda\delta_c,\qquad 0\le\lambda\le1.
$$

每个专家都是归一化分布，混合系数限制在凸集合中。
固定观测前缀与专家表时，$\ell(w)=D_{\rm TV}(p,q_w)$ 是凸函数。
单纯形上的一个等价切向次梯度为

$$
g_j=-\sum_v e_j(v)\mathbf1[q_w(v)<p(v)].
$$

省略的各坐标公共常数在单纯形投影中抵消。
投影更新为 $w_{t+1}=\Pi_\Delta(w_t-\eta_tg_t)$。
对任意固定比较点 $u\in\Delta$，

$$
\ell_t(w_t)-\ell_t(u)
\le
\frac{\|w_t-u\|^2-\|w_{t+1}-u\|^2}{2\eta_t}
+\frac{\eta_t}{2}\|g_t\|^2.
$$

单纯形直径平方至多为 2，$\|g_t\|^2\le M$。
取 $\eta_t=\eta/\sqrt t$，累加得到
$\sum_{t=1}^{T}[\ell_t(w_t)-\ell_t(u)]
\le(1/\eta+\eta M)\sqrt T$。
该界比较实际观测损失上的固定混合系数；多层神经起草器的优化另有其参数化与采样条件。

### 6.3 在线成本调度

设可选块长集合为 $\mathcal K$。维护各块长的产出和完整耗时估计：

$$
K^\star=\arg\max_{K\in\mathcal K}
\frac{\widehat{\mathbb E}[G_K]}
{\widehat C_d(K)+\widehat C_v(K)+\widehat C_h(K)+\widehat C_u(K)}.
$$

$C_h$ 包括采样、缓存、同步，$C_u$ 包括反馈与更新。
候选动作中可包含逐项 AR。轮次开始前确定动作，
轮次完成后更新统计，形成满足第 5.2 节条件的调度。
实测耗时曲线决定块长；GPU 图可为少数离散长度分别准备工作区。

## 7. 性能与验证

### 7.1 加速条件

稳定负载下，AR 每 token 耗时为 $C_A$，投机平均轮耗时为 $C$：

$$
\mathrm{TPS}_{\rm AR}\simeq\frac1{C_A},\qquad
\mathrm{TPS}_{\rm spec}\simeq\frac{\mathbb E[G]}{C},\qquad
S=\frac{\mathbb E[G]C_A}{C}.
$$

若扩展将平均产出从 $G_0$ 改为 $G_1$，轮耗时从 $C_0$ 改为 $C_1$，
净收益的条件为 $G_1/G_0>C_1/C_0$。
逐位置接受概率、整轮产出与完整耗时共同决定结论。

正式测量按总量计算：
$\mathrm{TPS}=N/\sum_rT_r$，$N$ 为实际返回的输出 token；
$T_r$ 包括请求 prefill、生成、反馈、更新和末尾同步。
模型加载、分词、图准备分别报告，另给含准备成本的吞吐。
TPF 为输出 token 总数除以解码前向次数；初始 prefill 和双向分支的初始锚点单独计数。

配对比较固定模型、精度、注意力后端、采样参数、提示模板、输出预算和停止条件。
同一请求的各方法交错执行，置信区间按请求重采样。
预学习数据、调参验证集和最终测试集各自独立。

### 7.2 数值与缓存验证

概率保持的推导采用精确算术。浮点验证同时检查 logits、概率 TV 与最大项。
若前两名 logit 的间隔为 $\delta$，每项误差至多为 $\varepsilon$，
$\delta>2\varepsilon$ 足以保持最大项相同。

验证层次包括：小词表联合分布枚举；整段与增量 AR；
每层共享 KV；起草对未来干净 token 的隔离；多锚点与单锚点蒸馏；
梯度与冻结参数；贪心输出；公开权重的外部 logits 对照。
统计一致性与浮点最大项一致性分别报告。

### 7.3 当前基线

硬件为 RTX 3090 24 GB / WSL2。以下为重构前已完成的验证结果，
使用 K2-Horizon-0.9B 及公开 r=128 适配器，BF16、块长 8、
温度 1、top-k=50、top-p=0.95。17 个问题各重复两次，每组生成 8,704 token。

| 方法 | TPS | 每轮产出 | 相对 AR |
|---|---:|---:|---:|
| AR | 125.106 | 1.0000 | 1.0000× |
| 固定条件低秩起草 | 132.252 | 3.0035 | 1.0571× |
| 深位置接续混合，冷启动在线 | 131.178 | 2.9890 | 1.0485× |
| 接续系数预学习后冻结 | 131.548 | 3.0086 | 1.0515× |
| 同一系数继续在线学习 | 131.303 | 3.0149 | 1.0495× |

最后一行相对固定起草为 0.9928×，95% 区间 $[0.9699,1.0170]$。
接受产出增加约 0.38%，轮耗时增加约 1.11%。
公共基座与起草适配器在各组中保持冻结。

执行优化的既有配对结果见 Git 提交 d50823c：
静态 TPS 从 108.655 增至 131.457，输出、接受与前向计数相同。
早期条件头见 578efef；接续混合见 09ce8ba。
其余取舍由相关 Git 历史追溯。

双向分支的性能复现采用公开的 Qwen3-1.7B 双视图权重。
本节随公开数值对齐与配对实验更新，模型更换后以其自身 AR 分支为分母。

## 8. 实现组织与来源

管线按四个职责组织：**骨干与状态、起草分支、概率校正、执行与测量**。
起草分支提供候选、对应概率和验证位置；
校正器返回提交前缀及干净缓存边界；
执行层管理算子与工作区，测量层统计各阶段成本。

已有条件低秩实现位于 [model.py](../src/blockspec/model.py)、
[distillation.py](../src/blockspec/distillation.py)；
共享概率规则位于 [sampling.py](../src/blockspec/sampling.py)。
[RUNNING.md](RUNNING.md) 记录运行入口，
[upstream.lock.json](../references/upstream.lock.json) 固定代码和模型来源。

双向分支按论文 Qwen3 架构与发布权重实现。
官方主分支新增的 Qwen3.5 训练入口使用 hard-label CE；
本文第 4.5 节对应论文 Qwen3 的完整教师概率 KL 配方。
两种训练对象与目标通过配置明确区分。

## 参考文献与实现来源

1. [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html)：接受与正残差校正。
2. [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)、[DeepSeek-V3](https://arxiv.org/abs/2412.19437)：多头与串联 MTP。
3. [EAGLE-3](https://arxiv.org/abs/2503.01840)：目标特征辅助起草。
4. [DFlash](https://arxiv.org/abs/2602.06036)：轻量扩散网络与目标特征 KV 注入。
5. [Unlocking Lossless Speedups in LLMs via Discrete Diffusion](https://arxiv.org/abs/2609.04010)、[源码](https://github.com/ifm-ai/uno)：条件低秩、因果噪声与配对蒸馏。
6. [Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion](https://arxiv.org/abs/2605.12825)、[固定源码](https://github.com/chiennv2000/orthrus/tree/4dceab65156b3dfb5dadbb11181a0e65d0ad314d)、[公开 1.7B 权重](https://huggingface.co/chiennv/Orthrus-Qwen3-1.7B)：双向起草、独立注意力与共享历史。
7. [DSpark](https://arxiv.org/abs/2607.05147)：半自回归修正与硬件感知调度。
8. [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)、[OnlineSPEC](https://arxiv.org/abs/2603.12617)、[Test-Time Speculation](https://arxiv.org/abs/2605.09329)：验证反馈与在线学习。
9. [LoRA](https://arxiv.org/abs/2106.09685)：低秩增量参数化。
10. [Prompt Lookup](https://github.com/apoorvumang/prompt-lookup-decoding)、[REST](https://arxiv.org/abs/2311.08252)、[SuffixDecoding](https://arxiv.org/abs/2411.04975)：历史接续候选。
