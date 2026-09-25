# IS 预算控制：候选数、补全数与块长

本文件集中说明 AR 条件 IS 在每个块边界联合选择候选数 $`M`$、补全数 $`K`$ 与块长 $`B_{\rm blk}`$ 的统计目标、实现与计费。
目标分布及 MH/IS 的基础算法见[ALGORITHMS.md](ALGORITHMS.md#alg-conditional-is)，设置字段见[设置说明](../SETTINGS.md)。
联合调度是 AR `is` 的默认规划（`ar.algorithms.is.planning = "full_horizon"`），目前完成实现与 CPU 测试，尚无模型质量或加速结论。

- [目标与方差分解](#budget-moments)
- [有限候选的误差界](#budget-tv)
- [联合动态调度](#budget-joint)
- [预算单位与执行边界](#budget-accounting)
- [调用、参数和代码索引](#budget-usage)

<a id="budget-moments"></a>
## 1. 目标与方差分解

固定提示及已生成前缀 $`g`$，候选块为 $`z`$，后续补全为 $`u`$。$`M`$ 是候选数，$`K`$ 是每候选补全数，
$`B_{\rm blk}`$ 是块长，$`T`$ 是生成总长度上限。生成在 EOS 处停止，思考段范围内也在思考段结束标记处停止。奖励和终止规则在本次生成中固定：

```math
G(g,z,u)=\exp\{r(g,z,u)/\tau\},\qquad
W_B(z)=\mathbb E_p[G\mid g,z],\qquad
\mu=\mathbb E_p[G\mid g].
```

下标 $`B`$ 表示所选块长 $`B_{\rm blk}`$。精确目标块分布为
$`\pi_B(z\mid g)=p(z\mid g)W_B(z)/\mu`$。独立生成 $`M`$ 个基础模型候选，各补全 $`K`$ 次，得到

```math
\widehat W_m=\frac1K\sum_{k=1}^K G(g,z_m,u_{mk}),\qquad
\Pr(I=m\mid\text{候选与补全})=\frac{\widehat W_m}{\sum_j\widehat W_j}.
```

这里估计的是指数权重的期望，通常不等于平均奖励再取指数。定义两类方差：

```math
v_{\rm between}(B)=\mathrm{Var}_z[W_B(z)],\qquad
v_{\rm within}(B)=\mathbb E_z[\mathrm{Var}(G\mid g,z)].
```

全方差公式给出

```math
v_{\rm between}(B)+v_{\rm within}(B)=\mathrm{Var}(G\mid g),\qquad
\mathrm{Var}\!\left(\frac1M\sum_m\widehat W_m\right)
=\frac{v_{\rm between}(B)}M+\frac{v_{\rm within}(B)}{MK}.
```

同一完整生成分布下，长块包含短块的全部 token。由条件期望的迭代性质，长块条件权重在给定短块时的期望
等于短块条件权重；再用全方差公式，得到块长增加时候选间方差不减、候选内方差不增。
这一结论要求奖励、最终长度上限及生成策略固定。改变 Consilience 的评分窗口定义或补全截断规则会改变目标。

固定块长，设整步共享前缀的成本为 $`c_0`$、每个候选的固定成本为 $`c_z`$、每条补全及评分成本为 $`c_u`$，
预算为 $`C=c_0+M(c_z+Kc_u)`$（第 4 节给出各项）。代入上式并求导，在两类方差和成本均为正的连续情形下得到

```math
K^*=\sqrt{\frac{v_{\rm within}c_z}{v_{\rm between}c_u}},\qquad
M^*=\frac{C-c_0}{c_z+K^*c_u}.
```

该式解释宽度与重复补全的权衡。实际实现枚举整数配置，同时处理终止块、最小候选数和完成预留。
候选已覆盖剩余全部序列时，直接评分，记录 $`K=0`$，内部确定性权重按一次评估处理。

<a id="budget-tv"></a>
## 2. 有限候选的误差界

总变差距离定义为 $`\|P-Q\|_{\rm TV}=\sup_A|P(A)-Q(A)|`$，衡量两种分布对同一事件所赋概率的最大差异。

假设 $`0\lt G\lt\infty`$、$`0\lt\mu\lt\infty`$、二阶矩有限，候选独立同分布，给定候选后的补全独立，
且 $`M,K,B_{\rm blk}`$ 在最终采样前已确定。令 $`P_{M,K,B}`$ 为对全部采样随机性取平均后的选中块分布，则

```math
\left\|P_{M,K,B}-\pi_B\right\|_{\rm TV}
\leq\min\left\{1,
\sqrt{\frac{v_{\rm between}(B)+v_{\rm within}(B)/K}{M\mu^2}}
\right\}.
```

**推导。** 对任意候选集合 $`A`$，令

```math
\widehat Z=\frac1M\sum_m\widehat W_m,\qquad
\widehat H_A=\frac1M\sum_m\widehat W_m\mathbf1\{z_m\in A\}.
```

条件权重无偏，因此 $`\mathbb E\widehat H_A=\mu\pi_B(A)`$，而选中块落入 $`A`$ 的概率为
$`\mathbb E[\widehat H_A/\widehat Z]`$。由于 $`0\leq\widehat H_A/\widehat Z\leq1`$，

```math
\begin{aligned}
|P_{M,K,B}(A)-\pi_B(A)|
&=\left|\mathbb E\left[
\frac{\widehat H_A}{\widehat Z}\left(1-\frac{\widehat Z}{\mu}\right)
\right]\right|\\
&\leq\frac{\mathbb E|\widehat Z-\mu|}{\mu}
\leq\frac{\sqrt{\mathrm{Var}(\widehat Z)}}{\mu}.
\end{aligned}
```

对 $`A`$ 取上确界，并代入第 1 节的方差分解即得结论。该结论不要求在一个离散候选池上近似整个目标支持集；
它约束最终输出的边缘分布。有限候选归一化本身仍有误差；准确率还取决于奖励与正确性的关系。

若第 $`t`$ 次选择在所有可达前缀和预算状态上的条件 TV 误差均不超过 $`\delta_t`$，逐步耦合两条生成路径，
每一步首次分开的概率至多为 $`\delta_t`$，于是完整序列误差至多为 $`\min\{1,\sum_t\delta_t\}`$。
随机块长也可使用相同调度规则耦合；精确条件采样的目标序列分布不依赖这种分块。
在线样本矩并未提供这些统一上界，因此下面的累计分数只用于调度，不作为全序列误差保证。

<a id="budget-joint"></a>
## 3. 联合动态调度

预算侧与采样侧分开实现：共享选择器 `choose_joint_budget` 与两种规划器 `FullHorizonPlanner`、`AdaptiveBudgetController` 位于 `shared/budget/`，只接收成本估计和初始样本矩；AR 执行器 `run_joint_budget_is` 只负责生成初始样本与正式样本并记账。每到一个新前缀：

1. 去重配置块长（只有接近输出上限时才截到剩余长度），并加入“生成至 EOS”的完成选项。
2. 在 `pilot_fraction` 限制内按块长升序选出可负担的块长：各块长在同一组共享的完整输出（pilot 池）上切出初始
   候选，所有块长的其余补全在一次批处理中生成；始终保留按期望长度计算的完成预算。
3. 估计各块长的两类相对方差，枚举整数 $`M,K`$，选择误差预测分数最低的可行配置。
4. 冻结本轮配置，执行一次[条件 IS](ALGORITHMS.md#alg-conditional-is) 步：0 号候选沿用当前完整序列的下一块及其
   补全，其余候选与补全使用独立随机种子新生成；初始样本不进入最终权重。
5. 保留选中的块及其一条补全作为新的完整序列，按实际消耗记账，用本步新生成补全的平均长度更新期望剩余
   长度，在下一个块边界重新执行上述步骤，直至完整序列的末尾。

### 初始统计量

所有初始样本的对数权重减去同一个最大值再取指数。记第 $`m`$ 组的均值和样本方差为
$`\overline G_m,s_m^2`$，该组数量为 $`K_m^{\rm pilot}`$。使用

```math
\widehat\mu=\frac1{M_{\rm pilot}}\sum_m\overline G_m,\qquad
\widehat v_{\rm within}=\frac1{M_{\rm pilot}}\sum_m s_m^2,
```

```math
\widehat v_{\rm between}
=\max\left\{0,
\mathrm{SampleVar}_m(\overline G_m)
-\frac1{M_{\rm pilot}}\sum_m\frac{s_m^2}{K_m^{\rm pilot}}
\right\}.
```

第二式扣除了补全噪声对候选均值方差的贡献；取非负部分引入估计偏差，但避免出现负方差。
共享模块只保存两类方差除以 $`\widehat\mu^2`$ 的值，因此对所有对数权重加同一常数不改变计划。
终止候选权重确定，可用单次评估并将组内方差设为零；其余候选至少需要两次独立补全。

统计预算不足的块长使用明确的默认值：候选间、候选内相对方差均为 1，终止块的后者为 0；
`candidate_count=0` 标记缺少初始观测。分数计算使用 `relative_variance_floor=1e-4`，防止极少量相同观测导致零误差预测。
这些默认值与下限是调度规则，不能排除未观测的高权重分支。

### 联合配置选择

对当前前缀，用期望剩余长度 $`\hat\ell`$（见[第 4 节](#budget-accounting)）估计剩余选择次数

```math
n(B)=\left\lceil\frac{\hat\ell}{B}\right\rceil,
```

完成选项记 $`n=1`$。旧版本以输出上限剩余量 $`T-|g|`$ 代替 $`\hat\ell`$，使预测随上限增长。

实现最小化下列插入样本矩的预测分数，枚举范围由配置给出：

```math
J(M,K,B)=n(B)
\sqrt{\frac{\widehat v_{\rm between}(B)+\widehat v_{\rm within}(B)/K}
{M\widehat\mu^2}}.
```

终止块删去组内项。排序前不将 $`J`$ 截到 1，以保留高噪声配置之间的差别。
可行配置同时满足

```math
n(B)\,[c_0+M(c_z(B)+Kc_u(B))]\leq C_{\rm remaining},
```

并在非终止块执行后保留一次最小完整候选 IS 的成本。相同分数依次按本轮成本更低、块长更长、候选数和补全数更少打破平局。

这个预测假设后续位置具有相近的方差与成本；每个新前缀都会重新规划。它可能偏好完整序列 IS，
并不保证多块路径更优，也没有全局预算最优或准确率提升保证。成本按期望长度估计，不拟合 GPU 时间模型。

初始样本只用于选择 $`M,K,B_{\rm blk}`$；除沿用保留序列的 0 号候选外，正式候选与补全都使用独立随机种子重新生成，
最终权重只含正式样本。先查看正式奖励再决定何时停止，或把初始样本混入最终平均值，均不满足第 2 节的独立性条件。
嵌套估计的非线性归一化和预算选择可参考
[On Nesting Monte Carlo Estimators](https://proceedings.mlr.press/v80/rainforth18a.html)及
[Bootstrap-based Budget Allocation for Nested Simulation](https://doi.org/10.1287/opre.2020.2071)；
本实现使用样本矩，不包含 bootstrap、训练得到的调度器或任务正确性标签。

<a id="budget-accounting"></a>
## 4. 预算单位与执行边界

### 计划成本与实际记账

补全和完成都生成到 EOS，成本取决于剩余输出的实际长度。设提示长度为 $`P`$，当前已生成长度为 $`L`$，
输出上限为 $`T`$，一次奖励需要 $`s`$ 次完整序列评分（复用生成概率的 `logprob` 奖励 $`s=0`$）。规划使用期望剩余长度

```math
\hat\ell=\min\{\hat\ell_{\rm obs},\,T-L\},
```

其中 $`\hat\ell_{\rm obs}`$ 是上一步新生成的正式补全的平均长度。第一步之前使用
`expected_output_tokens`；未给出时从提示生成一条普通补全测量长度，其消耗记为
`length_probe_forward_tokens` 并计入预算，这条补全不作为候选。

后端对同一批中相同的前缀只预填充一次（Transformers 在批内复用，vLLM 缓存前缀），成本按此计算。新候选以
完整输出生成，块与第一条补全来自同一请求：整步预填充一次前缀，每个候选解码 $`B+d`$ 个 token，
$`d=\max\{1,\hat\ell-B\}`$；$`K\gt1`$ 时每个候选再预填充一次自己的前缀 $`P+L+B`$，解码其余 $`K-1`$ 条补全。
于是非终止块

```math
c_0=P+L,\qquad
c_z(B)=B+\mathbf 1\{K\gt1\}(P+L+B),\qquad
c_u(B)=d+s\,(P+L+B+d).
```

到达输出上限的块（$`B=T-L`$）为终止块：候选就是生成至停止的完整输出并直接评分，
$`c_z=\hat\ell+s(P+L+\hat\ell)`$、$`c_u=0`$。完成预留 $`c_0+M_{\min}c_z`$ 随前缀更新；给出
`expected_output_tokens` 时，初始预算不足会在调用模型前报错。pilot 池本身就是 `pilot_candidates` 条完成候选，
由当前前缀的第一个 pilot 支付，每个非终止块长只再付其候选分支前缀与 `pilot_rollouts - 1` 条补全。

每步结束后按实际请求记账：每批中相同的前缀计一次，生成的 token 逐个计入，每条不同的完整序列只评分一次；
剩余预算取实际余额；
早于预期结束的补全因此不再占用预算。计划只依赖此前步骤的样本，在本步正式样本生成前固定，
第 2 节按可达前缀与预算状态条件化的逐步论证仍然适用。前面步骤超支时，规划器至少按完成预留看待剩余预算，
完成序列不会被拒绝。

这是请求级的前向 token 位置账本，不是任意后端的实际 FLOPs。`logprob` 与 `consilience` 各按一次完整序列评分计
（$`s=1`$），`verifier` 与 `vote` 按文本计算（$`s=0`$）；`python` 来源 verifier 的内部计算（例如外部评分模型）和
`vote` 样本池的生成都不进入账本。分块评分重复预填充、额外模型调用和后端内部实现可能使实测成本与账本不同；
实际开销由后端计数器另外报告（记录的 `cost.phases`）。

0 号候选的块及其保留补全已在之前的步骤生成并评分，实际记账不再计入。规划仍按 $`M`$ 个候选与 $`MK`$ 条补全
估计成本，因此没有 EOS 与重复序列时，实际消耗等于计划成本减去这部分复用，计划偏保守。

### 块长与输出上限解耦

块长网格本身不含 $`T`$，旧版本却让 $`B_{\rm blk}`$ 随 $`T`$ 变化：每次 rollout 与完成预留都按生成到
$`T`$ 计价（$`c_u=(1+s)(P+T)`$，完成预留 $`M_{\min}(1+s)(P+T)`$），且 EOS 提前结束不退款。
$`T`$ 越大，预留越早耗尽预算，调度越早进入块长为 $`T-L`$ 的收尾；`full_horizon` 还以 $`T-L`$
作为预测视界。根源是硬预算保证：rollout 必须生成到 EOS，只有按最坏情况 $`T`$ 预留才能保证不超预算。

现在 $`T`$ 只是生成的硬上限：它截断接近上限的块，决定哪个块是终止块，并在期望长度达到上限时进入成本。
只要输出没有触及上限，同一问题与随机种子下的块长、候选数、补全数、调整记录和输出都与 $`T`$ 无关；
[`test_joint_budget_is.py`](../../tests/test_joint_budget_is.py) 在 $`T=2048,16384,1048576`$ 下验证了这一点。
代价是预算变为按期望成本规划的软约束：补全比预期更长时，实际消耗可能超过计划和 `forward_token_budget`。
账本的实际消耗单独报告（`trace.planned_forward_tokens_used`），后端实测的前向 token 位置数见
`cost.forward_token_slots`；正式比较应使用实际值。

### 实测成本与公平比较

模型 $`j`$ 参数量为 $`N_j`$、实际前向 token 位置数为 $`S_j`$ 时，沿用

```math
\widehat F_{\rm forward}=2\sum_j N_jS_j.
```

预填充、解码、评分和批量填充均按后端实际执行记录；该估算省略注意力的长度平方项及逐元素计算。
墙钟、吞吐和显存独立测量。比较应包含长度测量、初始估计、`vote` 样本池和奖励评分。
相同计划预算不代表相同实际 FLOPs；正式实验需同时给出二者。

联合调度只接入 AR 的 `is`，要求同模型 on-policy 补全和固定逐序列奖励。`ar.output.sampling_scope = "thinking"`
时规划对象是思考段：候选与补全在思考段结束标记或 EOS 处停止，长度估计与账本都按实际生成的 token 数。Consilience 奖励
默认只读 thinking，切分失败沿用显式全序列回退；这与 IS 修改整个输出范围是两个独立设置。候选级不同
$`K_m`$ 与 dLLM 块长适配均未接入。共享选择器不绑定模型族，但其他适配层需要提供有效的统计量、成本和独立采样。

<a id="budget-usage"></a>
## 5. 调用、参数与代码索引

### 运行入口

联合预算是 AR `is` 的默认规划，统一入口的默认选择即运行它：

```bash
python -m inference_scaling --algorithm is --model ar --reward vote --dataset gsm8k
```

规划方式由 `ar.algorithms.is.planning` 选择：`fixed` 使用 `ar.algorithms.is.fixed` 中固定的 $`M,K,B_{\rm blk}`$，不做
预算调度；`full_horizon`（默认）与 `chunk_adaptive` 读取 `ar.algorithms.is.joint`，后者另读
`ar.algorithms.is.chunk_adaptive`。奖励取 `--reward` 指定的逐序列固定奖励，温度为 `rewards.<name>.temperature`。
下表的设置键省略前缀 `ar.algorithms.is.`，记录字段位于 `records.jsonl` 每行的 `trace` 中。

| 设置键 / 记录字段 | 含义 |
| --- | --- |
| `joint.forward_token_budget` | 包含长度测量、初始采样和奖励评分的总预算；按期望成本规划，按实际消耗记账 |
| `joint.block_sizes` | 块长网格；仅 `full_horizon` 模式将完整剩余长度加入正常竞争 |
| `joint.candidate_counts`、`joint.rollout_counts` | 整数网格，分别为 $`M\geq2`$、非终止时 $`K\geq1`$ |
| `joint.pilot_candidates`、`joint.pilot_rollouts` | 共享 pilot 池的完整输出数与每个被探测块长的每候选补全数 |
| `joint.pilot_fraction` | 每轮初始估计最多使用当前剩余预算的比例；还受完成预留限制 |
| `joint.relative_variance_floor` | 预测分数中的相对方差下限 |
| `joint.expected_output_tokens` | 第一步之前的期望输出长度；`null` 时先生成一条普通补全测量长度 |
| `chunk_adaptive.initial_block_size`、`initial_candidate_count`、`initial_rollout_count`、`adjustment_min_improvement` | 仅 `chunk_adaptive` 读取的初值与调整门槛 |
| `steps[].plan` | 选中的 $`M,K,B_{\rm blk}`$、局部误差估计、累计预测分数和是否有对应初始观测；收尾步（$`K=0`$）生成至 EOS，其 `block_size` 只记录生成上限 $`T-L`$ |
| `reserved_forward_tokens`、`planned_forward_tokens_used` | 各步计划成本合计与账本实际消耗合计（含长度测量与初始估计） |
| `steps[].pilot_forward_tokens`、`length_probe_forward_tokens` | 实际消耗中每步的初始估计部分与长度测量部分 |
| `steps[].expected_remaining_tokens`、`steps[].forward_tokens` | 规划该步时的期望剩余长度与该步正式样本的实际消耗 |
| `stopping_reason` | `eos` 或 `length` |

### 按下一块预算动态调整

默认 `planning = "full_horizon"`。选择 `chunk_adaptive` 时读取初值与调整门槛，例如把 `ar.algorithms.is` 中的
以下字段改为：

```json
{
  "planning": "chunk_adaptive",
  "chunk_adaptive": {
    "initial_block_size": 128,
    "initial_candidate_count": 4,
    "initial_rollout_count": 2,
    "adjustment_min_improvement": 0.1
  }
}
```

- 三个初值必须属于各自网格。
- 每次运行从初值开始，第一块不先做 pilot；后续仅在新 pilot 有效、存在方差信号、
  改善超过阈值且预算可负担时调整。没有证据或 pilot 预算不足则保持 B/M/K。
- B 每次最多探测一个相邻网格值；比较 B 需要当前块与邻居的两组 pilot（切分同一 pilot 池）。
  只容得下一组时，只允许调整 M/K；单元素 `block_sizes` 固定 B。
- 正式执行按期望成本检查下一块，另保护按期望长度计算的收尾预算；pilot 同时受比例上限和保护预算限制。
  每步结束后按实际消耗记账。15% 是可配置比例，不保证 pilot 能启动。
- 仅当当前 B/M/K 已无法负担，或剩余输出额度不超过当前 B 时，进入最少候选数、K=0 的收尾：
  候选生成至 EOS（至多到输出上限）并直接评分。收尾不是块长选择，完整剩余长度不参与正常块长竞争。
- 跨块长使用 `H = max(本次有效 pilot 的 B)`、`ceil(H/B) * local_error` 比较，
  不用最大输出上限预测整个 thinking。这是小样本启发式指标，不是正确率或显著性保证。
- `steps[].adjustment` 记录初值、保持/调整/收尾原因及比较分数；pilot 不进入正式候选池。

`chunk_adaptive` 统一采用成本优先规则，无需额外策略开关：先枚举本次有效
pilot 块长上的所有预算可行 M/K，筛选预测误差改善严格超过 `adjustment_min_improvement`
的方案，再选下一正式块计划成本最低者。同成本时按误差、较大 B、较小 M/K 确定性排序。
没有合格方案则保持当前值；信号检查、pilot 扣费、初值、收尾与预算保护不变。

此处成本指下一正式块按期望长度计算的计划成本，不是同覆盖长度总成本、实际 token、墙钟时间或 GPU FLOPs；
pilot 成本已在选择前扣除且对本次可选方案相同。跨 B 的误差仍按上述 H 比较。
该规则不保证每次都比当前配置便宜，只保证在超过改善门槛的可行方案中选择最便宜者；
它会牺牲进一步降低预测误差的机会，也不保证整题效率或正确率改善。
`steps[].adjustment` 额外记录 `comparisons[].eligible`、`eligible_count`、
`selection_reason`、`selected_relative_improvement` 和 `selected_reserved_cost`；
有合格方案时，`best_score` 指所选合格方案的分数，不一定是全局最低误差分数。

底层 `choose_joint_budget(..., forecast_full_horizon=False)` 只接受一个块长估计，
避免直接比较不同覆盖长度；运行层负责相邻块比较和独立收尾。
上述设置是示例，不代表已经运行真实模型或验证解题准确率。

### Python 接口

```python
from inference_scaling.arllm.algorithms.joint_budget_is import (
    JointBudgetISConfig, run_joint_budget_is,
)
from inference_scaling.shared.rng import SeedStream

result = run_joint_budget_is(
    backend, prompt_tokens,
    JointBudgetISConfig(forward_token_budget=2_000_000),
    reward, SeedStream(0), sampling=sampling,
)
```

`reward(prompt_tokens, sequences, token_logprobs)` 批量返回完整序列的奖励，`token_logprobs` 是各序列在生成策略下的
逐 token 对数概率；它必须是固定逐序列函数。`vote` 奖励因此先冻结一个独立样本池；
直接在当前候选池内重新统计多数标签会改变候选权重之间的依赖关系，不适用第 2 节证明。

| 职责 | 代码 / 测试 |
| --- | --- |
| 相对方差估计、联合整数选择 | [`shared/budget/joint.py`](../../src/inference_scaling/shared/budget/joint.py)：`estimate_weight_moments`、`choose_joint_budget` |
| 全视界规划与逐块自适应规划 | [`shared/budget/planners.py`](../../src/inference_scaling/shared/budget/planners.py)：`FullHorizonPlanner`、`AdaptiveBudgetController` |
| 计划成本与完成预留 | [`shared/budget/costs.py`](../../src/inference_scaling/shared/budget/costs.py)：`block_costs`、`completion_reserve` |
| AR 循环、独立随机数、预算记账 | [`arllm/algorithms/joint_budget_is.py`](../../src/inference_scaling/arllm/algorithms/joint_budget_is.py)：`JointBudgetISConfig`、`run_joint_budget_is` |
| 实际候选生成、补全和重采样 | [`arllm/algorithms/conditional_is.py`](../../src/inference_scaling/arllm/algorithms/conditional_is.py)：`conditional_is_step` |
| 统一入口的设置读取与记录 | [`app/ar.py`](../../src/inference_scaling/app/ar.py)：`ARFamily._is` |
| 前向 FLOPs 估算 | [`shared/compute.py`](../../src/inference_scaling/shared/compute.py)：`dense_forward_flops` |
| 矩估计、联合决策、精确枚举 TV 检查 | [`test_joint_budget.py`](../../tests/test_joint_budget.py) |
| 预算、EOS、随机数隔离与分布测试 | [`test_joint_budget_is.py`](../../tests/test_joint_budget_is.py) |
| 逐块自适应规划与成本优先规则 | [`test_joint_budget_adaptive.py`](../../tests/test_joint_budget_adaptive.py)、[`test_joint_budget_cost_policy.py`](../../tests/test_joint_budget_cost_policy.py) |
| 统一入口的默认选择与思考范围检查 | [`test_app.py`](../../tests/test_app.py) |

```powershell
python -m pytest -q tests/test_joint_budget.py tests/test_joint_budget_is.py tests/test_joint_budget_adaptive.py tests/test_joint_budget_cost_policy.py
```

小样本方差可能低估稀有分支，完整序列备选可能被频繁选中，初始估计也可能抵消调度收益。
