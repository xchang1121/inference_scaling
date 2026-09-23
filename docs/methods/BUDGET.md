# IS 预算控制：候选数、补全数与块长

本文件集中说明预算分配的统计目标、实现与计费。目标分布及 MH/IS 的基础算法见
[ALGORITHMS.md](ALGORITHMS.md#alg-conditional-is)；准确率和实测成本分别见
[算法质量报告](../reports/GSM8K_3090_ALIGNED_RESULTS.md)与
[执行成本报告](../reports/RTX3090_ROLLOUT_INFRA.md)。联合调度目前完成实现与 CPU 测试，尚无模型质量或加速结论。

- [目标与方差分解](#budget-moments)
- [有限候选的误差界](#budget-tv)
- [联合动态调度](#budget-joint)
- [固定候选的方差—成本分配](#budget-allocation)
- [初始估计与最终估计分离](#budget-pilot)
- [预算单位与执行边界](#budget-accounting)
- [调用、参数和代码索引](#budget-usage)

<a id="budget-moments"></a>
## 1. 目标与方差分解

固定提示及已生成前缀 $`g`$，候选块为 $`z`$，后续补全为 $`u`$。$`M`$ 是候选数，$`K`$ 是每候选补全数，
$`B_{\rm blk}`$ 是块长，$`T`$ 是生成总长度上限。EOS 后按吸收终止处理。奖励和终止规则在本次生成中固定：

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

固定块长，设生成一个候选成本为 $`c_z`$、一次补全及评分成本为 $`c_u`$，预算为 $`C=M(c_z+Kc_u)`$。
代入上式并求导，在两类方差和成本均为正的连续情形下得到

```math
K^*=\sqrt{\frac{v_{\rm within}c_z}{v_{\rm between}c_u}},\qquad
M^*=\frac{C}{c_z+K^*c_u}.
```

该式解释宽度与重复补全的权衡。实际实现枚举整数配置，同时处理终止块、最小候选数和完整生成预留量。
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

共享选择器为 `choose_joint_budget`；AR 执行器为 `run_joint_budget_is`。每到一个新前缀：

1. 将配置块长截到剩余长度、去重，并加入“直接生成至长度上限”的完整候选选项。
2. 在 `pilot_fraction` 限制内，按块长升序生成少量初始候选及补全；完成预算始终预留。
3. 估计各块长的两类相对方差，枚举整数 $`M,K`$，选择误差预测分数最低的可行配置。
4. 冻结本轮配置，使用独立随机种子重新生成候选及补全；仅这批样本进入最终权重。
5. 提交重采样选中的块，扣除预留成本，在新前缀重新执行上述步骤，直至 EOS 或长度上限。

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

对当前前缀，估计剩余选择次数

```math
n(B)=\left\lceil\frac{T-|g|}{B}\right\rceil.
```

实现最小化下列插入样本矩的预测分数，枚举范围由配置给出：

```math
J(M,K,B)=n(B)
\sqrt{\frac{\widehat v_{\rm between}(B)+\widehat v_{\rm within}(B)/K}
{M\widehat\mu^2}}.
```

终止块删去组内项。排序前不将 $`J`$ 截到 1，以保留高噪声配置之间的差别。
可行配置同时满足

```math
n(B)M[c_z(B)+Kc_u(B)]\leq C_{\rm remaining},
```

并在非终止块执行后保留一次最小完整候选 IS 的成本。相同分数依次按本轮成本更低、块长更长、候选数和补全数更少打破平局。

这个预测假设后续位置具有相近的方差与成本；每个新前缀都会重新规划。它可能偏好完整序列 IS，
并不保证多块路径更优，也没有全局预算最优或准确率提升保证。首版使用保守请求成本，不拟合 GPU 时间模型。

<a id="budget-allocation"></a>
## 4. 固定候选的方差—成本分配

已有 `allocate_variance_cost_budget` 在候选集合确定后分配历史/新样本；它与第 3 节的联合配置选择是独立入口。
对候选 $`i`$、来源 $`s\in\{\mathrm{history},\mathrm{fresh}\}`$，设外层概率比为 $`\rho_i`$，
实际估计项标准差为 $`\sigma_{i,s}`$，单样本成本为 $`c_{i,s}`$。连续代理目标为

```math
\min_{n_{i,s}}\sum_{i,s}\frac{\rho_i^2\sigma_{i,s}^2}{n_{i,s}}
\quad\mathrm{s.t.}\quad\sum_{i,s}c_{i,s}n_{i,s}\leq C.
```

对拉格朗日函数求导得到

```math
-\frac{\rho_i^2\sigma_{i,s}^2}{n_{i,s}^2}+\lambda c_{i,s}=0,
\qquad n_{i,s}\propto\frac{\rho_i\sigma_{i,s}}{\sqrt{c_{i,s}}}.
```

实现还处理每候选库存上限、同 replay 匹配键的共享容量、最少新样本数和整数取整。方差来自与最终样本分开的设计集，
并针对含 off-policy 或 replay 校正的实际估计项计算。连续解优化的是方差代理；容量约束与整数舍入后的分配不声称全局最优。
理论背景见 [Neyman 分层分配](https://doi.org/10.1111/j.2397-2335.1934.tb04184.x)及
[探索与重复采样的 IS 预算分析](https://doi.org/10.1080/24725854.2021.1953197)。

<a id="budget-pilot"></a>
## 5. 初始估计与最终估计分离

已有 `run_progressive_conditional_is` 固定候选数和块长，先为同一候选生成初始补全，估计标准差及相对成本，
然后由 `allocate_fresh_rollout_budget` 冻结每候选的最终补全数。设

```math
\ell_{ik}=r_{ik}/\tau+\log p(u_{ik}\mid g,z_i)-\log q(u_{ik}\mid g,z_i).
```

标准差由共用对数平移后的 $`e^{\ell_{ik}}`$ 估计；旧实现的相对成本由补全 token 数乘模型参数量估计，
off-policy 时另计基础模型重评分。最终条件权重只使用独立的新补全：

```math
\widehat W_i^{\rm final}
=\frac1{K_i^{\rm eval}}\sum_{k=1}^{K_i^{\rm eval}}e^{\ell_{ik}^{\rm eval}}.
```

给定候选和设计数据，样本数已固定、最终补全独立，因此使用精确概率比时条件权重无偏。
比值截断会改变这一性质。终止候选的确定性奖励可复用。初始轨迹可作为经过验证的推测解码草稿，不能重复当作独立统计观测。

联合调度还会改变候选数和块长，所以第 3 节连候选本身也重新独立生成。先查看最终奖励再决定何时停止，
或直接把设计样本混入最终平均值，均不自动满足上述证明条件。已有 replay 的独立性与校正要求见
[历史复用算法](ALGORITHMS.md#alg-base-replay)。

嵌套估计的非线性归一化和预算选择可参考
[On Nesting Monte Carlo Estimators](https://proceedings.mlr.press/v80/rainforth18a.html)及
[Bootstrap-based Budget Allocation for Nested Simulation](https://doi.org/10.1287/opre.2020.2071)。
本实现使用样本矩，不包含 bootstrap、训练得到的调度器或任务正确性标签。

<a id="budget-accounting"></a>
## 6. 预算单位与执行边界

### 预留成本

AR 联合调度器以 `forward_token_budget` 控制预留前向 token 位置数。设提示长度为 $`P`$，
当前已生成长度为 $`L`$，一次奖励最多需要 $`s`$ 次完整序列评分。非终止块采用

```math
c_z(B)=\max\{1,P+L+B\},\qquad
c_u(B)=(1+s)\max\{1,P+T\}.
```

每次补全包含重新预填充、解码以及奖励评分预留。终止块采用
$`c_z=(1+s)\max\{1,P+T\}`$、$`c_u=0`$。
始终保留 $`M_{\min}(1+s)\max\{1,P+T\}`$ 作为直接完成生成的成本；初始预算不足时在调用模型前报错。
EOS 提前结束和缓存命中不退回预留量，避免以随机长度改变已经冻结的样本集合。

这是一套明确的请求级预留账本，不是任意后端的实际 FLOPs 硬上限。自定义奖励应符合声明的
`reward_forward_passes`；分块评分重复预填充、额外模型调用、推测解码的未采用分支和后端内部实现
可能使实测成本与预留不同。当前 CLI 的两种模型奖励都预留一次评分；实际开销由后端计数器另外报告。

### 实测成本与公平比较

模型 $`j`$ 参数量为 $`N_j`$、实际前向 token 位置数为 $`S_j`$ 时，沿用

```math
\widehat F_{\rm forward}=2\sum_j N_jS_j.
```

预填充、解码、重评分、验证和批量填充均按后端实际执行记录；该估算省略注意力的长度平方项及逐元素计算。
墙钟、吞吐和显存独立测量。比较应包含初始估计、缓存建立和奖励评分，并分列主模型与辅助模型成本。
相同预留预算不代表相同实际 FLOPs；正式实验需同时给出二者。

联合调度当前仅接入 AR、同模型 on-policy、固定逐序列奖励和完整输出范围。Consilience 奖励仍默认只读 thinking，
切分失败沿用显式全序列 fallback；这与 IS 修改整个输出范围是两个独立设置。
thinking-only 采样的停止包装和正文续写、off-policy/replay、候选级不同 $`K_m`$、dLLM 块长适配均未接入这个新入口，
已有入口的相应功能保持不变。共享选择器不绑定模型族，但其他适配层需要提供有效的统计量、成本和独立采样。

<a id="budget-usage"></a>
## 7. 调用、参数与代码索引

### 独立入口

入口接受通用模型 ID 或本地目录，复用已有配置加载器、提示模板、输出解析和奖励工厂：

```powershell
python -m experiments.arllm.joint_budget_is `
  --model Qwen/Qwen3-1.7B --allow-download `
  --prompt "Find all positive integers n such that n squared plus 1 is divisible by n plus 1." `
  --thinking-mode enabled --sampling-scope full `
  --max-new-tokens 32768 --budget-forward-tokens 2000000 `
  --block-sizes 64 128 256 --candidate-counts 2 4 8 16 --rollout-counts 1 2 4 8 `
  --reward consilience --reward-temperature 2 --seed 0 `
  --output results/joint_budget/example.json
```

这是调用示例，未作为正式实验运行。`--config` 读取已有 TOML 的模型、后端、采样策略和奖励设置；
联合预算使用本入口的显式参数。`--backend transformers|vllm|vllm-sync` 选择后端；vLLM 须具备所选奖励所需的精确评分能力。
默认保持模型本身的上下文限制；实际有效长度和截断原因写入 JSON。`--reward sequence_log_probability` 使用按有效 token 数取均值的对数概率奖励，不计停止后的 padding。

| 参数 / 字段 | 含义 |
| --- | --- |
| `--budget-forward-tokens` | 包含初始采样和奖励评分的总预留量 |
| `--block-sizes` | 块长网格；运行时另加完整剩余长度 |
| `--candidate-counts`、`--rollout-counts` | 整数网格，分别为 $`M\geq2`$、非终止时 $`K\geq1`$ |
| `--pilot-candidates`、`--pilot-rollouts` | 每个被探测块长的初始样本数，均默认 2 |
| `--pilot-fraction` | 每轮初始估计最多使用当前剩余预算的比例，默认 0.15；还受完成预留量限制 |
| `JointBudgetISConfig.relative_variance_floor` | 调度用相对方差下限，默认 $`10^{-4}`$ |
| `JointBudgetISConfig.reward_forward_passes` | 自定义奖励的完整序列评分预留次数；CLI 固定为 1 |
| `steps[].plan` | 选中的 $`M,K,B_{\rm blk}`$、局部误差估计、累计预测分数和是否有对应初始观测 |
| `steps[].estimates` | 每个块长的相对方差、初始候选数和成本 |
| `reserved_forward_tokens`、`pilot_reserved_forward_tokens` | 总预留消耗及初始估计部分 |
| `actual_backend_cost`、`inference_seconds` | 实测前向计数、估算 FLOPs 与不含模型加载的执行时间 |

### Python 接口

```python
from inference_scaling.experimental.arllm.joint_budget_is import (
    JointBudgetISConfig, run_joint_budget_is,
)
from inference_scaling.shared.rng import SeedStream

result = run_joint_budget_is(
    backend, prompt_tokens,
    JointBudgetISConfig(forward_token_budget=2_000_000),
    reward, SeedStream(0), sampling=sampling,
)
```

`reward(prompt_tokens, complete_sequence_tokens)` 必须是固定逐序列函数。自一致性可以先固定一个独立参考池；
直接在当前候选池内重新统计多数标签会改变候选权重之间的依赖关系，不适用第 2 节证明。

| 职责 | 代码 / 测试 |
| --- | --- |
| 相对方差估计、联合整数选择 | [`shared/joint_budget.py`](../../src/inference_scaling/shared/joint_budget.py)：`estimate_weight_moments`、`choose_joint_budget` |
| AR 循环、独立随机数、完成预留 | [`experimental/arllm/joint_budget_is.py`](../../src/inference_scaling/experimental/arllm/joint_budget_is.py)：`JointBudgetISConfig`、`block_costs`、`run_joint_budget_is` |
| 实际候选生成、补全和重采样 | [`arllm/algorithms/conditional_is.py`](../../src/inference_scaling/arllm/algorithms/conditional_is.py)：`conditional_is_step` |
| CLI、通用加载与实际成本输出 | [`experiments/arllm/joint_budget_is.py`](../../experiments/arllm/joint_budget_is.py) |
| 历史/新样本方差—成本分配 | [`shared/budget.py`](../../src/inference_scaling/shared/budget.py)：`allocate_variance_cost_budget`、`allocate_fresh_rollout_budget` |
| 固定候选的两阶段估计 | [`AR progressive_is.py`](../../src/inference_scaling/experimental/arllm/progressive_is.py)、[`dLLM progressive_is.py`](../../src/inference_scaling/dllm/algorithms/progressive_is.py) |
| 动态候选和外层概率校正 | [`AR dynamic_is.py`](../../src/inference_scaling/experimental/arllm/dynamic_is.py)、[`dLLM dynamic_is.py`](../../src/inference_scaling/dllm/dynamic_is.py) |
| 前向 FLOPs 估算 | [`shared/compute.py`](../../src/inference_scaling/shared/compute.py)：`dense_forward_flops` |
| 矩估计、联合决策、精确枚举 TV 检查 | [`test_joint_budget.py`](../../tests/test_joint_budget.py) |
| 预算、EOS、随机数隔离与分布测试 | [`test_joint_budget_is.py`](../../tests/test_joint_budget_is.py) |
| CLI 及后端适配测试 | [`test_joint_budget_cli.py`](../../tests/test_joint_budget_cli.py) |

```powershell
python -m pytest -q tests/test_joint_budget.py tests/test_joint_budget_is.py tests/test_joint_budget_cli.py
```

小样本方差可能低估稀有分支，完整序列备选可能被频繁选中，初始估计也可能抵消调度收益。
现有固定候选方差—成本方案的成本结果见[执行成本报告](../reports/RTX3090_ROLLOUT_INFRA.md#infra-report-budget)；
这些历史结果不代表本次联合调度的效果。
