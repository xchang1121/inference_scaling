# inference_scaling

本仓库用于统一研究由基座语言模型诱导出的推理时采样分布。所有实验共享一套后端接口，包含四类算法：

1. 面向幂分布目标的后缀重采样 Metropolis--Hastings（MH）；
2. on-policy 条件能量重要性采样；
3. 保持候选来自基座模型、带 fresh-tail 修正的 off-policy rollout replay；
4. 动态候选 proposal、候选层重要性采样与方差—成本预算分配。

实现分为两个阶段。提交信息以 `basic implementation:` 开头时，目标是先忠实实现数学算法；以
`optimization:` 开头时，目标是在不暗中改变目标分布的前提下加入调度、缓存复用、向量化评分和
replay 系统优化。

## 设计约束

- base replay 算法的候选块始终由基座模型生成。
- 每条历史 rollout 都保存实际 behavior policy 及其真实采样概率，包括温度和截断设置。
- 当前决策使用过的数据不会重新进入未来的 evaluation pool。只有在决策完成后独立生成的 reserve
  rollout 才能进入该池。
- 异步执行使用请求级随机数流，并以 FP64 累加大词表 inverse-CDF；每次 benchmark 仍逐方法检查调度
  前后的 token 输出是否完全一致。
- 会改变估计器或 Markov 核的算法改动，与保持分布不变的系统优化分别标注。

## 开发环境

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
```

先安装支持 CUDA 的 PyTorch wheel，再安装本项目验证过的推理与训练依赖：

```powershell
# 选择本机 NVIDIA 驱动支持的 CUDA 运行时对应的 PyTorch 源。
.\.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python -m pip install -e ".[dev,gpu,training]"
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

PyTorch wheel 自带 CUDA 运行时；只有编译 CUDA 扩展时才需要单独安装 CUDA Toolkit。因此，`nvcc`
显示的版本不必与 `torch.version.cuda` 相同。原始模型权重和大型实验产物不会提交到 Git。

Transformers 后端默认使用 FP32。在实测 RTX 3090 上，BF16 logits 会随 batch 形状产生足以影响
生成时概率与随后批量重评分一致性的变化。低精度仍可用于吞吐实验，但若重要性权重依赖精确概率，
应先在目标模型和硬件上验证一致性；未经验证时使用 FP32。

## 统一的公开基准复现

实验套件使用 OpenAI 公开的 GSM8K，而不使用人工编造问题、付费偏好评审、受限数据集或不安全的
代码执行。GRPO 仅使用官方训练集的 7,473 条样本，其 SHA-256 为
`17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465`。所有准确率样本都来自官方
测试 split；`full` 配置使用全部 1,319 条，`standard` 和消融按预先固定的 seed 选择子集。测试文件
SHA-256 为
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。训练入口会核验两个文件的
哈希；若训练集和测试集出现完全相同的问题，也会立即终止。

统一使用同一模型家族：

- 主分布：`Qwen/Qwen2.5-1.5B-Instruct`，固定 revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；
- 低成本 off-policy rollout proposal：`Qwen/Qwen2.5-0.5B-Instruct`，固定 revision
  `7ae557604adf67be50417f59c2c2f167def9a775`；
- RL 对照：从上述同一个 1.5B checkpoint 出发，在固定的 GSM8K 训练集上本地训练 GRPO LoRA。
  `configs/gsm8k_grpo.toml` 固定奖励、prompt、优化器、LoRA、rollout 和 checkpoint 设置。

准备固定版本的数据和模型：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\prepare_gsm8k.py `
  --config configs\gsm8k_standard.toml
```

训练 GRPO 对照。默认配置对每个 prompt 生成四条回答，每个优化步累积四组 prompt，最大生成 192 个
token，共进行 205 个优化步；该步数与来源实验处于同一量级，每 25 步保存一个可恢复的 checkpoint：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\train_gsm8k_grpo.py --resume auto
```

输出 adapter 位于 `models/Qwen2.5-1.5B-Instruct-GRPO-GSM8K`。`run_manifest.json` 记录模型、数据版本
以及训练集/测试集零重合检查；`training_cost.json` 记录生成的 rollout 数和模型实际处理的 token。
wall time、峰值 CUDA 显存、采样得到的 GPU 功率及积分能耗仅作为硬件相关诊断。

本机已完成的训练及端到端加载检查记录在
`results/gsm8k_grpo_training_summary.json`。这次运行由前 100 步和从 `checkpoint-100` 继续的 105 步
组成，因此摘要把它明确记为两段调度，不把它解释成一次未中断的 205 步学习率轨迹。累计计算量、
rollout 数、adapter 哈希和 FP32 概率一致性检查均来自实际产物；训练 rollout 准确率只用于检查训练
过程，不能代替保留测试集上的方法比较。

先运行八条样本的集成检查。实验问题与比较结构围绕条件采样方法；主表沿用其单次最终回答
（pass@1）口径，比较 Base、Beam
Search、Best-of-N、Power Sampling 对应的幂分布 MH、条件能量重要性采样、小 proposal 加速版本与
GRPO；模型、数据集和适合 24 GiB 单卡的预算允许调整。每个命令都能从单条 GSM8K 记录处恢复；
manifest 同时固定配置、样本行号、模型权重和关键实现文件的 SHA-256，代码变化后不会混用旧记录：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\run_gsm8k_suite.py `
  --config configs\gsm8k_quick.toml `
  --with-replay `
  --with-async `
  --with-matched-target

.\.venv\Scripts\python experiments\summarize_gsm8k.py `
  --config configs\gsm8k_quick.toml `
  --output results\gsm8k_quick_comparison.json
```

本机已完成的 `validated` quick 集成检查见 `docs/GSM8K_QUICK_VALIDATION.md`。其中主方法与 replay
使用 8 条固定测试题，异步调度使用 32 条；这些结果用于验证代码路径和指标口径，不代替 128 条
standard 主实验。

在相同公开子集上复现 Base、幂分布 MH 与 GRPO 的多样性和 pass@k；这里使用 quick 预算、32 条固定
测试样本及 8 个独立 draw，避免把完整 128 条主表的高成本 MH 重复八遍：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\gsm8k_passk.py `
  --config configs\gsm8k_quick.toml `
  --limit 32 `
  --draws 8 `
  --tag passk
```

pass@k 使用每题 \(n\) 次独立采样中答对 \(c\) 次时的标准估计
`1 - choose(n-c,k)/choose(n,k)`；多样性另报告不同的可解析最终数值答案数，不把它冒充完整 token
序列多样性。

在单张 RTX 3090 上运行预注册的 128 条样本比较与算法消融：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\run_gsm8k_suite.py `
  --config configs\gsm8k_standard.toml `
  --with-replay `
  --with-async `
  --with-matched-target `
  --with-ablations `
  --with-budget-curve `
  --with-length-ablation `
  --ablation-limit 32

.\.venv\Scripts\python experiments\summarize_gsm8k.py `
  --config configs\gsm8k_standard.toml `
  --output results\gsm8k_standard_comparison.json

.\.venv\Scripts\python experiments\gsm8k_distribution_audit.py `
  --config configs\gsm8k_standard.toml `
  --problem-count 4 `
  --draws 16 `
  --output results\gsm8k_standard_distribution_audit.json

.\.venv\Scripts\python experiments\summarize_gsm8k_compute.py `
  --config configs\gsm8k_standard.toml `
  --training-cost models\Qwen2.5-1.5B-Instruct-GRPO-GSM8K\training_cost.json `
  --distribution-audit results\gsm8k_standard_distribution_audit.json `
  --output results\gsm8k_standard_compute.json

.\.venv\Scripts\python experiments\summarize_gsm8k_ablations.py `
  --config configs\gsm8k_standard.toml `
  --output results\gsm8k_standard_ablations.json
```

若要先在单张 RTX 3090 上完成同一测量结构、但缩放数据量和超参数，使用 32 条固定测试题的
`configs/gsm8k_3090_aligned.toml`。该配置保留 Base、Beam、Best-of-N、MH、标准/小 proposal 条件
IS、GRPO、共同 exact-verifier 目标、replay、异步、pass@k 及相同消融维度；它使用 192 token、
Beam/Best-of-8、8 个候选、每候选 3 条 rollout，以及 16 个 MH 长度阶段、每阶段 3 次更新：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\run_gsm8k_suite.py `
  --config configs\gsm8k_3090_aligned.toml `
  --tag validated `
  --with-matched-target `
  --with-replay `
  --with-async
```

这里缩放的是模型、样本数和算法预算，不改变要测的问题、方法对照、目标定义或 token/FLOPs 口径。
`gsm8k_standard.toml` 与 `gsm8k_full.toml` 仍分别保留 128 条和全部 1,319 条的更大预算，可从已有逐题
记录继续运行。

`configs/gsm8k_standard.toml` 使用 128 条固定测试样本、最大生成 256 token、20 beams、Best-of-20、
15 个候选、每个候选 3 条 rollout 和 4 个引导步。`configs/gsm8k_full.toml` 覆盖官方测试集全部
1,319 条样本，并采用来源主表的较大 GSM8K 预算：最大生成 512 token、20 beams、Best-of-30、
15 个候选、每个候选 3 条 rollout 和 4 个引导步。幂分布 MH 在两种配置中都把总长度划为 16 个
递增长度阶段，每阶段进行 10 次更新。单张 RTX 3090 上完整运行耗时很长，但所有原始记录都只追加
写入，可以用同一命令中断后继续。

主要比较包括 Base、Beam、Best-of-N self-consistency、后缀重采样 MH、条件重要性采样、小模型
off-policy 条件重要性采样，以及本地 GRPO adapter 的采样/贪心输出。两种条件采样方法的候选块都
来自主模型。加速版本只替换 completion rollout 生成器，并在权重中加入精确的主模型/proposal
后缀 log 概率比。来源对齐配置把该 log 比值截到 `[-10, 10]`，以有限偏差换取小 $K$ 下的方差
稳定性；每条记录同时保存截断前后修正值和被截断数量。将
`importance_log_ratio_clip` 设为空即可运行不截断的理论版本。
`--with-ablations` 会同时运行截断与不截断版本，直接报告这项方差—偏差取舍。

奖励设计消融也按条件采样的比较结构运行在 Best-of-N、标准条件采样和小 proposal 条件采样上，包括
平均 token 对数概率、平均负熵、自确定性、自一致性和测试集正确答案五种信号。前三个连续信号在每次
候选决策内做 min-max 归一化；自一致性仍跨引导步累计答案计数；正确答案奖励始终标记为 oracle，
不会混入可部署主表。连续奖励的额外基座模型评分会计入 token slot 与 FLOPs。

计算报告比较“一次 GRPO 训练 + 重复进行 GRPO 推理”和“无需训练、每次使用 MH/IS 推理”。主要
计算单位是实际 forward token slot 与估算的主导稠密矩阵 FLOPs。每个推理模型分别贡献
`2 * parameter_count * forward_token_slots`；GRPO 还计入 rollout 生成、参考模型评分、包含
gradient-checkpoint 重算的策略前向/反向，以及 LoRA 参数上的 AdamW 更新。GRPO 的 padded token slot
由每步累计 token、平均 completion 长度和 microbatch 最大 completion 长度重建，不把 padding 成本
藏在 wall time 中。主要的摊销临界查询数为

`ceil(training_FLOPs / (method_FLOPs_per_query - GRPO_FLOPs_per_query))`。

只有当实测准确率差不超过给定容差时，报告才会给出“准确率匹配”的临界点。若同时提供分布审计，
还必须让经验最终答案分布的 total variation 与 Jensen--Shannon divergence 都低于阈值，才会给出
“准确率与答案分布联合匹配”的临界点。wall time、显存和实测训练能耗仅作为补充诊断。

分布审计会重复随机解码，并报告经验最终答案分布之间的 Jensen--Shannon divergence 与 total
variation。有限次采样不能恢复完整 token 序列分布的差异，报告不会作这种声明。

为了让 GRPO、MH 与 IS 真正共享 `base * exp(reward / beta)` 目标，`verifier_*` 诊断会读取 GSM8K
测试集 gold answer。它们是计算量与分布的 oracle 对照，不是可部署的无监督算法；可部署主表仍使用
self-consistency 奖励。

所有计算缩减或耗时加速都明确给出分母：

- `compute_multiple_vs_base`：某方法的估算 FLOPs 除以相同样本上单次 Base 采样的估算 FLOPs；
- 小 proposal 的 FLOPs 比值：标准 on-policy 条件 IS 的 FLOPs 除以小 proposal off-policy 条件 IS
  的 FLOPs，并固定候选数、rollout 数、block、prompt、seed 和输出长度；比值大于 1 才表示缩减，
  小于 1 则明确报告为精确重要性修正带来的计算增加；
- 异步耗时加速：分别对 Base、Best-of-N、条件 IS 和小 proposal 条件 IS，使用逐 prompt 同步耗时
  除以相同配置与请求 seed 的连续批处理耗时；四种方法使用同一调度优化。报告逐方法给出精确 token
  匹配率、数值答案匹配率、共同前缀比例和分叉题号；若 CUDA batch 形状使长采样路径分叉，该比率只
  表示真实 workload 的墙钟对比，不冒充固定 token trace 的严格成对计时。该优化改善硬件利用率，
  不宣称减少算法 FLOPs；
- 重复前缀 KV 复用：相对“每条 rollout 都重新计算完整前缀”，逐个不同候选前缀只计算一次并复制
  KV；`shared_prefill_tokens_saved` 报告由此避免的非 padding 前缀 token；
- replay 在线 FLOPs 缩减：使用 `H+F` 条新 base rollout 的 fresh-only 计算量，除以已有 `H` 条
  off-policy 历史数据、只生成 `F` 条 fresh rollout 的在线计算量；历史 rollout 的 base/behavior
  概率在 cache 构建时已经一次性验证并保存，在线账本仍计入 fresh 样本所需的 behavior 评分；比值
  大于 1 才表示在线计算下降；
- replay 单次端到端 FLOPs 缩减：分母额外包含历史缓存构建成本，因而不会把 warm-cache 在线收益
  冒充第一次查询即可获得的收益；cache 构建包括历史生成以及历史的 base/behavior 概率评分；比值
  小于 1 时表示第一次查询总成本更高；
- 对应 wall-time 比率保留为同一硬件上的补充结果，并使用相同分母定义。

replay 性能实验把 GSM8K 公开答案当作固定 verifier，结果不会混入 self-consistency 准确率表。实验
重复同一个公开 prompt 和候选 seed，每条历史记录最多使用一次，并以实际使用的历史比例报告 rollout
复用率。详细公平性约束和完整消融矩阵见
[`docs/GSM8K_EXPERIMENT_DESIGN.md`](docs/GSM8K_EXPERIMENT_DESIGN.md)。

## 已实现内容

- MH 路径实现固定长度的分阶段算法，包括在所有后缀起点中均匀采样、完整四项 log-probability 接受
  比、从同一份生成 logits 同时记录 proposal 与基模概率、当前状态 token 分数缓存、独立链和接受率
  诊断；因此不会为了接受率再重跑一次相同后缀的模型前向。
- 精确表格测试枚举目标幂分布，并检查采样器的经验输出。
- 条件能量路径只从基座模型生成候选块，同时支持 on-policy completion 和具有完整 support 的
  off-policy rollout 模型。重要性比只作用于 completion 后缀，随机权重统一在 log 域聚合。
- `base-replay` 实现只使用元数据的设计冻结、单次使用的 evaluation record、behavior-mixture 分母、
  带独立 fresh-tail 修正的截断历史比，以及选择后 reserve record。当前决策的新样本和已消费历史
  都会不可逆地移入 design pool。
- `dynamic-is` 加入 defensive candidate mixture、精确的候选层概率比以及冻结后的 history/fresh 联合
  分配。默认冷启动分配可替换为基于 design pool 的经验方差和 token 成本估计器。
- `ContinuousBatchingBackend` 合并并发 prompt 的候选、rollout 与评分请求，同时保持请求级 seed 和
  原始结果顺序；其计数器记录实际形成的 batch 大小。
- `ScoreCachingBackend` 只有在模型、采样策略、prefix 和 continuation 全部一致时才复用 base/behavior
  分数；随机生成永不缓存。
- replay 的 fresh completion 和选择后 reserve completion 会跨候选展平成一个后端 batch，但保持
  各自的 seed 与 replay key。
- `TransformersBackend` 提供与真实采样策略一致的批量解码与重评分、请求级随机数流、KV cache 解码、
  多组重复 prefix 各自只做一次 prefill 后复制 KV、仅计算所需尾部 vocabulary logits，以及 forward
  token slot/FLOPs 计数。
- `gsm8k_passk.py` 对 Base、MH 与 GRPO 使用相同公开问题和独立 seed，报告标准 pass@k、解析答案
  多样性及累计 token/FLOPs。

运行独立的有限状态 smoke 实验：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\toy_mh.py
.\.venv\Scripts\python experiments\toy_conditional_is.py
.\.venv\Scripts\python experiments\toy_base_replay.py
.\.venv\Scripts\python experiments\toy_dynamic_is.py
```

在 NVIDIA GPU 上运行固定版本的真实模型复现：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\rtx3090_reproduction.py `
  --model models\Qwen2.5-0.5B-Instruct `
  --dtype float32 `
  --output results\rtx3090_reproduction.json
```

RTX 3090 的已核验测量与解释见
[`docs/RTX3090_REPRODUCTION.md`](docs/RTX3090_REPRODUCTION.md)。

## 仓库结构

- `src/inference_scaling/`：可复用算法、后端、调度器、replay 存储和指标；
- `configs/`：纳入版本控制的实验配置；
- `tests/`：精确表格测试与集成测试；
- `experiments/`：命令行实验入口；
- `docs/`：算法映射和复现报告；
- `results/`：只提交小型汇总，原始输出保留在本地。
