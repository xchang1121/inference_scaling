# GSM8K 统一复现实验设计

## 实验范围

本复现围绕条件采样的实验问题和比较结构，并用同一个公开且可自动评分的任务
OpenAI GSM8K 替代来源文章中彼此不同的评测栈。GRPO 只
使用 7,473 条官方训练样本；所有准确率样本都来自 1,319 条官方测试 split，`full` 使用全部样本，
`quick`、`standard` 与消融使用预先固定的子集。两个文件都固定字节级校验和，训练入口还会验证
训练集与测试集没有完全相同的问题。评测中不包含人工编造的问题。

本实验复现的是定性结论，并不声称复现来源文章中的具体数值。比较方法、单次最终回答（pass@1）
主指标、质量—成本曲线、加速分母及主要消融均与条件采样实验对齐；模型、公开任务和适合 24 GiB 单卡的
超参数可以调整。仓库中的 `quick`、`gsm8k_3090_aligned.toml`、`standard` 和 `full` 配置分别对应
8 条样本的集成检查、32 条固定题的单卡对齐实验、128 条固定题的较大预算实验，以及完整公开测试集。

3090 对齐配置不删减方法、目标对照或消融维度，只把最大长度改为 192、Beam/Best-of-N 改为 8、
候选数改为 8，并把 16 个 MH 长度阶段的每阶段更新数改为 3。所有差异均由 manifest 固定，结果不会
与 `standard` 或 `full` 的更大预算混用。

这里的“对齐”首先要求回答相同类型的问题：条件能量方法相对 Base、Beam Search 和 Best-of-N 的
质量—计算关系，小 proposal 相对标准条件采样的速度与复用收益，连续批处理相对逐请求执行的
wall-time 收益，以及候选数、rollout 数、引导轮数、奖励、温度和输出长度等因素的影响。幂分布 MH
和本地 GRPO 是额外的统一对照，用来检验 training-free 重分配与训练方法能否得到相近效果，并比较
达到相近效果所需的 token slot 与 FLOPs。

## 固定比较约定

主模型为 `Qwen/Qwen2.5-1.5B-Instruct`。RL 对照是在同一个冻结 checkpoint 上训练得到的本地 GRPO
LoRA，唯一奖励与评测时使用的精确数值正确性函数相同。奖励标准差缩放被关闭，使原始奖励尺度与
KL 系数之间仍有明确关系。低成本 rollout proposal 为 `Qwen/Qwen2.5-0.5B-Instruct`。两个下载的
权重文件都使用固定 SHA-256 核验；全部 GRPO 设置固定在 `configs/gsm8k_grpo.toml` 中。

在同一个实验配置下，所有主要方法都使用：

- 预先按固定 seed 选定的完全相同 GSM8K 行号；
- 完全相同的回答 prompt 与数值解析器；
- 完全相同的最大新增 token 数；
- 由公开样本行号派生的请求级随机 seed；
- 排除模型和数据加载、且在起止处同步 CUDA 的推理计时。

| 方法标识 | 分布或决策规则 | 主要比较对象 |
| --- | --- | --- |
| `base` | 温度 1 的一次主模型采样 | 单次推理基线 |
| `beam` | 确定性 beam search | 条件能量方案中的搜索基线 |
| `best_of_n` | 独立 base 采样，选择众数数值答案 | 并行采样基线 |
| `mh` | 固定长度、EOS 吸收状态下针对 \(p^4\) 的后缀重采样 MH | Base 与 RL sample |
| `conditional_is` | base 候选、base rollout 与累积 self-consistency 奖励 | Best-of-N 与 RL greedy |
| `conditional_is_small_proposal` | 相同候选和决策预算；用 1.5B/0.5B 后缀概率比修正 0.5B rollout | 标准 `conditional_is` |
| `rl_sample` | 从 GRPO checkpoint 进行一次温度 1 采样 | MH |
| `rl_greedy` | GRPO checkpoint 的贪心输出 | 条件 IS |
| `verifier_mh` | 针对 `base * exp(exact reward / beta)` 的完整序列后缀 MH | 与 GRPO 共享目标的比较 |
| `verifier_conditional_is` | 使用精确奖励与 GRPO beta 的条件 IS | 与 `rl_sample` 共享目标的比较 |
| `verifier_conditional_is_small_proposal` | 对上一目标使用经过修正的 0.5B rollout | 与 `rl_sample` 的 off-policy 比较 |

GRPO 与推理时采样对应两个相关但不同的问题。第一张表只检验它们能否在公开基准上达到相近准确率，
不声称 self-consistency IS 与正确性奖励训练的 GRPO 定义了相同的序列分布。涉及分布匹配的比较必须
使用同一个显式奖励，并报告经验答案分布诊断；结果报告会保留这一区别。

三个 `verifier_*` 方法在这个受控实验里直接读取测试集 gold answer，因而是“共享目标的 oracle
诊断”，不是可部署的无监督方法。它们只用于回答 GRPO、MH 和 IS 在同一个显式奖励目标下需要多少
计算；实际可部署的 training-free 主表仍使用 self-consistency 奖励。

奖励消融额外比较 条件采样使用的五类信号：完整回答的平均 token 对数概率、逐 token 预测分布的平均
负熵、自确定性、自一致性和正确答案。前三者都由主模型在完整回答上重新评分，并在每次候选决策的
全部回答内做 min-max 归一化；若所有值相同，则统一置零，因为相同的加性奖励不会改变该次选择。
自确定性按每个位置的 (D_{\mathrm{KL}}(U\|p_{\mathrm{base}})) 计算，其中 (U) 是词表上的均匀分布。
这些额外评分全部进入 token-slot/FLOPs 账本。正确答案仍只作为显式 oracle。

`verifier_mh` 从一条完整的 base 序列初始化。每次更新都在所有可能的后缀起点中均匀抽取一个，使用
具有完整 support 的 base policy 提议新后缀，并计算完整的目标/proposal 比。该转移核保持上述完整
序列目标不变；block size 只控制每轮的更新次数，不会排除任何后缀起点。

`standard` 主表使用 20 beams、Best-of-20，以及 $M=15,K=3,I=4$ 的条件采样；最大回答长度缩短为
256 token，以便在单张 RTX 3090 上重复运行。`full` 使用 512 token、20 beams、Best-of-30 和
$M=15,K=3,I=4$，对应来源实验在 GSM8K 上的主要算法结构与预算量级。两者的幂分布 MH 都采用
目标幂次 α=4、16 个递增长度阶段及每阶段 10 次更新；这里的 α 是 (p_{\mathrm{base}}^\alpha)
中的幂次，等价于来源实现所写的逆温度 0.25。主表的条件采样温度设为 1，使目标中的 `base` 就是
未经温度修改的基座分布；温度 0.7、1.0 和 1.5 另作消融。对任意非 1 温度，代码把温度缩放后的完整
支持采样策略明确视为参考分布，并在小 proposal 权重中使用相同温度下的精确后缀概率比。

小 proposal 的来源对齐配置先精确计算后缀 log 概率比，再把它截到 `[-10,10]`。这与参考实现的
实用稳定化设置一致，但会引入有限偏差；结果记录同时报告原始修正、实际修正和截断次数。关闭
`importance_log_ratio_clip` 时恢复不截断的重要性权重。因而报告会把该方法称为截断的有限 rollout
近似，而不会把它写成有限样本下严格无偏。

## GRPO 训练与计算量摊销

在共享目标的受控比较中，每个 prompt 的参考目标是

`maximize_pi E_pi[R] - beta * KL(pi || p_base)`。

不受参数化限制时，其最优解正比于 `p_base * exp(R / beta)`。GRPO 是这一目标的有限步、有裁剪的随机
优化，因此只近似该解；verifier-MH 以上述分布为平稳分布；verifier-IS 则用有限候选和有限 rollout
近似它的条件因子。因此报告同时测量准确率和经验答案分布距离，不会直接宣称有限计算下三者相同。

GRPO 进行 205 个优化步，并每 25 步保存可恢复 checkpoint。manifest 记录选中的公开训练行、基座权重哈希、软件包
版本、LoRA 参数量、生成 completion 数及 token 数，以及 trainer 实际观察到的 prompt+completion
token。峰值 CUDA 显存、同步 wall time、GPU 功率采样与积分能耗只作为硬件相关诊断。测试集仅用于
哈希与重合检查，从不传给 `GRPOTrainer`。

主要计算单位为 forward token slot 和估算的主导稠密矩阵 FLOPs。一个 forward token slot 表示一次
实际提交给模型的输入位置；重复的 prompt 或完整序列概率评分会再次计数。推理时，每个模型分别贡献

`2 * model parameter count * observed forward token slots`

FLOPs，1.5B 与 0.5B 的贡献分别计算后相加。GRPO 的计算拆为 rollout 生成、参考模型评分及策略
前向/反向。启用 gradient checkpointing 时，策略更新计为三个前向等价过程：前向、反向与重算。
每步累计 token、平均 completion 长度和 microbatch 最大 completion 长度共同重建实际 padded token
slot；AdamW 的小额开销只计到可训练 LoRA 参数。二次 attention、逐元素 kernel、tokenization、采样
与主机工作被明确列为未计项，不会用耗时代理冒充。

因此，MH 和两种条件 IS 与 GRPO 的主要摊销比较为

`GRPO training FLOPs + query count * GRPO inference FLOPs`

对比

`query count * training-free inference FLOPs`。

报告也给出原始 token-slot 临界点；但使用 0.5B proposal 时，FLOPs 更能体现不同模型大小。只有当
实测准确率差落在预设容差内时，才报告“准确率匹配”的临界查询数；若提供重复采样的答案分布审计，
还要同时通过 total variation 与 Jensen--Shannon divergence 阈值，才报告联合匹配临界点。有限答案
样本不能证明完整 token 序列分布相同。wall time、显存和实测训练能耗仅作补充；不会由 wall time
虚构推理能耗。

MH adapter 把 EOS 视为吸收 token。解码后的回答保持不变，但状态空间成为固定长度，当前后缀与提议
后缀都有显式概率，从而消除在固定长度 MH 证明里嵌入变长生成的歧义。

## 计算量与计时分母

每个计算缩减或 wall-time 加速都明确说明分母：

- `compute_multiple_vs_base`：某方法的估算稠密 FLOPs 除以完全相同样本上单次 Base 的估算稠密
  FLOPs。
- `standard_over_small_proposal_flop_factor`：标准 on-policy 条件 IS 的 FLOPs 除以小 proposal
  off-policy 条件 IS 的 FLOPs；候选、rollout、block、prompt、seed 和输出预算全部固定。大于 1
  才表示计算量下降，小于 1 表示为了精确重要性修正反而增加了 FLOPs。
- `runtime_multiple_vs_base`：某方法 wall time 除以相同样本上的 `base` wall time。它是硬件相关的
  补充成本倍数，不是计算量指标。
- 小 proposal wall-time 加速：标准 on-policy 条件 IS 耗时除以小 proposal off-policy 条件 IS
  耗时。只有 rollout 生成器和精确重要性修正发生变化；该比值大于 1 才称为加速。
- 异步加速：分别对 Base、Best-of-N、条件 IS 和小 proposal 条件 IS，以逐 prompt 同步耗时除以
  相同请求和 seed 的连续批处理耗时。四种方法统一使用这一调度方式；它衡量硬件利用率，不宣称减少
  算法 FLOPs。请求级 seed 固定随机流，但不同 CUDA batch 形状仍可能造成轻微 logits 差异；因此同时
  报告精确 token 匹配率、最终答案匹配率、共同前缀比例和分叉题号。输出不完全相同时，wall-time
  speedup 只表示真实 workload 对比，而不是固定 token trace 的严格成对计时。
- 重复前缀 KV 复用：分母是同一个生成 batch 对每条 rollout 分别重算完整 prefill 的前缀 token；
  分子只对每个不同的“prompt + 候选”前缀计算一次，再复制 KV 状态。结果直接报告没有重复处理的
  非 padding 前缀 token 数，不把它与连续批处理的 wall-time 收益混为一项。
- replay 在线 FLOPs 缩减：fresh-only 使用 `H+F` 条新 base rollout 的 FLOPs，除以 warm replay 已有
  `H` 条 off-policy rollout、只生成 `F` 条 fresh base rollout 的在线 FLOPs。候选来源、候选数、
  block size 和 `H+F` 都固定。历史 completion 的 base/behavior 概率在 cache 构建时一次性验证并
  保存；在线账本仍包含 fresh rollout 所需的 behavior 概率计算。该比值大于 1 才表示在线计算下降。
- replay 单次端到端 FLOPs 缩减：分母包括 cache 构建与 warm 决策。在线 warm-cache 收益不会冒充
  第一次查询即可获得的收益；cache 构建账本包含历史生成及上述 base/behavior 评分，wall-time 比率
  使用完全相同的两种分母。若该比值小于 1，就明确表示第一次查询的总成本更高。

replay 中每条历史记录最多使用一次。性能 benchmark 重复同一个公开 prompt 与候选 seed，保证历史
缓存中确实存在相同候选 key；因此它测量重复查询缓存，不代表跨无关 prompt 复用。该独立性能实验
把 GSM8K gold answer 当作固定 verifier，其准确率不混入 self-consistency 主表。

## 消融实验

实验入口包含以下定性检查：

- MH 目标幂次 1、2、4、8；
- 每个 MH block 更新 1、2、5、10 次；
- 固定 (K=3) 时取 (M=3,5,10,15)，并固定 (M=10) 比较 (K=1,3,5)；
- 固定总长度时取 2、4、8、16 个条件引导步；
- Beam Search、Best-of-N、条件 IS 和小 proposal 条件 IS 的质量—计算曲线；
- 平均 token 对数概率、平均负熵、自确定性、self-consistency 与测试集正确答案五种奖励；oracle
  结果单独标记，绝不作为可部署结果；
- 小 proposal 的不截断理论权重与 `[-10,10]` 实用截断；
- 采样温度 0.7、1.0、1.5，以及 Base、Beam Search、Best-of-N、两种条件方法和 GRPO 的最大回答
  长度 128、256、512；
- Base、Best-of-N、条件 IS 和小 proposal 条件 IS 的同步执行与连续批处理；
- fresh-only 与 warm off-policy rollout replay。
- Base、MH 与 GRPO 在 8 个独立 draw 下的标准 pass@k、题目 bootstrap 区间，以及解析答案和完整
  输出哈希多样性。三种方法使用同一连续批处理 worker 数；每个固定任务块单独保存实际 padded
  token slots、估算 FLOPs 与墙钟，不能把异步吞吐收益写成算法计算量下降。

每条原始结果只追加写入 JSONL。分布审计也按“方法 × draw × 题目”逐样本落盘，不等待整个方法结束。
manifest 会对有效配置和选中的 GSM8K 行号取 fingerprint，因此恢复
运行时不会把另一组实验结果静默追加到当前目录。fingerprint 还包含实际输入权重与关键实现文件的
SHA-256；算法或后端代码改变后必须使用新 tag 重新运行，不能沿用旧记录。
