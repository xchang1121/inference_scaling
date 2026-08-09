# 算法映射

实现对应 `HW_share/inference_scaling_article.tex` 中的数学对象。

| 框架标识 | 文档标签 | 必须保持的性质 |
| --- | --- | --- |
| `mh` | `alg:main-power-mh` | 后缀 MH 保持固定长度的幂分布目标不变 |
| `conditional-is` | `alg:main-onpolicy-is` | 候选与 completion 均从基座模型采样 |
| `base-replay` | `alg:main-replay-is` | 候选仍是 base 样本；replay 只用于估计条件能量 |
| `dynamic-is` | `alg:main-dynamic-is` | 辅助候选必须乘外层 base/proposal 概率比 |

replay 算法严格实现文档规定的数据生命周期：

1. design data 可以用于选择策略、方差估计和整数预算；
2. 设计冻结前不能读取 evaluation record 的数值，每条 evaluation record 最多消费一次；
3. 当前决策使用的每条 fresh rollout 都会转入 design data；
4. 只有在选择完成后独立生成的 reserve rollout 才能成为未来的 evaluation record。

动态实现先使用文档中的连续预算分配，再进行确定性整数舍入：候选层概率比同时乘到 history 和 fresh
的方差项上，每个来源还要除以其单样本成本的平方根。

动态 guidance step 在抽出候选后分成两个明确阶段。`design_prepare` 只接收候选对应的设计上下文，
可把独立 design rollout 和概率评分跨候选批量执行；随后 `statistics_provider` 只读取 design pool。
`rollout_budget_provider` 只能根据本轮候选、终止标记、逐候选容量与相同 replay key 共享的总库存冻结
成本预算，不能读取 evaluation completion 或 reward。重复候选必须共同遵守一次性库存上限。默认路径
仍使用配置中的固定预算，这两个接口只把文档允许的“先看元数据、再冻结设计”变成可测试的实现约束。

GSM8K 实验把各项基线映射为 `experiments/gsm8k_reproduction.py` 中的中性实现标识。
`conditional_is` 使用联合的累积 self-consistency 奖励；可复用估计器仍支持普通的固定逐序列奖励。
实验入口还可把该奖励替换成平均 token 对数概率、平均负熵、自确定性或正确答案 oracle，用于奖励
设计消融；这些替换不改变候选和 rollout 的概率修正公式。
动态候选的正式对照位于 `experiments/gsm8k_dynamic_is_benchmark.py`：固定组用于隔离候选 proposal 与
外层 IS，最优组才读取独立 design pool 并改变 rollout 配额。两组都记录 cache、design、稳定在线与
冷启动账本，不能用缓存命中率代替实际 FLOPs。
小 proposal 路径只改变 completion 的生成方式，并加入主模型/proposal likelihood ratio；候选块仍由
主模型生成。`AbsorbingEOSBackend` 提供 MH 所需的固定长度状态空间，同时不会把 chat prompt 内与
EOS 相同的 token 误判为生成终止。

条件算法允许把任意温度大于 0、且不做 top-k/top-p 硬截断的主模型策略定义为本次实验的参考分布。
候选与 on-policy rollout 使用同一策略；小 proposal 路径则在 completion 后缀上计算同一温度下的
精确主模型/proposal 概率比。候选不是由小 proposal 生成，因此外层候选块不乘这项比值。

实验配置可对 completion 的 log 概率比作显式对称截断。截断关闭时是理论上的普通重要性权重；截断
打开时会引入偏差，但能抑制小 $K$ 和长 completion 下的权重爆炸。原始比值、实际比值与截断数量
都会写入诊断，二者不能在结论中混为一谈。

公开 benchmark 还提供 `verifier_mh` 和两种基于 verifier 的条件 IS。它们使用与本地 GRPO 目标相同
的精确数值奖励和 reward temperature，因此以
`base * exp(reward / temperature)` 作为共同参考目标；有限候选、有限 rollout 与小 proposal 截断
分别作为近似误差报告。原有 `mh` 与 `conditional_is` 标识仍分别表示幂分布和 self-consistency
实验；结果报告会把这两个问题分开。
