# 算法映射

本文给出 `HW_share/inference_scaling_article.tex`、框架标识和核心实现之间的对应关系。公式、证明与
数据生命周期见[推理算法实现](ALGORITHMS.md)。

## 论文与框架

| 框架标识 | 论文标签 | 目标或估计量 | 实现 |
| --- | --- | --- | --- |
| `mh` | `alg:main-power-mh` | 固定长度幂分布的后缀 MH | `algorithms/mh.py` |
| `conditional-is` | `alg:main-onpolicy-is` | base 候选与 base completion 的条件能量 | `algorithms/conditional_energy.py` |
| `base-replay` | `alg:main-replay-is` | base 候选上的 history + fresh-tail 估计量 | `algorithms/base_replay.py` |
| `dynamic-is` | `alg:main-dynamic-is` | 混合候选、外层 `p/q_c` 与 rollout replay | `algorithms/dynamic_is.py` |

## 执行优化

| 实现 | 服务对象 | 保持量 |
| --- | --- | --- |
| `AsyncRolloutBroker` | IS / replay rollout | 完整轨迹进入估计器；部分轨迹保留为续跑状态 |
| `FrozenStreamingISEstimator` | on/off-policy IS | 冻结的 request id 集合与最终统计量 |
| 随机 `RolloutTokenTree` | base rollout 解码 | 经验 proposal + residual correction 后的 target 分布 |
| `run_reward_mh_chain_prefetched` | 奖励目标 MH | 普通 Hastings 更新序列 |
| `run_reward_mh_chain_delayed` | 昂贵 verifier 的 MH | 两阶段接受率对应的目标分布 |
| `FrozenReplaySuffixProposal` | 历史后缀 MH | 冻结混合 proposal 的正反概率 |

## 数据生命周期

| 数据集合 | 可见信息 | 用途 | 状态变化 |
| --- | --- | --- | --- |
| `design` | completion、reward、概率、成本 | 方差与成本估计 | 持久保存 |
| `evaluation` | key、behavior id、数量 | 最终能量估计 | 原子领取一次 |
| `reserved` | 已领取记录的句柄 | 冻结后的 evaluation | 揭示后转入 `design` |
| current fresh | 本轮新生成记录 | 本轮能量估计 | 本轮结束后转入 `design` |
| reserve rollout | 选择完成后独立生成 | 后续 evaluation | 写入 `evaluation` |

动态预算由 `design_prepare` 生成独立设计数据，`statistics_provider` 读取设计统计量，
`rollout_budget_provider` 根据候选、终止标记、容量和成本冻结整数配额。相同 replay key 的候选共享
一次性 evaluation 库存。

## GSM8K 方法标识

| 标识 | 候选 | rollout | 奖励或目标 | 修正 |
| --- | --- | --- | --- | --- |
| `conditional_is` | 1.5B base | 1.5B base | cumulative self-consistency | on-policy |
| `conditional_is_small_proposal` | 1.5B base | 0.5B proposal | cumulative self-consistency | 1.5B/0.5B 后缀概率比 |
| `conditional_is_small_proposal_uncorrected` | 1.5B base | 0.5B proposal | cumulative self-consistency | proposal-energy 目标 |
| `verifier_mh` | 完整序列状态 | 1.5B 后缀 | 数值正确性 | Hastings 比 |
| `verifier_conditional_is` | 1.5B base | 1.5B base | 数值正确性 | on-policy |
| `verifier_conditional_is_small_proposal` | 1.5B base | 0.5B proposal | 数值正确性 | 1.5B/0.5B 后缀概率比 |

sampling policy 的温度、top-k、top-p 与模型版本共同定义概率。权重截断关闭时使用普通重要性权重；
开启时记录原始比值、应用比值、截断次数与 ESS。
