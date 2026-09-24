# 归档实现

本目录保存被主线取代或筛选后未进入主线、但报告结果依赖的实现。主线包（`shared`、`arllm`、`dllm`）不导入这里的模块；
实验组装代码按方法名或对应基准选择它们，只用于复现已报告的结果。

- 主线只保留收益最高的版本。被取代、但报告结果依赖的实现移入本目录，并尽量复用主线组件；没有收益的实现直接删除，
  需要时从 git 历史检出。
- 归档方法不属于任何默认组件，只能按方法名显式运行；`tests/archive/` 覆盖它们与主线组件的配合。

## 当前内容

| 实现 | 归档原因 | 运行方式 |
| --- | --- | --- |
| [分块条件 IS](arllm/block_conditional_is.py) | 主线改为保留完整序列的[条件 IS](../arllm/algorithms/conditional_is.py)；GSM8K 报告依赖分块版本 | `--method block_conditional_is`、小模型补全变体（见下） |
| [迭代条件 IS](arllm/iterated_is.py)、[i-SIR 核](shared/iterated_sir.py) | 额外轮次的质量—成本收益不足 | `--method iterated_conditional_is` |
| [两阶段 IS](arllm/progressive_is.py) | 执行成本报告中墙钟与 FLOPs 均高于固定 IS | `benchmark_rollout_infra.py` 的算法组 |
| [SMC 多树搜索](arllm/smc_forest.py) | 非默认搜索；报告比较其后缀复用 | `benchmark_rollout_infra.py` 的算法组 |
| [动态候选 IS](arllm/dynamic_is.py) | 报告中设计阶段开销较大 | 组件 `dynamic_is`（`gsm8k_dynamic_is_benchmark.py`） |
| [流式 IS](arllm/streaming_is.py) | 报告中额外调度未形成收益 | `benchmark_is_mh_reuse.py` |

分块版本独有的 RQMC rollout 与精确提前停止，以及 0.5B 草稿模型推测解码，在筛选中没有收益且不再有运行入口，
已删除；最后的实现分别见提交 `642f617` 与 `4fcb376`。

## 复现

提交 `1c34687` 及更早版本的报告中，`conditional_is` 与 `verifier_conditional_is` 指分块条件 IS；复现时改用
`block_conditional_is` 与 `verifier_block_conditional_is`，它们沿用原方法名对应的随机种子。小模型补全变体
（`conditional_is_small_proposal` 等）保持原名。例如：

```powershell
python experiments/arllm/gsm8k_reproduction.py --config configs/gsm8k_standard.toml --method block_conditional_is
```

主线条件 IS 复用保留补全的奖励，只接受逐序列固定的奖励，GSM8K 配置因此把 `[conditional_is].reward` 设为
`frozen_consensus`；归档方法读取同表的 `block_reward`（未设置时回落到 `reward`），配置保留报告所用的
`self_consistency`。GSM8K 奖励消融包含组内归一化奖励，改由 `block_conditional_is` 执行；rollout 基础设施基准
也调用归档实现，与其报告一致。
