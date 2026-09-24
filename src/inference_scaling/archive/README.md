# 归档实现

本目录保存已被主线取代、但已有报告结果依赖的实现。主线包（`shared`、`arllm`、`dllm`、`experimental`）不导入这里的模块；
实验组装代码按方法名选择它们，只用于复现已报告的结果。

- 主线只保留收益最高的版本。被取代、但报告结果依赖的实现移入本目录，并尽量复用主线组件；没有收益的实现直接删除，
  需要时从 git 历史检出。
- 归档方法不属于任何默认组件，只能按方法名显式运行；`tests/archive/` 覆盖它们与主线组件的配合。

## 当前内容

[分块条件 IS](arllm/block_conditional_is.py)：每步按条件权重选择候选后只提交该块，用于估价的补全随即丢弃；支持小模型
补全的 $`p/q`$ 修正、比值截断与未校正消融。取代它的主线实现是[条件 IS](../arllm/algorithms/conditional_is.py)：保留一条
完整序列，在其块边界做条件 SIR。只属于分块版本的 RQMC rollout 与精确提前停止在筛选中没有收益，已删除，最后的实现见
提交 `642f617`。

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
