# 非默认研究实现

本目录保存已经实现并测试、但未进入生产默认执行链的算法。保留这些实现是为了复现实验结论和继续研究；调用方必须使用 `inference_scaling.experimental` 下的完整模块路径显式导入。

当前包括：

- 动态候选、逐轮 i-SIR、渐进式 IS、SMC rollout forest 和流式 IS；
- 有界提前停止、随机化 QMC rollout 设计和逐轮 SIR 公共算子；
- Qwen2.5-0.5B 草稿模型的精确 speculative decoding 实验实现。

生产入口只调度已经在 Qwen2.5-1.5B 消融中产生正收益的方法。完整实验结论与采用条件见 [`docs/reports/RTX3090_ROLLOUT_INFRA.md`](../../../docs/reports/RTX3090_ROLLOUT_INFRA.md) 和 [`results/optimization_attempts.json`](../../../results/optimization_attempts.json)。

