# Online Uno

在 RTX 3090 24 GB 上研究端到端吞吐优先的在线 speculative decoding。
当前主线是 **Recycling Uno**：复用上一轮 verifier 的尾部预测作为下一轮确定性候选，
以一遍 base-model verification 扩展输出；需要新候选时再调用两遍式 Uno。
用已经完成的 cycle 的真实耗时和提交 token 数在线决定是否复用、何时 refill。

新设计不要求首请求重复，也不需要先训练 residual head。任何候选都重新验证。
目标是提升包含在线控制和候选更新成本后的 TPS；数学正确性与系统加速分别检验。

## 当前状态

| 项目 | 证据与状态 |
| --- | --- |
| 硬件 | RTX 3090 24 GB，i7-12700K，32 GB RAM |
| 静态基线 | Uno 0.9B，Windows HF KV-cache，B=8 对 AR median decode speedup 1.352× |
| 官方 Linux runtime | WSL 安装中断后重新准备官方 MSI；尚未运行官方 Nano-vLLM 基线 |
| Recycling Uno | 设计与证明先行，随后实现、pilot、独立 held-out 评估 |
| 旧在线 residual/retrieval 试验 | 已归档；其中 residual 无可靠 TPS 收益，精确重复 retrieval 仅是工程上界 |

## 当前文档

- [新算法与数学证明](docs/RECYCLING_UNO_DESIGN_AND_PROOFS.md)
- [新实验协议](docs/RECYCLING_UNO_EXPERIMENT_PROTOCOL.md)
- [当前路线图](docs/ROADMAP.md)
- [硬件和复现边界](docs/HARDWARE_REPRODUCIBILITY_AUDIT.md)
- [Uno 静态基线](docs/STAGE2_UNO1B_RESULTS.md)
- [WSL/官方运行时协议](docs/STAGE9_WSL2_RUNTIME_PROTOCOL.md)
- [旧研究归档索引](archive/2026-09-05-v1/README.md)

## 原则

1. verifier 的 base 权重和离线 Uno adapter 固定；每一轮用实际的旧 proposal law 验证。
2. 在线状态只由过去反馈决定；未验证的尾部预测只是候选，不能直接提交。
3. TPS 分母包括 controller、同步、候选维护和在线学习时间；请求结束维护另报 inclusive E2E。
4. pilot 用于设计；confirmatory test 冻结配置与新 prompts/seeds。报告全部 workload 及负结果。
5. 每一阶段保存代码、结果和数学/实验文档，并直接 commit + push。

## 本地验证

从父仓库 `inference_scaling` 执行：

```powershell
.\.venv\Scripts\python -m pytest .\online-speculation\tests
.\.venv\Scripts\ruff check .\online-speculation\src .\online-speculation\tests
```

模型和安装包保存在被忽略的目录；版本锁、摘要结果和证明进入 Git。
