# Online Uno

在 RTX 3090 24 GB 上研究端到端吞吐优先的在线 speculative decoding。
当前主线是 **Budgeted Online Tree Uno**：一次 Uno draft 产生多个位置的候选，
用前缀闭合树覆盖多个可能 continuation，再通过 ancestor-only target attention 验证。
在线更新的是实际成本统计与验证节点预算；另有独立的在线 rank 校准消融。

新设计不要求首请求重复，也不需要先训练 residual head。任何候选都重新验证。
目标是提升包含在线控制和候选更新成本后的 TPS；数学正确性与系统加速分别检验。
没有在线更新 base/LoRA 网络权重。Uno 官方本身已有静态树，树验证不是本项目首创。

## 当前状态

| 项目 | 证据与状态 |
| --- | --- |
| 硬件 | RTX 3090 24 GB，i7-12700K，32 GB RAM |
| 静态基线 | Uno 0.9B，Windows HF KV-cache；早期静态复现与本轮完整生成计时分别保留 |
| WSL | Microsoft 签名/hash 验证通过，WSL 2.7.13.0 与虚拟机平台已安装；**等待 Windows 重启** |
| 官方 Linux runtime | Ubuntu/CUDA/PyTorch/Triton/FA2 和未修改官方 Nano-vLLM 基线仍待重启后验证 |
| 在线预算树 pilot | FP32：48.74 TPS vs linear B=8 的 41.27 TPS（+18.09%）；vs fixed tree 47.86 TPS 仅 +1.84% |
| 独立评估 | 360/360 完成；300 个 speculative 输出全部逐 token 等于 AR；频率门未通过，整组仅作描述性工程测量 |
| held-out TPS | linear B=8：42.41；fixed tree N=16：48.35；fixed tree N=32：50.60；online budget：49.58 |
| 在线收益边界 | 相对 linear +16.89%，相对 fixed N=16 +2.53%，但相对更强的 fixed N=32 -2.02%；尚无正式在线额外收益结论 |
| Recycling / warm-start | 已实现并测试，但没有可靠 TPS 收益，退出默认主线；负结果保留 |
| 旧在线 residual/retrieval 试验 | 已归档；其中 residual 无可靠 TPS 收益，精确重复 retrieval 仅是工程上界 |

## 当前文档

- [当前树算法与数学证明](docs/BUDGETED_TREE_UNO_DESIGN_AND_PROOFS.md)
- [完整 held-out 结果、审计与收益边界](docs/STAGE11_TREE_HELDOUT_RESULTS.md)
- [稳定 QoS pilot 与在线收益边界](docs/STAGE11_HIGHQOS_TREE_PILOT_RESULTS.md)
- [独立 held-out 协议](docs/TREE_HELDOUT_PROTOCOL_20260905.md)
- [WSL 安装完成与重启恢复点](docs/STAGE11_WSL_PROGRESS.md)
- [FA2/3090 树路径候选迁移与 attention 合并证明](docs/FA2_TREE_PORT_PLAN_AND_PROOF.md)
- [下一版嵌套树反事实反馈证明（尚未进入 GPU 控制）](docs/COUNTERFACTUAL_BUDGET_LEARNING_PROOF.md)
- [此前 recycling / warm-start 负结果](docs/STAGE11_RECYCLING_AND_WARMSTART_RESULTS.md)
- [当前路线图](docs/ROADMAP.md)
- [硬件和复现边界](docs/HARDWARE_REPRODUCIBILITY_AUDIT.md)
- [Uno 静态基线](docs/STAGE2_UNO1B_RESULTS.md)
- [WSL/官方运行时协议](docs/STAGE9_WSL2_RUNTIME_PROTOCOL.md)
- [旧研究归档索引](archive/2026-09-05-v1/README.md)

## 原则

1. verifier 的 base 权重和离线 Uno adapter 固定；每一轮用实际的旧 proposal law 验证。
2. 在线状态只由过去反馈决定；未验证的尾部预测只是候选，不能直接提交。
3. TPS 分母包括初始化、controller、同步、候选维护、在线学习和请求结束操作的完整生成调用。
4. pilot 用于设计；confirmatory test 冻结配置与新 prompts/seeds。报告全部 workload 及负结果。
5. 每一阶段保存代码、结果和数学/实验文档，并直接 commit + push。

## 本地验证

从父仓库 `inference_scaling` 执行：

```powershell
.\.venv\Scripts\python -m pytest .\online-speculation\tests
.\.venv\Scripts\ruff check .\online-speculation\src .\online-speculation\tests
```

模型和安装包保存在被忽略的目录；版本锁、摘要结果和证明进入 Git。

当前观察到的最快配置是 `tree:8:32`，不是在线 controller；该选择不构成全局最大 TPS 保证。
WSL 支持栈的下一步需要用户保存工作并重启 Windows，再运行
`scripts/resume_wsl_after_reboot.ps1`。Ubuntu 与 Linux GPU kernels 尚未在本机验证。

## 核心实现

- `src/online_speculation/tree_uno.py`：嵌套树、target-draw 遍历、rank 校准、在线成本预算。
- `src/online_speculation/hf_tree_uno.py`：真实模型 ancestor mask、position、KV 路径整理。
- `src/online_speculation/hf_recycling_benchmark.py`：配对顺序、完整计时、聚类统计及静态树对照。
- `patches/0001-experimental-fa2-tree.patch`：默认关闭的上游 FA2 树实验补丁，仅通过 apply-check，尚无 GPU 通过声明。
