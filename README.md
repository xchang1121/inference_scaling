# Online Uno

在 RTX 3090 24 GB 上研究端到端吞吐优先的在线 speculative decoding。
当前原生运行主线是 **R7 Native Online Uno**：保留官方 FA2/CUDA-graph draft/verify，
在请求内用真实接受量与耗时在线学习投机块长。更新成本计入完整 TPS。
之前的 Budgeted Online Tree Uno / ancestor-only HF 树实现保留作独立研究与证据，
不把不同 backend、dtype、长度的绝对 TPS 混作算法收益。

新设计不要求首请求重复，也不需要先训练 residual head。任何候选都重新验证。
目标是提升包含在线控制和候选更新成本后的 TPS；数学正确性与系统加速分别检验。
没有在线更新 base/LoRA 网络权重。Uno 官方本身已有静态树，树验证不是本项目首创。

## 当前状态

| 项目 | 证据与状态 |
| --- | --- |
| 硬件 | RTX 3090 24 GB，i7-12700K，32 GB RAM |
| 静态原生基线 | Uno 0.9B / BF16 / FA2，32/32 runs：128-token AR 184.37 TPS，官方 B=8 225.07 TPS |
| WSL | Windows 已重启；WSL2 + Ubuntu 22.04.5 正常运行，Linux nvidia-smi 已识别 RTX 3090 |
| 官方 Linux runtime | 已完成；Python 3.10、torch 2.11/cu128、Triton 3.6、FA2 2.8.3，14 项检查含 GPU forward/backward 全通过 |
| R7 原生在线 | 60/60 runs：256-token R7 211.31 TPS，原 B=8 208.98 TPS，AR 185.76 TPS；对 B8 +1.11%，区间跨 1，尚无稳定额外加速结论 |
| 原生行为控制 | 同块长 shadow8 与官方 B8：8/8 对 tokens/text/stats 完全一致，TPS 约 -0.77%；动态跨 B 仍有输出分叉 |
| 在线预算树 pilot | FP32：48.74 TPS vs linear B=8 的 41.27 TPS（+18.09%）；vs fixed tree 47.86 TPS 仅 +1.84% |
| 独立评估 | 360/360 完成；300 个 speculative 输出全部逐 token 等于 AR；频率门未通过，整组仅作描述性工程测量 |
| held-out TPS | linear B=8：42.41；fixed tree N=16：48.35；fixed tree N=32：50.60；online budget：49.58 |
| 在线收益边界 | 相对 linear +16.89%，相对 fixed N=16 +2.53%，但相对更强的 fixed N=32 -2.02%；尚无正式在线额外收益结论 |
| R6A 反馈校正 pilot | 72/72 完成、输出等于 AR；46.85 TPS < 同组 cost-only 48.45 < fixed N=32 49.63，保持默认关闭 |
| Recycling / warm-start | 已实现并测试，但没有可靠 TPS 收益，退出默认主线；负结果保留 |
| 旧在线 residual/retrieval 试验 | 已归档；其中 residual 无可靠 TPS 收益，精确重复 retrieval 仅是工程上界 |

## 当前文档

- [用户更新后的验收口径](docs/CURRENT_ACCEPTANCE_CRITERIA.md)
- [已重启后的 WSL/Ubuntu/CUDA 进度](docs/STAGE12_LINUX_RUNTIME_PROGRESS.md)
- [未修改官方 FA2 基线协议](docs/STAGE12_OFFICIAL_BASELINE_PROTOCOL.md)
- [R7 原生在线块长：设计、分布保持与成本证明](docs/R7_NATIVE_ONLINE_DESIGN_AND_PROOFS.md)
- [官方原生基线与跨 B 行为差异](docs/STAGE12_OFFICIAL_BASELINE_RESULTS.md)
- [R7 原生在线 60-run 完整结果](docs/STAGE12_NATIVE_ONLINE_RESULTS.md)
- [固定 B8 在线框架行为控制](docs/STAGE12_SHADOW_CONTROL_RESULTS.md)
- [当前树算法与数学证明](docs/BUDGETED_TREE_UNO_DESIGN_AND_PROOFS.md)
- [完整 held-out 结果、审计与收益边界](docs/STAGE11_TREE_HELDOUT_RESULTS.md)
- [稳定 QoS pilot 与在线收益边界](docs/STAGE11_HIGHQOS_TREE_PILOT_RESULTS.md)
- [独立 held-out 协议](docs/TREE_HELDOUT_PROTOCOL_20260905.md)
- [WSL 安装完成与重启恢复点](docs/STAGE11_WSL_PROGRESS.md)
- [FA2/3090 树路径候选迁移与 attention 合并证明](docs/FA2_TREE_PORT_PLAN_AND_PROOF.md)
- [嵌套树反事实反馈基础证明](docs/COUNTERFACTUAL_BUDGET_LEARNING_PROOF.md)
- [R6A 反馈校正控制器与冻结 pilot 协议](docs/R6A_FEEDBACK_CORRECTED_TREE_PROTOCOL.md)
- [R6A 完整结果：正确性通过、性能改进失败](docs/R6A_FEEDBACK_TREE_RESULTS.md)
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

Windows HF 实验中观察到的最快配置是 `tree:8:32`；该结果不能直接与 Linux/BF16 实验比较。
R7 已达到接近原 Uno 吞吐的工程目标，尚未证明稳定额外加速、任务质量等价或全局最大 TPS。
Windows 重启、Ubuntu 和 Linux kernel smoke 已完成，无需沿用旧重启请求。
临时下载/转发源脚本已清理、转发已关闭；分片删除被策略拦截而保留缓存。
原生模型基线与在线 pilot 以 Stage 12 的实际记录为准。

最终回归：218 tests passed、Ruff 通过、Linux pip check 通过。原始 JSON 与数学/结果文档均已提交。

## 核心实现

- `scripts/native_online_policy.py`：R7 原生官方引擎外围的请求内 EMA 块长学习，无在线 LoRA SGD。
- `scripts/wsl_official_baseline.py`：官方原版基线，以及 `--online` 启用的同运行时 R7 配对实验。
- `src/online_speculation/tree_uno.py`：嵌套树、target-draw 遍历、rank 校准、在线成本预算。
- `src/online_speculation/hf_tree_uno.py`：真实模型 ancestor mask、position、KV 路径整理。
- `src/online_speculation/feedback_budget.py`：独立启用的 R6A propensity-corrected 残差控制器；旧 R3E 配置不变。
- `src/online_speculation/hf_recycling_benchmark.py`：配对顺序、完整计时、聚类统计及静态树对照。
- `patches/0001-experimental-fa2-tree.patch`：默认关闭的上游 FA2 树实验补丁，仅通过 apply-check，尚无 GPU 通过声明。
