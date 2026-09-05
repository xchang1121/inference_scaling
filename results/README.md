# 当前结果

这里保留静态 Uno 复现、环境审计，以及当前预算树主线与其前置失败的原始记录。
旧 Stage 3–8、10 的实验和分析已移至
[`archive/2026-09-05-v1/results`](../archive/2026-09-05-v1/results)。

每个新结果须标注 backend、checkpoint/hash、工作负载、配置、pilot/confirmatory 范围，
以及控制/更新开销是否计入。TPS 的分母不能只统计 GPU forward。

- `stage11_tree_highqos_fp32_pilot.json`：完整、用于选择配置的 pilot，不是独立测试。
- `stage11_tree_heldout_fp32.json`：冻结协议的独立测试，360/360 完成，原始记录不删改。
- `stage11_tree_heldout_audit.json`：完整矩阵、原始 SHA-256、AR token 一致性与 GPU 快照审计。
  21 次 AR post-run memory clock 降低，按照预注册规则整组降级为描述性工程测量，
  `confirmatory_clock_gate_passed=false`；不能把 completed=true 等同于正式 TPS 成功。
- `stage11_tree_rank_fp32_pilot.json`：频率切换并触发概率检查的失败，completed=false，不能用于 TPS 声明。
- `stage11_tree_feedback_fp32_pilot.json` / `stage11_tree_feedback_audit.json`：R6A 完整 72-run pilot；
  正确性通过，但反馈校正低于同组静态树和旧控制器，保留负结果，不是新的 held-out。
- `stage11_wsl_install.json`：WSL 本体安装成功，但 restart_required=true 不代表 Linux runtime 已验证。

完整 held-out 在线预算树为 49.58 TPS，线性 B=8 为 42.41 TPS，固定 N=32 树为 50.60 TPS。
在线版没有超过本组最强静态对照；详见[结果与限制](../docs/STAGE11_TREE_HELDOUT_RESULTS.md)。
