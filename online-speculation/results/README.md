# 当前结果

这里保留静态 Uno 复现、环境审计，以及当前预算树主线与其前置失败的原始记录。
旧 Stage 3–8、10 的实验和分析已移至
[`archive/2026-09-05-v1/results`](../archive/2026-09-05-v1/results)。

每个新结果须标注 backend、checkpoint/hash、工作负载、配置、pilot/confirmatory 范围，
以及控制/更新开销是否计入。TPS 的分母不能只统计 GPU forward。

- `stage11_tree_highqos_fp32_pilot.json`：完整、用于选择配置的 pilot，不是独立测试。
- `stage11_tree_heldout_fp32.json`：冻结协议的独立测试；只有 completed=true 才算完成。
- `stage11_tree_rank_fp32_pilot.json`：频率切换并触发概率检查的失败，completed=false，不能用于 TPS 声明。
- `stage11_wsl_install.json`：WSL 本体安装成功，但 restart_required=true 不代表 Linux runtime 已验证。
