# 完整步骤边界的 FLARE＋证据缺口检查

已完成 16 题 × 11 条件＝176 个条件结果；8 道新题与 8 道回归题分别报告。主模型 thinking 开启，无 effort 提示。统一 1024 生成工作 token（全部草稿、丢弃与检查输出均计入）、512 回答 token、最多 256 个证据输入 token；原生图成本与耗时另列。

新题 F1（百分制）：静态 RAG **14.06**，现有 SC **11.09**，FLARE＋缺口检查直接追加 **11.38**，同检查点资料区重建 **11.43**。FLARE 减静态 RAG 为 −2.68 个百分点，按题配对 bootstrap 95% 区间 [−5.92, +0.87]，未显示稳定优势。每组仅 8 题、单种子，F1 不是语义正确率。

格式效应仍在：两个实际接受插入的检查点，相关及同长度无关正文在步骤末尾追加均立即闭合 thinking；同检查点移入用户资料区则均未立即闭合。回归组一次追加还在闭合后直接 EOS，得到空回答。逐字引用校验也有语义局限：一个通过的片段只复述症状，另一个才给出所需原因。

- [结果解释、预算和成本](../../experiments/online-injection-flare-gap-20261109/reports/OBSERVATIONS.md)
- [逐条件统计与区间](../../experiments/online-injection-flare-gap-20261109/reports/REPORT.md)
- [冻结方案](FLARE_GAP_PROTOCOL.md)
- [离线校验与复现](../../experiments/online-injection-flare-gap-20261109/README.md)
- [预选新题全部轨迹](../../experiments/online-injection-flare-gap-20261109/reports/traces/manifest.json)
- [所有实际接受插入的轨迹](../../experiments/online-injection-flare-gap-20261109/reports/accepted-traces/manifest.json)

正式生成源码冻结在 `28bc5e657e7d30d026bdf96d19d00b745be25ee9`；之后仅完善分析、导出、停止分类与产物校验，未改变正式推理控制器。176 条条件审计、112 组配对审计、25,629 项请求／执行单元检查通过；180 个执行单元的文本解码已核对。生成退出码 0，专用 NPU3 容器已停止，无 OOM。此前 312／336 条冻结归档保持原样。
