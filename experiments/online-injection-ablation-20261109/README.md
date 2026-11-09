# 无 effort 提示的思考预算与注入格式对照

本轮使用原 8 道 TechQA、3 档 thinking 预算、14 个方法，共 336 份完整轨迹。所有正式模型计算统一使用 198 物理 3 号单卡，TP=1。单轨迹采样，不包含 IS/MH 或候选选择。

- [完整结果、机制对照与图表](reports/REPORT.md)
- [逐条判分和停止行为](reports/grades.jsonl)、[完整统计](reports/summary.json)、[完整性审计](reports/audit-checks.json)
- [冻结方案](../../docs/reports/ONLINE_INJECTION_FOLLOWUP_PROTOCOL.md)
- [更智能的检索触发与插入：论文和接入取舍](../../docs/reports/ADAPTIVE_RETRIEVAL_OPTIONS.md)
- [首轮 pilot](../online-injection-20261109/README.md)

实际 prompt 没有任何 effort 指令。当前模型模板省略参数会默认插入 xhigh，因此通过其不生成 effort 文本的 medium 模板分支渲染，并逐题检查实际 token。thinking 预算 256 / 1024 / 4096；最终回答独立上限 512 token；外部资料不占 thinking 预算，但占上下文和 prefill。

无关对照只替换正文，保留来源标题与标签；外层去包装和仅包装使用等 token 数换行。五种资料变体逐题 token 数完全相同。relevant 标签指 BM25 检索资料，不是保证正确的 gold 证据。这些条件隔离当前正文和外层文字，不能涵盖所有格式效应。

## 校验与展开

在仓库根目录使用 Python 3.11 或更新版本：

```bash
python experiments/online-injection-ablation-20261109/unpack.py --verify-only
python experiments/online-injection-ablation-20261109/unpack.py --output analysis-workspace/injection-ablation
```

目标必须是新建或空目录。展开只恢复文件，不联网、不运行模型。manifest 保存直接文件、归档及每个成员的 SHA256。原始数据包括 snapshot/generation.jsonl、冻结 prepared.json、token IDs、生成概率、事件、成本、源码和运行时证据；初次纯 CPU 配置校验失败也保留，没有正式模型测量。

## 离线重建

安装 NumPy 和 Matplotlib 后，使用归档中的分析脚本与展开的冻结源码，不需要运行模型：

```bash
PYTHONPATH=analysis-workspace/injection-ablation/generation-source/src python experiments/online-injection-ablation-20261109/analysis/summarize_injection_ablation.py --snapshot analysis-workspace/injection-ablation/snapshot --gold analysis-workspace/injection-ablation/data/techqa/gold.jsonl --output analysis-workspace/injection-ablation/rebuilt
PYTHONPATH=analysis-workspace/injection-ablation/generation-source/src python experiments/online-injection-ablation-20261109/analysis/plot_injection_ablation.py --summary analysis-workspace/injection-ablation/rebuilt/summary.json --grades analysis-workspace/injection-ablation/rebuilt/grades.jsonl --output analysis-workspace/injection-ablation/rebuilt/figures
PYTHONPATH=analysis-workspace/injection-ablation/generation-source/src python experiments/online-injection-ablation-20261109/analysis/export_injection_ablation_traces.py --snapshot analysis-workspace/injection-ablation/snapshot --token-map analysis-workspace/injection-ablation/snapshot/trace-tokenizer-subset.json --output analysis-workspace/injection-ablation/rebuilt/traces
```

不重复包含模型权重、运行时镜像或完整 28,481 文档语料；prepared 输入包含实际检索资料，另保存全部检索及 gold 文档子集。重新检索需按原数据 manifest 的 NVIDIA 固定 revision 下载完整语料并核对 SHA256。TechQA 数据卡与 Apache-2.0 声明包含在归档中。

本轮是同八题的机制复核，三档预算不构成 24 道独立新题；F1 不能代替语义正确率。按完整原始轨迹判断自然结束、强制预算结束、无注入和空答案，失败及不符合预期的结果不丢弃。
