# Qwen3.8 十二方法评测：报告与完整分析材料

本目录保存 2026-11-06 完成的报告、全部已记录输出、判题、审阅标签、冻结输入和分析脚本。完整文件使用普通 Git 压缩归档保存，克隆仓库即可取得，无需 Git LFS 或连接实验服务器。

| 入口 | 内容 |
| --- | --- |
| [代码评测主报告](reports/coding/REPORT.md) | 正确率与性能表、跨题算法风格、原文证据；LiveCodeBench 100 题和 HumanEval+ 164 题，共 3168 份输出 |
| [十五组详细案例](reports/coding/CASE_COMPARISONS.md) | 保留逐题对照、42 段思维链摘录与判题问题说明 |
| [风格复核全文](reports/coding/STYLE_TRACES.md) / [原案例全文](reports/coding/SELECTED_TRACES.md) | 报告引用及固定风格样本的完整终选思维链与实际提交代码 |
| [代码评测统计](reports/coding/summary.json) | 十二方法的正确数、成本与配对统计 |
| [数学评测主报告](reports/math/REPORT.md) | 原九方法加三种 Metropolized + SC，100 题 × 12 方法，共 1200 份输出 |
| [数学完整统计](reports/math/all-methods-summary.json) | 合并对照、答案等价修正及原始判题口径 |
| [文件清单与 SHA-256](manifest.json) | 每个归档、归档内文件和直接展示文件的大小与哈希 |

“完整输出”指实验实际保存的全部记录：代码评测含终选思维链、最终回答、实际提交代码、内部选择追踪和计算量字段；数学评测保留原始测量记录中的全部字段。实验没有保存的落选候选全文不在这些材料中。

## 下载、校验与展开

需要 Python 3.11 或更新版本。在仓库根目录运行：

```bash
python experiments/qwen38-20261106/unpack.py --verify-only
python experiments/qwen38-20261106/unpack.py --output analysis-workspace
```

第二条命令先核验所有文件，再恢复到新的或空的目录；不会连接网络、执行生成代码或启动实验。`analysis-workspace/` 已被 Git 忽略，也可以把 `--output` 指向仓库外的空目录。

| 归档 | 展开后的内容 |
| --- | --- |
| [coding-workspace.tar.gz](full/coding-workspace.tar.gz) | 全部 3168 份生成记录和 3168 份判题记录、1739 份去重审阅材料、报告、语义复核数据、分析脚本及实际使用的控制器 |
| [coding-frozen-deployment.tgz](full/coding-frozen-deployment.tgz) | 原始冻结部署包，逐字节保留；含生成和判题源码、EvalPlus / LiveCodeBench 判题代码、264 题完整判题输入、配置和文件哈希 |
| [math-workspace.tar.gz](full/math-workspace.tar.gz) | 全部 1200 份数学测量、数据、原始源码快照、补测部署记录、标签、报告及分析脚本 |

代码评测的冻结源码提交为 `214a358c7ba2ff3037fdf7a8e188fb3877f81dcb`。归档约 56 MB；本目录的可直接阅读报告与归档内报告来自同一份最终文件。历史报告的浏览副本调整了指向仓库文档的相对链接，归档仍保留原字节。历史报告中的绝对路径与服务器状态是当时的溯源记录，分析时使用下面的相对路径。

## 后续分析的主要文件

展开后的路径均相对于 `analysis-workspace/`：

| 路径 | 用法 |
| --- | --- |
| `work/coding_bench_20261104/snapshots/{a,b}/generation/{a,b}.jsonl` | 完整生成流；筛选 `kind == "measurement"`，以 `(problem_id, mode)` 配对，共 3168 行测量 |
| `work/coding_bench_20261104/snapshots/{a,b}/evaluation/{a,b}-judgments.jsonl` | 完整判题流；筛选 `kind == "judgment"`，按 `problem_id`、`mode`、`submission_sha256` 与生成记录连接 |
| `outputs/qwen38-coding-20261104/review-cases.jsonl` | 1739 份去重后的题目、完整思维链和提交代码；不能代替逐方法原始记录 |
| `outputs/qwen38-coding-20261104/review-key.json` | 去重案例与原始方法记录的对应关系 |
| `outputs/qwen38-coding-20261104/review-labels.jsonl` | 全量内容复核标签与引文 |
| `outputs/qwen38-coding-20261104/style-review-labels.jsonl` | 固定 24 题配对风格复核；口径见 [STYLE_METHODS.md](reports/coding/STYLE_METHODS.md) |
| `work/coding_bench_20261104/frozen/public-manifest.json` | 固定题集、来源版本、采样规则及公开提示 |
| `work/coding_bench_20261104/frozen/judge-data.json` | 冻结的完整判题输入，供重新分析判题口径 |
| `work/met_sc_20261103/{a,b}/results/*/*.jsonl` | 数学原始测量流；筛选 `kind == "measurement"` 后共 1200 行 |

例如，读取全部代码生成记录：

```python
import json
from pathlib import Path

root = Path("analysis-workspace/work/coding_bench_20261104/snapshots")
records = [
    row
    for replica in ("a", "b")
    for line in (root / replica / "generation" / f"{replica}.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
    for row in [json.loads(line)]
    if row.get("kind") == "measurement"
]
assert len(records) == 3168
assert len({(row["problem_id"], row["mode"]) for row in records}) == 3168
```

## 重建报告

分析脚本的可阅读副本位于 [analysis](analysis)，展开后会恢复到 `analysis-workspace/work/`，保留原来的相对目录结构。只重建统计和报告不需要模型、NPU 或实验服务器。

```bash
python -m pip install "numpy>=1.26,<3"
python -X utf8 analysis-workspace/work/finalize_coding_report.py
python -X utf8 analysis-workspace/work/summarize_met_sc.py
```

`finalize_coding_report.py` 检查全部原始记录和引文后重建代码主报告，保留正确率、性能与长度两张表。`coding_study.py prepare` 可重新计算代码统计和配对区间；风格客观统计可通过 `analyze_coding_style.py` 重建。人工语义标签是保存的审阅结果，不会由关键词自动重判。旧的 `expand_coding_report.py` 用于追溯十五组案例版本；日常重建使用 `finalize_coding_report.py`。

原始生成和判题记录保留原字节。代码报告仍采用冻结判题成绩，其中浮点容差等已知限制见主报告；这里没有把案例反例直接换算成全题重新判分。归档不包含模型权重及完整运行时镜像，不能单凭这些文件保证重新生成相同采样结果。

## 上游材料

冻结包保留 EvalPlus 和 LiveCodeBench 的上游许可证，另提供 [EvalPlus 许可证](licenses/evalplus-LICENSE) 和 [LiveCodeBench 许可证](licenses/LiveCodeBench-LICENSE) 的直接副本。实际版本见 `manifest.json` 的 `vendor_commits`；题集版本与来源哈希见 [public-manifest.json](reports/coding/public-manifest.json)。这些声明与许可证保留原有适用范围。
