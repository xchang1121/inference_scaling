# FLARE + evidence-gap retrieval at complete reasoning steps

16 questions (8 newly held out, 8 regression), 11 conditions each, **176 formal condition outcomes**. Qwen3.8-27B on 198 physical NPU3 only, TP1. The prior 336-result archive remains unchanged.

- [Audited results and limits](reports/REPORT.md), [interpretation](reports/OBSERVATIONS.md)
- [Exact protocol](../../docs/reports/FLARE_GAP_PROTOCOL.md)
- [Per-condition grades/costs](reports/graded.jsonl), [summary](reports/summary.json)
- [All structural audits](reports/audit.jsonl), [paired controls](reports/pair-audit.jsonl), [gate decisions](reports/gate-events.jsonl)
- [Figure](reports/figures/comparison.png), [preselected full saved traces](reports/traces/manifest.json), [every accepted intervention](reports/accepted-traces/manifest.json)

Every condition has a 1024-token generated work cap, including all drafts, discarded previews, checker/planner output and sampled end/EOS markers. Final answer cap 512; evidence cap 256 including wrappers. Evidence input tokens consume context/prefill, rather than generated work. Equal token caps are not equal native FLOPs or equal actual use before natural termination. Main thinking is enabled; independent checkers use no-thinking greedy JSON. Rendered main prompts contain no effort instruction.

Primary FLARE inserts a short quoted packet directly after a complete assistant reasoning step. Its paired rebuild variant uses the exact same accepted checkpoint and packet in the original user reference region. Static RAG, legacy SC, and step-boundary SC are included. The legacy SC criterion is retained; all methods now share the short window retriever, so it is not a bitwise replay of the previous long-document SC arm. Unrelated bodies are substituted after the real-evidence support gate, with labels/wrappers and exact payload length unchanged.

Exact pre-intervention units are shared, not independently rerun. Each standalone condition is charged the full common stem plus its own cold branch. Actual study totals sum unique units. If no gate accepts, all corresponding content/format conditions share one completed result. All EOS, empty answers, rejected/invalid checks and capped answers remain in the denominator. One seed per question; F1 is normalized generative token overlap, not official extractive TechQA accuracy or semantic correctness.

## Verify and unpack

Python 3.11+:

```bash
python experiments/online-injection-flare-gap-20261109/unpack.py --verify-only
python experiments/online-injection-flare-gap-20261109/unpack.py --output analysis-workspace/flare-gap
```

The target must be new or empty. Extraction verifies every direct file and archive member, uses no network and runs no model. Generation source and final analysis scripts are separately hashed: analysis received output determinism/reporting fixes after generation was frozen.

## Rebuild results offline

No model weights, accelerator or full tokenizer are required. A verified ByteLevel decoder subset reconstructs every saved execution-unit thinking/answer and the two preselected heldout trace examples. NumPy and Matplotlib are needed only for the figure.

```bash
PYTHONPATH=analysis-workspace/flare-gap/generation-source/src python experiments/online-injection-flare-gap-20261109/analysis/summarize_flare_gap.py --generation analysis-workspace/flare-gap/snapshot/formal.jsonl --prepared analysis-workspace/flare-gap/snapshot/prepared.json --gold analysis-workspace/flare-gap/data/gold.jsonl --output analysis-workspace/flare-gap/rebuilt
PYTHONPATH=analysis-workspace/flare-gap/generation-source/src python experiments/online-injection-flare-gap-20261109/analysis/plot_flare_gap.py --summary analysis-workspace/flare-gap/rebuilt/summary.json --output analysis-workspace/flare-gap/rebuilt/figures
PYTHONPATH=analysis-workspace/flare-gap/generation-source/src python experiments/online-injection-flare-gap-20261109/analysis/export_flare_gap_traces.py --snapshot analysis-workspace/flare-gap/snapshot --output analysis-workspace/flare-gap/rebuilt/traces
```

An additional accepted-traces directory covers every accepted FLARE case with all 11 conditions, selected by gate decision rather than answer quality.

The archive includes formal raw requests, exact prefixes/IDs/logprobs/SC, discarded drafts, evidence/gates, both TRAIN smoke versions, a separately labeled synthetic positive checker contract, failed initial CPU preparation, runtime/container/source/input receipts and local grading gold. Gold was not deployed during generation. Corpus parent/gold document subsets, data card and license are included. Re-running BM25 parent selection needs the full 28,481-document corpus at the pinned NVIDIA revision and SHA256; subset replay is not full-index retrieval reproduction. Model weights/container images and full corpus are not duplicated.

Unit native graph figures estimate dense padded FLOPs; attention/host work and initialization profiling are not fully instrumented. Stage wall-time receipts include initialization/teardown. Smoke, warmup and synthetic contract costs are reported separately from formal condition costs. The synthetic fixture's provided reasoning is input, never presented as sampled prehistory or a benchmark answer.

[离线重建验证收据](reports/offline-verification.json)：16 份派生统计、PNG/SVG 与完整轨迹文件逐字节一致；模型与网络均未使用。
