# Results

本目录只提交小型、机器可读的实验清单与汇总。checkpoint、数据集、原始生成记录和 profiler trace
必须留在被忽略的 `models/`、`cache/` 或外部存储中。

- `preflight_rtx3090_windows.json`：本机硬件、Python/CUDA、关键包、Git 和官方 Uno runtime
  就绪状态。
- `stage1_lossless_validation.json`：static 与 post-round adaptive proposal 的完整序列 Monte Carlo
  分布检验，以及误用更新后 proposal 分母的解析负面对照。
