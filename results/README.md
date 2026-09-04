# Results

本目录只提交小型、机器可读的实验清单与汇总。checkpoint、数据集、原始生成记录和 profiler trace
必须留在被忽略的 `models/`、`cache/` 或外部存储中。

- `preflight_rtx3090_windows.json`：本机硬件、Python/CUDA、关键包、Git 和官方 Uno runtime
  就绪状态。
- `stage1_lossless_validation.json`：static 与 post-round adaptive proposal 的完整序列 Monte Carlo
  分布检验，以及误用更新后 proposal 分母的解析负面对照。
- `stage2_uno1b_rtx3090_hf.json`：锁定 Uno-1B checkpoint 的 Windows Hugging Face KV-cache
  回退实验；只把 TPF/接受率作为算法级复现，wall-clock 结论始终附带 backend 限定。
- `stage2_uno1b_rtx3090_hf_analysis.json`：对上述 10 个配对重复做 median percentile
  bootstrap，并把 algorithmic、fallback wall-clock、official runtime 三种结论分开判定。
- `stage2_official_runtime_probe.json`：锁定上游的顶层包可导入，但正式 `model_runner` 在当前
  Windows 环境因缺少 Triton 而阻塞的精确失败点。
- `stage3_online_markov.json`：20 seeds × 8 strategies 的 exact $\Psi$-Spec 非平稳 Markov 仿真；
  包含预注册主检验、TV regret/TPF/合成 cost proxy 的配对 bootstrap、update-cost 敏感性、segment
  分解，以及一个代表 seed 的 trace/controller events。`gpu_timing=false` 是强制结论边界。
