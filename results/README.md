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
- `stage4b_online_uno1b_rtx3090_hf.json`：真实 K2-Horizon-0.9B-Uno 上 static、online stride-10/20
  的 45 条固定长度 GPU 运行；含逐 update loss/rollback/reset、参数隔离、时间分解、显存和原始生成。
- `stage4b_online_uno1b_rtx3090_hf_analysis.json`：对 15 个 paired workload 的预注册判定、bootstrap、
  exact sign-test 和逐 prompt 诊断；主策略学习门失败且 HF TPS 显著变慢，不能表述为 online speedup。
- `stage5b_deferred_online_uno1b_rtx3090_hf.json`：future-validated deferred controller 的 30 条正式真机
  运行；包含候选 shadow evidence、promote/keep/reset、zero-head skip、逐组件计时与完整 paired metrics。
- `stage5b_deferred_online_uno1b_rtx3090_hf_analysis.json`：严格核验冻结设计、安全不变量，以配对均值为
  预注册主统计、配对中位数为稳健性统计；TPF 与 TPS 两个主门均未通过，并显示中文 workload 的明确退化。
- `stage6c_stream_uno1b_rtx3090_hf.json`：4-train/5-validation/10-test stationary request stream 的
  58 个真机生成记录；包含所有 snapshot score、validation-only selection、frozen head hash 和训练摊销。
- `stage6c_stream_uno1b_rtx3090_hf_analysis.json`：重算 stream continuity、zero/frozen/selection 不变量和
  paired mean/median bootstrap；snapshot 4 的 validation gain 未泛化，TPF/TPS test 主门均失败。
- `stage7_engineering_pilots_summary.json`：固定 `w=0.25` 与 verifier-EMA adaptive mixture 的小型工程
  pilot 聚合；固定权重未推广，adaptive validation 回退 zero。该文件只生成下一假设，不作为正式确认性结果。
- `stage8_greedy_stream_uno1b_rtx3090_hf.json`：预注册的 4-train/5-validation/20-test greedy request
  stream；含 78 个配对/快照生成、完整 token IDs、parameter isolation、head hash 和训练成本。
- `stage8_greedy_stream_uno1b_rtx3090_hf_analysis.json`：重算 Stage 8 全部安全门、validation-only 选择、
  50,000 次 paired bootstrap 和 sign tests；TPF 学习门通过，HF wall-clock 系统门失败。
- `stage10b_replay_uno1b_rtx3090_hf_pilot.json`：一次 verifier-confirmed cache-build request 后的
  exact-repeat 工程上界；记录 5 个 alternating static/replay pairs、greedy AR token 等价、真实 KV/forward
  计数、cache build cost、TPF/TPS bootstrap 与摊销。它是 Windows HF pilot，不是预注册或官方 runtime 结果。
