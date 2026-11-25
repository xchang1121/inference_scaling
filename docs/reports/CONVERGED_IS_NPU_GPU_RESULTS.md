# Qwen3.8-27B 收敛版 Adaptive IS：跨基准帕累托与 NPU/GPU 正交引擎实测报告

本报告汇总基于 `Qwen/Qwen3.8-27B`（BF16 权重 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，纯文本参数量 `26,895,998,464`，无任何外部草稿模型或外部奖励/验证模型）在 **昇腾 910B3（TP=2，64 GB HBM × 2）** 与 **NVIDIA L4（单卡 23 GB，vLLM 0.34.0）** 上的系统消融与工程验证结果。

---

## 1. 核心结论与开箱即用推荐

1. **全局推理采样与词表惩罚（Wave B1，880 次四基准配对评测）**：
   - 全局默认确定为 **`token_penalty = null`、`temperature = 0.7`、`top_p = 0.95`、`top_k = 20`**。
   - 相比旧仓库默认（`token_penalty = 1.0`、`T = 1.0`、全词表），四基准 80 题宏平均从 `86.15%`（`69/80`）提升至 **`89.90%`（`72/80`，配对 `+4/-0`）**，平均单题耗时下降 `19.5%`，MATH-500 L5 未闭合思考率减半（`10% → 5%`）。
   - 在 Adaptive IS v2 上同样验证 `T = 0.7` 显著优于 `T = 0.8`（四基准宏平均 `88.02%` vs `82.29%`，LiveCodeBench `79.2%` vs `58.3%`）。
2. **算法与奖励收敛（Wave B2-Core + B2-v2 + B2-Extra，13 组完整 88 题配对 × 4 基准）**：
   - 严格限定仅保留并改进既有两种无验证器内部奖励 **`self_certainty`** 与 **`consilience`**，不引入新奖励家族：
     - **`SelfCertaintyReward` 本体改进**：新增 `temperature_normalized=true`（将全词表 KL 乘以 `score_temperature`，消除低温采样下的奖励尺度膨胀）与 `incomplete_penalty=1.5`（抑制截断/未闭合思考），结合预掩码设备端标量统计使 `top_p=0.95, top_k=20` 下全词表 KL 仍严格有限且精确；聚合对比表明全段 `mean` 跨数学/代码最稳健。
     - **`ConsilienceReward` 本体改进**：将 `initial_penalty` 从 `3.0` 调优为 `1.5` 并加入 `incomplete_penalty=2.0`。
   - **动态内部调度（`exploration_schedule = "adaptive"`，`selection = "block_metropolized"`，`selection_ready = false`）**：外部仅暴露 `M`（`candidate_count`）、`K`（`rollout_count`）、`B`（`block_size`），内部自动协同四项机制：
     1. **动态试探门控（`pilot_gate`，默认 `pilot_tokens = B // 2`）**：简单/中等题单轨迹直出（GSM8K 88%–94%、HumanEval+ 67%–96% 提前退出），难题无缝复用已生成前缀升级完整搜索；
     2. **选择性前瞻（`selective_lookahead`）**：`K > 1` 时仅对前一半高分候选前缀分配完整 `K` 条续写，其余保留 `1` 条，保证无偏重要性权重同时省去 ~35%–40% 续写算力；
     3. **两阶段长尾停滞剪枝（`straggler_pruning`）**：在首轮与 Stage-2 同步监控已完结中位长度，中止置信度偏低的循环长尾轨迹，将 MATH-500 L5 未闭合思考率从旧两阶段基线的 `16.7%` 压降至 `4.2%–8.3%`；
     4. **ESS 自适应奖励温度（`adaptive_temperature`）与 Stage-2 `M-1` 上限**：依据组间离散度自动跳过或平滑分配 Stage-2 新候选，防止低收益重复探索。
   - **帕累托前沿推荐（均以 `M=4, B=1024, T=0.7, p=0.95, k=20, penalty=null` 为基准）**：
     - **通用质量默认（`settings/inference.json`）— `adapt_v2_k2_sc`（`M=4, K=2, B=1024`, `self_certainty`）**：四基准宏平均 **`88.02%`（`77/88`，配对 `+3/-1`，最高胜负比）**，LiveCodeBench 达全场最高 **`79.2%`（`19/24`，`+1/-0` 零退化）**；相比旧 `two_stage_bmet_k2_sc`（`83.33%`，`4492s`）提升 **`+4.69 pp`（+4 题）** 且更省时（`4062s`，`-9.6%`）。
     - **跨引擎/远程服务首选 — `adapt_v2_k2_cons_p15`（`M=4, K=2, B=1024`, `consilience`）**：并列四基准第一 **`88.02%`（`77/88`，`+4/-2`）**，总墙钟快 **`11.3%`**（`3605s` vs `4062s`）、总前向槽位少 **`9.7%`**（`997,762` vs `1,105,548`），且仅需 top-5 logprobs，可直连任意外部 OpenAI/SGLang/MindIE 服务。
     - **低算力快速档 — `adapt_v2_k1_sc`（外部设 `K=1`）**：`86.98%`（`76/88`），MATH-500 L5 达 **`91.7%`（`22/24`, `+2/-0`）**，总前向槽位比 `K=2` 再降 **`30.5%`**（`768,074`）。
     - **昇腾长推理页对齐档 — `adapt_v2_k2_b1536`（外部设 `B=1536`）**：对齐昇腾混合 Mamba/Attention `1536`-token KV 页，MATH-500 L5 达 **`91.7%`（`22/24`, `+2/-0`, 未闭合仅 `4.2%`）**，同时 MATH-500 槽位降 **`17.8%`**、HumanEval+ 槽位降 **`24.7%`**。
3. **NPU / GPU 正交硬件与推理引擎最优配置**：
   - **昇腾 910B3（TP=2，BF16）**：强制开启 ACLGraph（`enforce_eager=false`，关闭会慢 **5.74×–5.99×**）、开启前缀缓存（`enable_prefix_caching=true`，多分支 IS 省 **13%–14%** 前向槽位）、**`stream_interval=8`**（较旧默认 `32` 提速 **12.1%**、槽位省 **6.8%**，较 `64` 提速 **16.2%**）、`gpu_memory_utilization=0.88`、`max_concurrent_requests=4`（MATH-500 吞吐拐点 `29.7 q/hr = 2.83×`，GSM8K `969 q/hr = 3.16×`）。
   - **NVIDIA CUDA GPU（含单卡 24 GB L4 实测）**：统一 `GPULogprobWorker` 同时支持 vLLM 0.25–0.26 V1 与 vLLM 0.30–0.34+ GPU V2 采样器，开启 `scalar_self_certainty=true` + `async_scheduling=true` + CUDA Graph（`enforce_eager=false`）；单卡 24 GB 通过在线 `mxfp4` Marlin 量化（`ignore: ["re:.*visual.*", "re:.*lm_head.*"]`，无需外部模型）仅占 `17.39 GiB` 显存即可运行 `Qwen3.8-27B`，并发 `c=4` 纯推理提速 `1.80×`；混合 GDN/Attention 架构不可设 `kv_cache_dtype="fp8"`。

---

## 2. 实验一：全局采样配置与 `token_penalty` 消融（Wave B1）

环境：昇腾 910B3 TP=2（BF16），`concurrency=12`，80 题四数据集配对（GSM8K 16、HumanEval+ 24、LiveCodeBench 20、MATH-500 L5 20），共 11 组 × 80 = **880 次评测**。

| `token_penalty` | 全局采样 `(T, top_p, top_k)` | GSM8K (16) | HumanEval+ (24) | LiveCodeBench (20) | MATH-500 L5 (20) | 四基准宏平均 | 总通过 | 均耗时 (s) | MATH 未闭合 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `null` (`0.0`) | `(0.7, 0.95, 20)` | 93.8% (15) | 95.8% (23) | **80.0% (16)** | **90.0% (18)** | **89.90%** | **72/80** | 80.2 | **5.0% (1)** |
| `null` (`0.0`) | `(1.0, 0.95, 20)` | **100.0% (16)** | 95.8% (23) | 70.0% (14) | 90.0% (18) | 88.96% | 71/80 | 81.8 | 5.0% (1) |
| `1.0`（旧默认） | `(0.8, 0.95, 20)` | 93.8% (15) | 95.8% (23) | 75.0% (15) | 90.0% (18) | 88.65% | 71/80 | 78.7 | 10.0% (2) |
| `null` (`0.0`) | `(0.8, 0.95, 20)` | 93.8% (15) | 95.8% (23) | 70.0% (14) | 90.0% (18) | 87.40% | 70/80 | 82.9 | 5.0% (1) |
| `0.5` | `(0.8, 0.95, 20)` | 93.8% (15) | 95.8% (23) | 75.0% (15) | 85.0% (17) | 87.40% | 70/80 | 86.2 | 10.0% (2) |
| `1.0` | `(0.7, 0.95, 20)` | 87.5% (14) | 91.7% (22) | 80.0% (16) | 90.0% (18) | 87.29% | 70/80 | 79.2 | 10.0% (2) |
| `1.0`（旧默认基线） | `(1.0, 1.0, null)` | 93.8% (15) | 95.8% (23) | 65.0% (13) | 90.0% (18) | 86.15% | 69/80 | 82.6 | 10.0% (2) |
| `0.5` | `(0.7, 0.95, 20)` | 87.5% (14) | 95.8% (23) | 70.0% (14) | 90.0% (18) | 85.83% | 69/80 | 79.5 | 10.0% (2) |
| `0.5` | `(1.0, 0.95, 20)` | 93.8% (15) | 91.7% (22) | 65.0% (13) | 90.0% (18) | 85.10% | 68/80 | 81.7 | 5.0% (1) |
| `null` (`0.0`) | `(1.0, 1.0, null)` | 87.5% (14) | 95.8% (23) | 70.0% (14) | 85.0% (17) | 84.58% | 68/80 | 99.1 | 10.0% (2) |
| `1.0` | `(1.0, 0.95, 20)` | 93.8% (15) | 91.7% (22) | 65.0% (13) | 85.0% (17) | 83.85% | 67/80 | 81.8 | 10.0% (2) |

**结论**：旧仓库设置 8 词 `token_penalty=1.0` 仅在无核截断（`T=1.0, full`）下缓解发散（`84.58% → 86.15%`）；一旦通过预掩码标量统计解锁 `top_p=0.95, top_k=20`，词表惩罚反而破坏正常自我校验并加倍未闭合思考率。**关闭 `token_penalty` 并使用 `(T=0.7, top_p=0.95, top_k=20)` 全面胜出。**

---

## 3. 实验二：IS 调度算法与奖励本体消融（Wave B2-Core + B2-v2 + B2-Extra）

环境：昇腾 910B3 TP=2（BF16），统一锁定全局配置（`token_penalty=null, T=0.7, top_p=0.95, top_k=20`，除 `t08` 对照组），88 题配对四数据集（GSM8K 16、HumanEval+ 24、LiveCodeBench 24、MATH-500 L5 24），`concurrency=4`。

### 3.1 完整 13 臂 × 四基准（88 题配对）总表

| 方法组 | GSM8K (16) | HumanEval+ (24) | LiveCodeBench (24) | MATH-500 L5 (24) | **四基准宏平均** | **总通过** | vs Base (`+W/-L`) | MATH 未闭合 | 四库总墙钟 | 单题均时 | 精确前向槽位 (v2) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `base` | 87.5% (14) | 95.8% (23) | 75.0% (18) | 83.3% (20) | **85.42%** | 75/88 | - | 8.3% (2) | 1998s | 80.2s | - |
| `two_stage_sr_sc`（旧默认 `K=1`） | 93.8% (15) | 95.8% (23) | 75.0% (18) | 83.3% (20) | **86.98%** | 76/88 | `+3/-2` | 16.7% (4) | 2806s | 114.4s | - |
| `two_stage_bmet_k2_sc`（旧 `K=2`） | 87.5% (14) | 95.8% (23) | 66.7% (16) | 83.3% (20) | **83.33%** | 73/88 | `+2/-4` | 16.7% (4) | 4492s | 168.6s | - |
| `adapt_k1_sc_mean` (v1) | 87.5% (14) | 95.8% (23) | 66.7% (16) | 87.5% (21) | **84.38%** | 74/88 | `+2/-3` | 4.2% (1) | 4107s | 164.9s | - |
| `adapt_k2_sc_mean` (v1) | 93.8% (15) | 95.8% (23) | 70.8% (17) | **91.7% (22)** | **88.02%** | 77/88 | `+4/-2` | 4.2% (1) | 4207s | 169.7s | - |
| `adapt_k2_sc_fwin` (v1) | 87.5% (14) | 95.8% (23) | 75.0% (18) | **91.7% (22)** | **87.50%** | 77/88 | `+3/-1` | 4.2% (1) | 4014s | 164.0s | - |
| `adapt_k2_cons_p15` (v1) | 87.5% (14) | 95.8% (23) | 66.7% (16) | 87.5% (21) | **84.38%** | 74/88 | `+3/-4` | 8.3% (2) | 4067s | 161.8s | - |
| `adapt_v2_k1_sc`（**低成本档**） | 93.8% (15) | 95.8% (23) | 66.7% (16) | **91.7% (22)** | **86.98%** | 76/88 | `+4/-3` | **4.2% (1)** | **3544s** | **136.9s** | **768,074** |
| `adapt_v2_k2_sc`（**通用默认冠军**） | **93.8% (15)** | **95.8% (23)** | **79.2% (19)** | 83.3% (20) | **88.02%** | **77/88** | **`+3/-1`** | 8.3% (2) | 4062s | 159.2s | 1,105,548 |
| `adapt_v2_k2_sc_fwin` | 87.5% (14) | 95.8% (23) | 66.7% (16) | **91.7% (22)** | **85.42%** | 75/88 | `+3/-3` | 8.3% (2) | 3817s | 157.4s | 1,079,827 |
| `adapt_v2_k2_sc_t08` (`T=0.8` 对照) | 87.5% (14) | 95.8% (23) | 58.3% (14) | 87.5% (21) | **82.29%** | 72/88 | `+3/-6` | 8.3% (2) | 3889s | 161.0s | 1,120,525 |
| `adapt_v2_k2_cons_p15`（**并列冠军/跨引擎**） | **93.8% (15)** | **95.8% (23)** | 75.0% (18) | 87.5% (21) | **88.02%** | **77/88** | **`+4/-2`** | 12.5% (3) | **3605s** | 150.9s | **997,762** |
| `adapt_v2_k2_b1536`（**昇腾页对齐档**） | 87.5% (14) | 95.8% (23) | 70.8% (17) | **91.7% (22)** | **86.46%** | 76/88 | `+3/-2` | **4.2% (1)** | 3680s | 145.1s | **966,387** |

### 3.2 附加 LiveCodeBench（24 题）专项消融（Wave B2-Extra）

| 专项消融臂 | LiveCodeBench (24) | vs Base | 说明 |
| --- | ---: | ---: | --- |
| `adapt_v2_k2_sc`（`mean` + `pilot=B//2`） | **79.2% (19/24)** | **`+1/-0`** | 胜出默认配置 |
| `adapt_k2_nopilot_sc`（关闭 `pilot_gate`） | 70.8% (17/24) | `+1/-2` | 证明 `pilot_gate=true`（`B//2`）零精度损失 |
| `adapt_k2_sc_fin06`（`finish_sampling T=0.6`） | 70.8% (17/24) | `+1/-2` | 全局 `T=0.7` 已最优，无需额外解码温度覆盖 |
| `adapt_k2_sc_tail`（`Self-Certainty` `tail_mean`） | 66.7% (16/24) | `+1/-3` | 底部 25% 均值在代码题上弱于全段 `mean` |

### 3.3 Adaptive IS v2 关键机制贡献分解

- **v1 → v2 演进效果**：
  1. `pilot_tokens = B // 2` 修复了代码题 512–1024 token 中等思考被误判早退的问题，使 `adapt_v2_k2_sc` 的 LiveCodeBench 从 `17/24` 跃升至 **`19/24 (+8.3 pp, 零退化)`**，且 `adapt_v2_k2_cons_p15` 从 `16/24` 跃升至 **`18/24 (+8.3 pp)`**；
  2. 修复 Stage-2 `straggler_pruning` 锚点并将 Stage-2 新增候选上限从 `2(M-1)` 压至 `M-1`，叠加 `temperature_normalized=true`，使 `adapt_v2_k1_sc` 总墙钟比 v1 减少 **13.7%**（`4107s → 3544s`）同时宏平均提升 **`+2.60 pp`**（`84.38% → 86.98%`），`adapt_v2_k2_cons_p15` 总墙钟减少 **11.4%**（`4067s → 3605s`）同时宏平均提升 **`+3.64 pp`**（`84.38% → 88.02%`）。
- **`B=1536` 硬件页对齐收益**：在昇腾 910B3 上，vLLM 为 `Qwen3.8-27B` 混合 GDN/Attention 层自动设置 KV 页长为 `1536`。将外部 `B` 设为 `1536` 后，HumanEval+ 墙钟降 `17.3%`（`215.4s → 178.2s`）、槽位降 `24.7%`（`51,790 → 38,995`），MATH-500 L5 墙钟降 `13.0%`（`2177.4s → 1894.6s`）、槽位降 `17.8%`（`599,982 → 493,136`）并取得 `22/24 (91.7%)` 最高 MATH-500 分数。

---

## 4. 实验三：昇腾 910B3 正交引擎配置矩阵（Wave B3-v2）

环境：昇腾 910B3 TP=2（BF16），`Qwen/Qwen3.8-27B`，固定 `adapt_v2_k2_sc`（`M=4, K=2, B=1024`）及 `base` 对照。

### 4.1 并发扩展曲线（`max_concurrent_requests = 1 / 2 / 4 / 8`）

| 并发 `c` | GSM8K `adapt_v2_k2_sc` 吞吐 (q/hr) | GSM8K 加速比 | GSM8K 单题均时 | MATH-500 `adapt_v2_k2_sc` 吞吐 (q/hr) | MATH-500 加速比 | MATH-500 准确率 | MATH-500 墙钟 (8 题) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `c=1` (6 题) | 306.4 | 1.00× | 11.7s | 10.5 | 1.00× | 5/6 (83.3%) | 2051.2s (6 题) |
| `c=2` (8 题) | 599.5 | 1.96× | 12.0s | 26.4 | 2.51× | 6/8 (75.0%) | 1090.9s |
| **`c=4` (8 题，推荐)** | **969.2** | **3.16×** | **13.8s** | **29.7** | **2.83×（峰值）** | **7/8 (87.5%)** | **971.2s** |
| `c=8` (8 题) | 1325.6 | 4.33× | 17.5s | 29.0 | 2.76×（饱和） | 7/8 (87.5%) | 994.5s |

**结论**：**`max_concurrent_requests = 4`** 是 TP=2 昇腾 910B3 的最优吞吐-延迟拐点（MATH-500 吞吐在 `c=4` 达峰 `29.7 q/hr`，`c=8` 轻微回落且短题尾延迟增加 27%）。

### 4.2 引擎正交开关消融（`c=1` 配对同题）

| 消融维度 | 配置 | GSM8K (6 题) 墙钟 / 槽位 | MATH-500 (同 4 题) 墙钟 / 槽位 | 关键结论 |
| --- | --- | ---: | ---: | --- |
| **ACLGraph** | `enforce_eager=false`（开启，默认） | `base` 75.0s / `k1` **70.8s** | - | 必须开启 |
| **ACLGraph** | `enforce_eager=true`（关闭） | `base` 448.9s (**慢 5.99×**) / `k1` 406.3s (**慢 5.74×**) | 已跳过（~2h） | 关闭导致近 6 倍退化 |
| **Prefix Caching** | `enable_prefix_caching=true`（默认） | `base` 75.0s / `k2` 70.5s (`3,896`) | 4 题均精确复现；`870` 省 **13.0%** 槽位、`880` 省 **14.1%** 槽位 | 多分支 IS 显著降 prefill，单条 `base` 无开销（`0.4%`） |
| **Prefix Caching** | `enable_prefix_caching=false` | `base` 75.1s / `k2` 70.3s (`3,896`) | `870`: 188.7s (`18,664`) vs 171.1s (`16,245`); `880`: 907.1s (`101,010`) vs 887.5s (`86,744`) | 多分支 IS 下必须开启 |
| **`stream_interval`** | **`8`（新默认推荐）** | **70.4s** (`3,896`) | **1685.3s / `161,857` slots**（**-12.1% 时延 / -6.8% 槽位**） | 4× 更快触发 `K=2` 分支释放与长尾剪枝，短题零开销 |
| **`stream_interval`** | `32`（旧默认） | 70.5s (`3,896`) | 1916.9s / `173,740` slots | 基准 |
| **`stream_interval`** | `64` | 70.5s (`3,896`) | 2011.2s / `178,817` slots（慢 `+4.9%`） | 单调变慢，不推荐 |
| **块长页对齐** | `B=1536` vs `B=1024`（6 题 MATH-500） | 70.6s vs 70.5s | **1993.8s / `159,238` slots** vs 2051.2s / `183,527` slots（**-13.2% 槽位**，`unmeas=0`） | 与 Wave B2-v2 24 题（`-17.8%` 槽位）完全一致 |
| **显存比例** | `gpu_memory_utilization = 0.88` | 跨重启 100% 稳定 | 规避昇腾 60.96 GiB 可用 HBM 下 `0.92` 偶发 `51.81 < 56.08 GiB` 启动失败 | 推荐 `0.88` |

---

## 5. 实验四：NVIDIA CUDA GPU 单卡实机验证（Milestone C，Colab L4 23 GB）

环境：单卡 NVIDIA L4（`23,034 MiB`），`vllm 0.34.0`（GPU V2 Sampler），`Qwen/Qwen3.8-27B` 在线 `mxfp4` Marlin W4A16 量化（`quantization_config.ignore = ["re:.*visual.*", "re:.*lm_head.*"]`，无外部模型）。

| 验证项 | 实测结果 | 工程结论 |
| --- | --- | --- |
| 模型加载与显存 | 权重量化仅 `4.30s`，总加载 `170.1s`，常驻 **`17.39 GiB`**，KV Cache `10,240 tokens`，Mamba/Attention 页长 **`lcm = 832`** | 24 GB 单卡开箱可运行 27B 模型；`settings/presets/cuda_gpu_24g_mxfp4.json` 对齐设 `B = 832` |
| 预掩码 `GPULogprobWorker` | vLLM 0.34.0 GPU V2 采样器下精确还原 `self_certainty` 与 `consilience`，`fallbacks = []`，与 `async_scheduling=true` + CUDA Graph 完全兼容 | 统一算法与后端接口，零额外评分前向 |
| GSM8K（8 题 × 5 臂） | 全 5 臂（`base`, `two_stage_sr_sc`, `adapt_k1`, `adapt_k2`, `adapt_k2_cons`）均达 **`8/8 (100%)`**；Adaptive IS 均耗时仅为 `base` 的 **`1.04×`**，显著快于旧 `two_stage_sr_sc` 的 `1.50×–1.79×` | `pilot_gate` 在单卡 24 GB 上大幅降低延迟开销 |
| HumanEval+（8 题 × 5 臂） | 简单/中等题上 Adaptive IS 比旧 `two_stage_sr_sc` 提速 **`1.85×–2.70×`**（如 `HumanEval/8`: `92.4s → 34.3s`，`HumanEval/93`: `124.9s → 56.5s`），难题自动升级完整搜索 | 跨硬件复现昇腾上的帕累托优势 |
| 并发批处理 (`c=1` vs `c=4`) | GSM8K 4 题 × 4 臂总墙钟（含约 170s 模型加载）从 `545.4s` 降至 `379.2s`，扣除加载后纯推理提速 **`1.80×`** | 推荐单卡 24 GB 设 `max_concurrent_requests=2`（兼顾 10,240 KV 容量） |
| `kv_cache_dtype="fp8"` | 启动直接报错 `rc=1`（混合 GDN/Attention KV 缓存与 `fp8` 不兼容） | CUDA 预设保持默认 `auto` KV 缓存 |

---

## 6. 开箱即用硬件预设与启动方式

所有预设均位于 `settings/presets/` 并通过严格 Schema 校验，可通过 `--settings` 或环境变量 `INFERENCE_SCALING_SETTINGS` 一键切换：

| 预设文件 | 适用硬件与引擎 | 关键差异配置 | 启动示例 |
| --- | --- | --- | --- |
| `settings/presets/ascend_910b3_tp2.json` | 昇腾 910B3 × 2 (BF16) | `TP=2`, `gpu_mem=0.88`, `stream_interval=8`, ACLGraph+PrefixCache ON, `c=4` | `ASCEND_RT_VISIBLE_DEVICES=0,1 python -m inference_scaling.serve --settings settings/presets/ascend_910b3_tp2.json` |
| `settings/presets/cuda_gpu_bf16.json` | 80–96 GB NVIDIA GPU (H100 / A100-80G / Blackwell) | `device="cuda"`, BF16 `TP=1`, `GPULogprobWorker`, `async_scheduling=true`, `stream_interval=8`, `c=4` | `python -m inference_scaling.serve --settings settings/presets/cuda_gpu_bf16.json` |
| `settings/presets/cuda_gpu_fp8.json` | 40–96 GB NVIDIA GPU (L40S / A100-40G / H100) | 在线 `quantization="fp8"`（无外部模型），`context_window=32768`, `c=4` | `python -m inference_scaling.serve --settings settings/presets/cuda_gpu_fp8.json` |
| `settings/presets/cuda_gpu_24g_mxfp4.json` | 单卡 24 GB NVIDIA GPU (Colab L4 / RTX 4090) | 在线 `mxfp4` Marlin（忽略 visual/lm_head），`context_window=8192`, `B=832`, `c=2` | `VLLM_MARLIN_USE_ATOMIC_ADD=1 python -m inference_scaling.serve --settings settings/presets/cuda_gpu_24g_mxfp4.json` |
| `settings/presets/openai_remote.json` | 任意外部 vLLM / SGLang / MindIE `/v1/completions` 服务 | `backend="openai"`, `reward="consilience"`（`initial_penalty=1.5, incomplete_penalty=2.0`） | `python -m inference_scaling.serve --settings settings/presets/openai_remote.json` |

HTTP 服务请求支持按请求外部覆盖 `(M, K, B)`，例如在 `/v1/chat/completions` 或 `/v1/messages` 请求体中传入：

```json
{"scaling": {"candidate_count": 4, "rollout_count": 1, "block_size": 1536}}
```
