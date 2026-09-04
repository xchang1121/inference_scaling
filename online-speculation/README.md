# Online Speculation

本目录独立研究 [Uno](https://arxiv.org/abs/2609.04010) 的可复现性，以及如何在
`draft -> verify -> accept/reject` 循环中利用 verifier 已计算的分布在线更新 Uno 的扩散
adapter。目标不是只提高接受率，而是在保持原 AR 分布严格不变的前提下，提高包含在线更新成本后的端到端
tokens/s。

## 当前结论

- 本机 RTX 3090 24 GiB 足够运行 Uno 1B 推理、轻量 LoRA 在线更新以及 0.5B--3B
  级别的算法实验；现有 PyTorch CUDA 环境可直接支持原型。
- 论文 Qwen3-8B 的完整 diffusion distillation 使用 14.7B tokens，并报告约 32 小时、
  32 张 H200；本机不适合原尺度训练。
- 官方 Nano-vLLM Uno runtime 目前要求 Linux x86-64、Python 3.10、PyTorch 2.11、
  Triton 3.6 和 FlashAttention 2/3。当前 Windows 主机没有可用 WSL 发行版，因此正式性能
  复现需要先准备 Linux/WSL2 环境；Windows 路线用于算法正确性、checkpoint 接受率/TPF 和
  可微原型。Hugging Face 回退版保留 KV cache 和两次前向语义，但不把其 wall-clock 数字当作
  官方 Nano-vLLM 性能。
- 第一条正式性能路线选择公开的 `IFM/K2-Horizon-0.9B-Uno`，先比较同一 checkpoint 的
  AR、Uno linear sampler，再决定是否投入 Qwen3-8B。checkpoint 级实验现已完成：HF KV-cache
  回退 backend 上 $B=8$ 的 median TPF 为 1.401，paired median decode speedup 为 1.352×；
  官方 Nano-vLLM 路线仍等待 Linux/WSL2。
- exact $\Psi$-Spec 的非平稳仿真也已完成：预注册的 stride-10 discounted-tail 在线策略把
  TV regret 降低约 9.0%，TPF 提高 16.9%；在明确标注为合成的 update-cost proxy 下效率提高
  14.2%。这证明在线反馈有算法与成本空间，但尚不是 GPU online wall-clock 加速。
- 真实 Uno-1B fast-residual 实验已经给出更严格的负结果：安全/冻结门通过，但预注册 stride-10
  的 TPF ratio 为 0.980 [0.947, 1.000]，HF decode TPS ratio 为 0.962
  [0.953, 0.982]，显著变慢；stride-20 结果跨 1。下一版必须用未来 feedback 延迟批准 candidate，
  不能用同一批 held-out loss 假定 temporal generalization。

完整证据和边界见 [硬件与可复现性审计](docs/HARDWARE_REPRODUCIBILITY_AUDIT.md)，算法来源见
[文献矩阵](docs/LITERATURE_REVIEW.md)，阶段门和成功判据见 [研究路线图](docs/ROADMAP.md)。
Stage 2 的 checkpoint、采样语义和正式运行矩阵见
[Uno-1B 复现实验协议](docs/STAGE2_UNO1B_PROTOCOL.md)，实测结论见
[Uno-1B 结果报告](docs/STAGE2_UNO1B_RESULTS.md)。Stage 3 的冻结设计见
[在线仿真预注册协议](docs/STAGE3_ONLINE_SIMULATION_PROTOCOL.md)，结果、失败反例和 Online Uno v1
设计见[在线仿真结果报告](docs/STAGE3_ONLINE_SIMULATION_RESULTS.md)。真实模型 fast-weight 的结构、
top-K surrogate、rollback 与冻结不变量见
[Stage 4A fast residual 设计](docs/STAGE4_FAST_RESIDUAL_DESIGN.md)。
Stage 4B 的冻结矩阵见[真机在线协议](docs/STAGE4B_REAL_ONLINE_PROTOCOL.md)，正式负结果、prompt 分解与
shadow-candidate 改造见[真机在线结果报告](docs/STAGE4B_REAL_ONLINE_RESULTS.md)。
其跨未来窗口批准、exact filtered-overlap 动作和 feedback subsampling 设计见
[Stage 5A deferred controller](docs/STAGE5_DEFERRED_CONTROLLER_DESIGN.md)；Stage 5B 的三工作负载真机判定参数
已在[预注册协议](docs/STAGE5B_DEFERRED_ONLINE_PROTOCOL.md)中冻结。
正式[结果报告](docs/STAGE5B_DEFERRED_ONLINE_RESULTS.md)显示所有安全门通过，但 TPF 与 TPS 主门均未通过；
失败反例把下一版方向收敛到渐进 mixture 或跨请求摊销，而不是继续调 promotion margin。
Stage 6A 的[跨请求 Stream-Uno 设计](docs/STAGE6_STREAM_ONLINE_DESIGN.md)已加入 persistent learner 接口，
将用严格 train/validation/test 请求流检验是否能在未来请求中摊销在线学习。
可执行的 `hf_stream_uno.py` harness 已实现逐请求快照、validation-only 选择、zero fallback 和两种
break-even 计算；pilot 与正式 test 结果将分开保存。
stationary 英文 pilot 的 2 个 test seeds 给出正方向后，Stage 6C 的全新 seed、5-validation/10-test
[预注册协议](docs/STAGE6C_STREAM_UNO_PROTOCOL.md)已经冻结；其成功范围刻意限制为 repeated-query stream。
正式[Stage 6C 结果](docs/STAGE6C_STREAM_UNO_RESULTS.md)显示选中快照的 validation TPF +2.26%，但 10 个新
seed 的 test TPF 反而 -1.95%；head hash 与所有 frozen 审计通过。Stage 7 因此转向 static-anchored
probability mixture，限制 hard activation 的 trajectory 尾部风险。
Stage 7A 的[概率 mixture 设计](docs/STAGE7_STATIC_MIXTURE_DESIGN.md)已实现：static/candidate 各自过滤后再
在概率空间混合，实际 mixture 被原样保存给 exact verifier；当前只开放 frozen stream 评价。
固定 `w=0.25` 工程 pilot 在 validation 上选择了 +5.94% TPF 的 snapshot，但 5 个新 test seeds 的 mean
TPF ratio 只有 0.99493。Stage 7B 因而实现了
[verifier-feedback adaptive mixture](docs/STAGE7B_ADAPTIVE_MIXTURE_DESIGN.md)：每个请求从 static 开始，
只用已完成 verification 的 on-policy TV 证据控制下一轮 capped mixture；residual head 在评价请求中保持冻结。
后续 [adaptive pilot](docs/STAGE7B_ADAPTIVE_MIXTURE_PILOT_RESULTS.md) 对全部非零快照的 validation TPF
都低于 1，因而安全回退 zero；这证明 fail-safe 生效，却没有在线学习收益，Stage 8 转向 greedy target 来隔离
stochastic trajectory shift。
正式 [Stage 8 结果](docs/STAGE8_GREEDY_STREAM_RESULTS.md)首次得到真实 checkpoint 上可确认的跨请求学习
效果：20 个新 Uno noise seeds 的 mean TPF ratio 为 `1.00950 [1.00268, 1.01621]`，通过 +0.5% 实际
幅度门，所有 greedy 输出逐 token 相同；但 TPS ratio `1.00428 [0.98711, 1.02319]`，尚未证明净加速。

## 目录

```text
online-speculation/
|-- docs/                  # 数学推导、文献、实验设计和结论
|-- references/            # 上游论文、代码和 checkpoint 的不可变版本锁
|-- results/               # 只跟踪小型 JSON/汇总，不提交模型或大日志
|-- src/online_speculation # 可复用算法与实验基础设施
`-- tests/                 # 分布正确性和实现回归测试
```

## 环境清单

从父仓库执行：

```powershell
.\.venv\Scripts\python -m pip install -e .\online-speculation --no-deps
.\.venv\Scripts\python -m online_speculation.preflight `
  --repo-root . `
  --output .\online-speculation\results\preflight_rtx3090_windows.json
.\.venv\Scripts\python -m pytest .\online-speculation\tests
```

该命令不下载模型，也不修改系统环境；它只记录可复现性所需的只读机器信息。

## 研究原则

1. **lossless 是硬约束。** 每一轮的 proposal 必须用生成该 proposal 时保存的旧分布
   $q_{\phi_t}$ 做接受率分母；更新后的 $q_{\phi_{t+1}}$ 只能用于下一轮。
2. **接受率不是最终目标。** 所有方案同时报告接受长度、TPF、draft/verify/update 时间、峰值显存和
   端到端 tokens/s。
3. **先小后大。** 先用可枚举分布和小模型证明正确性与净收益，再迁移到 Uno 1B，最后才考虑 8B。
4. **静态对照优先。** 在线实验必须与相同 checkpoint、采样温度、随机种子和 workload 下的 AR 与
   static Uno 配对比较。
5. **逐阶段提交。** 每个达到阶段门的实现、测试和机器可读结果单独 commit 并 push。

## 状态

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 0 | 硬件审计、上游锁定、文献矩阵、实验阶段门 | 完成 |
| 1 | 可枚举的 lossless $\Psi$-Spec 核心与 Monte Carlo 分布检验 | 完成 |
| 2 | Uno 1B AR/linear 真机基线 | checkpoint/HF 回退完成；官方内核待 Linux |
| 3 | 静态与在线 proposer 的可控仿真，验证更新收益/成本边界 | 完成 |
| 4 | verifier-feedback 在线蒸馏和 fast-weight adapter | 完成；安全门通过、预注册性能门失败 |
| 5 | future-validated controller 与真机验证 | 完成；安全门通过、TPF/TPS 主门失败 |
| 6 | 跨请求 persistent learner 与 held-out stream | 完成；validation 收益未泛化到 test |
| 7 | static-anchored 与 verifier-gated probability mixture | 完成；安全回退通过、stochastic 学习门失败 |
| 8 | greedy repeated-query online residual | 完成；学习门通过、HF 系统门失败 |

Stage 1 的正式验证命令：

```powershell
.\.venv\Scripts\python -m online_speculation.lossless_validation `
  --samples 100000 --sequence-length 4 --vocabulary-size 3 --block-size 4 `
  --output .\online-speculation\results\stage1_lossless_validation.json
```

Stage 2 的 Windows checkpoint 级回退命令（需要先按文档取得锁定权重）：

```powershell
.\.venv\Scripts\python -m online_speculation.hf_uno `
  --model-path <K2-Horizon-0.9B目录> `
  --adapter-path <K2-Horizon-0.9B-Uno目录> `
  --block-sizes 2,4,8,16 --max-new-tokens 64 --repetitions 10 --ignore-stop `
  --output .\online-speculation\results\stage2_uno1b_rtx3090_hf.json
```

该命令首先流式校验 2.16 GB base 和 224 MB adapter 的 SHA-256；然后验证 clean/seed 行
不受 LoRA 影响、noise 行确实受到 LoRA 影响；最后交替测 AR 与各 block size。输出中的
`execution_backend` 和 `claim_scope` 是防止把回退数据误写成官方内核复现的强制字段。

Stage 3 的正式可控仿真命令：

```powershell
.\.venv\Scripts\python -m online_speculation.online_simulation `
  --tokens 12000 --seeds 20 --block-size 8 --bootstrap-samples 30000 `
  --output .\online-speculation\results\stage3_online_markov.json
```

该结果使用真实 Stage 2 forward 比例校准两次前向成本，但 online update 成本是透明的合成参数；JSON
同时报告 $0$--$4\times$ update-cost 敏感性和 break-even multiplier。
