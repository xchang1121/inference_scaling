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

完整证据和边界见 [硬件与可复现性审计](docs/HARDWARE_REPRODUCIBILITY_AUDIT.md)，算法来源见
[文献矩阵](docs/LITERATURE_REVIEW.md)，阶段门和成功判据见 [研究路线图](docs/ROADMAP.md)。
Stage 2 的 checkpoint、采样语义和正式运行矩阵见
[Uno-1B 复现实验协议](docs/STAGE2_UNO1B_PROTOCOL.md)，实测结论见
[Uno-1B 结果报告](docs/STAGE2_UNO1B_RESULTS.md)。

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
| 3 | 静态与在线 proposer 的可控仿真，验证更新收益/成本边界 | 待实现 |
| 4 | verifier-feedback 在线蒸馏和 fast-weight adapter | 待实现 |
| 5 | 自适应 block/update controller、消融和最终报告 | 待实现 |

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
