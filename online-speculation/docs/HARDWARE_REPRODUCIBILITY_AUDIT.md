# Uno 本机可复现性与硬件审计

审计日期：2026-09-05。上游固定为
[`ifm-ai/uno@ed2ee36`](https://github.com/ifm-ai/uno/commit/ed2ee36bb7a3aea8732ebc635b3f09490a032ea3)，
论文固定为 [arXiv:2609.04010v1](https://arxiv.org/abs/2609.04010)。机器可读信息位于
`results/preflight_rtx3090_windows.json`。

## 1. 本机资源

| 组件 | 实测值 | 对本项目的含义 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 3090，24,576 MiB，compute capability 8.6 | 足够 1B 级 Uno 推理和小 LoRA 反传；8B 需严格控制上下文、batch 和训练激活 |
| 驱动 | 596.49，`nvidia-smi` 显示 CUDA 13.2 capability | 能运行现有 CUDA 13 PyTorch wheel；不代表本地编译器也是 13.2 |
| CUDA toolkit | PATH 首个 `nvcc` 为 11.8；另装有 12.6 | 官方 cu128 环境应隔离，不能混用当前 PATH 编译扩展 |
| CPU | Intel i7-12700K，12 cores / 20 threads | 数据预处理和小规模评测充足 |
| RAM | 31.75 GiB | 1B/小模型充足；8B 训练时 CPU offload 和完整 optimizer state 风险较高 |
| 磁盘 | C: 约 301 GiB 空闲 | 足够 1B checkpoint、缓存和小数据；不应下载完整 OpenThoughts3 或保存大量 8B checkpoint |
| Python | 父仓库 `.venv`：Python 3.12、PyTorch 2.13.0+cu130 | CUDA 可用且支持 BF16，适合 Windows 原型，但不匹配官方 runtime |
| Linux | `wsl` 无可用发行版 | 当前不能直接运行官方 Nano-vLLM/FlashAttention 性能栈 |

Windows 的 `Win32_VideoController.AdapterRAM` 只报告 4 GiB 是 32-bit WMI 字段溢出；以
`nvidia-smi` 的 24,576 MiB 为准。

## 2. 官方 Uno 的真实要求

官方仓库固定：Linux x86-64、Python 3.10、PyTorch 2.11.0 cu128、Transformers
4.55.0、Triton 3.6.0、FlashAttention 2.8.3。linear sampler 需要 FA2；tree sampler 还需
从 Hopper 子目录构建 FA3。官方 inference engine 基于 Nano-vLLM，并非普通
`transformers.generate()` 的等价性能路径。详见
[官方安装与复现说明](https://github.com/ifm-ai/uno#installation)。

这带来两个结论：

1. **算法可复现与性能可复现必须拆开。** 当前 Windows PyTorch 可以验证拒绝采样、在线损失、
   fast weights 和收敛，但其 tokens/s 不能冒充官方 Nano-vLLM 数字。
2. **正式吞吐先跑 Uno 1B linear sampler。** 公开 1B adapter 约 224 MB，base 为 0.9B；
   在 24 GiB 显存上有充分余量。tree sampler 的 FA3/Hopper 路径不应作为 Ampere RTX 3090 的
   第一阶段依赖。

## 3. 能复现什么

| 层级 | 本机判断 | 计划 |
| --- | --- | --- |
| $\Psi$-Spec lossless 分布 | 可以完整复现 | 可枚举 categorical target/draft，Monte Carlo 与目标联合分布比较 |
| Uno 1B AR vs linear sampler | 硬件足够，软件环境未就绪 | Linux/WSL2 中跑官方 revision-pinned recipe |
| Uno 1B benchmark accuracy/TPF | 可以 | 先 GSM8K 小样本，再扩展；准确率与 TPF 和 wall clock 分开报告 |
| Qwen3-8B checkpoint 推理 | 24 GiB 边界内有机会 | 只在 1B 路线稳定后尝试短上下文、batch 1；先测峰值显存 |
| Qwen3-8B 全量 diffusion distillation | 不适合原尺度复现 | 不下载 1.2M 全集；在 0.5B--1.5B 或 tiny Qwen 上做缩放实验 |
| 在线 Uno 快速权重 | 可以研究 | 只更新小 rank、少数上层或输出校正器，并测 backward 的净成本 |
| 论文 H200 系统吞吐 | 不能数值复刻 | 只复现趋势与相对 speedup，明确硬件/软件差异 |

## 4. 原论文训练成本

论文的 Qwen3-8B 路线冻结 AR 权重，为每个投影加入 rank-128 LoRA，共 0.35B 可训练参数；
在 OpenThoughts3-1.2M 上训练 3 epochs、14.7B tokens、sequence length 4096，block curriculum
$B\in\{2,4,6,8,12,16\}$。论文报告约 32 小时、4 nodes × 8 H200。
[论文配置与硬件](https://arxiv.org/pdf/2609.04010#page=12)

即使显存通过 gradient checkpointing/offload 勉强压下，3090 相对 32×H200 的总算力和显存带宽差距
也使原尺度训练没有研究性价比。我们将保持以下结构不变而缩小规模：冻结 AR teacher、conditional LoRA、
uniform-noise block、TV/KL distillation、progressive block curriculum；只缩小模型、tokens、rank 和序列长度。

## 5. 发现的上游配置差异

复现必须记录而不能悄悄选择：

| 参数 | 论文 v1 主文 | 公开仓库 `ed2ee36` | 本项目处理 |
| --- | --- | --- | --- |
| Qwen LoRA scale | $\alpha_{\mathrm{LoRA}}=256$ | README/默认配置为 2048，即 $\alpha/r=16$；论文附录也支持 ratio 16 | 以公开 checkpoint/recipe 为主，并单独消融 256 vs 2048 |
| global batch | 主文写 64 | curriculum 与公开命令固定 128 | 正式复现记录两者；checkpoint 对齐用 128 |
| 训练硬件 | 4×8 H200 | curriculum 注释为 2×8 GPUs | 不推测原因；报告 source revision，并把差异列为待向作者核实项 |

这些差异不影响使用已发布 checkpoint 做推理基线，但会影响“从头训练 UnoQwen”的完全复现。

## 6. 环境路线

### 路线 A：立即可做的 Windows 原型

- 使用父仓库现有 CUDA `.venv` 和已下载的 Qwen2.5-0.5B/1.5B。
- 实现与模型无关的 exact speculative verification、在线 replay、TV/KL loss 和 controller。
- 在可微 Transformers 路径测 online backward 时间与显存。
- 所有结论标记为 prototype，不与 Nano-vLLM TPS 混为一谈。

### 路线 B：官方性能复现

- 准备 Ubuntu 24.04 WSL2 或独立 Linux。
- 在 Linux 文件系统而非 `/mnt/c` 创建 Python 3.10 环境。
- 严格安装官方 cu128/FA2 版本，先运行 Uno 1B、FA2 linear sampler。
- 固定 prompt tokens、generated tokens、warmup、seed、temperature、top-p/top-k 和 batch。
- 同一进程分别测 AR 和 Uno，保存 commit、checkpoint revision、GPU clocks/功耗和 peak memory。

安装 WSL/系统组件会改变主机配置且可能需要重启，本阶段不自动执行；算法工作不依赖它而继续推进。

## 7. 结论

这台电脑**能够复现 Uno 的核心算法和小模型加速趋势，也能够开发在线 Uno**；不能合理复现论文的
32×H200、14.7B-token 训练规模。正确的研究顺序是：可枚举正确性 → Uno 1B 官方 linear 基线 →
小模型在线更新 → 端到端成本 controller → 8B 可行性探测。
