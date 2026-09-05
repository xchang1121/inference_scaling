# Stage 9：WSL2 与 Uno 官方运行时迁移协议

协议冻结日期：2026-09-05。该阶段只建立并验证官方 Uno linear sampler 所需的 Linux 性能路径；
任何 Online Uno 性能改造必须在静态基线完成后另行预注册，不能用环境调试运行充当正式结果。

## 1. 当前主机与阻塞点

只读审计得到：Windows 11 Pro build 26200、RTX 3090 24 GiB、NVIDIA 596.49 驱动、
i7-12700K、约 32 GiB RAM，C: 约 299 GiB 可用。CPU firmware virtualization 和 SLAT
均已启用；当前没有 pending-reboot 标记。`wsl.exe` 存在，但 WSL optional features、WSL app 和
Linux distribution 均未安装，当前 Codex 进程也不是管理员。

因此安装分为三个明确检查点：

1. 普通权限下生成、审阅并提交安装器；
2. 只为 `wsl.exe --install`/`--update` 启动一个提升权限的 PowerShell；
3. 若 Windows 要求重启，则停止，不自动重启；重启后才初始化发行版和 Python runtime。

## 2. 版本选择

| 层 | 冻结选择 | 原因 |
| --- | --- | --- |
| WSL | WSL2，Microsoft web-download 路径 | 避免依赖 Microsoft Store UI；保留标准可更新内核 |
| Linux | Ubuntu 22.04 LTS | 原生 Python 3.10，与 Uno `>=3.10,<3.11` 约束一致 |
| GPU driver | 只使用 Windows NVIDIA 596.49 | NVIDIA 明确要求 WSL 使用宿主驱动，不安装 Linux display driver |
| Python | 3.10 venv，位于 WSL ext4 文件系统 | 避免 `/mnt/c` 元数据开销和 Windows/Linux wheel 混用 |
| PyTorch | 2.11.0 cu128 | 与锁定的 Uno 上游 recipe 一致 |
| Triton | 3.6.0 | 与锁定上游一致 |
| FlashAttention | 2.8.3 cu12/torch2.11/cp310 wheel | RTX 3090 linear sampler 使用 FA2；避免本机 32 GiB RAM 源码编译 |
| Uno | `ifm-ai/uno@ed2ee36bb7a3aea8732ebc635b3f09490a032ea3` | 与此前 Windows checkpoint 实验相同 |

上游 example 的 shell launcher当前默认 `fa3`，但 FA3 的公开构建目录和优化目标是 Hopper。
RTX 3090 是 Ampere，本阶段必须显式设置 `ATTENTION_BACKEND=fa2` 并使用 linear candidate top-k 1；
不安装或宣称复现 FA3 tree path。

## 3. 系统安装器的允许范围

仓库中的 `scripts/install_wsl2.ps1` 在执行前检查：管理员权限、BIOS virtualization、SLAT、
至少 25 GiB 系统盘空间以及不存在已有 pending reboot。它只执行：

```powershell
wsl.exe --install --distribution Ubuntu-22.04 --no-launch --web-download
wsl.exe --update --web-download
```

第二条只在第一条退出码为 0 时执行。安装器不会重启、不会修改 BIOS、不会删除已有发行版，也不会安装
CUDA 或 GPU driver。执行前后 feature state、命令参数、退出码、原始输出和 restart-required 判定写入
`results/stage9_wsl_install.json`。

## 4. 重启后的 Linux bootstrap

通过 WSL2/GPU 门后再建立环境，顺序冻结为：

1. 在发行版内验证 `uname -m == x86_64`、Ubuntu 22.04、`nvidia-smi` 可见 RTX 3090；
2. 安装最小系统包：`python3.10-venv`、`python3.10-dev`、`build-essential`、`git`、
   `curl`、`ca-certificates`、`ninja-build`；
3. 在 Linux ext4 home 下创建隔离 venv；
4. 安装官方 cu128 PyTorch wheel，再安装上游固定的 FA2 wheel 和 Uno editable package；
5. 导入已下载 Uno-1B base/adapter 到 Linux ext4，并重新核对 checkpoint hash；
6. 先跑 import/kernel smoke，再跑固定短 prompt 的 AR 与 linear Uno smoke。

重启后从父仓库运行以下可重入的 orchestrator；它以 WSL root 只安装 apt 系统包，再以新建的非 root
`singm` 用户建立 venv、复制模型并运行 smoke：

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\online-speculation\scripts\run_wsl_bootstrap.ps1
```

bootstrap 不给 Linux 用户设置口令或 passwordless sudo；后续系统包操作仍需由 Windows 侧显式指定
`wsl --user root`。源代码、venv 和模型的运行副本位于 WSL ext4 的
`/home/singm/online-speculation-work`，Windows 路径只作为可校验的导入源和结果落点。

除非预编译 FA2 或 Triton 明确报告缺少 CUDA compiler，否则不安装完整 CUDA toolkit。若确实需要，
只能安装 `cuda-toolkit-12-8`；禁止 NVIDIA 文档警告会尝试覆盖 WSL driver 的 `cuda`、`cuda-12-x`
或 `cuda-drivers` 元包。

## 5. Stage 9 通过门

所有条件必须同时满足：

- `wsl --status` 表明 WSL2 可用，Ubuntu-22.04 的 VERSION 为 2；
- Linux 中 `nvidia-smi`、`torch.cuda.is_available()` 和 device capability 8.6 均通过；
- `torch==2.11.0`、CUDA runtime 12.8、`triton==3.6.0`、`flash-attn==2.8.3`；
- FA2 forward/backward smoke 与 Uno 自带 CPU/GPU 相关 tests 通过；
- Uno-1B static linear smoke 生成完成，输出中 backend 明确为 `fa2`；
- 系统清单、package freeze、GPU/内核信息和峰值显存进入机器可读结果；
- 安装、runtime bootstrap、静态基线分别 commit + push。

## 6. 后续性能实验边界

完成 Stage 9 后，先在同一 WSL 进程中复现 static Uno 的 AR/linear paired baseline，扫描
$B\in\{2,4,8,16\}$。正式计时必须预热 CUDA graph/Triton/FA2，固定 output length，交替运行 static
对照，并分开报告 prefill、draft、verify 和 wall clock。

Online 路线只允许使用过去 verifier feedback 更新下一轮或下一请求的 proposal；生成本轮 token 时保存的
旧 $q_t$ 仍是 exact verifier 的接受率分母。首要系统改造是把 Stage 8 residual 更新从 Windows HF
词表投影路径迁移到 Nano-vLLM 的 GPU sampler/hidden-state 边界，并以 CUDA event 测量 update 是否能被
draft/verify stream 的空隙摊销。

## 7. 权威安装依据

- [Microsoft WSL 基本命令](https://learn.microsoft.com/en-us/windows/wsl/basic-commands)
- [NVIDIA CUDA on WSL User Guide](https://docs.nvidia.com/cuda/cuda-on-wsl-user-guide/index.html)
- [FlashAttention 官方安装与 GPU 支持](https://github.com/Dao-AILab/flash-attention#installation-and-features)
- [Uno 锁定上游安装说明](https://github.com/ifm-ai/uno/tree/ed2ee36bb7a3aea8732ebc635b3f09490a032ea3)
