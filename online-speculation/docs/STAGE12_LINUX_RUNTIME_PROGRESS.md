# Stage 12：重启后的 Linux Uno 运行时

2026-09-05。用户已重启 Windows，以下为实际执行结果，不再沿用此前的 reboot blocker。

## 已证实

- Windows LastBootUpTime：2026-09-05 12:16:10.500 +08:00；HypervisorPresent=true。
- WSL status 退出码 0；安装 Ubuntu-22.04 成功，CLI verbose 显示 VERSION=2。
- 发行版：Ubuntu 22.04.5 LTS，uname 架构 x86_64。
- 内核：6.18.33.2-microsoft-standard-WSL2。
- Linux `nvidia-smi`：NVIDIA GeForce RTX 3090，Windows 驱动 596.49，24576 MiB，compute capability 8.6。

## 恢复脚本修复

Ubuntu 首次安装后已是 WSL2。原 bootstrap 无条件执行 `--set-version Ubuntu-22.04 2`，
新 WSL 返回 WSL_E_VM_MODE_INVALID_STATE 并导致脚本退出。该失败没有发生在 CUDA 内。
现在先解析 `wsl --list --verbose`，仅在版本不是 2 时转换；缺失或无法解析时明确退出。
随后重新验证 uname 与 GPU 均成功，不删除、不重新注册该发行版。

## 后续检查点

系统 apt 包 -> 非 root 用户/隔离 Python 3.10 venv -> cu128 PyTorch/FA2/Triton ->
官方包导入与 FA2 forward/backward -> 未修改官方 AR/linear Uno。
在这些步骤实际通过之前，不把 Linux runtime 标记为完成。

验收以[用户更新后的口径](CURRENT_ACCEPTANCE_CRITERIA.md)为准；保留 exactness 与完整开销记录。
