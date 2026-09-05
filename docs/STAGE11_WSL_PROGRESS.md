# WSL2 系统安装进度与恢复点

2026-09-05。

## 已完成

- 通过管理员进程启用 Windows VirtualMachinePlatform，状态 Disabled -> Enabled。
- 系统返回 RestartNeeded=true。**未自动重启**，也没有中断用户其他应用。
- 可核对机器记录：[stage11_wsl_platform.json](../results/stage11_wsl_platform.json)。
- 修复 Windows PowerShell 5.1 参数默认值求值时 PSScriptRoot 为空的问题，
  路径改在脚本主体初始化。保留状态检查点和错误信息，避免后台失败无证据。

## WSL 本体安装已完成

官方 WSL 2.7.13 x64 MSI：
[Microsoft WSL release](https://github.com/microsoft/WSL/releases/tag/2.7.13)。
锁定长度 258985984 bytes，官方 SHA-256：

    a3505a50f4cc585551d11d9de824ba4375448d7a68f2e71d3fb315fa986fc754

GitHub release 大连接出现 TLS reset、长时间低速和超时。
当前下载器使用独立的 1 MiB Range、严格检查 206/Content-Range/长度、填补已有数据的缺口，
最终验证完整文件 SHA-256。没有关闭 TLS 验证，也没有使用非官方镜像。
旧 partial binary 数据保留在 ignored cache；无需重新下载已经收到的字节。

MSI 安装脚本额外要求有效 Microsoft Authenticode 签名，使用 /qn /norestart，
将每一步、退出码与 restart_required 保存到 JSON。

2026-09-05 03:00 UTC：完整文件 SHA-256 通过，Authenticode Valid，签名方 Microsoft Corporation。
管理员 MSI 安装退出码 0，结果：[stage11_wsl_install.json](../results/stage11_wsl_install.json)。
安装后 wsl --version 返回 WSL 2.7.13.0、kernel 6.18.33.2-2、WSLg 1.0.73.2。
wsl --status 默认版本为 2，但明确表示虚拟化尚未激活，WSL2 不能启动；尚无 Linux 分发版。
这是与 restart_required=true 一致的实际运行阻塞，不是仅凭安装器消息推测。

## 仍未完成

Windows 重启、Ubuntu 分发版初始化、Linux CUDA/PyTorch/Triton/FA2 smoke，
以及官方 Nano-vLLM Uno baseline。因此不能将本阶段 Windows HF 结果称为官方完整复现。

重启是已确认的外部步骤。完成下载和当前实验后，先 push 恢复点，再请用户自行重启；
恢复时读取此文档和 stage11_wsl_install.json，不重置模型/代码/实验记录。

已提供 scripts/resume_wsl_after_reboot.ps1：先验证上次成功安装之后确实发生过重启，
否则在任何发行版/软件安装之前退出。该保护已在 Windows PowerShell 5.1 下实测。
重启后它会先安装缺失的 Ubuntu-22.04，再检查 Linux 能启动，最后调用已有 bootstrap。
该后半段仍未在本机运行，不能把脚本准备完成视为 Linux 支持栈已安装。

补充只读管理员启动审计：[stage11_wsl_boot_audit.json](../results/stage11_wsl_boot_audit.json)。
firmware virtualization=true、SLAT=true、hypervisor_present=false、VirtualMachinePlatform=Enabled；
BCD 查询成功，未发现显式 hypervisorlaunchtype 值。没有修改 BIOS 或启动配置。
重启后若仍无法启动，将重新检查 hypervisor/BCD，不假定本次重启一定解决所有后续问题。

官方安装依据：[Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install)。

## 重启前追加的依赖完整性检查

FlashAttention wheel 地址来自锁定 Uno README 的安装步骤。2026-09-05 通过
[该发布的 GitHub API](https://api.github.com/repos/lesj0610/flash-attention/releases/tags/v2.8.3-cu12-torch2.11)
取得服务端资产 digest 与长度，并写入 [wheel lock](../config/flash_attn_wheel.lock.json)。
bootstrap 现在要求 SHA-256 和 253651546-byte 长度匹配后才安装，而不是仅记录下载后的 hash。
这锁定了已指定发布资产的一致性，不代表本机已经下载、安装或执行了该 Linux wheel。
