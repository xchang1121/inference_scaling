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

第一次 apt 更新遇到 archive.ubuntu.com:80 连接超时，基础索引缺失，随后 ninja-build 无候选。
同一官方域名的 HTTPS 以及 security 官方 HTTPS 均实际返回 200。bootstrap 因此只将官方源 URL
从 HTTP 升级为 HTTPS，原配置备份为 /etc/apt/sources.list.online-uno-original；保留 APT 签名校验，
未修改 Windows 的代理、DNS 或 VPN。增加 update --error-on=any，防止索引失败被当成成功继续。

## 已完成的系统层与下载恢复

随后 APT 安装成功，Python 3.10、venv、build-essential、cmake、ninja、git、rsync 等已就绪；
非 root 用户 singm、ext4 上的固定 Uno 源码和模型副本已建立，模型 SHA-256 与 Windows 副本一致。

PyTorch 官方索引默认选中的 download-r2.pytorch.org 主 wheel 多次断流；持续约 20 分钟仅取得
约 19 MiB。核实同一官方 download.pytorch.org 端点支持精确 HTTP 206 Range，且其对象元数据
SHA-256 与官方 wheel 索引一致后，保留 pip 临时文件的连续前缀，主动终止了经命令行核实的本任务
pip 进程 PID 401（退出 143）。这不是安装成功，也不是因一次观测超时就重启下载。

新增 Python 3.10 标准库分段续传器：最多 12 个并行 1 MiB 请求、每段最多 8 次重试、校验
Content-Range 与长度，最后以锁文件中的完整 SHA-256 验证；已有 NVIDIA 依赖缓存不删除。
未关闭 TLS 或更换到非官方 PyTorch 镜像。仅完整文件校验成功后才允许 pip 安装。

该 820,206,653-byte PyTorch wheel 现已通过锁定 SHA-256，分段续传完成。
重新检查 pip cache 时发现只有约 4.9 MiB 元数据，**并没有可恢复的大型 NVIDIA wheel 缓存**；
不能把此前下载输出直接视为安装或持久缓存成功。依赖的有效完成状态以之后 pip exit 0 为准。

WSL 到 github.com/pypi.nvidia.com 部分连接超时，而 Windows 同域名 HTTPS 可达。
临时 TLS passthrough helper 仅绑定 Windows WSL adapter 172.31.128.1:18743，
只允许 Ubuntu 当前 IP 172.31.134.151、仅 CONNECT:443 到硬编码的 PyPI/NVIDIA/PyTorch/GitHub
软件源白名单、最多存活 30 分钟。只对本次安装命令注入 HTTPS_PROXY，不改变全局代理、VPN、
DNS 或 TLS 校验；结束后主动关闭。经该通道官方 NVIDIA URL 返回 200，大型依赖恢复约 12 MB/s。

FA2 普通下载同样缓慢；保留约 4 MiB 连续前缀后改用相同 hash-locked 分段续传。
所有被中断下载的前缀仍在 task cache 中；没有删除实验数据或模型。

## 后续检查点

系统 apt 包 -> 非 root 用户/隔离 Python 3.10 venv -> cu128 PyTorch/FA2/Triton ->
官方包导入与 FA2 forward/backward -> 未修改官方 AR/linear Uno。
在这些步骤实际通过之前，不把 Linux runtime 标记为完成。

验收以[用户更新后的口径](CURRENT_ACCEPTANCE_CRITERIA.md)为准；保留 exactness 与完整开销记录。

## 实际完成：2026-09-05 13:23 +08:00

完整 bootstrap 退出码 0，pip check 通过。机器可读证据：
[stage9_wsl_runtime.json](../results/stage9_wsl_runtime.json)（沿用预先约定的文件名）。
所有 14 项 checks 均为 true，包括 Python 3.10、Linux x86_64、固定 Uno revision、
RTX 3090/SM 8.6、torch 2.11.0+cu128、Triton 3.6.0、FA2 2.8.3、Transformers 4.55.0、
没有安装 Linux NVIDIA display driver，以及 FA2 forward/backward finite。
kernel smoke shape=[2,128,8,64]，FP16，loss=0.07520008087158203。
这完成运行时验证，还不是模型基线或论文速度复现。

Triton 188,103,592 bytes 的官方索引/S3 SHA-256 一致。中断其 R2 慢连接前，先把所有完整
NVIDIA 临时 wheel 按 ZIP CRC 校验保存，再保存 Triton 前缀；续传完成后从本地安装。
三个关键 wheel 的来源/大小/SHA 锁保留在 config，pip freeze 记录安装来源及版本。

## 用户要求的临时工具清理

安装验证完成后关闭临时 Windows 转发进程，确认 18743 不再监听。
移除一次性 `download_verified_wheel.py`、`preserve_pip_downloads.py`、`wsl_package_tunnel.py`
及只服务于已删除下载器的测试。曾提交的脚本可从 Git 历史恢复。
正式 bootstrap 改为系统 curl + jq，继续保留 HTTPS、SHA-256/长度校验和普通断点续传，
不依赖已删除的临时 Python helper。

Windows 侧重复下载前缀已经移除。临时 Python 源脚本已删除，但其少量 bytecode cache 尚存。
三个已验证 wheel 的 .parts 目录与 .prefix.part 清理先遇到 shell quoting 错误；改用
PowerShell 严格路径检查后的 UNC 删除命令被执行策略拦截，未执行。不绕过该限制，
约 1.18 GiB Linux 分片暂保留，**不能声称已回收该空间**。已完成的完整 wheel、模型、venv、
官方源码及所有实验数据均保留；分片不是实验原始证据，不影响离线模型实验。
