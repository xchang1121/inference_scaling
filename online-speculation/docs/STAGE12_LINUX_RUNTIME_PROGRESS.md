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

## 后续检查点

系统 apt 包 -> 非 root 用户/隔离 Python 3.10 venv -> cu128 PyTorch/FA2/Triton ->
官方包导入与 FA2 forward/backward -> 未修改官方 AR/linear Uno。
在这些步骤实际通过之前，不把 Linux runtime 标记为完成。

验收以[用户更新后的口径](CURRENT_ACCEPTANCE_CRITERIA.md)为准；保留 exactness 与完整开销记录。
