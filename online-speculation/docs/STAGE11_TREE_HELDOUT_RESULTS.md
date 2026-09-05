# R3E：独立问题集的完整树验证结果

2026-09-05。12 个新 prompts × 5 seeds × 6 methods = **360 个运行，全部完成**。
每个运行固定输出 128 tokens，总计 46080 tokens。FP32 / Windows HF / RTX 3090。

- [冻结协议](TREE_HELDOUT_PROTOCOL_20260905.md)
- [原始完整数据](../results/stage11_tree_heldout_fp32.json)
- [包含原始文件 SHA-256 的审计](../results/stage11_tree_heldout_audit.json)

模型、adapter、所有算法配置均在运行之前冻结。协议和数据清单先本地 commit；
首次 push 遇到网络错误，随后重试成功。未因为测试结果修改任何配置或筛除任何运行。

## 1. 最重要的结论

**输出正确性门通过：300 个 speculative 运行全部逐 token 等于对应的 60 个 AR 参考。**
数学上的采样分布保持仍由独立小词表枚举和证明支持，不把 greedy equality 误称为
真实模型随机采样分布的完整统计检验。

在线预算版比线性 Uno 快，但**未超过本组更强的固定 32 节点树**。
所以没有“在线学习已超过最优静态树”的证据，更没有论文级在线收益声明。

另有预注册计时有效性限制：AR 的部分运行结束后出现 GPU memory clock 下降。
为遵守保守规则，**整组降级为完整 held-out 工程测量，未标记正式 TPS 成功门通过**。
下面全部 TPS/区间均需带着这一限定阅读；没有删除低频 AR 或它后面的任何运行。

## 2. 全部方法

TPS = 所有输出 tokens / 所有完整 generate-call 秒数。
TPF = 所有 decode tokens / 所有 decoder forwards，和 mean-of-pair TPF ratio 不是同一估计量。

| 方法 | 总 E2E TPS | aggregate decode TPF | 相对 linear B=8 |
| --- | ---: | ---: | ---: |
| AR | 27.6092 | 1.0000 | 0.6510 |
| linear Uno B=8 | 42.4133 | 1.4671 | 1.0000 |
| linear Uno B=16 | 41.9719 | 1.4671 | 0.9896 |
| fixed tree N=16 | 48.3518 | 1.6994 | 1.1400 |
| fixed tree N=32 | **50.5975** | **1.7762** | **1.1930** |
| online budget tree | 49.5762 | 1.7342 | 1.1689 |

在线版的 controller、初始 probe、周期 probe、建树、同步、KV 整理、初始化/结束成本均计入。
没有训练耗时挪到分母外，也没有通过精确重复请求获得缓存收益。

## 3. 如何读在线收益

- 对 linear B=8：总 TPS +16.89%，prompt-cluster ratio-of-sums 区间 [1.1306,1.2070]。
- 对 fixed tree N=16：仅约 +2.53%，区间 [1.0045,1.0438]，没有达到预定至少 +5% 的在线额外收益门。
- 对 fixed tree N=32：约 -2.02%，区间 [0.9658,0.9938]，更大的固定树在这组新问题上优于在线 controller。

以上均为 95% prompt-cluster ratio-of-sums bootstrap 区间；频率审计限制同样适用于区间解释。
N=32 在早期 smoke 上不如 N=16，但在 held-out 更强。两组的问题、输出长度和运行条件不同，
不能仅归因为任务分布迁移。当前 cost/surrogate controller 没有充分恢复这一收益；
不得据此在本组测试后换 preferred
或调 margin，再把调参后的结果继续称为同一独立测试。

本次参数选择不是“失败记录应删除”的理由：保留完整数据，下一版必须另开实验协议。

## 4. GPU 状态审计

每个方法均有 60 次 post-run 快照：

| 方法 | 9501 MHz | 5001 MHz |
| --- | ---: | ---: |
| AR | 39 | 21 |
| linear B=8 | 60 | 0 |
| linear B=16 | 60 | 0 |
| fixed tree N=16 | 60 | 0 |
| fixed tree N=32 | 60 | 0 |
| online budget | 60 | 0 |

低频 post-run 快照全部来自 AR，随后 speculative 方法通常恢复高频。
这与“方法负载不同导致频率反馈”的解释相容，但快照不足以证明其因果来源或完整时间轨迹。
尤其不能把 AR 侧的低频效应包装成纯算法加速。
尽管非 AR 方法结束时均为高频，我们仍不在读到结果后给预注册规则追加例外。

## 5. 实施决策与恢复点

当前本机最快的已测配置是 fixed tree N=32（观察值），在线 budget 保留为研究候选。
不为了保持“online”名称而默认使用更慢的配置，也不声称已找到全局最大 TPS。

下一次推进先由用户保存工作并重启 Windows：WSL 2.7.13.0 和 VirtualMachinePlatform
已经成功安装，但 hypervisor 尚未启动。恢复入口为 scripts/resume_wsl_after_reboot.ps1。

重启后：未修改官方 AR/linear Uno 基线 -> FA2 kernel smoke -> 独立启用候选 FA2 tree patch ->
固定树对照 -> 在线预算/嵌套树 counterfactual feedback。GPU graph、KV copy 与采样门逐层验证。
最后一项目前只有数学推导和 CPU 工具，尚未加入本次运行。

## 6. 完整性与复核

最终 Python 测试套件 **170 passed**；Ruff 检查通过；全部 PowerShell 脚本语法解析通过。
默认关闭的 FA2 tree patch 对锁定上游版本的 `git apply --check` 通过，仍未声称 GPU 运行通过。

审计脚本按冻结 design 重建全部 method/prompt/seed 键，拒绝整组方法缺失、重复替换、
无效时间和不足的输出预算；不会根据实际剩余方法数缩小 expected matrix。
原始 JSON 的 SHA-256 为：

    96dc0db1472989deaa47bb665fc102dad6c05a1374eb3a8b6b50f3ff5eb9c3ee

从父仓库 `inference_scaling` 复核：

```powershell
.\.venv\Scripts\python online-speculation\scripts\analyze_tree_heldout.py --input online-speculation\results\stage11_tree_heldout_fp32.json --output online-speculation\results\stage11_tree_heldout_audit.json
.\.venv\Scripts\python -m pytest online-speculation\tests -q
```

上一步只重建派生审计，不改动原始实验文件。历史失败和负结果仍保留在
[归档索引](../archive/2026-09-05-v1/README.md)及当前结果目录，清理不等同于删除不利证据。
