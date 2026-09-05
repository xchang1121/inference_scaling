# R6A：反馈校正在线树的完整 pilot 结果

2026-09-05。**算法实现和正确性验证完成，性能改进未成功，不提升为默认方案。**

本阶段把上一轮只有数学工具的嵌套反馈接入真实 Uno 推理：在线学习实际提交量相对
draft surrogate 的残差，使用已冻结的选择概率校正，并与相同 preferred=32 的旧控制器比较。
没有在线 LoRA SGD、没有额外 teacher forward、没有跨请求重复缓存。

## 1. 完整性

- [协议和数学推导](R6A_FEEDBACK_CORRECTED_TREE_PROTOCOL.md)在 GPU 结果前提交并 push：b0f2f8b。
- 实现与 191 项预运行测试检查点：1a9e208，启动前已经 push。
- [原始数据](../results/stage11_tree_feedback_fp32_pilot.json)：72/72 完成，18432 输出 tokens。
- [完整审计](../results/stage11_tree_feedback_audit.json)：60 个 speculative 运行全部逐 token 等于 12 个 AR 参考。
- base/adapter 参数冻结，optimizer_steps=0；所有 request 的 pending feedback 已完成。
- 运行中仅补充独立结果分析工具和待重启的 WSL 安装校验；未修改模型、控制器或 benchmark 参数。

FP32 / RTX 3090 / Windows HF / batch=1 / greedy。四个旧 pilot prompts，三次重复，
每请求输出 256 tokens。此处不是新的 held-out，不把四个 prompt 的 bootstrap 当论文级证据。
也不将本轮绝对 TPS 与 R3E 的 128-token/12-prompt 绝对 TPS 直接横向比较。

## 2. 全部结果

TPS 为总输出 tokens / 总完整生成调用秒数；TPF 为总 decode tokens / 总 decode forwards。
包含 controller、探索、反馈、同步、KV 整理、初始化与结束成本。

| 方法 | 总 E2E TPS | aggregate decode TPF |
| --- | ---: | ---: |
| AR | 27.9089 | 1.0000 |
| linear Uno B=8 | 41.1037 | 1.4037 |
| fixed tree N=16 | 47.6161 | 1.6452 |
| fixed tree N=32 | **49.6270** | **1.7076** |
| 旧 cost-only online，preferred=32 | 48.4546 | 1.6795 |
| R6A feedback online，preferred=32 | 46.8506 | 1.6523 |

R6A 相对线性 Uno 为 +13.98%，但这不能算在线算法改进成功：

- 对 fixed N=32：-5.59%，prompt-cluster ratio-of-sums 95% 区间 [0.9250,0.9618]。
- 对同 preferred 的旧 cost-only：-3.31%，区间 [0.9440,0.9891]。
- 对 fixed N=16：-1.61%，区间 [0.9300,1.0297]。

## 3. 是真实在线运行，但净收益失败

新控制器共执行 926 个 cycles：N=8 为 106 次，N=16 为 189 次，N=32 为 631 次。
选择原因：72 次初始探测、761 次 exploit 动作、93 次非 exploit 探索动作。
每条记录的小预算 reward update 次数恰好等于覆盖它的大预算执行次数；不是未生效的开关。

固定 N=32 共 896 个 cycles，旧控制器为 911 个。R6A 没有保住大树的接受长度优势，
同时还承担在线维护和探测成本。这是直接观测到的性能边界。
本实验没有单独分离 reward 方差、成本估计滞后和同步开销的因果贡献，不能武断归因于某一项。
条件平均创新正确与残差有界，并不蕴含每轮预测准确，更不蕴含有限请求 TPS 增加。

所有方法的 72 次 post-run 显存频率快照均为 9501 MHz。
因此这次没有触发上一轮的 post-run memory-clock 门失败；但快照不等于完整频率轨迹，
Windows 桌面背景仍存在，且数据自始至终是设计用 pilot，不会被升级成 confirmatory。

## 4. 决策与下一步

保留 R6A 的默认关闭实现、数学和原始负结果，不在这四个问题上继续调参后宣称独立成功。
当前最快的已测树配置仍是 fixed N=32。

WSL 2.7.13.0 已安装，但本轮再次检查 LastBootUpTime 仍为 2026-09-04 22:59:08，
HypervisorPresent=false。Linux runtime 仍被 Windows 重启阻塞；没有自动重启、修改 BIOS/BCD
或安装 Linux NVIDIA display driver。

下一项主工作是用户重启后，通过 scripts/resume_wsl_after_reboot.ps1 初始化 Ubuntu，
验证官方未修改 AR/linear runtime，再验证 FA2 树补丁。后续在线策略需要在该真实成本结构下
重新评估，不能假定 HF 中学到的预算最优点可直接迁移到 CUDA graph/FA2。

安装准备另有实质改进：根据 Uno 官方 README 指定发布资产的 GitHub digest，
FlashAttention wheel 现在必须在安装前通过预先锁定的 SHA-256 和字节数校验。
这只是安装保护，不表示 Linux wheel 已安装或 GPU 验证通过。

## 5. 复核

```powershell
.\.venv\Scripts\python online-speculation\scripts\analyze_tree_heldout.py --input online-speculation\results\stage11_tree_feedback_fp32_pilot.json --output online-speculation\results\stage11_tree_feedback_audit.json --candidate treefeedback:8:32
.\.venv\Scripts\python -m pytest online-speculation\tests -q
```

没有删除或覆写 R3E 原始数据。新增回归测试直接核对两份完整研究的原始 SHA-256，
并逐记录核对 R6A 的 AR 输出一致性和嵌套更新覆盖次数。
包含这两项结果回归后的最终套件为 **193 passed**，Ruff 检查通过。
