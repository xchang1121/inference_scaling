# Stage 10B：Verifier-Replay Uno 真机工程 Pilot

运行日期：2026-09-05。原始机器可读结果为
`results/stage10b_replay_uno1b_rtx3090_hf_pilot.json`。

## 结论边界

这次实验确认了一件重要但范围很窄的事：在 RTX 3090 上，真实 K2-Horizon-0.9B-Uno checkpoint 的
KV-cache 解码循环可以把**同一条已经由 target verifier 确认的 greedy trajectory**作为 delta proposal，
用一次 base AR block forward 代替 static Uno 的 draft + verify 两次 forward，并获得真实 wall-clock 收益。

它不是论文级收益，也不是通用请求分布上的结论：workload 是 exact repeated prompt，backend 是 Windows
Hugging Face fallback，样本只有 5 个配对重复。该实验的角色是 plumbing/性能上界 pilot；它只允许决定下一步
设计，不能替代 WSL 官方 Nano-vLLM 复现或后续预注册 held-out 评价。

## 环境与协议

- GPU：RTX 3090 24 GiB；PyTorch 2.13.0+cu130；
- 模型：锁定的 `IFM/K2-Horizon-0.9B` 与公开 Uno adapter，两个权重 SHA-256 均通过；
- HF compatibility：Transformers 5.16.1、PEFT 0.20.0；Transformers 5 被安装在项目私有
  `.hf-overrides`，没有覆盖父仓库固定的 Transformers 4.53.2；
- sampling：greedy，`temperature=0, top_k=1, top_p=1`；
- block size：8；每个请求固定生成 128 tokens，忽略 EOS；
- 一次 cache-build request，之后 cache 冻结；5 个 future requests 使用全新 Uno noise seeds；
- 每个 future pair 交替 static/replay 执行顺序；bootstrap 10,000 次；
- cache suffix 为 8--32 tokens，continuation 最长 7 tokens；router 只使用过去已完成 cycle 的证据。

为避免 K2 远程配置的 `PreTrainedConfig` 兼容问题污染父项目，运行前执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\online-speculation\scripts\bootstrap_windows_hf5.ps1
$env:PYTHONPATH = (Resolve-Path '.\online-speculation\.hf-overrides').Path
```

## 完整性与无损性

- routing probe：seed/clean rows 与 base 完全相同，noise rows 的 LoRA effect 非零；
- cache-build request 的 hybrid 空缓存路径与 static Uno token IDs、forward count 相同；
- 5/5 future pairs 的 replay、static Uno 和 greedy AR 128 个 token IDs 逐个相同；
- future replay 每次恰好 16 个 base forward，0 个 Uno fallback cycle；
- 真实 KV frontier 每轮按 commit length 检查，所有断言通过；
- cache 只由第一个已完成请求填充，future evaluation 中保持冻结。

## 工程结果

| 指标 | static Uno | verifier replay | 配对 ratio / 95% bootstrap CI |
| --- | ---: | ---: | ---: |
| mean tokens/forward | 1.5052 | 7.9375 | 5.2750× [5.2000, 5.3500] |
| mean decode tokens/s | 38.90 | 204.01 | 5.2535× [5.0180, 5.4943] |
| mean end-to-end tokens/s | 38.67 | 192.54 | 4.9878× [4.7546, 5.2375] |
| decoder forwards / request | 82--86 | 16 | — |

future request 的平均端到端节省为 2.6508 s [2.5240, 2.7777]。第一次请求建立 2,824 个 suffix keys，
纯 CPU cache update 为 3.85 ms。单个训练 pair 观察到的 hybrid/static 差值为 0.306 s；用它做保守一次性开销时，
pilot 的估算回本点为第 1 个 future repeat。把一次 build 和 5 次 future request 全部计入后，累计 wall-clock
speedup 为 2.965×。

这里 7.9375 TPF 基本等于固定 127 decoder tokens、16 forwards 的离散上界；它不是对开放域 cache hit rate
的估计。约 5.3× 的 ratio 同时受 static Uno 在这个 seed 上只有约 1.5 TPF 影响，不能外推到论文 Qwen3-8B、
batch serving 或 H200。

## 暴露的问题和下一改造

1. exact repeat 是 response trajectory replay 的最容易情形；下一轮必须覆盖 template near-repeat、mixed-domain
   和错误 continuation。
2. 当前 cache 只在请求结束后可见，首个请求没有收益。下一实现将增加**因果延迟 session cache**：只有当某段
   continuation 已完整落在 verifier-confirmed past 中才发布，使同一回答中的重复公式、模板或代码片段也能在
   后续位置复用，绝不把未验证未来写入 proposal store。
3. 当前 router 用 TPF EMA。HF pilot 先证明 forward-elimination 能转成 TPS；WSL 官方实现仍要把 lookup、
   host/device 和 CUDA event time 纳入效用分母。
4. 正式结论必须在工程参数冻结后使用 temporal train/validation/test 请求流；test cache 与 controller snapshot
   均冻结，且 exact-repeat 与 near-repeat 分开报告。

