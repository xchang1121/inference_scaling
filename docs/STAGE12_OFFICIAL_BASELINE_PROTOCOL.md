# 官方 WSL / FA2 Uno 基线协议

2026-09-05，在 Linux GPU 模型实验前记录。当前先建立真实官方运行路径，
不把 Windows HF 和 Linux 的不同 dtype/backend 绝对 TPS 之差当作单项算法收益。

- 固定 Uno ed2ee36bb7a3aea8732ebc635b3f09490a032ea3，工作副本 tracked source 必须干净。
- 固定已下载的 0.9B base 与 adapter，运行前重新核对 SHA-256。
- Ubuntu 22.04 / Python 3.10 / torch 2.11 cu128 / Triton 3.6 / FA2 2.8.3。
- 未启用 FA2 tree patch；直接调用官方 LLM.generate。仅将 parameter.requires_grad 设置为 false，
  不修改模型数值、verifier、sampler 或上游源代码。
- 模型 dtype 由官方 Config 决定并记录；本轮不改 Config 来强制对齐 Windows FP32。
- batch=1，max model/batched tokens=2048，GPU memory utilization=0.5，禁止 preemption。
- CUDA graphs 预捕获 batch=1、B=1/4/8/16；不开 torch.compile。
- AR 为 B=1，原线性 Uno 为 B=4/8/16；四个已使用 pilot prompts、每题两次重复，共 32 runs。
- 每次请求最多 128 tokens；temperature=0，top-k=32，top-p=.95，ignore_eos=true，
  random_uniform noise；seed 起点 20270005。每个宽度先预热 128 tokens。
- 所有 prompt 长度必须小于 256，避免完整 prompt-page 缓存命中混入重复请求收益。
- 同一持久引擎内按 prompt/repetition 轮换并反转方法顺序；每次记录 GPU 前后快照。

TPS 为完整官方 generate-call 返回 token 数 / 墙钟时间，包含 prefill、decode、调度和 detokenization，
不含模型加载/graph 初始捕获、共用 prompt 编码、GPU 状态查询、JSON I/O；初始化时间另报。
官方 finalize_output 会去掉末尾 stop tokens，即使 ignore_eos 为 true，因此同时记录实际返回长度，
不得假定一定返回 128 个 IDs。原始 decoder stats 不重写，TPF 根据其原语义解释。
源码检查确认：prefill 会提交一个 token，但不更新 seq.stats；因此 official stats 的
accepts/forwards 是 decode-only TPF，完整预算 128 对应 accepts=127。外层 TPS 仍包含 prefill。

该组是工程复现基线，不是新的 confirmatory study。逐 token 对比 AR，BF16/不同 kernel shape
若出现差异必须记录首差异，不能把理论 exactness 等同于 bitwise 一致。
若初始化或生成失败，保留 completed=false 和完整异常，使用新的结果文件记录修复后的重试。

之后才在同一官方运行时接入在线动作，并按[新的验收口径](CURRENT_ACCEPTANCE_CRITERIA.md)
评估与原 Uno 接近的行为和真实更新开销。没有要求新方法必须显著超过最强静态树。
