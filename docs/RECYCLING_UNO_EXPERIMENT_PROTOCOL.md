# Recycling Uno 实验协议

2026-09-05，在读取新算法 GPU 结果之前提交。下列 pilot 用于筛选，
confirmatory 的最终配置会在 pilot 后另行冻结。

## 实现门

- arbitrary deterministic proposals：解析 one-token law 等于 target。
- 小词表、多 token、历史依赖 tail reuse 与任意在线动作：枚举或 Monte Carlo 验证联合 law。
- fake AR model：all-accept、首/中/尾 rejection、候选缩短、refill、EOS、max_tokens、KV 内容与长度。
- base 和 offline adapter 无 optimizer、无权重写入。
- 空/禁用 recycling 路径与旧 Uno 的 token/RNG/forward 行为回归。
- 同一数学 target 的 greedy 输出严格相同；跨真实 BF16 kernel 形状单独记录数值差异。

## Pilot

模型：锁定 IFM/K2-Horizon-0.9B 与其 Uno adapter，BF16，batch 1。
Windows HF 可先做机制检验；官方 Nano-vLLM 的结果单独文件、单独结论。
greedy 首先隔离输出轨迹；随后至少验证 temperature=0.8/top-k=32 的采样正确性和系统开销。

首批四种未依赖重复请求的任务：英文解释、中文解释、Python 代码、整数/列表规律。
输出 256 tokens；warmup 各活跃形状，至少 3 paired seeds；方法顺序交替。
所有方法使用同 checkpoint、prompt 与 token budget；no-hit 和差工作负载全部保留。

比较 static Uno B=4/8/16、always-recycle、bounded recycling depth=1/2/4、
TPS-gated recycling。先固定 B=8 验证机制，再比较 refill 宽度。
控制状态默认 request-local；跨请求共享作为单独消融，按实际时间顺序运行。
不通过修改测试 prompts、删除失败 pairs 或增加重复率来提升汇总值。

## 吞吐指标

主指标为所有运行总输出 tokens/总 inclusive end-to-end 秒数，
及配对 E2E speedup 的 bootstrap CI。decode TPS、TTFT、TPF、forward 次数、候选命中深度、
recycle/refill 个数、控制开销和实际绝对 TPS 一并报告。
报告 mean paired ratio 与 ratio of sums 两者，不能混为同一估计量。
pilot 的少量 seed CI 只描述工程噪声，不用于宣称开放域泛化。

把背景下载、GPU 竞争、预热、温度等写入运行元数据；confirmatory 在安装下载结束后执行。
TPS 选择依据包含在线计时，不以 TPF 选择后宣称 TPS 最优。

## Confirmatory 预定框架

在 pilot 结束后提交明确选中配置，再使用全新 prompts 和 seeds。
至少 12 个 prompt、4 个任务域、每 prompt 5 paired repetitions，固定输出预算。
从 validation 选择的最优静态宽度是主要基线，另报 B=8 和 AR。
主要不确定性按 prompt 聚类 bootstrap；种子重复只估计同 prompt 的执行波动。
若模型提前 EOS，报告真实返回 tokens，并加固定 token-budget 机制分析。

系统成功：聚类 95% CI 下界大于 1，且总 E2E TPS 点估计至少提升 5%。
若只提升 TPF 或局限一个 domain，如实限定。该阈值并不意味着已经达到论文级结果。

## 后续优化

只有通过机制门后才花时间接入官方 CUDA graphs；固定捕获 B=4/8/16 与
recycle 的常用 K+1 形状，检查 eager fallback 比例。
优化目标依次为：减少 forward、减少 host sync、减少候选更新和采样开销、
降低无效 refill/recycle 探索。任何新设计都先记录再读取新 held-out 数据。
