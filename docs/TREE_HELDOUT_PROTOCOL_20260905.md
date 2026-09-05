# Tree Uno 独立 held-out 协议

2026-09-05，读取本组 GPU 结果之前提交。

## 冻结选择

HighQoS pilot 的总 E2E TPS：linear B=8 为 41.2743，B=4 为 40.8372；
fixed tree N=16 为 47.8604；rank-only 为 47.8194；budget-only 为 48.7403；
rank+budget 为 48.1241。所有输出 IDs 一致。

选择 **budget-only** 进入下一组；不选择 rank 学习作为默认，因为本组没有显示它稳定回本。
budget-only 相对固定树的 pilot 总 TPS 仅约 +1.84%，远小于它相对线性 Uno 的 +18.09%。
二者不可混淆：第一项才是当前在线控制的额外收益候选。

参数完全冻结：B=8，top-K=4，强制 greedy spine，预算 8/16/32，preferred=16，
每预算两次初始 probe，每 24 cycles 一次轮换 probe，cost EMA=0.8，switch margin=0.02，
rank 校准关闭。所有状态 request-local，无跨请求缓存，无在线网络权重更新。

## 数据与执行矩阵

新 prompts 文件：[tree_heldout_20260905.json](../benchmarks/tree_heldout_20260905.json)。
12 prompts，英文/中文/代码/数学各 3；不来自旧 pilot 的相同问题。
该文件与协议先 commit + push；不根据结果改 prompt、删除困难任务或更改主指标。

模型/adapter hash 与此前相同；FP32、Windows HF、batch=1、greedy、ignore_stop=true。
每 prompt 5 repetitions，全新 seed 起点 20268005；每次固定生成 **128 tokens**。
选择较短预算是为了在本机完成多 prompt 配对，并让在线探索开销完整地承受短请求考验。
结论仅限这一输出长度，不外推到长上下文或长推理链。

方法：AR、linear Uno B=8/B=16、fixed tree N=16/N=32、budget-only tree。
主要线性基线 B=8 由有效 pilot 选定；B=16 是预定保守补充对照。
主要静态树基线 N=16 由 pilot 选定；N=32 作为预定补充对照，不能在结果后忽略它。
总计 12 × 5 × 6 = 360 个运行。

每个方法预热 256 tokens；顺序在 prompt/seed 中轮换并反向交替。
同一进程显式关闭执行速度 EcoQoS，记录 API 前后 mask 与每次 GPU 状态。
MSI 下载/安装已经结束；不同时运行其他研究 GPU 任务。用户桌面应用仍可能产生背景干扰，需记录。

## 计时、正确性与统计

主 TPS 分母使用完整 generate call 墙钟，包括 prefill、draft、verify、候选树构建、
rank/budget 状态维护、KV 整理、同步与请求结束操作。不含模型加载、共用 prompt 编码和 JSON I/O。

每个新方法与相同 prompt 的 AR/静态输出逐 token 比较。若任意数值差异，报告首差异位置，
不得直接宣称 bitwise lossless。小词表联合 law 与 KV oracle 测试独立保留。

同时报告绝对总 TPS、ratio of total seconds、paired mean/median speedup、TPF。
以 prompt 为 cluster bootstrap：重复 seeds 不是独立任务。
对 ratio of total seconds 也直接按 prompt 聚类重采样，避免把 mean ratio 的 CI 当作 ratio-of-sums CI。

相对 linear 的系统成功：总 TPS 提升至少 5%，且 prompt-cluster CI 下界 >1。
“在线本身成功”必须另外相对 frozen tree 达到相同门。没有达到就报告未确认，不能将树收益代替在线收益。
如果 B=16 或 N=32 的结果更强，同时给出更强静态参照下的差值。

若系统频率再次发生大范围切换，不删除单个 pairs；将整组降级为诊断结果。
不因点估计接近门槛而追加更多 seeds，不在本组测试中继续调参数。
