# 推理性能设计

本框架采用大 batch 结构与跨 prompt 的连续批处理。独立算法 worker 把同步调用提交给共享
后端，后台 dispatcher 将时间上相邻且采样策略、生成长度、重复前缀数兼容的调用组进行合并。一次
`sample_batch` 或 `score_batch` 的请求不会被拆成零散单条再与其他 prompt 混排；超过预算的 rollout
组优先沿候选前缀的完整重复组切分。例如 15 个候选各 3 条 rollout 在 32 行上限下切成 30+15，而
不是 32+13。这样既保留跨 prompt 组批，也让 Transformers 后端仍能识别每个候选的重复 prefix 并
复制 KV。每个生成请求拥有自己的 seed，因此改变调度顺序不会改变该请求的随机数流；评分结果会按
原请求顺序拆回。GPU 浮点 kernel 仍可能因 batch 形状不同产生
轻微 logits 差异。为避免大词表上的 FP32 累加误差把固定随机阈值推过 token 边界，inverse-CDF 使用
FP64 累加和比较，但 token log-prob 仍来自同一个实际采样策略。异步 benchmark 还会逐方法检查同步与
异步 token 输出是否完全一致；这是每次运行都要验证的实现性质，而不是仅凭请求级 seed 假定成立。
长条件生成仍可能因不同 CUDA batch 形状下的轻微 logits 差异而分叉，因此报告还包含精确 token
匹配率、最终数值答案匹配率、共同前缀比例和分叉题号。若输出不完全一致，wall-time 比率只解释为
相同配置与 seed 下的真实 workload 对比，不解释为固定 token trace 的严格成对计时。

每个精确评分缓存 wrapper 只绑定一个模型；其内部 key 包含完整采样配置、prefix 与 continuation。这对 replay 尤其重要：
同一条历史 completion 往往要在 base 模型及多个 behavior policy 下重评分，而某一温度或截断策略的
分数绝不能复用于另一策略。随机生成结果不会缓存。

on-policy 条件 rollout 会直接携带生成时得到的精确 base-policy log-probability，因此后端不会再进行
冗余的完整序列评分。off-policy rollout 仍显式计算主模型分数。长评分请求会拆成有界 microbatch，
避免 `[batch, length, vocabulary]` logits 张量耗尽 24 GB GPU；该处理不改变概率与请求顺序。
幂分布 MH 的温度 proposal 与目标概率来自同一个主模型。后端因此会在每个生成位置对同一份 logits
同时计算实际 proposal 概率和温度 1 的基模概率，并把两者随 token 一起返回；MH 接受率直接使用这两组
概率，不再为同一后缀增加一次模型前向。这与来源实现读取 scaled/unscaled logits 的做法一致，减少
forward token slot，但不改变四项接受比。

多次采样实验进一步把同一道题的独立 MH 链按阶段和更新编号同步推进。每条链分别抽取后缀起点、
proposal seed 与接受随机数，只把该步所有不同长度的生成请求放进同一个物理 batch；因此它等价于逐链
执行相同随机流，而不是让链共享状态。有限状态测试逐字段比较向量化与逐链结果，真实模型 smoke 也检查
已有逐链 draw 0 的完整输出哈希。变长后缀会增加 padding，报告同时保留实际 forward token slots 与
墙钟：只有墙钟下降时才称为吞吐加速，不能把物理 batch 数减少直接写成 FLOPs 减少。

对支持 `logits_to_keep` 的 Qwen 模型，生成 prefill 只计算最后一个位置的 vocabulary logits，评分也
只保留覆盖 continuation 所需的尾部 logits；Transformer body 仍处理完整上下文，但不会为未使用的
prompt 位置反复执行大词表输出投影。

replay 生成也会跨候选展平：候选拥有不同 fresh 数量时，仍只发出一个异构生成 batch；选择后的
reserve completion 使用同一路径。这消除了逐候选同步点，同时保持 replay key、seed 和 behavior
log-probability 不变。
对于固定基模与固定 behavior 版本，历史 completion 的两侧概率在 cache 构建时完成验证并进入精确
评分缓存。重复查询的在线阶段只读取这些不可变分数；fresh base rollout 在 behavior 下的概率仍在
在线阶段计算并进入 token/FLOPs 账本。端到端口径则把历史生成、base 评分和 behavior 评分全部计入
cache 构建成本，因此预评分只移动计算发生的时间，不会凭空删掉第一次查询的成本。

Transformers 后端不仅会对完全相同的候选 prefix 执行一次 prefill，还会识别同一 batch 中的多组
重复 prefix。例如 (M) 个候选各有 (K) 条 rollout 时，只对 (M) 个不同的“prompt + 候选”前缀
做 prefill，再把每组 KV 状态复制 (K) 次，并继续用一个 (M K) 大 batch 解码。这样既避免了
重复候选前缀计算，也不牺牲 rollout 的并行度；请求级 seed、真实 log-probability 和输出顺序保持
不变。静态动态候选 proposal 按真实 sampling policy 分组，每个 policy 只生成一个 batch，并分别在
base 与辅助 policy 下批量评分。若 proposal factory 显式依赖先前候选，则该部分在算法上必须保持
串行。结果中的 `shared_prefill_tokens_saved` 明确以“同一个物理 batch 对每条序列都重新做完整
prefill”为分母，记录因 KV 复制而没有再次处理的非 padding 前缀 token 数。

计算量以实际 forward token slot 和估算 FLOPs 为主，而不是 wall time。后端对生成 prefill、KV decode
和完整序列评分分别计数；不同模型按 `2 * parameter_count * forward_token_slots` 分别计算后相加。
连续批处理的主要收益是提高硬件利用率，通常不会降低算法 FLOPs；replay 和小 proposal 是否降低
计算量则由上述 token/FLOPs 计数直接判断。耗时、显存和能耗只作为硬件相关补充。

仍可继续加入、且不改变分布的优化包括：

1. 在连续 MH 后缀 proposal 之间保留 paged KV 状态，而不只在一次 generation 内使用；
2. CPU 奖励解析与下一批 GPU 工作重叠；
3. 对变长请求分桶，同时保持请求级 seed 和真实采样概率；
4. 已消费 replay record 留在 design pool，用于改善方差和成本估计，但其数值不泄漏给未来 evaluation
   决策。

硬截断 proposal、未记录的 sampling transform，以及依赖当前 evaluation rollout 数值的数据复用会
改变估计器或 support，因此明确排除。
