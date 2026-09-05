# R7D：同前缀、同 KV 的官方 Uno 数值诊断

2026-09-05，在 R7D GPU 结果产生前冻结。此阶段只诊断，不改变在线策略，不申报速度/质量收益。

## 问题和可证伪假设

Stage 12 未修改的 BF16 / FA2 Uno 不同 B 已出现 greedy token 分叉，固定 B=8 的 shadow
却与官方原版 8/8 完全相同。不能据此把分叉统称为 BF16 误差，也不能把分叉归咎于 R7。

1. 同一个条件历史、已有 KV 完全相同，改变 B 后 root hidden/logits 是否不同？
2. 固定 B，改变未来 token 或 noise-row LoRA mask，因果 seed 行是否不变？
3. 固定所有输入，graph/eager、重复调用是否一致？
4. 第一处可观测差异在 embedding、第一层子模块、后续层输出还是最终 LM head？
5. 固定 hidden，BF16 LM head 的 full-B 与 single-row 计算是否不同？固定 single-row FP32
   参考 head 是否减少 argmax 分叉？后者只是局部参考，绝非全模型 FP32 真值。

## 冻结矩阵

输入为 `stage12_official_fa2_baseline.json`，原始 SHA-256
`c10f062b909e90ad96be802ee5221933e098578e0a1ee3ee0be2cb22ab785518`。
每题取最小 seed，对 B=4/8/16 与 AR 的首个不同生成位置去重，得到以下 6 个历史：

| 题目 | 下一生成位置（0-based） |
| --- | --- |
| english | 94 |
| chinese | 60、94 |
| code | 25、87 |
| math | 85 |

历史等于原 chat prompt + AR 输出中此位置以前的 token。每个历史先一次性 **base-only prefill**
历史除最后一个 token 以外的部分，再把已知最后一个 token 作为 uncached tail。
所以所有条件在同一个 prefill-built KV 上比较；它不等同于原生成轨迹的 incremental-built KV，
不能单凭本试验解释所有历史累计误差。

6 历史 × B∈{1,4,8,16} × LoRA∈{off, all-zero-mask, noise-only-mask}
× execution∈{graph,eager} × future∈{0,1} × repeat∈{0,1} = **576 forwards**。
future 0/1 使用各自固定 seed 的 15 个随机 token 的前 B−1 项，seed=20270205+100×context_index+future。
两次 repeat 输入完全一致。B=1 的三种 mask/两种 future 是显式冗余控制。

只保留模型参数冻结，不修改权重或 pinned upstream 文件。
同官方 baseline 的 BF16、FA2、TP=1、batch=1、graph shapes 1/4/8/16。
临时停用实例 graph runner 得到 eager 诊断，结束恢复。
完整 checkpoint SHA、upstream revision/clean 状态、软件版本、所有关键数值 flags 存档。
单行 FP32 参考 head 禁用 TF32；不把这个参考 head 接入生成。

## KV、hook 与计量不变量

所有历史及最长 probe 都在一个 256-token KV page 内。使用官方 BlockManager 独占分配，
不发布 prefix cache。每次 probe 前回滚到 frontier=len(history)−1，只写 frontier 及其后 scratch。
每次 probe 前后按位核对 `[0,frontier)` 的全部层 K/V；任何改写立即失败并保留记录。
每次存下 seed 自身 K/V 与最终 hidden/logits 的独立 clone，不能引用可被下轮 graph 覆盖的输出。

eager/off/future=0/repeat=0 额外抓第一层子模块及所有 decoder-layer 输出，hook 不改变输出。
额外 head 运算、hooks、clone、同步、CPU 统计都不属于性能测试。

记录 root argmax（以 torch.argmax 的 tie rule 为准）、top-5、top-1/2 gap、最大值并列数量、
raw logits/hidden/KV 的摘要 hash，以及成对 max-abs、RMS、变化元素数和
**temperature=1 未过滤 softmax TV**。最后一个 TV 不是本轮 greedy target 的 TV；
greedy 两个 delta 分布只要 argmax 不同，TV 就是 1。

必须完成全矩阵后才能分析；失败记录 `completed=false`，不得覆盖或删掉失败证据。
相同 shape 的相同输入重复差异、未来泄漏或 prefix KV 改写均不能用“数值不同很正常”忽略。

## 局部数学界及解释边界

若同一历史下两个 logits 满足 `||l-l'||∞≤δ`，且第一名相对第二名 margin>2δ，
则对任何竞争项 j，`l'_winner-l'_j ≥ margin−2δ>0`，argmax 保持。
反之不意味着一定分叉，只意味着该充分条件不给保证。输出 BF16 舍入可能产生 ties；
不能把 torch.topk 对 ties 的任意顺序当作 torch.argmax 的行为。

对于温度 τ>0 的未过滤 softmax，令 `r=l-l'`，可以先减去任意常数 c：softmax 平移不变。
若 `max(r)-min(r)=R`，中心化后 `||r-c||∞≤R/2`；沿路径的 softmax 导数
`p'_i=p_i (r_i-E_p r)/τ`，有 `TV ≤ min(1,R/(2τ))`（使用 `E|r-E r|≤R`）。
这是保守上界；实测 full-vocabulary TV 才是本试验报告值。
top-k/top-p 有不连续集合选择，不能把这个未过滤界直接搬到截断 target。

官方线性 `Attention.forward(BLOCK_DECODE)` 在此 pinned 版本使用 **causal=True**，
不是 noise-noise 全双向 attention。seed 行只能看历史及自身；row mask 的 noise 部分在因果图上
不应影响 seed。此前 R7 文档关于“双向 draft attention”的描述在本阶段更正。
不同 B 的实际数值 proposal 及 RNG 消耗仍可能不同，不能未经实测/构造就假定完整反事实反馈。

来源：[PyTorch 2.11 数值精度说明](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)
说明数学等价的 batch/slice 运算不保证 bitwise 相同；这支持检验假设，不替代本机诊断证据。
[Uno pinned source](https://github.com/ifm-ai/uno/tree/ed2ee36bb7a3aea8732ebc635b3f09490a032ea3)
提供 causal mask、KV staging、gated LoRA 和 graph 路径的被测实现。
