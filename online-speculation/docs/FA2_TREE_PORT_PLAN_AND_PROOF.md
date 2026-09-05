# RTX 3090 / FA2 树验证迁移：候选补丁与 softmax 合并证明

2026-09-05。**运行时实验候选，尚未 GPU 验证，不启用为默认配置。**
顺序仍是重启后先复现未修改官方 AR/linear Uno，再在独立副本验证本补丁。

## 上游边界

锁定 Uno commit ed2ee36bb7a3aea8732ebc635b3f09490a032ea3 的 Config 禁止 FA2 tree。
ModelRunner 仅为 FA3/FA4 启用 flash tree path，Attention 也把 FA2 排除在该路径之外。
官方自身已包含树构建、GPU 树遍历、paged KV 整理与 prefix/suffix cascade。
因此应优先尝试复用这些结构，而不是将 Windows HF 原型的 CPU heap/逐层 KV copy
直接宣称为最佳高性能实现。

核对 [FlashAttention v2.8.3 官方接口](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/flash_attn/flash_attn_interface.py)：
flash_attn_with_kvcache 已提供 return_softmax_lse，返回 LSE 为 [batch,heads,query]，
支持 grouped-query 与 paged KV；页大小要求 256 的倍数。Uno 默认页大小为 256。
这些接口证据支持进行兼容性实验，但不能替代真实 kernel 和 graph 测试。

## 定理：不相交 attention 分区可精确合并

对固定 query q，真实可见 token 集合分为不相交的长 prefix P 与短树祖先 S。
令 a_j = q dot k_j / sqrt(d)，Z_P = sum_(j in P) exp(a_j)，
Z_S 同理，O_P = sum exp(a_j) v_j / Z_P，O_S 同理。
线性展开分子即可得到：

    O = (Z_P O_P + Z_S O_S) / (Z_P + Z_S)。

若 L_P=log Z_P、L_S=log Z_S，则 L=logaddexp(L_P,L_S)，

    O = exp(L_P-L) O_P + exp(L_S-L) O_S。

任一空分区取 L=-infinity、O=0；只要总可见集合非空，公式仍成立。
实际 Uno 每个 query 至少看见自身，因此总集合非空。
该公式不要求 head 数等于 KV head 数；先按 GQA 映射找到正确 KV head 即可。

上游已有的 Triton suffix kernel 直接在同一个稳定缩放基准下累加 S 的分子、分母，
与这个分区恒等式等价。数学上不需要 FA3 特有性质，只需要 prefix kernel 返回
同一 softmax_scale、同一可见 prefix 对应的输出与自然对数 LSE。
若有窗口、ALiBi、softcap 或特殊 mask，分区双方必须采用相同 score 规则；
第一轮仅覆盖当前 K2-Horizon 的 full attention、无 ALiBi/softcap 配置。

数值 caveat：不同 kernel 分块和归约顺序会产生舍入差异；证明的是实数恒等式，
不保证 BF16 bitwise equality。需要同时记录 attention 误差、greedy 首差异与 token law 门。

## GPU 门与失败处理

1. 不修改默认路径，仅显式 UNO_EXPERIMENTAL_FA2_TREE=1 时允许候选 flash cascade。
2. 保留未修改官方 runtime 副本，先跑 AR/linear baseline。
3. GPU 对照 dense FP32 ancestor attention，覆盖 N=8/16/32、GQA32:8、D=64、
   prefix 跨越 256 页边界、不同有效树深度、BF16/FP16。
4. 对 accepted-path KV copy 做内容一致性测试，不只检查长度。
5. eager 通过后再捕获每个允许预算的独立 CUDA graph；采样反馈与 budget 只影响下一次选择。
6. 任一 GPU kernel、mask 或 KV 门失败，不启用补丁、不报告该运行时 speedup。

CPU 分区恒等式测试可先通过；它不能被替换成“FA2/RTX 3090 已经验证”。
此迁移可能降低开销，但当前没有时间结果，不能承诺提升幅度。
