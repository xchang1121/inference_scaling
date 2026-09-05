# Stage 10：Verifier-Replay Uno 设计

设计冻结日期：2026-09-05。本阶段针对 Stage 8 的核心矛盾：persistent neural residual 在受限 greedy
repeated-query 上把 TPF 提高了 0.95%，但 update、词表投影和两遍 Uno forward 都还存在，TPS 区间跨 1。
在单张 RTX 3090 上，下一候选应优先消除一次模型 forward，而不是只追求更小的 TV surrogate。

## 1. 归因与定位

该设计不是把 retrieval drafting 宣称为新概念。REST 已证明可从外部 token datastore 检索 continuation
做 speculative draft；CREST 研究了如何压缩 n-gram datastore；DReSD 研究 dense retrieval；RACER
组合 exact pattern 与 logits cue。OSD 则证明线上请求及 verifier feedback 可持续形成 replay 数据。

本项目新增的工程问题是：

> 如何把**本服务刚刚由同一个 AR verifier 确认的轨迹**变成 Uno 的一遍式 fast path，同时用静态 Uno
> 作为 miss/failure fallback，并在实际单 GPU 成本下在线决定何时启用？

因此暂名 **Verifier-Replay Uno（VR-Uno）**。它是已有 retrieval speculation、online feedback 和 Uno
runtime 的组合设计；是否有超出现有系统的研究新意，要等完整相关工作比对和多 workload 结果后再判断。

## 2. 两条执行路径

令当前 committed sequence 为 $x_{1:t}$，Uno block length 为 $B$。

### 2.1 Cache miss：原始 Uno

保持官方两遍路径：

```math
\text{diffusion draft forward}\rightarrow
\text{AR verify forward}.
```

第一遍产生一个 AR clean token 和 $B-1$ 个 diffusion proposals；第二遍做 $\Psi$-Spec。其实际
tokens/forward 是 $\tau_U/2$。

### 2.2 Cache hit：一遍 verifier replay

cache 根据当前 prefix 的最长 suffix 返回 $K=B-1$ 个候选
$c_{1:K}$。Nano-vLLM 的 KV frontier 保留在最后一个 uncached token $x_t$ 之前，因此一次输入

```math
[x_t,c_1,\ldots,c_K]
```

同时得到 $p(c_1|x_{1:t}),\ldots,p(c_K|x_{1:t},c_{<K})$ 和最终 lookahead。proposal law 是保存的

```math
q_i(y\mid x_{1:t},c_{<i})=\mathbf 1[y=c_i].
```

这一分支不运行 diffusion LoRA，只用一个 AR verifier forward。全接受时推进 $K+1=B$ 个 token；首次
拒绝在 $j$ 时推进前 $j-1$ 个缓存 token 和一个 correction。

## 3. Lossless 推导

对单个 cache proposal $c$，标准 speculative acceptance 是

```math
a(c)=\min\left(1,\frac{p(c)}{q(c)}\right)=p(c),
\qquad q=\delta_c.
```

接受分支贡献 $p(c)\delta_c$。拒绝概率是 $1-p(c)$，correction 为

```math
r(y)=\frac{[p(y)-\delta_c(y)]_+}{1-p(c)}
=\frac{p(y)\mathbf 1[y\ne c]}{1-p(c)}.
```

所以总输出律为

```math
p(c)\delta_c(y)+(1-p(c))r(y)=p(y).
```

逐位置条件化即可得到完整 AR 联合分布。cache 可以来自旧模型、错误 domain，甚至恶意低质量文本；只要
本轮实际 $q=\delta_c$ 被 verifier 使用且更新发生在下一轮之后，结果仍是 target $p$。cache 的可信度只决定
是否值得走 fast path，不承担正确性。

greedy/top-k 1 特例更简单：缓存 token 等于 target argmax 就接受；首次不等时直接提交 target argmax。
因此最终 token IDs 与普通 greedy AR 完全相同。

## 4. 为什么有机会把 TPF 变成 TPS

若 $K$ 个 replay token 独立地以概率 $a$ 匹配 greedy target，一遍分支的期望推进为

```math
\tau_R(a,K)=\sum_{j=0}^{K}a^j
=\frac{1-a^{K+1}}{1-a}.
```

原始 Uno 每轮两次 forward，replay 每轮一次；忽略不同 block kernel 的小差异时，必要条件近似为

```math
\tau_R(a,K)>\frac{\tau_U}{2}.
```

Stage 8 static TPF 约 1.4，$K=7$ 时 i.i.d. proxy 的 break-even token match 只约 0.29；精确重复轨迹
$a\approx1$ 时上界为 8 tokens/forward。这只是调度直觉，正式决策必须使用真实 CUDA event 和 wall clock，
因为一遍 base-only block 与 gated-LoRA draft/verify 的成本并不完全相同。

## 5. 有界在线 cache

已实现的 `VerifierReplayCache` 遵循以下约束：

- 只在一个请求完整结束后接收 `prompt + verified completion`，未结束请求不可见；
- 为 completion 每个位置保存长度 `min_suffix..max_suffix` 的 exact suffix 到后续 continuation 的计数；
- lookup 从最长 suffix 向短 suffix 搜索，按出现频率、continuation 长度和 token tuple 确定性打破并列；
- 要求最小 observation 和经验 confidence；
- 同一 key 的 alternatives 和总 keys 都有上限，按 LRU/低频确定性淘汰；
- namespace 必须绑定 model revision、tokenizer 和 sampling policy，避免不同实验静默共享状态；
- 不持久化自然语言或反解 token；正式用户流实验前仍需单独制定隐私、TTL 和跨租户隔离政策。

这不是 response cache：任何命中内容都要被当前 target model 逐 block 验证，绝不直接返回旧答案。

## 6. Past-only 成本路由

`CostAwareReplayRouter` 为每个 matched-suffix-length bucket 维护 replay tokens/forward EMA，并维护 static
Uno TPF EMA。决策顺序固定为：

1. namespace、match length、proposal length、cache confidence 不达标则 static；
2. 每个新 bucket 做固定次数的 exploration；
3. replay EMA 超过 static EMA 的预设 margin 才 exploit；
4. 被禁用的 bucket 每固定请求数做一次 probe，以适应 distribution return。

所有状态只来自已经完成的 static/replay cycles；当前 verification outcome 只能影响未来决策。第一版用 TPF
是为了纯逻辑测试，WSL 集成必须升级为

```math
\widehat U=\frac{\operatorname{EMA}(\text{committed tokens})}
{\operatorname{EMA}(\text{CUDA/event wall time})},
```

并把 lookup/host-to-device 开销计入 replay 分母。

## 7. 与 neural Online Uno 的组合

三层优先级为：

1. 高置信 cache hit：一遍 VR fast path；
2. cache miss/路由拒绝：static Uno 或已在 held-out validation 选出的 frozen residual Uno；
3. verifier feedback：请求结束后更新 cache；只有有明确空闲预算时才更新 neural residual。

这样把 neural learner 用在 retrieval 无法覆盖的新上下文，而 repeated patterns 不再支付 backward。stochastic
sampling 的第一版也使用严格 delta $q$；若接受率过低，再研究
$q_w=(1-w)q_{Uno}+w\delta_c$，但 mixture 需要运行 Uno draft，不能消除第一遍 forward，必须单独评价。

## 8. 实现和验证阶段

### Stage 10A：纯逻辑核心（已实现）

- bounded exact-suffix cache；
- past-only cost router；
- delta proposal 分布构造；
- greedy first-rejection/correction oracle；
- 单 token 枚举证明任意 target $p$ 下输出仍精确等于 $p$；
- cache closure、冲突 continuation、LRU/alternative bounds 和 controller 回归测试。

### Stage 10B：官方 Nano-vLLM 一遍分支

- 在 `TwoPassDecoder.run_cycle` 前做 batch-1 cache lookup；
- 命中时用 `[uncached seed, replay...]` 调用已有 `run_block` 一次；
- verifier payload 去掉已存在的 seed token，再按 commit length rollback KV；
- greedy 后实现 filtered stochastic delta verifier；
- stats 明确区分 `uno_forwards`、`replay_forwards`、hits、accepted prefix 和 lookup time；
- cache miss 的输出、forward count 和 random state 与上游 static path 回归等价。

### Stage 10C：预注册实验

工程 pilot 只允许确定协议参数，不进入正结论。之后用全新请求/seed 冻结：

- exact repeated prompt：验证上界与系统 plumbing；
- templated near-repeat：共享格式但替换实体/数值，排除完整答案 lookup 的单一解释；
- mixed-domain stream：量化错误命中和 fallback 下尾风险；
- static AR、static Uno、Stage 8 residual、VR-only、routed VR-Uno 配对比较；
- 分别报告 TPF、TPS、cache hit、prefix accepted、first-hit latency、RAM、GPU memory 和 break-even requests；
- formal test cache 只能由时间上更早的 train stream 填充，validation 只选 route snapshot，test 完全冻结。

## 9. 参考原始来源

- [REST: Retrieval-Based Speculative Decoding](https://arxiv.org/abs/2311.08252)
- [CREST: Effectively Compacting a Datastore](https://arxiv.org/abs/2408.04678)
- [DReSD: Dense Retrieval for Speculative Decoding](https://arxiv.org/abs/2502.15572)
- [RACER: Retrieval-Augmented Contextual Rapid Speculative Decoding](https://arxiv.org/abs/2604.14885)
- [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)
- [Test-Time Speculation](https://arxiv.org/abs/2605.09329)
- [Online Spec / When Drafts Evolve](https://arxiv.org/abs/2603.12617v2)

