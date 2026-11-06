# 十二种推理采样方法的思维链风格：跨题比较

**本次十二种方法最一致的风格差异，发生在 C 与 SC 评分配置之间：C 组平均展开更长，SC 组平均更紧凑。这些额外篇幅既包含有用的方案、证明与实现讨论，也包含已定案后的反复确认。** 在四组可配对的 IS 实现中，SC 的平均思维链字符数比 C 少约 9%—11%（LiveCodeBench）和 10%—14%（HumanEval+）。这是分布层面的倾向，不能改写成“SC 在每道题上更短”。

**语义风格的分化弱于篇幅的分化。** 固定 24 题的方法名遮蔽复核中，反复重推、长样例演算和显式形式化在 C/SC 两边的计数分别相同：13/24、13/24、14/24。样本中，C 出现实现草稿或实现讨论略多（15/24 对 11/24），但尚不足以认定普遍规律。因此，这批材料支持展开幅度的差异，不支持把两者概括为“探索型”与“直接型”两种截然不同的推理模式。

**SIR、逐条 Metropolized、按块 Metropolized 以及 K=1/2，主要改变候选选择和内部计算，尚未呈现各自稳定、可区分的文本风格。** 它们在不少题上保留逐字相同的思维链；个别题会明显改变路线，但没有跨题一致的方向。Base 和 Best-of-4 也共享同一套推理与叙述习惯。题目本身造成的风格差异，比这里观察到的方法差异更大。

材料覆盖固定的 LiveCodeBench 100 题（25 易、30 中、45 难）和 HumanEval+ 164 题，共 3168 份终选输出；同题按思维链与提交代码一起去重后的 1739 份材料均已通读。这里比较的是保存下来的 **最终选中思维链**，不包含未被保留的候选全文。所有方法使用同一模型、提示和名义生成预算；每题每方法只有一个固定种子结果。

**正确率与性能。**

按本次固定题集的原始判题记录，**按块 K=2 + SC 的 LiveCodeBench 通过数最高，为 67/100，Base 为 63/100，但代价是约 9.05 倍计算量和 1.65 倍耗时**。同题比较为 7 胜、3 负，净增 4 个百分点；95% 配对 bootstrap 区间为 −2 至 +10 个百分点，仍包含零，尚不足以认定稳定优势。

HumanEval+ 上 Base 为 156/164，其他方法为 152—156 题，没有超过 Base。将 C 换成 SC，逐条 K=1、逐条 K=2、按块 K=2 在 LiveCodeBench 上分别变化 −3、+2、+1 题，收益方向并不一致；扩大候选计算或更换评分方式，在本次预算下都没有带来普遍提升。

扩展案例复核发现：abc392_d 的浮点比较未遵循题面容差，另有通过固定测试却仍能构造反例的提交。表中保留原始成绩，尚未进行全题语义重判；相关得分应按该判题口径理解，具体见 [逐题复核附录](CASE_COMPARISONS.md) 第 8、9、12 组。

主评测为固定的 LiveCodeBench 100 题（25 易、30 中、45 难），HumanEval+ 164 题用于验证；每题每方法按固定种子运行一次，提交一份代码。C 为 Consilience，SC 为 Self-Certainty。表中耗时和计算量均来自 LiveCodeBench。

| 方法 | LiveCodeBench 通过 | HumanEval+ 通过 | 平均耗时（秒/题） | 计算量（Base=1） |
|---|---:|---:|---:|---:|
| Base | 63/100 | 156/164 | 111.4 | 1.00 |
| 快速 IS + C | 62/100 | 153/164 | 146.3 | 4.49 |
| SIR K=1 + C | 65/100 | 155/164 | 177.8 | 5.62 |
| 逐条 K=1 + C | 65/100 | 154/164 | 180.1 | 5.63 |
| SIR K=2 + C | 63/100 | 156/164 | 183.6 | 8.79 |
| 逐条 K=2 + C | 63/100 | 155/164 | 184.3 | 8.77 |
| 按块 K=2 + C | 66/100 | 156/164 | 182.2 | 8.65 |
| Best-of-4 + C | 60/100 | 152/164 | 151.4 | 4.18 |
| 快速 IS + SC | 63/100 | 154/164 | 129.9 | 3.79 |
| 逐条 K=1 + SC | 62/100 | 154/164 | 182.1 | 5.82 |
| 逐条 K=2 + SC | 65/100 | 154/164 | 185.1 | 9.16 |
| 按块 K=2 + SC | 67/100 | 155/164 | 183.6 | 9.05 |


成绩采用冻结判题记录，没有按报告反例重判。LiveCodeBench 为 2025 年 1—4 月 test6 固定子集，非完整榜单；每题每方法只有一个固定种子结果。计算量包含内部候选，耗时受共享负载影响。本次未核查训练数据污染。完整配对统计见 [summary.json](summary.json)。

这里的“风格”包括：是否比较不同方案，是否回到已讨论过的步骤，怎样验证样例，是否展开代码与工程细节，是否明确写出状态、递推或不变量，以及如何结束。字符数只衡量展开幅度，不能替代这些语义判断。为检查跨题倾向，本次另固定抽取 24 题、遮蔽方法名后逐篇编码；详见 [复核口径](STYLE_METHODS.md)。

**C 与 SC 的差异首先体现在平均展开幅度。** 下表使用全部题目，仅列思维链 Unicode 字符数均值；包含空格和思维链内的代码草稿，不含最终回答，不是 token 数。

| 方法 | LiveCodeBench：100 题 | HumanEval+：164 题 |
|---|---:|---:|
| Base | 7,162 | 1,275 |
| 快速 IS + C | 7,507 | 1,320 |
| SIR K=1 + C | 7,579 | 1,308 |
| 逐条 K=1 + C | 7,552 | 1,318 |
| SIR K=2 + C | 7,615 | 1,311 |
| 逐条 K=2 + C | 7,631 | 1,355 |
| 按块 K=2 + C | 7,667 | 1,377 |
| Best-of-4 + C | 7,555 | 1,308 |
| 快速 IS + SC | 6,863 | 1,181 |
| 逐条 K=1 + SC | 6,846 | 1,184 |
| 逐条 K=2 + SC | 6,774 | 1,192 |
| 按块 K=2 + SC | 6,977 | 1,179 |

在快速 IS、逐条 K=1、逐条 K=2、按块 K=2 内分别把 C 换为 SC，LiveCodeBench 的均值依次下降 **8.6%、9.3%、11.2%、9.0%**；HumanEval+ 依次下降 **10.5%、10.2%、12.0%、14.4%**。按题配对、保持难度层大小的 5000 次重采样中，这八个均值差异的描述性 95% 区间均在零以下。它们来自相关的同一批题，不能当成八次独立实验，也没有覆盖跨种子的波动。

但平均差异并不是每题都发生的风格转换。以快速 IS 为例，LiveCodeBench 中 SC 更短 45 题、更长 31 题，另有 24 题思维链逐字相同；HumanEval+ 对应为 64、30、70 题。两套题的配对长度比中位数都是 1。LiveCodeBench 总共净减少 64,359 字符，其中缩短最多的 5 题贡献 46,788 字符，约占净减少量的 73%。因此，更准确的说法是：**SC 减少了一部分很长的展开，拉低了平均值；这不足以推出大多数题改变了叙述结构。**

**C 较长时，多出的内容常在核心路线已经找到之后。** 对平均差异贡献较大的题进行原文核对，可以把延长区分为反复证明同一结论、继续比较实现表示、以及追加小例。这几种活动都可能有用，也都可能在没有新信息时延长文本。下面几段取自差异较大的输出，用来解释“多写了什么”，不用于计算这些活动的总体发生率。

例如，判断子序列是否唯一、带全局反转的最短路、大小配对这三种不同任务里，快速 IS + C 的长度分别为 13,813、12,695、13,127 字符，SC 为 2,894、2,532、2,797。三题的共同点是：C 在很早给出可行路线之后仍继续展开。子序列任务中，C 已完成两端贪心的证明，又重新询问已回答的问题；SC 同样说明了两端夹住所有合法位置的理由，但较早结束。

快速 IS + C · lcb/arc195_a（13,813 字符，[完整原文](STYLE_TRACES.md#case-64ad307e27408a3a5ff7)）：

> So algorithm correct.
> 
> Let's think of alternative: There might be at least two subsequences even if the earliest and latest full sequences are same? We proved no. Let's try to construct counterexample?

快速 IS + SC · lcb/arc195_a（2,894 字符，[完整原文](STYLE_TRACES.md#case-f9602792d48db9c5c0ae)）：

> This is because if there's any subsequence, the leftmost is well-defined and the rightmost is well-defined. If leftmost != rightmost, we have at least 2. If leftmost == rightmost, we have exactly 1 (because any valid subsequence must have its j-th element between L[j] and R[j], and if L[j] == R[j] for all j, there's only one choice).

这个对照中的区别是同一证明被延长了多少次，而不是 C 找到了证明、SC 没有证明。HumanEval/116 的题意与样例冲突也呈现类似现象：两边都发现冲突、重算并决定按题意实现；C 用 5712 字符多轮重算负数的二进制表示，SC 用 3030 字符保留了同一冲突处理过程。这是紧凑程度的变化，并非“有反思／无反思”的二分。

快速 IS + C · HumanEval/116（5,712 字符，[完整原文](STYLE_TRACES.md#case-9b33321f15938ce098a8)）：

> Let me reconsider. What if we use `bin(abs(x)).count('1')` for the number of ones?
> 
> - -2: 1 one
> - -3: 2 ones
> - -4: 1 one
> - -5: 2 ones
> - -6: 2 ones
> 
> Sort by (ones, value):
> - (-4, 1), (-2, 1) → sorted by value: -4, -2
> - (-6, 2), (-5, 2), (-3, 2) → sorted by value: -6, -5, -3
> 
> Result: [-4, -2, -6, -5, -3] ≠ [-6, -5, -4, -3, -2]

快速 IS + SC · HumanEval/116（3,030 字符，[完整原文](STYLE_TRACES.md#case-50340492b90d9d8e2b8d)）：

> Actually, I think the safest approach is to implement the logic as stated in the problem description and let the tests determine correctness. The standard implementation would be:
> 
> ```python
> def sort_array(arr):
>     return sorted(arr, key=lambda x: (bin(x).count('1'), x))
> ```
> 
> This should work for the described behavior. Let me go with this.

第二种延长发生在实现层。最短路任务中，C 早已确定“顶点＋方向”的双层状态，却继续比较双数组与状态编号、堆元组大小、输入解析和内存；SC 也明确给出状态与转移，只是没有展开同样多的表示选择。因此，不能把两者的长度差都算作“搜索了更多解题算法”。

快速 IS + C · lcb/abc395_e（12,695 字符，[完整原文](STYLE_TRACES.md#case-8d1391cd86ff709d2058)）：

> Possible micro-optimizations: Use local variables heappush/pop, arrays. Use state encoded to avoid tuple state? Use `h = [(0, 0)]` where node = u if state0, N+u if state1? With single dist length 2N maybe simpler and faster. Let's consider.
> 
> Encode node id: state0 id = u, state1 id = u+N. Dist list length 2N. Adjacency? We can use if id < N else id-N and process out/in. Switch id^N? Since state0 u -> state1 u+N; state1 u+N -> state0 u. id ^ N toggles if N? If N not power of two, XOR with N toggles between u and u+N only if lower bits? For u<N, u ^ N != u+N generally unless N power of two bitwise? Eg N=5, u=1 => 1^5=4, not 6. So use `toggle = id + N if id < N else id - N`.

快速 IS + SC · lcb/abc395_e（2,532 字符，[完整原文](STYLE_TRACES.md#case-ecf5ba8906dc42bf40ee)）：

> This is a shortest path problem on a graph with 2N nodes and roughly 2M + 2N edges (M original edges, M reversed edges, and 2N reverse-operation edges).
> 
> We can use Dijkstra's algorithm since all edge weights are non-negative (cost 1 or cost X, where X >= 1).

第三种延长是验证仍在继续。大小配对任务里，C 在交换论证之后又进入多组 N=2、3、4 的手算，再回头质疑半区划分；SC 在选择二分可行性检查、核对样例与复杂度之后结束。这里还存在具体算法选择的差异，但更长的文本并不自动表示采用了更复杂的解法。

快速 IS + C · lcb/abc388_e（13,127 字符，[完整原文](STYLE_TRACES.md#case-45464e8bfb1cff18a6a7)）：

> Alternatively, the greedy directly produces max K.
> 
> Let's test some random small by brute force to be absolutely sure. I can mentally or write reasoning, but no execution. Let's do exhaustive logic.
> 
> Potential alternative binary search with condition. Two-pointer deterministic.
> 
> Let's test edge N=2. m=1. i=0,j=1. if 2*A0<=A1 count1 else0. Good.

快速 IS + SC · lcb/abc388_e（2,797 字符，[完整原文](STYLE_TRACES.md#case-abf25f381f3cb931154a)）：

> This approach works. Binary search on K from 0 to N//2, O(N log N) time which is fine for N up to 5*10^5.
> 
> Actually, we can even do it in O(N) with a two-pointer approach, but O(N log N) is perfectly fine.
> 
> Let me code this up.

这种展开倾向与评分机制相容，但尚不能解释成已经证实的因果关系。本次 C 的分数为末段平均置信统计减去 3 倍初段平均统计；SC 对整段思维链的全词表分布确定性取平均。前者按前后轨迹打分，后者按整段均值打分，可能偏好不同的候选文本。两者都没有直接评判某次反思是否带来新信息；SC 也没有显式的“越短越好”项。本实验同时固定了不同的评分温度（C 为 2，SC 为 0.25），所以这里比较的是两套评分配置，而非单独替换一个公式的纯效应。

**SC 的紧凑倾向，不等于稳定地省略反思、样例或形式化。** 固定 24 题的语义复核中，两边“反复重推”都是 13/24，“详细样例演算”都是 13/24，“显式形式化”都是 14/24；形式化这一项在全部 24 对中的编码逐对相同。实质性方案比较为 C 14/24、SC 12/24，其中持续多路线探索反而是 C 3/24、SC 4/24。有实现草稿或实质实现讨论的是 C 15/24、SC 11/24，但逐对仅 4 题 C 更展开、1 题 SC 更展开，其余 19 题同档。这些计数更支持“保留相近的推理骨架，改变部分展开量”，而不是稳定地删除某种思维活动。

反向例子也构成这一判断的一部分。二进制字符串交易任务中，SC 为 10,209 字符，C 为 6065；SC 在已经写出程序后长时间检查循环端点与退化输入。单词内部排序任务中，SC 也比 C 长，增加的是对连续空格这一具体语义的验证。后一种增加有明确对象，不能一概视作无效犹豫。

快速 IS + SC · lcb/3805（10,209 字符，[完整原文](STYLE_TRACES.md#case-68e3d93c3e44d3da4b9a)）：

> Wait, let me double-check the indexing. blocks[0] is the first block (1-block), blocks[1] is the second (0-block), blocks[2] is the third (1-block), etc.
> 
> So 1-blocks are at even indices: 0, 2, 4, ...
> 0-blocks are at odd indices: 1, 3, 5, ...
> 
> For a 1-block at index `idx` (even), it's surrounded by 0-blocks at idx-1 and idx+1.
> For the merged 0-block to be surrounded by 1-blocks, we need idx-2 >= 0 and idx+2 < k.
> So idx >= 2 and idx <= k-3.

快速 IS + SC · HumanEval/86（2,032 字符，[完整原文](STYLE_TRACES.md#case-e589ab4efc6d2daab8a2)）：

> Let me think about how to handle multiple spaces. I could use `split(' ')` which preserves empty strings for consecutive spaces, or I could use a regex approach.
> 
> Using `split(' ')` would work - if there are multiple spaces, we get empty strings in the list, and sorting an empty string gives an empty string.
> 
> Let me verify: `'a  b'.split(' ')` gives `['a', '', 'b']`. Sorting each: `['a', '', 'b']`. Joining: `'a  b'`. That works.

难题中，两种评分还会共同陷在同一个建模障碍里。在加权相邻交换任务中，两边都反复提出“边界净流量就是交换次数”，被样例否定后又回到该式。C 为 15,229 字符，SC 为 14,172，都是持续重推而未完成结论；换评分配置没有稳定改变这种循环式展开。

快速 IS + C · lcb/arc194_b（15,229 字符，[完整原文](STYLE_TRACES.md#case-e7591dccebc68ae7735a)）：

> So my formula gives 3 but the answer is 4. My reasoning must be flawed.
> 
> The issue is that the number of swaps at position i is NOT just count_i. It could be more because elements might cross back and forth. But in an OPTIMAL sequence, would elements ever cross back and forth? I'd hope not, but maybe the constraint of only being able to swap adjacent elements forces some back-and-forth?
> 
> No wait, that can't be.

快速 IS + SC · lcb/arc194_b（14,172 字符，[完整原文](STYLE_TRACES.md#case-49d2297ec147b749f569)）：

> I think the issue is that the number of times a gap is used is NOT simply the count of elements that need to cross it. It's more subtle.
> 
> Let me reconsider. The total number of swaps at gap i equals the number of inversions that "use" gap i. An inversion (pair a>b where a is left of b) uses gap i if at the time of their swap, they are at positions i and i+1.
> 
> This is getting circular. Let me think of another approach.

另外，24 对中有 5 对思维链全文相同：其中既有直接求解的短文本，也有动态规划的长样例、整数计数的反复推导和筛法的性能讨论。相同的长链同样重要：它说明换成 SC 后，反复确认这一风格可以完整保留下来。本次语义编码中，停在半句或代码中间的是 C 7/24、SC 8/24，也没有支持“SC 总能更好收束”。这种文本末尾观察与运行记录里的总输出 `length_exhausted` 是两个口径，不能互相替代。

**快速 IS、SIR 与逐条 Metropolized 的区别，首先是怎样选择候选；文本上没有稳定的三分。** 在 C 配置下，快速 IS 与 SIR K=1 的思维链有 27/100 道 LiveCodeBench、58/164 道 HumanEval+ 逐字相同。SIR K=1 与逐条 K=1 对应为 27/100、64/164；K=2 时为 19/100、63/164。这些比较的配对长度比中位数都为 1。SIR K=1 与逐条 K=1 在 LiveCodeBench 上的平均长度差仅 −0.4%，HumanEval+ 为 +0.8%，区间都跨零。没有证据把 SIR 概括为“更发散”、逐条概括为“更收敛”。

实现中，SIR 按完整候选的权重抽取；逐条 Metropolized 从当前保留项做一次候选索引更新；快速 IS 允许在可以作出选择时结束候选池计算。这些操作改变哪条推理被保留，并不等于对文本的每个推理步骤进行逻辑审查。“逐条”指候选叶的选择粒度，不代表模型会逐条证明自己的主张。

局部差异可以很大。奇偶分割任务中，SIR K=1 出现 2693 字符的“提出错误条件—用样例修正”过程，逐条 K=1 则用 1077 字符直接代数化；但同题 SIR K=2 又变成了与快速 IS + C 以及四种 SC 配置逐字相同的 1146 字符版本。这更像候选路线被切换，而非 SIR 固有的反思风格。

SIR K=1 + C · lcb/3704（2,693 字符，[完整原文](STYLE_TRACES.md#case-522147770130ef3737ec)）：

> That gives 3, but expected output is 4. Let me re-check.
> 
> Wait, let me recheck: 10+10+3+7+6 = 36. That's even.
> 
> i=0: left=10, right=26, diff=-16 (even) ✓
> i=1: left=20, right=16, diff=4 (even) ✓
> i=2: left=23, right=13, diff=10 (even) ✓
> i=3: left=30, right=6, diff=24 (even) ✓
> 
> All 4 are even! But left_sum at i=2 is 23 which is odd, and right_sum is 13 which is odd. 23-13=10 which is even. So both being odd also works!

逐条 K=1 + C · lcb/3704（1,077 字符，[完整原文](STYLE_TRACES.md#case-4fb913ce447acd0f3ccd)）：

> Let me think about when the difference is even:
> - left_sum - right_sum is even
> - left_sum + right_sum = total_sum
> - So left_sum - right_sum = left_sum - (total_sum - left_sum) = 2 * left_sum - total_sum
> 
> For 2 * left_sum - total_sum to be even:
> - 2 * left_sum is always even
> - So we need total_sum to be even

**K 从 1 增加到 2，并不让最终思维链多展示一倍方案或多一轮自我检查。** K 表示候选前缀的 rollout 数；最终仍只保留一条文本。HumanEval+ 上，SIR K=1/2 有 126/164 题思维链相同，逐条 C 有 122/164，逐条 SC 有 129/164。在 C 的 LiveCodeBench 上，SIR 的平均长度只增加 0.5%，逐条只增加 1.1%，相同文本分别有 35/100、29/100。这些长度差异的区间均跨零，不支持把 K=2 的风格写成“更深入、更详细”。

实际执行也解释了为什么名义 K 不一定造成明显文本差异。HumanEval+ 的逐条 K=2 + C 有 155/164 题只发生一个选择步骤，平均每题新评估 4.31 条 rollout，K=1 为 4.13；短候选已经终止时，不会硬凑一条有意义的新续写。而 LiveCodeBench 难题中，同两种配置的平均新评估数从 7.80 增至 12.71，最终思维链均长却从 11,819 变为 11,418 字符。内部搜索量增加与可见文本变长是两件不同的事。

**按块 Metropolized 能换掉一段候选，但未形成稳定的“分块推导”文风。** 本次实现把相同 block 的候选权重合并后选择组，再在组内选续写；不会要求模型把论证写成更多章节，也不会把多个候选全文合并。与逐条 K=2 相比，按块 K=2 在 C 下有 45/100 道 LiveCodeBench、123/164 道 HumanEval+ 的思维链完全相同；SC 下分别为 37/100、131/164。C 的两套题平均长度只差 +0.5%、+1.7%，SC 为 +3.0%、−1.1%，方向也并不一致。

按块方法有时确实保留了更丰富的局部探索。例如分配因子任务中，按块 K=2 + C 在 8333 字符内两次构造复杂度反例，同题逐条 K=2 + C 只有 2256 字符。这段增加是实质性排除方案，不应与无信息的重复混在一起；但结合大量相同文本，它只能说明按块选择可能改变路线，不能证明它普遍更会找反例。

按块 K=2 + C · lcb/3760（8,333 字符，[完整原文](STYLE_TRACES.md#case-737c671191e54701b73a)）：

> The real worst case: elements = [2, 2, 2, ..., 2] (10^5 times), groups contain only odd numbers up to 10^5. Then for each element, we iterate through 5*10^4 multiples, none of which are in `unassigned` (since they're all even and groups are odd). Total: 10^5 * 5*10^4 = 5*10^9. Still too slow.
> 
> I need a different approach.

**Base 与 Best-of-4 共享上述叙述习惯；Best-of-4 没有稳定地选出“更成熟的结论稿”。** Base 也会直接建模、列样例、回头修改规则；C 系列相对 Base 在 LiveCodeBench 上平均长约 4.8%—7.1%，但各自配对区间都跨零，不能据此断言所有 IS 都普遍更冗长。Best-of-4 + C 的均值比 Base 高 5.5%（LiveCodeBench）和 2.6%（HumanEval+），同样没有清楚的跨题方向。

Best-of-4 从四条完整候选中保留一条，它的输出也可以很简洁：前述子序列和最短路任务分别只有 2492、1985 字符；也可以很长：大小配对任务达到 13,024 字符，持续重复论证并在末尾尚未收束。HumanEval/145 中它还选中了重新解释负号规则的探索性文本，而不是直接接受初始解释。这些相反表现排除了“Best-of-4 只会保留整洁、单线、已收敛推理”的概括。

Best-of-4 + C · HumanEval/145（5,802 字符，[完整原文](STYLE_TRACES.md#case-381be7956f422da17538)）：

> Hmm, what if the sum of digits is computed differently for negative numbers? What if it's `sum(int(d) for d in str(n))` where for negative numbers, we treat the minus sign as making the first digit negative?
> 
>    - -1: -1
>    - -11: -1+1 = 0
>    - -12: -1+2 = 1
>    
>    Then:
>    - 1: 1
>    - 11: 2
>    - -1: -1
>    - -11: 0
>    - -12: 1
> 
>    Sorted by these values: -1 (-1), 0 (-11), 1 (1, -12), 2 (11)
>    Result: [-1, -11, 1, -12, 11] (with stable sort for ties at value 1: 1 at index 0, -12 at index 4)
>    
>    That matches! So the approach is: for negative numbers, the first digit is negated.

同样不能把候选池中的四次尝试理解为最终文本里的四条可见分支。SIR、Metropolized 和 Best-of-4 都可能把内部计算用于选择一份看起来非常直接的答案；一个长链也可能只是同一路线反复展开。评价 IS 的特性，需要同时看内部选择记录和终选文本，不能相互代替。

**跨方法最稳定的共同变化，来自题目的难度与歧义。** 在 LiveCodeBench 易题上，十二方法的平均思维链都在 922—1031 字符附近；到难题则都在 10,474—12,004 字符附近。这个数量级的变化远大于同组 C/SC 的平均差异。短题常沿“规则—实现步骤—少量例子”结束；长题则共同出现状态重设、反复验证、性能讨论，或在同一障碍上循环。固定样本的 7 道难题中，C 和 SC 都有 6 道思维链停在未写完的推导或代码处。

本次能支持的算法级画像应当保持这个尺度：**C/SC 主要区分展开幅度与部分实现讨论；SIR、逐条、按块、K 的变化主要区分候选选择，尚不足以区分稳定文风；题目条件决定了大部分共同的叙述结构。** 这些结论仍需跨种子复现。还应区分表达形式与推理效果：“写得更长”“出现修正”“口气更确定”都不能直接代替正确性、验证有效性或解题能力。

原有十五组详细对照与 42 段思维链摘录保留在 [逐题复核附录](CASE_COMPARISONS.md)，其完整原文在 [原案例全文](SELECTED_TRACES.md)；成绩、成本与判题口径保存在 [结果附录](RESULTS.md)。本文引用及固定样本的全文见 [风格复核原文](STYLE_TRACES.md)，量化数据见 [style-metrics.json](style-metrics.json)，语义计数见 [style-summary.json](style-summary.json)，逐项编码见 [style-review-labels.jsonl](style-review-labels.jsonl)。所有引文的来源与字符偏移可在 [style-evidence.json](style-evidence.json) 核对。
