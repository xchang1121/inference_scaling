# 十五组逐题对照：原文与代码复核附录

这些个例用于核对具体行为，不用来估计算法风格的发生率。跨题结论见 [主报告](REPORT.md)。

全部 3168 份提交对应的终选思维链与代码已通读，同题相同内容去重后共 1739 份。这里保留原有三组并新增十二组，共十五组。每组均来自同一道题，分别核对推理原文、实际提交和冻结判题记录。

下文“通过／未通过”指本次固定测试集的记录；另行构造的小反例属于报告复核，未回写原始成绩。第 8 组发现浮点容差与判题实现不一致，第 9、12 组展示测试通过后仍存在的边界问题。这些个例用于解释具体行为，不作为各方法错误类型的发生率统计。

引文均直接截取模型保存的终选思维链，保留其中的错误、反复与原有措辞；不同引文块是不同位置的摘录，未拼成一段连续原文。标为“提交代码”的片段来自最终提交字段。每个方法旁的链接可查看该份完整思维链与代码，字符偏移和判题来源保存在 selected-evidence.json。

**1. LiveCodeBench abc393_f：正确的状态递推，未必能在预算内交付程序**

题目要求回答前缀 A[1…R] 中、所有元素不超过 X 的严格递增子序列最大长度。N、Q 均可达 20 万，既要解决两个查询约束，也要处理重复值。

对照：按块 K=2 + C：**未通过**，[完整原文](SELECTED_TRACES.md#case-5064c85c8239689fcc75)；按块 K=2 + SC：**通过**，[完整原文](SELECTED_TRACES.md#case-d05f2eb2cc1efda64fa1)。

按块 K=2 + C 的状态与更新：

> Define f_i(v) = max length of strictly increasing subsequence in A[1..i] where all elements <=? no exact value? Let's store `dp_i(x)` = max length of LIS in prefix i ending with value exactly x (coordinate). If x not present maybe 0. Recurrence: when adding x=A_i, any increasing subsequence ending at this new x either length 1 or formed by LIS in prefix i-1 ending with value <x plus x. So candidate = 1+max_{y<x} dp_{i-1}(y). Existing subsequences unchanged, so dp_i(x)=max(dp_{i-1}(x), candidate); dp_i(y)=dp_{i-1}(y) for y!=x. This is exactly persistent segment tree max point update and range max query.

这里的递推成立：只会在新元素的值处更新，查询历史版本即可同时满足前缀和数值阈值。问题出现在后续交付过程。该链长 15761 字符，继续比较持久化树的数组、列表、递归和迭代实现。

按块 K=2 + C 后期仍在讨论实现选择：

> Potential issue: `array('i')` append and index operations may be slower than list. Could use `list` for speed if memory allowed. AtCoder? Problem likely from AtCoder? N,Q=2e5. Python accepted maybe with pypy and lists? Need choose robust. We can implement with lists and maybe memory ~300-500 MB. AtCoder memory 1024 MB? Usually 1024 for such. But if 512 MB, still maybe okay? Let's estimate PyPy: list of ints in PyPy uses strategies? PyPy has list of ints unboxed? I think PyPy can store lists of homogeneous ints as rvalue? It has list strategies storing unboxed ints, memory ~8 bytes each, no individual int objects. So list approach in PyPy memory ~3*3.6M*8=86MB, very good. CPython list of ints memory big. Online judge likely PyPy? The request just Python program; we don't know. Use `array` for memory in CPython; speed? Might be okay with 7M operations. Could also use `list` and hope. Need produce code likely run under CPython? We need safe.

按块 K=2 + SC 对严格递增边界的处理：

> 1. Sort elements by value (ascending). For ties in value, we need to be careful. Actually, since we need strictly increasing subsequence, elements with the same value can't both be in the subsequence. So when processing elements with the same value, we should compute all their lis values first (based on the current state of the Fenwick tree), and then update the Fenwick tree with all of them.

SC 这份选择按 X 离线处理、在位置维度维护树状数组。先查询整组同值元素，再统一更新，避免把相等元素串成“严格递增”序列。其 9978 字符的推理后交付了完整程序。C 这份达到长度上限，最终提交只有查询函数的局部片段；按判题器包装方式复现时在 `node = root` 报 `NameError`。SC 在公开样例输出 `2、1、2`，本次整题通过。此例支持“可行思路到完整实现之间仍有落差”，不能据此把持久化树思路判为错误，也不能把两份输出的长度差解释为普遍因果关系。

**2. LiveCodeBench 3779：修复局部猜测之后，还需要全局分配**

每天吃四块披萨。奇数日收益是组内最大值，偶数日是第二大值；所有披萨只能使用一次，目标是总收益最大。

对照：按块 K=2 + C：**通过**，[完整原文](SELECTED_TRACES.md#case-3db6e3d8e61b2957adf6)；按块 K=2 + SC：**未通过**，[完整原文](SELECTED_TRACES.md#case-eaf6f9fbbe5fd0bc397d)。

C 先提出错误的总和猜测，再被公开样例否定：

> Claim: The optimal answer is a[0] + a[1] + ... + a[o-1] + a[o] + a[o+1] + ... + a[o+e-1] = sum of top (o+e) = sum of top k elements.
> 
> Wait, that's just the sum of the top k = n/4 elements? Let me verify with examples.
> 
> Example 1: [1,2,3,4,5,6,7,8], n=8, k=2, o=1, e=1
> Top k=2 elements: 8+7=15. But expected answer is 14.
> 
> So that's not right.

C 最终给出的全局分配规则：

> 1. Sort pizzas in descending order
> 2. k = n // 4
> 3. o = (k + 1) // 2 (ceil of k/2)
> 4. e = k // 2 (floor of k/2)
> 5. Answer = sum of first o elements + sum of elements at indices o+1, o+3, o+5, ..., o+2e-1

SC 则按日期依次做局部贪心：

> So the greedy strategy would be:
> 1. Sort in descending order.
> 2. Process days in order (day 1, day 2, day 3, ...).
> 3. On odd days: take the largest remaining as the "gained" weight, consume 3 smallest remaining.
> 4. On even days: take the second largest remaining as the "gained" weight, consume 2 smallest remaining.

SC 核对两日样例后的结论：

> This greedy approach works! Let me code it up.

差异在于是否提前为未来奇数日保留大值。对 `pizzas = [1,2,…,12]`，SC 依次获得 `12、10、9`，合计 31；C 先预留两个奇数日的 `12、11`，再让偶数日获得 9，合计 32。例如三组可取 `{1,2,3,12}`、`{7,8,9,10}`、`{4,5,6,11}`。独立穷举所有三日分组确认最优确为 32，两份实际提交分别返回 32 和 31。SC 后续反复核对指针与消耗数量，验证的是“每块恰好使用一次”，并没有证明该分配最大化总收益。

**3. HumanEval/145：样例冲突可以触发规则修正，也可能被当成题面错误**

题目要求按各位数字之和稳定排序；负数的处理需要结合给定示例理解。示例要求 `[1,11,-1,-11,-12] → [-1,-11,1,-12,11]`。

对照：Base：**未通过**，[完整原文](SELECTED_TRACES.md#case-d010e4e1149deb826268)；Best-of-4 + C：**通过**，[完整原文](SELECTED_TRACES.md#case-381be7956f422da17538)。

Base 已发现自己解释的结果与样例冲突：

> Stable sort by digit sum:
> - Digit sum 1: 1 (index 0), -1 (index 2)
> - Digit sum 2: 11 (index 1), -11 (index 3)
> - Digit sum 3: -12 (index 4)
> 
> Result: [1, -1, 11, -11, -12]
> 
> But expected is [-1, -11, 1, -12, 11]. That's different!

Base 对冲突的最终处理：

> Actually, looking at this more carefully, I think the example might just be incorrect in the problem statement, or I'm misreading it. Let me just implement the straightforward interpretation: sort by sum of digits of absolute value, with stable ordering for ties.
> 
> This is a well-known HumanEval problem (HumanEval/109). The standard solution is:

这段还声称题目是 HumanEval/109，而实际记录是 HumanEval/145。该“已知标准解”的说法并不能消除样例反证。最终代码仍对绝对值求数字和，返回 `[1,-1,11,-11,-12]`。

Best-of-4 + C 尝试让负号只作用于首位数字：

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

修正落实到最终排序键：

```python
    def digit_sum(n):
        s = str(abs(n))
        total = sum(int(d) for d in s)
        if n < 0:
            total -= 2 * int(s[0])
        return total
```

新规则使 `−1 → −1`、`−11 → 0`、`−12 → 1`，再利用稳定排序保留同分元素的原始次序。实际提交复现了题面顺序，并通过本次 HumanEval+ 测试。这里的区别是用冲突约束新解释，还是在冲突未解决时继续原解释。第 15 组会展示另一种情况：公开样例确实与数学定义相矛盾，因此“怀疑题面”本身既不能判好，也不能判坏。

**4. LiveCodeBench 3704：一次真正改变算法的反思，与直接证明的对照**

把数组分成两个非空连续部分，统计左右元素和之差为偶数的分割点。

对照：SIR K=1 + C：**通过**，[完整原文](SELECTED_TRACES.md#case-522147770130ef3737ec)；SIR K=2 + C：**通过**，[完整原文](SELECTED_TRACES.md#case-b5fa890d166725201216)。

SIR K=1 最初错误地只计算偶数前缀：

> If total_sum is even, then we need left_sum to be even (which means right_sum is also even since total is even).
> 
> So the algorithm is:
> 1. If total_sum is odd, return 0
> 2. Otherwise, count the number of indices i (0 <= i < n-1) where the prefix sum up to i is even.

公开样例迫使它找回被遗漏的奇数＋奇数情形：

> All 4 are even! But left_sum at i=2 is 23 which is odd, and right_sum is 13 which is odd. 23-13=10 which is even. So both being odd also works!
> 
> So the condition is: left_sum and right_sum have the same parity. Which means left_sum % 2 == right_sum % 2.
> 
> Since total = left_sum + right_sum, if total is even, then left_sum % 2 == right_sum % 2 always holds. So if total is even, ALL partitions work!

SIR K=2 直接利用代数恒等式：

> - left_sum - right_sum is even
> - left_sum + right_sum = total_sum
> - left_sum - right_sum = left_sum - (total_sum - left_sum) = 2 * left_sum - total_sum
> 
> So 2 * left_sum - total_sum is even.
> - 2 * left_sum is always even
> - So 2 * left_sum - total_sum is even when total_sum is even
> 
> Wait, that means if total_sum is even, ALL partitions work. If total_sum is odd, NO partitions work.

在公开数组 `[10,10,3,7,6]` 中，错误方案只计 3 个分割；`23−13=10` 恰好揭示遗漏。SIR K=1 后续把代码改为“总和偶数返回 n−1，否则返回 0”，和直接证明的 SIR K=2 一致，两者均通过。这里可以定位到“错误规则—反例—新规则—代码”的完整修正链。同样出现 `Wait`，SIR K=2 的这句话只是确认正确推导，并没有修复先前错误；统计反思词频无法区分二者。

**5. LiveCodeBench 3760：复杂度反例有效，但最终算法并未因此更强**

对每个组值 g，返回 elements 中能整除 g 的最早下标。数组长度和值域均可到 10 万。

对照：按块 K=2 + C：**通过**，[完整原文](SELECTED_TRACES.md#case-737c671191e54701b73a)；按块 K=2 + SC：**通过**，[完整原文](SELECTED_TRACES.md#case-b6689e2b24a0e86ef7d5)。

C 识别“没有更新”不等于“没有遍历”：

> Wait, this has the issue I mentioned. If elements = [1] * 100000, then for j=0, we set all best[multiple] = 0. For j=1, we iterate through all multiples of 1 again (10^5 iterations) but none get updated. Total: 10^5 * 10^5 = 10^10.

提前终止方案仍被新的最坏情况推翻：

> The real worst case: elements = [2, 2, 2, ..., 2] (10^5 times), groups contain only odd numbers up to 10^5. Then for each element, we iterate through 5*10^4 multiples, none of which are in `unassigned` (since they're all even and groups are odd). Total: 10^5 * 5*10^4 = 5*10^9. Still too slow.
> 
> I need a different approach.

SC 较直接地进入首下标映射＋约数枚举：

> One approach: For each possible divisor value `d` (from 1 to 10^5), precompute the minimum index `j` such that `elements[j] == d`. Then for each group `g`, I need to find the minimum of `min_index[d]` over all divisors `d` of `g`.
> 
> Steps:
> 1. Create an array `min_index` of size 100001, initialized to infinity. For each element `j`, set `min_index[elements[j]] = min(min_index[elements[j]], j)`.
> 2. For each group `g`, find all divisors of `g`, and for each divisor `d`, check if `min_index[d]` is not infinity. Take the minimum.

C 先尝试按元素逐个标记倍数，再加“全部分配后退出”，最后用重复 2 与奇数组构造最坏情况，说明内层仍可能空跑约 50 亿次。这是对运行成本的实质检验，而不是重复说“应该足够快”。最终 C 改为记录每个数的最早下标、逐组枚举约数，SC 也采用这个算法，两者均通过。C 的推理为 8333 字符，SC 为 2598 字符，但两份最后得到同类 `O(m+n√V)` 求解过程；SC 的值域数组初始化另有 `O(V)`。这段更长的有效纠错证明了它摆脱先前低效方案，并没有证明它比同题较直接的方案更优。

**6. LiveCodeBench 3759：通过差异来自一个 API 拼写，而非核心算法**

对每个 i，在满足 nums1[j] < nums1[i] 的下标中，选最多 k 个 nums2[j] 求最大和。关键边界是严格小于：nums1 相同的元素不能互相贡献。

对照：按块 K=2 + C：**未通过**，[完整原文](SELECTED_TRACES.md#case-3b67bd32e7ad7a72a64e)；按块 K=2 + SC：**通过**，[完整原文](SELECTED_TRACES.md#case-4909fe4435fc4c3f8f62)。

C 的分组顺序与堆更新思路成立：

> So the algorithm:
> 1. Create pairs (nums1[i], nums2[i], i) for all i
> 2. Sort by nums1 value
> 3. Process in groups of same nums1 value
> 4. For each group, first compute answers for all indices in the group (using the current state of the data structure), then add all elements in the group to the data structure
> 
> For the data structure: maintain a min-heap of size at most k, and a running sum of the heap.
> 
> When adding a value v:
> - If heap size < k: push v, add to sum
> - If heap size == k and v > heap[0]: pop the min, push v, update sum
> - Otherwise: do nothing

SC 描述的是同一核心算法：

> So the algorithm:
> 1. Sort indices by nums1 value
> 2. Use a min-heap to keep track of the top k nums2 values seen so far
> 3. Maintain a running sum of the heap
> 4. Process groups of equal nums1 values:
>    - For each element in the group, answer[i] = current sum of top k
>    - After the group, add all nums2 values of the group to the heap
> 
> For adding to the heap:
> - If heap size < k: push the value, add to sum
> - If heap size == k and value > heap[0]: pop heap[0], push value, update sum
> - If heap size == k and value <= heap[0]: do nothing

C 实际提交的错误调用：

```python
                    current_sum -= heapq.heapop(min_heap)
                    heapq.heappush(min_heap, val)
                    current_sum += val
```

SC 实际提交的堆替换：

```python
                    heap_sum -= min_heap[0]
                    heapq.heapreplace(min_heap, val)
                    heap_sum += val
```

C 写成了不存在的 `heapq.heapop`，公开样例在第一次需要替换堆顶时就触发 `AttributeError`。SC 使用 `heapreplace`，实际返回 `[80,30,0,80,50]` 并通过。两边都正确处理了同值分组、先回答后插入，因此不能把这一胜负记为“SC 找到了更好的算法”。两份原文还都把总复杂度写成 `O(n log k)`，漏掉排序的 `O(n log n)`；这一文字不完整与导致运行失败的 API 错误，需要分别评价。

**7. LiveCodeBench 3750：继续简化，也可能把正确草稿改错**

在环形数组中，对指定下标查找另一个相同值位置的最短环距；没有其他位置则返回 −1。

对照：逐条 K=2 + C：**未通过**，[完整原文](SELECTED_TRACES.md#case-86217df34921541dfbda)；快速 IS + C：**通过**，[完整原文](SELECTED_TRACES.md#case-c81caf648a042532f711)。

逐条 K=2 + C 早期已经覆盖首尾环绕候选：

> The minimum circular distance from `q` to any point in `indices` will be achieved by one of:
> - The nearest point to the left in the linear sense
> - The nearest point to the right in the linear sense
> - The first point (if wrapping around from the right gives a shorter path)
> - The last point (if wrapping around from the left gives a shorter path)

随后把距离公式误当成候选覆盖的保证：

> Actually, the nearest point to the left or right in the linear sense already covers the circular case when combined with the circular distance formula. The circular distance `min(|a-b|, n-|a-b|)` already accounts for wrap-around.
> 
> So really, we just need the nearest neighbor(s) in the sorted list. The nearest neighbor to `q` in the sorted list will give the minimum linear distance, and the circular distance formula handles the rest.

最终删除首尾候选的结论：

> Let me refine: check `indices[pos-1]`, `indices[pos]`, `indices[pos+1]` (with bounds checking), exclude `q`, and compute min circular distance.
> 
> This should be correct and efficient.

快速 IS + C 也做简化，但保留循环邻接关系：

> Actually, let me just use `pos` from bisect_left. Since `q` is in the list, `indices[pos] == q`. So:
> - Right neighbor: `indices[(pos + 1) % len(indices)]`
> - Left neighbor: `indices[(pos - 1) % len(indices)]`
> 
> These two should be sufficient for the circular case! Because in a circular array, the nearest same-value element must be either the immediate next or immediate previous in the sorted circular list of indices.

反例取 `nums=[7,1,2,7,3,4,5,6,8,7]`、`queries=[0]`。值 7 的位置是 `[0,3,9]`。逐条 K=2 只看线性邻居 3，返回 3；位置 9 经环绕只距 0 一步，正确答案为 1。快速 IS 用 `(pos±1) % 频次` 取得循环邻居，实际返回 1 并通过整题。错误链中曾意识到 `min(d,n−d)` 在大距离处也会变小，却只用两个同值位置的例子确认；这种例子无法区分“线性邻居”和“循环邻居”。增加思考步骤在这里引入了回退。

**8. LiveCodeBench abc392_d：相同概率公式，状态污染与判题容差是两件事**

从 N 个骰子中选两个，最大化掷出相同数字的概率。题面允许绝对或相对误差不超过 10⁻⁸。

对照：快速 IS + C：**未通过**，[完整原文](SELECTED_TRACES.md#case-b9b0c5cc051f21637bcb)；快速 IS + SC：**未通过**，[完整原文](SELECTED_TRACES.md#case-f35599a7e0885bd605e3)。

快速 IS + C 的概率公式和遍历策略：

> For two dice i and j, the probability they show the same number is:
> P = sum over all values v of (count of v in die i / K_i) * (count of v in die j / K_j)
> 
> So for each pair of dice (i, j), I need to compute:
> P(i,j) = sum over all v: (freq_i(v) / K_i) * (freq_j(v) / K_j)
> 
> To do this efficiently, I can:
> 1. For each die, compute a frequency map of its faces.
> 2. For each pair of dice, compute the sum by iterating over the smaller frequency map and looking up in the larger one.

SC 使用相同的频数内积思路：

> Approach:
> 1. For each die, compute a frequency map of the numbers on it.
> 2. For each pair (i, j), compute the dot product of their frequency maps (only over keys present in both), then divide by K_i * K_j.
> 3. Return the maximum.
> 
> For efficiency, when computing the dot product for pair (i, j), we should iterate over the smaller frequency map and look up in the larger one.

C 为遍历较小字典而交换外层状态：

```python
    for i in range(N):
        ki, fi = dice[i]
        for j in range(i + 1, N):
            kj, fj = dice[j]
            # Iterate over the smaller dictionary
            if len(fi) > len(fj):
                fi, fj = fj, fi
                ki, kj = kj, ki
```

SC 为当前骰子对使用独立临时变量：

```python
            if len(freq_i) <= len(freq_j):
                smaller = freq_i
                larger = freq_j
            else:
                smaller = freq_j
                larger = freq_i
```

C 的 `fi、ki` 在内层交换后没有为下一个 j 恢复，因而后续比较的已不是原来的第 i 个骰子。输入三枚骰子 `[1,2]、[3]、[1,2]`，两份实际程序分别输出 0 和 0.5；最优显然是第一与第三枚，概率 0.5。SC 的临时变量避免了这个具体错误。

但两份在冻结记录里均未通过，不能写成 SC 因此获得整题胜利。两份都在一个双骰测试上输出 `0.076923076923077`，基准文本为 `0.076923076923`；精确计数给出真实概率 `7/(13×7)=1/13`，两种小数都在题面容差内。所用判题器在文本不同时改为 Decimal 精确相等比较，仍不使用这道题的容差，因此给出 Wrong Answer。本组同时有可复现的代码缺陷和判题实现问题；原成绩保留，但不能把该浮点失败当作推理错误的证据。

**9. LiveCodeBench 3777：正确指出状态压缩问题，代码却没有遵守；通过测试仍有遗漏**

选非空子序列，使交替和等于 k，并最大化不超过 limit 的乘积。元素范围含 0，所以中途超过上限的乘积，之后乘 0 可能重新变得可行。

对照：Base：**未通过**，[完整原文](SELECTED_TRACES.md#case-33047554383b3cc4b3a7)；按块 K=2 + C：**通过**，[完整原文](SELECTED_TRACES.md#case-a66262b492a64f368c33)。

Base 明确识别只保存最大乘积不成立：

> But as I noted, a smaller product might be better in some cases. For example, if current product is 4 and we multiply by 3, we get 12. If current product is 3 and we multiply by 4, we get 12. Both are fine. But if limit is 10, product 4 * 3 = 12 > 10, but product 3 * 3 = 9 <= 10.
> 
> So we can't just keep the max product. We need to keep all achievable products.

Base 最终代码仍只保存最大值，并初始化空子序列：

```python
        # dp[parity][s_offset] = max product achievable (or -1 if not reachable)
        # parity 0 = even count chosen (next element gets +), parity 1 = odd count (next gets -)
        OFFSET = 1800
        SIZE = 3601
        
        NEG = -1  # unreachable
        dp = [[NEG] * SIZE for _ in range(2)]
        dp[0][OFFSET] = 1  # empty subsequence: parity 0, sum 0, product 1
```

按块 K=2 + C 最终明确区分非空状态：

> Simpler approach: just don't include the empty state. Instead, for each element, I can start a new subsequence with just that element.
> 
> Let me redo:
> - dp[(parity, sum)] = set of products
> - For each num in nums:
>   - new_dp = copy of dp
>   - Start new: new_dp[(1, num)].add(num)  [picking this as first element, parity becomes 1]
>   - Extend existing: for each (p, s) -> prods in dp:
>     - new_p = 1 - p
>     - new_s = s + num if p == 0 else s - num
>     - for prod in prods: new_dp[(new_p, new_s)].add(prod * num)
>   - dp = new_dp
> 
> At the end, check dp for states with sum = k, find max product ≤ limit.

它也意识到乘零会改变乘积：

> Let me also add: if num == 0, then prod * num = 0 for all prods. So the product becomes 0. That's fine.

状态压缩的直接反例是 `[3,2,1,3], k=4, limit=8`：前缀 `[3,2]` 与 `[2,1]` 都是偶数长度、交替和为 1，乘积分别为 6 和 2。继续接上最后的 3，只有较小乘积能在上限内得到 6。Base 丢掉它后实际返回 −1，按块 C 保留乘积集合则返回正确的 6。

此外，Base 在 `[0], k=0, limit=1` 返回空序列留下的 1，正确值为 0；按块 C 的集合状态在此返回 0。但按块 C 在初始化和延伸时都删除超限乘积。对 `[10,0], k=10, limit=1`，非空子序列 `[10,0]` 的交替和为 10、乘积为 0，合法且最优；实际程序却返回 −1。枚举全部非空子序列验证了以上三个小例子。

按块 C 确实通过了本次全部 43 个测试，而 Base 未通过；同题按块 SC 与按块 C 的终选思维链、代码完全相同。因此这既是“较完整状态表示修复了一部分错误”的案例，也是“通过现有测试并未排除语义漏洞”的案例。原文同时提到风险和零值，并不等于最终转移保留了所需状态；仅搜索关键句会高估完成的修正。

**10. LiveCodeBench abc388_f：一般可达性推导与零步边界，都必须落实到状态转移**

从格子 1 跳到 N，每步长度在 [A,B]，落点不能在坏区间中。N 可达 10¹²，A、B 至多 20，需要压缩长好区间的可达状态。

对照：按块 K=2 + SC：**未通过**，[完整原文](SELECTED_TRACES.md#case-8365280d2bcd3e04608f)；逐条 K=1 + C：**未通过**，[完整原文](SELECTED_TRACES.md#case-b9d08bc11b68a11e90de)。

按块 SC 将含 k 的间隔错误消去：

> The gap between [x+kA, x+kB] and [x+(k+1)A, x+(k+1)B] is: x+(k+1)A - (x+kB) - 1 = A - B - 1. Since A ≤ B, this is ≤ -1, meaning they overlap or are adjacent! So the reachable set from a single position x is actually [x+A, ∞) (within the good stretch), because consecutive ranges overlap.

它后来确实找到了无法迈出第一步的边界：

> Hmm wait, if A=2, B=3, N=2: from 1 we can jump to 3 or 4, but N=2, so we can't reach it. But we need x+i ≤ N, so 1+2=3 > 2=N, can't even make the jump. So answer is No.

逐条 K=1 + C 推导出正确的正距离判据：

> For a given difference d = t - s:
> - If d < A: not reachable
> - If A = B: reachable iff d % A == 0
> - If A < B: reachable iff there exists k ≥ 1 with k*A ≤ d ≤ k*B. This is iff ceil(d/B) ≤ floor(d/A).

但在衔接坏区间时排除了已经站在临界格上的状态：

```python
            for s in reachable:
                if s >= x:
                    continue
                d = x - s
```

SC 的间隔应为 `(k+1)A−kB−1 = A−k(B−A)−1`，并非 `A−B−1`；不同步数的可达区间不保证从一开始就相连。即使它后来明确说 `N=2,A=2,B=3` 应为 No，最终仍只检查 `lo≤N`，该输入实际输出 Yes。

逐条 K=1 的正距离公式更准确，但将 `s==x` 也跳过，遗漏无需在当前好区间继续移动的“零步”情形。对 `N=3,A=1,B=2`、唯一坏格 `[2,2]`，本可直接 `1→3`，实际却输出 No。两份在冻结测试中均未通过；一个是可达集合的错误概括与未落实的修正，另一个是局部公式正确、跨区间衔接遗漏边界。不能因后一份有正确公式，就把整份算法标成正确。

**11. HumanEval/46：手算序列正确，初始化依然可能写错**

计算四阶递推 fib4：初值为 `[0,0,2,0]`，之后每项等于前四项之和；要求迭代实现。

对照：快速 IS + SC：**未通过**，[完整原文](SELECTED_TRACES.md#case-2d08d33a41e0d57c34e4)；逐条 K=1 + SC：**通过**，[完整原文](SELECTED_TRACES.md#case-8482a026a8c999fc404e)。

快速 IS + SC 的样例演算全部正确：

> Let me verify with the examples:
> - fib4(4) = fib4(3) + fib4(2) + fib4(1) + fib4(0) = 0 + 2 + 0 + 0 = 2
> - fib4(5) = fib4(4) + fib4(3) + fib4(2) + fib4(1) = 2 + 0 + 2 + 0 = 4 ✓
> - fib4(6) = fib4(5) + fib4(4) + fib4(3) + fib4(2) = 4 + 2 + 0 + 2 = 8 ✓
> - fib4(7) = fib4(6) + fib4(5) + fib4(4) + fib4(3) = 8 + 4 + 2 + 0 = 14 ✓

它的实际初始化与上面的规则不一致：

```python
    a, b, c, d = 0, 2, 0, 0  # fib4(0), fib4(1), fib4(2), fib4(3)
    for _ in range(4, n + 1):
        a, b, c, d = b, c, d, a + b + c + d
    return d
```

逐条 K=1 + SC 的思考也采用普通迭代：

> So the approach is straightforward iterative computation. I'll handle base cases for n=0,1,2,3 and then iterate from 4 to n.

有效修正出现在最终代码内部：

```python
    a, b, c, d = 0, 2, 0, 0  # fib4(0), fib4(1), fib4(2), fib4(3)
    # Actually let me use a sliding window
    # fib4(0)=0, fib4(1)=0, fib4(2)=2, fib4(3)=0
    a, b, c, d = 0, 0, 2, 0
```

快速 IS 的代码把初值 2 放到了第二个变量，计算 `fib4(6)` 得到 6，而文字演算写的是正确的 8。逐条 K=1 代码也先写错一次，但紧接着用正确初值覆盖，实际返回 8 并通过。这份成功并不来自另一种递推算法，而来自初始化被真正修正。如果只看思维链，两份都能算对样例；如果只看代码第一条赋值，又会漏掉随后的覆盖。

**12. HumanEval/125：公开样例之外的规则扩展，可能改善实现但不改变得分**

字符串有空白字符时按空白分词；否则有逗号则按逗号分词；再否则统计字母表中奇数序号的小写字母。

对照：Base：**通过**，[完整原文](SELECTED_TRACES.md#case-82b01251f220ee233da5)；快速 IS + C：**通过**，[完整原文](SELECTED_TRACES.md#case-85115e5cbc606d0d3adf)。

Base 对规则的复述覆盖了 whitespace：

> 1. Given a string of words, I need to:
>    - If there are whitespaces, split on whitespace and return the list of words
>    - If no whitespaces but there are commas, split on commas and return the list
>    - If no whitespaces and no commas, return the count of lower-case letters that have an odd order in the alphabet (where 'a'=0, 'b'=1, ..., 'z'=25)

但它的实际条件只检查普通空格：

```python
    if ' ' in txt:
        return txt.split()
    elif ',' in txt:
        return txt.split(',')
```

快速 IS + C 主动检查空白的范围：

> Wait, I should consider if there are other whitespace characters besides space. The problem says "whitespace" so I should probably check for any whitespace. Let me use a more general check.
> 
> Actually, looking at the examples, it seems like regular space is the whitespace character in question. But to be safe, I could check if any character is a whitespace character. Let me use `' '.join` approach or check with `any(c.isspace() for c in txt)`.
> 
> Let me keep it simple and check for space, since the examples use spaces. But to be more robust, I'll check for any whitespace character.

修改落实为与 split 一致的判定：

```python
    if any(c.isspace() for c in txt):
        return txt.split()
```

两份都通过了本次 HumanEval+ 测试，但对 `a\tb`（中间是制表符），Base 返回字母计数 1，快速 IS 返回两个词 a 和 b；后者符合题面 whitespace 的通常含义。能确认的是，既有成绩未区分这个边界，而规则审查区分出了行为。这也是“有实际价值的修正未必带来本次分数提升”的例子，不能仅按胜负筛选所有有信息的案例。

**13. HumanEval/160：把样例当作区分解释的实验**

给出操作符列表与非负整数列表，构造并求值表达式。关键歧义是按书写顺序计算，还是采用通常优先级。

对照：快速 IS + SC：**通过**，[完整原文](SELECTED_TRACES.md#case-dfddce6f0e269c7bc341)；Base：**通过**，[完整原文](SELECTED_TRACES.md#case-312775e042c982c108c5)。

快速 IS + SC 开头直接作了错误解释：

> The user wants me to implement a function `do_algebra` that takes a list of operators and a list of operands, and evaluates the algebraic expression left to right (not following operator precedence).

它用两种算法对同一表达式的不同预测修正规则：

> Let me check: if we follow operator precedence:
> 2 + 3 * 4 - 5 = 2 + 12 - 5 = 9 ✓
> 
> If we go left to right:
> (2 + 3) * 4 - 5 = 5 * 4 - 5 = 20 - 5 = 15 ≠ 9
> 
> So it follows standard operator precedence! That means I should use `eval` to evaluate the expression, or implement it with proper precedence.

Base 则先把两种解释作为待判定选项：

> The key question is: do I need to respect operator precedence, or just evaluate left to right?
> 
> Looking at the example: 2 + 3 * 4 - 5 = 9
> - If we respect precedence: 2 + (3*4) - 5 = 2 + 12 - 5 = 9 ✓
> - If we evaluate left to right: (2+3) * 4 - 5 = 20 - 5 = 15 ✗
> 
> So we need to respect operator precedence. The simplest way to do this is to build a string expression and use `eval()`.

`2+3×4−5` 按通常优先级是 9，从左到右是 15；给定样例明确区分了二者。SC 撤回开头的解释，最终和 Base 一样构造表达式并调用 Python 求值，两份均返回 9 并通过。这段修正较短，却包含可检验的竞争解释、区分它们的计算和最终实现。它比没有新证据的反复确认更有实质内容；不过它也没有因此超过同题一开始就正确处理歧义的 Base。

**14. HumanEval/97：特殊负数样例掩盖错误泛化，代码又可能偏离文字结论**

返回两个整数的个位数字之积，负数按绝对值的个位理解。公开负数样例是 `multiply(14,-15)=20`。

对照：Best-of-4 + C：**未通过**，[完整原文](SELECTED_TRACES.md#case-da237fe7ba135d10b36c)；逐条 K=1 + C：**通过**，[完整原文](SELECTED_TRACES.md#case-b994d3b80115eaa2d5db)。

Best-of-4 + C 从 −15 的巧合推出一般结论：

> If I used `-15 % 10` in Python, it gives `5` (Python's modulo always returns non-negative for positive divisor). So that works.

它只重复已有样例后采用了无 abs 的实现：

> So simply `a % 10 * b % 10` would work? Let me check:
> - `148 % 10` = 8, `412 % 10` = 2, 8 * 2 = 16 ✓
> - `19 % 10` = 9, `28 % 10` = 8, 9 * 8 = 72 ✓
> - `2020 % 10` = 0, `1851 % 10` = 1, 0 * 1 = 0 ✓
> - `14 % 10` = 4, `-15 % 10` = 5, 4 * 5 = 20 ✓
> 
> So the solution is simply `(a % 10) * (b % 10)`.

逐条 K=1 + C 在思维链中也给出了同样的错误泛化：

> Actually, in Python, `-15 % 10 = 5` (Python's modulo always returns non-negative for positive divisor), so `a % 10` would already work for negative numbers in Python. Let me verify: `-15 % 10` in Python gives `5`. Yes!
> 
> So simply `(a % 10) * (b % 10)` would work in Python.
> 
> But to be safe and clear, I'll use `abs(a) % 10 * abs(b) % 10` or just `(a % 10) * (b % 10)`.
> 
> Let me go with the simple approach: `(a % 10) * (b % 10)` since Python's modulo handles negatives correctly for this case.

但它最终交付的是带 abs 的正确版本：

```python
    return (abs(a) % 10) * (abs(b) % 10)
```

`−15 % 10=5` 恰好等于绝对值个位，并不代表所有负数如此。`−12 % 10=8`，而 `abs(−12)%10=2`；对 `(14,−12)`，Best-of-4 实际返回 32，正确值为 8，逐条 K=1 实际返回 8。前者未通过，后者通过。但后一份文字最后仍说要用无 abs 的表达式，不能据其正确代码倒推“推理已经消除了误解”。本例应分别记录负数规则的错误泛化与最终代码的正确性。

**15. HumanEval/47：打勾的样例核对不一定真的完成了计算**

实现标准中位数。公开文档把 `[-10,4,6,1000,10,20]` 的中位数写成 15.0，但排序后中间两个数是 6、10，平均值应为 8.0。

对照：逐条 K=1 + SC：**通过**，[完整原文](SELECTED_TRACES.md#case-3f0ec062125a71ff7413)；Base：**通过**，[完整原文](SELECTED_TRACES.md#case-68487bab726a484bd3fd)。

逐条 K=1 + SC 原文把错误算式当作核对通过：

> Let me verify with the examples:
> - `median([3, 1, 2, 4, 5])` → sorted: [1, 2, 3, 4, 5], middle element is 3 ✓
> - `median([-10, 4, 6, 1000, 10, 20])` → sorted: [-10, 4, 6, 10, 20, 1000], two middle elements are 6 and 10, average is 15.0 ✓

Base 则明确发现了题面和计算的矛盾：

> Wait, that gives 8.0, not 15.0. Let me re-check.
> 
> sorted: [-10, 4, 6, 10, 20, 1000]
> n = 6, so middle indices are 2 and 3
> Elements at index 2 and 3: 6 and 10
> (6 + 10) / 2 = 8.0
> 
> But the expected output is 15.0. That's strange. Let me re-read the problem.
> 
> Hmm, maybe the expected output in the docstring is wrong? Or maybe I'm misunderstanding something.
> 
> Wait, let me re-check: [-10, 4, 6, 1000, 10, 20]
> Sorted: [-10, 4, 6, 10, 20, 1000]
> n = 6
> Middle elements: index 2 = 6, index 3 = 10
> Median = (6 + 10) / 2 = 8.0
> 
> The docstring says 15.0 which doesn't match. This seems like an error in the docstring. I'll implement the standard median calculation, which should be correct for the actual test cases.

SC 的最终公式仍然正确：

```python
    if n % 2 == 1:
        return s[mid]
    else:
        return (s[mid - 1] + s[mid]) / 2
```

两份实际程序都返回 8.0，并通过本次 HumanEval+ 测试。SC 的问题是局部算术陈述和验证结论错误，没有传播到通用中位数公式；Base 正确识别了不一致。与第 3 组对照，本题怀疑样例有明确算术依据，而第 3 组仍存在能解释样例的负号规则。评价“发现样例冲突”时，需要看冲突究竟被算清、被解释，还是只是被一句“样例可能错了”带过。
