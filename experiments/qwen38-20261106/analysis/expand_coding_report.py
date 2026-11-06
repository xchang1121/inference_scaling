"""Build fifteen detailed comparisons from immutable experiment traces."""
from __future__ import annotations

import hashlib
import json

import coding_study as cs
from validate_expanded_examples import validate

manifest, problems, records, servers, attempts = cs.load()
cases = {r['case_id']: r for r in cs.read_jsonl(cs.OUT / 'review-cases.jsonl')}
key = json.loads((cs.OUT / 'review-key.json').read_text(encoding='utf-8'))
labels = cs.labels()
assert len(records) == 3168 and len(cases) == len(labels) == 1739 and not attempts
assert set(cases) == set(labels)
sections, selected, evidence = [], [], []
lines = [
    '全部 3168 份提交对应的终选思维链与代码已通读，同题相同内容去重后共 1739 份。'
    '这里保留原有三组并新增十二组，共十五组。每组均来自同一道题，分别核对推理原文、实际提交和冻结判题记录。', '',
    '下文“通过／未通过”指本次固定测试集的记录；另行构造的小反例属于报告复核，未回写原始成绩。'
    '第 8 组发现浮点容差与判题实现不一致，第 9、12 组展示测试通过后仍存在的边界问题。'
    '这些个例用于解释具体行为，不作为各方法错误类型的发生率统计。', '',
    '引文均直接截取模型保存的终选思维链，保留其中的错误、反复与原有措辞；'
    '不同引文块是不同位置的摘录，未拼成一段连续原文。标为“提交代码”的片段来自最终提交字段。'
    '每个方法旁的链接可查看该份完整思维链与代码，字符偏移和判题来源保存在 selected-evidence.json。', '',
]


def paragraph(text):
    lines.extend([text, ''])


def section(title, question, pair):
    number = len(sections) + 1
    assert len({cases[cid]['problem_id'] for cid, _ in pair}) == 1
    entry = {'number': number, 'title': title, 'problem_id': cases[pair[0][0]]['problem_id'], 'cases': []}
    sections.append(entry)
    paragraph(f'**{number}. {title}**')
    paragraph(question)
    captions = []
    for cid, mode in pair:
        case = cases[cid]
        row = records[case['problem_id'], mode]
        assert row['thinking_text'] == case['thinking_text'] and row['submitted_code'] == case['submitted_code']
        assert cid in labels
        selected.append(cid)
        item = {
            'section': number, 'case_id': cid, 'problem_id': case['problem_id'], 'mode': mode,
            'matching_methods': [r['mode'] for r in key[cid]],
            'correct': row['judgment']['correct'], 'thinking_characters': len(case['thinking_text']),
            'seconds': row['seconds'], 'estimated_flops': cs.native_flops(row),
            'length_exhausted': row['length_exhausted'], 'submission_sha256': row['submission_sha256'],
            'thinking_sha256': hashlib.sha256(case['thinking_text'].encode()).hexdigest(),
            'source_replica': row['replica'], 'source_line': row['source_line'], 'excerpts': [],
        }
        evidence.append(item)
        entry['cases'].append(cid)
        result = '通过' if item['correct'] else '未通过'
        captions.append(f'{cs.NAMES[mode]}：**{result}**，[完整原文](SELECTED_TRACES.md#case-{cid})')
    paragraph('对照：' + '；'.join(captions) + '。')


def excerpt(cid, label, begin, end=None, *, field='thinking_text'):
    source = cases[cid][field]
    start = source.index(begin)
    stop = start + len(begin) if end is None else source.index(end, start) + len(end)
    quote = source[start:stop]
    assert quote and len(quote) < 2300
    item = next(x for x in evidence if x['case_id'] == cid)
    item['excerpts'].append({'label': label, 'field': field, 'start': start, 'end': stop, 'quote': quote})
    paragraph(label + '：')
    if field == 'submitted_code':
        lines.extend(['```python', quote, '```', ''])
    else:
        lines.extend(['\n'.join('> ' + s for s in quote.split('\n')), ''])


# The original three comparisons are expanded as well.
a, b = '5064c85c8239689fcc75', 'd05f2eb2cc1efda64fa1'
section('LiveCodeBench abc393_f：正确的状态递推，未必能在预算内交付程序',
        '题目要求回答前缀 A[1…R] 中、所有元素不超过 X 的严格递增子序列最大长度。'
        'N、Q 均可达 20 万，既要解决两个查询约束，也要处理重复值。',
        [(a, 'block-met-k2'), (b, 'block-met-k2-self-certainty')])
excerpt(a, '按块 K=2 + C 的状态与更新', 'Define f_i(v)',
        'This is exactly persistent segment tree max point update and range max query.')
paragraph('这里的递推成立：只会在新元素的值处更新，查询历史版本即可同时满足前缀和数值阈值。'
          '问题出现在后续交付过程。该链长 15761 字符，继续比较持久化树的数组、列表、递归和迭代实现。')
excerpt(a, '按块 K=2 + C 后期仍在讨论实现选择',
        "Potential issue: `array('i')` append and index operations may be slower than list.",
        'Need produce code likely run under CPython? We need safe.')
excerpt(b, '按块 K=2 + SC 对严格递增边界的处理',
        '1. Sort elements by value (ascending).', 'and then update the Fenwick tree with all of them.')
paragraph('SC 这份选择按 X 离线处理、在位置维度维护树状数组。先查询整组同值元素，再统一更新，'
          '避免把相等元素串成“严格递增”序列。其 9978 字符的推理后交付了完整程序。'
          'C 这份达到长度上限，最终提交只有查询函数的局部片段；按判题器包装方式复现时在 `node = root` 报 `NameError`。'
          'SC 在公开样例输出 `2、1、2`，本次整题通过。此例支持“可行思路到完整实现之间仍有落差”，'
          '不能据此把持久化树思路判为错误，也不能把两份输出的长度差解释为普遍因果关系。')

a, b = '3db6e3d8e61b2957adf6', 'eaf6f9fbbe5fd0bc397d'
section('LiveCodeBench 3779：修复局部猜测之后，还需要全局分配',
        '每天吃四块披萨。奇数日收益是组内最大值，偶数日是第二大值；所有披萨只能使用一次，目标是总收益最大。',
        [(a, 'block-met-k2'), (b, 'block-met-k2-self-certainty')])
excerpt(a, 'C 先提出错误的总和猜测，再被公开样例否定', 'Claim: The optimal answer is', "So that's not right.")
excerpt(a, 'C 最终给出的全局分配规则', '1. Sort pizzas in descending order\n2. k = n // 4',
        '5. Answer = sum of first o elements + sum of elements at indices o+1, o+3, o+5, ..., o+2e-1')
excerpt(b, 'SC 则按日期依次做局部贪心', 'So the greedy strategy would be:', 'consume 2 smallest remaining.')
excerpt(b, 'SC 核对两日样例后的结论', 'This greedy approach works! Let me code it up.')
paragraph('差异在于是否提前为未来奇数日保留大值。对 `pizzas = [1,2,…,12]`，SC 依次获得 `12、10、9`，'
          '合计 31；C 先预留两个奇数日的 `12、11`，再让偶数日获得 9，合计 32。'
          '例如三组可取 `{1,2,3,12}`、`{7,8,9,10}`、`{4,5,6,11}`。'
          '独立穷举所有三日分组确认最优确为 32，两份实际提交分别返回 32 和 31。'
          'SC 后续反复核对指针与消耗数量，验证的是“每块恰好使用一次”，并没有证明该分配最大化总收益。')

a, b = 'd010e4e1149deb826268', '381be7956f422da17538'
section('HumanEval/145：样例冲突可以触发规则修正，也可能被当成题面错误',
        '题目要求按各位数字之和稳定排序；负数的处理需要结合给定示例理解。'
        '示例要求 `[1,11,-1,-11,-12] → [-1,-11,1,-12,11]`。',
        [(a, 'base'), (b, 'bon-consilience')])
excerpt(a, 'Base 已发现自己解释的结果与样例冲突', 'Stable sort by digit sum:',
        "But expected is [-1, -11, 1, -12, 11]. That's different!")
excerpt(a, 'Base 对冲突的最终处理',
        'Actually, looking at this more carefully, I think the example might just be incorrect',
        'This is a well-known HumanEval problem (HumanEval/109). The standard solution is:')
paragraph('这段还声称题目是 HumanEval/109，而实际记录是 HumanEval/145。该“已知标准解”的说法并不能消除样例反证。'
          '最终代码仍对绝对值求数字和，返回 `[1,-1,11,-11,-12]`。')
excerpt(b, 'Best-of-4 + C 尝试让负号只作用于首位数字',
        'Hmm, what if the sum of digits is computed differently for negative numbers?',
        'That matches! So the approach is: for negative numbers, the first digit is negated.')
excerpt(b, '修正落实到最终排序键', '    def digit_sum(n):', '        return total', field='submitted_code')
paragraph('新规则使 `−1 → −1`、`−11 → 0`、`−12 → 1`，再利用稳定排序保留同分元素的原始次序。'
          '实际提交复现了题面顺序，并通过本次 HumanEval+ 测试。这里的区别是用冲突约束新解释，'
          '还是在冲突未解决时继续原解释。第 15 组会展示另一种情况：公开样例确实与数学定义相矛盾，'
          '因此“怀疑题面”本身既不能判好，也不能判坏。')

a, b = '522147770130ef3737ec', 'b5fa890d166725201216'
section('LiveCodeBench 3704：一次真正改变算法的反思，与直接证明的对照',
        '把数组分成两个非空连续部分，统计左右元素和之差为偶数的分割点。',
        [(a, 'sir-k1'), (b, 'sir-k2')])
excerpt(a, 'SIR K=1 最初错误地只计算偶数前缀',
        'If total_sum is even, then we need left_sum to be even', 'where the prefix sum up to i is even.')
excerpt(a, '公开样例迫使它找回被遗漏的奇数＋奇数情形',
        'All 4 are even! But left_sum at i=2', 'So if total is even, ALL partitions work!')
excerpt(b, 'SIR K=2 直接利用代数恒等式', '- left_sum - right_sum is even',
        'Wait, that means if total_sum is even, ALL partitions work. If total_sum is odd, NO partitions work.')
paragraph('在公开数组 `[10,10,3,7,6]` 中，错误方案只计 3 个分割；`23−13=10` 恰好揭示遗漏。'
          'SIR K=1 后续把代码改为“总和偶数返回 n−1，否则返回 0”，和直接证明的 SIR K=2 一致，两者均通过。'
          '这里可以定位到“错误规则—反例—新规则—代码”的完整修正链。'
          '同样出现 `Wait`，SIR K=2 的这句话只是确认正确推导，并没有修复先前错误；统计反思词频无法区分二者。')

a, b = '737c671191e54701b73a', 'b6689e2b24a0e86ef7d5'
section('LiveCodeBench 3760：复杂度反例有效，但最终算法并未因此更强',
        '对每个组值 g，返回 elements 中能整除 g 的最早下标。数组长度和值域均可到 10 万。',
        [(a, 'block-met-k2'), (b, 'block-met-k2-self-certainty')])
excerpt(a, 'C 识别“没有更新”不等于“没有遍历”',
        'Wait, this has the issue I mentioned. If elements = [1] * 100000,', 'Total: 10^5 * 10^5 = 10^10.')
excerpt(a, '提前终止方案仍被新的最坏情况推翻',
        'The real worst case: elements = [2, 2, 2, ..., 2]', 'I need a different approach.')
excerpt(b, 'SC 较直接地进入首下标映射＋约数枚举',
        'One approach: For each possible divisor value `d`', 'Take the minimum.')
paragraph('C 先尝试按元素逐个标记倍数，再加“全部分配后退出”，最后用重复 2 与奇数组构造最坏情况，'
          '说明内层仍可能空跑约 50 亿次。这是对运行成本的实质检验，而不是重复说“应该足够快”。'
          '最终 C 改为记录每个数的最早下标、逐组枚举约数，SC 也采用这个算法，两者均通过。'
          'C 的推理为 8333 字符，SC 为 2598 字符，但两份最后得到同类 `O(m+n√V)` 求解过程；'
          'SC 的值域数组初始化另有 `O(V)`。这段更长的有效纠错证明了它摆脱先前低效方案，'
          '并没有证明它比同题较直接的方案更优。')

a, b = '3b67bd32e7ad7a72a64e', '4909fe4435fc4c3f8f62'
section('LiveCodeBench 3759：通过差异来自一个 API 拼写，而非核心算法',
        '对每个 i，在满足 nums1[j] < nums1[i] 的下标中，选最多 k 个 nums2[j] 求最大和。'
        '关键边界是严格小于：nums1 相同的元素不能互相贡献。',
        [(a, 'block-met-k2'), (b, 'block-met-k2-self-certainty')])
excerpt(a, 'C 的分组顺序与堆更新思路成立',
        'So the algorithm:\n1. Create pairs', 'Otherwise: do nothing')
excerpt(b, 'SC 描述的是同一核心算法',
        'So the algorithm:\n1. Sort indices by nums1 value',
        '- If heap size == k and value <= heap[0]: do nothing')
excerpt(a, 'C 实际提交的错误调用',
        '                    current_sum -= heapq.heapop(min_heap)',
        '                    current_sum += val', field='submitted_code')
excerpt(b, 'SC 实际提交的堆替换',
        '                    heap_sum -= min_heap[0]',
        '                    heap_sum += val', field='submitted_code')
paragraph('C 写成了不存在的 `heapq.heapop`，公开样例在第一次需要替换堆顶时就触发 `AttributeError`。'
          'SC 使用 `heapreplace`，实际返回 `[80,30,0,80,50]` 并通过。'
          '两边都正确处理了同值分组、先回答后插入，因此不能把这一胜负记为“SC 找到了更好的算法”。'
          '两份原文还都把总复杂度写成 `O(n log k)`，漏掉排序的 `O(n log n)`；'
          '这一文字不完整与导致运行失败的 API 错误，需要分别评价。')

a, b = '86217df34921541dfbda', 'c81caf648a042532f711'
section('LiveCodeBench 3750：继续简化，也可能把正确草稿改错',
        '在环形数组中，对指定下标查找另一个相同值位置的最短环距；没有其他位置则返回 −1。',
        [(a, 'met-k2'), (b, 'is')])
excerpt(a, '逐条 K=2 + C 早期已经覆盖首尾环绕候选',
        'The minimum circular distance from `q` to any point in `indices`',
        '- The last point (if wrapping around from the left gives a shorter path)')
excerpt(a, '随后把距离公式误当成候选覆盖的保证',
        'Actually, the nearest point to the left or right in the linear sense already covers',
        'and the circular distance formula handles the rest.')
excerpt(a, '最终删除首尾候选的结论',
        'Let me refine: check `indices[pos-1]`', 'This should be correct and efficient.')
excerpt(b, '快速 IS + C 也做简化，但保留循环邻接关系',
        'Actually, let me just use `pos` from bisect_left.',
        'the immediate next or immediate previous in the sorted circular list of indices.')
paragraph('反例取 `nums=[7,1,2,7,3,4,5,6,8,7]`、`queries=[0]`。值 7 的位置是 `[0,3,9]`。'
          '逐条 K=2 只看线性邻居 3，返回 3；位置 9 经环绕只距 0 一步，正确答案为 1。'
          '快速 IS 用 `(pos±1) % 频次` 取得循环邻居，实际返回 1 并通过整题。'
          '错误链中曾意识到 `min(d,n−d)` 在大距离处也会变小，却只用两个同值位置的例子确认；'
          '这种例子无法区分“线性邻居”和“循环邻居”。增加思考步骤在这里引入了回退。')

a, b = 'b9b0c5cc051f21637bcb', 'f35599a7e0885bd605e3'
section('LiveCodeBench abc392_d：相同概率公式，状态污染与判题容差是两件事',
        '从 N 个骰子中选两个，最大化掷出相同数字的概率。题面允许绝对或相对误差不超过 10⁻⁸。',
        [(a, 'is'), (b, 'is-self-certainty')])
excerpt(a, '快速 IS + C 的概率公式和遍历策略',
        'For two dice i and j, the probability they show the same number is:',
        'looking up in the larger one.')
excerpt(b, 'SC 使用相同的频数内积思路',
        'Approach:\n1. For each die, compute a frequency map',
        'look up in the larger one.')
excerpt(a, 'C 为遍历较小字典而交换外层状态',
        '    for i in range(N):\n        ki, fi = dice[i]',
        '                ki, kj = kj, ki', field='submitted_code')
excerpt(b, 'SC 为当前骰子对使用独立临时变量',
        '            if len(freq_i) <= len(freq_j):',
        '                larger = freq_i', field='submitted_code')
paragraph('C 的 `fi、ki` 在内层交换后没有为下一个 j 恢复，因而后续比较的已不是原来的第 i 个骰子。'
          '输入三枚骰子 `[1,2]、[3]、[1,2]`，两份实际程序分别输出 0 和 0.5；最优显然是第一与第三枚，概率 0.5。'
          'SC 的临时变量避免了这个具体错误。')
paragraph('但两份在冻结记录里均未通过，不能写成 SC 因此获得整题胜利。'
          '两份都在一个双骰测试上输出 `0.076923076923077`，基准文本为 `0.076923076923`；'
          '精确计数给出真实概率 `7/(13×7)=1/13`，两种小数都在题面容差内。'
          '所用判题器在文本不同时改为 Decimal 精确相等比较，仍不使用这道题的容差，因此给出 Wrong Answer。'
          '本组同时有可复现的代码缺陷和判题实现问题；原成绩保留，但不能把该浮点失败当作推理错误的证据。')

a, b = '33047554383b3cc4b3a7', 'a66262b492a64f368c33'
section('LiveCodeBench 3777：正确指出状态压缩问题，代码却没有遵守；通过测试仍有遗漏',
        '选非空子序列，使交替和等于 k，并最大化不超过 limit 的乘积。元素范围含 0，'
        '所以中途超过上限的乘积，之后乘 0 可能重新变得可行。',
        [(a, 'base'), (b, 'block-met-k2')])
excerpt(a, 'Base 明确识别只保存最大乘积不成立',
        'But as I noted, a smaller product might be better in some cases.',
        "So we can't just keep the max product. We need to keep all achievable products.")
excerpt(a, 'Base 最终代码仍只保存最大值，并初始化空子序列',
        '        # dp[parity][s_offset]', '        dp[0][OFFSET] = 1  # empty subsequence: parity 0, sum 0, product 1',
        field='submitted_code')
excerpt(b, '按块 K=2 + C 最终明确区分非空状态',
        "Simpler approach: just don't include the empty state.",
        'At the end, check dp for states with sum = k, find max product ≤ limit.')
excerpt(b, '它也意识到乘零会改变乘积',
        "Let me also add: if num == 0, then prod * num = 0 for all prods. So the product becomes 0. That's fine.")
paragraph('状态压缩的直接反例是 `[3,2,1,3], k=4, limit=8`：前缀 `[3,2]` 与 `[2,1]` 都是偶数长度、交替和为 1，'
          '乘积分别为 6 和 2。继续接上最后的 3，只有较小乘积能在上限内得到 6。'
          'Base 丢掉它后实际返回 −1，按块 C 保留乘积集合则返回正确的 6。')
paragraph('此外，Base 在 `[0], k=0, limit=1` 返回空序列留下的 1，正确值为 0；按块 C 的集合状态在此返回 0。'
          '但按块 C 在初始化和延伸时都删除超限乘积。对 `[10,0], k=10, limit=1`，'
          '非空子序列 `[10,0]` 的交替和为 10、乘积为 0，合法且最优；实际程序却返回 −1。'
          '枚举全部非空子序列验证了以上三个小例子。')
paragraph('按块 C 确实通过了本次全部 43 个测试，而 Base 未通过；同题按块 SC 与按块 C 的终选思维链、代码完全相同。'
          '因此这既是“较完整状态表示修复了一部分错误”的案例，也是“通过现有测试并未排除语义漏洞”的案例。'
          '原文同时提到风险和零值，并不等于最终转移保留了所需状态；仅搜索关键句会高估完成的修正。')

a, b = '8365280d2bcd3e04608f', 'b9d08bc11b68a11e90de'
section('LiveCodeBench abc388_f：一般可达性推导与零步边界，都必须落实到状态转移',
        '从格子 1 跳到 N，每步长度在 [A,B]，落点不能在坏区间中。N 可达 10¹²，A、B 至多 20，'
        '需要压缩长好区间的可达状态。',
        [(a, 'block-met-k2-self-certainty'), (b, 'met-k1')])
excerpt(a, '按块 SC 将含 k 的间隔错误消去',
        'The gap between [x+kA, x+kB]', 'because consecutive ranges overlap.')
excerpt(a, '它后来确实找到了无法迈出第一步的边界',
        'Hmm wait, if A=2, B=3, N=2:', 'So answer is No.')
excerpt(b, '逐条 K=1 + C 推导出正确的正距离判据',
        'For a given difference d = t - s:', 'This is iff ceil(d/B) ≤ floor(d/A).')
excerpt(b, '但在衔接坏区间时排除了已经站在临界格上的状态',
        '            for s in reachable:', '                d = x - s', field='submitted_code')
paragraph('SC 的间隔应为 `(k+1)A−kB−1 = A−k(B−A)−1`，并非 `A−B−1`；'
          '不同步数的可达区间不保证从一开始就相连。即使它后来明确说 `N=2,A=2,B=3` 应为 No，'
          '最终仍只检查 `lo≤N`，该输入实际输出 Yes。')
paragraph('逐条 K=1 的正距离公式更准确，但将 `s==x` 也跳过，遗漏无需在当前好区间继续移动的“零步”情形。'
          '对 `N=3,A=1,B=2`、唯一坏格 `[2,2]`，本可直接 `1→3`，实际却输出 No。'
          '两份在冻结测试中均未通过；一个是可达集合的错误概括与未落实的修正，另一个是局部公式正确、'
          '跨区间衔接遗漏边界。不能因后一份有正确公式，就把整份算法标成正确。')

a, b = '2d08d33a41e0d57c34e4', '8482a026a8c999fc404e'
section('HumanEval/46：手算序列正确，初始化依然可能写错',
        '计算四阶递推 fib4：初值为 `[0,0,2,0]`，之后每项等于前四项之和；要求迭代实现。',
        [(a, 'is-self-certainty'), (b, 'met-k1-self-certainty')])
excerpt(a, '快速 IS + SC 的样例演算全部正确',
        'Let me verify with the examples:',
        '- fib4(7) = fib4(6) + fib4(5) + fib4(4) + fib4(3) = 8 + 4 + 2 + 0 = 14 ✓')
excerpt(a, '它的实际初始化与上面的规则不一致',
        '    a, b, c, d = 0, 2, 0, 0', '    return d', field='submitted_code')
excerpt(b, '逐条 K=1 + SC 的思考也采用普通迭代',
        'So the approach is straightforward iterative computation.', 'then iterate from 4 to n.')
excerpt(b, '有效修正出现在最终代码内部',
        '    a, b, c, d = 0, 2, 0, 0', '    a, b, c, d = 0, 0, 2, 0', field='submitted_code')
paragraph('快速 IS 的代码把初值 2 放到了第二个变量，计算 `fib4(6)` 得到 6，而文字演算写的是正确的 8。'
          '逐条 K=1 代码也先写错一次，但紧接着用正确初值覆盖，实际返回 8 并通过。'
          '这份成功并不来自另一种递推算法，而来自初始化被真正修正。'
          '如果只看思维链，两份都能算对样例；如果只看代码第一条赋值，又会漏掉随后的覆盖。')

a, b = '82b01251f220ee233da5', '85115e5cbc606d0d3adf'
section('HumanEval/125：公开样例之外的规则扩展，可能改善实现但不改变得分',
        '字符串有空白字符时按空白分词；否则有逗号则按逗号分词；再否则统计字母表中奇数序号的小写字母。',
        [(a, 'base'), (b, 'is')])
excerpt(a, 'Base 对规则的复述覆盖了 whitespace',
        '1. Given a string of words, I need to:', "(where 'a'=0, 'b'=1, ..., 'z'=25)")
excerpt(a, '但它的实际条件只检查普通空格',
        "    if ' ' in txt:", "        return txt.split(',')", field='submitted_code')
excerpt(b, '快速 IS + C 主动检查空白的范围',
        'Wait, I should consider if there are other whitespace characters besides space.',
        "But to be more robust, I'll check for any whitespace character.")
excerpt(b, '修改落实为与 split 一致的判定',
        '    if any(c.isspace() for c in txt):', '        return txt.split()', field='submitted_code')
paragraph('两份都通过了本次 HumanEval+ 测试，但对 `a\\tb`（中间是制表符），Base 返回字母计数 1，'
          '快速 IS 返回两个词 a 和 b；后者符合题面 whitespace 的通常含义。'
          '能确认的是，既有成绩未区分这个边界，而规则审查区分出了行为。'
          '这也是“有实际价值的修正未必带来本次分数提升”的例子，不能仅按胜负筛选所有有信息的案例。')

a, b = 'dfddce6f0e269c7bc341', '312775e042c982c108c5'
section('HumanEval/160：把样例当作区分解释的实验',
        '给出操作符列表与非负整数列表，构造并求值表达式。关键歧义是按书写顺序计算，还是采用通常优先级。',
        [(a, 'is-self-certainty'), (b, 'base')])
excerpt(a, '快速 IS + SC 开头直接作了错误解释',
        'The user wants me to implement a function `do_algebra`', '(not following operator precedence).')
excerpt(a, '它用两种算法对同一表达式的不同预测修正规则',
        'Let me check: if we follow operator precedence:', 'or implement it with proper precedence.')
excerpt(b, 'Base 则先把两种解释作为待判定选项',
        'The key question is: do I need to respect operator precedence,',
        'build a string expression and use `eval()`.')
paragraph('`2+3×4−5` 按通常优先级是 9，从左到右是 15；给定样例明确区分了二者。'
          'SC 撤回开头的解释，最终和 Base 一样构造表达式并调用 Python 求值，两份均返回 9 并通过。'
          '这段修正较短，却包含可检验的竞争解释、区分它们的计算和最终实现。'
          '它比没有新证据的反复确认更有实质内容；不过它也没有因此超过同题一开始就正确处理歧义的 Base。')

a, b = 'da237fe7ba135d10b36c', 'b994d3b80115eaa2d5db'
section('HumanEval/97：特殊负数样例掩盖错误泛化，代码又可能偏离文字结论',
        '返回两个整数的个位数字之积，负数按绝对值的个位理解。公开负数样例是 `multiply(14,-15)=20`。',
        [(a, 'bon-consilience'), (b, 'met-k1')])
excerpt(a, 'Best-of-4 + C 从 −15 的巧合推出一般结论',
        'If I used `-15 % 10` in Python,', 'So that works.')
excerpt(a, '它只重复已有样例后采用了无 abs 的实现',
        'So simply `a % 10 * b % 10` would work?', 'So the solution is simply `(a % 10) * (b % 10)`.')
excerpt(b, '逐条 K=1 + C 在思维链中也给出了同样的错误泛化',
        'Actually, in Python, `-15 % 10 = 5`',
        "since Python's modulo handles negatives correctly for this case.")
excerpt(b, '但它最终交付的是带 abs 的正确版本',
        '    return (abs(a) % 10) * (abs(b) % 10)', field='submitted_code')
paragraph('`−15 % 10=5` 恰好等于绝对值个位，并不代表所有负数如此。'
          '`−12 % 10=8`，而 `abs(−12)%10=2`；对 `(14,−12)`，Best-of-4 实际返回 32，正确值为 8，'
          '逐条 K=1 实际返回 8。前者未通过，后者通过。'
          '但后一份文字最后仍说要用无 abs 的表达式，不能据其正确代码倒推“推理已经消除了误解”。'
          '本例应分别记录负数规则的错误泛化与最终代码的正确性。')

a, b = '3f0ec062125a71ff7413', '68487bab726a484bd3fd'
section('HumanEval/47：打勾的样例核对不一定真的完成了计算',
        '实现标准中位数。公开文档把 `[-10,4,6,1000,10,20]` 的中位数写成 15.0，'
        '但排序后中间两个数是 6、10，平均值应为 8.0。',
        [(a, 'met-k1-self-certainty'), (b, 'base')])
excerpt(a, '逐条 K=1 + SC 原文把错误算式当作核对通过',
        'Let me verify with the examples:', 'two middle elements are 6 and 10, average is 15.0 ✓')
excerpt(b, 'Base 则明确发现了题面和计算的矛盾',
        'Wait, that gives 8.0, not 15.0.',
        "The docstring says 15.0 which doesn't match. This seems like an error in the docstring. I'll implement the standard median calculation, which should be correct for the actual test cases.")
excerpt(a, 'SC 的最终公式仍然正确',
        '    if n % 2 == 1:', '        return (s[mid - 1] + s[mid]) / 2', field='submitted_code')
paragraph('两份实际程序都返回 8.0，并通过本次 HumanEval+ 测试。SC 的问题是局部算术陈述和验证结论错误，'
          '没有传播到通用中位数公式；Base 正确识别了不一致。'
          '与第 3 组对照，本题怀疑样例有明确算术依据，而第 3 组仍存在能解释样例的负号规则。'
          '评价“发现样例冲突”时，需要看冲突究竟被算清、被解释，还是只是被一句“样例可能错了”带过。')

assert len(sections) == 15 and len(selected) == len(set(selected)) == 30
validation = validate(cases, records)
validation.update(
    all_1739_cases_reviewed=True, selected_quotes_match=True,
    comparison_sections=len(sections), selected_cases=len(selected),
    thinking_excerpt_count=sum(q['field'] == 'thinking_text' for e in evidence for q in e['excerpts']),
    submitted_code_excerpt_count=sum(q['field'] == 'submitted_code' for e in evidence for q in e['excerpts']),
    exact_excerpt_characters=sum(len(q['quote']) for e in evidence for q in e['excerpts']),
    original_judgments_unchanged=True,
)
for item in evidence:
    assert item['excerpts']
    for q in item['excerpts']:
        assert cases[item['case_id']][q['field']][q['start']:q['end']] == q['quote']

source_lines = ['# 十五组对照的完整模型原文', '',
    '以下按报告顺序保存三十份被引用的终选思维链与实际提交，未经改写。'
    '“通过／未通过”是冻结判题记录；反例与判题器问题见主报告。'
    '同一文本可能对应多个方法，具体映射见 review-key.json。', '']
for item in evidence:
    cid = item['case_id']
    c = cases[cid]
    source_lines.extend([
        f'<a id="case-{cid}"></a>', '',
        f"## {item['section']}. {c['problem_id']} · {cs.NAMES[item['mode']]}", '',
        f"case_id：`{cid}`；本次判题：{'通过' if item['correct'] else '未通过'}；"
        f"思维链 {len(c['thinking_text'])} 字符；同文本方法："
        + '、'.join(cs.NAMES[m] for m in item['matching_methods']) + '。', '',
        '思维链原文：', '', '````text', c['thinking_text'], '````', '',
        '实际提交代码：', '', '````python', c['submitted_code'], '````', '',
    ])
observations = '\n'.join(lines)
cs.save(cs.OUT / 'selected-excerpts.json', selected)
cs.save(cs.OUT / 'selected-evidence.json', evidence)
cs.save(cs.OUT / 'comparison-index.json', sections)
cs.save(cs.OUT / 'report-validation.json', validation)
(cs.OUT / 'THOUGHT_OBSERVATIONS.md').write_text(observations, encoding='utf-8')
(cs.OUT / 'SELECTED_TRACES.md').write_text('\n'.join(source_lines), encoding='utf-8')

findings_path = cs.OUT / 'FINDINGS.md'
findings = findings_path.read_text(encoding='utf-8')
if '原始判题记录' not in findings:
    findings = findings.replace('本次固定题集上，', '按本次固定题集的原始判题记录，', 1)
    findings += ('\n扩展案例复核发现：abc392_d 的浮点比较未遵循题面容差，'
                 '另有通过固定测试却仍能构造反例的提交。表中保留原始成绩，'
                 '尚未进行全题语义重判；相关得分应按该判题口径理解，具体见第 8、9、12 组。\n')
    findings_path.write_text(findings, encoding='utf-8')
cs.prepare()
summary = json.loads((cs.OUT / 'summary.json').read_text(encoding='utf-8'))
assert summary['status'] == 'complete' and summary['review']['remaining'] == 0
print(json.dumps({k: validation[k] for k in ('comparison_sections', 'selected_cases',
    'thinking_excerpt_count', 'submitted_code_excerpt_count', 'exact_excerpt_characters')}, ensure_ascii=False))
