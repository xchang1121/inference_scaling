"""Build a cross-problem style report, preserving the earlier case and score appendices."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import statistics

import numpy as np

import coding_study as cs


STYLE = cs.ROOT / 'style_review'
FIELDS = ('exploration', 'revisiting', 'walkthrough', 'implementation', 'formalization', 'closure')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    manifest, problems, records, servers, attempts = cs.load()
    cases = {r['case_id']: r for r in cs.read_jsonl(cs.OUT / 'review-cases.jsonl')}
    case_key = json.loads((cs.OUT / 'review-key.json').read_text(encoding='utf-8'))
    all_labels = cs.labels()
    summary = json.loads((cs.OUT / 'summary.json').read_text(encoding='utf-8'))
    assert len(records) == 3168 and len(cases) == len(all_labels) == 1739 and not attempts
    assert set(cases) == set(all_labels) and summary['status'] == 'complete'
    assert all(s['generation_exit'] == s['judge_exit'] == '0' for s in servers.values())
    assert (cs.OUT / 'CASE_COMPARISONS.md').exists() and (cs.OUT / 'RESULTS.md').exists()

    baseline = json.loads((STYLE / 'prior-report/baseline.json').read_text(encoding='utf-8'))
    for name, expected in baseline['raw_source_sha256'].items():
        assert sha(cs.ROOT.parents[1] / Path(name.replace('\\', '/'))) == expected, name
    old_summary = json.loads((STYLE / 'prior-report/summary.json').read_text(encoding='utf-8'))
    assert summary['datasets'] == old_summary['datasets']

    metrics = json.loads((STYLE / 'metrics.json').read_text(encoding='utf-8'))
    protocol = json.loads((STYLE / 'protocol.json').read_text(encoding='utf-8'))
    packets = json.loads((STYLE / 'packets.json').read_text(encoding='utf-8'))
    key = json.loads((STYLE / 'key.json').read_text(encoding='utf-8'))
    labels = cs.read_jsonl(STYLE / 'labels.jsonl')
    assert len(labels) == len(packets) == 24 and len(key) == 48
    assert len({r['problem_id'] for r in labels}) == 24
    annotated = {r['problem_id']: r for r in labels}
    scores = {'is': [], 'is-self-certainty': []}
    used_cases = {}

    def use(pid, mode):
        row = records[pid, mode]
        cid = cs.digest([pid, row['thinking_text'], row['submitted_code']])[:20]
        assert cid in all_labels and cases[cid]['thinking_text'] == row['thinking_text']
        used_cases.setdefault(cid, (pid, mode))
        return cid, row

    for packet in packets:
        row = annotated[packet['problem_id']]
        assert [c['review_id'] for c in packet['cases']] == [c['review_id'] for c in row['cases']]
        for source, label in zip(packet['cases'], row['cases'], strict=True):
            info = key[label['review_id']]
            cid, record = use(info['problem_id'], info['mode'])
            assert info['case_id'] == cid and record['thinking_text'] == source['thinking_text']
            assert source['thinking_text'][label['quote_start']:label['quote_end']] == label['quote']
            assert label['note'] and all(label[f] in (0, 1, 2) for f in FIELDS)
            scores[info['mode']].append({'problem_id': packet['problem_id'], **label})

    strata = {'livecodebench': [p for p in problems if problems[p]['dataset'] == 'livecodebench'],
              'humanevalplus': [p for p in problems if problems[p]['dataset'] == 'humanevalplus']}
    for d in ('easy', 'medium', 'hard'):
        strata['livecodebench_' + d] = [p for p in strata['livecodebench'] if problems[p]['difficulty'] == d]
    comparison_checks = 0
    for name, data in metrics['strata'].items():
        ids = strata[name]
        for mode, values in data['methods'].items():
            texts = [records[p, mode]['thinking_text'] for p in ids]
            assert values['n'] == len(ids)
            assert abs(values['characters']['mean'] - statistics.mean(map(len, texts))) < 1e-9
            assert values['characters']['median'] == statistics.median(map(len, texts))
            assert values['thinking_with_code_fence'] == sum('```' in t for t in texts)
            assert values['thinking_over_10000_characters'] == sum(len(t) > 10000 for t in texts)
            assert values['single_search_step'] == sum(len(records[p, mode].get('trace', {}).get('steps', [])) == 1 for p in ids)
        for comp in data['comparisons']:
            left, right = comp['reference'], comp['alternative']
            a = np.array([len(records[p, left]['thinking_text']) for p in ids])
            b = np.array([len(records[p, right]['thinking_text']) for p in ids])
            assert comp['identical_thinking'] == sum(records[p, left]['thinking_text'] == records[p, right]['thinking_text'] for p in ids)
            assert comp['alternative_shorter'] == int((b < a).sum())
            assert comp['alternative_longer'] == int((b > a).sum())
            assert abs(comp['aggregate_character_percent_change'] - (b.sum() / a.sum() - 1) * 100) < 1e-9
            comparison_checks += 1

    counts = {mode: {f: dict(sorted(Counter(r[f] for r in rows).items())) for f in FIELDS}
              for mode, rows in scores.items()}
    ordinal = {}
    for field in FIELDS:
        counts_pair = Counter()
        for row in labels:
            values = {key[c['review_id']]['mode']: c[field] for c in row['cases']}
            delta = values['is'] - values['is-self-certainty']
            counts_pair['C_higher' if delta > 0 else 'SC_higher' if delta < 0 else 'equal'] += 1
        ordinal[field] = dict(counts_pair)
    # These report sentences are tied to the completed, still-unaltered semantic labels.
    assert counts['is']['revisiting'][2] == counts['is-self-certainty']['revisiting'][2] == 13
    assert counts['is']['formalization'][1] == counts['is-self-certainty']['formalization'][1] == 14
    assert counts['is']['walkthrough'][2] == counts['is-self-certainty']['walkthrough'][2] == 13
    assert sum(r['implementation'] > 0 for r in scores['is']) == 15
    assert sum(r['implementation'] > 0 for r in scores['is-self-certainty']) == 11
    assert ordinal['formalization']['equal'] == 24

    variance, tails = {}, {}
    for ds in ('livecodebench', 'humanevalplus'):
        ids = strata[ds]
        x = np.log(np.array([[len(records[p, m]['thinking_text']) for m in cs.MODES] for p in ids]))
        grand, problem, method = x.mean(), x.mean(axis=1, keepdims=True), x.mean(axis=0, keepdims=True)
        total = ((x - grand) ** 2).sum()
        variance[ds] = {'problem_fraction': float(len(cs.MODES) * ((problem - grand) ** 2).sum() / total),
                        'method_fraction': float(len(ids) * ((method - grand) ** 2).sum() / total),
                        'interaction_and_unseparated_sampling_fraction': float(((x - problem - method + grand) ** 2).sum() / total),
                        'interpretation': 'Descriptive two-way decomposition of log character length, not causal attribution.'}
        delta = [(p, len(records[p, 'is-self-certainty']['thinking_text']) - len(records[p, 'is']['thinking_text'])) for p in ids]
        top = sorted(delta, key=lambda item: item[1])[:5]
        tails[ds] = {'net_SC_minus_C_characters': sum(d for _, d in delta),
                     'five_largest_shortenings': [{'problem_id': p, 'SC_minus_C_characters': d} for p, d in top],
                     'five_largest_sum': sum(d for _, d in top),
                     'selection': 'Post-hoc descriptive contribution; not a representative semantic sample.'}

    audit = {'scope': 'All 264 problems x 12 methods for objective metrics; a fixed 24-problem paired semantic recoding of fast IS C/SC.',
             'all_original_unique_cases_read': 1739, 'semantic_sample_problems': 24, 'semantic_sample_texts': 48,
             'semantic_counts': counts, 'paired_ordinal_counts': ordinal,
             'semantics_limits': 'One assistant reviewer; method names masked during coding; all texts had been accessible in the earlier full review. Not an independent multi-rater or experiment-preregistered study. Quota sample counts are not population estimates.',
             'length_variance_decomposition': variance, 'length_tail_contribution': tails,
             'protocol_sha256': sha(STYLE / 'protocol.json'), 'labels_sha256': sha(STYLE / 'labels.jsonl'),
             'metrics_sha256': sha(STYLE / 'metrics.json')}

    frozen = json.loads((cs.ROOT / 'FROZEN.json').read_text(encoding='utf-8'))
    source_names = ['source/scripts/benchmark_is_gpu.py',
                    'source/src/inference_scaling/shared/rewards/consilience.py',
                    'source/src/inference_scaling/arllm/rewards/intrinsic.py',
                    'source/src/inference_scaling/shared/sampling/conditional_is.py',
                    'source/src/inference_scaling/arllm/algorithms/conditional_is.py', 'settings-a.json', 'settings-b.json']
    verified_sources = {}
    for name in source_names:
        path = cs.ROOT / name
        assert sha(path) == frozen['files'][name], name
        verified_sources[name] = sha(path)

    findings = (
        '**本次十二种方法最一致的风格差异，发生在 C 与 SC 评分配置之间：C 组平均展开更长，SC 组平均更紧凑。'
        '这些额外篇幅既包含有用的方案、证明与实现讨论，也包含已定案后的反复确认。**'
        '在四组可配对的 IS 实现中，SC 的平均思维链字符数比 C 少约 9%—11%（LiveCodeBench）和 10%—14%（HumanEval+）。'
        '这是分布层面的倾向，不能改写成“SC 在每道题上更短”。\n\n'
        '**语义风格的分化弱于篇幅的分化。** 固定 24 题的方法名遮蔽复核中，反复重推、长样例演算和显式形式化在 C/SC 两边的计数分别相同：'
        '13/24、13/24、14/24。样本中，C 出现实现草稿或实现讨论略多（15/24 对 11/24），但尚不足以认定普遍规律。'
        '因此，这批材料支持展开幅度的差异，不支持把两者概括为“探索型”与“直接型”两种截然不同的推理模式。\n\n'
        '**SIR、逐条 Metropolized、按块 Metropolized 以及 K=1/2，主要改变候选选择和内部计算，尚未呈现各自稳定、可区分的文本风格。**'
        '它们在不少题上保留逐字相同的思维链；个别题会明显改变路线，但没有跨题一致的方向。Base 和 Best-of-4 也共享同一套推理与叙述习惯。'
        '题目本身造成的风格差异，比这里观察到的方法差异更大。'
    )
    lines = ['# 十二种推理采样方法的思维链风格：跨题比较', '', findings, '']
    evidence = []

    def p(text):
        lines.extend([text, ''])

    def quote(pid, mode, start_text, end_text=None, *, occurrence=0, description=None):
        cid, row = use(pid, mode)
        source = row['thinking_text']
        start = -1
        for _ in range(occurrence + 1):
            start = source.index(start_text, start + 1)
        end = start + len(start_text) if end_text is None else source.index(end_text, start) + len(end_text)
        text = source[start:end]
        assert 0 < len(text) < 2000
        item = {'problem_id': pid, 'mode': mode, 'case_id': cid, 'field': 'thinking_text',
                'start': start, 'end': end, 'quote': text,
                'thinking_sha256': hashlib.sha256(source.encode()).hexdigest(),
                'source_replica': row['replica'], 'source_line': row['source_line'],
                'thinking_characters': len(source)}
        evidence.append(item)
        p(f"{cs.NAMES[mode]} · {pid}（{len(source):,} 字符，"
          f"[完整原文](STYLE_TRACES.md#case-{cid})）" + (f'，{description}' if description else '') + '：')
        p('\n'.join('> ' + line for line in text.split('\n')))

    p('材料覆盖固定的 LiveCodeBench 100 题（25 易、30 中、45 难）和 HumanEval+ 164 题，共 3168 份终选输出；'
      '同题按思维链与提交代码一起去重后的 1739 份材料均已通读。这里比较的是保存下来的 **最终选中思维链**，'
      '不包含未被保留的候选全文。所有方法使用同一模型、提示和名义生成预算；每题每方法只有一个固定种子结果。')
    results_body = (cs.OUT / 'RESULTS.md').read_text(encoding='utf-8').split('\n', 1)[1].strip()
    results_table = '\n'.join(line for line in results_body.splitlines() if line.startswith('|'))
    assert len(results_table.splitlines()) == len(cs.MODES) + 2
    p('**正确率与性能。**')
    p(results_body)
    p('这里的“风格”包括：是否比较不同方案，是否回到已讨论过的步骤，怎样验证样例，是否展开代码与工程细节，'
      '是否明确写出状态、递推或不变量，以及如何结束。字符数只衡量展开幅度，不能替代这些语义判断。'
      '为检查跨题倾向，本次另固定抽取 24 题、遮蔽方法名后逐篇编码；详见 [复核口径](STYLE_METHODS.md)。')

    p('**C 与 SC 的差异首先体现在平均展开幅度。** 下表使用全部题目，仅列思维链 Unicode 字符数均值；'
      '包含空格和思维链内的代码草稿，不含最终回答，不是 token 数。')
    lines.extend(['| 方法 | LiveCodeBench：100 题 | HumanEval+：164 题 |', '|---|---:|---:|'])
    for mode in cs.MODES:
        l = metrics['strata']['livecodebench']['methods'][mode]['characters']['mean']
        h = metrics['strata']['humanevalplus']['methods'][mode]['characters']['mean']
        lines.append(f'| {cs.NAMES[mode]} | {l:,.0f} | {h:,.0f} |')
    lines.append('')
    p('在快速 IS、逐条 K=1、逐条 K=2、按块 K=2 内分别把 C 换为 SC，LiveCodeBench 的均值依次下降 '
      '**8.6%、9.3%、11.2%、9.0%**；HumanEval+ 依次下降 **10.5%、10.2%、12.0%、14.4%**。'
      '按题配对、保持难度层大小的 5000 次重采样中，这八个均值差异的描述性 95% 区间均在零以下。'
      '它们来自相关的同一批题，不能当成八次独立实验，也没有覆盖跨种子的波动。')
    p('但平均差异并不是每题都发生的风格转换。以快速 IS 为例，LiveCodeBench 中 SC 更短 45 题、更长 31 题，'
      '另有 24 题思维链逐字相同；HumanEval+ 对应为 64、30、70 题。两套题的配对长度比中位数都是 1。'
      'LiveCodeBench 总共净减少 64,359 字符，其中缩短最多的 5 题贡献 46,788 字符，约占净减少量的 73%。'
      '因此，更准确的说法是：**SC 减少了一部分很长的展开，拉低了平均值；这不足以推出大多数题改变了叙述结构。**')

    p('**C 较长时，多出的内容常在核心路线已经找到之后。** 对平均差异贡献较大的题进行原文核对，可以把延长区分为'
      '反复证明同一结论、继续比较实现表示、以及追加小例。这几种活动都可能有用，也都可能在没有新信息时延长文本。'
      '下面几段取自差异较大的输出，用来解释“多写了什么”，不用于计算这些活动的总体发生率。')
    p('例如，判断子序列是否唯一、带全局反转的最短路、大小配对这三种不同任务里，快速 IS + C 的长度分别为 '
      '13,813、12,695、13,127 字符，SC 为 2,894、2,532、2,797。三题的共同点是：C 在很早给出可行路线之后仍继续展开。'
      '子序列任务中，C 已完成两端贪心的证明，又重新询问已回答的问题；SC 同样说明了两端夹住所有合法位置的理由，但较早结束。')
    quote('lcb/arc195_a', 'is', 'So algorithm correct.\n\nLet\'s think of alternative:',
          "We proved no. Let's try to construct counterexample?")
    quote('lcb/arc195_a', 'is-self-certainty', "This is because if there's any subsequence,",
          "there's only one choice).")
    p('这个对照中的区别是同一证明被延长了多少次，而不是 C 找到了证明、SC 没有证明。'
      'HumanEval/116 的题意与样例冲突也呈现类似现象：两边都发现冲突、重算并决定按题意实现；'
      'C 用 5712 字符多轮重算负数的二进制表示，SC 用 3030 字符保留了同一冲突处理过程。'
      '这是紧凑程度的变化，并非“有反思／无反思”的二分。')
    quote('HumanEval/116', 'is', "Let me reconsider. What if we use `bin(abs(x)).count('1')`",
          'Result: [-4, -2, -6, -5, -3] ≠ [-6, -5, -4, -3, -2]')
    quote('HumanEval/116', 'is-self-certainty', 'Actually, I think the safest approach is to implement the logic as stated',
          'This should work for the described behavior. Let me go with this.')

    p('第二种延长发生在实现层。最短路任务中，C 早已确定“顶点＋方向”的双层状态，却继续比较双数组与状态编号、'
      '堆元组大小、输入解析和内存；SC 也明确给出状态与转移，只是没有展开同样多的表示选择。'
      '因此，不能把两者的长度差都算作“搜索了更多解题算法”。')
    quote('lcb/abc395_e', 'is', 'Possible micro-optimizations:',
          'So use `toggle = id + N if id < N else id - N`.')
    quote('lcb/abc395_e', 'is-self-certainty', 'This is a shortest path problem on a graph with 2N nodes',
          "We can use Dijkstra's algorithm since all edge weights are non-negative (cost 1 or cost X, where X >= 1).")
    p('第三种延长是验证仍在继续。大小配对任务里，C 在交换论证之后又进入多组 N=2、3、4 的手算，'
      '再回头质疑半区划分；SC 在选择二分可行性检查、核对样例与复杂度之后结束。'
      '这里还存在具体算法选择的差异，但更长的文本并不自动表示采用了更复杂的解法。')
    quote('lcb/abc388_e', 'is', 'Alternatively, the greedy directly produces max K.',
          "Let's test edge N=2. m=1. i=0,j=1. if 2*A0<=A1 count1 else0. Good.")
    quote('lcb/abc388_e', 'is-self-certainty', 'This approach works. Binary search on K from 0 to N//2,',
          'Let me code this up.')
    p('这种展开倾向与评分机制相容，但尚不能解释成已经证实的因果关系。本次 C 的分数为末段平均置信统计减去 '
      '3 倍初段平均统计；SC 对整段思维链的全词表分布确定性取平均。前者按前后轨迹打分，后者按整段均值打分，'
      '可能偏好不同的候选文本。两者都没有直接评判某次反思是否带来新信息；SC 也没有显式的“越短越好”项。'
      '本实验同时固定了不同的评分温度（C 为 2，SC 为 0.25），所以这里比较的是两套评分配置，而非单独替换一个公式的纯效应。')

    p('**SC 的紧凑倾向，不等于稳定地省略反思、样例或形式化。** 固定 24 题的语义复核中，两边“反复重推”都是 '
      '13/24，“详细样例演算”都是 13/24，“显式形式化”都是 14/24；形式化这一项在全部 24 对中的编码逐对相同。'
      '实质性方案比较为 C 14/24、SC 12/24，其中持续多路线探索反而是 C 3/24、SC 4/24。'
      '有实现草稿或实质实现讨论的是 C 15/24、SC 11/24，但逐对仅 4 题 C 更展开、1 题 SC 更展开，其余 19 题同档。'
      '这些计数更支持“保留相近的推理骨架，改变部分展开量”，而不是稳定地删除某种思维活动。')
    p('反向例子也构成这一判断的一部分。二进制字符串交易任务中，SC 为 10,209 字符，C 为 6065；'
      'SC 在已经写出程序后长时间检查循环端点与退化输入。单词内部排序任务中，SC 也比 C 长，'
      '增加的是对连续空格这一具体语义的验证。后一种增加有明确对象，不能一概视作无效犹豫。')
    quote('lcb/3805', 'is-self-certainty', 'Wait, let me double-check the indexing.',
          'So idx >= 2 and idx <= k-3.')
    quote('HumanEval/86', 'is-self-certainty', 'Let me think about how to handle multiple spaces.',
          "Joining: `'a  b'`. That works.")
    p('难题中，两种评分还会共同陷在同一个建模障碍里。在加权相邻交换任务中，两边都反复提出“边界净流量就是交换次数”，'
      '被样例否定后又回到该式。C 为 15,229 字符，SC 为 14,172，都是持续重推而未完成结论；'
      '换评分配置没有稳定改变这种循环式展开。')
    quote('lcb/arc194_b', 'is', 'So my formula gives 3 but the answer is 4. My reasoning must be flawed.',
          "No wait, that can't be.")
    quote('lcb/arc194_b', 'is-self-certainty', 'I think the issue is that the number of times a gap is used',
          'This is getting circular. Let me think of another approach.')
    p('另外，24 对中有 5 对思维链全文相同：其中既有直接求解的短文本，也有动态规划的长样例、整数计数的反复推导和筛法的性能讨论。'
      '相同的长链同样重要：它说明换成 SC 后，反复确认这一风格可以完整保留下来。'
      '本次语义编码中，停在半句或代码中间的是 C 7/24、SC 8/24，也没有支持“SC 总能更好收束”。'
      '这种文本末尾观察与运行记录里的总输出 `length_exhausted` 是两个口径，不能互相替代。')

    p('**快速 IS、SIR 与逐条 Metropolized 的区别，首先是怎样选择候选；文本上没有稳定的三分。**'
      '在 C 配置下，快速 IS 与 SIR K=1 的思维链有 27/100 道 LiveCodeBench、58/164 道 HumanEval+ 逐字相同。'
      'SIR K=1 与逐条 K=1 对应为 27/100、64/164；K=2 时为 19/100、63/164。'
      '这些比较的配对长度比中位数都为 1。SIR K=1 与逐条 K=1 在 LiveCodeBench 上的平均长度差仅 −0.4%，'
      'HumanEval+ 为 +0.8%，区间都跨零。没有证据把 SIR 概括为“更发散”、逐条概括为“更收敛”。')
    p('实现中，SIR 按完整候选的权重抽取；逐条 Metropolized 从当前保留项做一次候选索引更新；'
      '快速 IS 允许在可以作出选择时结束候选池计算。这些操作改变哪条推理被保留，'
      '并不等于对文本的每个推理步骤进行逻辑审查。“逐条”指候选叶的选择粒度，不代表模型会逐条证明自己的主张。')
    p('局部差异可以很大。奇偶分割任务中，SIR K=1 出现 2693 字符的“提出错误条件—用样例修正”过程，'
      '逐条 K=1 则用 1077 字符直接代数化；但同题 SIR K=2 又变成了与快速 IS + C 以及四种 SC 配置逐字相同的 1146 字符版本。'
      '这更像候选路线被切换，而非 SIR 固有的反思风格。')
    quote('lcb/3704', 'sir-k1', 'That gives 3, but expected output is 4. Let me re-check.',
          'So both being odd also works!')
    quote('lcb/3704', 'met-k1', 'Let me think about when the difference is even:',
          'So we need total_sum to be even')

    p('**K 从 1 增加到 2，并不让最终思维链多展示一倍方案或多一轮自我检查。** K 表示候选前缀的 rollout 数；'
      '最终仍只保留一条文本。HumanEval+ 上，SIR K=1/2 有 126/164 题思维链相同，逐条 C 有 122/164，逐条 SC 有 129/164。'
      '在 C 的 LiveCodeBench 上，SIR 的平均长度只增加 0.5%，逐条只增加 1.1%，相同文本分别有 35/100、29/100。'
      '这些长度差异的区间均跨零，不支持把 K=2 的风格写成“更深入、更详细”。')
    p('实际执行也解释了为什么名义 K 不一定造成明显文本差异。HumanEval+ 的逐条 K=2 + C 有 155/164 题只发生一个选择步骤，'
      '平均每题新评估 4.31 条 rollout，K=1 为 4.13；短候选已经终止时，不会硬凑一条有意义的新续写。'
      '而 LiveCodeBench 难题中，同两种配置的平均新评估数从 7.80 增至 12.71，最终思维链均长却从 11,819 变为 11,418 字符。'
      '内部搜索量增加与可见文本变长是两件不同的事。')

    p('**按块 Metropolized 能换掉一段候选，但未形成稳定的“分块推导”文风。** 本次实现把相同 block 的候选权重合并后选择组，'
      '再在组内选续写；不会要求模型把论证写成更多章节，也不会把多个候选全文合并。'
      '与逐条 K=2 相比，按块 K=2 在 C 下有 45/100 道 LiveCodeBench、123/164 道 HumanEval+ 的思维链完全相同；'
      'SC 下分别为 37/100、131/164。C 的两套题平均长度只差 +0.5%、+1.7%，SC 为 +3.0%、−1.1%，方向也并不一致。')
    p('按块方法有时确实保留了更丰富的局部探索。例如分配因子任务中，按块 K=2 + C 在 8333 字符内两次构造复杂度反例，'
      '同题逐条 K=2 + C 只有 2256 字符。这段增加是实质性排除方案，不应与无信息的重复混在一起；'
      '但结合大量相同文本，它只能说明按块选择可能改变路线，不能证明它普遍更会找反例。')
    quote('lcb/3760', 'block-met-k2', 'The real worst case: elements = [2, 2, 2, ..., 2]',
          'I need a different approach.')

    p('**Base 与 Best-of-4 共享上述叙述习惯；Best-of-4 没有稳定地选出“更成熟的结论稿”。**'
      'Base 也会直接建模、列样例、回头修改规则；C 系列相对 Base 在 LiveCodeBench 上平均长约 4.8%—7.1%，'
      '但各自配对区间都跨零，不能据此断言所有 IS 都普遍更冗长。Best-of-4 + C 的均值比 Base 高 5.5%（LiveCodeBench）'
      '和 2.6%（HumanEval+），同样没有清楚的跨题方向。')
    p('Best-of-4 从四条完整候选中保留一条，它的输出也可以很简洁：前述子序列和最短路任务分别只有 2492、1985 字符；'
      '也可以很长：大小配对任务达到 13,024 字符，持续重复论证并在末尾尚未收束。'
      'HumanEval/145 中它还选中了重新解释负号规则的探索性文本，而不是直接接受初始解释。'
      '这些相反表现排除了“Best-of-4 只会保留整洁、单线、已收敛推理”的概括。')
    quote('HumanEval/145', 'bon-consilience', 'Hmm, what if the sum of digits is computed differently for negative numbers?',
          'That matches! So the approach is: for negative numbers, the first digit is negated.')
    p('同样不能把候选池中的四次尝试理解为最终文本里的四条可见分支。'
      'SIR、Metropolized 和 Best-of-4 都可能把内部计算用于选择一份看起来非常直接的答案；'
      '一个长链也可能只是同一路线反复展开。评价 IS 的特性，需要同时看内部选择记录和终选文本，不能相互代替。')

    p('**跨方法最稳定的共同变化，来自题目的难度与歧义。** 在 LiveCodeBench 易题上，十二方法的平均思维链都在 '
      '922—1031 字符附近；到难题则都在 10,474—12,004 字符附近。这个数量级的变化远大于同组 C/SC 的平均差异。'
      '短题常沿“规则—实现步骤—少量例子”结束；长题则共同出现状态重设、反复验证、性能讨论，或在同一障碍上循环。'
      '固定样本的 7 道难题中，C 和 SC 都有 6 道思维链停在未写完的推导或代码处。')
    p('本次能支持的算法级画像应当保持这个尺度：**C/SC 主要区分展开幅度与部分实现讨论；SIR、逐条、按块、K 的变化主要区分候选选择，'
      '尚不足以区分稳定文风；题目条件决定了大部分共同的叙述结构。**'
      '这些结论仍需跨种子复现。还应区分表达形式与推理效果：'
      '“写得更长”“出现修正”“口气更确定”都不能直接代替正确性、验证有效性或解题能力。')
    p('原有十五组详细对照与 42 段思维链摘录保留在 [逐题复核附录](CASE_COMPARISONS.md)，其完整原文在 '
      '[原案例全文](SELECTED_TRACES.md)；成绩、成本与判题口径保存在 [结果附录](RESULTS.md)。'
      '本文引用及固定样本的全文见 [风格复核原文](STYLE_TRACES.md)，量化数据见 [style-metrics.json](style-metrics.json)，'
      '语义计数见 [style-summary.json](style-summary.json)，逐项编码见 [style-review-labels.jsonl](style-review-labels.jsonl)。'
      '所有引文的来源与字符偏移可在 [style-evidence.json](style-evidence.json) 核对。')

    report = re.sub(r'([。！？；])\*\*(?=\w)', r'\1** ', '\n'.join(lines))
    findings = re.sub(r'([。！？；])\*\*(?=\w)', r'\1** ', findings)
    assert report.count('| 方法 |') == 2 and results_table in report
    assert '## 典型思维链对照' not in report
    assert len(evidence) >= 16 and len({e['problem_id'] for e in evidence}) >= 10
    for item in evidence:
        source = records[item['problem_id'], item['mode']]['thinking_text']
        assert source[item['start']:item['end']] == item['quote']

    trace_lines = ['# 风格复核的完整模型原文', '',
                   '包括固定 24 题语义复核及正文引用的补充材料。按题目、思维链与提交代码去重；'
                   '相同思维链但代码不同的记录仍分别保存。方法映射来自冻结输出，原文未经改写。'
                   '这些是模型输出材料，不是报告作者的推理；其中的陈述可能错误。', '']
    for cid, (pid, mode) in used_cases.items():
        case = cases[cid]
        trace_lines.extend([f'<a id="case-{cid}"></a>', '', f'## {pid} · {cs.NAMES[mode]}', '',
                            f'case_id：`{cid}`；思维链 {len(case["thinking_text"]):,} 字符；同一思维链与代码对应方法：'
                            + '、'.join(cs.NAMES[r['mode']] for r in case_key[cid]) + '。', '',
                            '````text', case['thinking_text'], '````', '',
                            '实际提交代码：', '', '````python', case['submitted_code'], '````', ''])

    methods = '''# 风格复核口径与实现依据

本报告以同一固定题目内的方法对照为基础。全量客观统计覆盖 264 题 × 12 方法；原先的内容通读覆盖同题思维链与代码去重后的 1739 份材料。本次追加语义编码的样本仅比较快速 IS + C 与快速 IS + SC，不能把这 24 题的计数直接转用到其他方法。

抽样规则先固定为按 `SHA256(style-audit-20261106-v1|problem_id)` 排序，在 LiveCodeBench 易、中、难分别取 4、5、7 题，HumanEval+ 取 8 题。没有按长度或通过与否筛选。每题两份文本用 A/B 和随机编号遮蔽方法名，通读全部思维链、记录六项语义编码和精确引文后再查看映射。两份相同的前缀可以引用另一份已经读过的原文；没有用关键词次数替代阅读。

这是同一个助手进行的复核，文本此前已在全量通读中出现过。方法名遮蔽不等于独立审阅者盲评；没有多人一致性检验。抽样和量表是本次复核前固定的，不是生成实验前预注册的。分层配额也不等于全题集的自然比例。因此，24 题计数仅描述这个探索性样本，不报告为总体风格发生率。

六项编码为：方案探索（0 单路线，1 实质比较或修订两种方案，2 持续多路线切换）；回访（0 无实质回头，1 一次局部澄清，2 反复重推）；样例（0 无，1 简短核对，2 详细逐步或多次长演算）；实现（0 无扩展实现讨论，1 一个实质草稿或实现讨论，2 多草稿或持续表示、性能、边界工程讨论）；形式化（0 直接规则或口头说明，1 明确状态、递推、不变量或代数推导）；结尾（0 普通收束，1 尚有未解决选项，2 半句或代码处中断）。量表描述活动，不评价它是否正确或有益。

字符数采用 Python 对 `thinking_text` 的 Unicode 长度，包括空格与思维链内代码草稿，排除最终回答；没有把字符数称为 token 数。全文相同是字符串严格相等，不能推出最终代码也相同。超过 10000 字符、含代码围栏是附加客观描述，不作为语义风格分类器。配对百分比为 `sum(SC) / sum(C) - 1`，另保存每题长度比中位数和短／长／相同的题数。区间用 5000 次按题配对重采样，LCB 保持易、中、难层大小；仅反映这一固定实验中的题目变化，未处理跨种子误差或多重比较。补充保存的对数长度双因素分解也只是描述性分解，残差混合方法与题目交互、采样变化，不能作因果比例解释。

正文用于解释长度成分的补充案例，部分来自缩短幅度较大的题，是事后选择的说明材料，未加入固定 24 题的语义计数。旧的十五组逐题案例同样不是风格发生率样本。

**冻结实现中的评分与选择机制。** 模型为 Qwen3.8-27B，同题同提示同种子，生成温度 1；thinking effort 为 low，对应 4096 token 名义预算，总输出上限 8192。所有方法都共享对 wait、hmm 等词的 token penalty 配置，故这些词的出现次数尤其不适合直接当作反思程度。

C（Consilience）使用 top-5 置信统计轨迹，跳过最初 5%，以长度 20% 的初段窗口和末段窗口计算 `末段均值 - 3 × 初段均值`；评分温度为 2。SC（Self-Certainty）对思维链上逐 token 的全词表分布相对均匀分布的 KL 取均值，评分温度为 0.25。SC 的均值不是长度惩罚，也不是判题器给出的正确性概率。两种配置均只对 thinking 段评分。

SIR 按候选的完整权重抽样。逐条 Metropolized 在候选叶层面，从候选 0 的首个完整 rollout 做一次索引更新；按块版本先对相同 block 汇总权重，再条件抽取叶。它们没有逐句检查逻辑，也不通过文字是否含“反思”来打分。K 表示每个候选前缀的 rollout 数；已终止的候选组可折叠为一条，记录中的“新评估数”排除复用候选。Base 和 Best-of-4 没有 IS step 字段，不能把其 step 统计值 0 理解为没有计算。快速 IS 的 selection-ready 执行还可能在完成选择后取消其余生成，因此名义配置相同不等于实际计算相同。

运行记录的 `length_exhausted` 指总输出长度口径；语义编码的“中断”只描述思维链末尾文本。强制闭合 thinking、转入回答后，结构上的 thinking_status 可以 complete，原文仍可能停在半句。报告不混用这三种口径。

实现依据来自本次冻结源代码，提交 `214a358c7ba2ff3037fdf7a8e188fb3877f81dcb`。下列文件均已与 FROZEN.json 的 SHA-256 校验：

- `scripts/benchmark_is_gpu.py`：模式、K、评分类型与 thinking 范围的选择。
- `src/inference_scaling/shared/rewards/consilience.py`：C 的窗口和算式。
- `src/inference_scaling/arllm/rewards/intrinsic.py`：SC 均值和 C 的输入统计。
- `src/inference_scaling/shared/sampling/conditional_is.py`：SIR、逐条与按块的选择核。
- `src/inference_scaling/arllm/algorithms/conditional_is.py`：保留完整序列、两阶段计划与实际步骤。
- `settings-a.json`、`settings-b.json`：生成、预算、评分温度与共同 token penalty。

全文、引文和标签均保留可核对的来源标识。验证结果见 [style-validation.json](style-validation.json)，原始实验版本见 [FROZEN.json](FROZEN.json)。性能得分及浮点判题容差等限制见 [结果附录](RESULTS.md)；本次重写未修改生成结果或原始判题。
'''

    for source, target in [('protocol.json', 'style-protocol.json'), ('packets.json', 'style-review-packets.json'),
                           ('key.json', 'style-review-key.json'), ('labels.jsonl', 'style-review-labels.jsonl'),
                           ('metrics.json', 'style-metrics.json')]:
        shutil.copyfile(STYLE / source, cs.OUT / target)
    cs.save(cs.OUT / 'style-summary.json', audit)
    cs.save(cs.OUT / 'style-evidence.json', evidence)
    (cs.OUT / 'REPORT.md').write_text(report, encoding='utf-8')
    (cs.OUT / 'FINDINGS.md').write_text(findings + '\n', encoding='utf-8')
    (cs.OUT / 'THOUGHT_OBSERVATIONS.md').write_text(report, encoding='utf-8')
    (cs.OUT / 'STYLE_TRACES.md').write_text('\n'.join(trace_lines), encoding='utf-8')
    (cs.OUT / 'STYLE_METHODS.md').write_text(methods, encoding='utf-8')
    # Keep the existing full-trace appendix's evidence pointer correct after moving case discussions.
    trace_path = cs.OUT / 'SELECTED_TRACES.md'
    trace_text = trace_path.read_text(encoding='utf-8')
    trace_text = trace_text.replace('反例与判题器问题见主报告。', '反例与判题器问题见 [逐题复核附录](CASE_COMPARISONS.md)。')
    trace_path.write_text(trace_text, encoding='utf-8')
    validation = {'all_1739_original_cases_reviewed': True, 'all_3168_outputs_preserved': True,
                  'dataset_score_and_cost_aggregates_unchanged': True,
                  'accuracy_and_performance_table_in_main_report': True,
                  'accuracy_and_performance_method_rows': len(cs.MODES),
                  'fixed_semantic_pairs_checked': len(labels), 'fixed_semantic_labels_checked': len(key),
                  'all_semantic_quotes_match': True, 'objective_comparisons_checked': comparison_checks,
                  'style_quote_count': len(evidence), 'style_quote_problem_count': len({e['problem_id'] for e in evidence}),
                  'style_quote_characters': sum(len(e['quote']) for e in evidence),
                  'all_style_quotes_match': True, 'full_trace_cases_written': len(used_cases),
                  'raw_source_sha256': baseline['raw_source_sha256'], 'verified_implementation_sha256': verified_sources,
                  'artifact_sha256': {n: sha(cs.OUT / n) for n in ('REPORT.md', 'STYLE_TRACES.md', 'STYLE_METHODS.md',
                                      'style-evidence.json', 'style-summary.json', 'style-metrics.json',
                                      'style-review-labels.jsonl', 'CASE_COMPARISONS.md', 'RESULTS.md')}}
    cs.save(cs.OUT / 'style-validation.json', validation)
    return {k: validation[k] for k in ('fixed_semantic_pairs_checked', 'style_quote_count', 'style_quote_problem_count',
                                      'style_quote_characters', 'full_trace_cases_written')}


if __name__ == '__main__':
    print(json.dumps(build(), ensure_ascii=False))
