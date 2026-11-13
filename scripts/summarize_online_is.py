"""Audited paired F1, generated-token cost and latency for from-start IS."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_online_is import audit_snapshot
from summarize_cot_guidance import read
from summarize_flare_gap import bootstrap, grade

ARMS = ('direct', 'closed_is', 'static_facts', 'static_facts_noise', 'online_is', 'online_is_noise')
PAIRS = (('online_is', 'closed_is'), ('online_is', 'static_facts'),
         ('online_is', 'online_is_noise'), ('static_facts', 'static_facts_noise'),
         ('static_facts', 'closed_is'), ('online_is_noise', 'closed_is'))
LABELS = {'direct': '直接生成', 'closed_is': '无检索 IS', 'static_facts': '静态事实 + IS',
          'static_facts_noise': '静态等长无关文本 + IS', 'online_is': '全程在线事实 + IS',
          'online_is_noise': '全程在线等长无关文本 + IS'}


def aggregate(records):
    """Bootstrap questions after averaging their seeds, preserving pairing."""
    indexed = {(r['question_id'], r['seed'], r['arm']): r for r in records}
    questions = sorted({r['question_id'] for r in records})
    seeds = sorted({r['seed'] for r in records})
    assert len(indexed) == len(records) == len(questions)*len(seeds)*len(ARMS)
    assert set(indexed) == {(q, s, a) for q in questions for s in seeds for a in ARMS}
    baseline = [r for r in records if r['arm'] == 'direct']
    baseline_seconds = statistics.fmean(r['seconds'] for r in baseline)
    baseline_tokens = statistics.fmean(r['generated_tokens'] for r in baseline)
    assert baseline_seconds > 0 and baseline_tokens > 0
    summary = []
    for arm in ARMS:
        group = [r for r in records if r['arm'] == arm]
        qf1 = [statistics.fmean(indexed[q, s, arm]['f1'] for s in seeds) for q in questions]
        s = {'arm': arm, 'n': len(group), 'f1': statistics.fmean(qf1),
             'f1_question_bootstrap_95': bootstrap(qf1)['bootstrap_95'],
             'exact_match_percent': 100*statistics.fmean(r['exact_match'] for r in group),
             'mean_generated_tokens': statistics.fmean(r['generated_tokens'] for r in group),
             'mean_search_tokens': statistics.fmean(r['search_tokens'] for r in group),
             'mean_answer_tokens': statistics.fmean(r['answer_tokens'] for r in group),
             'mean_seconds': statistics.fmean(r['seconds'] for r in group),
             'mean_retrieval_seconds': statistics.fmean(r['retrieval_seconds'] for r in group),
             'mean_selected_injected_tokens': statistics.fmean(r['selected_injected_tokens'] for r in group),
             'mean_retrieval_calls': statistics.fmean(r['retrieval_calls'] for r in group),
             'mean_native_prefill_tokens': statistics.fmean(r['native_prefill_tokens'] for r in group),
             'empty': sum(r['empty'] for r in group), 'truncated': sum(r['truncated'] for r in group),
             'thinking_eos': sum(r['thinking_eos'] for r in group),
             'forced_thinking_end': sum(r['forced_thinking_end'] for r in group)}
        s['tokens_relative_direct'] = s['mean_generated_tokens']/baseline_tokens
        s['latency_relative_direct'] = s['mean_seconds']/baseline_seconds
        s['paired_latency_geomean_relative_direct'] = math.exp(statistics.fmean(
            math.log(indexed[q, seed, arm]['seconds']/indexed[q, seed, 'direct']['seconds'])
            for q in questions for seed in seeds))
        summary.append(s)
    pairs = []
    for left, right in PAIRS:
        deltas = [statistics.fmean(indexed[q, s, left]['f1']-indexed[q, s, right]['f1'] for s in seeds) for q in questions]
        pairs.append({'left': left, 'right': right, **bootstrap(deltas),
            'question_deltas': dict(zip(questions, deltas, strict=True)),
            'wins_ties_losses': [sum(v > 0 for v in deltas), sum(v == 0 for v in deltas), sum(v < 0 for v in deltas)]})
    return {'questions': len(questions), 'seeds': seeds, 'measurements': len(records), 'arms': summary, 'paired_f1': pairs}


def summarize(snapshot, output, gold_path, frozen_path, protocol_path, stage):
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    gold = {r['id']: r['answer'] for r in read(gold_path)}
    records, audits, identities, setup, raw = [], [], [], [], {}
    for folder in sorted(snapshot.iterdir()):
        if not folder.is_dir() or not (folder/(stage+'.jsonl')).exists():
            continue
        audited = audit_snapshot(folder, stage, frozen_path, output/('audit-'+folder.name+'.json'))
        audits.append(audited)
        rows = read(folder/(stage+'.jsonl'))
        meta = next(r for r in rows if r['kind'] == 'metadata')
        engine = next(r for r in rows if r['kind'] == 'engine')
        assert tuple(meta['arms']) == ARMS and set(meta['seeds']) == set(protocol['seeds'] if stage == 'formal' else [20261112])
        identities.append({'worker': folder.name, 'source': meta['source'], 'settings_sha256': meta['settings_sha256'],
            'prepared_sha256': meta['prepared_sha256'], 'model_id': engine['model_id'],
            'raw_sha256': hashlib.sha256((folder/(stage+'.jsonl')).read_bytes()).hexdigest()})
        start = json.loads((folder/(stage+'-start.json')).read_text())
        finish = json.loads((folder/(stage+'-finish.json')).read_text())
        measures = [r for r in rows if r['kind'] == 'measurement']
        setup.append({'worker': folder.name, 'job_seconds': finish['finished']-start['started'],
            'pipeline_seconds': sum(r['seconds'] for r in measures), 'index_seconds': engine['fact_build_seconds'],
            'container_stopped': not finish['state']['Running']})
        for r in measures:
            answer = r['answer']
            graded = grade(answer['answer_text'], gold[r['question_id']])
            selected_events = [r['events'][i] for i in r['selected_event_ids']]
            records.append({'question_id': r['question_id'], 'seed': r['seed'], 'arm': r['arm'], 'worker': folder.name,
                'f1': 100*graded['f1'], 'exact_match': graded['exact_match'], 'answer_text': answer['answer_text'],
                'generated_tokens': r['total_generation_tokens'], 'search_tokens': r['search_work'],
                'answer_tokens': len(answer['answer_ids']), 'seconds': r['seconds'],
                'retrieval_seconds': r['stage_seconds'].get('static_retrieval_seconds', 0.)+r['stage_seconds']['materialization_seconds'],
                'retrieval_calls': len(r['events'])+len(r.get('static_retrievals', [])),
                'selected_injections': len(selected_events)+len(r.get('static_packets', [])),
                'selected_injected_tokens': sum(len(e['injected_ids']) for e in selected_events)+len(r.get('static_injected_ids', [])),
                'native_prefill_tokens': r['cost'].get('prefill_tokens', 0),
                'empty': answer['empty_answer'], 'truncated': answer['truncated'],
                'thinking_eos': answer['eos_inside_thinking'], 'forced_thinking_end': answer['forced_thinking_end']})
            raw[r['question_id'], r['seed'], r['arm']] = r
    expected_cases = set(protocol['cases'] if stage == 'formal' else protocol['smoke_cases'])
    assert {r['question_id'] for r in records} == expected_cases
    assert all(i['source'] == identities[0]['source'] and i['settings_sha256'] == identities[0]['settings_sha256']
               and i['model_id'] == identities[0]['model_id'] for i in identities)
    assert all(len({r['worker'] for r in records if r['question_id'] == q}) == 1 for q in expected_cases)
    # Static real/noise use identical question-only retrieved windows and packet lengths.
    exposures = []
    for q in sorted(expected_cases):
        for seed in sorted({r['seed'] for r in records}):
            real, noise = raw[q, seed, 'static_facts'], raw[q, seed, 'static_facts_noise']
            assert real['static_packets'] == noise['static_packets']
            assert len(real['static_injected_ids']) == len(noise['static_injected_ids']) == 512
            left, right = raw[q, seed, 'online_is'], raw[q, seed, 'online_is_noise']
            exposures.append({'question_id': q, 'seed': seed, 'relevant_selected_injections': len(left['selected_event_ids']),
                'noise_selected_injections': len(right['selected_event_ids']),
                'equal_selected_exposure': len(left['selected_event_ids']) == len(right['selected_event_ids'])})
    result = aggregate(records)
    result.update(stage=stage, audit={'passed': all(a['passed'] for a in audits), 'checks': sum(a['checks'] for a in audits)},
        identities=identities, setup=setup, total_generated_tokens=sum(r['generated_tokens'] for r in records),
        total_pipeline_hours=sum(r['seconds'] for r in records)/3600,
        total_device_hours=sum(s['job_seconds'] for s in setup)/3600,
        online_noise_equal_exposure_pairs=sum(e['equal_selected_exposure'] for e in exposures),
        online_noise_total_pairs=len(exposures))
    def write(name, data):
        (output/name).write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n', encoding='utf-8', newline='\n')
    write('summary.json', result)
    write('records.json', records)
    write('noise-exposure.json', exposures)
    with (output/'scores.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    lines = ['# 从起点 IS 在线知识注入：正式指标' if stage == 'formal' else '# TRAIN 完整性验证指标', '',
        f'{len(expected_cases)} 道固定问题 × {len(result["seeds"])} 个种子 × 6 组，完成 {len(records)} 次原生运行；全部纳入统计。',
        '模型、IS 算法与参数保持原设定。F1 为最终答案与参考答案的词级重叠 F1（0–100）。', '',
        '| 方法 | F1 | 平均总生成 token | token / 直接生成 | 平均耗时（秒） | 耗时 / 直接生成 | 空答案 | 截断 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['arms']:
        lines.append(f'| {LABELS[r["arm"]]} | {r["f1"]:.3f} | {r["mean_generated_tokens"]:.1f} | {r["tokens_relative_direct"]:.2f}× | {r["mean_seconds"]:.2f} | {r["latency_relative_direct"]:.2f}× | {r["empty"]} | {r["truncated"]} |')
    lines += ['', '## 配对差异', '', '| 比较 | F1 差值 | 95% 问题级 bootstrap 区间 |', '|---|---:|---:|']
    for p in result['paired_f1']:
        lo, hi = p['bootstrap_95']
        lines.append(f'| {LABELS[p["left"]]} − {LABELS[p["right"]]} | {p["difference"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
    grouped = {r['arm']: r for r in result['arms']}
    online, closed = grouped['online_is'], grouped['closed_is']
    lines += ['', '先对每道题的两个种子求平均，再对问题重采样 5,000 次；没有把同题种子当作独立问题。', '', '## 本批结果解读', '',
        f'在线事实组相对无检索 IS 的实际生成量为 {online["mean_generated_tokens"]/closed["mean_generated_tokens"]:.2f}×，'
        f'耗时为 {online["mean_seconds"]/closed["mean_seconds"]:.2f}×。这些是实测成本，不是配置上限的倍数。']
    for p in result['paired_f1'][:3]:
        lo, hi = p['bootstrap_95']
        conclusion = '区间支持这批题上的正向差异' if lo > 0 else '区间支持这批题上的负向差异' if hi < 0 else '区间包含 0，尚不能确认稳定差异'
        lines.append(f'相对{LABELS[p["right"]]}，在线事实组 F1 差值为 {p["difference"]:+.3f}，95% 区间 [{lo:+.3f}, {hi:+.3f}]；{conclusion}。')
    lines += ['',
        '## 统计口径与范围', '',
        '- 所有 IS 组：M=3、K=2、block=512、SC 温度 0.25，保留思考上限 8192、搜索生成预算 65536、答案上限 16384。直接生成是单条 8192-token 思考轨迹的参照。',
        '- 总生成量包含全部候选、被丢弃 rollout 和最终答案；没有额外检索代理或临时答案生成。注入资料作为输入成本，另记录实际注入量和原生 prefill 量。',
        '- 延迟包含检索、IS 搜索与答案生成；模型加载、索引初始化、预热和前缀缓存重置在单次计时之外。表中倍数为均值之比，summary.json 另提供配对耗时比的几何均值。',
        '- 每道题的所有条件在同一张卡上执行，每组开始前清空前缀缓存；原生 EOS、空答案和截断均保留，不重试补答案、不剔除低分。',
        '- 静态组注入 2×256 token；在线组在生成 128、512 token 后各触发一次，提前结束时可能只有一次或零次。噪声组逐包等长，但整条轨迹的注入总量不保证相等。',
        '- 静态检索只使用问题；在线检索还使用各候选的已有思考。因此在线／静态对比同时包含查询、注入时机和实际曝光量的变化，不能单独归因于注入位置。',
        f'- 在线事实／噪声的选中路径注入量完全相等：{result["online_noise_equal_exposure_pairs"]}/{len(exposures)} 对，详见 noise-exposure.json。',
        '- 本轮是此前使用的固定 16 题开发集复测，不是新的独立测试集；区间与结论仅描述这批题。词重叠 F1 也不等同于人工语义正确率。',
        f'- 完整轨迹审计通过 {result["audit"]["checks"]} 项检查；累计生成 {result["total_generated_tokens"]:,} token，测量流程 {result["total_pipeline_hours"]:.3f} 小时，分配设备 {result["total_device_hours"]:.3f} 小时；自身容器均已停止。', '']
    (output/'RESULTS_ZH.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot', 'output', 'gold', 'frozen', 'protocol'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--stage', choices=['smoke', 'formal'], default='formal')
    a = p.parse_args()
    r = summarize(a.snapshot, a.output, a.gold, a.frozen, a.protocol, a.stage)
    print(json.dumps({k: r[k] for k in ('questions', 'measurements', 'audit', 'total_generated_tokens')}))
