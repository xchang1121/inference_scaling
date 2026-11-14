from pathlib import Path
import json

root=Path('/data/disk/changxiangyu/agentic_scaling/results')
for worker in globals().get('workers',['a3']):
    out=root/('online-injection-full-is-metrics-20261113-'+worker)
    if out.exists() and not (out/'smoke.pid').exists():
        print(json.dumps({'worker':worker,'preparation':{p.name:p.stat().st_size for p in out.glob('prepared-*.json')},'logs':{p.name:p.read_text(errors='replace')[-1200:] for p in out.glob('prepare-*.log')}}))
    for stage in ['smoke','formal','diagnostic']:
        if not (out/(stage+'.pid')).exists():continue
        path=out/(stage+'.jsonl')
        counts={};last=None;empty=trunc=0;last_stats=None
        if path.exists():
            with path.open() as f:
                for line in f:
                    try:r=json.loads(line)
                    except ValueError:continue
                    k=r['kind'];counts[k]=counts.get(k,0)+1
                    if k=='diagnosis':
                        last=r['question_id']+':'+r['variant'];last_stats={'tokens':len(r['token_ids']),'first':r['token_ids'][:1],'seconds':r['seconds']}
                    if k=='measurement':
                        last=r['unit_id'];empty+=sum(x['empty_answer'] for x in [r['answer']]);trunc+=sum(x['truncated'] for x in [r['answer']])
                        last_stats={'seconds':round(r['seconds'],1),'search_tokens':r['search_work'],'answer_lengths':[len(x['answer_ids']) for x in [r['answer']]],
                            'candidate_thinking_lengths':[len(x) for x in r['candidates']],'forced_thinking_ends':sum(x['forced_thinking_end'] for x in [r['answer']])}
        log=out/(stage+'.log');launch=out/(stage+'-launch.log');exitpath=out/(stage+'.exit')
        failed=exitpath.exists() and exitpath.read_text().strip()!='0'
        print(json.dumps({'worker':worker,'stage':stage,'counts':counts,'last':last,'last_stats':last_stats,'empty_candidates':empty,'truncated_candidates':trunc,
            'exit':exitpath.read_text().strip() if exitpath.exists() else None,
            'log_tail':log.read_text(errors='replace')[-1700 if failed else -300:] if log.exists() else '',
            'launch_tail':launch.read_text(errors='replace')[-800:] if launch.exists() else ''}))
