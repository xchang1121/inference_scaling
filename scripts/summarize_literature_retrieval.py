"""Offline F1, paired latency, generation accounting, and frozen-IS replay audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from types import SimpleNamespace

from summarize_cot_guidance import Decoder, read
from summarize_flare_gap import bootstrap, grade
from inference_scaling.retrieval.ablation import static_prompt
from inference_scaling.retrieval.cot_guidance import GuidanceConfig, GuidanceLedger
from inference_scaling.retrieval.literature import unrelated
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation, select_metropolized, weigh

EOS,CLOSE=248046,248069
BACKEND=SimpleNamespace(model_id='offline',tokenizer=SimpleNamespace(eos_token_id=EOS))


def injection_stops(row):
    """Selected-path stops at actual insertion boundaries, distinct from IS roots."""
    h,arm,probe=row['handoff'],row['arm'],row['probe']
    suffix=row['candidates'][0]
    firsts=[];after_last=None
    if h['injected_ids'] and arm.startswith(('static_','ircot')):
        firsts=[probe['calls'][0]['token_ids'][:1]] if probe and probe['calls'] else [suffix[:1]]
        reason=(probe['reason_ids'] if probe and not row['audit'].get('terminal_checkpoint') else [])+suffix
        after_last=next((i for i,t in enumerate(reason) if t in {EOS,CLOSE}),len(reason))
    elif h['injected_ids']:
        events=h['events']
        for i,event in enumerate(events):
            if event['kind']!='injection':continue
            subsequent=next((e for e in events[i+1:] if e['kind']=='thought'),None)
            firsts.append(subsequent['token_ids'][:1] if subsequent else suffix[:1])
            after_last=(subsequent['retained_tokens'] if subsequent else
                next((j for j,t in enumerate(suffix) if t in {EOS,CLOSE}),len(suffix)))
    return {'injection_episodes':len(firsts),'post_injection_immediate_close_events':sum(x==[CLOSE] for x in firsts),
        'post_injection_immediate_eos_events':sum(x==[EOS] for x in firsts),
        'thinking_after_last_injection':after_last}


def audit_row(row,cfg,decode,noise,check):
    uid=row['unit_id'];arm=row['arm'];h=row['handoff'];probe=row['probe'];audit=row['audit']
    def ck(ok,label):check(ok,uid+':'+label)
    calls=h['controller_calls']+h['interim_calls']+(probe['calls'] if probe else [])+row['calls']
    controller_tokens=sum(len(c['token_ids']) for c in h['controller_calls']+h['interim_calls'])
    probe_tokens=probe['work'] if probe else 0
    search_tokens=sum(len(c['token_ids']) for c in row['calls'])
    answer=row['answer'];answer_tokens=len(answer['answer_ids'])
    ck(controller_tokens==h['controller_work'],'controller_cost')
    ck(controller_tokens+probe_tokens==row['presearch_work'],'presearch_cost')
    ck(search_tokens==row['search_work'],'search_cost')
    ck(row['total_generation_tokens']==controller_tokens+probe_tokens+search_tokens+answer_tokens,'total_cost')
    ck(controller_tokens+probe_tokens+search_tokens<=cfg.search_tokens,'same_work_budget')
    ck(answer_tokens<=cfg.answer_tokens,'answer_cap')
    ck(row['total_generation_tokens']<=cfg.search_tokens+cfg.answer_tokens,'total_budget')
    ck(row['cost'].get('generated_tokens')==row['total_generation_tokens'],'native_token_counter')
    ck(all(row['cost'].get(k,0)==0 for k in ('scored_tokens','cancelled_requests','preemptions')),'no_hidden_scoring_or_cancellation')
    ck(sum(row['stage_seconds'].values())<=row['seconds']+1e-6,'timing_components')
    for i,c in enumerate(calls):
        ck(token_hash(c['prefix_ids'])==c['prefix_sha256'],f'call{i}:prefix_hash')
        ck(len(c['token_ids'])<=c['max_new_tokens'],f'call{i}:length')
        ck(len(c['prefix_ids'])+len(c['token_ids'])<=cfg.context_tokens,f'call{i}:context')
        ck(EOS not in c['token_ids'] or c['token_ids'][-1]==EOS,f'call{i}:eos_last')
        if 'sc' in c:
            ck(all(len(c[k])==len(c['token_ids']) and all(math.isfinite(v) for v in c[k]) for k in ('sc','cn','lp')),f'call{i}:stats')
            ck(CLOSE not in c['token_ids'] or c['token_ids'][-1]==CLOSE,f'call{i}:close_last')
    ck(len({c['request_id'] for c in h['controller_calls']})==len(h['controller_calls']),'unique_controller_requests')
    for p in h['packets']:
        ck(token_hash(p['ids'])==p['sha256'],'packet_hash')
        ck(not {EOS,CLOSE,248068}.intersection(p['ids']),'packet_no_structure')
    if arm in {'direct','closed_is'}:
        ck(h['root_ids']==row['base_ids'] and not h['injected_ids'] and not h['events'],'fully_closed')
    elif arm.startswith(('static_','ircot')):
        payload=[t for p in h['packets'] for t in (unrelated(p,noise) if arm.endswith('_noise') else p['ids'])]
        expected=static_prompt(tuple(row['base_ids']),tuple(payload),EOS,()) if payload else tuple(row['base_ids'])
        ck(tuple(h['root_ids'])==expected,'fresh_reader_exact_references')
        ck(h['carried']=={'sc':[],'cn':[],'lp':[]},'no_agent_cot_in_reader')
        ck(payload==[t for p in h['injected_ids'] for t in p],'packet_variant')
    else:
        current=list(row['base_ids']);carried={k:[] for k in ('sc','cn','lp')}
        thought_calls=iter(h['controller_calls']);answer_calls=iter(h['interim_calls'])
        interim_count=0
        for event in h['events']:
            if event['kind']=='thought':
                c=next(thought_calls)
                ck(c['prefix_ids']==current==event['root_before'],'preserved_thinking_prefix')
                ck(c['token_ids']==event['token_ids'],'thought_event_tokens')
                n=next((i for i,t in enumerate(c['token_ids']) if t in {EOS,CLOSE}),len(c['token_ids']))
                ck(n==event['retained_tokens'],'retained_thought_length')
                for k in carried:carried[k].extend(c[k][:n])
                current+=c['token_ids'][:n]
            elif event['kind']=='retrieval':
                c=next(answer_calls);interim_count+=1
                ck(c['prefix_ids'][:len(current)]==current and c['prefix_ids'][len(current)]==CLOSE,'interim_answer_prefix')
                ck(decode(c['token_ids'],True).strip()==c['text'],'interim_text')
                ck(c['text'] in event['query'],'interim_in_retrieval_query')
            elif event['kind']=='injection':
                current+=event['ids'];ck(token_hash(current)==event['root_after_sha256'],'append_only_injection')
        ck(current==h['root_ids'],'controller_handoff_prefix')
        ck(carried==h['carried'],'reward_excludes_injected_tokens_and_interim_answer')
        ck(sum(len(c['token_ids']) for c in h['interim_calls'])<=512*interim_count,'interim_cap')
        ck(len(h['carried']['sc'])==len(h['retained_reason_ids']),'reason_statistics_alignment')
        ck(h['terminal_eos']==any(e.get('trigger')=='eos' for e in h['events']),'terminal_eos_flag')
        if arm=='rethink':ck(not h['packets'] and not h['interim_calls'],'rethink_has_no_retrieval')
    if probe:
        ck(probe['probe_root_ids']==h['root_ids'],'checkpoint_root')
        ck(sum(len(c['token_ids']) for c in probe['calls'])==probe['work'],'checkpoint_cost')
        pl=GuidanceLedger(BACKEND,None,probe['probe_root_ids'],(CLOSE,),0,cfg)
        pl.calls=probe['calls']
        ck(pl.statistics(probe['reason_ids'])==probe['statistics'],'checkpoint_statistics')
        ck(row['carried']==probe['statistics'],'checkpoint_score_carried')
        expected=list(h['root_ids'])+([] if audit.get('terminal_checkpoint') else probe['reason_ids'])
        ck(row['root_ids']==expected,'checkpoint_preserved')
    else:ck(row['root_ids']==h['root_ids'],'handoff_preserved')
    seed=SeedStream(SeedStream(row['seed']).derive('question',row['question_id'])).derive('search')
    ledger=GuidanceLedger(BACKEND,None,row['root_ids'],(CLOSE,),seed,cfg,carried=row['carried'])
    ledger.calls=row['calls']
    for c in row['calls']:ck(c['prefix_ids'][:len(row['root_ids'])]==row['root_ids'],'immutable_search_epoch')
    if audit.get('steps'):
        config=audit['config']
        expected={'candidate_count':3,'rollout_count':2,'block_size':512,'reward_temperature':.25,
            'block_first':True,'exploration_schedule':'fixed','selection_ready':False,'selection':'block_metropolized'}
        ck(all(config[k]==v for k,v in expected.items()),'frozen_IS_parameters')
        ck(config['total_length']==cfg.thinking_tokens-len(row['carried']['sc']),'frozen_IS_horizon')
        kept=();spent=row['presearch_work']
        for si,record in enumerate(audit['steps']):
            fixed=kept[:record['generated_length_before']]
            remaining=config['total_length']-len(fixed);width=min(config['block_size'],remaining);carried=int(bool(kept))
            bound=(3-carried)*(width+2*(remaining-width))+carried*(remaining-width)
            ck(bound==record['reserved'],'complete_pool_reservation')
            ck(spent+bound<=cfg.search_tokens,'pool_afforded_before_draw')
            ck(sum(len(c['token_ids']) for c in row['calls'] if c['request_id'].startswith(f'conditional-is:step:{si}:'))==record['work'],'pool_actual_cost')
            blocks,leaves=[],[]
            for candidate in record['candidates']:
                block=tuple(candidate['token_ids']);group=[]
                for leaf in candidate['rollouts']:
                    suffix=tuple(leaf['token_ids']);score,=ledger.scores([fixed+block+suffix],'sc')
                    ck(math.isclose(score,leaf['reward'],abs_tol=1e-10),'SC_reward_replay')
                    ck(math.isclose(score/.25,leaf['log_weight'],abs_tol=1e-10),'SC_log_weight_replay')
                    group.append(RolloutEvaluation(suffix,leaf['reward'],leaf['log_weight']))
                blocks.append(block);leaves.append(group)
            candidates=weigh(blocks,leaves)
            mi,ki=select_metropolized(candidates,SeedStream(seed),'conditional_is',si,group_blocks=True)
            ck((mi,ki)==(record['selected_index'],record['completion_index']),'IS_selection_replay')
            ck(all(math.isclose(c.log_weight,r['log_weight'],abs_tol=1e-10) for c,r in zip(candidates,record['candidates'],strict=True)),'IS_group_weights')
            kept=fixed+candidates[mi].token_ids+candidates[mi].rollouts[ki].token_ids
            spent+=record['work']
        ck(kept==tuple(row['candidates'][0]),'selected_trajectory_replay')
        ck(len(kept)+len(row['carried']['sc'])<=cfg.thinking_tokens,'retained_thinking_cap')
    suffix=row['candidates'][0]
    ck(decode(answer['answer_ids'],True).strip()==answer['answer_text'],'decoded_final_answer')
    ck([t for c in answer['calls'] for t in c['token_ids']]==answer['answer_ids'],'answer_call_concatenation')
    ck(answer['empty_answer']==(not answer['answer_text'].strip()),'empty_answer_flag')
    ck(not answer['truncated'],'no_final_truncation')
    ck(answer['eos_inside_thinking']==(EOS in suffix),'selected_EOS_flag')
    prefix=list(row['root_ids'])+suffix+([] if suffix[-1:]==[CLOSE] else [CLOSE])
    for c in answer['calls']:
        ck(token_hash(prefix)==c['prefix_sha256'],'final_answer_prefix')
        ck(len(prefix)+len(c['token_ids'])<=cfg.context_tokens,'answer_context_cap')
        prefix+=c['token_ids']
    if answer['eos_inside_thinking']:ck(not answer['calls'] and not answer['answer_ids'],'EOS_never_repaired')
    retrievals=[e for e in h['events'] if e['kind']=='retrieval']
    think_calls=h['controller_calls']+(probe['calls'] if probe else [])+row['calls']
    requested_prefix_tokens=sum(len(c['prefix_ids']) for c in calls)
    current_len=len(row['root_ids'])+len(suffix)+(0 if suffix[-1:]==[CLOSE] else 1)
    for c in answer['calls']:
        requested_prefix_tokens+=current_len;current_len+=len(c['token_ids'])
    return {'unit_id':uid,'question_id':row['question_id'],'seed':row['seed'],'arm':arm,'worker':row['worker'],
        'answer_text':answer['answer_text'],'generation_tokens':row['total_generation_tokens'],
        'controller_tokens':controller_tokens,'probe_tokens':probe_tokens,'search_tokens':search_tokens,
        'answer_tokens':answer_tokens,'injected_tokens':sum(map(len,h['injected_ids'])),
        'requested_prefix_tokens':requested_prefix_tokens,'retrieval_calls':len(retrievals),
        'retrieval_seconds':sum(e['seconds'] for e in retrievals),'seconds':row['seconds'],
        'native_prefill_tokens':row['cost'].get('prefill_tokens',0),
        'engine_requests':row['cost'].get('engine_requests',0),
        'encoder_queries_truncated':sum(e.get('encoder_query_tokens',0)>512 for e in retrievals),
        **row['stage_seconds'],'empty':answer['empty_answer'],'truncated':answer['truncated'],
        'thinking_eos':answer['eos_inside_thinking'],'forced_thinking_end':answer['forced_thinking_end'],
        'raw_thinking_eos_calls':sum(EOS in c['token_ids'] for c in think_calls),
        'raw_thought_close_calls':sum(CLOSE in c['token_ids'] for c in think_calls),
        'intercepted_thought_closes':sum(e.get('trigger')=='natural_close' for e in h['events']),
        'controller_periodic_triggers':sum(e.get('trigger')=='periodic' for e in h['events']),
        'interim_length_caps':sum(len(c['token_ids'])>=c['max_new_tokens'] and c['finish_reason']=='length' for c in h['interim_calls']),
        'selected_immediate_close':suffix[:1]==[CLOSE],'selected_immediate_eos':suffix[:1]==[EOS],
        'search_budget_stops':audit.get('stop')=='whole_pool_budget',**injection_stops(row),
        'insertion_count':len(h['injected_ids']),
        'forward_token_slots':row.get('graph_execution',{}).get('forward_token_slots',0),
        'max_search_call_tokens':max((len(c['token_ids']) for c in row['calls']),default=0)}


def summarize(snapshot,output,gold_path,stage,frozen_path,manifest_path):
    output.mkdir(parents=True,exist_ok=True)
    gold_rows={r['id']:r for r in read(gold_path)}
    gold={q:r['answer'] for q,r in gold_rows.items()}
    frozen=json.loads(frozen_path.read_text());manifest=json.loads(manifest_path.read_text())
    expected_cases=set(manifest['smoke_ids'] if stage=='smoke' else manifest['fresh_ids'])
    expected_seeds={20261112} if stage=='smoke' else set(manifest['seeds'])
    failures=[];checks=0;all_rows=[];native_rows={};metadata=[];pair_notes=[];setup=[]
    def check(ok,name):
        nonlocal checks
        checks+=1
        if not ok:failures.append(name)
    for folder in sorted(snapshot.iterdir()):
        if not folder.is_dir():continue
        path=folder/(stage+'.jsonl')
        if not path.exists():path=folder/(stage+'.jsonl.gz')
        if not path.exists():continue
        rows=read(path);meta=next(r for r in rows if r['kind']=='metadata');cfg=GuidanceConfig(**meta['config'])
        metadata.append(meta)
        for engine in (r for r in rows if r['kind']=='engine'):
            bank=engine['retrieval']['examples']
            setup.append({'worker':meta.get('worker',folder.name),'fact_index_build_seconds':engine['fact_build_seconds'],
                'example_index_build_seconds':bank['build_seconds'],'example_bank_entries':bank['entries'],
                'encoder_model':bank['encoder']['model'],'encoder_revision':bank['encoder']['revision']})
        check(meta['config']==GuidanceConfig().__dict__,'unchanged_guidance_config')
        check(all(meta['source'].get(n)==h for n,h in frozen['files'].items()),'frozen_reader_source_hashes')
        check(set(meta['seeds'])==expected_seeds,'expected_seeds')
        check(not any(r['kind']=='failed' for r in rows),'native_no_failure')
        measures=[r for r in rows if r['kind']=='measurement'];done=[r for r in rows if r['kind']=='completed']
        check(len(done)==1 and done[0]['units']==len(measures)==len(meta['cases'])*len(meta['seeds'])*len(meta['arms']),'native_complete')
        prepared_path=folder/('prepared-'+stage+'.json');prepared=json.loads(prepared_path.read_text(encoding='utf-8'))
        check(hashlib.sha256(prepared_path.read_bytes()).hexdigest()==meta['prepared_sha256'],'prepared_hash')
        decode=Decoder(json.loads((folder/('vocabulary-'+stage+'.json')).read_text(encoding='utf-8')))
        bases={c['id']:c['prompt_ids'] for c in prepared['cases']}
        for r in measures:
            check(r['base_ids']==bases[r['question_id']],r['unit_id']+':original_prompt')
            try:s=audit_row(r,cfg,decode,prepared['noise_ids'],check)
            except Exception as e:
                check(False,r['unit_id']+':audit_exception:'+repr(e));continue
            s['f1']=100*grade(s['answer_text'],gold[s['question_id']])['f1']
            fact_arm=s['arm'].startswith(('static_facts','ircot'))
            gold_docs=set(gold_rows[s['question_id']].get('document_ids',[]))
            retrieved=[p['id'].split('@',1)[0] for p in r['handoff']['packets']] if fact_arm else []
            s['retrieved_gold_document']=bool(gold_docs.intersection(retrieved)) if fact_arm else None
            s['dynamic_query_added_gold_document']=(bool(gold_docs.intersection(retrieved[1:]))
                and not bool(gold_docs.intersection(retrieved[:1]))) if s['arm'].startswith('ircot') else None
            all_rows.append(s)
            check(r['unit_id'] not in native_rows,'unique_unit')
            native_rows[r['unit_id']]=r
    check({r['question_id'] for r in all_rows}==expected_cases,'exact_same_question_cohort')
    arms=metadata[0]['arms'] if metadata else []
    expected={(q,s,a) for q in expected_cases for s in expected_seeds for a in arms}
    keyed={(r['question_id'],r['seed'],r['arm']):r for r in all_rows}
    check(set(keyed)==expected,'complete_paired_design')
    for q in sorted(expected_cases):
        for s in sorted(expected_seeds):
            for left,right in [('static_facts','static_facts_noise'),('static_examples','static_examples_noise'),('ircot','ircot_noise'),('thinkretrieve','thinkretrieve_noise')]:
                if (q,s,left) not in keyed or (q,s,right) not in keyed:continue
                a=native_rows[keyed[q,s,left]['unit_id']];b=native_rows[keyed[q,s,right]['unit_id']]
                lengths_a=list(map(len,a['handoff']['injected_ids']));lengths_b=list(map(len,b['handoff']['injected_ids']))
                matched=lengths_a==lengths_b
                pair_notes.append({'question_id':q,'seed':s,'left':left,'right':right,'exact_packet_lengths':matched,
                    'relevant_lengths':lengths_a,'noise_lengths':lengths_b})
                if left!='thinkretrieve':check(matched,f'{q}:{s}:{left}:equal_noise_length')
                if left=='ircot':check(a['handoff']['controller_calls']==b['handoff']['controller_calls'],f'{q}:{s}:matched_IRCoT_agent')
                if left=='thinkretrieve':
                    check(matched or b['handoff']['terminal_eos'],f'{q}:{s}:TR_mismatch_requires_observed_terminal_EOS')
    for r in all_rows:
        baseline=keyed.get((r['question_id'],r['seed'],'direct'))
        check(baseline is not None,'direct_baseline_present')
        if baseline is None:continue
        check(r['worker']==baseline['worker'],'latency_same_worker')
        r['latency_ratio_vs_direct']=r['seconds']/baseline['seconds']
        r['generation_ratio_vs_direct']=r['generation_tokens']/max(1,baseline['generation_tokens'])
    audit={'passed':not failures,'checks':checks,'failures':failures,'measurements':len(all_rows),
        'final_truncated':sum(r['truncated'] for r in all_rows),'noise_pair_count':len(pair_notes),
        'noise_length_mismatches':[p for p in pair_notes if not p['exact_packet_lengths']]}
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    if failures:raise RuntimeError(str(failures[:20]))
    groups={a:[r for r in all_rows if r['arm']==a] for a in arms}
    mean=lambda rows,k:statistics.fmean(r[k] for r in rows)
    baseline_seconds=mean(groups['direct'],'seconds')
    summary=[]
    numeric=['generation_tokens','controller_tokens','probe_tokens','search_tokens','answer_tokens','injected_tokens',
        'requested_prefix_tokens','native_prefill_tokens','engine_requests','retrieval_calls','retrieval_seconds','seconds','controller_seconds','search_seconds','answer_seconds','forward_token_slots']
    counts=['empty','truncated','thinking_eos','forced_thinking_end','raw_thinking_eos_calls','raw_thought_close_calls',
        'intercepted_thought_closes','controller_periodic_triggers','interim_length_caps','selected_immediate_close','selected_immediate_eos','encoder_queries_truncated',
        'search_budget_stops','injection_episodes','post_injection_immediate_close_events','post_injection_immediate_eos_events']
    for arm,rows in groups.items():
        summary.append({'arm':arm,'n':len(rows),'f1':mean(rows,'f1'),
            **{'mean_'+k:mean(rows,k) for k in numeric},**{k:sum(r[k] for r in rows) for k in counts},
            'latency_ratio_of_means':mean(rows,'seconds')/baseline_seconds,
            'geomean_paired_latency_ratio':math.exp(statistics.fmean(math.log(r['latency_ratio_vs_direct']) for r in rows)),
            'mean_thinking_after_last_injection':statistics.fmean(r['thinking_after_last_injection'] for r in rows
                if r['thinking_after_last_injection'] is not None) if any(r['thinking_after_last_injection'] is not None for r in rows) else None,
            'retrieved_gold_document_count':sum(bool(r['retrieved_gold_document']) for r in rows) if arm.startswith(('static_facts','ircot')) else None,
            'dynamic_query_added_gold_document_count':sum(bool(r['dynamic_query_added_gold_document']) for r in rows) if arm.startswith('ircot') else None,
            'max_answer_tokens':max(r['answer_tokens'] for r in rows)})
    qmeans={a:{q:mean([r for r in rows if r['question_id']==q],'f1') for q in expected_cases} for a,rows in groups.items()}
    comparisons=[(a,'closed_is') for a in arms if a not in {'direct','closed_is'}]
    comparisons += [('closed_is','direct'),('ircot','static_facts'),('thinkretrieve','static_examples'),('thinkretrieve','rethink')]
    comparisons += [('static_facts','static_facts_noise'),('ircot','ircot_noise'),('static_examples','static_examples_noise'),('thinkretrieve','thinkretrieve_noise')]
    contrasts=[{'left':a,'right':b,**bootstrap(qmeans[a][q]-qmeans[b][q] for q in sorted(expected_cases))} for a,b in comparisons]
    result={'stage':stage,'questions':len(expected_cases),'seeds':sorted(expected_seeds),'measurements':len(all_rows),
        'summary':summary,'contrasts':contrasts,'audit':audit,'noise_pairs':pair_notes,'setup_costs':setup}
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (output/'answers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows),encoding='utf-8')
    with (output/'scores.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows(all_rows)
    text=['# IRCoT / ThinkRetrieve: frozen IS reader','',f'{len(expected_cases)} questions, {len(expected_seeds)} seeds, {len(all_rows)} measurements.',
        '', 'F1 is word overlap on a 0–100 scale. Equal total generation allowances; actual costs differ. All controller and discarded candidate generation is included. Direct means one ordinary thinking trajectory, with no retrieval or IS. Every latency is measured from a cold prefix cache; initial model/index loading and warm-up are excluded.',
        '', '| Arm | F1 | Generated tokens | Retrieval seconds | End-to-end seconds | Latency / direct | Paired geometric ratio | Empty | Truncated |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"| {r['arm']} | {r['f1']:.3f} | {r['mean_generation_tokens']:.1f} | {r['mean_retrieval_seconds']:.3f} | {r['mean_seconds']:.2f} | {r['latency_ratio_of_means']:.3f} | {r['geomean_paired_latency_ratio']:.3f} | {r['empty']} | {r['truncated']} |")
    text+=['','## Generation and stop diagnostics','','| Arm | Controller tokens | Checkpoint tokens | IS/direct tokens | Answer tokens | Input injection tokens | Thought EOS outputs | Intercepted close | Immediate IS suffix close | Forced close |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"| {r['arm']} | {r['mean_controller_tokens']:.1f} | {r['mean_probe_tokens']:.1f} | {r['mean_search_tokens']:.1f} | {r['mean_answer_tokens']:.1f} | {r['mean_injected_tokens']:.1f} | {r['thinking_eos']} | {r['intercepted_thought_closes']} | {r['selected_immediate_close']} | {r['forced_thinking_end']} |")
    text+=['','Immediate stops below are measured at actual insertion boundaries on the selected path. Several static packets form one insertion episode; ThinkRetrieve counts each sequential insertion separately. These differ from immediately closing an IS suffix after a checkpoint.',
        '', '| Arm | Insertion episodes | Immediately close thinking | Immediate EOS | Mean thought tokens after last insertion |',
        '|---|---:|---:|---:|---:|']
    for r in summary:
        after=r['mean_thinking_after_last_injection'];shown='—' if after is None else f'{after:.1f}'
        text.append(f"| {r['arm']} | {r['injection_episodes']} | {r['post_injection_immediate_close_events']} | {r['post_injection_immediate_eos_events']} | {shown} |")
    text+=['','## Paired F1 differences','','Seeds are averaged within question; intervals are a descriptive 5,000-draw question-cluster bootstrap, without multiplicity adjustment.','']
    for c in contrasts:text.append(f"- {c['left']} − {c['right']}: {c['difference']:+.3f}, 95% CI [{c['bootstrap_95'][0]:+.3f}, {c['bootstrap_95'][1]:+.3f}].")
    text+=['',f"Audit: {checks} checks passed. Exact-length noise pairs: {len(pair_notes)-len(audit['noise_length_mismatches'])}/{len(pair_notes)}.",
        '', 'IRCoT uses a separate fresh reader; its noise control corrupts reader evidence only. ThinkRetrieve retains and extends its trace; its noise control follows its own evolving trace with matched packet lengths. Their F1/cost differences therefore compare complete adapted pipelines, not an isolated placement change.']
    for r in setup:text+=['',f"One-time index construction on {r['worker']}: facts {r['fact_index_build_seconds']:.2f}s; {r['example_bank_entries']} complete worked examples {r['example_index_build_seconds']:.2f}s. These startup costs are excluded from per-question latency. Encoder: {r['encoder_model']} at {r['encoder_revision']}."]
    text+=['','## Offline retrieval diagnostic','','Gold document IDs are used only here, after generation. Retrieving the right source document does not guarantee that the selected short window contains the answer. Noise arms retain retrieval metadata but do not expose the original contents.','']
    for r in summary:
        if r['arm'] in {'static_facts','ircot'}:
            text.append(f"- {r['arm']}: retrieved a gold source document in {r['retrieved_gold_document_count']}/{r['n']} runs; dynamic-query additions: {r['dynamic_query_added_gold_document_count']}.")
    (output/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'stage':stage,'questions':len(expected_cases),'measurements':len(all_rows),'audit':audit}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('snapshot','output','gold','frozen','manifest'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stage',choices=['smoke','formal'],required=True)
    a=p.parse_args();summarize(a.snapshot,a.output,a.gold,a.stage,a.frozen,a.manifest)
