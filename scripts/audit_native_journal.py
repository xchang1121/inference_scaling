"""Reconcile append-only native outputs with complete per-unit measurements."""
from collections import defaultdict


def audit_native_journal(journal, measurements, check):
    starts, outputs, totals = {}, {}, defaultdict(int)
    def ck(ok,label):
        check(ok,'native_journal:'+label)
    for record in journal:
        key = record['batch']
        if record['kind'] == 'native_start':
            ck(key not in starts,'unique_start')
            starts[key] = record
        elif record['kind'] == 'native_done':
            ck(key in starts and key not in outputs,'one_completion_per_start')
            start = starts[key]
            ck(start['unit'] == record['unit'],'same_unit')
            ck(len(start['requests']) == len(record['samples']),'complete_batch')
            for request,sample in zip(start['requests'],record['samples'],strict=True):
                ck(request['prefix'] == sample['prefix'] and request['request_id'] == sample['request_id'],'sample_identity')
                totals[record['unit']] += len(sample['token_ids'])
            outputs[key] = record
        else:
            ck(False,'no_failure_in_completed_job')
    ck(set(starts) == set(outputs),'all_native_batches_completed')
    expected_units = {':'.join(map(str,(r['question_id'],r['seed'],r['config_id']))):r for r in measurements}
    ck(set(totals) == set(expected_units)|{'warmup'},'exact_unit_coverage')
    for unit,row in expected_units.items():
        ck(totals[unit] == row['total_generation_tokens'] == row['cost']['generated_tokens'],'complete_unit_generation_cost')
        recorded = {}
        for key,done in outputs.items():
            if done['unit'] != unit:
                continue
            for request,sample in zip(starts[key]['requests'],done['samples'],strict=True):
                name = request['request_id']
                ck(name not in recorded,'unique_request_per_unit')
                recorded[name] = (request,sample,done['statistics'].get(name))
        for call in row['native_calls']+row.get('auxiliary_calls',[]):
            name = call.get('request_id') or call['request']['request_id']
            ck(name in recorded,'measurement_request_exists')
            if name not in recorded:
                continue
            request,sample,statistics = recorded[name]
            ck(call['prefix_ids'] == request['prefix'] and call['token_ids'] == sample['token_ids'],'measurement_tokens_match_journal')
            ck(call['lp'] == sample['token_logprobs'] and statistics == [call['sc'],call['cn']],'measurement_statistics_match_journal')
