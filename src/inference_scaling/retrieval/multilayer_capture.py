"""Explicit diagnostic schema; multi-layer policies require separate validation."""

SCHEMA = {
    'version': 1, 'block_dimensions': 128,
    'blocks': [{'source': 'input_layernorm_output', 'layer': n} for n in (16, 32, 48)]
        + [{'source': 'final_normalized_pre_lm_head'}],
    'representation_dimensions': 512, 'alignment_diagnostic_index': 512,
    'projection_seed': 2026111502, 'state_offset': -1,
    'same_projection_for_each_real_block': True,
}


def recapture_batches(plan):
    """Bound one-token recapture work and preserve explicit repeat/layout identity."""
    from inference_scaling.retrieval.stepwise import token_hash
    items = plan['items']
    if plan['kind'] == 'multilayer_prefix_recapture_pilot':
        valid = plan['version'] in (1, 2) and len(items) == 4
        size = 4
    else:
        valid = (plan['kind'] == 'multilayer_prefix_recapture_pool' and plan['version'] in (3,4)
                 and plan.get('pool') == 32 and plan.get('batch_size') == 4
                 and (len(items) == 64 if plan['version'] == 3 else 64 <= len(items) <= 128 and len(items)%32 == 0))
        if plan['version'] == 4:
            pairs = plan.get('source_pairs', [])
            valid = valid and 2 <= len(pairs) <= 4 and len(items) == 32*len(pairs)
            valid = valid and len({(p['job'],p['question_id']) for p in pairs}) == len(pairs)
            for pair in pairs:
                group = [i for i in items if i.get('source_job') == pair['job'] and i['question_id'] == pair['question_id']]
                valid = valid and [i['observed_branch_position'] for i in group] == list(range(pair['position']-32,pair['position']))
        size = 4
    if not valid or plan['repeats'] != 2 or plan['labels_added'] != 0 or plan['independent_groups_added'] != 0:
        raise ValueError('unsupported or unbounded recapture plan')
    if len({i['prefix_sha256'] for i in items}) != len(items) or not all(
            i['question_id'].startswith('TRAIN_') and token_hash(i['prefix_ids']) == i['prefix_sha256'] for i in items):
        raise ValueError('duplicate, non-TRAIN or corrupt causal prefix')
    return [(repeat, start//size, ordered[start:start+size])
            for repeat, ordered in enumerate((items, items[::-1]))
            for start in range(0, len(items), size)]


def validate_diagnostic_batch(batch):
    """Prevent passing concatenated layers into existing final-only policies."""
    if batch.get('diagnostic_worker', 'none') not in {'none', 'multilayer'}:
        raise ValueError('multilayer diagnostic cannot mix workers')
    for config in batch['configs']:
        if config.get('collect_signals', 'none') not in {'none', 'multilayer'}:
            raise ValueError('multilayer diagnostic cannot mix signal schemas')
        if config['arm'] != 'tuned_online' or any(key in config for key in
                ('auxiliary_policy', 'probe_policy', 'alignment_policy', 'gap_policy', 'example_policy')):
            raise ValueError('multilayer capture currently permits diagnostic tuned IS only')
        policy = config.get('policy', {})
        if policy.get('trigger', 'fixed') not in {'fixed', 'never'} or policy.get('query', 'concat') != 'concat':
            raise ValueError('multilayer diagnostic requires a signal-independent policy')


def validate_multilayer_batch(batch):
    """Separate opt-in layer policies from the unchanged diagnostic validator."""
    if batch.get('multilayer_policy') is not True:
        return validate_diagnostic_batch(batch)
    if batch.get('diagnostic_worker') != 'multilayer':
        raise ValueError('layer policy requires an explicit multilayer worker')
    from inference_scaling.retrieval.probe_online_is import ProbePolicy
    learned = 0
    for config in batch['configs']:
        if config.get('collect_signals') != 'multilayer' or config['arm'] != 'tuned_online':
            raise ValueError('every layer policy/control must collect the same multilayer schema')
        if any(key in config for key in ('auxiliary_policy','alignment_policy','gap_policy','example_policy')):
            raise ValueError('layer policy cannot mix other signal consumers')
        policy = config.get('policy', {})
        if policy.get('trigger','fixed') not in {'fixed','never'} or policy.get('query','concat') not in {'concat','question'}:
            raise ValueError('layer policy requires fixed checkpoints and concat/question retrieval')
        if 'probe_policy' in config:
            model = ProbePolicy(**config['probe_policy']).model
            if model['feature_schema'].get('version') != 2 or policy.get('trigger','fixed') != 'fixed':
                raise ValueError('layer policy requires an explicit captured-layer model')
            learned += 1
    if not learned:
        raise ValueError('use the diagnostic route for batches without learned layer policies')
