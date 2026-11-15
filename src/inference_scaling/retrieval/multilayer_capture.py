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
        valid = (plan['kind'] == 'multilayer_prefix_recapture_pool' and plan['version'] == 3
                 and plan.get('pool') == 32 and len(items) == 64 and plan.get('batch_size') == 4)
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
