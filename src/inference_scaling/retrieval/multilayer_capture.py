"""Explicit diagnostic schema; multi-layer policies require separate validation."""

SCHEMA = {
    'version': 1, 'block_dimensions': 128,
    'blocks': [{'source': 'input_layernorm_output', 'layer': n} for n in (16, 32, 48)]
        + [{'source': 'final_normalized_pre_lm_head'}],
    'representation_dimensions': 512, 'alignment_diagnostic_index': 512,
    'projection_seed': 2026111502, 'state_offset': -1,
    'same_projection_for_each_real_block': True,
}


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
