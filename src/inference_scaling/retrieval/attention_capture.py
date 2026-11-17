"""Bound tagged attention to signal-independent full-IS validation pairs."""
from inference_scaling.retrieval.attention_geometry import geometry


def validate_attention_batch(batch):
    schema = geometry(batch.get('attention_geometry'))
    repeats = batch.get('attention_repeats', 1)
    if type(repeats) is not int or repeats not in (1, 2):
        raise ValueError('attention diagnostics permit one or two explicit repetitions')
    if (batch.get('family') != 'engineering' or batch.get('diagnostic_worker') != 'attention'
            or len(batch.get('cases', [])) != 2
            or len(batch.get('seeds', [])) != 1 or len(batch.get('configs', [])) != 2*repeats):
        raise ValueError('tagged attention currently requires two-case OFF/ON engineering pairs')
    configs = batch['configs']
    modes = [c.get('collect_signals', 'none') for c in configs]
    if modes != ['none', 'attention']*repeats:
        raise ValueError('tagged attention requires ordered OFF/ON controls')
    normalized = [{k: v for k, v in c.items() if k not in {'id', 'collect_signals'}} for c in configs]
    if any(c != normalized[0] for c in normalized[1:]):
        raise ValueError('attention controls must have identical policy and evidence settings')
    for config in configs:
        if (config['arm'] != 'tuned_online' or any(k in config for k in
                ('attention_geometry', 'auxiliary_policy', 'gap_policy', 'example_policy'))):
            raise ValueError('tagged attention diagnostic cannot mix policy consumers')
        policy = config.get('policy', {})
        if policy.get('trigger', 'fixed') != 'fixed' or policy.get('query', 'concat') != 'concat':
            raise ValueError('attention diagnostic requires fixed actual injection with concat queries')
    return schema
