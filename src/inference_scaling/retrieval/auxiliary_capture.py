"""Explicit real-layer views for candidate-local auxiliary disagreement."""
import math

from inference_scaling.retrieval.auxiliary_signals import hidden_disagreement
from inference_scaling.retrieval.multilayer_capture import SCHEMA


def layer_block(layer):
    if type(layer) is int and layer in (16, 32, 48):
        return (layer//16-1)*128
    if type(layer) is str and layer == 'final':
        return 384
    raise ValueError('explicit captured layer must be 16, 32, 48, or final')


def validate_captured_rows(rows, tokens):
    if rows is None or not tokens or len(rows) != len(tokens):
        raise ValueError('missing aligned auxiliary captured rows')
    for row in rows:
        vector = row['hidden_projection']
        if (len(vector) != 513 or vector[512] != 0
                or not all(math.isfinite(x) for x in vector)):
            raise ValueError('captured auxiliary rows require finite513 and exact alignment zero')


def policy_disagreement(policy, calls):
    """Preserve legacy averaging; select one verified block before any pooling."""
    layer = policy.capture_layer
    if layer is None:
        if any(len(s['hidden_projection']) == 513 for c in calls
               for s in (c['internal_signals'] or [])):
            raise ValueError('captured layers require explicit auxiliary layer selection')
        return hidden_disagreement(calls, tail=policy.hidden_tail, metric=policy.metric, ridge=policy.ridge)
    start = layer_block(layer)
    views = []
    for call in calls:
        validate_captured_rows(call['internal_signals'], call['token_ids'])
        views.append({**call, 'internal_signals': [
            {**row, 'hidden_projection': row['hidden_projection'][start:start+128]}
            for row in call['internal_signals']]})
    feature = hidden_disagreement(views, tail=policy.hidden_tail, metric=policy.metric, ridge=policy.ridge)
    feature['capture'] = {'schema': SCHEMA, 'layer': layer, 'block_start': start,
                          'source': SCHEMA['blocks'][start//128]['source']}
    return feature


def validate_auxiliary_batch(batch):
    """Keep auxiliary-layer policies separate from learned probes and diagnostics."""
    from inference_scaling.retrieval.auxiliary_online_is import AuxiliaryPolicy
    if (batch.get('multilayer_auxiliary') is not True or batch.get('multilayer_policy')
            or batch.get('attention_geometry') is not None or batch.get('family') != 'seakr'
            or batch.get('diagnostic_worker') != 'multilayer'):
        raise ValueError('explicit separate multilayer auxiliary batch and worker required')
    if (not 1 <= len(batch.get('cases', [])) <= 4 or len(batch.get('seeds', [])) != 1
            or not 1 <= len(batch.get('configs', [])) <= 3):
        raise ValueError('auxiliary layer batches require bounded TRAIN cases and configurations')
    for config in batch['configs']:
        if (config['arm'] != 'tuned_online' or config.get('collect_signals') != 'multilayer'
                or any(k in config for k in ('probe_policy', 'alignment_policy', 'gap_policy', 'example_policy'))):
            raise ValueError('all auxiliary layer configurations need the same captured schema')
        policy = AuxiliaryPolicy(**config.get('auxiliary_policy', {}))
        layer_block(policy.capture_layer)
        if policy.method != 'seakr':
            raise ValueError('captured auxiliary layers require SeaKR')
        tuning = config.get('policy', {})
        if tuning.get('trigger', 'fixed') != 'fixed' or tuning.get('query', 'concat') not in {'concat', 'question'}:
            raise ValueError('captured auxiliary layers require fixed checkpoints and concat/question retrieval')
