"""Explicit experimental attention geometry, carried by the native signal payload."""
from math import isfinite

GEOMETRIES = {
    'layer63-all': (63, 'all'),
    'layer31-all': (31, 'all'),
    'layer63-even': (63, 'even'),
    'layer63-odd': (63, 'odd'),
}
HEAD_CODES = {'all': 0, 'even': 1, 'odd': 2}
WORKERS = {
    'layer63-all': 'TaggedAttention63Worker',
    'layer31-all': 'TaggedAttention31Worker',
    'layer63-even': 'TaggedAttentionEvenWorker',
    'layer63-odd': 'TaggedAttentionOddWorker',
}


def geometry(name):
    if not isinstance(name, str) or name not in GEOMETRIES:
        raise ValueError('unknown experimental attention geometry')
    layer, heads = GEOMETRIES[name]
    return {'version': 1, 'layer': layer, 'head_pooling': heads,
            'head_order': 'flattened kv_head then query_group',
            'source': 'native post-RoPE query and full paged KV', 'metadata_dimensions': 4}


def head_indices(count, mode):
    if type(count) is not int or count < 2 or mode not in HEAD_CODES:
        raise ValueError('invalid attention head selection')
    return tuple(range(count)) if mode == 'all' else tuple(range(HEAD_CODES[mode]-1, count, 2))


def decode_geometry(values, name):
    expected = geometry(name)
    if len(values) != 4 or any(type(v) not in (int, float) or not isfinite(v) or v != int(v) for v in values):
        raise RuntimeError('invalid native attention geometry tags')
    layer, code, count, selected = map(int, values)
    if layer != expected['layer'] or code != HEAD_CODES[expected['head_pooling']]:
        raise RuntimeError('native attention geometry does not match requested schema')
    indices = head_indices(count, expected['head_pooling'])
    if selected != len(indices):
        raise RuntimeError('native attention head count mismatch')
    return {**expected, 'heads': count, 'selected_heads': list(indices)}
