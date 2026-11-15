"""Causal, fixed-schema features for small frozen-model retrieval probes."""
import math
import re

from inference_scaling.retrieval.corpus import terms
from inference_scaling.retrieval.smart_online_is import clean_question


def prefix_features(rows, *, pool=32, capture_layer=None):
    if not rows or pool<1:raise ValueError('observed prefix rows required')
    rows=rows[-pool:]
    vectors=[r['hidden_projection'] for r in rows]
    if capture_layer is not None:
        schema = pooled_feature_schema(pool, capture_layer=capture_layer)
        if any(len(v) != 513 or not all(math.isfinite(x) for x in v) or v[512] != 0 for v in vectors):
            raise ValueError('invalid multilayer dimensions, values or causal alignment')
        start = schema['capture_block_start']
        vectors = [v[start:start+128] for v in vectors]
    if any(len(v)!=128 for v in vectors):raise ValueError('expected validated 128-dimensional hidden projection')
    hidden=[math.fsum(v[i] for v in vectors)/len(vectors) for i in range(128)]
    scalar=[math.fsum(r[k] for r in rows)/len(rows) for k in ('entropy','top2_gap','lp','sc')]
    scalar += [rows[-1]['p_eos'],rows[-1]['p_think_end']]
    result=hidden+scalar
    if not all(math.isfinite(v) for v in result):raise ValueError('nonfinite probe input')
    return result


FEATURE_SCHEMA={'version':1,'hidden_source':'final_normalized_pre_lm_head','hidden_dimensions':128,
    'projection_seed':2026111502,'pool':32,'scalars':['mean_entropy','mean_top2_gap','mean_logprob','mean_sc','last_p_eos','last_p_think_end'],
    'state_offset':-1,'dimension':134}


def pooled_feature_schema(pool=32, *, capture_layer=None):
    if type(pool) is not int or pool not in (8,16,32,64,128):
        raise ValueError('unsupported causal feature pooling window')
    if capture_layer is None:
        return {**FEATURE_SCHEMA,'pool':pool}
    from copy import deepcopy
    from inference_scaling.retrieval.multilayer_capture import SCHEMA
    layers = (16,32,48,'final')
    if capture_layer not in layers or (capture_layer != 'final' and type(capture_layer) is not int):
        raise ValueError('unsupported captured layer')
    return {**FEATURE_SCHEMA, 'version':2, 'pool':pool,
        'hidden_source':'final_normalized_pre_lm_head' if capture_layer == 'final' else 'input_layernorm_output',
        'capture_layer':capture_layer, 'capture_block_start':128*layers.index(capture_layer),
        'capture_dimensions':513, 'capture_schema':deepcopy(SCHEMA)}


def validate_feature_schema(schema):
    expected = pooled_feature_schema(schema.get('pool'), capture_layer=schema.get('capture_layer'))
    if schema != expected:
        raise ValueError('unsupported probe feature schema')


EVIDENCE_SCHEMA={'version':1,'dimension':8,'names':['question_coverage','query_coverage','question_jaccard',
    'query_jaccard','identifier_coverage','body_length_1024','nonpadding_fraction','log1p_retrieval_score']}


def evidence_features(question,query,packet,decode):
    q=set(terms(clean_question(question)));search=set(terms(query))
    body_ids=packet['ids'][packet['body_start']:packet['body_stop']]
    body=decode(body_ids,True);words=set(terms(body))
    identifiers={t for t in q if re.search(r'[0-9_.:/-]',t)}
    def coverage(source):return len(source&words)/max(1,len(source))
    def jaccard(source):return len(source&words)/max(1,len(source|words))
    actual=packet.get('original_body_tokens',len(body_ids))
    values=[coverage(q),coverage(search),jaccard(q),jaccard(search),coverage(identifiers),
        len(body_ids)/1024,min(actual,len(body_ids))/max(1,len(body_ids)),math.log1p(max(0,float(packet.get('score',0))))]
    if not all(math.isfinite(v) for v in values):raise ValueError('nonfinite evidence features')
    return values
