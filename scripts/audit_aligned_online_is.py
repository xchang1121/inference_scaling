"""Replay causal hidden projection and full frozen-index dense search."""
from types import MethodType,SimpleNamespace

import numpy as np

from inference_scaling.retrieval.aligned_online_is import AlignmentPolicy,fuse_documents
from inference_scaling.retrieval.hidden_alignment import project_hidden
from inference_scaling.retrieval.probe_features import prefix_features
from inference_scaling.retrieval.tuned_online_is import TunedOnlineLedger


def prepare_alignment_audit(row,cfg,decode,eos,close,ck,vectors):
    if vectors is None:raise ValueError('full frozen document index required for alignment audit')
    p=AlignmentPolicy(**row['alignment_policy']);queries={}
    ck(vectors.sha256==row['document_vector_sha256']==p.model['provenance']['vector_sha256'],'alignment_index_identity')
    def resolve(d):
        native=row['native_calls'][:d['observed_native_calls']]
        proxy=SimpleNamespace(root=tuple(row['root_ids']),native_calls=native,
            tap=SimpleNamespace(history={c['request_id']:c['internal_signals'] for c in native}))
        proxy.recent=MethodType(TunedOnlineLedger.recent,proxy)
        features=prefix_features(proxy.recent(d['branch_ids'],32));vector=project_hidden(p.model,features[:128])
        signal=d['signal']
        ck(signal['probe_features']==features and signal['model_sha256']==p.model['sha256'],'alignment_causal_features')
        ck(np.allclose(signal['projected_query'],vector,atol=1e-10,rtol=1e-10),'alignment_projection')
        queries[d['branch_sha256']]=vector
        return True,signal
    def rank(d,ranked,used,default):
        r=d['retrieval'];a=r['alignment'];expected=vectors.search(queries[d['branch_sha256']],k=p.candidate_documents)
        ck(a['vector_sha256']==vectors.sha256,'alignment_search_index')
        ck([v[0] for v in expected]==[v[0] for v in a['dense_hits']],'alignment_full_index_topk')
        ck(np.allclose([v[1] for v in expected],[v[1] for v in a['dense_hits']],atol=1e-9,rtol=1e-9),'alignment_cosine_scores')
        order,scores=fuse_documents([name for name,_ in expected],[name for name,_ in a['lexical_hits']],p.dense_weight)
        ck(a['parent_order']==order and a['fusion_scores']==scores,'alignment_parent_fusion')
        ck(r['parent_ids']==order[:len(r['parent_ids'])],'alignment_parent_selection')
        return default
    return resolve,lambda d:None,lambda:None,rank
