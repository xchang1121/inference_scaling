import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

from test_probe_pairs import FeaturesNative
from inference_scaling.retrieval.aligned_online_is import AlignmentPolicy
from inference_scaling.retrieval.corpus import Document,Retriever
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.hidden_alignment import DocumentVectors,fit_alignment
from inference_scaling.retrieval.literature import FactBank
from inference_scaling.retrieval.tuned_online_is import TuningPolicy,run_tuned_is
from inference_scaling.shared.rng import SeedStream


def test_hidden_retrieval_full_is_audit_replays_alignment_and_rank():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_tuned_online_is import audit_tuned_row
    vectors=object.__new__(DocumentVectors);vectors.ids=['a','b'];vectors.sha256='synthetic'
    vectors.vectors=np.zeros((2,768));vectors.vectors[0,0]=1.;vectors.vectors[1,1]=1.
    model=fit_alignment(np.zeros((2,128)),np.tile(vectors.vectors[0],(2,1)),provenance={'vector_sha256':'synthetic'})
    b=FeaturesNative();cfg=replace(GuidanceConfig(),thinking_tokens=640,answer_tokens=1024,context_tokens=8192)
    index=Retriever([Document('a','Q support','Q facts for this question.'),Document('b','Q related','Q other facts.')],kind='bm25')
    result=run_tuned_is(b,b,(1,2),(901,),SeedStream(1).derive('question','synthetic'),cfg,question='Q',
        facts=FactBank(b,index),policy=TuningPolicy(),alignment_policy=AlignmentPolicy(model),document_vectors=vectors)
    row=json.loads(json.dumps({'unit_id':'synthetic','question':'Q','question_id':'synthetic','seed':1,
        'base_ids':[1,2],'static_packets':[],'static_injected_ids':[],
        'cost':{'generated_tokens':result['total_generation_tokens']},**result}))
    assert row['events'] and b.requests[0].prefix==(1,2)
    def audit(r):
        failures=[]
        audit_tuned_row(r,cfg,lambda ids,skip=False:b.decode(ids,skip_special_tokens=skip),900,901,
            lambda ok,label:failures.append(label) if not ok else None,document_vectors=vectors)
        return failures
    assert not audit(row)
    broken=copy.deepcopy(row);broken['decisions'][0]['signal']['probe_features'][0]+=1.
    assert any('alignment_causal_features' in s for s in audit(broken))
    broken=copy.deepcopy(row);broken['decisions'][0]['retrieval']['alignment']['dense_hits'][0][1]+=.1
    assert any('alignment_cosine_scores' in s for s in audit(broken))
