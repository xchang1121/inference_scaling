"""Small regularized probes on frozen internal features; no LLM weight updates."""
import hashlib
import json
import math

import numpy as np

from inference_scaling.retrieval.probe_features import FEATURE_SCHEMA,EVIDENCE_SCHEMA


def indices(target,mode):
    if target not in {'knowledge','value','risk'} or mode not in {'all','hidden','scalar'}:
        raise ValueError('unsupported probe target or feature subset')
    selected=list(range(134)) if mode=='all' else list(range(128)) if mode=='hidden' else list(range(128,134))
    if target!='knowledge':selected+=list(range(134,142))
    return selected


def fingerprint(model):
    payload={k:v for k,v in model.items() if k!='sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def fit_probe(features,targets,*,target,mode='all',alpha=10.,provenance=None):
    if not math.isfinite(alpha) or alpha<=0:raise ValueError('positive ridge regularization required')
    x=np.asarray(features,dtype=np.float64);y=np.asarray(targets,dtype=np.float64)
    dimension=134 if target=='knowledge' else 142
    if x.ndim!=2 or x.shape[1]!=dimension or y.shape!=(len(x),) or len(x)<2:
        raise ValueError('invalid probe training shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite training data')
    select=indices(target,mode);x=x[:,select]
    center=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    z=(x-center)/scale;intercept=float(y.mean())
    weights=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@(y-intercept))
    model={'version':1,'target':target,'mode':mode,'alpha':alpha,'indices':select,'dimension':dimension,
        'feature_schema':FEATURE_SCHEMA,'evidence_schema':EVIDENCE_SCHEMA if target!='knowledge' else None,
        'center':center.tolist(),'scale':scale.tolist(),'weights':weights.tolist(),'intercept':intercept,
        'target_min':float(y.min()),'target_max':float(y.max()),'training_rows':len(y),
        'provenance':provenance or {},'prediction_semantics':{'knowledge':'predicted reference-F1 shortfall of no-evidence local suffix',
            'value':'predicted reference-F1 gain of candidate evidence in local suffix',
            'risk':'uncalibrated ridge score for empty/truncated/regressed local suffix'}[target]}
    model['sha256']=fingerprint(model)
    return model


def predict_probe(model,features):
    if model['sha256']!=fingerprint(model):raise ValueError('probe artifact hash mismatch')
    if model['feature_schema']!=FEATURE_SCHEMA or model['indices']!=indices(model['target'],model['mode']):
        raise ValueError('probe schema mismatch')
    if len(features)!=model['dimension'] or not all(math.isfinite(v) for v in features):
        raise ValueError('invalid inference features')
    if model['target']!='knowledge' and model['evidence_schema']!=EVIDENCE_SCHEMA:
        raise ValueError('evidence schema mismatch')
    x=np.asarray(features,dtype=np.float64)[model['indices']]
    center,scale,weights=(np.asarray(model[k],dtype=np.float64) for k in ('center','scale','weights'))
    if any(v.shape!=x.shape or not np.isfinite(v).all() for v in (center,scale,weights)) or (scale<=0).any():
        raise ValueError('invalid learned parameters')
    value=float(model['intercept']+((x-center)/scale)@weights)
    if not math.isfinite(value):raise ValueError('nonfinite probe prediction')
    return value
