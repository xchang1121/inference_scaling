"""Small regularized probes on frozen internal features; no LLM weight updates."""
import hashlib
import json
import math

import numpy as np

from inference_scaling.retrieval.probe_features import EVIDENCE_SCHEMA,pooled_feature_schema,validate_feature_schema


def indices(target,mode):
    if target not in {'knowledge','value','risk'} or mode not in {'all','hidden','scalar'}:
        raise ValueError('unsupported probe target or feature subset')
    selected=list(range(134)) if mode=='all' else list(range(128)) if mode=='hidden' else list(range(128,134))
    if target!='knowledge':selected+=list(range(134,142))
    return selected


def fingerprint(model):
    payload={k:v for k,v in model.items() if k!='sha256'}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def fit_probe(features,targets,*,target,mode='all',alpha=10.,provenance=None,pool=32):
    schema=pooled_feature_schema(pool)
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
        'feature_schema':schema,'evidence_schema':EVIDENCE_SCHEMA if target!='knowledge' else None,
        'center':center.tolist(),'scale':scale.tolist(),'weights':weights.tolist(),'intercept':intercept,
        'target_min':float(y.min()),'target_max':float(y.max()),'training_rows':len(y),
        'provenance':provenance or {},'prediction_semantics':{'knowledge':'predicted reference-proxy shortfall of no-evidence local suffix; exact formula in provenance',
            'value':'predicted reference-proxy gain of candidate evidence in local suffix; exact formula in provenance',
            'risk':'uncalibrated ridge score for empty/truncated/regressed local suffix; exact formula in provenance'}[target]}
    model['sha256']=fingerprint(model)
    return model


def predict_probe(model,features):
    if model['sha256']!=fingerprint(model):raise ValueError('probe artifact hash mismatch')
    validate_feature_schema(model['feature_schema'])
    if model['indices']!=indices(model['target'],model['mode']):
        raise ValueError('probe schema mismatch')
    if len(features)!=model['dimension'] or not all(math.isfinite(v) for v in features):
        raise ValueError('invalid inference features')
    if model['target']!='knowledge' and model['evidence_schema']!=EVIDENCE_SCHEMA:
        raise ValueError('evidence schema mismatch')
    x=np.asarray(features,dtype=np.float64)[model['indices']]
    if model.get('architecture')=='tanh_mlp':
        return predict_mlp(model,x)
    center,scale,weights=(np.asarray(model[k],dtype=np.float64) for k in ('center','scale','weights'))
    if any(v.shape!=x.shape or not np.isfinite(v).all() for v in (center,scale,weights)) or (scale<=0).any():
        raise ValueError('invalid learned parameters')
    value=float(model['intercept']+((x-center)/scale)@weights)
    if not math.isfinite(value):raise ValueError('nonfinite probe prediction')
    return value


def mlp_loss_grad(z,y,w1,b1,w2,b2,penalty):
    hidden=np.tanh(z@w1+b1);predicted=hidden@w2+b2;residual=predicted-y
    loss=float(np.mean(residual**2)+penalty*(np.sum(w1**2)+np.sum(w2**2)))
    d=2*residual/len(y)
    dh=d[:,None]*w2[None,:]*(1-hidden**2)
    return loss,[z.T@dh+2*penalty*w1,dh.sum(0),hidden.T@d+2*penalty*w2,np.array(d.sum())]


def fit_mlp_probe(features,targets,*,target,mode='all',alpha=.1,width=8,epochs=300,seed=20261115,provenance=None,pool=32):
    schema=pooled_feature_schema(pool)
    if alpha<=0 or not math.isfinite(alpha) or width not in (8,16) or epochs<1:
        raise ValueError('invalid MLP configuration')
    x=np.asarray(features,dtype=np.float64);y=np.asarray(targets,dtype=np.float64)
    dimension=134 if target=='knowledge' else 142
    if x.ndim!=2 or x.shape[1]!=dimension or y.shape!=(len(x),) or len(x)<2:
        raise ValueError('invalid MLP training shape')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('nonfinite training data')
    select=indices(target,mode);x=x[:,select];center=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    z=(x-center)/scale;rng=np.random.default_rng(seed)
    params=[rng.normal(0,1/math.sqrt(z.shape[1]),(z.shape[1],width)),np.zeros(width),
            rng.normal(0,.05,width),np.array(y.mean())]
    moments=[np.zeros_like(v) for v in params];variances=[np.zeros_like(v) for v in params];curve=[]
    for epoch in range(1,epochs+1):
        loss,grads=mlp_loss_grad(z,y,*params,alpha/len(y))
        if not math.isfinite(loss):raise ValueError('MLP optimization diverged')
        if epoch==1 or epoch%50==0 or epoch==epochs:curve.append({'epoch':epoch,'train_objective':loss})
        for i,gradient in enumerate(grads):
            moments[i]=.9*moments[i]+.1*gradient;variances[i]=.999*variances[i]+.001*gradient**2
            params[i]-=.01*(moments[i]/(1-.9**epoch))/(np.sqrt(variances[i]/(1-.999**epoch))+1e-8)
    model={'version':1,'architecture':'tanh_mlp','target':target,'mode':mode,'alpha':alpha,'width':width,
        'epochs':epochs,'seed':seed,'learning_rate':.01,'optimizer':'full_batch_Adam','indices':select,'dimension':dimension,
        'feature_schema':schema,'evidence_schema':EVIDENCE_SCHEMA if target!='knowledge' else None,
        'center':center.tolist(),'scale':scale.tolist(),'w1':params[0].tolist(),'b1':params[1].tolist(),
        'w2':params[2].tolist(),'b2':float(params[3]),'training_rows':len(y),'training_curve':curve,
        'provenance':provenance or {},'prediction_semantics':'unbounded local suffix surrogate score; not calibrated probability'}
    model['sha256']=fingerprint(model)
    return model


def predict_mlp(model,x):
    center,scale,w1,b1,w2=(np.asarray(model[k],dtype=np.float64) for k in ('center','scale','w1','b1','w2'))
    width=model['width']
    if (center.shape!=x.shape or scale.shape!=x.shape or w1.shape!=(len(x),width) or b1.shape!=(width,)
            or w2.shape!=(width,) or not all(np.isfinite(v).all() for v in (center,scale,w1,b1,w2)) or (scale<=0).any()):
        raise ValueError('invalid MLP parameters')
    value=float(np.tanh(((x-center)/scale)@w1+b1)@w2+model['b2'])
    if not math.isfinite(value):raise ValueError('nonfinite MLP prediction')
    return value
