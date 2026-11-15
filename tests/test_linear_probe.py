import copy

import numpy as np
import pytest

from inference_scaling.retrieval.linear_probe import fit_probe,predict_probe


def test_probe_fits_predictive_signal_and_rejects_corrupt_artifact():
    rng=np.random.default_rng(40);x=rng.normal(size=(80,134));x[:,0]=2.
    y=2*x[:,128]-.5*x[:,129]+.3
    model=fit_probe(x,y,target='knowledge',mode='scalar',alpha=.1)
    held=rng.normal(size=(20,134));truth=2*held[:,128]-.5*held[:,129]+.3
    predictions=np.array([predict_probe(model,v.tolist()) for v in held])
    assert np.mean((predictions-truth)**2)<.001
    broken=copy.deepcopy(model);broken['weights'][0]+=1
    with pytest.raises(ValueError,match='hash'):predict_probe(broken,held[0].tolist())
    with pytest.raises(ValueError,match='features'):predict_probe(model,[1.]*142)


def test_value_probe_uses_evidence_features_with_hidden_subset():
    rng=np.random.default_rng(41);x=rng.normal(size=(400,142));y=.5*x[:,134]
    model=fit_probe(x,y,target='value',mode='hidden',alpha=.1)
    base=np.zeros(142);left=predict_probe(model,base.tolist());base[134]=1
    assert predict_probe(model,base.tolist())>left+.4
    assert not set(range(128,134)).intersection(model['indices'])
