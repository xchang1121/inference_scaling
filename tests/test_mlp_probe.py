import numpy as np

from inference_scaling.retrieval.linear_probe import fit_mlp_probe,mlp_loss_grad,predict_probe


def test_mlp_gradients_match_finite_difference():
    rng=np.random.default_rng(7);z=rng.normal(size=(7,3));y=rng.normal(size=7)
    params=[rng.normal(size=(3,8)),rng.normal(size=8),rng.normal(size=8),np.array(.2)]
    _,grads=mlp_loss_grad(z,y,*params,.03)
    directions=[rng.normal(size=v.shape) for v in params]
    eps=1e-6
    upper=mlp_loss_grad(z,y,*[p+eps*d for p,d in zip(params,directions,strict=True)],.03)[0]
    lower=mlp_loss_grad(z,y,*[p-eps*d for p,d in zip(params,directions,strict=True)],.03)[0]
    expected=sum(float(np.sum(g*d)) for g,d in zip(grads,directions,strict=True))
    assert np.isclose((upper-lower)/(2*eps),expected,atol=1e-6)


def test_mlp_fit_reproducible_and_shared_prediction_interface():
    rng=np.random.default_rng(8);x=rng.normal(size=(80,134));y=np.tanh(x[:,128])
    model=fit_mlp_probe(x,y,target='knowledge',mode='scalar',epochs=100)
    again=fit_mlp_probe(x,y,target='knowledge',mode='scalar',epochs=100)
    assert model['sha256']==again['sha256']
    assert model['training_curve'][-1]['train_objective']<model['training_curve'][0]['train_objective']/4
    assert all(np.isfinite(predict_probe(model,row.tolist())) for row in x[:5])
