from collections import namedtuple
import math
from types import SimpleNamespace

import pytest

from inference_scaling.retrieval.internal_signals import install_distribution_signals, unpack_signals
from inference_scaling.arllm.backends.vllm_statistics import install_scalar_statistics, scalar_generated_logprobs


def check_tensor_truth():
    import torch
    payload = namedtuple('Payload', 'logprob_token_ids logprobs selected_token_ranks')
    def gather(lp, n, tokens):
        values, ids = lp.topk(n, dim=-1)
        return payload(torch.cat((tokens[:, None], ids), dim=-1),
                       torch.cat((lp.gather(-1, tokens[:, None]), values), dim=-1), None)
    sampler = SimpleNamespace(gather_logprobs=gather)
    install_scalar_statistics(sampler, 7)
    lp = torch.tensor([[0., 0., 0., 0., 0., 0., 0.], [3., 2., 1., 0., -1., -2., -3.]]).log_softmax(-1)
    saved = lp.clone()
    chosen = torch.tensor([0, 6])
    baseline = sampler.gather_logprobs(lp, 6, chosen)
    install_distribution_signals(sampler, 7, 6, 5)
    result = sampler.gather_logprobs(lp, 10, chosen)
    rows = [dict(zip(ids.tolist(), values.tolist(), strict=True))
            for ids, values in zip(result.logprob_token_ids, result.logprobs, strict=True)]
    clean, signals = unpack_signals(rows, chosen.tolist(), 7)
    reference = [dict(zip(ids.tolist(), values.tolist(), strict=True))
                 for ids, values in zip(baseline.logprob_token_ids, baseline.logprobs, strict=True)]
    assert clean == reference
    for top_k in (None, 5):
        assert scalar_generated_logprobs(clean, chosen.tolist(), top_k) == scalar_generated_logprobs(reference, chosen.tolist(), top_k)
    assert torch.equal(lp, saved)
    assert torch.equal(sampler.gather_logprobs(lp, 6, chosen).logprobs, baseline.logprobs)
    for i, signal in enumerate(signals):
        probs = [math.exp(x) for x in lp[i].tolist()]
        ordered = sorted(probs, reverse=True)
        truth = {'entropy': -sum(p*math.log(p) for p in probs),
                 'top2_gap': ordered[0]-ordered[1], 'p_eos': probs[6], 'p_think_end': probs[5]}
        for key in truth:
            assert math.isclose(truth[key], signal[key], abs_tol=1e-6), (key, truth, signal)
    hidden_sampler = SimpleNamespace(gather_logprobs=gather)
    install_scalar_statistics(hidden_sampler, 7)
    install_distribution_signals(hidden_sampler, 7, 6, 5, hidden_dimensions=3)
    hidden_sampler._retrieval_hidden_projection = torch.tensor([[1.,2.,3.],[-1.,-2.,-3.]])
    hidden_result = hidden_sampler.gather_logprobs(lp, 13, chosen)
    hidden_rows = [dict(zip(ids.tolist(), values.tolist(), strict=True))
                   for ids, values in zip(hidden_result.logprob_token_ids, hidden_result.logprobs, strict=True)]
    stripped, features = unpack_signals(hidden_rows, chosen.tolist(), 7, hidden_dimensions=3)
    assert stripped == reference
    assert [s['hidden_projection'] for s in features] == [[1.,2.,3.],[-1.,-2.,-3.]]
    from inference_scaling.retrieval.attention_signals import reconstruct_attention
    q = torch.tensor([[1.,0.],[0.,1.]])
    k = torch.tensor([[[1.,0.]],[[0.,1.]],[[1.,1.]]])
    v = torch.tensor([[[2.,1.]],[[3.,4.]],[[5.,6.]]])
    probabilities = (q@k[:,0].T/math.sqrt(2)).softmax(-1)
    native_output = probabilities@v[:,0]
    attention = reconstruct_attention(q,k,v,native_output,topk=2)
    assert attention[-2] < 1e-6 and attention[-1] < 1e-6
    attn_sampler = SimpleNamespace(gather_logprobs=gather)
    install_scalar_statistics(attn_sampler,7)
    install_distribution_signals(attn_sampler,7,6,5,attention_topk=2)
    attn_sampler._retrieval_attention = attention[None].repeat(2,1)
    attn_result = attn_sampler.gather_logprobs(lp,17,chosen)
    attn_rows = [dict(zip(ids.tolist(),values.tolist(),strict=True))
                 for ids,values in zip(attn_result.logprob_token_ids,attn_result.logprobs,strict=True)]
    stripped,features = unpack_signals(attn_rows,chosen.tolist(),7,attention_topk=2)
    assert stripped == reference and features[0]['attention']['prefix_tokens'] == 3


def test_tensor_truth():
    pytest.importorskip('torch')
    check_tensor_truth()


def test_missing_signal_fails_closed():
    with pytest.raises(RuntimeError, match='missing'):
        unpack_signals([{1: -.3, -1: 5.}], [1], 7)


def test_invalid_probability_fails_closed():
    row = {1: -.3, -1: 5., -2: 1., -3: .4, -4: 1.2, -5: .1}
    with pytest.raises(RuntimeError, match='probability'):
        unpack_signals([row], [1], 7)


def test_flat_stream_rows_align():
    class Flat:
        start_indices = [0, 7]
        end_indices = [7, 14]
        token_ids = [1, -1, -2, -3, -4, -5, 2]*2
        logprobs = [-.3, 5., 1., .4, .2, .1, -.5]*2
        def __len__(self):
            return 2
    clean, signals = unpack_signals(Flat(), [1, 2], 7)
    assert clean == [{1: -.3, -1: 5., 2: -.5}]*2
    assert signals == ({'entropy': 1., 'top2_gap': .4, 'p_eos': .2, 'p_think_end': .1},)*2
