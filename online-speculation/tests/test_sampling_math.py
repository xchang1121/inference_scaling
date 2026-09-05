import itertools

import pytest
import torch

from blockspec.diffusion import corrupt, posterior, psi_transition
from blockspec.sampling import SamplingConfig, probabilities, residual, verify_linear


@pytest.mark.parametrize("p,q", [
    ([.2, .5, .3], [.6, .1, .3]),
    ([1., 0., 0.], [0., .5, .5]),
    ([.2, .5, .3], [.2, .5, .3]),
    ([0., .4, .6], [1., 0., 0.]),
])
def test_exhaustive_one_step_output_law(p, q):
    p, q = torch.tensor(p, dtype=torch.float64), torch.tensor(q, dtype=torch.float64)
    r, mass = residual(p, q)
    # Enumerate every proposed token, both acceptance branches and correction.
    law = torch.zeros_like(p)
    for y in range(len(p)):
        a = min(1., float(p[y] / q[y])) if q[y] else 0.
        law[y] += q[y] * a
        law += q[y] * (1 - a) * r
    torch.testing.assert_close(law, p, atol=1e-15, rtol=1e-15)
    torch.testing.assert_close(mass, .5 * (p - q).abs().sum())


def test_adaptive_proposals_preserve_two_token_joint_law():
    # The second proposal is allowed to depend on the first emitted token.
    p0 = torch.tensor([.3, .7], dtype=torch.float64)
    q0 = torch.tensor([.9, .1], dtype=torch.float64)
    p1 = torch.tensor([[.2, .8], [.6, .4]], dtype=torch.float64)
    q1 = torch.tensor([[.8, .2], [0., 1.]], dtype=torch.float64)
    def law(p, q):
        r, _ = residual(p, q)
        return torch.minimum(p, q) + (q - p).clamp_min(0).sum() * r
    actual = law(p0, q0)[:, None] * torch.stack([law(p1[h], q1[h]) for h in range(2)])
    torch.testing.assert_close(actual, p0[:, None] * p1, atol=1e-15, rtol=1e-15)


def test_rejection_and_all_accept_alignment():
    q = torch.tensor([[1., 0., 0.], [0., 1., 0.]])
    p = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
    result = verify_linear(torch.tensor([0, 1]), q, p, acceptance_uniforms=torch.zeros(2))
    assert result.tokens == [0, 2]
    assert (result.accepted, result.rejected_at, result.supervised) == (1, 1, 2)
    result = verify_linear(torch.tensor([0, 1]), q, torch.cat((q, p[-1:])),
                           acceptance_uniforms=torch.full((2,), .999))
    assert result.tokens == [0, 1, 1]
    assert (result.accepted, result.rejected_at, result.supervised) == (2, None, 2)


def test_empty_proposal_is_a_bonus_draw():
    result = verify_linear(torch.empty(0, dtype=torch.long), torch.empty(0, 2),
                           torch.tensor([[0., 1.]]))
    assert result.tokens == [1] and result.supervised == 0


@pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.0, float("inf")])
def test_invalid_acceptance_randomness(bad):
    with pytest.raises(ValueError):
        verify_linear(torch.tensor([0]), torch.tensor([[1., 0.]]),
                      torch.tensor([[1., 0.], [1., 0.]]),
                      acceptance_uniforms=torch.tensor([bad]))


def test_temperature_topk_topp_and_greedy():
    logits = torch.tensor([3., 2., 1., 0.], dtype=torch.float64)
    assert torch.equal(probabilities(logits, SamplingConfig()), torch.tensor([1., 0., 0., 0.]))
    p = probabilities(logits, SamplingConfig(temperature=2, top_k=2))
    torch.testing.assert_close(p[:2], (logits[:2] / 2).softmax(-1))
    assert p[2:].sum() == 0
    p = probabilities(logits, SamplingConfig(temperature=1, top_p=.7))
    assert torch.equal(p > 0, torch.tensor([True, True, False, False]))


@pytest.mark.parametrize("alpha_s,alpha_t", [(1., 0.), (.8, .3), (.7, 0.), (1., .7)])
def test_reverse_posterior_against_enumerated_markov_joint(alpha_s, alpha_t):
    prior = torch.tensor([.1, .3, .6], dtype=torch.float64)
    for clean, observed in itertools.product(range(3), repeat=2):
        x = torch.nn.functional.one_hot(torch.tensor(clean), 3).double()
        got = posterior(torch.tensor(observed), x, prior, alpha_s, alpha_t)
        joint = torch.empty(3, dtype=torch.float64)
        for intermediate in range(3):
            first = alpha_s * (intermediate == clean) + (1 - alpha_s) * prior[intermediate]
            ratio = alpha_t / alpha_s
            second = ratio * (intermediate == observed) + (1 - ratio) * prior[observed]
            joint[intermediate] = first * second
        torch.testing.assert_close(got, joint / joint.sum(), atol=1e-15, rtol=1e-15)


@pytest.mark.parametrize("kappa", [0., .3, 1.])
def test_one_step_psi_reduces_to_prediction(kappa):
    prior = torch.tensor([.1, .3, .6], dtype=torch.float64)
    q = torch.tensor([.4, .5, .1], dtype=torch.float64)
    for token in range(3):
        got = psi_transition(torch.tensor(token), q, prior, 1., 0., kappa)
        torch.testing.assert_close(got, q, atol=1e-15, rtol=1e-15)


def test_corruption_endpoints():
    clean = torch.tensor([[0, 1, 2], [2, 1, 0]])
    prior = torch.tensor([0., 1., 0.])
    assert torch.equal(corrupt(clean, prior, 1), clean)
    assert torch.equal(corrupt(clean, prior, 0), torch.ones_like(clean))
