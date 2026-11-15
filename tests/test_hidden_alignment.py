import copy

import numpy as np
import pytest

from inference_scaling.retrieval.hidden_alignment import fit_alignment, project_hidden


def test_alignment_learns_synthetic_shared_space_and_rejects_tampering():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(400, 128))
    # Constant hidden dimensions must remain finite after train-only scaling.
    x[:, -1] = 5.
    truth = rng.normal(size=(128, 768))
    y = x@truth
    model = fit_alignment(x, y, alpha=.1)
    held = rng.normal(size=(128,));held[-1] = 5.
    predicted = project_hidden(model, held)
    target = held@truth;target /= np.linalg.norm(target)
    assert predicted@target > .98
    assert np.isclose(np.linalg.norm(predicted), 1.)
    broken = copy.deepcopy(model);broken['weights'][0][0] += 1.
    with pytest.raises(ValueError, match='hash'):
        project_hidden(broken, held)
    with pytest.raises(ValueError, match='paired'):
        fit_alignment(x[:, :127], y)


def test_fold_pca_compiles_to_original_input_and_reduces_rank():
    rng = np.random.default_rng(420)
    factors = rng.normal(size=(40, 3))
    mixing = rng.normal(size=(3, 128))
    x = factors@mixing
    targets = factors@rng.normal(size=(3, 768))
    model = fit_alignment(x, targets, alpha=.1, input_rank=3)
    assert np.linalg.matrix_rank(np.asarray(model['weights'])) == 3
    assert model['input_pca']['effective_rank'] == 3
    assert np.isclose(np.linalg.norm(project_hidden(model, x[0])), 1.)
    # A request above the training fold's row rank is explicitly reduced.
    small = fit_alignment(x[:3], targets[:3], input_rank=16)
    assert small['input_pca']['effective_rank'] == 2
    # The default route preserves the original deployed model schema.
    assert fit_alignment(x, targets) == fit_alignment(x, targets, input_rank=128)
    assert 'input_pca' not in fit_alignment(x, targets)
    for invalid in (0, 129, 3.5, True):
        with pytest.raises(ValueError, match='input rank'):
            fit_alignment(x, targets, input_rank=invalid)
