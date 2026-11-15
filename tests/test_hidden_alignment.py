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
