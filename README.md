# inference_scaling

Unified experiments for direct inference-time sampling from distributions induced by a base language model.
The project is organized around one backend interface and four algorithm families:

1. suffix-resampling Metropolis--Hastings for power targets;
2. on-policy conditional-energy importance sampling;
3. base-candidate off-policy rollout replay with fresh tail correction;
4. dynamic candidate proposals, candidate-level importance sampling, and variance--cost budget allocation.

The implementation is being built in two passes. Commits prefixed with `basic implementation:` reproduce the
mathematical algorithms first. Commits prefixed with `optimization:` add scheduling, cache reuse, vectorized
scoring, and replay-system optimizations without silently changing the target distribution.

## Design constraints

- Candidate blocks in the base replay algorithm always come from the base model.
- A stored rollout records the exact behavior policy and actual sampling probability, including temperature and
  truncation settings.
- Data used by the current decision never returns to the future evaluation pool. Only independently generated
  post-decision reserve rollouts may enter that pool.
- Asynchronous execution uses request-local random streams so scheduling order does not change a request's random
  draws.
- Optimizations that alter the estimator or Markov kernel are labelled separately from distribution-preserving
  systems changes.

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m pytest
```

GPU installation and reproducible RTX 3090 commands will be added with the Hugging Face backend. Raw model
weights and experiment artifacts are deliberately excluded from Git.

## Repository layout

- `src/inference_scaling/`: reusable algorithms, backends, schedulers, replay storage, and metrics;
- `configs/`: checked-in experiment configurations;
- `tests/`: exact tabular tests and integration tests;
- `experiments/`: command-line experiment entry points;
- `docs/`: algorithm-to-document mapping and reproduction reports;
- `results/`: small checked-in summaries only; raw outputs remain local.

