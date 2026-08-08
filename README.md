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

## Implemented so far

- The MH path follows the staged fixed-length algorithm in the article, including a uniformly sampled suffix
  start, the four log-probability acceptance ratio, cached current-state token scores, independent chains, and
  acceptance diagnostics.
- Exact tabular tests enumerate the target power distribution and check the sampler's empirical output.
- The conditional-energy path samples candidate blocks only from the base model and supports either on-policy completions or a
  full-support off-policy rollout model. The importance ratio is computed only over the completion suffix, and
  all random weights are aggregated in the log domain.
- `base-replay` implements metadata-only design freezing, single-use evaluation records, behavior-mixture
  denominators, truncated historical ratios with an independent fresh tail correction, and post-selection reserve
  records. Current-decision fresh samples and consumed history are irreversibly moved to the design pool.
- `dynamic-is` adds defensive candidate mixtures, exact candidate-level probability ratios, and a frozen joint
  history/fresh allocation. Its default cold-start allocation is replaceable by the included design-pool empirical
  variance and token-cost estimator.
- `ContinuousBatchingBackend` merges candidate, rollout, and score requests from concurrently running prompts while
  preserving request-local seeds and original result order; its counters expose achieved batch sizes.
- `ScoreCachingBackend` reuses exact base and behavior scores only when model, sampling policy, prefix, and
  continuation all match; random generation is never cached.
- Replay fresh completions and post-selection reserve completions are flattened across candidates into one backend
  batch, while their request-local seeds and replay keys remain distinct.

Run the standalone finite-state smoke experiments with:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\toy_mh.py
.\.venv\Scripts\python experiments\toy_conditional_is.py
.\.venv\Scripts\python experiments\toy_base_replay.py
.\.venv\Scripts\python experiments\toy_dynamic_is.py
```

## Repository layout

- `src/inference_scaling/`: reusable algorithms, backends, schedulers, replay storage, and metrics;
- `configs/`: checked-in experiment configurations;
- `tests/`: exact tabular tests and integration tests;
- `experiments/`: command-line experiment entry points;
- `docs/`: algorithm-to-document mapping and reproduction reports;
- `results/`: small checked-in summaries only; raw outputs remain local.
