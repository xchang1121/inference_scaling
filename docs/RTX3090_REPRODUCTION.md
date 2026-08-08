# RTX 3090 reproduction

## Scope

This is a small real-model behavioral and systems check, not a paper-scale accuracy claim. Exact finite-state tests
in `tests/` check the target distributions and replay estimators; this run checks that the same implementation works
with a causal language model on a physical GPU and exhibits the intended qualitative behavior.

The checked JSON artifacts are `results/rtx3090_reproduction.json` (FP32 algorithms and systems) and
`results/rtx3090_backend_bfloat16.json` (reduced-precision diagnostic).

## Environment

- GPU: NVIDIA GeForce RTX 3090, 24 GiB;
- driver: 596.49 (`nvidia-smi` reports support through CUDA 13.2);
- PyTorch: 2.13.0+cu130, with its bundled CUDA 13.0 runtime;
- Transformers: 5.14.1;
- model: Qwen2.5-0.5B-Instruct at revision
  `7ae557604adf67be50417f59c2c2f167def9a775`;
- model weight SHA-256:
  `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`;
- operating system: Windows 11; Python 3.12.5.

The machine also has CUDA Toolkits 11.8 and 12.6, and the current `PATH` resolves `nvcc` to 11.8. That does not
prevent this experiment from using CUDA: ordinary PyTorch inference uses the runtime bundled with its wheel, while
`nvcc` matters when compiling a CUDA extension. The installed driver is new enough to run the cu130 wheel.

## Command

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python experiments\rtx3090_reproduction.py `
  --model models\Qwen2.5-0.5B-Instruct `
  --dtype float32 `
  --output results\rtx3090_reproduction.json
```

The BF16 diagnostic changes `--dtype` to `bfloat16` and writes the second JSON artifact.

## Backend efficiency

Eight requests with a common 47-token prompt and 24 generated tokens each were run sequentially and as one batch.

| Measurement | Result |
| --- | ---: |
| Sequential generation | 2.820 s |
| Batched generation | 0.352 s |
| Speedup | 8.02x |
| Batched throughput | 545.7 generated tokens/s |
| Shared-prefix prefill tokens avoided | 329 |
| Peak allocated CUDA memory | 2.390 GB |
| Largest continuous batch formed from eight concurrent callers | 8 |

The continuous dispatcher combined all eight synchronous single-request callers into one physical model batch. The
shared-prefix path computed the 47-token prefill once rather than eight times, then forked the KV cache.

In FP32, generated token log-probabilities and later full-sequence rescoring differed by `5.33e-6` on average and
`1.09e-4` at worst. In BF16 the same measurements were `4.93e-2` and `1.26`; its single-run throughput was lower
at 486.0 tokens/s, although allocated memory fell to 1.229 GB. This is why FP32 is the default for importance
weights in this environment. The timing comparison is one warm run rather than a confidence interval.

Even FP32 sequential and batched text was not bitwise identical. Different batch shapes can change a logit by a few
floating-point ulps; a fixed uniform draw can then fall on the other side of a categorical CDF boundary, after which
the autoregressive contexts diverge. The random stream and mathematical policy are scheduling-independent, but
GPU bitwise identity across batch shapes is not assumed.

## Algorithm behavior

### Suffix-resampling Metropolis-Hastings

Four fixed-length chains used `alpha=2`, length 16, block size 8, and three MH updates per block. The aggregate
acceptance rate was 54.2%. Mean base-model log-probability improved from `-1.761` per token for direct samples to
`-0.971` per token for the final chain states, a gain of `0.790` per token. This is the expected sharpening behavior;
four chains are not enough for a distributional accuracy estimate, which is instead covered by the enumerated
tests.

### Conditional importance sampling

The check used four fixed arithmetic prompts, four candidates, four rollouts per candidate, block size 8, and a
binary final-answer reward. Direct sampling answered three of four prompts correctly. Both on-policy rollouts and
temperature-0.7 off-policy rollouts with the exact completion likelihood ratio answered all four in this fixed run.

The on-policy mean absolute log correction was `2.56e-6`, which measures only FP32 recomputation noise. The
off-policy correction was nontrivial at `0.223`; its average completion ESS was `1.73` out of at most four, compared
with `1.84` on-policy. Matching outputs in this seed show that the correction can preserve the decision despite a
different rollout policy, but this four-prompt check is not a statistical equivalence proof.

### Off-policy rollout replay

A controlled decision first generated two temperature-0.7 historical completions for each of four reproduced base
candidates. The replay decision used all two historical completions per candidate and only one new base completion
per candidate. All four historical ESS values were 2.0.

Before the decision the evaluation pool contained eight records. Afterwards it contained zero evaluation records,
zero reserved records, and twelve design records: eight consumed histories plus four fresh completions. This checks
the intended metadata-only claim, single-use evaluation, and fresh-data lifecycle on a real model.

### Dynamic candidate importance sampling

The defensive proposal mixed the base candidate policy and a temperature-0.7 auxiliary policy equally. Across six
two-token decisions it drew 24 candidates from each component. Candidate-level outer-weight ESS ranged from 2.20
to 8.0 out of eight. The run produced the correct answer `27 * 14 = 378`; all nonterminal candidates received one
fresh rollout under the fixed eight-rollout budget.

## Interpretation and limits

These measurements establish that CUDA execution, exact-policy metadata, off-policy correction, replay
consumption, outer candidate correction, KV reuse, and continuous batching operate together on the reference
machine. They do not estimate benchmark-level confidence intervals, long-chain mixing, or scaling behavior for
multi-billion-parameter models. Those require more prompts, multiple seeds, and a larger model while retaining the
same pinned model, reward, and behavior-policy metadata recorded here.
