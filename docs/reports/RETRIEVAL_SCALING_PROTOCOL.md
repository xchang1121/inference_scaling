# Retrieval combined with inference scaling: frozen pilot

This pilot tests whether search after a short evidence checkpoint helps technical
QA. It does not assume that successes on mathematical tasks transfer to TechQA.
The preceding 176-condition FLARE experiment and all its archives remain frozen.

## Rationale and scope

[Self-Certainty Best-of-N](https://arxiv.org/abs/2502.18581v3) supports distribution-based
selection without an external reward model, including open-ended tasks. Here it
is explicitly adapted to selecting a **thinking suffix** followed by one answer,
not a literal replication of selection over N complete final answers.
[Compute-optimal inference scaling](https://arxiv.org/abs/2408.03314v1) motivates
comparing allocations rather than assuming longer thinking is always better.

The repository's previous Qwen3.8 MATH pilot gave block-Metropolized SC K=2
95/100 versus base 92/100, but the paired interval included zero; coding gains
also did not generalize across datasets. This is an empirically promising local
variant, not an algorithm proved to improve all QA. Its Metropolized operation
updates the candidate **index**, not a power-target suffix MH sampler.

## Frozen design

- Same pinned Qwen3.8-27B weights, tokenizer, image and runtime as the previous
  FLARE pilot. A physical NPU3 only, TP1. No training or external reward model.
- Eight answerable DEV questions: four new document-disjoint hash-selected
  questions, plus four old questions chosen by gate category (supported accept,
  false-positive accept, invalid quotation, rejection), never answer F1.
  Regression and fresh cohorts are reported separately. Public questions and the
  full 28,481-document corpus enter the runtime; gold answers remain offline.
- Three methods: single long continuation; SC-BoN up to eight suffixes of at most
  1024 tokens; complete-pool block-Metropolized conditional IS, M=4, K=2,
  block=256, suffix horizon=1024, reward temperature=.25, fixed block schedule,
  block-first batch reference, no approximate selection-ready stopping.
- Five evidence views: closed, static real/noise, dynamic FLARE-gap real/noise.
  8 x 3 x 5 = 120 logical conditions, one seed. This is a feasibility pilot with
  four independent fresh questions, not a powered effectiveness study.
- Every standalone condition has an **8192 generated-token upper limit** for
  thinking, provisional/discarded text, planner/checker JSON including EOS,
  every search candidate and rollout, and rejected/unselected candidates.
  Final answer has a separate 512-token limit and is generated once. Natural
  termination is respected, so realized compute can differ substantially.
  Evidence input tokens are recorded separately; native graph FLOPs and wall
  time are reported, never claimed to be FLOPs matched.
- SC-BoN funds a complete batch from the remaining worst-case suffix capacity:
  `min(8, remaining_work // horizon)` candidates. Thus the dynamic probe can
  reduce N from eight to seven at horizon 1024. Early endings do not refill
  that batch. IS likewise stops between complete trees. These conservative
  allocations and unused budgets must remain visible in the results.
- Same full-support T=1 main policy and SC=mean KL(U||p). SC is read from native
  generation statistics, over the new suffix only, excluding EOS/think-close.
  Empty thinking has neutral score zero and is not filtered out. No rescoring
  forwards are permitted. Auxiliary generation remains greedy, thinking off,
  at most 96 output tokens per check. No low/effort instruction is rendered.

## Retrieval/search interface

The old FLARE/gap prompts, first eligible completed-step check after 64 confirmed
tokens, subsequent low-confidence trigger, two planner checks, single retrieval,
literal-quote gate, BM25 parent/window search and 256-token packet are unchanged.
A new bounded probe scans at most 256 main tokens. It stops after a completed
retrieval decision, two planner decisions, or scan exhaustion; it rolls back
pending provisional tokens to the last confirmed complete step. All discarded
tokens and provisional terminal markers are charged and preserved. If a natural
end is committed before the checkpoint, the completed answer is reused and the
search-bypass status is explicit; thinking is not forcibly reopened.

Accepted evidence is rebuilt into the original user reference region at the
saved completed-step boundary. This uses the safer format from the previous
experiment. A search epoch then has exactly one immutable prompt, including its
confirmed history and evidence. Old scores/candidates are never reused across
evidence changes. No retrieval happens inside an IS candidate pool. Thus this
is a staged adaptive-retrieval plus suffix-search composition, not multi-hop
tree retrieval or a claim of unbiased inference over changing contexts.

Real/noise variants share the actual probe. Noise substitutes source body IDs
only after the real support gate, retaining all labels, wrappers and exact token
length. If no packet is accepted, real/noise reuse the identical search output.
The closed/static searches do not pay a dynamic probe. Each standalone view is
charged its full common plus branch cost; actual study cost deduplicates shared
execution units. Branches cold-prefill; native KV-fork equivalence is not assumed.

Before an IS step the entire worst-case tree is reserved. With remaining horizon
R and width B, the initial bound is M[B+K(R-B)]; carried-state bound is
(M-1)[B+K(R-B)]+(K-1)(R-B). If the next whole tree cannot fit, return the already
selected complete rollout. Never select from partial weights. This reservation
can leave budget unused, and the pilot must report that utilization.

## Validation and reporting

CPU tests cover reservations, multi-step completion, reward-context scope, EOS
retention and evidence-epoch resets. Two disjoint TRAIN questions are a native
smoke test before the frozen formal run. Every raw prefix, seed, generated ID,
log probability, SC statistic and finish reason is retained. Audit total work,
per-request context, complete-pool scores, exact paired prehistory and evidence
length; record empty answers, selected/raw EOS, immediate think-close, forced
closure, search bypass, wall time and native graph estimates. Normalized answer
token F1/EM and paired question-level bootstrap intervals are descriptive only.
No hyperparameters or question selection are tuned on formal outputs.

Wall times cover measured model execution units, with online dynamic lookup
included. Offline static-packet preparation, index construction and model
startup are excluded from those per-condition times and recorded separately
where possible. Native graph FLOPs are dense-padded estimates, excluding some
attention and host operations. The two TRAIN smoke questions are not quality
selection data. A pre-formal input correction restored the exact prior DEV_Q119
prompt (one trailing space); both input versions and hashes are preserved.
