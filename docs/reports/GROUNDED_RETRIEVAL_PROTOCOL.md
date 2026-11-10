# Question-anchored retrieval with answer-verified IS

This follow-up tests the concrete failures in the preceding frozen pilot:
invented product assumptions, symptom excerpts mistaken for causal evidence,
lost edition/license exceptions, and self-certainty favoring unsupported claims.
All preceding experiments remain unchanged.

## Resources and validation split

The user explicitly requested checking and using more available cards. Read-only
inventory found A3 free, A7 attached to another active experiment, and B0/4/5/6/7
without accelerator processes. Use only A3 and B4/5/6, after rechecking each at
launch. Every worker is an isolated TP1 replica. Do not change other jobs or the
shared main checkout. Four replicas improve experimental throughput, not the
per-answer compute budget.

Both hosts use the exact same v0.23 base image, the already validated patched
runner and all 804 native operator files, same Qwen3.8-27B weights/tokenizer and
settings. Verify runtime packages and a native smoke on each card. There is no
cross-device or stochastic KV equivalence claim. Assign complete question/seed
blocks to workers so paired arms always run on the same physical device.

Eight previous questions are regression diagnostics, including the four previously
fresh questions. Twelve additional nonempty answerable DEV questions are selected
by SHA256(`grounded-is-20261110:ID`), with gold document groups disjoint from prior
gold and all parent groups retrieved in the two preceding FLARE/scaling runs,
and from one another. Two seeds (20261110, 20261111). Gold stays offline and only
determines answerability, group exclusions and eventual evaluation. Two separate
TRAIN fixtures check native contracts before formal execution. No tuning on fresh
outputs; old regressions must never be pooled into generalization estimates.

## Eight arms and budget

1. Static legacy BM25 windows + previous block-Met SC (M4/K2, horizon1024/block256).
2. Dynamic legacy FLARE/gap + the same previous block-Met SC.
3. Static question-anchored evidence + M3/K2 SC (horizon768/block256).
4. Dynamic question-anchored evidence + the same M3/K2 SC.
5. Static anchored evidence + M3/K2 answer-verified IS.
6. Dynamic anchored evidence + the same answer-verified IS.
7. Same as 5, substituting unrelated body text of exactly the same token length.
8. Same as 6, substituting unrelated body text of exactly the same token length.

20 questions x 2 seeds x 8 arms = 320 logical formal conditions. The independent
fresh question count is 12, not 24 seeds or 192 conditions. Report question-level
paired intervals after averaging seeds and disclose unadjusted descriptive tests.

Each condition has an 8192 generated-work cap. It includes probes, discarded
prefixes, planner/extractor JSON, every candidate and rollout, candidate final
answers and verifier outputs, including EOS. Returned answers have a 512-token
limit. Legacy and SC arms generate one answer after selecting thinking, with its
tokens separately charged. Verified IS already pays all candidate answers inside
the work cap and returns the selected cached answer, with no regeneration. Thus
the verified arm has a stricter total-work allowance; disclose both realized
generation and native graph estimates. No claim of equal realized FLOPs.

Verified IS reserves a full tree plus a 512-token answer and 96-token checker for
each possible new leaf. Initial worst-case cost is 3840 + 6*(512+96) = 7488.
The bounded dynamic probe costs at most 256+96+224=576, leaving room under 8192.
Never choose among partial weights. Reuse cached scores/answers only for an
identical suffix under the same immutable evidence root. Terminal/empty leaves
remain in the pool; no hidden rejection sampling or free retries.

## Retrieval and prefix repair

After the first complete reasoning step with at least 32 confirmed tokens, one
paid gap check anchors product/version/edition to the original question. The
provisional plan is explicitly treated as a hypothesis. Scan at most 256 main
tokens. Natural committed termination is respected.

Fuse question-only BM25 and question+gap BM25 rankings with 2:1 reciprocal rank
weights, retaining eight parents. Rank 384-token windows with stride192, expose
six ranked windows plus the final window of each of the top two parents. This
makes document solutions available even when only symptoms share query terms.
No answer labels or gold parent IDs participate in retrieval.

A paid greedy extractor (224 tokens maximum) checks entity/edition/version and
whether an exact contiguous excerpt supplies a requested fact, including relevant
exceptions. It accepts at most one excerpt, packaged within the original 256-input
token cap. Never truncate an accepted quote to fit; reject oversized/invalid
outputs. The extractor checks every assumption in the confirmed prefix. If the
prefix is unsafe, roll back to reasoning offset0 (a complete-step boundary).
Otherwise preserve the exact saved prefix. Provisional drafts are always discarded
at intervention. Rebuild the original user reference region; no assistant-inline
injection. These are paid inference-only adaptations, not trained Self-RAG/CRAG.

Static anchored arms pay the same extraction/checking without a main-model probe.
Real/noise share the real gate and rollback. Only body token IDs are replaced;
titles/wrappers remain as in the prior control. The verifier sees the actual
relevant/noise context. No-packet twins share the exact same execution.

## Answer-grounded reward

Each complete thinking candidate generates a final answer (up to512) once. A
separate greedy 96-token check scores coverage0/1/2, unsupported material facts,
scope errors and source contradictions against the original question and the
actual exposed reference. Positive coverage requires a literal 3–16-word source
quote; malformed output receives a pessimistic score. Reward is coverage/2 minus
each error flag plus 0.01*tanh(SC/10). SC is a small tie breaker. The reward
temperature remains .25 and selection is the existing complete-pool block-Met
index update. This remains a noisy self-verifier, not an oracle or proof of
correctness. Final answers are selected within the budget, never re-generated
after scoring.

## Evaluation and diagnostics

Report token F1/normalized EM with their known limits on long technical answers;
inspect old wrong-entity cases and a fixed sample of fresh cases. Also report gate
acceptance, prefix rollback, invalid extractor/checker output, source group hit and
evidence coverage offline, actual tokens/FLOPs/time, early thinking EOS, immediate
post-injection close/EOS, forced suffix closure, final-answer truncation and empty
answers. Preserve all candidate IDs/statistics/answers/verdicts and raw errors.

Relevant primary work: [Self-RAG](https://arxiv.org/abs/2310.11511),
[CRAG](https://arxiv.org/abs/2401.15884v3), and
[Sample, Scrutinize and Scale](https://arxiv.org/abs/2502.01839v2). These motivate
retrieval quality checks and verification allocation; they do not establish that
this untrained adaptation improves TechQA. The SC control isolates the new
retrieval/allocation package; verified-vs-SC compares answer-scored search with
thinking-scored search, including its extra generation cost.
