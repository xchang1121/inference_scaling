# Question-anchored retrieval with answer-verified IS

This follow-up tests the concrete failures in the preceding frozen pilot:
invented product assumptions, symptom excerpts mistaken for causal evidence,
lost edition/license exceptions, and self-certainty favoring unsupported claims.
All preceding experiments remain unchanged.

## Resources and validation split

The user explicitly requested checking and using more available cards. Read-only
inventory initially found A3 and B4/5/6 available and used them for the first
smoke. Subsequent B host-memory failures and an external job occupying B5 made
that allocation unstable. A6/7 were tried after their model processes exited but
failed device initialization while the old pair container remained running.
The user then explicitly requested stopping the container on A0/1 and moving
host198 experiments onto those two devices. B4's next startup also failed a
32-byte pinned-host allocation, so the final workers are A0 and A1, two isolated
TP1 replicas,160 formal conditions each. The A0/1 container was stopped as asked;
no other existing jobs or the shared main checkout are changed. Replicas improve
experimental throughput, not the per-answer compute budget.

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
exceptions. It selects source/span indices from numbered original sentences or
adjacent sentence pairs, each5–50 words and at most500 characters without URLs.
If overlapping pairs exceed the auxiliary context limit, only single sentences
are exposed. No model copying is needed; the program resolves the chosen index
to literal source text. The parser also retains exact-quote compatibility with
earlier tests, enforcing at most64 words/600 characters and rejecting URLs.
It accepts at most one excerpt, packaged within the original256-input
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

## Pre-formal smoke correction

The first native smoke on TRAIN_Q559 exposed a literal-copy formatting failure:
the extractor copied a very long download URL, exhausted its224-token output cap,
and returned truncated JSON. The original run remains archived. Before any formal
DEV execution, a second version constrained quotes to short facts and rejected
URLs. Its native TRAIN_Q545 run still copied a nonliteral table row or an overlong
paragraph, rejecting both static and dynamic evidence. Preserve both versions and
all infrastructure failures. The final pre-formal correction replaces text copying
with numbered source/span selection. Repeat all eight arms on A0 and A1,
covering both TRAIN fixtures. No gold answer or fresh-model outcome is used for
these formatting corrections. Formal assignments are fixed round-robin across
the two workers before execution, with every question's seeds/arms together.
The supplemental FORMAL-WORKER-PLAN.json overrides the initial three-worker
WORKER-PLAN.json; generator source and prepared questions remain unchanged.

## Supplementary routing policy fixed before fresh cases

At 2026-11-10 02:50:43 UTC, after the native smoke and while only regression
questions DEV_Q184 and DEV_Q119 had produced formal model units, a supplementary
policy was registered: use answer-verified IS if the shared extractor accepts an
evidence packet; otherwise use anchored SC. Static and dynamic versions apply
the same rule, and real/noise controls share the gate. Selection sees only packet
presence, never candidate answers, verifier scores or evaluation labels.

This rule is evaluated by exact dispatch among the already paired native arms;
it adds no model calls or independent trials. Every selected branch pays its
common probe/extraction cost. Preserve the original eight-arm analysis and label
the four supplementary outputs as derived views. Record realized total generated
tokens and verify each derived condition stays below 8192, including its answer.
The implementation also exposes a direct runtime dispatcher for future use;
the running native generator remains frozen at commit 2a806da.

The registration stores SHA256 hashes and byte lengths of the append-only native
logs at that time. Offline evaluation verifies these prefixes, the recorded
regression-only model-unit IDs, identical paid probes and evidence roots, and
all primary audits. This is a before-fresh registration, not a claim that the
policy predates the regression observations or the original study.
