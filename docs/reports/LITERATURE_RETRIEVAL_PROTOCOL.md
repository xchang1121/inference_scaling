# IRCoT / ThinkRetrieve with a frozen IS reader

Frozen before native smoke and formal generation, 2026-11-12.

## Question and scope

Can changing retrieval and injection improve TechQA word F1 while leaving the
previous Block-Metropolized IS + Self-Certainty implementation and parameters
unchanged? This is a small, paired mechanism study, not a reproduction of the
papers' benchmark scores. The 16 DEV questions and seeds 20261112/20261113 are
exactly those in `online-injection-cot-bon-20261110`. No reselection or tuning on
this round's DEV outputs. Two disjoint TRAIN fixtures are used for native smoke.

## Frozen reader

Import `GuidanceConfig`, `GuidanceLedger`, `checkpoint`, `search`, and
`finish_answer` from the previous experiment without editing that module or the
underlying IS implementation. Model Qwen3.8-27B, same revision and Ascend runtime;
sampling temperature 1, full support; M=3, K=2, block=512, SC temperature=.25,
block-first, fixed exploration, block-metropolized selection. Search work cap
65,536; total retained generated thinking 8,192; final answer cap 16,384, continued
in 4,096-token chunks; context 32,768. No EOS suppression or minimum answer length.

All controller generation, discarded previews, and interim-answer tokens are
subtracted from the 65,536 work allowance. The final answer has the same separate
16,384 cap in every arm. Thus total generated-token allowance is at most 81,920.
Injected text is input and is reported separately. The IS horizon continues to
be 8,192 minus retained generated reasoning, as in the previous interface;
algorithm, M/K, block size, score and acceptance are unchanged. Equal allowances
do not imply equal realized token counts, forward work, or elapsed time.

## Controllers and deviations from the papers

**IRCoT:** BM25 question retrieval, then at most two sentence-generation / latest-
sentence-retrieval rounds, accumulating previously unseen token windows. The
retrieval agent conditions on all accumulated references and previous sentences.
Stop early if the agent closes thinking or no complete sentence is found within
256 generated tokens. A fresh reader receives the accumulated references in its
user prompt and uses the unchanged 64-token-minimum/512-token-max checkpoint and
IS. Agent CoT is not copied into this reader. Up to three 256-token reference
packets, from BM25 top-8 parent documents and ranked 192-token / 96-stride windows.
Static facts retrieve three windows in one question-only search. No gold-selected
parent, document, window, query, or answer is available to runtime retrieval.

The IRCoT noise arm repeats the **same relevant retrieval agent** with the same
seed, but replaces all retrieved titles and body text at the independent reader
with equal-length unrelated prose. It measures reader evidence content, not the
effect of corrupting the retrieval agent itself. This choice permits an exact
content comparison without leaking agent reasoning into the final reader.

**ThinkRetrieve:** generate a reasoning segment until natural `</think>` or
2,048 tokens, elicit an interim answer (maximum 512 tokens), embed original
question + interim answer, retrieve one unused worked example, append it to the
preserved thinking prefix, and repeat for at most two insertions. Then hand off
to the unchanged IS reader. Intercepted thought-closing tokens are recorded and
removed before continuation, following the official trace-based mechanism. A
sampled EOS remains terminal and yields an explicit empty-answer event. This
controller has a smaller insertion count and interval than the public library's
8 / 6,144 defaults to reserve thinking and search capacity for the fixed IS.

Example retrieval uses E5-base-v2 joint-QA normalized embeddings and exact cosine
search on CPU (equivalent to an exact normalized inner-product FAISS index).
Both sides use the official `query: question\nsolution: answer` convention.
Encoder revision and file hashes are preserved. E5's 512-token encoder limit is
reported; injected examples remain complete. This uses the paper's E5-base
variant rather than the public library's default E5-large. No new LLM-generated
reasoning is used to manufacture examples.

The example bank contains answerable TRAIN questions with complete procedural
solutions verified as whitespace-normalized substrings of their source documents.
Exclude every document group shared with the 16 DEV questions or two smoke
fixtures, and question word-set Jaccard >= .65 against any held question. Admit
only complete question+solution bodies <=1,024 model tokens. This is a technical
procedure bank, not mathematical CoT data. Bank construction never sees model
results. Runtime never receives DEV gold answers.

ThinkRetrieve noise uses its own evolving reasoning and interim queries, but
replaces the complete example body with unrelated tokens matching the relevant
arm's body length at each insertion ordinal. It shares only intervention lengths,
not relevant reasoning or example contents. Differences in early EOS or insertion
count are retained and explicitly audited; no silently repaired pair. The static
example arms retrieve two examples from the original question. The rethink arm
uses the same two segment boundaries and only the continuation phrase, with no
retrieval and no interim-answer generation.

## Arms

1. Direct: ordinary one-trajectory thinking and final answer, no retrieval or IS.
2. Closed IS: unchanged checkpoint + Block-Metropolized SC, no injected content.
3. Static fact references + IS.
4. Static fact references replaced by equal-length unrelated prose + IS.
5. IRCoT retrieval agent + IS reader.
6. IRCoT reader with equal-length unrelated references + IS.
7. Static worked examples + IS.
8. Static worked examples replaced by equal-length unrelated prose + IS.
9. ThinkRetrieve + IS.
10. ThinkRetrieve with equal-length unrelated example bodies + IS.
11. Rethink-only continuation + IS.

Total formal measurements: 16 questions x 2 seeds x 11 arms = 352. Two TRAIN
fixtures x one seed x 11 arms = 22 smoke measurements. All answers, including
empty ones, remain in denominators. Do not tune protocol using smoke F1.

## Cost, timing, and validity

Each question/seed's arms run on the same worker. Rotate arm order; move paired
ThinkRetrieve noise just after its reference to obtain lengths. Reset native
prefix cache before every arm, outside measurement. Every arm executes its full
controller afresh; no cross-arm output reuse or amortized latency masquerading
as measurement. Allow prefix reuse *within* an arm. Model loading, initial index
construction, graph warm-up and cache reset are excluded, and index construction
is recorded separately. Time includes controller, BM25/dense query encoding and
lookup, probe, complete IS candidate pools, and final answer generation.

Report F1 (0-100), paired question-cluster bootstrap intervals, total generation
tokens (including discarded candidates and interim answers), breakdown by phase,
injected input tokens, full requested prefix tokens, retrieval calls/seconds,
end-to-end seconds, and latency relative to the same question/seed's direct arm.
Include both ratio of mean elapsed time and geometric mean of paired ratios.
Native forward-token slots are useful additional work estimates, not exact FLOPs.

Record all sampled thought-close and EOS events, selected immediately closing
suffixes, controller-intercepted closes, empty final answers, forced thinking
closure, auxiliary length caps, final-answer truncation and actual maximum
lengths. If a final answer is truncated, preserve the failure and stop the study
before scoring; fix the issue before formal launch. Do not equate a thought-close
with EOS. Independently recompute costs, prefix preservation, rewards and IS
selection from raw calls. Verify frozen reader hashes before native runs.

## Sources

- IRCoT: <https://aclanthology.org/2023.acl-long.557/>;
  <https://github.com/StonyBrookNLP/ircot>.
- ThinkRetrieve: <https://arxiv.org/html/2608.10928v1>;
  <https://github.com/itsvaibhav01/ThinkRetrieve>, inspected commit
  `2fb909fdd2bf6aa9abe7310608e3bd2c3cf26704`.
- E5 model card: <https://huggingface.co/intfloat/e5-base-v2>.
