# IS from the start with online fact retrieval

This supersedes the fixed-controller-prefix integration for new online-IS runs.
Historical IRCoT/ThinkRetrieve artifacts retain their original implementation.

The logical IS prompt is exactly the prepared question and opening thinking marker.
There is no checkpoint, generated prefix, temporary answer, or retrieval-agent
generation before IS. `run_online_is` immediately invokes the original `search`
and `ConditionalISAdapter` with empty state and no carried statistics.

The original Block-Metropolized IS / SC implementation and `GuidanceConfig` are
unchanged: M=3, K=2, block=512, fixed exploration, block-first, selection-ready
disabled, SC temperature=0.25, full-support model sampling temperature=1,
8192 retained thinking tokens, 65536 generated search tokens, 16384 answer tokens
in 4096-token chunks, and 32768 context tokens. Existing complete-pool reservation,
selection, rollouts, and sequential block fixation still apply. Already fixed
blocks are not retroactively resampled.

An experimental backend adapter materializes physical context from each logical
candidate prefix. At generated-token positions 128 and 512, it queries the frozen
BM25 fact bank using the original question plus that branch's last 256 generated
thinking tokens. Each insertion is 256 input tokens including its source-data
wrapper; short bodies receive newline padding. The second insertion excludes the
first selected window. Trigger positions and retrieval limits are separate from
IS parameters. This is a custom online fact retriever, not an IRCoT or
ThinkRetrieve reproduction.

Evidence is cached by the exact generated prefix within a single run. Identical
prefixes reuse identical evidence. Different branches cannot inherit each other's
evidence. Native generation pauses exactly at the insertion boundary, appends the
evidence, and continues the same branch without discarding generated text.
Logical candidate IDs, block sizes, budgets and SC scores contain model-generated
tokens only. Native SC/logprob statistics come from the actual augmented context.
All discarded candidates and rollouts remain charged to generation cost.

Native RNG handling uses the backend's existing `uniform_offset` behavior: vLLM
derives an independent seed for a continued segment rather than replaying one
uninterrupted RNG stream. Thus retrieval-on runs do not promise token-identical
randomness to an unsplit generation. Retrieval-disabled requests retain original
prefixes, lengths, seeds and offsets.

Final readout reconstructs only the selected trajectory's evidence; it cannot
initiate an unused insertion exactly at the thinking-budget boundary. EOS and
thinking-close are never suppressed, empty answers are retained, and truncation
is recorded. No generated thought is moved into the immutable IS prompt.

`scripts/benchmark_online_is.py` exposes direct, closed_is, static_facts,
static_facts_noise, online_is, and online_is_noise arms. Static arms retrieve two
distinct windows from the original question, fix each packet to 256 tokens, and
insert them before the user-message end. They also start IS from zero generated
thinking tokens. Static-noise replaces every factual body with equal-length
unrelated prose. Online noise executes its own branch-local queries and
replaces every factual body token with unrelated prose at exactly the same
packet size. Branches can terminate at different positions, so total evidence
exposure need not match across arms; logs retain this difference. Index building,
model warmup and cache reset are outside unit timing; per-branch retrieval and
all generation are inside. This runner preserves the existing prepared data and
model settings, with separate outputs from previous experiments.

The metric run uses the same fixed 16 DEV questions and seeds 20261112/20261113:
192 fresh runs, with all six arms for a question on the same worker and a cold
prefix cache before each arm. Only A3 was both process-free and unbound at launch; all conditions use that
worker. Other cards were occupied by processes or bound to other active containers. A separate 12-run
TRAIN smoke validates all six arms before formal evaluation. No partial F1 is
used to tune the generator. Report token-overlap F1, paired question-bootstrap
differences, all generated tokens, pipeline latency relative to direct, actual
evidence exposure, EOS, empty answers and truncation. Actual empty/truncated
answers remain in the cohort and scoring; reaching the answer cap does not
discard or rerun that condition or prevent other conditions from finishing.
The 65536 search-token cap
is identical across IS arms; direct is the one-trajectory reference with 8192
thinking tokens. These are equal configured IS budgets, not equal realized cost.

Example (native Linux environment):

```sh
python scripts/benchmark_online_is.py --settings settings.json \
  --prepared prepared-smoke.json --corpus data/corpus.jsonl \
  --output smoke.jsonl --seeds 20261112
```

`tests/test_online_is.py` covers zero-state entry, frozen-IS equivalence with
retrieval disabled, candidate isolation, replay, actual termination, accounting,
noise lengths, and final answer context. Native validation receipts are stored
separately; unit tests alone are not evidence of model answer quality.
