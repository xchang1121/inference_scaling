# Frozen short evidence in the user prompt versus ongoing reasoning

Registered before native measurements on 2026-11-10. This is a new study; earlier
grounded retrieval runs and artifacts are immutable.

## Question and interventions

Does placing exactly the same short evidence in an unfinished reasoning chain
improve final TechQA lexical token F1, and which inference-scaling selector works
best? No history reset, evidence repair, folding, or retrieval-trigger learning.

One question-only BM25 + indexed exact-span extraction per question, frozen across
seeds/conditions. The existing grounded extractor uses no answer annotations.
Packet <=256 tokens. Static placement is before the user end marker. Inline
placement is after the earliest complete sentence/paragraph at or after 64
generated reasoning tokens, found in a <=512-token probe. Retain that exact
prefix; charge all probe work, including unused lookahead. If no boundary or
natural reasoning termination precedes it, do not invent an injection point.
Both static and inline groups use this SAME checkpoint stopping policy before
branching candidates. Static checkpoints are generated with evidence already in
the user message; inline checkpoints are generated without evidence and then
receive the packet. This avoids confounding evidence placement with shared-prefix
versus independent-from-start candidate generation. These are checkpointed BoN
and IS comparisons, not independently sampled whole-trajectory BoN. Naturally
finished checkpoints bypass search and are explicitly recorded.
Report these cases separately. Both placements use identical packet IDs and
wrapper. Noise replaces body tokens only, preserving length, labels and wrapper.
No accepted evidence means no packet; report all questions and packet-present
questions without selecting questions by measured outcomes.

## Frozen comparison

16 fresh answerable DEV questions, 2 seeds (20261112, 20261113), 4 contexts:
static relevant, inline relevant, static unrelated, inline unrelated. Selection
uses SHA256("cot-bon-20261110:" + ID), excluding previous exclusion/gold document
groups and duplicate gold groups. Training questions TRAIN_Q545/TRAIN_Q559 are
only smoke tests. Gold is local offline evaluation data, never deployed.

Per context, one shared BoN-8 candidate pool yields single (candidate 0),
BoN-Self-Certainty, BoN-Consilience and BoN-mean-logprob results. All 8 answers
are completed once, permitting paired selector audits and offline oracle ceilings
(oracle is diagnostic only and never an executable selector).

Four conditional IS runs: SIR-SC, Metropolized-SC, Block-Metropolized-SC,
Block-Metropolized-Consilience. M=3, K=2, block=512, complete pools only, fixed
schedule, no selection-ready approximation. SC temperature .25; Consilience 2.
SC is mean KL(U||p); Consilience uses top-5 negative mean logprob, default 5%
skip / 20% endpoint windows / initial penalty 3. Scores cover all retained
model-generated thinking (including pre-injection reasoning), excluding evidence,
terminal markers and final answers. Native generation returns BOTH statistics
from the same logits; no separate sampling or scoring forwards.

Thus 1024 logical selected answers, from 640 search units before exact no-packet
reuse. BoN's four views are paired selectors, not independent generation runs.

## Length and work

Model and probability policy unchanged: Qwen3.8-27B, BF16, temperature=1,
full support; medium chat-template setting without an explicit low instruction.
Context=32768; total retained thinking horizon=8192 (including inline prefix).
Search generated-token cap=65536 including the probe in BOTH placements. BoN completes
8 candidates; IS reserves each full pool before starting. Both have the same
maximum search cap, not necessarily equal realized work. Extraction and final
answer generation are accounted separately. Independent candidate answers are
batched, with per-candidate/per-chunk seeds. Candidate answers can require up to
16384 tokens each, in <=4096-token calls, until natural EOS. No short-answer
word cap. Common final instruction asks direct technical answers without repeating
analysis, identical in every group. A selected thinking horizon can still force
thinking close; record this independently from final-answer truncation.

Any final answer reaching its hard/context cap stops the study and is NOT accepted
as a completed scored run. Preserve failed traces before increasing limits.
Final-answer continuation seeds are keyed by question/run/selected suffix and
chunk, never global call counters. Backend does not promise chunking-invariant
randomness. Main search seeds are shared across placements/methods independently
of auxiliary calls. IS uses an immutable root within every search epoch.

## Evaluation and completion

Use the existing lexical F1 implementation. Average seeds per question before
paired bootstrap CIs; report raw scores, packet coverage, context-specific
rankings, position deltas and noise deltas. Single is a low-compute baseline.
Report search tokens, answer tokens, auxiliary tokens, prefill/decode costs and
latency; no equal-FLOPs claim. BoN selectors share generation costs, while final
readout costs differ from IS; expose both search-only and end-to-end costs.
Audit raw/selected immediate EOS and thinking-close, empty answers, forced
thinking ends, all answer truncations, packet identity and exact preserved prefix,
native SC/Consilience reductions, IS selection records and budget accounting.
Freeze generation source/settings before formal execution; do not tune on DEV F1.

Native resources: use currently idle devices in new task-owned containers;
leave previous artifacts and other workloads untouched. Save hashes, raw logs,
answers and reproducible offline analysis; stop own containers on completion.

Pre-formal revision: v1 smoke exposed the required extra SC transport slot
(6 rather than 5 logprob slots). v2 tested all native statistic/IS paths but was
retired during TRAIN-only smoke to match the checkpoint policy in both placements
and batch independent answer readouts. Preserve both attempts; no DEV generation
or F1-guided tuning preceded these corrections.
