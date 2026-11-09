# Step-boundary FLARE + evidence-gap experiment

Frozen design, 2026-11-09. This is a new inference-only combination inspired by
[FLARE](https://aclanthology.org/2023.emnlp-main.495/) and external context
sufficiency checks, not a reproduction of a trained retrieval policy.

## Inputs and budgets

- Qwen3.8-27B revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, same pinned
  tokenizer/runtime as the previous ablation. A=27.106.124.198 physical NPU3
  only, TP1. The prior 312/336 results and archives remain frozen.
- 8 old DEV questions are regression cases. 8 newly selected answerable DEV
  questions are held out from this implementation: SHA256-ranked IDs, gold
  document groups disjoint from prior gold/frozen BM25 groups and one another.
  Selection uses metadata, never model performance. Report cohorts separately.
- Two disjoint TRAIN questions test format/runtime contracts before the formal
  run. Gold remains local and is never deployed to the model or retriever.
- Every condition has 1024 generated work tokens and 512 final answer tokens.
  Work includes retained thoughts, discarded previews, thinking end/EOS tokens,
  and all planner/checker output. No effort instruction in the rendered prompt.
  Thinking is enabled for the main model, disabled for independent checkers.
- At most one retrieval attempt and one accepted packet, at most 256 evidence
  tokens including labels/wrappers. Evidence input tokens are not generated
  work tokens; their prefill/cache/graph cost is measured. A token cap does not
  imply identical FLOPs or equal actual work used by naturally ending samples.

## Conditions and paired execution

Closed book; static RAG; legacy SC; step-boundary SC; FLARE-gap. Every retrieval
method has relevant and exactly length-matched unrelated bodies: 11 conditions
per question, 176 formal condition outcomes, one seed per question. These are
not 176 independent pre-intervention generations.

For each question/method, execute one common pre-intervention unit, including
all drafts and checks. Save its exact token IDs. Cold-reprefill relevant and
unrelated continuations separately with APC cleared. Each condition is charged
the full measured common unit plus its branch. Study-wide actual cost sums
unique executed units. If no intervention is accepted, both content conditions
reference the same completed unit and are explicitly marked as shared results.
No native KV fork, seed-replay equivalence, or independent replication is claimed.

Main sampling is full-support temperature 1, 32-token requests, with actual
sampled log probabilities and KL(U||p) Self-Certainty. Main and auxiliary seed
streams are separate. Auxiliary generation is greedy top-1, at most 96 tokens
per check. Its independent prompts/context and all output are logged and charged.

## Trigger, query, boundary, placement

Legacy SC retains the previous 5% relative SC drop, 32-token window, minimum 64
thinking tokens, 128-token cooldown, assistant-inline packet. It now uses the
same short window retriever/packet formatter as the other methods, so it is the
existing trigger criterion, not a bit-for-bit replay of the prior long-doc arm.

Step SC evaluates that criterion only after complete steps. FLARE-gap keeps a
next completed step provisional. After at least 64 confirmed thinking tokens,
the first eligible step gets a gap check even if confident; later candidates
require a non-stopword alphanumeric token with p<0.15. At most two planner
checks, separated by 64 confirmed tokens, and only if at least 320 work tokens
remain (two 96-token checks plus 128 main tokens). These constants are fixed
before formal outputs; no performance-based calibration on either DEV cohort.

Sentence punctuation or a blank-line paragraph outside code defines a complete
step. Decimal/version dots, common abbreviations, numbered list prefixes, inline
code and fenced code do not create boundaries. A crossing tokenizer token is
rejected unless its remainder contains whitespace only. Incomplete steps cannot
trigger insertion. Retained generated tokens are never decoded and re-encoded.

The planner returns a missing fact and scoped query. The corpus-only retriever
BM25-selects 8 parent documents, splits their bodies into 192-token windows with
96-token stride, BM25-ranks windows, exposes at most 2 short bodies. Static/SC
use the question query; FLARE-gap uses question plus planned missing-fact query.
A second checker must say the exposed evidence supplies that fact and copy
5–24 words verbatim from an exposed body. JSON/type/source/quote failures skip
insertion without retries. This gate can be semantically wrong; it is no oracle.

For FLARE-gap, discard the unconfirmed draft/tail only after the gate accepts.
The primary FLARE-gap continuation appends its short packet directly after the
confirmed complete assistant reasoning step. A companion `flare_gap_rebuild`
condition shares that same common stem, trigger, payload and work already spent,
and changes only reference placement to the user region.
For step SC discard only the lookahead tail past its accepted boundary. Step SC
and the FLARE-gap companion rebuild the original user reference region, preserve
the confirmed thinking IDs, and regenerate from that boundary. Static uses the
same reference region at offset zero. Legacy SC keeps the old assistant-inline
placement. The FLARE-gap format pair directly tests placement from one exact
checkpoint; step SC versus the rebuilt FLARE-gap companion further compares the
additional trigger/query/gate changes. This does not isolate every component.

Unrelated bodies are substituted **after** the real-evidence gate accepts.
Source labels/wrappers, trigger checkpoint and payload length are identical.
This is a deliberate content perturbation, not an independent test of whether
the gate rejects unrelated evidence. Retrieved corpus text is sanitized against
chat/thinking marker injection. No gold-derived query, body selection or judge.

## Required measurements

Preserve every request prefix, seed, raw generated IDs, log probabilities,
SC values, auxiliary output, parse failures, discarded draft, packet and body
offsets. Report generated work/auxiliary/retained/discarded/answer/evidence tokens,
retrieval and acceptance coverage, token F1/EM against local gold, latency,
native token accounting, native graph token slots and dense FLOP estimates.
The estimates do not include every attention/host operation.

Count natural thinking close, forced budget close, immediate post-insertion
thinking close, immediate EOS, EOS inside thinking, empty final answer, answer
cap, provisional/discarded EOS and failed execution separately. Never mask EOS,
force-reopen thinking, drop empty answers or retry based on quality. Paired
question bootstrap intervals describe this small sample, not independent seeds.
Smoke/initialization/failed attempts are kept separate from formal results and
included in study resource receipts. Raw traces and source/input/runtime hashes
must support an offline structural audit before handoff.

Before any formal generation, the design added paired literal assistant-step insertion and user-region reconstruction. The initial two-TRAIN smoke source used reconstruction only; its outputs remain a separately versioned runtime check. The final source receives a focused two-TRAIN FLARE smoke check (8 format/content outcomes) before formal launch; unchanged baselines are covered by the original 18-outcome smoke. No DEV model outcomes informed this change.

The focused final-source smoke also includes a labeled synthetic positive contract for the supported-evidence path: a fictional Mercury API v2 source explicitly specifies 27 seconds. Its provided reasoning/draft is input rather than sampled prehistory. Planner/checker generations and two short format continuations are recorded as separate probe cost, excluded from formal conditions and quality scores. The probe never supplies a DEV answer or changes formal thresholds.
