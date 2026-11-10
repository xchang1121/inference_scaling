# Grounded dynamic retrieval + inference scaling

See `reports/OBSERVATIONS.md` for interpretation and `reports/REPORT.md` for all
320 formal conditions. After multi-card smokes and host failures, the user moved
host198 experiments to A0/A1, two isolated TP1 replicas; each paired question
block stayed on one device. The independent fresh sample is12 questions,
with two seeds, and the8 old regressions are reported separately.

The archive preserves every formal candidate, auxiliary call, score and selected
answer, all three native smoke versions, initialization failures, frozen generation
source, model/runtime hashes, the exact patched runner and804 custom operators,
offline gold, evidence provenance and all retrieved/gold parent documents.
Full BM25 re-retrieval requires the pinned full public corpus; the included parent
subset is for inspection only. Model weights are hash-pinned, not redistributed.

Run `python unpack.py --output NEW_DIRECTORY` to verify and extract every member.
Python3.12+, NumPy and Matplotlib suffice for offline analysis; pytest also runs
the CPU contracts. The actual local CPU versions are recorded separately in
`protocol/analysis-environment.json`, with package pins in
`analysis/requirements-offline.txt`. Byte-identical replay is verified in that
environment; plot bytes may differ with other versions or platforms.
Set PYTHONPATH to `generation-source/src;analysis` on Windows,
or `generation-source/src:analysis` on POSIX. From the extracted root:

```sh
python analysis/summarize_grounded_retrieval.py --snapshot snapshot/formal --stage formal --gold data/gold.jsonl --output rebuilt
python analysis/export_grounded_retrieval.py --snapshot snapshot/formal --stage formal --output rebuilt/traces
python analysis/diagnose_grounded_retrieval.py --snapshot snapshot/formal --stage formal --gold data/gold.jsonl --output rebuilt
python analysis/evaluate_grounded_route.py --snapshot snapshot/formal --graded rebuilt/graded.jsonl --registration protocol/adaptive-policy-preregistration.json --output rebuilt
python analysis/validate_grounded_review.py --notes reports/manual-case-notes.jsonl --cases rebuilt/fixed-case-review.jsonl --manifest data/manifest.json --diagnostics rebuilt/evidence-diagnostics.json --output rebuilt/manual-review-audit.json
python analysis/plot_grounded_retrieval.py --summary rebuilt/summary.json --output rebuilt/figures
```

Rebuilding the first smoke uses `generation-source-smoke-v1/src` on PYTHONPATH,
because its original quote parser allowed longer excerpts. The second smoke uses
`generation-source-smoke-v2/src`. The final native
generation source commit is recorded in `protocol/generation-revision.json` and
every worker's `protocol/SOURCE-FROZEN.json`. The final two-worker assignment
overrides the initial three-worker plan in `protocol/FORMAL-WORKER-PLAN.json`
inside each formal snapshot. Analysis may have later commits.
No model, network, tokenizer package or accelerator is needed for offline replay.
Native re-execution is a separate operation with the pinned weights, corpus,
base container image, operator overlays and CANN environment shown in the records.

Same8,192 generated-work cap does not mean equal realized FLOPs. Verified IS pays
candidate answers and checkers inside that cap; SC arms pay one final answer
separately. F1 measures lexical overlap, not factual correctness.

The eight reused regression questions retain a concise-answer suffix; the twelve
fresh questions use the original benchmark wording without that suffix. All arms
for a given question share identical wording. This frozen difference, discovered
during the fixed case review, limits comparisons of answer length, truncation and
F1 between cohorts. See `protocol/prompt-scope-audit.json`; no prompt or selection
rule was changed after inspecting fresh answers.

SC final answers use the inherited32-token chunks and call-counter-based seeds;
verified leaves use one call of up to512 tokens with a suffix-based seed.
Dynamic probes can therefore change an SC answer's sampling seed even when its
final input matches the static arm. Context diagnostics expose exact prefix,
suffix and first-answer-seed comparisons; do not attribute those answer changes
to retrieval timing alone.

`reports/ADAPTIVE.md` evaluates a supplementary gate-based policy registered
before any fresh model unit: accepted evidence selects verified IS; otherwise
anchored SC. Its160 outputs are exact derived views of the320 native conditions,
not additional trials. The registration's append-only log prefixes are verified
during offline replay. `analysis/grounded_route.py` is a later policy module;
it is not part of the frozen native generation source.

Offline handoff verification passed:973 archived members,107 rebuilt result/trace/figure files byte-identical, and47 CPU tests. The exact archive hash is bound in [OFFLINE-VERIFICATION.json](OFFLINE-VERIFICATION.json). This receipt is outside the archive to avoid a self-referential hash. The first packaging check exposed a Windows case-insensitive lookup of the local worker plan; the corrected verifier prioritizes the frozen worker namespace. The failed-attempt receipt is retained in the archive. Native generation and frozen inputs were unchanged.
