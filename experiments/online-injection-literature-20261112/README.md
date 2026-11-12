# IRCoT and ThinkRetrieve with frozen Block-Metropolized SC

See [Chinese interpretation](reports/formal/RESULTS_ZH.md), [formal results](reports/formal/REPORT.md), [all answers and costs](reports/formal/answers.jsonl),
[the complete audit](reports/formal/audit.json), and [the frozen protocol](PROTOCOL.md).
The [overview](visuals/overview.png) compares F1, generation cost, latency and empty
answers. The [paired question view](visuals/question_differences.png) shows F1
differences relative to no-injection IS, with empty-answer markers. SVG originals
and the per-question comparison CSV are included in `visuals/`.
The [empty-answer diagnostics](diagnostics/EMPTY_ANSWERS.md) locate the observed
termination phase, distinguishing controller EOS, checkpoint EOS, selected IS
trajectory EOS and empty final readout. These locations do not establish why
the model selected EOS.

16 identical TechQA DEV questions × 2 seeds × 11 arms = 352 native measurements.
The independent TRAIN smoke contributes 22 measurements. The same IS algorithm,
SC reward, candidate/rollout counts, sampling, and generation limits are used in
all IS arms. Runtime retrieval receives no DEV gold. Every pipeline's controller,
interim answers, discarded candidates, and final answer are included in costs.
Cold-prefix-cache end-to-end latency is compared against a separately executed
ordinary generation on the same question, seed, and worker. Loading and initial
index construction are excluded and recorded separately.

These are explicit TechQA/native-token adaptations, not paper benchmark replicas.
IRCoT uses a separate retrieval agent and fresh IS reader. ThinkRetrieve uses E5
joint-QA retrieval over complete source-supported TRAIN troubleshooting examples,
intercepts thought-closing markers, and appends examples to retained thinking.
Actual EOS is never suppressed. The protocol explains the controls and differences.

Before formal evaluation, a TRAIN-only cold-cache diagnostic compared our initial
paraphrased insertion template against the official ThinkRetrieve template.
The official template improved continuation on one of two TRAIN fixtures, so V2
restored its exact text and example format. No DEV output or TRAIN F1 was used
to make this correction. The retired V1 source, all 22 V1 smoke measurements,
six diagnostic calls, and the failed diagnostic startup receipt are retained
under `retired/`. Formal results use V2 exclusively, after a fresh 22-run smoke.
The upstream MIT license is included under `licenses/`.

Raw archives include generated token IDs, SC values, exact prompt prefixes, all
IS pools and selection records, final answers, query and evidence records, native
cost counters, elapsed times, and a tokenizer-vocabulary subset. Reader source
hashes are frozen in `inputs/FROZEN_IS.json`. `generation-source.tar.gz` contains
the exact deployed source. E5 weights are not duplicated here; their pinned
revision and all file hashes are in `inputs/encoder-manifest.json`.
The exact retrieval corpus, original example-bank source, and unrelated text
are included under `inputs/`; large text inputs use gzip with uncompressed
SHA-256 values in `compressed-input-hashes.json`.

Offline replay needs Python and NumPy, without a model or network:

```text
python tools/replay.py --package . --workdir /absolute/path/to/new-replay-directory
```

Replay verifies package hashes and reconstructs both reports byte-for-byte from
raw records. The receipt is `OFFLINE-VERIFICATION.json`. Task-owned native
containers stopped after completion; unrelated workloads were left untouched.

To also regenerate and byte-check all visual artifacts, install the pinned
`analysis/plot-requirements.txt` dependencies and add `--figures` to the replay
command. `--plot-python /path/to/python` optionally selects a separate plotting
environment. The complete delivery was checked with this figure replay enabled.
Synthetic plotting fixtures are tests only and are never included in results.
