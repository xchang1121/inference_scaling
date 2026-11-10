# Short evidence in CoT: BoN and conditional IS

Start with [formal results](reports/formal/REPORT.md),
[all selected answers](reports/formal/answers.jsonl), and
[paired BoN budget curves](reports/formal/budget-curves.json).
The full protocol is [PROTOCOL.md](PROTOCOL.md).

16 fresh TechQA questions × 2 seeds × 4 evidence conditions × 8 selector views.
BoN selectors share one pool of 8 candidates. Both static and inline evidence
conditions branch after the same complete-step checkpoint policy; existing
reasoning is preserved. Gold is available only to offline evaluation.

Generation source and settings are frozen. Thinking horizon: 8192 tokens;
final-answer cap: 16384; context: 32768; generated search cap: 65536 including
the checkpoint. Final answers continue in chunks until EOS. Any answer cap
failure would invalidate completion. Actual cost varies between algorithms;
the raw traces and budget curves expose that variation.

Raw archives contain full sampled token IDs, SC/top-5 statistics, exact roots,
candidate pools, selection records, final answers, runtime metadata and a
tokenizer vocabulary subset. The `retired` directory preserves TRAIN-only
pre-formal attempts; those are excluded from the formal results.

Offline replay needs Python and NumPy, without a model or network:

```text
python tools/replay.py --package . --workdir /absolute/path/to/new-replay-folder
```

Replay verifies file hashes and rebuilds both reports byte-for-byte from raw
tokens, probabilities and local gold. `OFFLINE-VERIFICATION.json` records the
verified rebuild. Both task-owned native containers stopped after completion.
