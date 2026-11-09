# Retrieval + inference scaling pilot

See `reports/OBSERVATIONS.md` for conclusions and `reports/REPORT.md` for all
numbers. The archive contains frozen generation source, raw native generation,
all candidates and scores, protocol/input hashes, both TRAIN smoke questions,
offline gold and all retrieved/gold parent documents needed to inspect evidence.
Full BM25 re-retrieval requires the pinned public corpus; the included parent
subset is for inspection, not a replacement full-corpus ranking index.

Run `python unpack.py --output NEW_DIRECTORY` to extract and verify every archive
member. Python 3.12+ with
NumPy and Matplotlib suffices for offline audits, grading and figures. No model,
network, tokenizer or accelerator is used by these analysis commands. On Windows
set `PYTHONPATH` to `generation-source/src;analysis`; on POSIX use
`generation-source/src:analysis`. Run from the extracted root:

```sh
python analysis/summarize_retrieval_scaling.py --input snapshot/formal.jsonl --gold data/gold.jsonl --output rebuilt
python analysis/plot_retrieval_scaling.py --summary rebuilt/summary.json --output rebuilt/figures
python analysis/export_retrieval_scaling.py --snapshot snapshot --output rebuilt/traces
```

`MANIFEST.json` records every archive member's length and SHA256. Generation
source and model/runtime pins are in `protocol/FORMAL-FROZEN.json` and the
prepared inputs. Native re-execution is separate from the offline reproduction.
The native run used A physical NPU3, TP1; each stage stops its own container on exit.
