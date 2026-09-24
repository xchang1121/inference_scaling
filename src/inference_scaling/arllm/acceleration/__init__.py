"""Distribution-preserving acceleration of rollout execution.

- ``primitives``: speculative drafting, rollout token trees, streaming rewards, run-ahead
- ``rollout_broker``: resumable partial-rollout scheduling shared by Transformers and vLLM
- ``vllm_suffix_proposer``: vLLM suffix-cache proposer that accepts dynamic draft lengths

These utilities depend only on the request contracts; backends and algorithms use them.
"""
