"""The unified inference entry: ``python -m inference_scaling``.

- ``cli``: the four choices (algorithm, model family, reward, dataset) and the output root
- ``settings``: the fixed settings file and its strict schema
- ``run``: results directory, resume and records
- ``ar`` / ``dllm``: each model family's backends and algorithms
- ``rewards``: the chosen reward bound to one problem
- ``records``: manifest identities, record IO and the summary
"""
