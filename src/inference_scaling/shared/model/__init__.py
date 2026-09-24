"""Model-facing configuration shared by every model family.

- ``loading``: checkpoint resolution and model identity
- ``generation``: generation length limits against model context windows
- ``prompting``: chat-template rendering and thinking-mode switches
- ``output`` / ``structured_output``: thinking/content segmentation of outputs

Algorithms never read these settings directly; the app turns them into
backends, sampling policies and length limits.
"""
