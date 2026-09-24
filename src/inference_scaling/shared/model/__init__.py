"""Model-facing configuration shared by every model family.

- ``loading``: checkpoint resolution and per-role loading options
- ``generation``: generation length limits against model context windows
- ``prompting``: chat-template rendering and thinking-mode switches
- ``output`` / ``structured_output``: thinking/content segmentation of outputs

Algorithms never read these settings directly; experiment assembly turns them
into backends, sampling policies and length limits.
"""
