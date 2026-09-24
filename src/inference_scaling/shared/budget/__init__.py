"""Forward-token planning of budgeted IS; budget code never samples.

- ``joint``: pilot weight moments and the integer (B, M, K) choice
- ``planners``: next-block planners built on ``joint``
- ``costs``: reserved forward-token cost model used by the planners
"""
