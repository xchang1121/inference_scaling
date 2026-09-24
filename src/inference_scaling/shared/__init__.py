"""Infrastructure shared by autoregressive and diffusion language models.

``sampling`` (algorithm kernels), ``budget`` (forward-token planning of IS),
``model`` (loading, prompting, generation limits, output parsing) and
``rewards`` (verifiers, votes and Consilience arithmetic).
"""
