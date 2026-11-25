"""Declarative backend capabilities shared by algorithms, CLI, and serving.

Algorithms and the serving layer query :func:`backend_capabilities` rather than
branching on concrete engine classes or hardware vendor names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    """Feature matrix exposed by an autoregressive backend instance."""

    backend: str = "custom"
    device: str = "unknown"
    streaming: bool = False
    native_continuous_batching: bool = False
    native_token_stops: bool = False
    scalar_self_certainty: bool = False
    pre_mask_statistics: bool = False
    exact_scoring: bool = False
    constrained_beam_search: bool = False
    sampling_constraints: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def backend_capabilities(backend: Any) -> EngineCapabilities:
    """Return declared capabilities of ``backend``, falling back to duck-typed flags."""

    getter = getattr(backend, "capabilities", None)
    if callable(getter):
        caps = getter()
        if isinstance(caps, EngineCapabilities):
            return caps
    inner = getattr(backend, "backend", None)
    if inner is not None and inner is not backend:
        base = backend_capabilities(inner)
        return EngineCapabilities(
            backend=base.backend,
            device=base.device,
            streaming=bool(getattr(backend, "supports_streaming_generation", base.streaming)),
            native_continuous_batching=bool(
                getattr(backend, "supports_native_continuous_batching", base.native_continuous_batching)
            ),
            native_token_stops=bool(getattr(backend, "supports_native_token_stops", base.native_token_stops)),
            scalar_self_certainty=base.scalar_self_certainty,
            pre_mask_statistics=base.pre_mask_statistics,
            exact_scoring=base.exact_scoring,
            constrained_beam_search=bool(
                getattr(backend, "supports_constrained_beam_search", base.constrained_beam_search)
            ),
            sampling_constraints=base.sampling_constraints,
        )
    return EngineCapabilities(
        backend=type(backend).__name__,
        device=str(getattr(backend, "device", "unknown")),
        streaming=bool(getattr(backend, "supports_streaming_generation", False)),
        native_continuous_batching=bool(getattr(backend, "supports_native_continuous_batching", False)),
        native_token_stops=bool(getattr(backend, "supports_native_token_stops", False)),
        scalar_self_certainty=bool(getattr(backend, "_scalar_self_certainty", False)),
        pre_mask_statistics=bool(getattr(backend, "_scalar_self_certainty", False)),
        exact_scoring=getattr(backend, "scoring_backend", None) is not None
        or hasattr(backend, "token_statistics"),
        constrained_beam_search=bool(getattr(backend, "supports_constrained_beam_search", False)),
        sampling_constraints=bool(getattr(backend, "_initial_strata_supported", False)),
    )


__all__ = ["EngineCapabilities", "backend_capabilities"]
