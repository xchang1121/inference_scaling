"""External retrieval and append-only context interventions for autoregressive models.

This experimental controller runs beside the sampling kernels. Evidence tokens
are context, never model samples or importance-sampling probability factors.
"""
from .corpus import Document, Hit, Retriever
from .controller import InjectionConfig, OnlineInjectionController
from .triggers import TriggerConfig

__all__ = ["Document", "Hit", "Retriever", "InjectionConfig", "OnlineInjectionController", "TriggerConfig"]
