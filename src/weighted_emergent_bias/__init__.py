"""weighted-emergent-bias: runtime circuit-breaker for DoT in multi-agent LLM systems.

Pre-alpha. See docs/DESIGN.md and docs/plans/PHASE-1.md.
"""

from __future__ import annotations

from .clients import EmbeddingClient, LLMClient
from .types import (
    Axis,
    BiasScore,
    DivergenceMethod,
    NodeId,
    Perturbation,
    PerturbationKind,
    TaskMode,
)

__version__ = "0.0.0"

__all__ = [
    "Axis",
    "BiasScore",
    "DivergenceMethod",
    "EmbeddingClient",
    "LLMClient",
    "NodeId",
    "Perturbation",
    "PerturbationKind",
    "TaskMode",
    "__version__",
]
