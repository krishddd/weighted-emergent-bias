"""weighted-emergent-bias: runtime circuit-breaker for DoT in multi-agent LLM systems.

Pre-alpha. See docs/DESIGN.md and docs/plans/PHASE-1.md.
"""

from __future__ import annotations

from .clients import EmbeddingClient, LLMClient
from .scoring import AxisSpec, Substitution, perturb
from .types import (
    Axis,
    BiasScore,
    DivergenceMethod,
    NodeId,
    Payload,
    Perturbation,
    PerturbationKind,
    TaskMode,
)

__version__ = "0.0.0"

__all__ = [
    "Axis",
    "AxisSpec",
    "BiasScore",
    "DivergenceMethod",
    "EmbeddingClient",
    "LLMClient",
    "NodeId",
    "Payload",
    "Perturbation",
    "PerturbationKind",
    "Substitution",
    "TaskMode",
    "__version__",
    "perturb",
]
