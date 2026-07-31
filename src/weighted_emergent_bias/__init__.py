"""weighted-emergent-bias: runtime circuit-breaker for DoT in multi-agent LLM systems.

Pre-alpha. See docs/DESIGN.md and docs/plans/PHASE-1.md.
"""

from __future__ import annotations

from .clients import EmbeddingClient, LLMClient
from .scoring import (
    AxisSpec,
    LOOCProbe,
    ProbeError,
    Substitution,
    compute_local_bias,
    perturb,
)
from .topology import AgentDAG
from .types import (
    Axis,
    AxisFailure,
    AxisScore,
    BiasScore,
    DivergenceMethod,
    NodeId,
    Payload,
    Perturbation,
    PerturbationKind,
    ProbeResult,
    TaskMode,
)

__version__ = "0.1.0"

__all__ = [
    "AgentDAG",
    "Axis",
    "AxisFailure",
    "AxisScore",
    "AxisSpec",
    "BiasScore",
    "DivergenceMethod",
    "EmbeddingClient",
    "LLMClient",
    "LOOCProbe",
    "NodeId",
    "Payload",
    "Perturbation",
    "PerturbationKind",
    "ProbeError",
    "ProbeResult",
    "Substitution",
    "TaskMode",
    "__version__",
    "compute_local_bias",
    "perturb",
]
