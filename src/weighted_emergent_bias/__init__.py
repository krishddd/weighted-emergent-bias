"""weighted-emergent-bias: runtime circuit-breaker for DoT in multi-agent LLM systems.

Pre-alpha. See docs/DESIGN.md and docs/plans/PHASE-1.md.
"""

from __future__ import annotations

from .accumulation import NetworkAccumulator, NetworkState, node_magnitude
from .breaker import CircuitBreaker, ControlMachine, freeze
from .calibration import ThresholdConfig, calibrate_thresholds
from .clients import EmbeddingClient, LLMClient
from .intervention import (
    InterventionContext,
    InterventionResult,
    InterventionStrategy,
    PanelDecision,
    PanelResult,
    SkepticAgent,
    SkepticPanel,
    SkepticStance,
    SkepticVerdict,
    TrustGraph,
    TrustScore,
    route_intervention,
)
from .scoring import (
    AxisSpec,
    LOOCProbe,
    ProbeError,
    Substitution,
    compute_local_bias,
    perturb,
)
from .topology import AgentDAG, WeightResult, dependency_weights, katz_weight
from .types import (
    Axis,
    AxisFailure,
    AxisScore,
    BiasScore,
    BreakerAction,
    BreakerDecision,
    BreakerState,
    DivergenceMethod,
    NodeId,
    Payload,
    Perturbation,
    PerturbationKind,
    ProbeResult,
    TaskMode,
)

__version__ = "0.3.0"

__all__ = [
    "AgentDAG",
    "Axis",
    "AxisFailure",
    "AxisScore",
    "AxisSpec",
    "BiasScore",
    "BreakerAction",
    "BreakerDecision",
    "BreakerState",
    "CircuitBreaker",
    "ControlMachine",
    "DivergenceMethod",
    "EmbeddingClient",
    "InterventionContext",
    "InterventionResult",
    "InterventionStrategy",
    "LLMClient",
    "LOOCProbe",
    "NetworkAccumulator",
    "NetworkState",
    "NodeId",
    "PanelDecision",
    "PanelResult",
    "Payload",
    "Perturbation",
    "PerturbationKind",
    "ProbeError",
    "ProbeResult",
    "SkepticAgent",
    "SkepticPanel",
    "SkepticStance",
    "SkepticVerdict",
    "Substitution",
    "TaskMode",
    "ThresholdConfig",
    "TrustGraph",
    "TrustScore",
    "WeightResult",
    "__version__",
    "calibrate_thresholds",
    "compute_local_bias",
    "dependency_weights",
    "freeze",
    "katz_weight",
    "node_magnitude",
    "perturb",
    "route_intervention",
]
