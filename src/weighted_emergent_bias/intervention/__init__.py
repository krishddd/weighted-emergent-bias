"""Intervention layer (M4) — routing seam shipped in M3, implementations to follow.

M3 ships only the *seam*: given how bias is distributed across the graph, decide which mitigation
strategy a halted run should route to. The strategies themselves (skeptic debate, MADERA repair)
are M4. The default classifier here is an explicit heuristic, not a solved bias-type diagnosis.
"""

from __future__ import annotations

from .router import InterventionContext, InterventionStrategy, route_intervention
from .types import (
    InterventionResult,
    PanelDecision,
    PanelResult,
    PrunedAgent,
    SkepticStance,
    SkepticVerdict,
    TrustScore,
)

__all__ = [
    "InterventionContext",
    "InterventionResult",
    "InterventionStrategy",
    "PanelDecision",
    "PanelResult",
    "PrunedAgent",
    "SkepticStance",
    "SkepticVerdict",
    "TrustScore",
    "route_intervention",
]
