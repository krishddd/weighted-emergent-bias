"""Value types for the intervention layer (M4).

Frozen and self-validating, like the rest of the library's value types. ``TrustScore`` clamps its
denominator so the CortexDebate ``T = (C+R+I)/S`` heuristic cannot blow up when self-orientation
approaches zero (the review's ``S → 0 ⇒ T → ∞`` hazard).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..types import Payload
from .router import InterventionStrategy

_S_FLOOR = 1e-3  # clamp on self-orientation so trust cannot diverge


class SkepticStance(str, Enum):
    """One skeptic's verdict on a disputed output."""

    STANDS = "stands"
    """The output survives scrutiny."""

    REVISE = "revise"
    """The output is biased and a revision is proposed."""

    REJECT = "reject"
    """The output should be discarded outright."""


class PanelDecision(str, Enum):
    """The panel's aggregate decision after weighing verdicts."""

    STANDS = "stands"
    REVISED = "revised"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class SkepticVerdict:
    """One skeptic agent's structured critique of a disputed output."""

    agent_type: str
    stance: SkepticStance
    critique: str
    confidence: float
    has_evidence: bool = False
    proposed_payload: Payload | None = None

    def __post_init__(self) -> None:
        if not self.agent_type:
            raise ValueError("agent_type must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class TrustScore:
    """A CortexDebate-style trust score ``T = (C + R + I) / S`` with a clamped denominator.

    Treated as a heuristic (see docs/plans/PHASE-4.md R2): ``S`` (self-orientation / overconfidence)
    is best set from normalized entropy, and the whole score is opt-in — uniform aggregation is the
    default baseline it must beat.
    """

    credibility: float
    reliability: float
    intimacy: float
    self_orientation: float

    def __post_init__(self) -> None:
        for name in ("credibility", "reliability", "intimacy", "self_orientation"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def value(self) -> float:
        """The trust weight, with ``S`` clamped to ``_S_FLOOR`` to stay finite."""
        return (self.credibility + self.reliability + self.intimacy) / max(
            self.self_orientation, _S_FLOOR
        )


@dataclass(frozen=True, slots=True)
class PrunedAgent:
    """An agent removed from aggregation, with the reason (never "dissented" alone)."""

    agent_type: str
    reason: str


@dataclass(frozen=True, slots=True)
class PanelResult:
    """The skeptic panel's outcome: a decision, a corrected payload, and an audit of pruning."""

    decision: PanelDecision
    corrected_payload: Payload | None
    confidence: float
    verdicts: tuple[SkepticVerdict, ...] = ()
    pruned: tuple[PrunedAgent, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class InterventionResult:
    """The result of running a mitigation strategy on a halted run."""

    strategy: InterventionStrategy
    repaired_payload: Payload | None
    converged: bool
    trail: tuple[str, ...] = ()
