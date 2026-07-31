"""The optional pre-trigger verifier -- factual grounding, kept deliberately out of the core.

The 2026-07 review asked for a verification layer between detection and the breaker, so the system
halts on outputs that are *wrong* rather than only ones that are demographically non-invariant. That
was a scope fork, and the decision (DESIGN section 8, PHASE-2 R3) was to ship it as an **optional
injectable hook** rather than fold factual checking into the core -- because the noise-floor
apparatus supports the counterfactual claim and nothing broader.

This module is that hook. It is opt-in: no core module imports it, `Verifier` has no default
implementation, and a run without one behaves exactly as before. What it adds is a place for a
user-supplied oracle (retrieval, a formal checker, a database lookup) to speak *before* the breaker
decides, plus the honest distinction the review was really asking for:

    counterfactual bias  -- what M1 measures (invariance, anchored to the node's own noise floor)
    factual error        -- what only an external oracle can measure

:func:`gate_decision` keeps them separate rather than blending them into one number, so a run can
report "biased", "unverified", or both without the audit trail losing which is which.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ..types import Payload


class VerificationStatus(str, Enum):
    """What an external oracle concluded about a payload's factual accuracy."""

    VERIFIED = "verified"
    """The oracle found supporting evidence."""

    REFUTED = "refuted"
    """The oracle found the claim contradicted -- a factual error, not a bias signal."""

    UNVERIFIED = "unverified"
    """The oracle could not reach a conclusion. Explicitly *not* the same as ``VERIFIED``."""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """One oracle's verdict, kept distinct from any bias score."""

    status: VerificationStatus
    confidence: float
    evidence: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def blocks_promotion(self) -> bool:
        """True only when the oracle actively refuted the payload.

        ``UNVERIFIED`` deliberately does **not** block: an oracle that failed to reach a conclusion
        is not evidence of error, and treating silence as refutation would halt healthy runs.
        """
        return self.status is VerificationStatus.REFUTED


class Verifier(Protocol):
    """A user-supplied factual oracle. There is intentionally no default implementation."""

    async def verify(self, payload: Payload, context: str = "") -> VerificationResult: ...


VerifierHook = Callable[[Payload], Awaitable[VerificationResult]]


@dataclass(frozen=True, slots=True)
class GateOutcome:
    """The combined pre-trigger verdict, with the two signals kept separate."""

    bias_halts: bool
    verification: VerificationResult | None

    @property
    def halts(self) -> bool:
        """Halt if bias breached *or* an oracle refuted the payload."""
        return self.bias_halts or (
            self.verification is not None and self.verification.blocks_promotion
        )

    @property
    def reason(self) -> str:
        parts: list[str] = []
        if self.bias_halts:
            parts.append("counterfactual bias breached threshold")
        if self.verification is not None and self.verification.blocks_promotion:
            parts.append(f"factually refuted: {self.verification.detail or 'no detail'}")
        if not parts:
            if self.verification is not None:
                return f"clear (verification: {self.verification.status.value})"
            return "clear"
        return "; ".join(parts)


async def gate_decision(
    payload: Payload,
    *,
    bias_halts: bool,
    verifier: Verifier | None = None,
    context: str = "",
) -> GateOutcome:
    """Combine the bias verdict with an optional external verification, without conflating them.

    With no ``verifier`` this is a pass-through: the outcome is exactly the bias decision, which is
    why the core is unaffected by this module's existence.
    """
    result = await verifier.verify(payload, context) if verifier is not None else None
    return GateOutcome(bias_halts=bias_halts, verification=result)
