"""Deterministic test doubles for the intervention layer (M4).

Ground-truth fakes so the panel / trust graph / MADERA can be exercised without a real model:
a skeptic that returns a canned verdict, and (for MADERA) fakes that diagnose, retrieve, and edit
toward a known unbiased target.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

from ..intervention.types import SkepticStance, SkepticVerdict
from ..types import Payload


class FakeSkeptic:
    """A ``SkepticAgent`` giving a fixed verdict, optionally delayed (for concurrency tests)."""

    def __init__(
        self,
        agent_type: str,
        stance: SkepticStance = SkepticStance.REVISE,
        *,
        confidence: float = 0.8,
        has_evidence: bool = True,
        revision: Payload | None = None,
        critique: str = "canned critique",
        delay: float = 0.0,
    ) -> None:
        self.agent_type = agent_type
        self._stance = stance
        self._confidence = confidence
        self._has_evidence = has_evidence
        self._revision = revision
        self._critique = critique
        self._delay = delay

    async def review(self, disputed: Payload, context: str = "") -> SkepticVerdict:
        if self._delay > 0.0:
            await asyncio.sleep(self._delay)
        return SkepticVerdict(
            agent_type=self.agent_type,
            stance=self._stance,
            critique=self._critique,
            confidence=self._confidence,
            has_evidence=self._has_evidence,
            proposed_payload=self._revision,
        )


# --- MADERA fakes: a payload with a "bias" field that an editor drives toward zero ----------


async def bias_field_scorer(payload: Payload) -> float:
    """Score a payload by its ``"bias"`` field (0 if absent). The ground truth for MADERA tests."""
    if isinstance(payload, Mapping):
        return float(payload.get("bias", 0.0))
    return 0.0


class FakeDiagnoser:
    """Returns a fixed diagnosis of the biased reasoning step."""

    async def diagnose(self, payload: Payload, context: str = "") -> str:
        return "unsupported leap from group to individual"


class FakeRetriever:
    """Returns canned counter-evidence (no real retrieval)."""

    def __init__(self, evidence: Sequence[str] | None = None) -> None:
        self._evidence = tuple(evidence) if evidence is not None else ("counter-example",)

    async def retrieve(self, query: str) -> Sequence[str]:
        return self._evidence


class DecayEditor:
    """Rewrites a payload's ``"bias"`` field down by a fixed factor -- models a repair that works.

    ``decay >= 1.0`` models a repair that does *not* reduce bias (for the non-convergence test).
    """

    def __init__(self, decay: float = 0.4) -> None:
        self._decay = decay

    async def rewrite(self, payload: Payload, diagnosis: str, evidence: Sequence[str]) -> Payload:
        bias = float(payload["bias"]) if isinstance(payload, Mapping) else 0.0
        base = dict(payload) if isinstance(payload, Mapping) else {}
        return {**base, "bias": bias * self._decay, "text": f"revised given: {diagnosis}"}
