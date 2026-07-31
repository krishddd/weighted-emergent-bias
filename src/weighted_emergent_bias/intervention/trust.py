"""Trust-graph aggregation of skeptic verdicts (M4).

Turns a panel's raw verdicts into a decision. Two knobs, and one hard safety rule:

- **Weighting** is the ablation dimension. ``uniform`` (the default *baseline*) counts every verdict
  equally; ``trust`` weights by a clamped CortexDebate score ``T = (C+R+I)/S``. Trust-weighting is
  opt-in and has to *beat* uniform in the WP7 study before it earns its place.
- **Pruning** (trust mode) drops **overconfident** verdicts only -- high certainty with no evidence.
  It is **never** allowed to prune a verdict for *dissenting* from the emerging consensus: trust is
  computed per verdict from its own content, with no term that references the other votes
  (anonymization). That is the guarantee that stops the panel silencing a correct minority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..types import Payload
from .types import (
    PanelDecision,
    PanelResult,
    PrunedAgent,
    SkepticStance,
    SkepticVerdict,
    TrustScore,
)

_STANCE_TO_DECISION = {
    SkepticStance.STANDS: PanelDecision.STANDS,
    SkepticStance.REVISE: PanelDecision.REVISED,
    SkepticStance.REJECT: PanelDecision.REJECTED,
}


class TrustGraph:
    """Aggregates verdicts into a :class:`PanelResult` under a weighting scheme.

    ``weighting="uniform"`` is a pure majority baseline (no pruning). ``weighting="trust"`` weights
    by trust and prunes overconfident verdicts. ``reliability`` optionally supplies a per-agent
    historical accuracy (a legitimate signal, distinct from rewarding conformity).
    """

    def __init__(
        self,
        *,
        weighting: str = "uniform",
        reliability: Mapping[str, float] | None = None,
        overconfidence_threshold: float = 0.9,
    ) -> None:
        if weighting not in ("uniform", "trust"):
            raise ValueError(f"weighting must be 'uniform' or 'trust', got {weighting!r}")
        self._weighting = weighting
        self._reliability = dict(reliability or {})
        self._overconfidence_threshold = overconfidence_threshold

    def _is_overconfident(self, verdict: SkepticVerdict) -> bool:
        # Overconfidence is high certainty with NO evidence -- never "disagreed with the majority".
        return not verdict.has_evidence and verdict.confidence >= self._overconfidence_threshold

    def trust(self, verdict: SkepticVerdict) -> TrustScore:
        """The clamped trust score for one verdict, computed from its own content only."""
        credibility = 1.0 if verdict.has_evidence else 0.3
        reliability = self._reliability.get(verdict.agent_type, 1.0)
        intimacy = 1.0
        # Self-orientation (overconfidence): high when confident without evidence, low when backed.
        self_orientation = verdict.confidence if not verdict.has_evidence else 0.25
        return TrustScore(credibility, reliability, intimacy, self_orientation)

    def _weight(self, verdict: SkepticVerdict) -> float:
        return 1.0 if self._weighting == "uniform" else self.trust(verdict).value

    def aggregate(self, verdicts: Sequence[SkepticVerdict]) -> PanelResult:
        """Weigh the verdicts (pruning overconfident ones in trust mode) into a decision."""
        kept: list[SkepticVerdict] = []
        pruned: list[PrunedAgent] = []
        for v in verdicts:
            if self._weighting == "trust" and self._is_overconfident(v):
                pruned.append(
                    PrunedAgent(v.agent_type, "overconfident: high confidence, no evidence")
                )
            else:
                kept.append(v)

        if not kept:
            return PanelResult(PanelDecision.STANDS, None, 0.0, tuple(verdicts), tuple(pruned))

        stance_weight = dict.fromkeys(SkepticStance, 0.0)
        total = 0.0
        best_revision: Payload | None = None
        best_revision_weight = -1.0
        for v in kept:
            w = self._weight(v)
            stance_weight[v.stance] += w
            total += w
            if (
                v.stance is SkepticStance.REVISE
                and v.proposed_payload is not None
                and w > best_revision_weight
            ):
                best_revision = v.proposed_payload
                best_revision_weight = w

        winning = max(stance_weight, key=lambda s: stance_weight[s])
        decision = _STANCE_TO_DECISION[winning]
        corrected = best_revision if decision is PanelDecision.REVISED else None
        confidence = stance_weight[winning] / total if total > 0.0 else 0.0
        return PanelResult(decision, corrected, confidence, tuple(verdicts), tuple(pruned))
