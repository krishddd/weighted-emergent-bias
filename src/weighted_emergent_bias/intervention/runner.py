"""The intervention runner -- route a halted run to a strategy and return a repaired payload (M4).

This is what closes the loop with M3. Given the halt context, it calls :func:`route_intervention`
(broad contamination → skeptics, concentrated → MADERA), runs the chosen strategy, and returns an
:class:`InterventionResult`. Because M3's recovery **re-measures** ``B_net``, the runner only has to
return a corrected payload -- a genuine repair drives the control machine back to Normal on the next
step, with no tighter coupling.

The runner is pure orchestration; all model work is inside the injected panel / MADERA pipeline.
"""

from __future__ import annotations

from ..types import Payload
from .madera import MaderaEditor
from .router import InterventionContext, route_intervention
from .skeptics import SkepticPanel
from .trust import TrustGraph
from .types import InterventionResult, PanelDecision


class InterventionRunner:
    """Routes a halted run to the skeptic panel or MADERA and returns the repaired payload."""

    def __init__(
        self,
        *,
        panel: SkepticPanel | None = None,
        trust: TrustGraph | None = None,
        madera: MaderaEditor | None = None,
        breadth_threshold: float = 0.5,
    ) -> None:
        self._panel = panel
        self._trust = trust if trust is not None else TrustGraph()
        self._madera = madera
        self._breadth_threshold = breadth_threshold

    async def _run_skeptics(self, payload: Payload) -> InterventionResult:
        if self._panel is None:
            raise ValueError("routing chose 'skeptics' but no skeptic panel was supplied")
        verdicts = await self._panel.review(payload)
        result = self._trust.aggregate(verdicts)
        pruned = f"{len(result.pruned)} pruned" if result.pruned else "none pruned"
        trail = (f"panel: {result.decision.value} ({pruned})",)
        if result.decision is PanelDecision.REVISED and result.corrected_payload is not None:
            return InterventionResult("skeptics", result.corrected_payload, True, trail)
        if result.decision is PanelDecision.REJECTED:
            return InterventionResult("skeptics", None, True, trail)
        return InterventionResult("skeptics", payload, False, (*trail, "output stands"))

    async def run(self, context: InterventionContext, payload: Payload) -> InterventionResult:
        """Route ``context`` and run the chosen mitigation on ``payload``."""
        strategy = route_intervention(context, breadth_threshold=self._breadth_threshold)
        if strategy == "skeptics":
            return await self._run_skeptics(payload)
        if strategy == "madera":
            if self._madera is None:
                raise ValueError("routing chose 'madera' but no MADERA pipeline was supplied")
            return await self._madera.repair(payload)
        return InterventionResult("none", payload, False, ("no intervention needed",))
