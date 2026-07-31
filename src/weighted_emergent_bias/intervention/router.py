"""Route a halted run to a mitigation strategy (the M3 seam).

The bifurcated intervention (DESIGN §6): a **conformity spiral** — many agents homogenizing around
the same biased register — routes to parallel **skeptic** agents that break the false consensus; an
entrenched **parametric** bias — concentrated in one model — routes to **MADERA** surgical repair.
So the routing signal is *breadth*: broad contamination → skeptics, concentrated → MADERA.

This is a documented heuristic, not a solved diagnosis. It is the seam M4 plugs a real classifier
into; the default is deliberately simple and its threshold is exposed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InterventionStrategy = Literal["skeptics", "madera", "none"]


@dataclass(frozen=True)
class InterventionContext:
    """How bias is distributed across the graph at the moment of a halt.

    ``biased_node_count`` / ``total_node_count`` give the breadth of contamination;
    ``max_node_magnitude`` is the worst single node. ``halted`` gates routing — an un-halted run
    needs no intervention.
    """

    biased_node_count: int
    total_node_count: int
    max_node_magnitude: float
    halted: bool = True

    def __post_init__(self) -> None:
        if self.total_node_count < 0 or self.biased_node_count < 0:
            raise ValueError("node counts must be non-negative")
        if self.biased_node_count > self.total_node_count:
            raise ValueError("biased_node_count cannot exceed total_node_count")

    @property
    def breadth(self) -> float:
        """Fraction of the graph that is biased, in [0, 1] (0 if the graph is empty)."""
        if self.total_node_count == 0:
            return 0.0
        return self.biased_node_count / self.total_node_count


def route_intervention(
    context: InterventionContext,
    *,
    breadth_threshold: float = 0.5,
) -> InterventionStrategy:
    """Pick a mitigation strategy for a halted run.

    Returns ``"none"`` if the run is not halted or no node is biased; ``"skeptics"`` when
    contamination is *broad* (breadth ≥ ``breadth_threshold`` — a conformity spiral across many
    agents); ``"madera"`` when it is *concentrated* (a single/few persistently-biased nodes, the
    signature of entrenched parametric bias).
    """
    if not 0.0 < breadth_threshold <= 1.0:
        raise ValueError(f"breadth_threshold must be in (0, 1], got {breadth_threshold}")
    if not context.halted or context.biased_node_count == 0:
        return "none"
    return "skeptics" if context.breadth >= breadth_threshold else "madera"
