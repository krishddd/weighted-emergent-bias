"""Turning per-node BiasScores into the magnitude the network accumulator integrates.

This module is where the central finding of the Phase 1 study becomes an architectural constraint.
``BiasScore.effect_size`` is a *detection* statistic that inflates with sample count (the study
measured the same true bias at 1.9 sigma with n=3 and 22.9 sigma with n=30, because ``null_std``
-> 0 as n grows). Weighting on it would make the network signal depend on each node's sample budget,
not its bias. So M2 does **not** accumulate effect size.

The magnitude it accumulates is the **significance-gated excess divergence**:

    b_i = max(0, raw_divergence - null_mean)   if the score is significant, else 0

As n grows, ``raw_divergence`` converges to the true divergence and ``null_mean`` -> 0, so ``b_i``
converges to a fixed per-node quantity (bounded [0, 1] for CHOICE / JSD). The p-value carries the
"is this real?" judgement the whole noise-floor apparatus exists to produce; the excess divergence
carries an n-stable "how big?". ``effect_size`` is used only through the significance gate, never as
the magnitude.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .types import BiasScore, NodeId


def node_magnitude(score: BiasScore, *, alpha: float = 0.05) -> float:
    """The n-stable, significance-gated bias magnitude ``b_i`` for a node.

    Returns ``max(0, raw_divergence - null_mean)`` when ``score`` is significant at ``alpha``, else
    ``0.0``. This is the quantity the network accumulator (WP5) integrates -- never ``effect_size``,
    which is sample-size-dependent (see module docstring).
    """
    if not score.is_significant(alpha):
        return 0.0
    return max(0.0, score.raw_divergence - score.null_mean)


_REDUCTIONS = ("weighted_mean", "max")


@dataclass(frozen=True)
class NetworkState:
    """The network bias signal at one superstep, at two timescales.

    ``fast`` reacts quickly (spike detection — conformity that converges in a few steps); ``slow``
    integrates over many steps (cultural drift). Both are bias-corrected, so a constant input reads
    at its true level from step 1 rather than ramping up from 0. ``step`` is the superstep count.
    """

    fast: float
    slow: float
    step: int


def _reduce(
    magnitudes: Mapping[NodeId, float],
    weights: Mapping[NodeId, float],
    reduction: str,
) -> float:
    """Collapse a superstep's firing nodes into one value. Order-independent by construction."""
    nodes = list(magnitudes)
    if not nodes:
        return 0.0
    missing = [n for n in nodes if n not in weights]
    if missing:
        raise ValueError(f"no dependency weight for firing node(s): {missing}")
    if reduction == "max":
        return max(magnitudes[n] for n in nodes)
    # weighted_mean: a central (high-w) biased node dominates the step.
    wsum = sum(weights[n] for n in nodes)
    if wsum <= 0.0:
        return 0.0
    return sum(weights[n] * magnitudes[n] for n in nodes) / wsum


class NetworkAccumulator:
    """Bias-corrected, multi-scale EWMA of weighted node bias across graph execution.

    Call :meth:`update` once per superstep with the firing nodes' magnitudes (from
    :func:`node_magnitude`) and their dependency weights (from ``dependency_weights``). The firing
    set is reduced to a single value (``weighted_mean`` by weight, or ``max``), then folded into a
    fast and a slow EWMA. Both carry the standard warm-up bias correction ``/ (1 - (1-alpha)^t)``,
    without which ``B_net`` would read low for the first ~1/alpha steps -- exactly the early graph
    stages where interception matters most.
    """

    def __init__(
        self,
        *,
        fast_alpha: float = 0.7,
        slow_alpha: float = 0.1,
        reduction: str = "weighted_mean",
    ) -> None:
        for name, a in (("fast_alpha", fast_alpha), ("slow_alpha", slow_alpha)):
            if not 0.0 < a <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {a}")
        if reduction not in _REDUCTIONS:
            raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")
        self._fast_alpha = fast_alpha
        self._slow_alpha = slow_alpha
        self._reduction = reduction
        self._fast_raw = 0.0
        self._slow_raw = 0.0
        self._step = 0

    def _corrected(self, raw: float, alpha: float) -> float:
        if self._step == 0:
            return 0.0
        return raw / (1.0 - (1.0 - alpha) ** self._step)

    @property
    def state(self) -> NetworkState:
        """Current bias-corrected state, without advancing a step."""
        return NetworkState(
            fast=self._corrected(self._fast_raw, self._fast_alpha),
            slow=self._corrected(self._slow_raw, self._slow_alpha),
            step=self._step,
        )

    def update(
        self,
        magnitudes: Mapping[NodeId, float],
        weights: Mapping[NodeId, float],
    ) -> NetworkState:
        """Advance one superstep with the firing nodes' magnitudes and weights; return new state.

        An empty ``magnitudes`` is a valid quiet step (contributes 0), letting ``B_net`` decay.
        """
        step_value = _reduce(magnitudes, weights, self._reduction)
        self._step += 1
        self._fast_raw = self._fast_alpha * step_value + (1.0 - self._fast_alpha) * self._fast_raw
        self._slow_raw = self._slow_alpha * step_value + (1.0 - self._slow_alpha) * self._slow_raw
        return self.state
