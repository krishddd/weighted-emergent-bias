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

from .types import BiasScore


def node_magnitude(score: BiasScore, *, alpha: float = 0.05) -> float:
    """The n-stable, significance-gated bias magnitude ``b_i`` for a node.

    Returns ``max(0, raw_divergence - null_mean)`` when ``score`` is significant at ``alpha``, else
    ``0.0``. This is the quantity the network accumulator (WP5) integrates -- never ``effect_size``,
    which is sample-size-dependent (see module docstring).
    """
    if not score.is_significant(alpha):
        return 0.0
    return max(0.0, score.raw_divergence - score.null_mean)
