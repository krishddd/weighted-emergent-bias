"""Core value types for the detection layer.

These are deliberately small, frozen, and self-validating. Everything downstream
(propagation, control, evidence) consumes a ``BiasScore``, so its invariants are
enforced at construction rather than trusted.

Design notes:

- ``BiasScore`` is the result of *one* baseline-vs-counterfactual comparison,
  standardized against the node's own sampling-noise null distribution. A node with
  several perturbation axes yields several scores; how they are aggregated is the
  *consumer's* choice (see ``docs/plans/PHASE-1.md`` R3), so no collapsing happens here.
- ``ProbeResult`` is intentionally *not* defined in this module yet. Its shape is
  dictated by the probe's actual data flow, which lands in WP5; defining it now would
  be guessing. It joins this file when ``scoring/probe.py`` exists.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

# Semantic aliases — these are just ``str`` at runtime, but they document intent.
NodeId = str
Axis = str

# A node's input/output payload: a string, or a nested structure of them. Real agent state
# is structured (dicts of fields, lists of messages), so perturbation traverses it and edits
# only the string leaves. ``str`` is listed first because it is both the base case and,
# technically, a ``Sequence`` — traversal must check it first.
Payload = str | Mapping[str, Any] | Sequence[Any]


class TaskMode(str, Enum):
    """How a node's output is compared against its counterfactual.

    The two modes are not on the same raw scale and must never be averaged together;
    the mode travels with every :class:`BiasScore` so a consumer always knows which
    estimator produced it. Standardization against the per-node null (see
    :class:`BiasScore.effect_size`) is what makes the two loosely comparable.
    """

    CHOICE = "choice"
    """A fixed candidate set exists (classifier, router, scorer, ranker, judge).
    Divergence is exact Jensen-Shannon over a shared support — the high-confidence path."""

    GENERATIVE = "generative"
    """Free-form text with no candidate set. Divergence is an embedding-space distance —
    a semantic proxy, not a calibrated divergence."""


class DivergenceMethod(str, Enum):
    """The concrete estimator that produced a score. Recorded for audit and to refuse
    silently mixing incomparable scales."""

    JENSEN_SHANNON = "jensen_shannon"
    """JS over top-k logprobs on a shared candidate support. Bounded [0, 1] (log base 2)."""

    EMBEDDING = "embedding"
    """Distance between output texts in embedding space."""

    DETERMINISTIC = "deterministic"
    """Node sampled at temperature 0: the null collapses, so any nonzero divergence is
    real by construction and no standardization is applied."""


class PerturbationKind(str, Enum):
    """Whether a counterfactual edits an attribute directly or via a correlated proxy."""

    EXPLICIT = "explicit"
    """A named protected attribute was substituted (e.g. a gendered pronoun)."""

    PROXY = "proxy"
    """A feature correlated with a protected attribute was substituted (zip code,
    institution, sociolect) — the demographic marker is never stated outright."""


@dataclass(frozen=True, slots=True)
class Perturbation:
    """One counterfactual: the original payload and its demographically perturbed twin.

    The estimator computes divergence between a node's output on ``original`` and its
    output on ``perturbed``; everything except the targeted ``axis`` (and its proxies)
    is held fixed. ``kind`` records whether this counterfactual edits an explicit attribute
    or only a correlated proxy — the two are separate probes and are never merged, because
    proxy bias (a model reacting to a zip code or sociolect) is a distinct, more insidious
    signal than reacting to a stated attribute.
    """

    axis: Axis
    original: Payload
    perturbed: Payload
    kind: PerturbationKind = PerturbationKind.EXPLICIT

    def __post_init__(self) -> None:
        if not self.axis:
            raise ValueError("Perturbation.axis must be a non-empty string")


@dataclass(frozen=True, slots=True)
class BiasScore:
    """A calibrated, single-comparison bias magnitude with honest uncertainty.

    ``effect_size`` is the number M2 (propagation) consumes: the counterfactual
    divergence expressed in units of the node's own sampling-noise standard deviation,

        effect_size = (raw_divergence - null_mean) / null_std,

    so a value near 0 means "indistinguishable from resampling the same input" and a
    value of, say, 3 means "three noise-sigmas beyond what non-determinism explains."
    The confidence interval and permutation ``p_value`` are what stop a wide, uncertain
    score from being treated as a confident one downstream.
    """

    effect_size: float
    ci_low: float
    ci_high: float
    p_value: float
    method: DivergenceMethod
    task_mode: TaskMode
    n_samples: int
    raw_divergence: float
    null_mean: float
    null_std: float
    axis: Axis | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")
        if self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must not exceed ci_high ({self.ci_high})")
        if self.n_samples < 0:
            raise ValueError(f"n_samples must be non-negative, got {self.n_samples}")
        if self.null_std < 0.0:
            raise ValueError(f"null_std must be non-negative, got {self.null_std}")

    @property
    def ci(self) -> tuple[float, float]:
        """The confidence interval as a ``(low, high)`` pair."""
        return (self.ci_low, self.ci_high)

    def is_significant(self, alpha: float = 0.05) -> bool:
        """True if the counterfactual divergence beats the noise floor at level ``alpha``."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        return self.p_value <= alpha

    @property
    def ci_excludes_zero(self) -> bool:
        """True if the whole confidence interval sits above zero — a stricter,
        magnitude-based confidence signal than the p-value alone."""
        return self.ci_low > 0.0
