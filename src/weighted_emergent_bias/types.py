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

import numpy as np
import numpy.typing as npt

# The canonical numeric array type. Annotate numpy locals with this explicitly so mypy's
# shape inference cannot narrow it differently across numpy stub versions (see CONTRIBUTING).
FloatArray = npt.NDArray[np.float64]

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


@dataclass(frozen=True, slots=True)
class AxisScore:
    """A bias score for one (axis, kind) counterfactual, keeping the perturbation kind that
    ``BiasScore`` alone does not carry (explicit attribute vs. proxy)."""

    axis: Axis
    kind: PerturbationKind
    score: BiasScore


@dataclass(frozen=True, slots=True)
class AxisFailure:
    """A probe that could not be scored for one (axis, kind) — recorded, never silently dropped."""

    axis: Axis
    kind: PerturbationKind
    error: str


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """The outcome of probing one node across its perturbation axes.

    Scores are kept per (axis, kind) rather than collapsed — how to aggregate them (max across
    axes is the safety-relevant default; see ``docs/plans/PHASE-1.md`` R3) is the consumer's
    choice. Any axis that errored lands in ``failures`` with its reason, so a partial scan is
    never mistaken for a complete one. ``n_client_calls`` is the number of LLM calls issued,
    for cost accounting.
    """

    task_mode: TaskMode
    n_samples: int
    n_client_calls: int
    scores: tuple[AxisScore, ...]
    failures: tuple[AxisFailure, ...]

    @property
    def ok(self) -> bool:
        """True if every axis was scored (no failures)."""
        return not self.failures

    def worst(self) -> AxisScore | None:
        """The highest-effect-size axis score — the max-aggregation default. ``None`` if empty."""
        if not self.scores:
            return None
        return max(self.scores, key=lambda a: a.score.effect_size)


class BreakerState(str, Enum):
    """The control plane's state (M3). Transitions live in the control state machine."""

    NORMAL = "normal"
    """Bias below thresholds; the run proceeds untouched."""

    WARNING = "warning"
    """Slow-scale drift crossed its threshold; observe, do not yet halt."""

    INTERVENTION = "intervention"
    """Fast-scale bias crossed the enter threshold; the run is halted and rerouted for repair."""

    RECOVERY = "recovery"
    """A mitigation ran; awaiting cool-down and a passing re-check before returning to normal."""

    ESCALATED = "escalated"
    """Repeated repair attempts failed; terminal, handed to human review."""


class BreakerAction(str, Enum):
    """The concrete, deterministic action a :class:`BreakerDecision` directs."""

    PROCEED = "proceed"
    """Promote the node's output and continue."""

    HALT = "halt"
    """Freeze the compromised payload; stop before the downstream node consumes it."""

    REROUTE = "reroute"
    """Halt and hand control to the mitigation subgraph."""

    ESCALATE = "escalate"
    """Give up automated repair and route to human review."""


@dataclass(frozen=True, slots=True)
class BreakerDecision:
    """The outcome of one breaker check: a deterministic action plus advisory severity.

    ``action`` is discrete on purpose — a control plane must be predictable. ``mixing_ratio`` is the
    smooth ``sigmoid((B_net - tau_enter)/kappa)`` severity in ``[0, 1]``; M3 computes it, M4 may use
    it to blend consensus vs. skeptic output. ``frozen_payload`` holds the compromised state when
    the run is halted, so it is recoverable and auditable.
    """

    state: BreakerState
    action: BreakerAction
    mixing_ratio: float
    b_net: float
    reason: str
    frozen_payload: Payload | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.mixing_ratio <= 1.0:
            raise ValueError(f"mixing_ratio must be in [0, 1], got {self.mixing_ratio}")

    @property
    def halted(self) -> bool:
        """True if this decision stops the run (halt, reroute, or escalate)."""
        return self.action is not BreakerAction.PROCEED
