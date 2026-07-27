"""Detection-layer scoring: perturbation, divergence, noise floor, probes (M1)."""

from __future__ import annotations

from .divergence import (
    assert_same_estimator,
    cosine_distance,
    jensen_shannon,
    js_divergence_from_logits,
    raw_divergence,
    softmax,
)
from .perturbation import AxisSpec, Substitution, perturb

__all__ = [
    "AxisSpec",
    "Substitution",
    "assert_same_estimator",
    "cosine_distance",
    "jensen_shannon",
    "js_divergence_from_logits",
    "perturb",
    "raw_divergence",
    "softmax",
]
