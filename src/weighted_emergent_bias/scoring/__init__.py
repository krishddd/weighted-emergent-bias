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
from .noise import compute_local_bias
from .perturbation import AxisSpec, Substitution, perturb
from .probe import LOOCProbe, ProbeError

__all__ = [
    "AxisSpec",
    "LOOCProbe",
    "ProbeError",
    "Substitution",
    "assert_same_estimator",
    "compute_local_bias",
    "cosine_distance",
    "jensen_shannon",
    "js_divergence_from_logits",
    "perturb",
    "raw_divergence",
    "softmax",
]
