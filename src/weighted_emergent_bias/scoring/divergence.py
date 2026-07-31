"""Divergence estimators — the raw distance between a node's two outputs.

Two task modes, two primitives:

- ``CHOICE`` → :func:`jensen_shannon` over a shared candidate support. **True** Jensen-Shannon,
  using the mixture ``M = ½(P + Q)``:

      JSD(P‖Q) = ½·D_KL(P‖M) + ½·D_KL(Q‖M)

  This is bounded in ``[0, 1]`` (log base 2) and stays finite even when the two supports are
  disjoint, because ``M`` has mass wherever *either* input does. Note that the symmetrized-KL
  form ``½·KL(P‖Q) + ½·KL(Q‖P)`` — which some sources mislabel "Jensen-Shannon" — is a
  different, *unbounded* quantity (Jeffreys divergence) that blows up to ∞ on the exact
  zero-probability case JSD handles cleanly. We use true JSD; see the disjoint-support test.

- ``GENERATIVE`` → :func:`cosine_distance` between output embeddings, a semantic proxy.

Scores from the two modes are on different scales and are never mixed. :func:`assert_same_estimator`
enforces that at the point where any consumer would try to combine :class:`~..types.BiasScore`s.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from ..types import BiasScore, DivergenceMethod, FloatArray, TaskMode

_SUM_TOL = 1e-9


def _as_distribution(x: FloatArray | list[float]) -> FloatArray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"a distribution must be a non-empty 1-D array, got shape {a.shape}")
    if bool(np.any(a < 0.0)):
        raise ValueError("distribution has negative entries; probabilities must be >= 0")
    total = float(a.sum())
    if not math.isclose(total, 1.0, abs_tol=_SUM_TOL):
        raise ValueError(f"distribution must sum to 1, got {total}")
    return a


def _kl_bits(p: FloatArray, m: FloatArray) -> float:
    """KL(p ‖ m) in bits, with the 0·log0 = 0 convention. ``m`` is a mixture, so mathematically
    it has mass wherever ``p`` does; the ``maximum`` clamp only guards float underflow to 0.0 for
    denormal inputs (where ``log2(0)`` would be -inf), and is a no-op otherwise."""
    mask = p > 0.0
    p_masked = p[mask]
    m_masked = np.maximum(m[mask], np.finfo(np.float64).tiny)
    return float(np.sum(p_masked * (np.log2(p_masked) - np.log2(m_masked))))


def softmax(logits: FloatArray | list[float]) -> FloatArray:
    """Convert raw log-scores to a probability distribution, numerically stably."""
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"logits must be a non-empty 1-D array, got shape {a.shape}")
    shifted = a - np.max(a)
    exp: FloatArray = np.exp(shifted)
    result: FloatArray = exp / exp.sum()
    return result


def jensen_shannon(p: FloatArray | list[float], q: FloatArray | list[float]) -> float:
    """True Jensen-Shannon divergence between two distributions on a shared support.

    Bounded in ``[0, 1]`` (log base 2): ``0`` iff the distributions are equal, ``1`` iff they
    have disjoint supports. Symmetric in its arguments.
    """
    pd = _as_distribution(p)
    qd = _as_distribution(q)
    if pd.shape != qd.shape:
        raise ValueError(
            f"distributions must share a support (same shape); got {pd.shape} vs {qd.shape}"
        )
    m = 0.5 * (pd + qd)
    jsd = 0.5 * _kl_bits(pd, m) + 0.5 * _kl_bits(qd, m)
    # Clamp only to absorb floating-point overshoot at the [0, 1] boundaries.
    return float(min(1.0, max(0.0, jsd)))


def js_divergence_from_logits(a: FloatArray | list[float], b: FloatArray | list[float]) -> float:
    """Jensen-Shannon divergence between two candidate-sets given their raw log-scores."""
    return jensen_shannon(softmax(a), softmax(b))


def cosine_distance(u: FloatArray | list[float], v: FloatArray | list[float]) -> float:
    """Cosine distance ``1 - cos(u, v)`` between two embedding vectors. Range ``[0, 2]``:
    ``0`` for identical direction, ``1`` for orthogonal, ``2`` for opposite."""
    uu = np.asarray(u, dtype=np.float64)
    vv = np.asarray(v, dtype=np.float64)
    if uu.shape != vv.shape or uu.ndim != 1 or uu.size == 0:
        raise ValueError(
            f"embeddings must be non-empty 1-D arrays of equal shape; got {uu.shape}, {vv.shape}"
        )
    nu = float(np.linalg.norm(uu))
    nv = float(np.linalg.norm(vv))
    if nu == 0.0 or nv == 0.0:
        raise ValueError("cosine distance is undefined for a zero-norm vector")
    cos = float(np.dot(uu, vv) / (nu * nv))
    cos = min(1.0, max(-1.0, cos))
    return 1.0 - cos


def raw_divergence(
    a: FloatArray | list[float],
    b: FloatArray | list[float],
    *,
    task_mode: TaskMode,
) -> tuple[float, DivergenceMethod]:
    """Dispatch to the estimator for ``task_mode``, returning ``(value, method)``.

    ``CHOICE`` treats ``a``/``b`` as candidate log-scores (JSD); ``GENERATIVE`` treats them as
    embedding vectors (cosine distance). Returning the method alongside the value keeps the two
    inseparable, so a score can never be recorded without the estimator that produced it.
    """
    if task_mode is TaskMode.CHOICE:
        return js_divergence_from_logits(a, b), DivergenceMethod.JENSEN_SHANNON
    if task_mode is TaskMode.GENERATIVE:
        return cosine_distance(a, b), DivergenceMethod.EMBEDDING
    raise ValueError(f"unsupported task_mode: {task_mode!r}")


def assert_same_estimator(scores: Iterable[BiasScore]) -> None:
    """Raise if a collection of scores mixes divergence methods or task modes.

    Effect sizes from different estimators are not on a common raw scale; combining them (a mean,
    a max, an accumulator step) is a silent scale error. Consumers call this before aggregating.
    """
    items = list(scores)
    if not items:
        return
    methods = {s.method for s in items}
    modes = {s.task_mode for s in items}
    if len(methods) > 1 or len(modes) > 1:
        raise ValueError(
            "cannot combine BiasScores from different estimators: "
            f"methods={sorted(m.value for m in methods)}, "
            f"task_modes={sorted(t.value for t in modes)}"
        )
