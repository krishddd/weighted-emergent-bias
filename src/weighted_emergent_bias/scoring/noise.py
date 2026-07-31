"""The noise floor — turning raw divergence into a calibrated bias score.

This is the load-bearing module. An LLM sampled twice on the *identical* input diverges from
itself, so a raw baseline-vs-counterfactual divergence is always > 0 even for an unbiased node.
Thresholding that raw number is thresholding sampling temperature. :func:`compute_local_bias`
instead measures the counterfactual divergence *in excess of* the node's own sampling noise, and
reports how confident that measurement is.

**Method (permutation test).** Given `n` resampled baseline representations and `m` resampled
counterfactual ones, the statistic is the divergence between the two groups' mean representations,

    T(A, B) = d(mean(A), mean(B))

where `d` is Jensen-Shannon (``CHOICE``) or cosine distance (``GENERATIVE``). The null
distribution comes from repeatedly shuffling the baseline/counterfactual labels and recomputing
`T` — i.e. "what values of `T` arise purely from resampling noise, if the counterfactual label
means nothing?" Because the observed statistic and the null are *the same statistic*, there is no
scale mismatch, and for a genuinely unbiased node the observed `T` is a typical draw from the null,
so the p-value is ~uniform and the false-positive rate tracks alpha. That property is the WP4
acceptance gate (see the calibration test).

- **effect_size** = `(T_obs - mean(null)) / max(std(null), std_floor)` -- the magnitude M2 consumes,
  in units of the node's sampling-noise sigma.
- **p_value** = one-sided permutation p-value (fraction of null ≥ observed).
- **ci_low/ci_high** = bootstrap confidence interval on the effect size.
- **Deterministic path**: if the baseline resamples are identical (no sampling noise, e.g. a
  temperature-0 node), the score is tagged ``DivergenceMethod.DETERMINISTIC`` — any nonzero
  divergence is real by construction, and the standardization falls back to ``std_floor``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..types import BiasScore, DivergenceMethod, FloatArray, TaskMode
from .divergence import cosine_distance, jensen_shannon


def _stack(samples: Sequence[FloatArray], label: str) -> FloatArray:
    if len(samples) < 2:
        raise ValueError(
            f"need >= 2 {label} samples to estimate the noise floor, got {len(samples)}"
        )
    arr: FloatArray = np.asarray(np.vstack(samples), dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{label} samples must each be 1-D of equal length")
    return arr


def _stat(group_a: FloatArray, group_b: FloatArray, task_mode: TaskMode) -> float:
    """Divergence between the two groups' mean representations."""
    mean_a: FloatArray = group_a.mean(axis=0)
    mean_b: FloatArray = group_b.mean(axis=0)
    if task_mode is TaskMode.CHOICE:
        # Renormalize to absorb float drift; the mean of distributions is a distribution.
        mean_a = mean_a / mean_a.sum()
        mean_b = mean_b / mean_b.sum()
        return jensen_shannon(mean_a, mean_b)
    if task_mode is TaskMode.GENERATIVE:
        return cosine_distance(mean_a, mean_b)
    raise ValueError(f"unsupported task_mode: {task_mode!r}")


def compute_local_bias(
    baseline: Sequence[FloatArray],
    counterfactual: Sequence[FloatArray],
    *,
    task_mode: TaskMode,
    rng: np.random.Generator,
    axis: str | None = None,
    n_permutations: int = 1000,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    std_floor: float = 1e-9,
) -> BiasScore:
    """Compute a calibrated :class:`~..types.BiasScore` from resampled representations.

    ``baseline`` / ``counterfactual`` are sequences of a node's output *representations* under
    the standard and perturbed inputs — probability distributions (summing to 1) for ``CHOICE``,
    embedding vectors for ``GENERATIVE``. Each side must have >= 2 samples so the sampling-noise
    floor can be estimated. ``rng`` drives the permutation and bootstrap resampling (required, so
    results are reproducible).
    """
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    base = _stack(baseline, "baseline")
    cf = _stack(counterfactual, "counterfactual")
    if base.shape[1] != cf.shape[1]:
        raise ValueError(
            f"baseline and counterfactual representations must have equal width; "
            f"got {base.shape[1]} and {cf.shape[1]}"
        )

    n, m = base.shape[0], cf.shape[0]
    pooled: FloatArray = np.vstack([base, cf])
    t_obs = _stat(base, cf, task_mode)

    # Permutation null: shuffle group labels, recompute the statistic.
    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        order = rng.permutation(n + m)
        perm_a = pooled[order[:n]]
        perm_b = pooled[order[n:]]
        null[i] = _stat(perm_a, perm_b, task_mode)

    null_mean = float(null.mean())
    null_std = float(null.std(ddof=1))
    eff_std = max(null_std, std_floor)
    effect_size = (t_obs - null_mean) / eff_std

    # One-sided p-value (bias means a larger-than-noise divergence); +1 keeps it non-zero.
    p_value = float((1 + np.sum(null >= t_obs - 1e-12)) / (1 + n_permutations))

    # Bootstrap CI on the effect size: resample within each group with replacement.
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        a = base[rng.integers(0, n, n)]
        b = cf[rng.integers(0, m, m)]
        boot[i] = (_stat(a, b, task_mode) - null_mean) / eff_std
    lo_pct = 100.0 * (1.0 - ci) / 2.0
    ci_low = float(np.percentile(boot, lo_pct))
    ci_high = float(np.percentile(boot, 100.0 - lo_pct))

    # A node whose baseline resamples are identical has no sampling noise: any divergence is real.
    is_deterministic = bool(np.allclose(base, base[0]))
    if is_deterministic:
        method = DivergenceMethod.DETERMINISTIC
    elif task_mode is TaskMode.CHOICE:
        method = DivergenceMethod.JENSEN_SHANNON
    else:
        method = DivergenceMethod.EMBEDDING

    return BiasScore(
        effect_size=effect_size,
        ci_low=min(ci_low, ci_high),
        ci_high=max(ci_low, ci_high),
        p_value=p_value,
        method=method,
        task_mode=task_mode,
        n_samples=n,
        raw_divergence=t_obs,
        null_mean=null_mean,
        null_std=null_std,
        axis=axis,
    )
