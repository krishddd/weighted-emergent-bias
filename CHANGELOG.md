# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/) once it reaches a release.

## [Unreleased]

Building **M1 — Detection core (v0.1)**. Nothing released yet; `version = 0.0.0`.

### Added
- **WP4 — Noise floor** (`scoring/noise.py`): `compute_local_bias` turns resampled baseline /
  counterfactual representations into a calibrated `BiasScore` via a permutation test — the
  observed statistic and the null are the same quantity (divergence between group means), giving
  proper false-positive control. Reports a standardized effect size, a one-sided permutation
  p-value, a bootstrap CI, and a deterministic-node path. **Calibration gate met:** false-positive
  rate within tolerance of α = 0.05 over 200 unbiased simulated nodes. `compute_local_bias` is now
  the top-level public entry point.
- **WP3 — Divergence estimators** (`scoring/divergence.py`): true Jensen-Shannon over a shared
  candidate support (mixture-based, bounded `[0, 1]`, finite on disjoint supports where KL
  would be ∞), `softmax`, `js_divergence_from_logits`, `cosine_distance` for the generative
  path, a `raw_divergence` mode→method dispatcher, and `assert_same_estimator` to refuse
  combining scores from different estimators. The review's symmetrized-KL "JS" formula is
  explicitly not adopted (see module docstring and the disjoint-support test).
- **WP2 — Perturbation engine** (`scoring/perturbation.py`): `perturb(payload, axes)` with
  `AxisSpec` / `Substitution`. Traverses nested payloads (dict/list/tuple), editing only
  string leaves with whole-word, case-preserving substitution. Explicit and proxy
  substitutions emit separate `Perturbation`s. No default axis list ships.
  [docs/example-axes.md](docs/example-axes.md) documents illustrative axis sets.
- **WP1 — Core types & protocols**: `BiasScore` (effect size, CI, permutation p-value,
  per-axis, estimator + null stats), `TaskMode` / `DivergenceMethod` / `PerturbationKind`
  enums, `Perturbation`, `Payload`; async `LLMClient` / `EmbeddingClient` protocols;
  `FakeLLMClient` with a dial-able per-axis bias knob and `FakeEmbeddingClient` (the project's
  ground-truth test asset).
- **WP0 — Foundations**: src-layout package, `py.typed`, `mypy --strict`, ruff, pytest with a
  seeded-RNG fixture, and a GitHub Actions matrix across Python 3.10/3.11/3.12.
- **Planning docs**: `DESIGN.md` (incl. scope §0 and review-driven revisions §8),
  `ROADMAP.md`, `plans/PHASE-1.md`, and the 2026-07 external-review response.

### Changed
- `Perturbation.original` / `.perturbed` generalized from `str` to the `Payload` alias to carry
  structured payloads (WP2).
- `FloatArray` promoted from `clients.py` to `types.py` as its canonical home (WP3); `clients`
  re-exports it for compatibility.

### Fixed
- `mypy --strict` failures surfaced only on specific interpreters: a 3.12-only numpy stub
  syntax error (dropped the `python_version` pin so each matrix job validates its own version)
  and a 3.10-only ndarray shape-typing error (explicit `FloatArray` annotations).

### Notes
- Scope made explicit: WEB detects **counterfactual demographic bias**, not factual error or
  consensus deviation. Benchmark numbers from prior work (MALIBU, BBQ-Hard, the 57.5% SDC
  figure) are cited as prior work only, never as this library's validated results.
