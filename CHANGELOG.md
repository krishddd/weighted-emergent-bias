# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/) once it reaches a release.

## [Unreleased]

Building **M1 — Detection core (v0.1)**. Nothing released yet; `version = 0.0.0`.

### Added
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

### Fixed
- `mypy --strict` failures surfaced only on specific interpreters: a 3.12-only numpy stub
  syntax error (dropped the `python_version` pin so each matrix job validates its own version)
  and a 3.10-only ndarray shape-typing error (explicit `FloatArray` annotations).

### Notes
- Scope made explicit: WEB detects **counterfactual demographic bias**, not factual error or
  consensus deviation. Benchmark numbers from prior work (MALIBU, BBQ-Hard, the 57.5% SDC
  figure) are cited as prior work only, never as this library's validated results.
