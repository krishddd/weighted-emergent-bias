# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/) once it reaches a release.

## [Unreleased]

Next: **M5 — Evidence** (append-only causal audit trail, SARIF 2.1.0 export, HTML/JSON reporting).

## [0.4.0] — 2026-07-31

**M4 — Intervention.** Repairs a halted run instead of just stopping it: conformity spirals go to a
parallel skeptic panel under a trust graph; entrenched parametric bias goes to a MADERA-style
diagnose→retrieve→rewrite pipeline. Protocols + reference impls with injected callables — no vendor
SDK. Characterized by the [Phase 4 study](docs/studies/phase4-intervention.md): trust-weighting
recovers the correct dissenting output where uniform majority-voting fails (a conformity spiral), the
minority is never pruned, MADERA converges, and a full halt→repair loop drops `B_net` 0.80→0.072.

### Added
- **M4 WP1 — intervention types**: `SkepticVerdict`, `PanelResult`, `TrustScore` (clamped `S`),
  `InterventionResult`, stances/decisions.
- **M4 WP2 — skeptic panel** (`intervention/skeptics.py`): `SkepticAgent` protocol + empirical /
  devil's-advocate / diversity reference skeptics; `SkepticPanel` runs ≥2 distinct agents concurrently.
- **M4 WP3 — trust graph** (`intervention/trust.py`): uniform (baseline) vs. trust weighting, with
  overconfidence-only pruning and the **minority-suppression guarantee** (dissenters never pruned).
- **M4 WP4 — MADERA** (`intervention/madera.py`): bounded, re-scored diagnose→retrieve→rewrite.
- **M4 WP5 — runner** (`intervention/runner.py`): routes + runs a strategy; closes the M3→M4
  halt→repair→recover loop.
- **M4 WP6 — LangGraph `Send` fan-out** (`integrations/langgraph/skeptic_fanout.py`, optional).
- **M4 WP7 — intervention study** ([report](docs/studies/phase4-intervention.md)).

## [0.3.0] — 2026-07-31

**M3 — Control.** A deterministic control plane over `B_net`: halt, freeze, and reroute on breach,
with hysteresis to prevent thrashing and a recovery state machine. Characterized by the
[Phase 3 control study](docs/studies/phase3-control.md): hysteresis enters Intervention once where a
binary threshold flips 3× on the same near-boundary trajectory; the machine recovers on a clearing
spike and escalates on a persistent one; calibration hits its target false-halt rate. Still no
validated claims about real models.

### Added
- **M3 WP1 — control types**: `BreakerState`, `BreakerAction`, `BreakerDecision` (discrete action +
  advisory sigmoid `mixing_ratio` + recoverable `frozen_payload`).
- **M3 WP2/WP3 — `breaker.py`**: `CircuitBreaker` (two-threshold hysteresis + mixing ratio, anti-
  thrash) and `ControlMachine` (Normal→Warning→Intervention→Recovery, cool-down, `max_retries`→
  Escalated, injectable intervention hook); `freeze` for payload snapshots. Pure, framework-free.
- **M3 WP4 — `route_intervention`** (`intervention/router.py`): breadth-based routing seam
  (broad→skeptics, concentrated→MADERA); a documented heuristic for M4 to refine.
- **M3 WP5 — LangGraph reference adapter** (`integrations/langgraph/`): `BiasState` with a staging
  buffer, `guarded_node`, and a `Command`-based breaker node giving edge-level interception (a biased
  payload is frozen, never promoted to `committed`). Optional; not imported by the core; tests behind
  `importorskip`.
- **M3 WP6 — `calibrate_thresholds`** (`calibration.py`): pick `tau` from control-run percentiles at
  a target false-halt rate — no magic constant.
- **M3 WP7 — control study** (`studies/phase3_control.py`, [report](docs/studies/phase3-control.md)).

## [0.2.0] — 2026-07-31

**M2 — Propagation.** Turns per-node bias scores into a network-level signal `B_net`, weighted by
each node's downstream blast radius and integrated across execution. Characterized by the
[Phase 2 propagation study](docs/studies/phase2-propagation.md): the same bias is ~5.8× more
impactful seeded at a central node than a leaf, and `B_net` tracks downstream reach. Still no
validated claims about real models — demonstrated on the ground-truth fake.

### Added
- **M2 WP1 — `AgentDAG`** (`topology/dag.py`): framework-agnostic directed graph with
  deterministic (insertion-ordered) nodes, adjacency matrix, cycle detection, and topological order
  (Kahn). Builds from an adjacency dict or a `networkx.DiGraph`; depends only on numpy.
- **M2 WP2 — Katz centrality** (`topology/centrality.py`): `dependency_weights` / `katz_weight`
  compute blast radius as row-sums of `(I − αA)⁻¹` (transposed orientation; upstream outranks
  leaves). DAG nilpotency gives free α; cyclic graphs clamp α below `1/ρ` and record it. Out-degree
  fallback.
- **M2 WP3 — n-stable magnitude** (`accumulation.py`): `node_magnitude` = significance-gated excess
  divergence, the quantity M2 accumulates (never `effect_size`, which is sample-size-dependent).
- **M2 WP4 — composite weight**: `dependency_weights` accepts an optional injected error-history
  prior (folded into Katz before normalization); injected-only, no self-updating loop.
- **M2 WP5 — `NetworkAccumulator`** (`accumulation.py`): bias-corrected multi-scale EWMA (fast +
  slow) producing `B_net`, with an order-independent superstep reduction (`weighted_mean` / `max`).
- **M2 WP6 — DoT simulation harness** (`testing/dot_harness.py`): `simulate_dot` runs the full
  M1+M2 stack; `B_net` rises on a seeded graph and stays flat on a control.
- **M2 WP7 — propagation study** (`studies/phase2_propagation.py`, [report](docs/studies/phase2-propagation.md)).

## [0.1.0] — 2026-07-31

**M1 — Detection core.** Single-node counterfactual bias detection, calibrated against its own
sampling-noise floor, with no graph, framework, or vendor dependency. From a prompt + axes +
client you get a per-axis, calibrated `BiasScore` via `LOOCProbe`. Characterized end-to-end by
the [Phase 1 calibration study](docs/studies/phase1-calibration.md). No validated claims about
real models — the estimator is proven correct on the ground-truth fake.

### Added
- **WP7 — Phase 1 calibration study** (`studies/phase1_calibration.py`,
  [docs/studies/phase1-calibration.md](docs/studies/phase1-calibration.md)): reproducible
  five-experiment study — the first numbers this project owns. Detector is calibrated (null
  p-values uniform, FPR tracks α across noise levels); detection floor ≈ β 1–2 at n=8; usable
  sample budget n≥5 (n=3 has 0 power); CHOICE↔GENERATIVE agreement ρ≈0.56; and a **measured cost
  multiplier of `n × (1 + axes)` = 10–32×**, correcting DESIGN's 3–6× estimate. Also surfaced
  that effect size is n-dependent (a detection statistic, not a portable magnitude — an input to
  M2). Optional `study` extra adds matplotlib for figures.
- **WP5 — LOOCProbe orchestration** (`scoring/probe.py`): `LOOCProbe.run` resamples a node's
  output on the standard input and each perturbed input **concurrently** (`asyncio.gather`),
  feeds each pair to `compute_local_bias`, and returns a per-axis `ProbeResult`. One axis failing
  is isolated and recorded in `failures`; a baseline failure is fatal (`ProbeError`). Supports
  `CHOICE` and `GENERATIVE` modes. Resolves the deferred `ProbeResult` / `AxisScore` / `AxisFailure`
  types (shape now dictated by the probe's data flow).
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
- Jensen-Shannon divide-by-zero on denormal inputs: the mixture could underflow to 0.0 even
  where `p > 0`, producing `inf`/`nan`. `_kl_bits` now clamps the mixture to the smallest positive
  float in the log (a no-op for normal inputs). Found by a Hypothesis property test (WP5).
- `mypy --strict` failures surfaced only on specific interpreters: a 3.12-only numpy stub
  syntax error (dropped the `python_version` pin so each matrix job validates its own version)
  and a 3.10-only ndarray shape-typing error (explicit `FloatArray` annotations).

### Notes
- Scope made explicit: WEB detects **counterfactual demographic bias**, not factual error or
  consensus deviation. Benchmark numbers from prior work (MALIBU, BBQ-Hard, the 57.5% SDC
  figure) are cited as prior work only, never as this library's validated results.
