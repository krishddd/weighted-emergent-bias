# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project uses
[Semantic Versioning](https://semver.org/) once it reaches a release.

## [Unreleased]

## [0.6.0] — 2026-08-16

**Correctness and distribution.** Six defects fixed where the code contradicted its own
documented guarantees — two of them in the safety path (trust weighting, breaker escalation) and
live in tagged v0.5.0. First release published to PyPI, with documentation on GitHub Pages.

### Added
- **Optional pre-trigger `Verifier`** (`intervention/verifier.py`): the scope fork decided in the
  2026-07 review, now built. Opt-in and un-imported by the core; keeps *counterfactual bias* and
  *factual error* as separate signals rather than blending them. `UNVERIFIED` never blocks — an
  oracle that reached no conclusion is not evidence of error.
- **Real-model adapter + validation harness** (`integrations/anthropic_client.py`,
  `studies/validate_real_model.py`, optional `[anthropic]` extra): makes real-model validation one
  command once an API key is supplied. Never runs in CI; makes billable calls.
  **Resolves Phase 1 spike S1** — the Anthropic Messages API exposes no token logprobs, so
  `TaskMode.CHOICE` is unreachable analytically; the adapter recovers it by Monte Carlo (one-hot
  draws whose group mean is the empirical distribution), at the cost of needing a larger `n`.

### Fixed
- **Trust weighting was inverted** (`intervention/trust.py`): self-orientation `S` took the raw
  confidence, so a no-evidence verdict at `confidence=0.0` landed on `S=0`, hit the `_S_FLOOR` clamp
  and was handed ~2300× the weight of an evidence-backed verdict — the clamp added to *prevent*
  `S→0` divergence was the thing triggering it. `S` is now an interpolation over `[0.25, 1.0]`, so
  an unevidenced verdict never outweighs an evidenced one of equal stance and loses weight as it
  grows more assertive. In `trust` mode this had let one content-free `STANDS` override three
  evidenced `REVISE`s at 98.5% reported confidence.
- **`max_retries` had become a lifetime cap** (`breaker.py`): `_attempts` was never reset on a
  successful recovery, so a long run that cleanly self-healed `max_retries` separate times escalated
  to human review on the strength of its recoveries. The counter now resets when an incident closes;
  the bound is again per-incident. An escalated decision also retains the payload that caused the
  halt even after `B_net` falls back, instead of handing human review `frozen: None`.
- **The append-only audit trail was mutable** (`audit/trail.py`): `detail` was a shallow `dict()`
  copy, rewritable both directly through the frozen event and through nested objects the caller
  still held. It is now a deep copy behind a `MappingProxyType`, so a recorded event cannot be
  edited — the guarantee the module documents.
- **`has_evidence` fired on prose about evidence** (`intervention/skeptics.py`): a substring match
  for `"evidence"` set the flag true on "no evidence provided", and the `empirical_auditor` prompt
  contains the word itself. Detection now requires an explicit `EVIDENCE:` marker or a URL. The
  reference prompt also now offers `REJECT` (previously unreachable) and asks for the marker.
- **Skeptic panel was all-or-nothing** (`intervention/skeptics.py`): one failing client took the
  whole `asyncio.gather` down and discarded the verdicts that did arrive. Failures are now dropped
  and surviving verdicts returned. Panel-decision ties break toward the more cautious stance
  (`REJECT > REVISE > STANDS`) instead of silently failing open on enum declaration order.
- **`calibrate_thresholds` returned unachievable thresholds** (`calibration.py`): a target rate
  finer than `1/n` made `np.percentile` interpolate to the sample maximum. It now requires
  `n ≥ ⌈1/target⌉` control samples and rejects an explicitly-empty `control_slow`.

### Changed
- **`compute_local_bias` permutation/bootstrap loops are vectorized** (`scoring/noise.py`):
  block-evaluated instead of one validated estimator call per draw. ~4–6× faster with **bit-for-bit
  identical** outputs on the same seed (the RNG stream is consumed unchanged).
- **`AgentDAG` caches its adjacency matrix and topological order** (`topology/dag.py`): the graph is
  immutable, so centrality no longer rebuilds them per lookup. `adjacency_matrix()` still returns a
  fresh copy. Corrected the `dag.py` docstring that told a maintainer to transpose `A` — the code
  correctly consumes it as-is, and a transpose would re-invert the blast-radius ranking. The same
  misleading "transposed" wording is corrected in the README.

### Packaging
- **Published to PyPI** as [`weighted-emergent-bias`](https://pypi.org/project/weighted-emergent-bias/),
  built and uploaded by `.github/workflows/release.yml` on a `v*` tag via PyPI Trusted Publishing
  (OIDC — no API token is stored). The workflow re-runs the full lint/type/test gate and asserts the
  git tag matches the packaged version before building.
- **Documentation site** at <https://krishddd.github.io/weighted-emergent-bias/>, built with
  MkDocs Material (`[docs]` extra, `mkdocs.yml`) and deployed by `.github/workflows/docs.yml`.
  Mermaid diagrams render natively.

### Note for anyone who ran the M4 ablation before this release
The WP7 trust-vs-uniform comparison was measuring the inverted trust score described above.
Any numbers produced from it before `baf4fdb` should be re-run.

### Still not claimed
Benchmark reproduction remains unscheduled, and **no validated-performance claim on real models has
been made** — the harness exists, but until someone runs it and publishes numbers the claim stays
"demonstrated on the ground-truth fake". A single run is evidence, not validation.

## [0.5.0] — 2026-08-01

**M5 — Evidence.** Makes every decision auditable: an append-only causal trail, SARIF 2.1.0 export
(validated against the published schema in CI), and self-contained HTML/JSON reports with a breach
trace-back. Characterized by the [Phase 5 study](docs/studies/phase5-evidence.md): a real M1→M5 run
records a trail whose SARIF validates, and whose report traces the breach to the originating node and
axis. **This completes the pipeline** — detect → weight/accumulate → control → intervene → audit —
still with no validated-performance claims on real models.

### Added
- **M5 WP1 — audit trail** (`audit/trail.py`): append-only `AuditTrail` of monotonic-seq `AuditEvent`s
  (wall-clock optional + injected), with recorders for `BiasScore` / `BreakerDecision` / `PanelResult`.
- **M5 WP2 — SARIF export** (`audit/sarif.py`): `to_sarif` producing a SARIF 2.1.0 log, validated
  against the bundled schema with `jsonschema` in CI.
- **M5 WP3 — reporting** (`audit/report.py`): `to_json`, self-contained `to_html`, and `trace_breach`.
- **M5 WP4 — evidence study** (`studies/phase5_audit.py`, [report](docs/studies/phase5-evidence.md)).

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
