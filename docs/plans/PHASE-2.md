# PHASE 2 — Propagation (M2, v0.2)

**Goal.** Turn the per-node `BiasScore`s from M1 into a network-level signal `B_net` that tracks
how bias accumulates and propagates as a graph executes — weighting each node's bias by its
downstream blast radius, and integrating over execution steps at two timescales.

**Explicit non-goals.** No threshold, no breaker, no halt (M3). No LangGraph (M3). No
interventions (M4). M2 produces a *number that rises when bias propagates*; deciding what to do
about it is M3's job. Adding a threshold here would let a mis-weighted signal hide behind a tuned
cutoff — the same discipline that kept M1 honest.

**The one-sentence success test.** On a scripted multi-agent DAG with one deliberately biased
seed node, `B_net` must rise monotonically as the seeded bias propagates downstream, and stay flat
on an unbiased control graph with identical topology. If it rises on the control, the weighting or
accumulation is wrong.

**Inheritance from Phase 1.** Two M1 findings are load-bearing here and are treated as settled
inputs, not open questions:

- **Effect size is n-dependent** ([WP7 study](../studies/phase1-calibration.md), finding #4). It is
  a standardized *detection* statistic that inflates with sample count, so it is **not** the
  quantity M2 weights on. See R1.
- **Scores from different `task_mode`s are not comparable** (M1). `B_net` is computed per mode; a
  graph mixing CHOICE and GENERATIVE nodes needs per-mode accumulators, not one blended number.

---

## 1. Research questions

### R1 — What per-node magnitude `b_i` feeds accumulation? ⚠️ resolved by WP7, do not re-open

`effect_size` is tempting (it is *the* headline number) and wrong: WP7 showed the same true bias
reads as 1.9σ at n=3 and 22.9σ at n=30, because `null_std → 0` as n grows. Weighting on it would
make `B_net` depend on each node's sample budget, not its bias.

The n-stable magnitude is the **excess divergence over the noise floor**:

    b_i = max(0, raw_divergence − null_mean)

As n→∞, `raw_divergence →` the true divergence and `null_mean → 0`, so `b_i` converges to a fixed
per-node quantity (bounded [0,1] for CHOICE/JSD). `effect_size` and `p_value` are kept as a
**significance gate**, not a magnitude:

    b_i = (raw_divergence − null_mean)⁺   if p_value ≤ α   else 0

**Recommendation:** accumulate the significance-gated excess divergence. This combines "is the bias
real?" (the p-value the whole noise-floor apparatus exists to produce) with "how big is it?" (an
n-stable magnitude). Expose the gate α as a parameter; document that raw `effect_size` must never be
fed to the accumulator.

### R2 — Katz orientation and cyclic graphs

Two facts from DESIGN §2a, now to be implemented:

- **Transposed adjacency.** A node's blast radius is what it can *reach*, not what reaches it.
  `networkx.katz_centrality` measures the latter; M2 computes Katz on `Aᵀ`. Getting this backwards
  ranks terminal leaves as maximally critical — plausible-looking and exactly inverted.
- **DAG nilpotency.** On an acyclic graph the adjacency is nilpotent (spectral radius 0), so the
  Katz series `(I − αAᵀ)⁻¹` converges for *every* attenuation α — α is a free "how far downstream do
  I care" knob, not a stability constraint.

**Cyclic fallback (must-have before v0.3, designed now).** LangGraph permits cycles, at which point
nilpotency is gone and α must satisfy `α < 1/λ_max(A)`. M2 must detect cycles and either (a) clamp α
below the spectral-radius bound, or (b) fall back to out-degree centrality. **Recommendation:**
detect via `networkx.is_directed_acyclic_graph`; on a DAG use free α; on a cyclic graph clamp
`α = safety · 1/λ_max` (safety ≈ 0.5) and record in output that the clamp was applied — never
silently return a divergent or meaningless number.

### R3 — Composite weight: Katz alone, or Katz × error-history?

The review banked a Bayesian error-history score `P(biased | history_i)` as a complement to
topological Katz (DESIGN §8). There is a trap in it: **if error-history updates from the detector's
own unverified detections, it becomes a self-reinforcing suppression loop** — the exact failure
gap #4 warned about. A node flagged once gets a higher prior, gets weighted up, gets flagged more.

**Recommendation for v0.2:** Katz is the primary, always-on weight. The composite weight *accepts an
optional, externally-supplied* error-history prior (an injected per-node `P(biased)`), but M2 does
**not** self-update it from its own detections. Automatic error-history from a verified signal is
deferred to when the optional `Verifier` hook exists (the scope fork the user kept optional). This
keeps v0.2's weighting grounded in graph structure, which cannot feed a suppression loop.

    w_i = katz_i × (prior_i if supplied else 1.0),  then normalized over the graph

### R4 — Reducing concurrently-firing nodes within a superstep

A superstep can fire several nodes at once; applying the EWMA recurrence per node in arbitrary order
makes `B_net` order-dependent and non-reproducible (DESIGN §2a). The reduction must be a single
order-independent operation per superstep.

**Recommendation:** weighted mean of `b_i` by `w_i` across the firing set is the defensible default
(a central biased node dominates the step); expose `max` as the paranoid alternative. Never apply the
recurrence node-by-node within a step.

### R5 — Multi-scale EWMA, warm-up correction, normalization

The review banked a two-timescale accumulator (DESIGN §8): `S_fast` catches conformity spikes,
`S_slow` catches slow cultural drift, so one α no longer has to trade off between them.

- **Warm-up bias correction** (DESIGN §2a): initializing `S₀ = 0` biases the average downward for
  the first ~1/α steps — exactly the early graph stages where interception matters most. Divide by
  `1 − (1−α)ᵗ`.
- **Normalization:** `b_i` is already bounded [0,1] for CHOICE, so `S_fast`/`S_slow` are too; document
  that GENERATIVE (cosine, [0,2]) is a different scale and normalize per mode if needed.

**Recommendation:** `S_fast` with `α ≈ 0.7`, `S_slow` with `α ≈ 0.1` (the review's constants) as
documented, tunable defaults — no magic numbers pretending to be validated. `B_net` exposes both; M3
will trip on fast and drift-alert on slow.

---

## 2. Work packages

### WP1 — DAG protocol and adapters (`topology/dag.py`)
A framework-agnostic protocol for a directed graph: nodes, directed edges, and iteration. Adapters
from a plain `dict[node, list[node]]` and from a `networkx.DiGraph`. No LangGraph.
*Accept when:* a graph can be built from both sources and round-trips its edges; a property test
confirms adapter equivalence.

### WP2 — Centrality (`topology/centrality.py`)
`katz_weight(dag, node)` and `dependency_weights(dag)` on the **transposed** adjacency, with cycle
detection and the α-clamp fallback; out-degree as the cheap alternative.
*Accept when:* on a hand-checked chain `A→B→C`, the *upstream* node A outranks terminal C (blast
radius, not in-degree); a cyclic graph triggers the clamp and records it; results are cached and
invalidated only on topology change.

### WP3 — Per-node magnitude (`accumulation.py` or `scoring/`)
A pure function `BiasScore → b_i`: significance-gated excess divergence (R1). Refuses `effect_size`.
*Accept when:* `b_i` is invariant to n on the fake (the WP7 failure mode does not reappear); gated to
0 when `p_value > α`.

### WP4 — Composite weight
Combine Katz with an optional injected error-history prior (R3); normalize over the graph. No
self-updating.
*Accept when:* with no prior, weights equal normalized Katz; an injected prior shifts them
predictably; a supplied self-updating source is *not* accepted (there is no such code path).

### WP5 — Multi-scale accumulator (`accumulation.py`)
`NetworkAccumulator` maintaining `S_fast` and `S_slow` with warm-up correction and the superstep
reduction (R4, R5). One update per superstep.
*Accept when:* order-independence property test passes (shuffling a superstep's nodes gives identical
`B_net`); warm-up correction verified against a hand-computed early-step trajectory; both scales
respond with the expected lag.

### WP6 — Synthetic DoT simulation harness
A scripted multi-agent DAG with a biased seed node whose bias propagates to downstream nodes, plus an
unbiased control graph of identical topology. Uses the M1 fake.
*Accept when:* the success test holds — `B_net` rises monotonically on the seeded graph, flat on the
control — and `S_fast` leads `S_slow`.

### WP7 — Propagation study
A short written study (like Phase 1's): `B_net` trajectories, the effect of `w_i` (central vs leaf
seed), and fast-vs-slow behavior. Reproducible from committed code.
*Accept when:* the report exists with reproducible figures showing central-node bias produces a
larger `B_net` rise than leaf-node bias of the same magnitude — the whole point of topological
weighting.

---

## 3. Test strategy

Same four layers as Phase 1. Specific must-haves:
- **Property:** superstep reduction is order-independent; Katz on `Aᵀ` matches a direct
  reachability-weighted computation on small graphs; `b_i` is n-invariant.
- **Statistical:** on the DoT harness, monotonic rise on seeded vs flat on control, asserted over
  many seeds (not one trajectory).
- No network; all randomness via the seeded `rng` fixture.

## 4. Risks

| Risk | Signal | Response |
| --- | --- | --- |
| Someone feeds `effect_size` to the accumulator | `B_net` depends on sample budget | WP3 refuses it by construction; document loudly (R1). |
| Katz computed on `A`, not `Aᵀ` | leaf nodes rank as critical | Hand-checked chain test (WP2) catches the inversion. |
| Cyclic graph diverges | NaN/huge Katz on a LangGraph cycle | Cycle detection + α-clamp, recorded in output (R2). |
| Error-history suppression loop | correct minority silenced over runs | No self-updating error-history in v0.2; prior is injected-only (R3). |
| Superstep order-dependence | non-reproducible `B_net` | Single order-independent reduction per step (R4); property test. |
| Recomputing Katz every step | O(1) guard becomes dominant cost | Compute once at graph-compile, cache, invalidate on topology change (DESIGN risk #1). |

## 5. Definition of done

- [ ] `dependency_weights` on transposed adjacency; upstream-outranks-leaf test passes
- [ ] Cyclic-graph detection + α-clamp, recorded when applied
- [ ] `b_i` = significance-gated excess divergence; verified n-invariant on the fake
- [ ] `NetworkAccumulator` with fast+slow scales, warm-up correction, order-independent superstep reduction
- [ ] DoT harness: `B_net` monotonic on seeded graph, flat on control, over many seeds
- [ ] Propagation study published with reproducible figures (central vs leaf seed)
- [ ] `mypy --strict` clean on 3.10–3.12; no LangGraph, no breaker, no threshold
