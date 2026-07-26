# DESIGN.md — weighted-emergent-bias

**Status:** planning document. No implementation exists yet. Every number quoted from
prior work is attributed, and none of it is a result produced by this library.

---

## 1. Elevator pitch

In a multi-agent LLM pipeline, one agent's mildly stereotyped output becomes the next
agent's ground truth. Downstream agents do not re-litigate the premise they were handed —
they build on it, and the bias compounds through the graph until every stage has
homogenized around the same skewed register. This failure mode is called
Degeneration-of-Thought (DoT), and single-model alignment does not catch it: the bias is
not a property of any one model's weights, it is a property of how the agents are wired
together. `weighted-emergent-bias` is a runtime circuit-breaker for that failure. It
probes each node with a demographically perturbed counterfactual to get a local bias score,
weights that score by the node's downstream blast radius via graph centrality, accumulates
the weighted scores into a network-level moving average as the graph executes, and halts
the run deterministically when that average crosses a threshold — freezing the compromised
payload and rerouting control to a mitigation subgraph instead of letting the contaminated
state propagate.

---

## 0. Scope: what this detects — and what it does not

This boundary is load-bearing and was made explicit after the 2026-07 external review (see
[reviews/2026-07-external-review-response.md](reviews/2026-07-external-review-response.md)).

The core detector is a **within-node counterfactual invariance test**. It runs a node on an
input and again on a demographically perturbed twin of that input, and measures how much the
node's *own* output distribution shifts, standardized against the node's *own* sampling-noise
floor. The anchor is the node's unperturbed output — **not** the other agents' consensus.

**Therefore WEB detects:** demographic / stereotype bias — a node treating an input
differently because of a protected attribute or its proxies.

**WEB does not detect:**

- **Factual error.** A node can be perfectly invariant across demographics and still be
  wrong. Catching that needs an external ground-truth anchor, which counterfactual probing
  does not provide. It is offered only as an *optional injectable verifier* (see §8), never
  asserted by the core.
- **Consensus deviation.** WEB never flags a node for disagreeing with its peers. A lone
  correct dissenter that answers identically across demographics scores at the noise floor.
  This is deliberate: consensus-based flagging is exactly the "punish the correct minority"
  failure the review warned about, and the counterfactual design sidesteps it by never
  comparing a node to other nodes to compute `B_i`.
- **Style drift** as a bias signal — it is a separate axis and is not conflated with the
  demographic divergence measure.

Keeping the claim this narrow is what makes it defensible: every part of the noise-floor
apparatus supports *this* statement and no broader one.

---

## 2. Gap analysis

### 2a. Genuinely novel engineering — build this carefully

These are the parts where the work is real and the design decisions are load-bearing.

**The noise floor: separating non-determinism from systematic bias.** This is the single
most important piece of the project and the source documents assert it without specifying
it. An LLM sampled twice on the *identical* prompt produces different text, so a raw
divergence between a standard probe and a counterfactual probe is *always* greater than
zero — even for a perfectly unbiased model. Any system that thresholds on raw divergence is
thresholding on sampling temperature. The fix: for every probe, also compute a
*same-input* divergence (baseline vs. a re-sample of baseline) to estimate the model's
intrinsic output variance at that node, and define the local bias score as the
counterfactual divergence *in excess of* that null distribution, ideally as a standardized
effect size with a bootstrap confidence interval rather than a bare subtraction. Without
this, every downstream number in the system is uninterpretable. This is the load-bearing
contribution and it should be built first, tested hardest, and documented most carefully.

**Divergence estimation under partial observability.** LLM output distributions are
frequently inaccessible — hosted chat APIs often expose no logprobs, and agentic
tool-calling outputs have no meaningful token distribution at all. The design uses a tiered
estimator: Jensen–Shannon divergence over top-k logprobs where available, falling back to
an embedding-space distance over the two output texts where not. JS specifically (not KL)
because it is symmetric, bounded in [0, 1] under log base 2, and finite when the two
supports differ — KL diverges to infinity the moment the counterfactual puts mass on a
token the baseline did not, which for top-k truncated distributions is the common case, not
the edge case. Every `BiasScore` must carry the method that produced it so that scores from
different estimators are never silently averaged together; they are not on the same scale.

**Katz centrality over an agent DAG, correctly oriented.** Two non-obvious properties worth
getting right. First, on a true DAG the adjacency matrix is nilpotent, so its spectral
radius is zero and the Katz series `(I − αA)^-1` converges for *every* attenuation factor α
— the usual `α < 1/λ_max` constraint simply does not bind, which makes Katz unusually
well-behaved here and lets α be tuned purely as a "how far downstream do I care" knob.
Second, and this is an easy correctness trap: the standard Katz formulation (and
`networkx.katz_centrality`) measures influence flowing *in* along incoming edges. We want
the opposite — a node's blast radius is the set of nodes reachable *from* it. The
computation must run on the transposed adjacency matrix. Getting this backwards produces a
plausible-looking number that ranks terminal leaf nodes as maximally critical, which is
exactly inverted from the intent.

**EWMA accumulation with parallel supersteps and warm-up correction.** The source
formulation, `B_net,t = α·(B_i·w_i) + (1−α)·B_net,t−1`, assumes one node updates per step.
Real graphs fan out: a superstep can fire five nodes at once, and applying the recurrence
five times in arbitrary order makes the result order-dependent and non-reproducible. The
accumulator needs a defined reduction over concurrently-firing nodes (weighted mean by `w_i`
is the defensible default; max is the paranoid alternative) applied once per superstep.
Separately, initializing `B_net,0 = 0` biases the average downward for the first ~1/α steps,
which systematically *delays* threshold breach exactly during the early graph stages where
interception is most valuable. Apply the standard bias correction, dividing by
`1 − (1−α)^t`, or the breaker is least sensitive precisely when it matters most.

### 2b. Framework plumbing — use the platform, don't reinvent it

Real work, but the design questions are already answered by LangGraph. Do not
over-engineer these:

- Threading a `BiasState` sub-schema through shared graph state — a `TypedDict` with the
  appropriate reducer annotations.
- Deterministic halt and reroute via `Command(goto=..., update=...)`.
- Parallel skeptic fan-out via `Send` — a standard map-reduce pattern.
- Append-only audit accumulation — a list channel with an `operator.add` reducer.
- Node wrapping/instrumentation — a decorator over the user's node callables.

The one genuinely non-trivial integration concern is that the LOOC probe doubles (at
minimum) the inference calls on the critical path, so the wrapper must support async
concurrent execution of the standard and counterfactual probes rather than running them
sequentially.

### 2c. Claims to soften — and numbers you must NOT present as your own

Read this section before writing any marketing copy. Each of these is a real published
result from someone else's paper, measured on someone else's setup. None of them is
evidence that *this library* works, and several do not even measure the thing this library
does.

| Claim in source docs | Actual provenance | How to present it |
| --- | --- | --- |
| **"57.5% average Bias Score reduction" from SDC + LOOC** | Beyond Generation (ACL 2025 Findings), on Super-NaturalInstructions classification and multiple-choice tasks | **Do not claim this.** Beyond the attribution problem, there is a scope mismatch that matters more: that figure measures *single-model label bias* — a model's tendency to favor certain answer options — not multi-agent emergent social bias. It is the wrong metric on the wrong unit of analysis. Cite as "the LOOC+SDC calibration approach this design borrows from reports a 57.5% label-bias reduction in its original single-model setting." |
| **MADERA on BBQ-Hard: +8pp accuracy, bias −0.08; GPT-4 0.71→0.96, bias −0.29→−0.04** | Towards Fairer AI (AAAI-SS), specific model versions, specific prompt set | **Do not claim this.** These are prior-work results for a standalone debiasing pipeline, not for this library's reimplementation of it. Say "inspired by MADERA," and note the reimplementation is unvalidated. |
| **MALIBU findings on collaborative amplification and inverse bias** | MALIBU benchmark paper (arXiv 2507.01019) | Safe to cite as *motivation* — it documents the problem this library targets. Never imply this library has been evaluated on MALIBU until it has. |
| **"No architecture has closed the operational loop"** | Novelty claim in the source doc | Soften to "we are not aware of an existing runtime, graph-aware implementation." Unfalsifiable priority claims age badly and invite easy rebuttal. |
| **McKinsey trust formula `T = (C+R+I)/S` as a debate aggregation weight** | Business-consulting heuristic, adapted by CortexDebate | Present as a heuristic, explicitly. It has no theoretical justification as an ML aggregation weight and needs its own ablation before anyone should believe it beats uniform averaging. It also has a hard numerical failure: `S → 0` sends `T → ∞`. Clamp the denominator and document the clamp. |
| **EEOC four-fifths rule as a compliance gate** | US employment-discrimination guidance | Handle carefully. The four-fifths rule applies to *selection rates* in employment decisions, not to divergence scores between text outputs. Wiring it to `B_net` is a category error, and shipping it as a "compliance check" invites someone to rely on it legally. If included at all, gate it behind an explicitly-named employment-screening module with a prominent not-legal-advice disclaimer. |
| **"SARIF-style" audit reporting** | Static Analysis Results Interchange Format | Only claim "SARIF 2.1.0 export" if the output validates against the published JSON schema in CI. Otherwise say "SARIF-inspired JSON." |
| **"Doubles baseline inference cost"** | Estimate in source doc | It is a floor, not an estimate. With SDC prior generation, noise-floor re-sampling, and multi-axis perturbation, realistic overhead is 3–6× per instrumented node. State the honest range. |

**Bottom line:** until you have run your own evaluation, the only defensible framing is
"this implements mechanisms from *[cited papers]*; independent validation of this
implementation is pending." Ship the synthetic DoT simulation harness (see v0.2) so there
is *something* reproducible demonstrating the mechanics, and keep the benchmark
reproduction as an explicit, un-promised future milestone.

---

## 3. Public API surface

Twelve imports a user should need. Everything else is internal.

```python
# --- types ---
class BiasScore:
    """Local bias magnitude for one node, with the estimator, noise floor, and CI that produced it."""

class BreakerDecision:
    """Outcome of a threshold check: proceed, or halt with a reason, bias type, and frozen payload."""

# --- scoring ---
def compute_local_bias(baseline, counterfactual, *, noise_floor=None, method="auto") -> BiasScore:
    """Divergence between a standard and a counterfactual output, net of the node's sampling noise."""

class LOOCProbe:
    """Wraps one agent node: runs standard, counterfactual, and null re-sample probes concurrently."""

class SyntheticPriorGenerator:
    """SDC — has the model synthesize in-domain priors at runtime when no calibration data exists."""

def perturb(payload, axes) -> list[Perturbation]:
    """Generate demographic/proxy-variable counterfactuals of an input payload along the given axes."""

# --- topology ---
def katz_weight(dag, node_id, *, attenuation=0.5) -> float:
    """Downstream blast radius of a node: Katz centrality on the transposed agent DAG."""

def dependency_weights(dag, *, method="katz") -> dict[str, float]:
    """Normalized w_i for every node in one pass; the form you actually want at graph-compile time."""

# --- accumulation & breaking ---
class NetworkAccumulator:
    """Bias-corrected EWMA of weighted node scores, with a defined reduction over parallel supersteps."""

class CircuitBreaker:
    """Holds tau; converts an accumulated B_net into a proceed/halt BreakerDecision."""

# --- intervention ---
def route_intervention(decision) -> Literal["skeptics", "madera", "none"]:
    """Bifurcated routing: conformity spirals go to debate, parametric bias goes to reasoning repair."""

class SkepticPanel:
    """Parallel adversarial reviewers aggregated under a trust graph that prunes overconfident agents."""

class MaderaEditor:
    """Diagnose the biased logical jump, retrieve counter-evidence, rewrite the chain until B_net clears."""

# --- audit ---
class AuditTrail:
    """Append-only causal log of every probe, divergence, weight, and routing decision; exports SARIF/HTML."""
```

---

## 4. Module layout

No code yet — one line of responsibility per file.

```
src/weighted_emergent_bias/
  __init__.py              Public re-exports; the twelve names above and nothing else.
  types.py                 BiasScore, BreakerDecision, BiasType, Perturbation, NodeId dataclasses.
  clients.py               LLMClient / EmbeddingClient protocols — user supplies the implementation.

  scoring/
    divergence.py          Jensen-Shannon over logprobs; embedding-distance fallback; scale metadata.
    noise.py               Null-distribution estimation: same-input re-sampling, effect size, bootstrap CI.
    perturbation.py        Demographic axes and proxy-variable substitution to build counterfactual inputs.
    probe.py               LOOCProbe — orchestrates concurrent standard/counterfactual/null probes.
    synthetic.py           SDC — runtime generation of in-domain priors when no calibration set exists.

  topology/
    dag.py                 Framework-agnostic DAG protocol (nodes, edges) + adapters from concrete graphs.
    centrality.py          Katz on the transposed adjacency; out-degree and betweenness alternatives.

  accumulation.py          NetworkAccumulator — bias-corrected EWMA, superstep reduction policies.
  breaker.py               CircuitBreaker — tau comparison, payload freezing, bias-type classification.

  intervention/
    router.py              Maps a BreakerDecision to a mitigation strategy.
    skeptics.py            SkepticAgent protocol + Empirical Auditor / Devil's Advocate / Diversity Champion.
    trust.py               Trust graph: clamped (C+R+I)/S scoring, edge pruning, response anonymization.
    madera.py              Three-phase diagnose -> retrieve -> iteratively-rewrite reasoning repair.

  integrations/
    langgraph/
      state.py             BiasState TypedDict and channel reducers for the shared graph state.
      guard.py             Node decorator that instruments an existing LangGraph node with a LOOC probe.
      nodes.py             Prebuilt breaker / router / skeptic-fanout / aggregator nodes using Command+Send.

  audit/
    trail.py               Append-only causal event log; the single source of truth for reporting.
    sarif.py               Schema-validated SARIF 2.1.0 serialization of the trail.
    report.py              HTML and JSON run reports with the bias trajectory over supersteps.
```

---

## 5. The five riskiest decisions

| # | Decision | Mechanism options | Runtime suitability | Risk | Recommendation |
| --- | --- | --- | --- | --- | --- |
| 1 | **Centrality measure for `w_i`** | Out-degree `O(1)`; Katz `O(n³)` dense but `O(V+E)` per-node on a DAG via reverse-topological DP; betweenness `O(V·E)` | Katz is fine *if computed once at graph-compile time*, not per superstep. Betweenness is not viable for dynamic graphs. | Recomputing Katz on every state update silently turns an `O(1)` guard into the dominant cost on large graphs. | **Katz, computed once at compile time and cached**, invalidated only on topology change. Because DAG adjacency is nilpotent the series always converges, so α is a free tuning knob. Expose out-degree as a cheap fallback for graphs that rewire every step. |
| 2 | **EWMA α: user-tunable or fixed** | Fixed constant; user-tunable; auto-derived from graph depth | Tunable costs nothing at runtime. | A free α is the easiest way for a user to make the breaker never fire — and there is no principled default, since the right α depends on graph depth and whether you fear spikes or slow drift. | **Tunable with a documented default of α ≈ 2/(depth+1)**, plus mandatory warm-up bias correction. Ship a calibration utility that sweeps α and τ against recorded runs; refuse to ship a magic number pretending to be validated. |
| 3 | **Skeptic/MADERA agents: built-in or injectable** | Built-in with a bundled SDK; protocols only; protocol + reference impl with injected client | Identical at runtime. | Bundling an SDK pins a vendor and makes the test suite need network access or heavy mocking. Protocols-only means the repo cannot demonstrate its headline feature. | **Protocol + reference implementation, BYO LLM callable.** No hard provider dependency; prompts overridable; the whole intervention path testable against a deterministic fake client. |
| 4 | **Distinguishing bias from sampling noise** | Threshold raw divergence; fixed noise constant; per-node empirical null distribution | Per-node null estimation adds an extra inference call per probe. | Skipping it makes the entire system unfalsifiable — you cannot tell a biased node from a hot-temperature node, which is the exact failure the project claims to fix. | **Per-node empirical null, non-negotiable.** Make the extra sample configurable (cached, sampled, or amortized across supersteps) but never default it off. Report `B_i` as a standardized effect size with a bootstrap CI, not a bare difference. |
| 5 | **Probe cost on the critical path** | Probe every node; probe by centrality; sample probabilistically | 3–6× inference on instrumented nodes is the real range, not 2×. | Uniform probing makes the library too expensive to run in production, so it gets switched off — a guard that is disabled protects nothing. | **Centrality-scaled probing:** high-`w_i` nodes get full multi-axis probes, low-`w_i` nodes get sampled or skipped, with the sampling rate recorded in the audit trail so coverage is never silently overstated. Run standard and counterfactual probes concurrently, never sequentially. |

---

## 6. Milestones

Each ships something independently testable.

**v0.1 — Single-node LOOC scoring.** No graph, no LangGraph, no network. `perturb`,
`compute_local_bias`, the JS/embedding tiered divergence, and the noise-floor estimator.
*Done when:* a synthetic fake client with a known injected bias produces `B_i` that
separates cleanly from an unbiased control, and an unbiased-but-high-temperature client
scores at the noise floor rather than above it. That second test is the one that matters.

**v0.2 — Topology and accumulation.** `dependency_weights`, transposed Katz,
`NetworkAccumulator` with superstep reduction and warm-up correction, plus a synthetic DoT
simulation harness: a scripted multi-agent DAG with a deliberately biased seed node.
*Done when:* the harness shows `B_net` rising monotonically as the seeded bias propagates,
and staying flat on the unbiased control graph.

**v0.3 — LangGraph integration.** `BiasState`, the node-instrumenting decorator,
`CircuitBreaker` wired to `Command()` for deterministic halt and reroute.
*Done when:* a real LangGraph run halts at the seeded node and the frozen payload is
recoverable from state.

**v0.4 — Interventions.** `SkepticPanel` with `Send()` fan-out, the clamped trust graph
with pruning and anonymization, and the MADERA-style diagnose/retrieve/rewrite pipeline.
*Done when:* a halted run completes the mitigation loop, returns to the primary graph, and
post-intervention `B_net` measurably drops — with an ablation against uniform aggregation,
because the trust formula is a heuristic that has to earn its place.

**v0.5 — Audit and reporting.** `AuditTrail`, schema-validated SARIF 2.1.0 export, HTML and
JSON run reports.
*Done when:* SARIF output validates against the published schema in CI and the trail traces
a breach back to its originating node and perturbation.

**Explicitly not scheduled:** MALIBU / BBQ-Hard reproduction. It is a large, API-cost-heavy
workstream. Until it happens, the README makes no validated-performance claims.

---

## 7. Open questions

- Which demographic axes ship as defaults, and does shipping a fixed list encode its own
  bias? Leaning toward: no defaults, a required explicit config, and documented example sets.
- Should `B_net` be per-run or persist across runs to catch drift over a deployment's
  lifetime? Per-run for v0.x; cross-run is a separate telemetry concern.
- How does the breaker behave on cyclic graphs? LangGraph permits cycles, at which point
  Katz's nilpotency guarantee disappears and the `α < 1/λ_max` constraint returns. Needs an
  explicit detection-and-fallback path before v0.3.
- **Pre-Trigger Verification scope fork (pending user decision).** Should factual-error
  verification be part of the core, or an optional injectable `Verifier` hook? Recommended:
  optional hook, keeping the core claim counterfactual-only. See §8 and the review response.

---

## 8. Revisions from the 2026-07 external review

A three-reviewer architecture review raised 14 gaps; the full point-by-point response,
including two P0 items reframed as based on a consensus-detection misreading, is in
[reviews/2026-07-external-review-response.md](reviews/2026-07-external-review-response.md).
None required changes to shipped code (WP0/WP1). The accepted mechanism changes are recorded
here against their owning module so they are not lost before those modules are built.

**Detection (M1).** Scope boundary made explicit (§0): counterfactual invariance, not factual
correctness or consensus deviation. Divergence measure formalization tracked in PHASE-1 R1.

**Propagation (M2).**
- **Bayesian error-history score** `P(biased | history_i)` as a *complement* to topological
  `w_i`, not a replacement: topology gives blast radius, history gives base rate; both feed a
  composite risk. (Addresses the minority-suppression critique at its real location — a node
  is scrutinized on structure and track record, never penalized for dissent.)
- **Multi-scale EWMA:** maintain a fast and a slow accumulator (distinct decay rates) so
  sudden spikes and slow drift are caught simultaneously instead of traded off against one α.

**Control (M3).**
- **Two-threshold hysteresis** (`τ_enter` > `τ_exit`) plus a continuous mixing ratio replaces
  the single binary `τ`, to stop breaker thrashing when `B_net` sits near the boundary.
- **Edge-level interception:** trip on the edge *before* the downstream node consumes the
  payload, not after the upstream node emits — otherwise the biased context has already
  propagated by the time the breaker fires.
- **Four-state machine** Normal → Warning → Intervention → Recovery, with transition guards, a
  cool-down window, and a human-escalation escape condition. Defines the recovery/re-entry
  protocol left unspecified in the source docs.
- **Router self-monitoring:** the reroute node is itself probed and audited; it is not an
  exempt, unmonitored amplifier.

**Intervention (M4).**
- Minority-suppression guardrails in trust-graph pruning: response anonymization and a
  "never prune on divergence from consensus alone" rule, so a correct dissenter cannot be
  silenced for dissenting.
- Skeptic SPOF mitigation: require ≥2 skeptics of diverse provenance; the BYO-callable design
  already avoids a single bundled agent.

**Evidence (M5).**
- Evaluation protocol: four-condition ablation (baseline → LOOC-only → LOOC+centrality → full
  breaker) with bias reduction, accuracy, latency/step, intervention frequency, recovery rate,
  and a reverse-bias symmetry check.
- Per-failure-mode error budget (target FPR and missed-bias rate).
- Observability: extend the SARIF trail with trigger reason, intervention path, and routing
  entropy over time.

**Cross-cutting.** Monitor independence (WP5): the probe may take an optional independent
monitor client (different model family, read-only) so the monitor does not inherit the blind
spots of the agent it watches.
