# Response to external architecture review (2026-07)

Three independent reviewers (DeepSeek, Codex, Claude Code) raised 14 gaps and proposed 7
new document sections. This is a point-by-point response: what is valid and banked, what is
already handled, and — for two P0 items — what is based on a reading of the architecture
that does not match what is being built.

**TL;DR.** Most of the feedback is valid and lands in M2–M5; it is banked below and folded
into the design docs. No item is a defect in the code shipped so far (WP0/WP1). Two P0
detection gaps (#2, #4) partly rest on interpreting the detector as *consensus-based*; it is
not, and the distinction changes the fix. One proposed section (Pre-Trigger Verification) is
a genuine scope fork and is flagged for a decision rather than silently adopted.

---

## The distinction that reframes gaps #2 and #4

Several critiques assume WEB flags a node for **diverging from the other agents' consensus**.
It does not. The core detector is a **within-node counterfactual invariance test**:

> Run node *i* on input *x*, and again on *x′* where only a demographic attribute (and its
> proxies) has been perturbed. The bias signal is the divergence between the node's *own two
> output distributions*, standardized against the node's *own* sampling-noise floor.

The anchor is the node's unperturbed output, not its peers. Consequently:

- **"A 5th correct agent gets flagged while 4 wrong agents reinforce each other"** (the
  consensus≠truth problem) **does not arise in detection.** WEB never compares a node to
  other nodes to produce `B_i`. A lone correct dissenter that answers *identically* for `he`
  and `she` scores at the noise floor, regardless of what the majority says.
- **Centrality does not suppress minorities in the detection layer.** `w_i` is purely
  topological (downstream blast radius from graph structure). It *amplifies the scrutiny* a
  high-fan-out node's bias score receives; it never downweights a node's *content* in any
  vote. There is no vote.

**What the critiques get right, and where the concern actually lands:**

1. Counterfactual invariance ≠ factual correctness. WEB detects **demographic/stereotype
   bias**, not **factual error** or **style drift**. A node can be perfectly invariant and
   perfectly wrong; WEB will not catch that, by design. This scope boundary was implicit and
   is now made explicit (DESIGN.md new §0). Factual-error detection needs an external
   ground-truth anchor — that is the Pre-Trigger Verification fork (below), not the core.
2. The consensus-suppression concern is real **in the M4 skeptic debate**, where trust-graph
   pruning *could* silence a correct minority. That is where gap #4's fix belongs, and the
   planned anonymization + "never prune on divergence alone" rules are the response.
3. The Bayesian error-history score (#4's proposed replacement) is a good **complement** to
   topological `w_i` in M2 — not a replacement. Topology answers "how far would this node's
   bias spread"; error-history answers "how often has this node been biased before". Both
   feed the composite risk. Banked.

---

## Triage table

| # | Gap | Owner | Verdict | Action |
|---|-----|-------|---------|--------|
| 1 | `b_i(t)` divergence undefined | M1 / WP3 | Valid; already specified | Formalize JS-over-shared-support in `divergence.py`; worked example in the WP7 study. Justification (JS vs KL/Wasserstein) already in PHASE-1 R1. |
| 2 | LOOC lacks ground-truth anchor | M1 | Reframe | Detector is invariance-anchored, not consensus. Scope out factual error explicitly (DESIGN §0). External verification = optional fork below. |
| 3 | Hard threshold τ thrashing | M3 | Valid | Replace binary τ with two-threshold hysteresis (τ_enter, τ_exit) + continuous mixing ratio. Banked to M3. |
| 4 | Centrality minority-suppression loop | M1→M4 | Reframe | No suppression in detection (`w_i` is topological). Concern applies to M4 trust-pruning. Add Bayesian error-history as an M2 complement to `w_i`. |
| 5 | Node-level interception misses edge propagation | M3 | Valid | Intercept on the **edge**, before the downstream node consumes the payload — not after the node emits. Banked to M3. |
| 6 | Monitor independence not enforced | WP5 | Valid | Probe accepts an optional **independent monitor client** (different model family / read-only). Document the independence contract. Flagged in `clients.py`. |
| 7 | Skeptic SPOF + shared-prior risk | M4 | Valid | Require ≥2 skeptics of diverse provenance; BYO-callable design already avoids a single bundled agent. Banked to M4. |
| 8 | LOOC cost — no sampling strategy | WP5 | Partly handled | Centrality-scaled probing already in DESIGN risk #5; formalize the sampling **policy** (rates, caching, amortized null) in WP5. |
| 9 | McKinsey trust formula ungrounded | M4 | Already addressed | DESIGN §2c already treats it as a heuristic that must beat uniform aggregation in an ablation, with a denominator clamp. No change. |
| 10 | Routing layer itself unmonitored | M3 | Valid | The reroute/router node gets its own probe + audit entry; it is not exempt. Banked to M3. |
| 11 | Single-scale EWMA misses drift + spike | M2 | Valid | Run **fast + slow** EWMAs (S_fast, S_slow) concurrently; fast catches spikes, slow catches drift. Banked to M2. |
| 12 | No formal error budget | M5 / study | Valid | Define per-failure-mode error budget (FPR, missed-bias rate) in the evaluation protocol. Banked. |
| 13 | Recovery / re-entry unspecified | M3 | Valid | Four-state machine Normal→Warning→Intervention→Recovery with guards, cool-down, and a human-escalation escape. Banked to M3. |
| 14 | No evaluation protocol w/ baselines + ablations | M5 / study | Valid | Four-condition ablation: baseline → LOOC-only → LOOC+centrality → full breaker. Metrics incl. reverse-bias symmetry. Banked. |

---

## Proposed new sections → where each lands

| Proposed section | Disposition |
|---|---|
| 1. Formal Bias Score Definition | PHASE-1 R1 (done) + `divergence.py` (WP3) + worked example in WP7 study. |
| 2. **Pre-Trigger Verification Layer** | **Scope fork — see below.** Adopted as an *optional injectable verifier*, not core. |
| 3. Continuous Mixing + Hysteresis | M3 design — banked. |
| 4. Multi-scale EWMA + Bayesian reliability score | M2 design — banked (error-history complements, not replaces, centrality). |
| 5. Monitor-independence contract + state machine | Independence: WP5 / `clients.py`. State machine: M3. |
| 6. Observability & audit log | M5 — extends the already-planned SARIF trail with routing entropy + trigger-reason fields. |
| 7. Evaluation protocol w/ baselines & ablations | M5 / validation study — banked. |

---

## The one genuine scope fork: Pre-Trigger Verification

Proposed section 2 asks for an external verifier *between* detection and the breaker, so the
system halts on **wrong** outputs, not just **demographically non-invariant** ones. This is a
real, defensible idea — and it changes what the project *is*:

- **Keep counterfactual scope (recommended).** WEB stays a bias circuit-breaker: it detects
  and interrupts demographic bias propagation, with an *honest, narrow, anchor-free* claim.
  Factual verification is offered as an **optional injectable `Verifier` hook** the user can
  wire in (and MADERA already does retrieval in M4). The core makes no correctness claim.
- **Expand to error+bias detection.** WEB becomes a general agent-output verifier. Much
  larger scope, needs a ground-truth oracle on the critical path, and inherits every hard
  problem of automated fact-checking. Different project.

**Recommendation: keep counterfactual scope, ship the verifier as an optional hook.** The
narrow claim is the defensible one, and it is the claim the whole noise-floor apparatus
actually supports. This is surfaced to the user for an explicit decision; nothing is
foreclosed.

---

## Net effect on the plan

- **No code change in WP0/WP1.** The types already carry uncertainty (CI + p-value), estimator
  identity (`method`), and audit fields; the fake already separates bias from model variance.
- **WP2 (next) is unchanged** — perturbation is the same regardless of these decisions.
- **M2 gains:** Bayesian error-history score + multi-scale EWMA (composite risk).
- **M3 gains:** hysteresis controller, edge-level interception, the four-state machine,
  router self-monitoring, recovery protocol.
- **M4 gains:** the minority-suppression guardrails (anonymization, no-prune-on-divergence),
  skeptic diversity requirement.
- **M5 gains:** the ablation evaluation protocol, error budget, extended observability.
- **One decision pending from the user:** the Pre-Trigger Verification scope fork.
