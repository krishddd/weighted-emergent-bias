# PHASE 3 — Control (M3, v0.3)

**Goal.** A deterministic control plane that watches the `B_net` signal from M2 and, when bias
crosses a threshold, halts the run, freezes the compromised payload, and reroutes to a mitigation
subgraph — with **hysteresis** so it does not thrash near the boundary, and a **recovery protocol**
so a repaired run can re-enter. Ships a framework-agnostic core plus a LangGraph reference adapter.

**Explicit non-goals.** M3 does not *implement* interventions — skeptic debate and MADERA repair are
M4. The breaker routes control *to* a mitigation hook (an injectable callback) and consumes its
result; the hook's default is a no-op, so the whole control plane is testable without M4. No audit
trail / SARIF (M5). No new detection or propagation.

**The one-sentence success test.** On a `B_net` trajectory that rises through `tau_enter` and then
oscillates near it, the breaker enters Intervention **exactly once** and does not flip state on the
oscillation; it returns to Normal only after `B_net` drops below `tau_exit` *and* a cool-down
elapses — deterministically. Under LangGraph, the halt freezes the node's output in shared state and
reroutes via `Command` **before the downstream node runs**.

**Inheritance from prior phases (settled inputs, not open questions).**

- **`B_net` has two scales** (M2): a fast EWMA (spikes) and a slow EWMA (drift). The breaker
  consumes both — see R1.
- **`B_net`'s absolute scale depends on the Katz α and the propagation model** ([Phase 2 study §2](../studies/phase2-propagation.md)).
  So there is *no* portable magic threshold; M3 must ship a calibration path, not a constant — see R6.
- **Framework-agnostic core** (project principle): the breaker and state machine are pure Python,
  fully testable with no LangGraph and no numpy. LangGraph is an optional reference adapter.
- **Detection is counterfactual-invariance, scoped narrow** (DESIGN §0): the breaker halts on
  *demographic-bias propagation*, never on consensus deviation. Nothing here widens that claim.

---

## 1. Research questions

### R1 — Which scale(s) does the breaker act on?

M2 exposes `fast` and `slow`. The review (DESIGN §8) says trip on fast, drift-alert on slow.

**Recommendation:** map the two scales to two states. **`slow` crossing `tau_warn` → Warning**
(drift building; observe, maybe raise probing intensity later). **`fast` crossing `tau_enter` →
Intervention** (a spike that warrants halting). This uses both scales for what each is good at and
gives the state machine (R3) its natural triggers. A single-scale fallback (fast only) stays
available for users who do not want drift alerts.

### R2 — Hysteresis and the continuous mixing ratio

A single binary `tau` thrashes: a `B_net` sitting near it flips the breaker on/off every step
(gap #3). Two fixes, both banked:

- **Two thresholds** `tau_enter > tau_exit`. Enter Intervention at `tau_enter`; only leave once
  `B_net` falls below the *lower* `tau_exit`. The gap between them is the dead-band that kills
  oscillation.
- **Continuous mixing ratio** `m = sigmoid((B_net - tau_enter) / kappa)` in `[0, 1]` — a smooth
  "how compromised" severity rather than a hard 0/1.

**Recommendation:** the breaker returns a discrete decision (for the deterministic halt/reroute)
**and** carries `m` as advisory severity. The *application* of `m` (blending consensus vs. skeptic
output) is M4's job — M3 computes it, M4 consumes it. This keeps the halt deterministic (a decision
plane must be predictable) while still surfacing the smooth signal the review wanted.

### R3 — The recovery state machine

Four states plus a terminal escape (gap #13):

```
NORMAL --slow>=tau_warn--> WARNING --fast>=tau_enter--> INTERVENTION
   ^                          |                              |
   |                    fast>=tau_enter                 (run mitigation hook)
   |                          v                              v
   +--B_net<tau_exit & cool-down elapsed & recheck OK-- RECOVERY
                                                            |
                              max_retries exceeded ---------+--> ESCALATED (terminal, human review)
```

- **NORMAL → WARNING:** `slow >= tau_warn`. **WARNING → NORMAL:** `slow < tau_warn` (drift subsided).
- **any → INTERVENTION:** `fast >= tau_enter`. Halt, freeze the payload, run the mitigation hook.
- **INTERVENTION → RECOVERY:** hook returned. Enter a cool-down window.
- **RECOVERY → NORMAL:** cool-down elapsed **and** `B_net < tau_exit` **and** the corrected output
  passes its *own* re-check (its `B_net` contribution is below `tau_exit`).
- **RECOVERY → INTERVENTION:** re-check still bad → retry (increment attempt count).
- **→ ESCALATED:** attempts exceed `max_retries`. Terminal; hand to human review. Prevents the
  infinite correction loop the review flagged.

**Recommendation:** implement this as a pure state machine whose Intervention action is an
**injectable callable** (`Callable[[InterventionContext], InterventionResult]`), defaulting to a
no-op that leaves `B_net` unchanged (so the default machine, with no M4, escalates after
`max_retries` — a safe, honest default). All transition guards, the cool-down counter, and the
retry counter live here and are unit-tested without any framework.

### R4 — Edge-level interception (gap #5)

If the breaker checks *after* a node commits its output to shared state, the biased payload has
already reached the next node's context. The check must sit on the **edge**: node output goes to an
`unverified_output` staging buffer; the breaker inspects it; if clear it is **promoted** to main
state, if breached it is **frozen** and control reroutes — all before any downstream node activates.

**Recommendation:** the core exposes `freeze(payload)` and the promote/replace decision; the
LangGraph adapter realizes the staging buffer as a `BiasState` channel and gates promotion in the
node guard (R5). The core never assumes a framework; the adapter wires the buffer.

### R5 — LangGraph adapter scope, and how to test it

v0.3 ships the reference adapter: a `BiasState` `TypedDict` (local scores, weights, `network_ewma`,
`unverified_output`, `causal_audit_trail` placeholder for M5), a node-guard decorator that probes +
stages + gates promotion, and breaker/reroute nodes using `Command(goto=..., update=...)`.
`Send`-based skeptic fan-out is **M4**, not here.

**Recommendation:** keep the adapter thin (LangGraph's API churns; minimize surface). Test it with
`pytest.importorskip("langgraph")`, and add `langgraph` to the `dev` extra so CI actually exercises
it. If it destabilizes the 3.10–3.12 mypy matrix, isolate the adapter in a per-module mypy override
rather than weakening strictness elsewhere. The default test suite must still pass with LangGraph
absent (the core is what most tests cover).

### R6 — Threshold calibration (no magic tau)

The Phase 2 study showed `B_net`'s scale is deployment-dependent. Shipping a constant `tau` would be
the same sin as thresholding raw divergence.

**Recommendation:** ship a calibration utility that, given recorded `B_net` trajectories from
*control* (unbiased) runs, picks `tau_enter` at a target false-halt rate (e.g. the 99th percentile
of control `B_net`), and sets `tau_exit = c * tau_enter` (c ≈ 0.6–0.8) and `tau_warn` on the slow
scale similarly. Document the defaults as *starting points requiring calibration*, never as
validated. This mirrors the Phase 1 discipline: the library ships the method, the user owns the number.

### R7 — Bias-type routing seam

M4 routes conformity spirals to skeptics and parametric bias to MADERA. Classifying which is which is
genuinely uncertain and needs signals M3 only partly has.

**Recommendation:** M3 ships the *seam* — `route_intervention(context) -> "skeptics" | "madera" |
"none"` — with a simple, documented default heuristic (broad multi-node contamination → conformity →
skeptics; a single persistently-high node → parametric → madera) and a pluggable classifier. It is
explicitly a heuristic to be validated in M4; M3 does not pretend to have solved bias-type diagnosis.

### R8 — Router self-monitoring (gap #10)

The reroute/orchestrator is itself an agent that can amplify bias, yet nothing monitors it.

**Recommendation for v0.3:** treat the router as a normal node — it is probed and its `B_net`
contribution counts like any other (no exemption). The routing-**entropy** signal from the review
(low selection entropy → the orchestrator is concentrating on agreeing agents → feed the breaker
independently) is designed here but built when a real router exists in M4; v0.3 records the intent
and leaves the hook.

---

## 2. Work packages

### WP1 — Control types (`types.py`)
`BreakerState` (NORMAL/WARNING/INTERVENTION/RECOVERY/ESCALATED), `BreakerAction`
(PROCEED/HALT/REROUTE/ESCALATE), `BreakerDecision` (state, action, mixing ratio, `b_net`, reason,
frozen payload). Frozen, self-validating.
*Accept when:* invariants enforced (e.g. mixing ratio in [0,1]); a decision round-trips its fields.

### WP2 — Hysteresis breaker (`breaker.py`, pure)
`CircuitBreaker(tau_enter, tau_exit, tau_warn, kappa)` with `.check(fast, slow) -> BreakerDecision`.
Two-threshold dead-band + sigmoid mixing ratio. No framework, no numpy needed.
*Accept when:* a trajectory oscillating between `tau_exit` and `tau_enter` does **not** flip the
decision (the anti-thrash test); mixing ratio is monotonic in `B_net`; `tau_enter > tau_exit`
enforced.

### WP3 — Control state machine (`breaker.py`)
`ControlMachine` driving the R3 transitions with cool-down, `max_retries` → ESCALATED, and an
injectable intervention hook (default no-op). Pure and deterministic.
*Accept when:* the full lifecycle is unit-tested — enter on spike, cool-down gating, recovery on
recheck-OK, retry on recheck-bad, escalation after `max_retries`; no thrash near the boundary.

### WP4 — Payload freezing + routing seam (`breaker.py`, `intervention/router.py`)
`freeze(payload)` (immutable snapshot of the compromised state) and `route_intervention(context)`
with the default heuristic classifier.
*Accept when:* a frozen payload is recoverable and unmutated; routing returns the documented
strategy for representative broad-vs-concentrated contamination patterns.

### WP5 — LangGraph reference adapter (`integrations/langgraph/`)
`BiasState` TypedDict + reducers; a node-guard decorator (probe → stage in `unverified_output` →
gate promotion); prebuilt breaker/reroute nodes using `Command`. Edge-level interception via the
staging buffer.
*Accept when (langgraph installed):* a real graph run halts at the seeded node, the frozen payload is
recoverable from state, and the downstream node does **not** observe the biased output. Tests use
`importorskip`; default suite passes without langgraph.

### WP6 — Threshold calibration (`breaker.py` or `calibration.py`)
`calibrate_thresholds(control_trajectories, *, target_false_halt_rate)` → `(tau_enter, tau_exit,
tau_warn)`.
*Accept when:* on recorded control `B_net`, the chosen `tau_enter` yields ≤ the target false-halt
rate; documented as a starting point, not a validated constant.

### WP7 — Control study
Written report (like Phase 1/2): breaker behavior on DoT trajectories — trips once at breach, no
thrash near boundary, recovery after cool-down, escalation after repeated failure. Reproducible.
*Accept when:* the report shows a near-boundary oscillating `B_net` producing a single state entry
(vs. the many flips a binary threshold would produce), with reproducible figures.

---

## 3. Test strategy

Four layers as before. M3-specific must-haves:
- **Property:** the breaker never flips state while `B_net` stays within the `[tau_exit, tau_enter]`
  dead-band; state transitions are a function of (state, fast, slow, counters) only (deterministic).
- **Statistical/integration:** on DoT trajectories (reuse the M2 harness), a single Intervention
  entry at breach; escalation after `max_retries` no-op recoveries.
- **Adapter:** LangGraph tests behind `importorskip`; the default suite needs neither LangGraph nor
  network. All randomness via the seeded `rng` fixture.

## 4. Risks

| Risk | Signal | Response |
| --- | --- | --- |
| Thrashing near threshold | breaker flips every step | Two-threshold dead-band + anti-thrash property test (WP2). |
| Magic `tau` that doesn't transfer | breaker never/always fires on a new deployment | Ship calibration, not a constant (R6/WP6); document defaults as starting points. |
| Biased payload leaks downstream | next node sees pre-check output | Edge-level staging buffer; promotion gated by the breaker (R4/WP5). |
| Infinite correction loop | Recovery↔Intervention forever | `max_retries` → ESCALATED terminal state (R3/WP3). |
| LangGraph API churn / heavy dep | adapter breaks or bloats CI | Thin adapter; `importorskip`; default suite framework-free. |
| Over-claiming bias-type diagnosis | routing treated as solved | `route_intervention` is a documented heuristic seam, validated in M4 (R7). |
| Deterministic-halt vs continuous-mixing confusion | non-reproducible halts | Decision is discrete; mixing ratio is advisory only (R2). |

## 5. Definition of done

- [ ] Pure `CircuitBreaker` with two-threshold hysteresis + sigmoid mixing ratio; anti-thrash test passes
- [ ] `ControlMachine`: 4 states + ESCALATED, cool-down, `max_retries`, injectable hook; full-lifecycle tests
- [ ] Edge-level interception: payload frozen and *not* visible downstream (LangGraph test)
- [ ] LangGraph adapter (`BiasState`, guard, `Command` halt/reroute) behind `importorskip`; default suite framework-free
- [ ] `calibrate_thresholds` meets a target false-halt rate on control trajectories
- [ ] `route_intervention` seam with a documented default heuristic
- [ ] Control study published with reproducible figures (single entry vs. binary-threshold thrash)
- [ ] `mypy --strict` clean on 3.10–3.12; core has no LangGraph import; no M4/M5 work
