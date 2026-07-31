# ROADMAP — module segregation

The pipeline splits into **five modules**. Each is independently useful, independently
testable, and depends only on the modules before it. Nothing later can be trusted if
something earlier is wrong, so the order is not negotiable.

```
  M1 Detection        "is this one node biased, and how sure are we?"
       |                 pure math + LLM probes. no graph, no framework.
       v
  M2 Propagation      "how much does that node's bias matter to the whole run?"
       |                 graph centrality + accumulation over time.
       v
  M3 Control          "stop the run when it matters, deterministically."
       |                 threshold, freeze, reroute. LangGraph adapter lands here.
       v
  M4 Intervention     "repair the biased state and resume."
       |                 skeptic debate + trust pruning, or MADERA reasoning edits.
       v
  M5 Evidence         "prove what happened, in a form someone can audit."
                         causal trail, SARIF, HTML/JSON reports.
```

| # | Module | Owns | Ships as | Depends on |
| --- | --- | --- | --- | --- |
| **M1** | **Detection core** | Counterfactual perturbation, divergence estimation, per-node noise floor, significance testing, SDC priors | v0.1 | nothing (numpy only) |
| **M2** | **Propagation model** | DAG protocol, transposed Katz centrality, Bayesian error-history score, composite dependency weights, multi-scale (fast+slow) bias-corrected EWMA, superstep reduction | v0.2 | M1 |
| **M3** | **Control plane** | Two-threshold hysteresis breaker, edge-level interception, Normal→Warning→Intervention→Recovery state machine, router self-monitoring, payload freezing, `BiasState`, LangGraph `Command()` halt/reroute | v0.3 | M1, M2 |
| **M4** | **Intervention** | Skeptic panel (≥2, diverse provenance) + `Send()` fan-out, trust-graph pruning with anonymization + no-prune-on-dissent guardrail, MADERA diagnose→retrieve→rewrite | v0.4 | M3 |
| **M5** | **Evidence** | Append-only causal trail, extended SARIF 2.1.0 export (trigger reason, routing entropy), HTML/JSON reports, ablation evaluation protocol + error budget | v0.5 | M3, M4 |

Mechanism additions in M2–M5 are banked from the 2026-07 external review — see
[DESIGN.md §8](DESIGN.md) and [reviews/2026-07-external-review-response.md](reviews/2026-07-external-review-response.md).

## Why this order, and why M1 gets disproportionate effort

M2 multiplies M1's score by a weight. M3 compares it to a threshold. M4 fires when that
threshold trips. M5 reports on all of it. **Every one of those is a transformation of a
number M1 produces.** If `B_i` is not a calibrated, meaningful quantity, then M2 is
weighting noise, M3 is thresholding noise, M4 is spending real inference budget repairing
outputs that were never biased, and M5 is generating audit reports that certify nothing.

There is no way to recover from a bad M1 further down the pipeline, and a plausible-looking
`B_i` that is actually measuring sampling temperature will not announce itself — the system
will run, produce numbers, trip occasionally, and be entirely wrong. So M1 is not "the easy
first bit." It is the only module with a genuine research question in it, and it gets built
slowly, with an empirical validation study attached.

M2 through M5 are comparatively well-understood engineering: known algorithms (Katz, EWMA)
and known framework primitives (`Command`, `Send`, `TypedDict` reducers). They are work, but
they are not risk.

## Current status

- **M1** — **shipped as v0.1.0.** Complete and characterized: [plans/PHASE-1.md](plans/PHASE-1.md),
  [calibration study](studies/phase1-calibration.md). WP6 (SDC) deferred as optional.
- **M2** — **shipped as v0.2.0.** Complete and characterized: [plans/PHASE-2.md](plans/PHASE-2.md),
  [propagation study](studies/phase2-propagation.md). DAG, transposed Katz, n-stable magnitude,
  composite weight, multi-scale accumulator, DoT harness.
- **M3** — next up. Sketched in [DESIGN.md §8](DESIGN.md): two-threshold hysteresis breaker,
  edge-level interception, Normal→Warning→Intervention→Recovery state machine, LangGraph adapter.
  Detailed plan to be written at phase start.
- **M4–M5** — sketched in [DESIGN.md](DESIGN.md). Planned in detail at the start of each phase,
  not before; earlier phases will change what later ones need.
