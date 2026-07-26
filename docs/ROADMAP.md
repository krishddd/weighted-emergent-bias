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
| **M2** | **Propagation model** | DAG protocol, transposed Katz centrality, dependency weights, bias-corrected EWMA, superstep reduction | v0.2 | M1 |
| **M3** | **Control plane** | `CircuitBreaker`, payload freezing, bias-type classification, `BiasState`, LangGraph `Command()` halt/reroute | v0.3 | M1, M2 |
| **M4** | **Intervention** | Skeptic panel + `Send()` fan-out, trust-graph pruning, anonymization, MADERA diagnose→retrieve→rewrite | v0.4 | M3 |
| **M5** | **Evidence** | Append-only causal trail, SARIF 2.1.0 export, HTML/JSON run reports | v0.5 | M3, M4 |

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

- **M1** — planned in detail: [plans/PHASE-1.md](plans/PHASE-1.md). Not started.
- **M2–M5** — sketched in [DESIGN.md](DESIGN.md). Planned in detail at the start of each phase,
  not before; earlier phases will change what later ones need.
