# weighted-emergent-bias

**A runtime circuit-breaker for the Degeneration-of-Thought (DoT) problem in multi-agent LLM systems.**

> **Status: pre-alpha, design phase.** No implementation yet. See [docs/DESIGN.md](docs/DESIGN.md)
> for the full plan. This project makes no validated performance claims — see
> [Prior work](#prior-work-and-what-this-project-has-not-yet-shown).

---

In a multi-agent LLM pipeline, one agent's mildly stereotyped output becomes the next
agent's ground truth. Downstream agents do not re-litigate the premise they were handed —
they build on it, and the bias compounds through the graph until every stage has homogenized
around the same skewed register. This failure mode is called Degeneration-of-Thought (DoT),
and single-model alignment does not catch it: the bias is not a property of any one model's
weights, it is a property of how the agents are wired together. `weighted-emergent-bias` is
a runtime circuit-breaker for that failure. It probes each node with a demographically
perturbed counterfactual to get a local bias score, weights that score by the node's
downstream blast radius via graph centrality, accumulates the weighted scores into a
network-level moving average as the graph executes, and halts the run deterministically when
that average crosses a threshold — freezing the compromised payload and rerouting control to
a mitigation subgraph instead of letting the contaminated state propagate.

## How it works

| Stage | Mechanism |
| --- | --- |
| **Detect** | LOOC (Leave-One-Out Calibration) probes each node with a demographically perturbed counterfactual and measures the divergence between the two outputs — net of the node's own sampling noise. |
| **Weight** | Katz centrality over the transposed agent DAG gives each node a dependency weight `w_i`. A biased terminal leaf is harmless; a biased central router contaminates everything downstream. |
| **Accumulate** | A bias-corrected EWMA tracks `B_net` across execution supersteps, catching slow drift that no single-point evaluation would see. |
| **Break** | When `B_net` crosses `tau`, execution halts deterministically and the payload is frozen. |
| **Intervene** | Conformity spirals route to parallel Skeptic Agents under a trust graph that prunes overconfident participants. Parametric bias routes to a MADERA-style diagnose → retrieve → rewrite repair pipeline. |
| **Audit** | Every probe, divergence, weight, and routing decision lands in an append-only causal trail, exportable as SARIF or HTML. |

## Design principles

- **Framework-agnostic core.** The scoring, topology, accumulation, and breaker logic depend
  only on numpy and networkx. LangGraph is the reference adapter, shipped as an optional
  extra — not a hard dependency.
- **Bring your own model.** No bundled LLM SDK. You supply a client callable; the library
  supplies the protocols, the math, and reference agent implementations.
- **Noise floor, always.** An LLM sampled twice on identical input diverges from itself.
  Every bias score is reported net of an empirically estimated per-node null distribution, as
  a standardized effect size with a confidence interval. Thresholding raw divergence would
  just be thresholding on temperature.
- **No silent coverage gaps.** When probing is sampled or skipped for cost reasons, the
  sampling rate is recorded in the audit trail. A partial scan never reports as a full one.

## Prior work, and what this project has *not* yet shown

This design implements and adapts mechanisms from published research. Those papers' results
are **theirs, measured on their setups** — they are not evidence that this implementation
works, and this project will not present them as such.

- **LOOC + Synthetic Data Calibration** — *Beyond Generation: Leveraging LLM Creativity to
  Overcome Label Bias in Classification* (ACL 2025 Findings). Reports a 57.5% label-bias
  reduction **in a single-model classification setting**, which is a different metric on a
  different unit of analysis than multi-agent emergent bias.
- **MADERA** — *Towards Fairer AI: Multi-Agent Debiasing of LLMs With Online Evidence
  Retrieval* (AAAI-SS). Reports accuracy and bias improvements on BBQ-Hard for its own
  pipeline; the reimplementation here is unvalidated.
- **MALIBU** — *Multi-Agent LLM Implicit Bias Uncovered* ([arXiv:2507.01019](https://arxiv.org/abs/2507.01019)).
  Cited as motivation: it documents that collaborative agent discussion amplifies implicit
  bias. This library has not been evaluated on it.
- **CortexDebate** — source of the trust-graph pruning approach. The underlying
  `T = (C+R+I)/S` trust formula is a consulting heuristic, treated here as a heuristic that
  must earn its place against uniform aggregation in an ablation.

Benchmark reproduction is deliberately **not** on the v0.x roadmap. Until it happens, the
only claim made here is that the mechanics are implemented and demonstrable on a synthetic
DoT simulation harness.

## Roadmap

- **v0.1** — single-node LOOC scoring with noise-floor separation, no graph dependency
- **v0.2** — Katz topology weighting, EWMA accumulation, synthetic DoT simulation harness
- **v0.3** — LangGraph `BiasState`, node instrumentation, `Command()` halt and reroute
- **v0.4** — Skeptic panel with trust-graph pruning, MADERA repair pipeline
- **v0.5** — causal audit trail, SARIF 2.1.0 export, HTML/JSON reporting

## License

MIT
