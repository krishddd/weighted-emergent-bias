# FAQ

## Does this detect factual errors?

**No.** A node can be perfectly counterfactually invariant and still completely wrong. Factual
correctness needs an external oracle, which is offered only as an optional injectable `Verifier`
hook — deliberately kept out of the core claim.

Keeping bias and error as separate signals is a scope decision, not an oversight. Blending them
would mean neither number means anything specific.

## Does it flag an agent for disagreeing with the others?

**No, and it must never.** The detector is a *within-node* test: a node's output is compared to
its **own** output on a perturbed input, anchored to its **own** sampling-noise floor. The other
agents' consensus is never an input.

A lone correct dissenter against a biased majority is exactly the thing this must not punish. The
same rule holds in the skeptic panel: pruning fires on overconfidence (high certainty, no
evidence), never on divergence from consensus.

## Why not just threshold the divergence?

Because an LLM sampled twice on identical input diverges from itself. Raw baseline-vs-counterfactual
divergence is always > 0 even for a completely unbiased node, and it scales with temperature.
Thresholding it means thresholding sampling temperature.

Every score is therefore reported net of an empirically estimated per-node null.

## Why is `rng` a required argument?

So results are reproducible. A defaulted RNG would make the numbers unverifiable, and the entire
point of the noise-floor apparatus is to produce a number someone else can check.

## Why does no default list of demographic axes ship?

Because shipping a fixed list encodes its own bias — whose categories, in whose taxonomy, for
whose deployment? Axes are a required explicit argument. Illustrative sets are documented as
examples, never baked into code.

## Has this been benchmarked?

**No, and the README says so.** Benchmark reproduction (MALIBU / BBQ-Hard) is deliberately
unscheduled — it is a large, API-cost-heavy workstream needing datasets and budget.

The papers whose mechanisms are adapted here report their own results on their own setups. Those
are cited as prior work and are **not** evidence that this implementation works. What is
demonstrated is that the mechanics run and behave correctly on a synthetic harness where the
ground-truth bias is known by construction.

A real-model validation harness exists and is one command — but it has not been run, and a single
run would be evidence, not validation.

## Does it work with cyclic graphs?

Yes. LangGraph permits cycles, at which point Katz's nilpotency guarantee disappears and the
`α < 1/ρ(A)` stability constraint returns. Cycles are detected and `α` is clamped to half that
bound, with the clamp **recorded** in the result — never a silently divergent number.

## Does it need an API key or network access?

Not for the core, and not for the test suite. No LLM SDK is bundled; you supply a client callable.
Optional integrations sit behind extras and `importorskip`.

## What overhead does it add?

Probing costs extra model calls — that is inherent to counterfactual testing, and the honest answer
is that it depends on your sampling rate. When probing is sampled or skipped for cost, the sampling
rate is recorded in the audit trail, so a partial scan never reports as a full one.

## Is it production-ready?

It is **alpha**. The mechanics are complete, typed strictly, and tested, but the scope is narrow by
design and there are no validated-performance claims on real models. Treat it as a well-engineered
implementation of a specific published idea, not a turnkey guarantee.
