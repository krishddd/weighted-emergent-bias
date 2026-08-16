# Invariants

Rules that must not be broken. **Every one was learned the hard way** — most have a regression
test pinning them, and the ones that don't are called out below. If you are about to change code
that touches any of these, read the reasoning first.

## Detection (M1)

### Never accumulate `effect_size`

`BiasScore.effect_size` is a *detection* statistic that inflates with sample count. The Phase 1
study measured the same true bias at **1.9σ with n=3 and 22.9σ with n=30**, because `null_std → 0`
as n grows. Weighting on it makes the network signal depend on each node's sample budget rather
than its bias.

Accumulate **significance-gated excess divergence** instead:

```
b_i = max(0, raw_divergence - null_mean)   if significant at alpha, else 0
```

The p-value carries "is this real?"; the excess divergence carries an n-stable "how big?".

### The noise floor is non-negotiable

An LLM sampled twice on identical input diverges from itself. Thresholding raw divergence is
thresholding sampling temperature. Every score is reported net of an empirically estimated
per-node null.

### True JSD, not symmetrized KL

Use the mixture form `JSD = ½·KL(P‖M) + ½·KL(Q‖M)` with `M = ½(P+Q)`. It is bounded in `[0,1]`
and finite on disjoint supports. The symmetrized-KL ("Jeffreys") form that a reviewer proposed is
**unbounded** and blows up to ∞ on exact-zero probabilities.

## Propagation (M2)

### Blast radius counts walks *leaving* a node

Dependency weight is the row-sums of `(I − α·A)⁻¹` using `A` **directly** — walks a node can
reach. This is Katz on the *reversed* graph relative to `networkx.katz_centrality`, which measures
influence *arriving* along incoming edges.

> ⚠️ **Do not "fix" this by adding a transpose.** Transposing ranks terminal leaves as maximally
> critical — exactly inverted. A stale docstring in `dag.py` told a maintainer to do precisely
> that and was corrected in v0.6.0. The A→B→C orientation test guards it.

### Priors are injected, never self-updated

An error-history prior may be supplied externally. M2 **never** updates a prior from its own
detections — a self-reinforcing error history is the minority-suppression loop the external
review warned about.

## Control (M3)

### Recovery is re-measured, never assumed

Returning to Normal requires the cool-down to elapse **and** `B_net` to independently fall below
`tau_exit`. A repair that claims success does not get to declare itself recovered.

### `max_retries` is per-incident

A clean recovery closes the incident and resets the attempt counter. Before v0.6.0 it did not
reset, silently degrading into a lifetime cap that escalated runs *because* they had self-healed
successfully.

### No magic thresholds

`tau` comes from `calibrate_thresholds` on control (unbiased) runs, targeting a false-halt rate
directly. A hardcoded `tau` is the M3 analogue of thresholding raw divergence. Calibration also
refuses targets finer than `1/n` — a quantile cannot resolve a rate the data cannot support.

## Intervention (M4)

### No prune-on-dissent

Trust is computed per verdict **from its own content**, with no term referencing the other votes.
Pruning fires on *overconfidence* (high certainty, no evidence) — **never** on disagreeing with an
emerging consensus. This is the guarantee that stops the panel silencing a correct minority.

### Trust must be monotonic in evidence

An unevidenced verdict must never outweigh an evidence-backed one, and must lose weight as it
grows more assertive. Before v0.6.0, self-orientation `S` took the raw confidence, so a
`confidence=0.0` no-evidence verdict hit the `_S_FLOOR` clamp and received **~2300× the weight** of
an evidenced verdict — the clamp added to prevent `S → 0` divergence was the thing causing it.

### Ties break toward caution

`REJECT > REVISE > STANDS`. Falling back on enum declaration order silently favoured STANDS —
fail-open on a safety component.

## Evidence (M5)

### The trail is genuinely append-only

`detail` is deep-copied behind a `MappingProxyType`. A shallow copy left recorded events
rewritable both directly and through nested objects the caller still held. An evidence log that
can be quietly edited is not an audit trail.

## Cross-cutting

### Determinism

Seeded `rng` (required, never defaulted), monotonic sequence index rather than wall-clock,
insertion-ordered nodes. Python randomizes `set` iteration of strings across processes, so a
set-backed graph would silently produce different centrality numbers run to run.

Optimizations must preserve the RNG stream. The v0.6.0 vectorization of the permutation and
bootstrap loops is **bit-for-bit identical** on the same seed — verified to 12 decimals against
the previous implementation.

### Honesty

- No validated-performance claims on real models.
- Prior-work numbers cited **as prior work only**.
- Scope stays narrow: demographic counterfactual bias — not factual error, not consensus deviation.
- A claim is made only when a test or a study in this repository backs it.
