# Example perturbation axes

These are **illustrative examples, not defaults.** The library ships no built-in axis list on
purpose: a hardcoded set of demographic categories is itself an editorial position about which
identities count and how they are named, and that is not a decision a fairness tool should make
silently on a user's behalf. `perturb(payload, axes)` requires you to pass axes explicitly;
calling it with none is an error.

Copy, edit, and — most importantly — **scrutinize** these before using them. The right axes,
terms, and proxies depend entirely on your domain, your population, and what a counterfactual
*should* hold fixed in your task. Treat every list below as a starting point to be reviewed by
someone who understands the context, not a canonical taxonomy.

## How an axis works

An `AxisSpec` is a named set of `Substitution`s. Each substitution is a whole-word,
case-insensitive replacement (with case preserved: `He → She`, `HIS → HER`). A substitution is
tagged **explicit** (it edits a stated attribute) or **proxy** (it edits a feature merely
*correlated* with the attribute). The two kinds produce separate perturbations, because "the
model reacts to a stated attribute" and "the model reacts to a proxy for it" are different
findings — the second is usually the more insidious one.

```python
from weighted_emergent_bias import AxisSpec, Substitution, PerturbationKind, perturb

gender = AxisSpec(
    name="gender",
    substitutions=(
        Substitution("he", "she"),
        Substitution("him", "her"),
        Substitution("his", "her"),
        Substitution("man", "woman"),
        Substitution("mr", "ms"),
    ),
)

perts = perturb({"letter": "He is a hardworking man; his record is strong."}, [gender])
# perts[0].perturbed == {"letter": "She is a hardworking woman; her record is strong."}
```

## Illustrative sets

### Gender (explicit)

Pronoun and honorific swaps. Note that a single directional list (`he → she`) probes one
direction only; for a symmetric study, run the reverse list as a second axis and compare.

| pattern | replacement |
| --- | --- |
| `he` | `she` |
| `him` | `her` |
| `his` | `her` |
| `man` | `woman` |
| `men` | `women` |
| `mr` | `ms` |

### Name-as-proxy (proxy)

Names correlate with gender and ethnicity without stating either. This is a **proxy** axis —
the demographic attribute is never named, only implied. Use with care and with names chosen
for and validated against *your* context; the pairs below are placeholders, not endorsements.

```python
name_proxy = AxisSpec(
    name="name_ethnicity_proxy",
    substitutions=(
        Substitution("Emily", "Lakisha", kind=PerturbationKind.PROXY),
        Substitution("Greg", "Jamal", kind=PerturbationKind.PROXY),
    ),
)
```

This mirrors classic correspondence-testing methodology (identical résumés, only the name
changes). If bias fires on this axis but not on an explicit one, the model is keying off the
proxy — exactly the case post-hoc filters miss.

### Geography-as-proxy (proxy)

Zip codes, neighborhoods, and institutions correlate with race and class. Substitute one
plausible-but-demographically-different locale for another; keep everything else fixed.

```python
geo_proxy = AxisSpec(
    name="geography_proxy",
    substitutions=(
        Substitution("90210", "10453", kind=PerturbationKind.PROXY),
    ),
)
```

## Guidance for building your own

- **Hold everything else fixed.** A good counterfactual changes *only* the attribute under
  test. If your swap also changes tense, formality, or meaning, the divergence you measure is
  confounded.
- **Prefer whole, unambiguous tokens.** Matching is whole-word; short patterns (`he`) are safe
  because of word boundaries, but check they do not collide with domain terms.
- **Separate explicit from proxy deliberately.** Tag proxies as `PerturbationKind.PROXY` so the
  audit trail can tell you *which* channel the bias came through.
- **Review with someone who knows the domain.** These are counterfactuals about people. The
  cost of a careless axis is a fairness tool that encodes the very assumptions it claims to check.
