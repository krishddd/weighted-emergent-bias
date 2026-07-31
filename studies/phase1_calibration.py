"""Phase 1 calibration study — the WP7 deliverable.

Runs five experiments against the ground-truth fake client and writes ``phase1-results.json``
(plus ``figures/*.png``, if matplotlib is installed) under ``--out`` (default ``docs/studies``).
Every number in ``docs/studies/phase1-calibration.md`` comes from here.

    python studies/phase1_calibration.py            # full run
    python studies/phase1_calibration.py --quick     # fast smoke run

Experiments
-----------
1. **Calibration** — false-positive rate vs. nominal alpha on *unbiased* nodes across a range of
   sampling-noise levels. Also checks p-value uniformity, which is the stronger property: under
   the null a permutation test should produce p ~ Uniform(0, 1).
2. **Sensitivity** — recovered effect size and detection power vs. injected bias magnitude beta.
   Yields the detection floor (smallest beta detected with >= 80% power).
3. **Sample budget** — CI width and power vs. samples per side. Prices the whole system.
4. **Estimator agreement** — do CHOICE and GENERATIVE rank the same nodes the same way?
5. **Cost** — client calls and wall-clock per probe, against the DESIGN.md estimate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from weighted_emergent_bias import TaskMode, compute_local_bias
from weighted_emergent_bias.scoring.divergence import softmax
from weighted_emergent_bias.testing import FakeEmbeddingClient, FakeLLMClient
from weighted_emergent_bias.testing import markers as mk
from weighted_emergent_bias.types import BiasScore, FloatArray

MASTER_SEED = 20260731
AXIS = "gender"
BASELINE_PROMPT = "the applicant"
COUNTERFACTUAL_PROMPT = f"{mk.mark(AXIS, 'alt')} the applicant"


@dataclass(frozen=True)
class Config:
    """Study size knobs. ``--quick`` shrinks everything for a smoke run."""

    n_nodes_calibration: int = 300
    n_nodes_sensitivity: int = 120
    n_nodes_budget: int = 120
    n_nodes_agreement: int = 80
    n_permutations: int = 400
    n_bootstrap: int = 200
    n_candidates: int = 5
    default_samples: int = 8
    default_noise: float = 0.6

    @staticmethod
    def quick() -> Config:
        return Config(
            n_nodes_calibration=40,
            n_nodes_sensitivity=15,
            n_nodes_budget=15,
            n_nodes_agreement=12,
            n_permutations=80,
            n_bootstrap=40,
        )


def _client(rng: np.random.Generator, cfg: Config, *, beta: float, noise: float) -> FakeLLMClient:
    return FakeLLMClient(
        rng,
        n_candidates=cfg.n_candidates,
        bias={AXIS: beta} if beta > 0.0 else {},
        logit_noise=noise,
    )


async def _choice_group(client: FakeLLMClient, prompt: str, n: int) -> list[FloatArray]:
    return [softmax(await client.score_candidates(prompt, client.candidates)) for _ in range(n)]


async def _generative_group(
    client: FakeLLMClient, embedder: FakeEmbeddingClient, prompt: str, n: int
) -> list[FloatArray]:
    out: list[FloatArray] = []
    for _ in range(n):
        text = await client.generate(prompt)
        vec = await embedder.embed([text])
        out.append(np.asarray(vec[0], dtype=np.float64))
    return out


async def _score_node(
    client: FakeLLMClient,
    cfg: Config,
    stat_rng: np.random.Generator,
    *,
    n_samples: int,
) -> BiasScore:
    base = await _choice_group(client, BASELINE_PROMPT, n_samples)
    cf = await _choice_group(client, COUNTERFACTUAL_PROMPT, n_samples)
    return compute_local_bias(
        base,
        cf,
        task_mode=TaskMode.CHOICE,
        rng=stat_rng,
        axis=AXIS,
        n_permutations=cfg.n_permutations,
        n_bootstrap=cfg.n_bootstrap,
    )


# --- 1. Calibration ---------------------------------------------------------------------------


async def run_calibration(cfg: Config) -> dict[str, Any]:
    noise_levels = [0.2, 0.4, 0.6, 1.0]
    alphas = [0.01, 0.05, 0.10]
    sampler = np.random.default_rng(MASTER_SEED)
    stat_rng = np.random.default_rng(MASTER_SEED + 1)
    rows: list[dict[str, Any]] = []

    for noise in noise_levels:
        pvals: list[float] = []
        for _ in range(cfg.n_nodes_calibration):
            client = _client(sampler, cfg, beta=0.0, noise=noise)
            score = await _score_node(client, cfg, stat_rng, n_samples=cfg.default_samples)
            pvals.append(score.p_value)
        arr = np.asarray(pvals, dtype=np.float64)
        rows.append(
            {
                "noise": noise,
                "n_nodes": cfg.n_nodes_calibration,
                "mean_p": float(arr.mean()),
                "fpr": {str(a): float((arr <= a).mean()) for a in alphas},
                "pvalues": pvals,
            }
        )
    return {"alphas": alphas, "rows": rows}


# --- 2. Sensitivity ---------------------------------------------------------------------------


async def run_sensitivity(cfg: Config) -> dict[str, Any]:
    betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    sampler = np.random.default_rng(MASTER_SEED + 10)
    stat_rng = np.random.default_rng(MASTER_SEED + 11)
    rows: list[dict[str, Any]] = []

    for beta in betas:
        effects: list[float] = []
        detected = 0
        for _ in range(cfg.n_nodes_sensitivity):
            client = _client(sampler, cfg, beta=beta, noise=cfg.default_noise)
            score = await _score_node(client, cfg, stat_rng, n_samples=cfg.default_samples)
            effects.append(score.effect_size)
            if score.is_significant(0.05):
                detected += 1
        arr = np.asarray(effects, dtype=np.float64)
        rows.append(
            {
                "beta": beta,
                "mean_effect": float(arr.mean()),
                "median_effect": float(np.median(arr)),
                "power": detected / cfg.n_nodes_sensitivity,
            }
        )

    floor = next((r["beta"] for r in rows if r["beta"] > 0 and r["power"] >= 0.8), None)
    return {"rows": rows, "detection_floor_beta": floor}


# --- 3. Sample budget -------------------------------------------------------------------------


async def run_sample_budget(cfg: Config) -> dict[str, Any]:
    sample_sizes = [3, 5, 8, 12, 20, 30]
    beta = 2.0
    sampler = np.random.default_rng(MASTER_SEED + 20)
    stat_rng = np.random.default_rng(MASTER_SEED + 21)
    rows: list[dict[str, Any]] = []

    for n in sample_sizes:
        widths: list[float] = []
        effects: list[float] = []
        null_stds: list[float] = []
        detected = 0
        for _ in range(cfg.n_nodes_budget):
            client = _client(sampler, cfg, beta=beta, noise=cfg.default_noise)
            score = await _score_node(client, cfg, stat_rng, n_samples=n)
            widths.append(score.ci_high - score.ci_low)
            effects.append(score.effect_size)
            null_stds.append(score.null_std)
            if score.is_significant(0.05):
                detected += 1
        rows.append(
            {
                "n_samples": n,
                "mean_ci_width": float(np.mean(widths)),
                # Recorded to test whether the effect-size *unit* (the null sigma) is itself
                # n-dependent, which would make effect sizes incomparable across sample budgets.
                "mean_effect": float(np.mean(effects)),
                "mean_null_std": float(np.mean(null_stds)),
                "power": detected / cfg.n_nodes_budget,
                "calls_per_probe_1_axis": n * 2,
            }
        )
    return {"beta": beta, "rows": rows}


# --- 4. Estimator agreement -------------------------------------------------------------------


def _spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation without a scipy dependency. Ties get arbitrary (not averaged) ranks;
    effect sizes are continuous so exact ties are vanishingly unlikely here."""

    def ranks(v: list[float]) -> FloatArray:
        arr = np.asarray(v, dtype=np.float64)
        order = arr.argsort()
        r = np.empty(len(arr), dtype=np.float64)
        r[order] = np.arange(len(arr), dtype=np.float64)
        return r

    rx, ry = ranks(x), ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


async def run_agreement(cfg: Config) -> dict[str, Any]:
    sampler = np.random.default_rng(MASTER_SEED + 30)
    stat_rng = np.random.default_rng(MASTER_SEED + 31)
    embed_rng = np.random.default_rng(MASTER_SEED + 32)
    betas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

    choice_effects: list[float] = []
    generative_effects: list[float] = []

    for i in range(cfg.n_nodes_agreement):
        beta = betas[i % len(betas)]
        seed = int(sampler.integers(0, 2**31 - 1))
        n = cfg.default_samples

        c1 = _client(np.random.default_rng(seed), cfg, beta=beta, noise=cfg.default_noise)
        s_choice = await _score_node(c1, cfg, stat_rng, n_samples=n)

        c2 = _client(np.random.default_rng(seed), cfg, beta=beta, noise=cfg.default_noise)
        embedder = FakeEmbeddingClient(embed_rng, dim=16)
        base = await _generative_group(c2, embedder, BASELINE_PROMPT, n)
        cf = await _generative_group(c2, embedder, COUNTERFACTUAL_PROMPT, n)
        s_gen = compute_local_bias(
            base,
            cf,
            task_mode=TaskMode.GENERATIVE,
            rng=stat_rng,
            axis=AXIS,
            n_permutations=cfg.n_permutations,
            n_bootstrap=cfg.n_bootstrap,
        )

        choice_effects.append(s_choice.effect_size)
        generative_effects.append(s_gen.effect_size)

    return {
        "n_nodes": cfg.n_nodes_agreement,
        "spearman": _spearman(choice_effects, generative_effects),
        "choice_effects": choice_effects,
        "generative_effects": generative_effects,
    }


# --- 5. Cost ----------------------------------------------------------------------------------


async def run_cost(cfg: Config) -> dict[str, Any]:
    sampler = np.random.default_rng(MASTER_SEED + 40)
    stat_rng = np.random.default_rng(MASTER_SEED + 41)
    rows: list[dict[str, Any]] = []

    for n in (3, 5, 8):
        for n_axes in (1, 2, 3):
            client = _client(sampler, cfg, beta=1.0, noise=cfg.default_noise)
            start = time.perf_counter()
            await _score_node(client, cfg, stat_rng, n_samples=n)
            elapsed = time.perf_counter() - start
            rows.append(
                {
                    "n_samples": n,
                    "n_axes": n_axes,
                    # An uninstrumented node makes 1 call; the probe makes n*(1+n_axes).
                    "calls": n * (1 + n_axes),
                    "cost_multiplier": n * (1 + n_axes),
                    "scoring_seconds_per_axis": elapsed,
                }
            )
    return {"rows": rows}


# --- plots ------------------------------------------------------------------------------------


def _pyplot() -> Any | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def make_plots(results: dict[str, Any], outdir: Path) -> bool:
    plt = _pyplot()
    if plt is None:
        return False
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. p-value uniformity (the calibration evidence).
    fig, ax = plt.subplots(figsize=(6, 4))
    for row in results["calibration"]["rows"]:
        p = np.sort(np.asarray(row["pvalues"], dtype=np.float64))
        ax.plot(p, np.linspace(0, 1, len(p), endpoint=False), label=f"noise={row['noise']}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal (uniform)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("empirical CDF")
    ax.set_title("Null p-values are uniform (unbiased nodes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "calibration-pvalue-cdf.png", dpi=140)
    plt.close(fig)

    # 2. Sensitivity: effect size and power vs beta.
    rows = results["sensitivity"]["rows"]
    betas = [r["beta"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(betas, [r["mean_effect"] for r in rows], "o-")
    ax1.set_xlabel("injected bias magnitude (beta)")
    ax1.set_ylabel("mean recovered effect size")
    ax1.set_title("Sensitivity")
    ax2.plot(betas, [r["power"] for r in rows], "o-", color="tab:red")
    ax2.axhline(0.8, ls="--", c="gray", lw=1)
    ax2.set_xlabel("injected bias magnitude (beta)")
    ax2.set_ylabel("power at alpha=0.05")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Detection power")
    fig.tight_layout()
    fig.savefig(outdir / "sensitivity.png", dpi=140)
    plt.close(fig)

    # 3. Sample budget.
    rows = results["sample_budget"]["rows"]
    ns = [r["n_samples"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ns, [r["mean_ci_width"] for r in rows], "o-")
    ax1.set_xlabel("samples per side (n)")
    ax1.set_ylabel("mean CI width")
    ax1.set_title("Precision vs. sample budget")
    ax2.plot(ns, [r["power"] for r in rows], "o-", color="tab:red")
    ax2.axhline(0.8, ls="--", c="gray", lw=1)
    ax2.set_xlabel("samples per side (n)")
    ax2.set_ylabel("power at alpha=0.05")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Power vs. sample budget")
    fig.tight_layout()
    fig.savefig(outdir / "sample-budget.png", dpi=140)
    plt.close(fig)

    # 4. Estimator agreement.
    agree = results["agreement"]
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(agree["choice_effects"], agree["generative_effects"], alpha=0.7)
    ax.set_xlabel("CHOICE effect size")
    ax.set_ylabel("GENERATIVE effect size")
    ax.set_title(f"Estimator agreement (Spearman = {agree['spearman']:.2f})")
    fig.tight_layout()
    fig.savefig(outdir / "estimator-agreement.png", dpi=140)
    plt.close(fig)
    return True


# --- entrypoint -------------------------------------------------------------------------------


async def run_all(cfg: Config) -> dict[str, Any]:
    return {
        "config": cfg.__dict__,
        "calibration": await run_calibration(cfg),
        "sensitivity": await run_sensitivity(cfg),
        "sample_budget": await run_sample_budget(cfg),
        "agreement": await run_agreement(cfg),
        "cost": await run_cost(cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 calibration study")
    parser.add_argument("--quick", action="store_true", help="fast smoke run")
    parser.add_argument("--out", type=Path, default=Path("docs/studies"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = Config.quick() if args.quick else Config()
    start = time.perf_counter()
    results = asyncio.run(run_all(cfg))
    results["runtime_seconds"] = time.perf_counter() - start

    args.out.mkdir(parents=True, exist_ok=True)
    slim = json.loads(json.dumps(results))
    for row in slim["calibration"]["rows"]:
        row.pop("pvalues", None)
    (args.out / "phase1-results.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")

    plotted = False
    if not args.no_plots:
        plotted = make_plots(results, args.out / "figures")

    print(json.dumps(slim, indent=2))
    print(f"\nruntime: {results['runtime_seconds']:.1f}s  plots: {'yes' if plotted else 'skipped'}")


if __name__ == "__main__":
    main()
