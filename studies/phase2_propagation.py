"""Phase 2 propagation study -- the WP7 deliverable.

Demonstrates the point of topological weighting: the *same* bias does far more damage seeded at a
central node than at a leaf. Runs against the DoT harness (real M1+M2 stack) and emits a JSON
results file plus figures (if matplotlib is installed). Every number in
``docs/studies/phase2-propagation.md`` comes from here.

    python studies/phase2_propagation.py            # full run
    python studies/phase2_propagation.py --quick     # fast smoke run

Experiments
-----------
1. **Central vs. leaf seed** -- identical bias magnitude seeded at a high-blast-radius hub vs. a
   terminal leaf; compare the resulting B_net trajectories.
2. **B_net vs. seed centrality** -- seed each node in turn; B_net should track the seed Katz weight.
3. **Fast vs. slow** -- the two accumulator scales on the central-seed run (fast leads).
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

from weighted_emergent_bias import AgentDAG, dependency_weights
from weighted_emergent_bias.testing import DoTResult, simulate_dot

MASTER_SEED = 20260731


@dataclass(frozen=True)
class Config:
    beta: float = 3.0
    n_samples: int = 8
    n_permutations: int = 300

    @staticmethod
    def quick() -> Config:
        return Config(n_samples=6, n_permutations=80)


def hub_dag() -> tuple[AgentDAG, str, str]:
    """root -> hub -> {w1..w4}. Returns (dag, central_node, leaf_node)."""
    edges = [("root", "hub"), ("hub", "w1"), ("hub", "w2"), ("hub", "w3"), ("hub", "w4")]
    dag = AgentDAG(edges, nodes=["root", "hub", "w1", "w2", "w3", "w4"])
    return dag, "hub", "w1"


async def _run(dag: AgentDAG, seed: str, cfg: Config, rng_seed: int) -> DoTResult:
    return await simulate_dot(
        dag,
        seed,
        rng=np.random.default_rng(rng_seed),
        beta=cfg.beta,
        n_samples=cfg.n_samples,
        n_permutations=cfg.n_permutations,
    )


async def run_central_vs_leaf(cfg: Config) -> dict[str, Any]:
    dag, central, leaf = hub_dag()
    weights = dependency_weights(dag).weights
    central_res = await _run(dag, central, cfg, MASTER_SEED)
    leaf_res = await _run(dag, leaf, cfg, MASTER_SEED)
    return {
        "central_node": central,
        "leaf_node": leaf,
        "central_weight": weights[central],
        "leaf_weight": weights[leaf],
        "central_fast": central_res.fast,
        "leaf_fast": leaf_res.fast,
        "central_final": central_res.fast[-1],
        "leaf_final": leaf_res.fast[-1],
        "ratio": central_res.fast[-1] / leaf_res.fast[-1] if leaf_res.fast[-1] > 0 else None,
    }


async def run_bnet_vs_centrality(cfg: Config) -> dict[str, Any]:
    dag, _, _ = hub_dag()
    weights = dependency_weights(dag).weights
    rows: list[dict[str, Any]] = []
    for i, seed in enumerate(("w1", "hub", "root")):
        res = await _run(dag, seed, cfg, MASTER_SEED + i)
        contaminated = sum(1 for v in res.node_bias.values() if v > 0.0)
        rows.append(
            {
                "seed": seed,
                "katz_weight": weights[seed],
                "downstream_reach": contaminated - 1,  # nodes contaminated besides the seed
                "bnet_final": res.fast[-1],
            }
        )
    return {"rows": rows}


async def run_fast_vs_slow(cfg: Config) -> dict[str, Any]:
    dag, central, _ = hub_dag()
    res = await _run(dag, central, cfg, MASTER_SEED)
    return {"fast": res.fast, "slow": res.slow}


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

    cvl = results["central_vs_leaf"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cvl["central_fast"], "o-", label=f"central seed (w={cvl['central_weight']:.2f})")
    ax.plot(cvl["leaf_fast"], "s--", label=f"leaf seed (w={cvl['leaf_weight']:.2f})")
    ax.set_xlabel("superstep")
    ax.set_ylabel("B_net (fast)")
    ax.set_title("Same bias, central vs. leaf seed")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "central-vs-leaf.png", dpi=140)
    plt.close(fig)

    rows = results["bnet_vs_centrality"]["rows"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([r["downstream_reach"] for r in rows], [r["bnet_final"] for r in rows])
    for r in rows:
        ax.annotate(r["seed"], (r["downstream_reach"], r["bnet_final"]))
    ax.set_xlabel("seed downstream reach (nodes contaminated)")
    ax.set_ylabel("final B_net (fast)")
    ax.set_title("B_net tracks seed blast radius")
    fig.tight_layout()
    fig.savefig(outdir / "bnet-vs-centrality.png", dpi=140)
    plt.close(fig)

    fvs = results["fast_vs_slow"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fvs["fast"], "o-", label="fast (spikes)")
    ax.plot(fvs["slow"], "s--", label="slow (drift)")
    ax.set_xlabel("superstep")
    ax.set_ylabel("B_net")
    ax.set_title("Two timescales, central-seed run")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fast-vs-slow.png", dpi=140)
    plt.close(fig)
    return True


async def run_all(cfg: Config) -> dict[str, Any]:
    return {
        "config": cfg.__dict__,
        "central_vs_leaf": await run_central_vs_leaf(cfg),
        "bnet_vs_centrality": await run_bnet_vs_centrality(cfg),
        "fast_vs_slow": await run_fast_vs_slow(cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 propagation study")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("docs/studies"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    cfg = Config.quick() if args.quick else Config()
    start = time.perf_counter()
    results = asyncio.run(run_all(cfg))
    results["runtime_seconds"] = time.perf_counter() - start

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "phase2-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    plotted = False if args.no_plots else make_plots(results, args.out / "figures")

    print(json.dumps(results, indent=2))
    print(f"\nruntime: {results['runtime_seconds']:.1f}s  plots: {'yes' if plotted else 'skipped'}")


if __name__ == "__main__":
    main()
