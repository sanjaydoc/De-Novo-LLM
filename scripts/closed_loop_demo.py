#!/usr/bin/env python
"""End-to-end closed-loop optimization demo (runs offline, no GPU/network).

Simulates a de novo campaign: a candidate pool is scored by a noisy, budgeted
oracle; a Bayesian-optimization loop (deep-ensemble surrogate + Expected
Improvement) chooses which candidates to "measure" each round. We compare it
against random screening and plot best-found vs. number of experiments,
averaged over several seeds.

    python scripts/closed_loop_demo.py --rounds 12 --batch 5 --seeds 5

Outputs: docs/assets/closed_loop.png, docs/results/closed_loop.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from denovo.closedloop import ActiveLearningLoop, DeepEnsemble, SyntheticOracle  # noqa: E402
from denovo.closedloop.featurize import identity_featurizer  # noqa: E402
from denovo.closedloop.oracle import negative_ackley  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_method(acquisition, *, pool, dim, rounds, batch, init, noise, seed):
    oracle = SyntheticOracle(negative_ackley, noise_std=noise, seed=seed)
    loop = ActiveLearningLoop(
        pool,
        identity_featurizer,
        oracle,
        acquisition=acquisition,
        surrogate_factory=lambda: DeepEnsemble(n_models=5, epochs=120, seed=seed),
        init_size=init,
        batch_size=batch,
        n_rounds=rounds,
        seed=seed,
    )
    hist = loop.run()
    # Interpolate best_true onto a common query axis.
    return np.array(hist.n_queries), np.array(hist.best_true)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dim", type=int, default=6)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--init", type=int, default=10)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    methods = {"Bayesian opt (EI)": "ei", "Random screening": "random"}
    curves = {name: [] for name in methods}
    common_q = None

    for name, acq in methods.items():
        for s in range(args.seeds):
            rng = np.random.default_rng(1000 + s)
            pool = list(rng.uniform(-3.0, 3.0, size=(args.pool, args.dim)))
            q, best = run_method(
                acq, pool=pool, dim=args.dim, rounds=args.rounds,
                batch=args.batch, init=args.init, noise=args.noise, seed=s,
            )
            if common_q is None:
                common_q = q
            best_i = np.interp(common_q, q, best)
            curves[name].append(best_i)

    results = {"n_queries": common_q.tolist(), "global_optimum": 0.0, "methods": {}}
    for name in methods:
        arr = np.vstack(curves[name])
        results["methods"][name] = {
            "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(),
        }

    os.makedirs(os.path.join(ROOT, "docs", "results"), exist_ok=True)
    with open(os.path.join(ROOT, "docs", "results", "closed_loop.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    _plot(results)
    # Report the headline: fewer experiments to reach a target.
    bo = np.array(results["methods"]["Bayesian opt (EI)"]["mean"])
    rnd = np.array(results["methods"]["Random screening"]["mean"])
    print(f"Final best  — BO: {bo[-1]:.3f}   random: {rnd[-1]:.3f}   (optimum: 0.0)")


def _plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q = np.array(results["n_queries"])
    colors = {"Bayesian opt (EI)": "#2563eb", "Random screening": "#9ca3af"}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for name, c in colors.items():
        m = np.array(results["methods"][name]["mean"])
        s = np.array(results["methods"][name]["std"])
        ax.plot(q, m, color=c, lw=2.4, label=name)
        ax.fill_between(q, m - s, m + s, color=c, alpha=0.15)
    ax.axhline(results["global_optimum"], color="#059669", ls="--", lw=1.2, label="global optimum")
    ax.set_xlabel("experiments run (oracle queries)")
    ax.set_ylabel("best property found")
    ax.set_title("Closed-loop de novo optimization: BO vs. random screening")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, color="#e5e7eb")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "assets", "closed_loop.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
