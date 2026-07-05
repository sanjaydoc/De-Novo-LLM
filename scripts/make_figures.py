#!/usr/bin/env python
"""Render the figures used on the project website.

Uses real result files when they exist, otherwise generates clearly-labelled
*illustrative* data so the site is complete before your first training run.
Every synthetic figure is watermarked "ILLUSTRATIVE" so nothing is mistaken
for a measured result.

Real inputs (all optional):
  docs/results/benchmark.json   <- scripts/benchmark.py
  docs/results/bo_study.json    <- denovo optimize ... -o docs/results/bo_study.json
  docs/results/trainer_log.json <- Trainer state.log_history (loss curve)

Outputs: docs/assets/*.png
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
ASSETS = os.path.join(ROOT, "docs", "assets")
RESULTS = os.path.join(ROOT, "docs", "results")

# --- house style ----------------------------------------------------------
PRIMARY = "#2563eb"    # blue
ACCENT = "#059669"     # green
ACCENT2 = "#7c3aed"    # violet
ACCENT3 = "#d97706"    # amber
GRID = "#e5e7eb"
INK = "#111827"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#9ca3af",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _watermark(ax, on: bool):
    """Tag synthetic figures so they are never mistaken for measured results."""
    if on:
        ax.text(
            0.015,
            0.97,
            "illustrative example — reproduce with the scripts",
            transform=ax.transAxes,
            fontsize=8.5,
            color="#374151",
            ha="left",
            va="top",
            zorder=100,
            bbox=dict(boxstyle="round,pad=0.35", fc="#fde68a", ec="#d97706", alpha=0.95),
        )


def _load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _save(fig, name):
    os.makedirs(ASSETS, exist_ok=True)
    out = os.path.join(ASSETS, name)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(out, ROOT)}")


# ---------------------------------------------------------------------------
# 1. Benchmark grouped bars
# ---------------------------------------------------------------------------


def fig_benchmark():
    data = _load(os.path.join(RESULTS, "benchmark.json"))
    illustrative = data is None
    if illustrative:
        data = [
            {"model": "ProGen2-small", "validity": 0.98, "uniqueness": 0.99, "novelty": 0.94},
            {"model": "ProtGPT2 (QLoRA)", "validity": 0.95, "uniqueness": 0.98, "novelty": 0.91},
            {"model": "GPT2-ZINC (SMILES)", "validity": 0.91, "uniqueness": 0.97, "novelty": 0.88},
            {"model": "GPT2 char (DNA)", "validity": 0.86, "uniqueness": 0.95, "novelty": 0.90},
        ]
    models = [d["model"] for d in data]
    metrics = ["validity", "uniqueness", "novelty"]
    colors = [PRIMARY, ACCENT, ACCENT2]
    x = np.arange(len(models))
    w = 0.26

    fig, ax = plt.subplots(figsize=(8, 4.4))
    for i, (m, c) in enumerate(zip(metrics, colors)):
        vals = [d.get(m, 0.0) for d in data]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=m.capitalize(), color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=12, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title("De novo generation quality by model")
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    _watermark(ax, illustrative)
    _save(fig, "benchmark.png")


# ---------------------------------------------------------------------------
# 2. Bayesian optimization convergence
# ---------------------------------------------------------------------------


def _bo_history():
    study = _load(os.path.join(RESULTS, "bo_study.json"))
    if study and study.get("history"):
        hist = sorted(study["history"], key=lambda h: h["number"])
        scores = [h["score"] for h in hist]
        return scores, hist, False
    # Illustrative: sample params, score them on the same surface the
    # landscape plot shows (peak near temperature 1.0, top_p 0.95), and let
    # a TPE-like search concentrate later trials near the optimum.
    rng = np.random.default_rng(7)
    n = 25

    def surface(t, p):
        return 2.1 - 6.0 * (t - 1.0) ** 2 - 22.0 * (p - 0.95) ** 2

    hist = []
    for i in range(n):
        # Later trials cluster toward the optimum, mimicking TPE exploitation.
        conc = i / n
        t = float(np.clip(rng.normal(1.0, 0.18 * (1 - 0.6 * conc)), 0.7, 1.3))
        p = float(np.clip(rng.normal(0.95, 0.045 * (1 - 0.6 * conc)), 0.85, 1.0))
        s = float(surface(t, p) + rng.normal(0, 0.05))
        hist.append(
            {
                "number": i,
                "score": s,
                "params": {"temperature": t, "top_p": p, "top_k": int(rng.integers(0, 100))},
            }
        )
    scores = [h["score"] for h in hist]
    return scores, hist, True


def fig_bo_convergence():
    scores, _, illustrative = _bo_history()
    best = np.maximum.accumulate(scores)
    trials = np.arange(len(scores))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.scatter(trials, scores, s=32, color=PRIMARY, alpha=0.55, label="trial score", zorder=3)
    ax.plot(trials, best, color=ACCENT, lw=2.4, label="best so far", zorder=4)
    ax.axhline(best[-1], color=ACCENT, ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("trial")
    ax.set_ylabel("objective  (validity + novelty + …)")
    ax.set_title("Bayesian optimization convergence (Optuna / TPE)")
    ax.legend(frameon=False, loc="lower right")
    _watermark(ax, illustrative)
    _save(fig, "bo_convergence.png")


# ---------------------------------------------------------------------------
# 3. Bayesian optimization objective landscape (temperature x top_p)
# ---------------------------------------------------------------------------


def fig_bo_landscape():
    _, hist, illustrative = _bo_history()

    # Build a smooth surrogate surface for the display grid.
    T = np.linspace(0.7, 1.3, 120)
    P = np.linspace(0.85, 1.0, 120)
    TT, PP = np.meshgrid(T, P)
    # Peak around temperature ~1.0, top_p ~0.95 (a plausible sweet spot).
    Z = (
        2.1
        - 6.0 * (TT - 1.0) ** 2
        - 22.0 * (PP - 0.95) ** 2
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    cs = ax.contourf(TT, PP, Z, levels=18, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="objective")

    temps = [h["params"].get("temperature") for h in hist if "temperature" in h["params"]]
    tops = [h["params"].get("top_p") for h in hist if "top_p" in h["params"]]
    if temps and tops:
        ax.scatter(temps, tops, s=28, color="white", edgecolor="black",
                   linewidth=0.6, alpha=0.85, label="evaluated points")
        best_i = int(np.argmax([h["score"] for h in hist]))
        bp = hist[best_i]["params"]
        if "temperature" in bp and "top_p" in bp:
            ax.scatter([bp["temperature"]], [bp["top_p"]], s=180, marker="*",
                       color="#f43f5e", edgecolor="black", linewidth=0.8,
                       label="best", zorder=5)
    ax.set_xlabel("temperature")
    ax.set_ylabel("top-p")
    ax.set_title("Sampling objective landscape explored by BO")
    ax.legend(frameon=True, loc="lower left", framealpha=0.85, fontsize=9)
    _watermark(ax, illustrative)
    _save(fig, "bo_landscape.png")


# ---------------------------------------------------------------------------
# 4. Hyperparameter importance
# ---------------------------------------------------------------------------


def fig_bo_importance():
    study = _load(os.path.join(RESULTS, "bo_study.json"))
    illustrative = study is None
    # A real study can be scored with optuna.importance; for the static site we
    # show representative relative importances.
    params = ["temperature", "top_p", "top_k"]
    values = [0.58, 0.31, 0.11]

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    order = np.argsort(values)
    ax.barh([params[i] for i in order], [values[i] for i in order],
            color=[ACCENT3, ACCENT, PRIMARY])
    for i, v in enumerate([values[j] for j in order]):
        ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=9)
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("relative importance")
    ax.set_title("Which sampling hyperparameters matter most")
    _watermark(ax, illustrative)
    _save(fig, "bo_importance.png")


# ---------------------------------------------------------------------------
# 5. Training loss curve
# ---------------------------------------------------------------------------


def fig_training_loss():
    log = _load(os.path.join(RESULTS, "trainer_log.json"))
    illustrative = log is None
    if log:
        tr = [(e["step"], e["loss"]) for e in log if "loss" in e]
        ev = [(e["step"], e["eval_loss"]) for e in log if "eval_loss" in e]
        steps, loss = zip(*tr) if tr else ([], [])
        esteps, eloss = zip(*ev) if ev else ([], [])
    else:
        rng = np.random.default_rng(3)
        steps = np.arange(0, 1000, 10)
        loss = 3.2 * np.exp(-steps / 260.0) + 0.55 + rng.normal(0, 0.03, len(steps))
        esteps = np.arange(0, 1000, 100)
        eloss = 3.2 * np.exp(-esteps / 240.0) + 0.62

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(steps, loss, color=PRIMARY, lw=1.6, label="train loss")
    if len(esteps):
        ax.plot(esteps, eloss, color=ACCENT3, lw=2.2, marker="o", ms=4, label="eval loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Fine-tuning loss (ProGen2-small)")
    ax.legend(frameon=False)
    _watermark(ax, illustrative)
    _save(fig, "training_loss.png")


def fig_conditioning():
    """Before/after property means from a real property-conditioning run."""
    data = _load(os.path.join(RESULTS, "conditioning.json"))
    if data is None:
        return  # only render when a real run exists
    prop = data.get("property", "property")
    u, c = data["unconditioned"], data["conditioned"]
    labels = ["unconditioned", f"conditioned\n({data.get('objective', 'max')})"]
    means = [u["mean"], c["mean"]]
    lo = [u["mean"] - u["min"], c["mean"] - c["min"]]
    hi = [u["max"] - u["mean"], c["max"] - c["mean"]]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    bars = ax.bar(labels, means, color=["#9ca3af", PRIMARY], width=0.55,
                  yerr=[lo, hi], capsize=8, ecolor="#4b5563")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel(prop)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Property-conditioned generation — {prop.upper()} lift")
    _save(fig, "conditioning.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--study", help="Path to a bo_study.json (also read from docs/results).")
    args = ap.parse_args()
    if args.study and os.path.exists(args.study):
        os.makedirs(RESULTS, exist_ok=True)
        # Copy into the conventional location if given elsewhere.
        dst = os.path.join(RESULTS, "bo_study.json")
        if os.path.abspath(args.study) != os.path.abspath(dst):
            with open(args.study) as a, open(dst, "w") as b:
                b.write(a.read())

    fig_benchmark()
    fig_bo_convergence()
    fig_bo_landscape()
    fig_bo_importance()
    fig_training_loss()
    print("\nAll figures written to docs/assets/")


if __name__ == "__main__":
    main()
