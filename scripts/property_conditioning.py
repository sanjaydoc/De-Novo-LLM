#!/usr/bin/env python
"""Demonstrate property-conditioned generation and plot the distribution shift.

Generates an unconditioned pool and a property-steered selection from the same
model, then overlays their property histograms so the steering is visible.

    python scripts/property_conditioning.py -m entropy/gpt2_zinc_87m \
        --property qed --mode max --num 200 --oversample 10

Outputs: docs/assets/conditioning.png, generated/conditioned.txt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402

from denovo.condition import guided_generate  # noqa: E402
from denovo.config import GenerateConfig  # noqa: E402
from denovo.properties import PROPERTIES, build_objective  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", "-m", default="entropy/gpt2_zinc_87m")
    ap.add_argument("--property", "-p", default="qed", choices=list(PROPERTIES))
    ap.add_argument("--mode", choices=["max", "min", "target"], default="max")
    ap.add_argument("--target", type=float, default=None)
    ap.add_argument("--num", "-n", type=int, default=200)
    ap.add_argument("--oversample", type=int, default=10)
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    objective = build_objective(args.property, mode=args.mode, target=args.target)
    gen = GenerateConfig(num_samples=args.num, batch_size=50, max_new_tokens=128,
                         do_sample=True, temperature=1.0, top_p=0.95)

    print(f"Steering {args.model} -> {objective.describe()}")
    res = guided_generate(args.model, gen, objective, oversample=args.oversample,
                          trust_remote_code=args.trust_remote_code)
    print(res.summary(objective))

    os.makedirs(os.path.join(ROOT, "generated"), exist_ok=True)
    with open(os.path.join(ROOT, "generated", "conditioned.txt"), "w", encoding="utf-8") as fh:
        for smi, val in res.selected:
            fh.write(f"{smi}\t{val:.4f}\n")

    _plot(res, args.property, objective.describe())


def _plot(res, prop, desc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = np.array(res.baseline_values)
    sel = np.array(res.selected_values)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    bins = 30
    ax.hist(base, bins=bins, density=True, alpha=0.55, color="#9ca3af", label="unconditioned")
    ax.hist(sel, bins=bins, density=True, alpha=0.7, color="#2563eb", label="conditioned")
    ax.axvline(base.mean(), color="#6b7280", ls="--", lw=1)
    ax.axvline(sel.mean(), color="#1d4ed8", ls="--", lw=1.5)
    ax.set_xlabel(prop)
    ax.set_ylabel("density")
    ax.set_title(f"Property-conditioned generation — {desc}")
    ax.legend(frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(True, color="#e5e7eb")
    fig.tight_layout()
    out = os.path.join(ROOT, "docs", "assets", "conditioning.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
