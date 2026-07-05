#!/usr/bin/env python
"""Benchmark one or more trained checkpoints and emit a results table.

For each (name, checkpoint) pair it generates ``--num`` sequences and computes
the standard de novo metrics, then writes a JSON + Markdown table under
``--out-dir`` (default ``docs/results``). Feed that JSON to
``scripts/make_figures.py`` to render the website charts.

Example
-------
    python scripts/benchmark.py \
        --config configs/progen2_protein.yaml \
        --model ProGen2-small outputs/progen2_small \
        --model ProtGPT2      outputs/protgpt2_qlora \
        --num 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from a source checkout without installation.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from denovo.config import load_config           # noqa: E402
from denovo.evaluate import evaluate_sequences  # noqa: E402
from denovo.generate import generate            # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", "-c", required=True)
    ap.add_argument(
        "--model",
        "-m",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
        help="A display name and a checkpoint path. Repeatable.",
    )
    ap.add_argument("--num", "-n", type=int, default=1000)
    ap.add_argument("--out-dir", default="docs/results")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.generate.num_samples = args.num
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for name, path in args.model:
        print(f"== Benchmarking {name} ({path}) ==")
        seqs = generate(path, cfg.generate, trust_remote_code=cfg.model.trust_remote_code)
        metrics = evaluate_sequences(
            seqs, cfg.data.modality, training_file=cfg.data.train_file
        )
        print(metrics.pretty())
        row = {"model": name, "modality": cfg.data.modality, **metrics.as_dict()}
        rows.append(row)

    json_path = os.path.join(args.out_dir, "benchmark.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    md_path = os.path.join(args.out_dir, "benchmark.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("| Model | Validity | Uniqueness | Novelty | Diversity |\n")
        fh.write("|-------|----------|------------|---------|-----------|\n")
        for r in rows:
            div = f"{r['diversity']:.3f}" if r.get("diversity") is not None else "-"
            fh.write(
                f"| {r['model']} | {r['validity']:.1%} | {r['uniqueness']:.1%} "
                f"| {r['novelty']:.1%} | {div} |\n"
            )

    print(f"\nWrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
