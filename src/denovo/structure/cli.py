"""CLI for the SE(3)-equivariant flow-matching molecule generator.

    denovo-mol train    -c configs/mol_flow.yaml
    denovo-mol sample   -c configs/mol_flow.yaml -o generated/
    denovo-mol pipeline -c configs/mol_flow.yaml     # train -> sample -> evaluate
"""

from __future__ import annotations

import argparse
import json
import os

from denovo.structure.config import load_structure_config


def cmd_train(args):
    from denovo.structure.train import train

    cfg = load_structure_config(args.config)
    out = train(cfg)
    print(f"Done. Model in {out}")


def cmd_sample(args):
    from denovo.structure.evaluate import evaluate_molecules
    from denovo.structure.sample import sample_molecules, write_outputs

    cfg = load_structure_config(args.config)
    model_dir = args.model or cfg.train.output_dir
    n = args.num or cfg.sample.n_samples
    mols = sample_molecules(
        model_dir, n_samples=n, n_steps=cfg.sample.n_steps, batch_size=cfg.sample.batch_size
    )
    out_dir = args.output or os.path.join(model_dir, "generated")
    smiles, n_valid = write_outputs(mols, out_dir)
    print(f"Generated {len(mols)} molecules -> {out_dir}  ({n_valid} valid SMILES)")
    metrics = evaluate_molecules(mols)
    print(metrics.pretty())


def cmd_pipeline(args):
    from denovo.structure.evaluate import evaluate_molecules
    from denovo.structure.sample import sample_molecules, write_outputs
    from denovo.structure.train import train

    cfg = load_structure_config(args.config)
    print("== train ==")
    out = train(cfg)
    print("\n== sample ==")
    mols = sample_molecules(out, n_samples=cfg.sample.n_samples, n_steps=cfg.sample.n_steps,
                            batch_size=cfg.sample.batch_size)
    smiles, n_valid = write_outputs(mols, os.path.join(out, "generated"))
    print(f"generated {len(mols)} molecules ({n_valid} valid)")
    print("\n== evaluate ==")
    metrics = evaluate_molecules(mols)
    print(metrics.pretty())
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics.as_dict(), fh, indent=2)


def build_parser():
    p = argparse.ArgumentParser(prog="denovo-mol",
                                description="SE(3)-equivariant flow-matching 3D molecule generation.")
    sub = p.add_subparsers(dest="command", required=True)

    def cfg(sp):
        sp.add_argument("--config", "-c", required=True)

    sp = sub.add_parser("train", help="Train the equivariant flow model.")
    cfg(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("sample", help="Sample molecules from a trained model.")
    cfg(sp)
    sp.add_argument("--model", "-m")
    sp.add_argument("--num", "-n", type=int)
    sp.add_argument("--output", "-o")
    sp.set_defaults(func=cmd_sample)

    sp = sub.add_parser("pipeline", help="train -> sample -> evaluate.")
    cfg(sp)
    sp.set_defaults(func=cmd_pipeline)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
