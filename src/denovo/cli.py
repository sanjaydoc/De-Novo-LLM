"""Command-line interface: ``denovo <command> --config config.yaml``.

Commands
--------
prepare   Clean/validate/canonicalise a raw data file into train/eval splits.
train     Fine-tune the configured model.
generate  Sample sequences from a trained checkpoint.
evaluate  Score generated sequences (validity/uniqueness/novelty/diversity).
pipeline  prepare -> train -> generate -> evaluate in one go.
info      Show registered modalities and the resolved model for a config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from denovo.config import Config, load_config
from denovo.modalities import get_modality, list_modalities


def _load(args) -> Config:
    if not args.config:
        raise SystemExit("This command requires --config PATH.yaml")
    return load_config(args.config)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_info(args) -> None:
    print("Registered modalities:")
    for name in list_modalities():
        m = get_modality(name)
        print(f"  - {name:10s} [{m.kind}]  default: {m.default_model}")
        if m.description:
            print(f"               {m.description}")
    if args.config:
        cfg = load_config(args.config)
        print("\nResolved for this config:")
        print(f"  modality : {cfg.data.modality}")
        print(f"  model    : {cfg.resolved_model()}")
        print(f"  lora     : {cfg.lora.use_lora}  4bit: {cfg.model.load_in_4bit}")


def cmd_prepare(args) -> None:
    from denovo.data import prepare_dataset

    cfg = _load(args)
    inp = args.input or cfg.data.train_file
    train_path, eval_path, stats = prepare_dataset(
        inp,
        cfg.data.modality,
        out_dir=args.out_dir,
        text_column=cfg.data.text_column,
        filter_invalid=cfg.data.filter_invalid,
        canonicalize=cfg.data.canonicalize,
        validation_split=cfg.data.validation_split,
        seed=cfg.train.seed,
    )
    print("Prepared dataset:")
    print(json.dumps(stats, indent=2))
    print(f"  train -> {train_path}")
    if eval_path:
        print(f"  eval  -> {eval_path}")
    print(
        "\nSet these in your config's `data` section:\n"
        f"  train_file: {train_path}"
        + (f"\n  eval_file: {eval_path}" if eval_path else "")
    )


def cmd_train(args) -> None:
    from denovo.train import train

    cfg = _load(args)
    print(f"Fine-tuning {cfg.resolved_model()} on modality={cfg.data.modality}")
    out = train(cfg)
    print(f"\nDone. Model saved to: {out}")


def cmd_generate(args) -> None:
    from denovo.generate import generate

    cfg = _load(args)
    model_path = args.model or cfg.train.output_dir
    if args.num:
        cfg.generate.num_samples = args.num
    seqs = generate(
        model_path,
        cfg.generate,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    out_path = args.output
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            for s in seqs:
                fh.write(s + "\n")
        print(f"Wrote {len(seqs)} sequences to {out_path}")
    else:
        for s in seqs:
            print(s)


def cmd_evaluate(args) -> None:
    from denovo.data import read_sequences
    from denovo.evaluate import evaluate_sequences

    cfg = _load(args)
    generated = read_sequences(args.input)
    metrics = evaluate_sequences(
        generated,
        cfg.data.modality,
        training_file=cfg.data.train_file,
    )
    print(f"Evaluation (modality={cfg.data.modality}):")
    print(metrics.pretty())
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(metrics.as_dict(), fh, indent=2)
        print(f"\nMetrics written to {args.output}")


def cmd_condition(args) -> None:
    """Property-conditioned generation: steer molecules toward a target property."""
    from denovo.condition import guided_generate, iterative_generate
    from denovo.properties import PROPERTIES, build_objective

    cfg = _load(args)
    if args.num:
        cfg.generate.num_samples = args.num
    objective = build_objective(
        args.property, mode=args.mode, target=args.target, tolerance=args.tolerance
    )
    model_path = args.model or cfg.train.output_dir
    runner = iterative_generate if args.rounds > 1 else guided_generate
    kwargs = dict(
        modality_name=cfg.data.modality,
        oversample=args.oversample,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if args.rounds > 1:
        kwargs["rounds"] = args.rounds
    result = runner(model_path, cfg.generate, objective, **kwargs)

    print(f"Property-conditioned generation ({PROPERTIES[args.property]}):")
    print(result.summary(objective))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            for smi, val in result.selected:
                fh.write(f"{smi}\t{val:.4f}\n")
        print(f"\nWrote {len(result.selected)} molecules (+values) to {args.output}")


def cmd_scaffold(args) -> None:
    """Scaffold-constrained generation: keep molecules containing a substructure."""
    from denovo.properties import build_objective
    from denovo.scaffold import scaffold_generate

    cfg = _load(args)
    if args.num:
        cfg.generate.num_samples = args.num
    objective = None
    if args.property:
        objective = build_objective(args.property, mode=args.mode, target=args.target)
    model_path = args.model or cfg.train.output_dir
    result = scaffold_generate(
        model_path, cfg.generate, args.scaffold,
        is_smarts=args.smarts, modality_name=cfg.data.modality,
        objective=objective, oversample=args.oversample,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    print("Scaffold-constrained generation:")
    print(result.summary(args.scaffold, objective))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            for smi, val in result.matches:
                fh.write(smi + (f"\t{val:.4f}" if val is not None else "") + "\n")
        print(f"\nWrote {len(result.matches)} molecules to {args.output}")


def cmd_nim(args) -> None:
    """Call NVIDIA hosted NIMs (MolMIM / ESMFold / Evo 2)."""
    from denovo.nim import NIM_MODELS

    if args.service == "list":
        print("NVIDIA NIM services (set NVIDIA_API_KEY; keys at build.nvidia.com):")
        for name, desc in NIM_MODELS.items():
            print(f"  - {name:8s} {desc}")
        return

    from denovo.nim import NIMClient

    client = NIMClient()  # reads NVIDIA_API_KEY

    if args.service == "molmim":
        if not args.smi:
            raise SystemExit("nim molmim requires --smi <seed SMILES>")
        algorithm = "none" if args.no_optimize else "CMA-ES"
        mols = client.molmim_generate(
            args.smi, num_molecules=args.num, algorithm=algorithm,
            property_name=args.property, minimize=args.minimize,
        )
        print(f"MolMIM returned {len(mols)} molecules"
              + ("" if args.no_optimize else f" (optimizing {args.property}):"))
        for m in mols[:20]:
            extra = f"  score={m['score']:.3f}" if m.get("score") is not None else ""
            print(f"  {m['smiles']}{extra}")
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as fh:
                for m in mols:
                    fh.write((m["smiles"] or "") + "\n")
            print(f"Wrote {len(mols)} molecules to {args.output}")

    elif args.service == "esmfold":
        if not args.sequence:
            raise SystemExit("nim esmfold requires --sequence <amino acids>")
        pdb = client.esmfold_predict(args.sequence)
        out = args.output or "structure.pdb"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(pdb or "")
        print(f"ESMFold structure written to {out} ({len(pdb or '')} chars)")

    elif args.service == "evo2":
        if not args.sequence:
            raise SystemExit("nim evo2 requires --sequence <DNA>")
        seq = client.evo2_generate(args.sequence, num_tokens=args.num)
        print(f"Evo 2 generated:\n{seq}")
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write((seq or "") + "\n")


def cmd_optimize(args) -> None:
    """Bayesian (TPE) hyperparameter optimization over quality metrics."""
    from denovo.optimize import ObjectiveWeights, optimize

    cfg = _load(args)
    result = optimize(
        cfg,
        mode=args.mode,
        model_path=args.model or cfg.train.output_dir,
        n_trials=args.trials,
        eval_samples=args.eval_samples,
        weights=ObjectiveWeights(),
    )
    print(f"\nBest score: {result.best_score:.4f}")
    print("Best hyperparameters:")
    print(json.dumps(result.best_params, indent=2))
    out = args.output or os.path.join(cfg.train.output_dir, "bo_study.json")
    result.save(out)
    print(f"\nStudy saved to {out}")
    print("Render figures with:  python scripts/make_figures.py --study " + out)


def cmd_pipeline(args) -> None:
    """End-to-end: prepare -> train -> generate -> evaluate."""
    from denovo.data import prepare_dataset, read_sequences
    from denovo.evaluate import evaluate_sequences
    from denovo.generate import generate
    from denovo.train import train

    cfg = _load(args)

    if not args.skip_prepare:
        print("== [1/4] prepare ==")
        train_path, eval_path, stats = prepare_dataset(
            args.input or cfg.data.train_file,
            cfg.data.modality,
            out_dir=args.out_dir,
            text_column=cfg.data.text_column,
            filter_invalid=cfg.data.filter_invalid,
            canonicalize=cfg.data.canonicalize,
            validation_split=cfg.data.validation_split,
            seed=cfg.train.seed,
        )
        print(json.dumps(stats, indent=2))
        cfg.data.train_file = train_path
        cfg.data.eval_file = eval_path

    print("\n== [2/4] train ==")
    out = train(cfg)

    print("\n== [3/4] generate ==")
    seqs = generate(out, cfg.generate, trust_remote_code=cfg.model.trust_remote_code)
    gen_path = os.path.join(out, "generated.txt")
    with open(gen_path, "w", encoding="utf-8") as fh:
        for s in seqs:
            fh.write(s + "\n")
    print(f"Wrote {len(seqs)} sequences to {gen_path}")

    print("\n== [4/4] evaluate ==")
    metrics = evaluate_sequences(
        seqs, cfg.data.modality, training_file=cfg.data.train_file
    )
    print(metrics.pretty())
    with open(os.path.join(out, "metrics.json"), "w") as fh:
        json.dump(metrics.as_dict(), fh, indent=2)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="denovo",
        description="Fine-tune LLMs to generate de novo biomolecules.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_config(sp):
        sp.add_argument("--config", "-c", help="Path to a YAML config file.")

    sp = sub.add_parser("info", help="List modalities / resolved model.")
    add_config(sp)
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("prepare", help="Clean raw data into train/eval splits.")
    add_config(sp)
    sp.add_argument("--input", "-i", help="Raw input file (overrides config).")
    sp.add_argument("--out-dir", default="data/processed")
    sp.set_defaults(func=cmd_prepare)

    sp = sub.add_parser("train", help="Fine-tune the configured model.")
    add_config(sp)
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("generate", help="Sample from a trained checkpoint.")
    add_config(sp)
    sp.add_argument("--model", "-m", help="Checkpoint dir (overrides config).")
    sp.add_argument("--num", "-n", type=int, help="Number of samples.")
    sp.add_argument("--output", "-o", help="Write sequences here (else stdout).")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("evaluate", help="Score generated sequences.")
    add_config(sp)
    sp.add_argument("--input", "-i", required=True, help="File of generated seqs.")
    sp.add_argument("--output", "-o", help="Write metrics JSON here.")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("condition", help="Property-conditioned generation (logP/QED/...).")
    add_config(sp)
    sp.add_argument("--property", "-p", required=True,
                    help="Property: logp | qed | mw | tpsa | hbd | hba | rings | rotbonds")
    sp.add_argument("--mode", choices=["max", "min", "target"], default="max")
    sp.add_argument("--target", type=float, help="Target value (for --mode target).")
    sp.add_argument("--tolerance", type=float, default=0.5)
    sp.add_argument("--num", "-n", type=int, help="Molecules to keep.")
    sp.add_argument("--oversample", type=int, default=8, help="Generate N x this, keep the best.")
    sp.add_argument("--rounds", type=int, default=1, help=">1 runs iterative rounds.")
    sp.add_argument("--model", "-m", help="Checkpoint / HF id.")
    sp.add_argument("--output", "-o", help="Write 'smiles<TAB>value' lines here.")
    sp.set_defaults(func=cmd_condition)

    sp = sub.add_parser("scaffold", help="Scaffold/substructure-constrained generation.")
    add_config(sp)
    sp.add_argument("--scaffold", "-s", required=True, help="Scaffold SMILES (or SMARTS with --smarts).")
    sp.add_argument("--smarts", action="store_true", help="Treat --scaffold as SMARTS.")
    sp.add_argument("--property", "-p", help="Optionally rank matches by this property.")
    sp.add_argument("--mode", choices=["max", "min", "target"], default="max")
    sp.add_argument("--target", type=float)
    sp.add_argument("--num", "-n", type=int, help="Molecules to keep.")
    sp.add_argument("--oversample", type=int, default=20)
    sp.add_argument("--model", "-m", help="Checkpoint / HF id.")
    sp.add_argument("--output", "-o", help="Write matches here.")
    sp.set_defaults(func=cmd_scaffold)

    sp = sub.add_parser("nim", help="Call NVIDIA hosted NIMs (MolMIM/ESMFold/Evo2).")
    sp.add_argument("--service", choices=["molmim", "esmfold", "evo2", "list"], default="list")
    sp.add_argument("--smi", help="Seed SMILES (molmim).")
    sp.add_argument("--sequence", help="Protein (esmfold) or DNA (evo2) sequence.")
    sp.add_argument("--num", "-n", type=int, default=30, help="Molecules / tokens.")
    sp.add_argument("--property", "-p", default="QED", help="MolMIM property to optimize.")
    sp.add_argument("--minimize", action="store_true", help="Minimize the property.")
    sp.add_argument("--no-optimize", action="store_true", help="Sample only (no CMA-ES).")
    sp.add_argument("--output", "-o", help="Output file.")
    sp.set_defaults(func=cmd_nim)

    sp = sub.add_parser("optimize", help="Bayesian (TPE) hyperparameter search.")
    add_config(sp)
    sp.add_argument("--mode", choices=["sampling", "training"], default="sampling")
    sp.add_argument("--model", "-m", help="Trained checkpoint (sampling mode).")
    sp.add_argument("--trials", "-t", type=int, default=25)
    sp.add_argument("--eval-samples", type=int, default=200)
    sp.add_argument("--output", "-o", help="Where to write the study JSON.")
    sp.set_defaults(func=cmd_optimize)

    sp = sub.add_parser("pipeline", help="prepare -> train -> generate -> evaluate.")
    add_config(sp)
    sp.add_argument("--input", "-i", help="Raw input file (overrides config).")
    sp.add_argument("--out-dir", default="data/processed")
    sp.add_argument("--skip-prepare", action="store_true")
    sp.set_defaults(func=cmd_pipeline)

    return p


def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
