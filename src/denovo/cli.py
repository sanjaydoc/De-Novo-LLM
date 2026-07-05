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
