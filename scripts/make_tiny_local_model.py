#!/usr/bin/env python
"""Build a tiny, randomly-initialised GPT-2 + tokenizer on the LOCAL disk.

This exists so the full fine-tuning pipeline can be exercised end-to-end with
**no network** (the Hugging Face Hub is not always reachable, e.g. in CI or
sandboxes). The tokenizer is trained on a sample file; the model is a 2-layer
toy. It will not produce meaningful molecules -- it proves the plumbing.

    python scripts/make_tiny_local_model.py \
        --data data/samples/smiles_sample.txt --out outputs/_tiny_local

Then:  denovo pipeline -c configs/smoke_local.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def build(data_file: str, out_dir: str, vocab_size: int = 500, n_layer: int = 2,
          n_embd: int = 64, n_head: int = 2, n_positions: int = 128) -> str:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    os.makedirs(out_dir, exist_ok=True)

    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    raw = Tokenizer(BPE(unk_token="<unk>"))
    raw.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=specials, min_frequency=1)
    raw.train([data_file], trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(out_dir)

    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=n_positions,
        n_ctx=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/samples/smiles_sample.txt")
    ap.add_argument("--out", default="outputs/_tiny_local")
    args = ap.parse_args()
    out = build(args.data, args.out)
    print(f"Tiny local model + tokenizer written to: {out}")
    print("Now run:  denovo pipeline -c configs/smoke_local.yaml")


if __name__ == "__main__":
    main()
