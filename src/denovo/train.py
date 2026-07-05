"""Fine-tuning loop built on the Hugging Face ``Trainer``."""

from __future__ import annotations

import json
import os
from typing import Optional

from denovo.config import Config
from denovo.data import build_tokenized_datasets
from denovo.model import load_model_and_tokenizer


def train(cfg: Config) -> str:
    """Fine-tune according to ``cfg`` and return the output directory."""
    import torch
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(cfg.train.seed)

    model, tokenizer = load_model_and_tokenizer(cfg)

    train_ds, eval_ds = build_tokenized_datasets(
        tokenizer,
        cfg.data.train_file,
        cfg.data.eval_file,
        max_length=cfg.data.max_length,
        text_column=cfg.data.text_column,
        validation_split=cfg.data.validation_split,
        seed=cfg.train.seed,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    os.makedirs(cfg.train.output_dir, exist_ok=True)

    # Only enable fp16/bf16 on CUDA; on CPU they slow things down or error.
    cuda = torch.cuda.is_available()
    fp16 = cfg.train.fp16 and cuda
    bf16 = cfg.train.bf16 and cuda and torch.cuda.is_bf16_supported()

    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        gradient_accumulation_steps=cfg.train.grad_accum,
        learning_rate=cfg.train.lr,
        warmup_ratio=cfg.train.warmup_ratio,
        weight_decay=cfg.train.weight_decay,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=cfg.train.eval_steps if eval_ds is not None else None,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        fp16=fp16,
        bf16=bf16,
        report_to=[] if cfg.train.report_to == "no" else [cfg.train.report_to],
        seed=cfg.train.seed,
        dataloader_pin_memory=cuda,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()

    # Persist final model + tokenizer + the exact config used.
    trainer.save_model(cfg.train.output_dir)
    tokenizer.save_pretrained(cfg.train.output_dir)
    with open(os.path.join(cfg.train.output_dir, "denovo_config.json"), "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)

    return cfg.train.output_dir
