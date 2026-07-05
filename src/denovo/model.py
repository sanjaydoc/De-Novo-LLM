"""Model & tokenizer construction with optional quantisation and LoRA.

Three fine-tuning regimes, chosen from config:

* **full**  -- update all weights (fine for the tiny SMILES/SELFIES models).
* **LoRA**  -- freeze the base, train small adapters (``lora.use_lora: true``).
* **QLoRA** -- 4-bit base + LoRA (``model.load_in_4bit: true`` + LoRA), the
  only way ~700M models like ProtGPT2 fit in 6GB.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from denovo.config import Config


_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(name: str):
    if name in ("auto", "", None):
        return "auto"
    if name not in _DTYPES:
        raise ValueError(f"Unknown torch_dtype {name!r}. Options: {list(_DTYPES)}.")
    return _DTYPES[name]


# Known attention projection module names per architecture family.
_TARGET_HINTS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",  # llama / neo / opt
    "c_attn", "c_proj",                                    # gpt2
    "query_key_value", "dense",                            # bloom / falcon
    "Wqkv",                                                # mpt
)


def find_target_modules(model) -> List[str]:
    """Best-effort discovery of attention Linear layers for LoRA.

    Returns the set of *leaf* module names (the last dotted component) that
    look like attention projections, which is what PEFT matches against.
    """
    import torch.nn as nn

    found = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf = name.split(".")[-1]
            if leaf in _TARGET_HINTS:
                found.add(leaf)
    if not found:
        # Fall back to letting PEFT decide; empty tuple triggers its defaults.
        return []
    return sorted(found)


def load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    name = cfg.model.tokenizer_name or cfg.resolved_model()
    tokenizer = AutoTokenizer.from_pretrained(
        name, trust_remote_code=cfg.model.trust_remote_code
    )
    # Causal LMs need a pad token for batching; reuse EOS when missing.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def _quant_config(cfg: Config):
    if not (cfg.model.load_in_4bit or cfg.model.load_in_8bit):
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "4-bit/8-bit loading needs bitsandbytes: pip install bitsandbytes"
        ) from exc
    if cfg.model.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_model(cfg: Config, tokenizer=None):
    """Build the (optionally quantised + LoRA-wrapped) causal LM."""
    from transformers import AutoModelForCausalLM

    quant = _quant_config(cfg)
    dtype = _resolve_dtype(cfg.model.torch_dtype)

    kwargs = dict(trust_remote_code=cfg.model.trust_remote_code)
    if quant is not None:
        kwargs["quantization_config"] = quant
    else:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(cfg.resolved_model(), **kwargs)

    # Resize embeddings if we added a pad token above.
    if tokenizer is not None and len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    use_qlora = quant is not None and cfg.lora.use_lora
    if use_qlora:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
        )

    if cfg.lora.use_lora:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        targets = cfg.lora.target_modules
        if not targets:
            targets = find_target_modules(model)
        peft_cfg = PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=targets or None,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    return model


def load_model_and_tokenizer(cfg: Config) -> Tuple[object, object]:
    tokenizer = load_tokenizer(cfg)
    model = load_model(cfg, tokenizer=tokenizer)
    return model, tokenizer


def load_for_generation(model_path: str, *, trust_remote_code: bool = False,
                        base_model: Optional[str] = None):
    """Load a trained checkpoint for sampling.

    Handles both full checkpoints and PEFT adapter directories (the latter
    detected by the presence of an ``adapter_config.json``).
    """
    import os

    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))

    if is_adapter:
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        tok_src = model_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        tok_src = model_path

    tokenizer = AutoTokenizer.from_pretrained(
        tok_src, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer
