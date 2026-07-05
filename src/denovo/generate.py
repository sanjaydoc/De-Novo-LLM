"""Sampling novel biomolecules from a trained checkpoint."""

from __future__ import annotations

from typing import List, Optional

from denovo.config import GenerateConfig
from denovo.model import load_for_generation


def _clean_output(text: str, prompt: str) -> str:
    """Strip the prompt prefix and surrounding whitespace from a completion."""
    if prompt and text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()


def generate(
    model_path: str,
    gen_cfg: GenerateConfig,
    *,
    trust_remote_code: bool = False,
    seed: Optional[int] = 42,
) -> List[str]:
    """Generate ``gen_cfg.num_samples`` sequences from the model at ``model_path``.

    Returns raw decoded strings (no validity filtering -- that is the
    evaluator's job, so callers can measure validity honestly).
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    model, tokenizer = load_for_generation(
        model_path, trust_remote_code=trust_remote_code
    )
    device = model.device

    prompt = gen_cfg.prompt or (tokenizer.bos_token or "")
    results: List[str] = []

    remaining = gen_cfg.num_samples
    batch = max(1, gen_cfg.batch_size)

    # Encode the prompt once; empty prompt -> start from BOS/eos-free seed.
    if prompt:
        enc = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")
    else:
        input_ids = None
        attention_mask = None

    gen_kwargs = dict(
        max_new_tokens=gen_cfg.max_new_tokens,
        do_sample=gen_cfg.do_sample,
        temperature=gen_cfg.temperature,
        top_p=gen_cfg.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if gen_cfg.top_k and gen_cfg.top_k > 0:
        gen_kwargs["top_k"] = gen_cfg.top_k

    with torch.no_grad():
        while remaining > 0:
            n = min(batch, remaining)
            if input_ids is not None:
                ids = input_ids.repeat(n, 1)
                mask = attention_mask.repeat(n, 1) if attention_mask is not None else None
                out = model.generate(
                    input_ids=ids, attention_mask=mask, num_return_sequences=1,
                    **gen_kwargs,
                )
            else:
                # No prompt: let generate seed from BOS by passing num_return_sequences.
                out = model.generate(
                    num_return_sequences=n,
                    bos_token_id=tokenizer.bos_token_id,
                    **gen_kwargs,
                )
            decoded = tokenizer.batch_decode(out, skip_special_tokens=True)
            results.extend(_clean_output(t, prompt) for t in decoded)
            remaining -= n

    return results[: gen_cfg.num_samples]
