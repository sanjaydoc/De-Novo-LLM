"""Data loading, cleaning and tokenisation.

Input formats supported (auto-detected by extension):

* ``.txt`` / ``.smi`` -- one sequence per line.
* ``.csv``            -- a column named by ``data.text_column`` (or the first
  column if unset).
* ``.parquet``        -- same column rules as CSV.

``prepare_dataset`` cleans a raw file into deduplicated, optionally
canonicalised train/eval files.  ``build_tokenized_datasets`` turns cleaned
files into tokenised ``datasets.Dataset`` objects ready for the Trainer.
"""

from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

from denovo.modalities import Modality, get_modality


# ---------------------------------------------------------------------------
# Raw reading / writing
# ---------------------------------------------------------------------------


def read_sequences(path: str, text_column: Optional[str] = None) -> List[str]:
    """Read a list of raw sequence strings from ``path``."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".smi", ".seq", ""):
        with open(path, "r", encoding="utf-8") as fh:
            # For .smi the first whitespace-delimited token is the SMILES.
            lines = []
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(line.split()[0] if ext == ".smi" else line)
            return lines
    if ext in (".csv", ".parquet"):
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                f"Reading {ext} files requires pandas: pip install pandas"
            ) from exc
        df = pd.read_csv(path) if ext == ".csv" else pd.read_parquet(path)
        col = text_column or df.columns[0]
        if col not in df.columns:
            raise KeyError(
                f"Column {col!r} not found in {path!r}. "
                f"Available columns: {list(df.columns)}."
            )
        return [str(x) for x in df[col].dropna().tolist()]
    raise ValueError(f"Unsupported input extension {ext!r} for {path!r}.")


def write_sequences(path: str, sequences: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for seq in sequences:
            fh.write(seq + "\n")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_sequences(
    sequences: List[str],
    modality: Modality,
    *,
    filter_invalid: bool = True,
    canonicalize: bool = True,
    dedup: bool = True,
) -> Tuple[List[str], dict]:
    """Validate / canonicalise / dedup a list of sequences.

    Returns the cleaned list plus a small stats dict for logging.
    """
    kept: List[str] = []
    seen: set = set()
    n_invalid = 0
    n_dup = 0

    for seq in sequences:
        seq = seq.strip()
        if not seq:
            continue
        canon = modality.canonicalize(seq)
        if canon is None:
            n_invalid += 1
            if filter_invalid:
                continue
            out = seq  # keep original if not filtering
        else:
            out = canon if canonicalize else seq
        if dedup:
            key = canon if canon is not None else out
            if key in seen:
                n_dup += 1
                continue
            seen.add(key)
        kept.append(out)

    stats = {
        "input": len(sequences),
        "kept": len(kept),
        "invalid": n_invalid,
        "duplicates": n_dup,
    }
    return kept, stats


def prepare_dataset(
    input_path: str,
    modality_name: str,
    *,
    out_dir: str = "data/processed",
    text_column: Optional[str] = None,
    filter_invalid: bool = True,
    canonicalize: bool = True,
    validation_split: float = 0.05,
    seed: int = 42,
) -> Tuple[str, Optional[str], dict]:
    """Clean a raw file and split into train/eval ``.txt`` files.

    Returns ``(train_path, eval_path_or_None, stats)``.
    """
    modality = get_modality(modality_name)
    raw = read_sequences(input_path, text_column=text_column)
    cleaned, stats = clean_sequences(
        raw, modality, filter_invalid=filter_invalid, canonicalize=canonicalize
    )
    if not cleaned:
        raise ValueError(
            f"No valid sequences left after cleaning {input_path!r} "
            f"(modality={modality_name}). Stats: {stats}"
        )

    rng = random.Random(seed)
    rng.shuffle(cleaned)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    train_path = os.path.join(out_dir, f"{base}.train.txt")

    eval_path = None
    if validation_split and validation_split > 0 and len(cleaned) > 20:
        n_eval = max(1, int(len(cleaned) * validation_split))
        eval_seqs = cleaned[:n_eval]
        train_seqs = cleaned[n_eval:]
        eval_path = os.path.join(out_dir, f"{base}.eval.txt")
        write_sequences(eval_path, eval_seqs)
    else:
        train_seqs = cleaned

    write_sequences(train_path, train_seqs)
    stats["train"] = len(train_seqs)
    stats["eval"] = len(cleaned) - len(train_seqs)
    return train_path, eval_path, stats


# ---------------------------------------------------------------------------
# Tokenisation for training
# ---------------------------------------------------------------------------


def build_tokenized_datasets(
    tokenizer,
    train_file: str,
    eval_file: Optional[str],
    *,
    max_length: int = 128,
    text_column: Optional[str] = None,
    validation_split: float = 0.05,
    seed: int = 42,
):
    """Load cleaned files and return tokenised (train, eval) datasets.

    Each line becomes one example: ``bos? + tokens + eos`` (bos/eos added only
    if the tokenizer defines them).  Examples are padded dynamically by the
    data collator, not here.
    """
    from datasets import Dataset

    train_seqs = read_sequences(train_file, text_column=text_column)
    if eval_file:
        eval_seqs = read_sequences(eval_file, text_column=text_column)
    elif validation_split and validation_split > 0 and len(train_seqs) > 20:
        rng = random.Random(seed)
        rng.shuffle(train_seqs)
        n_eval = max(1, int(len(train_seqs) * validation_split))
        eval_seqs = train_seqs[:n_eval]
        train_seqs = train_seqs[n_eval:]
    else:
        eval_seqs = None

    bos = tokenizer.bos_token or ""
    eos = tokenizer.eos_token or ""

    def _wrap(seqs):
        return {"text": [f"{bos}{s}{eos}" for s in seqs]}

    def _tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_ds = Dataset.from_dict(_wrap(train_seqs))
    train_ds = train_ds.map(_tokenize, batched=True, remove_columns=["text"])

    eval_ds = None
    if eval_seqs:
        eval_ds = Dataset.from_dict(_wrap(eval_seqs))
        eval_ds = eval_ds.map(_tokenize, batched=True, remove_columns=["text"])

    return train_ds, eval_ds
