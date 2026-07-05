"""Typed configuration objects loaded from YAML.

A single YAML file fully describes a run (data, model, LoRA, training,
generation).  Everything has a sensible default tuned for a 6GB laptop GPU, so
config files can stay short and only override what matters.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    modality: str = "smiles"
    train_file: str = "data/samples/smiles_sample.txt"
    eval_file: Optional[str] = None
    #: For .csv / .parquet inputs, the column holding the sequence string.
    text_column: Optional[str] = None
    #: Fraction split off for evaluation when ``eval_file`` is not given.
    validation_split: float = 0.05
    max_length: int = 128
    #: Drop rows that fail the modality's validity check during ``prepare``.
    filter_invalid: bool = True
    #: Canonicalise sequences during ``prepare`` (dedup on the normal form).
    canonicalize: bool = True


@dataclass
class ModelConfig:
    #: Hugging Face repo id, or empty to use the modality default.
    pretrained_model: str = ""
    tokenizer_name: Optional[str] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    #: "auto", "float16", "bfloat16" or "float32".
    torch_dtype: str = "auto"


@dataclass
class LoraConfig:
    use_lora: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    #: ``None`` -> auto-detect attention projection modules for the model.
    target_modules: Optional[List[str]] = None


@dataclass
class TrainConfig:
    output_dir: str = "outputs/run"
    epochs: float = 3.0
    batch_size: int = 8
    grad_accum: int = 4
    lr: float = 5e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 20
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    gradient_checkpointing: bool = False
    fp16: bool = False
    bf16: bool = False
    #: "no", "tensorboard" -- passed through to the HF Trainer.
    report_to: str = "no"


@dataclass
class GenerateConfig:
    num_samples: int = 100
    batch_size: int = 25
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 0.95
    do_sample: bool = True
    #: Optional textual prompt / prefix to seed generation.
    prompt: str = ""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)

    # -- helpers ---------------------------------------------------------
    def resolved_model(self) -> str:
        """Model id to use, falling back to the modality default."""
        if self.model.pretrained_model:
            return self.model.pretrained_model
        from denovo.modalities import get_modality

        return get_modality(self.data.modality).default_model

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "lora": LoraConfig,
    "train": TrainConfig,
    "generate": GenerateConfig,
}


def _build_section(cls, values: Optional[Dict[str, Any]]):
    values = values or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(
            f"Unknown key(s) {sorted(unknown)} in '{cls.__name__}' config "
            f"section. Valid keys: {sorted(known)}."
        )
    return cls(**values)


def load_config(path: str) -> Config:
    """Load a :class:`Config` from a YAML file.

    Unknown top-level sections or keys raise a clear error rather than being
    silently ignored -- typos in a config are a common, costly mistake.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path!r} must be a YAML mapping.")

    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(
            f"Unknown config section(s) {sorted(unknown)}. "
            f"Valid sections: {sorted(_SECTIONS)}."
        )

    kwargs = {name: _build_section(cls, raw.get(name)) for name, cls in _SECTIONS.items()}
    return Config(**kwargs)
