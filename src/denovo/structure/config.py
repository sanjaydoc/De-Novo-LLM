"""Configuration for the SE(3)-equivariant flow-matching molecule generator."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class StructData:
    #: "toy" (offline) or "sdf".
    dataset: str = "toy"
    sdf_path: Optional[str] = None
    n_toy: int = 512
    min_atoms: int = 4
    max_atoms: int = 16


@dataclass
class StructModel:
    hidden: int = 128
    n_layers: int = 4


@dataclass
class StructTrain:
    output_dir: str = "outputs/mol_flow"
    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-10
    grad_clip: float = 1.0
    log_every: int = 20
    seed: int = 42
    fp16: bool = False


@dataclass
class StructSample:
    n_samples: int = 100
    n_steps: int = 100
    batch_size: int = 50


@dataclass
class StructureConfig:
    data: StructData = field(default_factory=StructData)
    model: StructModel = field(default_factory=StructModel)
    train: StructTrain = field(default_factory=StructTrain)
    sample: StructSample = field(default_factory=StructSample)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "data": StructData,
    "model": StructModel,
    "train": StructTrain,
    "sample": StructSample,
}


def _build(cls, values):
    values = values or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown key(s) {sorted(unknown)} in '{cls.__name__}'. Valid: {sorted(known)}.")
    return cls(**values)


def load_structure_config(path: str) -> StructureConfig:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    unknown = set(raw) - set(_SECTIONS)
    if unknown:
        raise ValueError(f"Unknown section(s) {sorted(unknown)}. Valid: {sorted(_SECTIONS)}.")
    return StructureConfig(**{name: _build(cls, raw.get(name)) for name, cls in _SECTIONS.items()})
