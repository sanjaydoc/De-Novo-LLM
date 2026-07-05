"""Tests for YAML config loading and validation."""

import textwrap

import pytest

from denovo.config import Config, load_config


def _write(tmp_path, text):
    p = tmp_path / "cfg.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_defaults_load_with_empty_file(tmp_path):
    cfg = load_config(_write(tmp_path, ""))
    assert isinstance(cfg, Config)
    assert cfg.data.modality == "smiles"
    assert cfg.train.output_dir == "outputs/run"


def test_partial_override(tmp_path):
    cfg = load_config(_write(tmp_path, """
        data:
          modality: protein
        train:
          epochs: 7
    """))
    assert cfg.data.modality == "protein"
    assert cfg.train.epochs == 7
    # untouched fields keep defaults
    assert cfg.train.batch_size == 8


def test_resolved_model_uses_modality_default(tmp_path):
    cfg = load_config(_write(tmp_path, "data:\n  modality: selfies\n"))
    assert cfg.resolved_model() == "ncfrey/ChemGPT-4.7M"


def test_explicit_model_wins(tmp_path):
    cfg = load_config(_write(tmp_path, """
        data:
          modality: smiles
        model:
          pretrained_model: my/custom-model
    """))
    assert cfg.resolved_model() == "my/custom-model"


def test_unknown_section_raises(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "bogus:\n  x: 1\n"))


def test_unknown_key_raises(tmp_path):
    with pytest.raises(ValueError):
        load_config(_write(tmp_path, "train:\n  not_a_field: 1\n"))
