"""Tests for the NVIDIA NIM client (no network / no API key required)."""

import pytest

from denovo.nim import NIM_MODELS, NIMClient


def test_nim_models_listed():
    assert {"molmim", "esmfold", "evo2"} <= set(NIM_MODELS)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NGC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        NIMClient()


def test_client_reads_env_key(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    client = NIMClient()
    assert client.api_key == "nvapi-test"
    assert client._headers()["Authorization"] == "Bearer nvapi-test"
    assert client.base.endswith("/biology")
