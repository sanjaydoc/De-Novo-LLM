"""Tests for property objectives (scoring math, no RDKit required)."""

import math

import pytest

from denovo.properties import PROPERTIES, build_objective


def test_known_properties_registered():
    assert {"logp", "qed", "mw", "tpsa"} <= set(PROPERTIES)


def test_maximize_and_minimize_transform():
    mx = build_objective("logp", mode="max")
    mn = build_objective("logp", mode="min")
    assert mx.transform(3.0) == 3.0
    assert mn.transform(3.0) == -3.0
    # maximize prefers larger values; minimize prefers smaller
    assert mx.transform(5.0) > mx.transform(1.0)
    assert mn.transform(1.0) > mn.transform(5.0)


def test_target_transform_peaks_at_target():
    obj = build_objective("qed", mode="target", target=0.8, tolerance=0.1)
    assert math.isclose(obj.transform(0.8), 1.0, rel_tol=1e-9)
    # closer to target scores higher
    assert obj.transform(0.75) > obj.transform(0.5)
    assert obj.transform(0.85) > obj.transform(1.0)


def test_validation_errors():
    with pytest.raises(KeyError):
        build_objective("nonsense")
    with pytest.raises(ValueError):
        build_objective("logp", mode="target")  # missing target
    with pytest.raises(ValueError):
        build_objective("logp", mode="bogus")


def test_describe():
    assert "logp" in build_objective("logp", mode="max").describe()
    assert "0.8" in build_objective("qed", mode="target", target=0.8).describe()
