"""Tests for scaffold-constrained generation (structure/logic without a model)."""

import pytest

from denovo.scaffold import ScaffoldResult, contains, parse_pattern


def test_match_rate_and_summary():
    res = ScaffoldResult(matches=[("c1ccccc1O", None)], n_pool=100, n_valid=80, n_match=1)
    assert res.match_rate == 1 / 80
    text = res.summary("c1ccccc1")
    assert "scaffold" in text and "c1ccccc1" in text


def test_rdkit_paths_graceful_without_install():
    """Without RDKit, contains() is False and parse_pattern raises ImportError."""
    from denovo import scaffold

    if scaffold._rdkit_chem() is None:
        assert contains("c1ccccc1", object()) is False
        with pytest.raises(ImportError):
            parse_pattern("c1ccccc1")
    else:
        # With RDKit available, benzene contains a benzene scaffold; ethanol does not.
        benzene = parse_pattern("c1ccccc1")
        assert contains("c1ccccc1O", benzene) is True
        assert contains("CCO", benzene) is False
