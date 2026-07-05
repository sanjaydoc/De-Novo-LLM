"""Tests for the modality registry (no ML dependencies required)."""

import pytest

from denovo.modalities import get_modality, list_modalities


def test_registry_has_core_modalities():
    names = set(list_modalities())
    assert {"smiles", "selfies", "protein", "dna", "rna"} <= names


def test_aliases_resolve():
    assert get_modality("molecule").name == "smiles"
    assert get_modality("peptide").name == "protein"


def test_unknown_modality_raises():
    with pytest.raises(KeyError):
        get_modality("unobtanium")


def test_protein_alphabet_validation():
    m = get_modality("protein")
    assert m.validate("MKTAYIAKQR")           # valid amino acids
    assert not m.validate("MKTZ123")          # digits are not residues
    assert m.canonicalize(" mktay ") == "MKTAY"  # trims + uppercases


def test_dna_rna_alphabets():
    dna = get_modality("dna")
    rna = get_modality("rna")
    assert dna.validate("ACGTACGT")
    assert not dna.validate("ACGU")           # U is RNA, not DNA
    assert rna.validate("ACGUACGU")
    assert not rna.validate("ACGT")           # T is DNA, not RNA


def test_smiles_loose_fallback_or_rdkit():
    """Without RDKit we fall back to a char check; with it, real validity.

    Either way a clearly-bogus string must be rejected and ethanol accepted.
    """
    m = get_modality("smiles")
    assert m.validate("CCO")                  # ethanol
    assert not m.validate("this is not smiles")  # spaces -> invalid
