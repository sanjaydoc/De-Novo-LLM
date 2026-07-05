"""Featurizers: turn candidate objects into fixed-length vectors for the surrogate.

* :func:`identity_featurizer` -- candidates are already feature vectors.
* :func:`morgan_featurizer` -- SMILES -> Morgan/ECFP fingerprint (needs RDKit).
* :func:`descriptor_featurizer` -- SMILES -> a few interpretable RDKit descriptors.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

import numpy as np


def identity_featurizer(candidates: Sequence) -> np.ndarray:
    return np.asarray(candidates, dtype=float)


def morgan_featurizer(n_bits: int = 1024, radius: int = 2) -> Callable[[Sequence[str]], np.ndarray]:
    """Return a SMILES -> ECFP bit-vector featurizer (requires RDKit)."""

    def _feat(smiles: Sequence[str]) -> np.ndarray:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        out = np.zeros((len(smiles), n_bits), dtype=np.float32)
        for i, smi in enumerate(smiles):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            for b in fp.GetOnBits():
                out[i, b] = 1.0
        return out

    return _feat


def descriptor_featurizer() -> Callable[[Sequence[str]], np.ndarray]:
    """Return a SMILES -> [MW, logP, TPSA, HBD, HBA, rings] featurizer (RDKit)."""

    def _feat(smiles: Sequence[str]) -> np.ndarray:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski

        rows: List[List[float]] = []
        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rows.append([0.0] * 6)
                continue
            rows.append([
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                Descriptors.TPSA(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Descriptors.RingCount(mol),
            ])
        return np.asarray(rows, dtype=float)

    return _feat
