"""Metrics for generated 3D molecules.

Reports the EDM-style set:

* **atom stability** -- fraction of atoms with a valid valence.
* **molecule stability** -- fraction of molecules whose every atom is valid.
* **validity** -- fraction that sanitise in RDKit (needs RDKit).
* **uniqueness / novelty** -- over canonical SMILES of the valid molecules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from denovo.structure.chem import molecule_stability, mol_to_smiles


@dataclass
class MoleculeMetrics:
    n: int
    atom_stability: float
    mol_stability: float
    validity: float
    uniqueness: float
    novelty: float

    def as_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        return (
            f"  molecules         : {self.n}\n"
            f"  atom stability    : {self.atom_stability:.1%}\n"
            f"  mol stability     : {self.mol_stability:.1%}\n"
            f"  validity          : {self.validity:.1%}\n"
            f"  uniqueness        : {self.uniqueness:.1%}\n"
            f"  novelty           : {self.novelty:.1%}"
        )


def evaluate_molecules(
    molecules: Sequence[Tuple[List[str], np.ndarray]],
    training_smiles: Optional[Sequence[str]] = None,
) -> MoleculeMetrics:
    n = len(molecules)
    if n == 0:
        return MoleculeMetrics(0, 0, 0, 0, 0, 0)

    atom_frac = []
    mol_stable = 0
    smiles = []
    for symbols, coords in molecules:
        af, ms = molecule_stability(symbols, np.asarray(coords))
        atom_frac.append(af)
        mol_stable += int(ms)
        smi = mol_to_smiles(symbols, np.asarray(coords))
        if smi:
            smiles.append(smi)

    validity = len(smiles) / n
    unique = set(smiles)
    uniqueness = (len(unique) / len(smiles)) if smiles else 0.0

    train_set = set(training_smiles or [])
    novel = [s for s in unique if s not in train_set] if train_set else list(unique)
    novelty = (len(novel) / len(unique)) if unique else 0.0

    return MoleculeMetrics(
        n=n,
        atom_stability=float(np.mean(atom_frac)),
        mol_stability=mol_stable / n,
        validity=validity,
        uniqueness=uniqueness,
        novelty=novelty,
    )
