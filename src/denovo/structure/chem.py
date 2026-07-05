"""Chemistry helpers: atom vocabulary, distance-based bond inference, metrics.

Bond orders are inferred from interatomic distances using typical bond-length
tables (in picometres) with small margins -- the standard approach used by 3D
molecule generators such as EDM to turn a raw point cloud into an RDKit
molecule for validity/stability scoring. RDKit is optional: without it you can
still generate coordinates and write ``.xyz`` files.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# QM9 atom vocabulary.
ATOM_SYMBOLS: List[str] = ["H", "C", "N", "O", "F"]
ATOM_NUMBERS: Dict[str, int] = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
SYMBOL_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(ATOM_SYMBOLS)}
IDX_TO_SYMBOL: Dict[int, str] = {i: s for i, s in enumerate(ATOM_SYMBOLS)}
ALLOWED_VALENCE: Dict[str, int] = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1}

# Typical bond lengths in picometres (single / double / triple).
_BONDS1 = {
    "H": {"H": 74, "C": 109, "N": 101, "O": 96, "F": 92},
    "C": {"H": 109, "C": 154, "N": 147, "O": 143, "F": 135},
    "N": {"H": 101, "C": 147, "N": 145, "O": 140, "F": 136},
    "O": {"H": 96, "C": 143, "N": 140, "O": 148, "F": 142},
    "F": {"H": 92, "C": 135, "N": 136, "O": 142, "F": 142},
}
_BONDS2 = {
    "C": {"C": 134, "N": 129, "O": 120},
    "N": {"C": 129, "N": 125, "O": 121},
    "O": {"C": 120, "N": 121, "O": 121},
}
_BONDS3 = {
    "C": {"C": 120, "N": 116},
    "N": {"C": 116, "N": 110},
}
_MARGINS = {1: 10, 2: 5, 3: 3}  # picometres


def _lookup(table, a, b):
    if a in table and b in table[a]:
        return table[a][b]
    return None


def get_bond_order(sym_a: str, sym_b: str, distance_pm: float) -> int:
    """Return bond order 0/1/2/3 from element symbols and distance (pm)."""
    t3 = _lookup(_BONDS3, sym_a, sym_b)
    if t3 is not None and distance_pm < t3 + _MARGINS[3]:
        return 3
    t2 = _lookup(_BONDS2, sym_a, sym_b)
    if t2 is not None and distance_pm < t2 + _MARGINS[2]:
        return 2
    t1 = _lookup(_BONDS1, sym_a, sym_b)
    if t1 is not None and distance_pm < t1 + _MARGINS[1]:
        return 1
    return 0


def infer_bonds(symbols: List[str], coords: np.ndarray) -> List[Tuple[int, int, int]]:
    """List of ``(i, j, order)`` bonds inferred from a 3D structure (Angstrom)."""
    bonds = []
    n = len(symbols)
    for i in range(n):
        for j in range(i + 1, n):
            d_pm = float(np.linalg.norm(coords[i] - coords[j])) * 100.0
            order = get_bond_order(symbols[i], symbols[j], d_pm)
            if order > 0:
                bonds.append((i, j, order))
    return bonds


def molecule_stability(symbols: List[str], coords: np.ndarray) -> Tuple[float, bool]:
    """Fraction of atoms with a chemically valid valence, and whole-mol stability."""
    n = len(symbols)
    if n == 0:
        return 0.0, False
    valence = [0] * n
    for i, j, order in infer_bonds(symbols, coords):
        valence[i] += order
        valence[j] += order
    stable_atoms = sum(1 for k in range(n) if valence[k] == ALLOWED_VALENCE.get(symbols[k], -1))
    return stable_atoms / n, stable_atoms == n


def build_rdkit_mol(symbols: List[str], coords: np.ndarray):
    """Build an RDKit molecule with inferred bonds (or ``None`` if RDKit absent)."""
    try:
        from rdkit import Chem
        from rdkit.Chem import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return None
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    for s in symbols:
        rw.AddAtom(Chem.Atom(ATOM_NUMBERS[s]))
    order_map = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for i, j, order in infer_bonds(symbols, coords):
        rw.AddBond(i, j, order_map[order])
    mol = rw.GetMol()
    conf = Chem.Conformer(len(symbols))
    for k, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(k, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf)
    return mol


def mol_to_smiles(symbols: List[str], coords: np.ndarray) -> Optional[str]:
    """Sanitise the inferred molecule and return canonical SMILES, or ``None``."""
    mol = build_rdkit_mol(symbols, coords)
    if mol is None:
        return None
    from rdkit import Chem

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def types_to_symbols(type_idx) -> List[str]:
    return [IDX_TO_SYMBOL[int(i)] for i in type_idx]


def write_xyz(path: str, symbols: List[str], coords: np.ndarray, comment: str = "") -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{len(symbols)}\n{comment}\n")
        for s, (x, y, z) in zip(symbols, coords):
            fh.write(f"{s} {x:.4f} {y:.4f} {z:.4f}\n")
