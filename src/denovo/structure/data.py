"""3D molecule datasets and batching for flow-matching training.

* :class:`ToyMoleculeDataset` -- random point clouds for offline smoke
  tests / CI (not chemically meaningful).
* :class:`SDFDataset` -- real molecules with 3D conformers from an ``.sdf``
  file (e.g. QM9), via RDKit.

``collate`` pads a batch to its largest molecule and returns
``(coords, onehot_types, node_mask)`` plus the per-molecule atom counts (used to
sample molecule sizes at generation time).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from denovo.structure.chem import ATOM_SYMBOLS, SYMBOL_TO_IDX


class ToyMoleculeDataset(Dataset):
    """Random small molecules for exercising the pipeline offline."""

    def __init__(self, n_samples: int = 512, min_atoms: int = 4, max_atoms: int = 12, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.items = []
        for _ in range(n_samples):
            n = int(self.rng.integers(min_atoms, max_atoms + 1))
            coords = self.rng.normal(0, 1.3, size=(n, 3)).astype(np.float32)
            # Bias toward carbon so bond inference is not pure noise.
            probs = np.array([0.3, 0.4, 0.1, 0.15, 0.05])
            types = self.rng.choice(len(ATOM_SYMBOLS), size=n, p=probs)
            self.items.append((coords, types.astype(np.int64)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class SDFDataset(Dataset):
    """Molecules with 3D conformers read from an SDF file (requires RDKit)."""

    def __init__(self, path: str, max_atoms: int = 64):
        try:
            from rdkit import Chem
            from rdkit.Chem import RDLogger

            RDLogger.DisableLog("rdApp.*")
        except Exception as exc:  # pragma: no cover
            raise ImportError("SDFDataset requires RDKit: pip install rdkit") from exc

        self.items = []
        supplier = Chem.SDMolSupplier(path, removeHs=False)
        for mol in supplier:
            if mol is None or mol.GetNumConformers() == 0:
                continue
            syms = [a.GetSymbol() for a in mol.GetAtoms()]
            if any(s not in SYMBOL_TO_IDX for s in syms) or len(syms) > max_atoms:
                continue
            conf = mol.GetConformer()
            coords = np.array(
                [[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                 for i in range(len(syms))],
                dtype=np.float32,
            )
            types = np.array([SYMBOL_TO_IDX[s] for s in syms], dtype=np.int64)
            self.items.append((coords, types))
        if not self.items:
            raise ValueError(f"No usable molecules parsed from {path!r}.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def atom_count_histogram(dataset) -> dict:
    """Empirical distribution of molecule sizes, for sampling molecule sizes."""
    counts: dict = {}
    for coords, _ in dataset:
        n = len(coords)
        counts[n] = counts.get(n, 0) + 1
    return counts


def sample_sizes(histogram: dict, k: int, rng: np.random.Generator) -> np.ndarray:
    sizes = np.array(sorted(histogram))
    probs = np.array([histogram[s] for s in sizes], dtype=float)
    probs /= probs.sum()
    return rng.choice(sizes, size=k, p=probs)


def collate(batch: List[Tuple[np.ndarray, np.ndarray]], n_atom_types: int = len(ATOM_SYMBOLS)):
    """Pad a batch to its largest molecule; return tensors + node mask."""
    max_n = max(len(c) for c, _ in batch)
    B = len(batch)
    coords = torch.zeros(B, max_n, 3)
    onehot = torch.zeros(B, max_n, n_atom_types)
    node_mask = torch.zeros(B, max_n)
    for b, (c, types) in enumerate(batch):
        n = len(c)
        coords[b, :n] = torch.from_numpy(np.asarray(c, dtype=np.float32))
        onehot[b, torch.arange(n), torch.from_numpy(np.asarray(types))] = 1.0
        node_mask[b, :n] = 1.0
    # Centre coordinates (training assumes zero-CoM data).
    n_per = node_mask.sum(1, keepdim=True).clamp(min=1).unsqueeze(-1)
    mean = (coords * node_mask.unsqueeze(-1)).sum(1, keepdim=True) / n_per
    coords = (coords - mean) * node_mask.unsqueeze(-1)
    return coords, onehot, node_mask
