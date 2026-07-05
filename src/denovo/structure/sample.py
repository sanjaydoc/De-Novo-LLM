"""Sampling from a trained equivariant flow-matching model."""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np
import torch

from denovo.structure.chem import types_to_symbols, write_xyz
from denovo.structure.data import sample_sizes
from denovo.structure.model import EquivariantFlowModel


def load_model(output_dir: str, device: Optional[str] = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(os.path.join(output_dir, "model.pt"), map_location=device)
    model = EquivariantFlowModel(
        n_atom_types=ckpt["n_atom_types"], hidden=ckpt["hidden"], n_layers=ckpt["n_layers"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def sample_molecules(
    output_dir: str,
    n_samples: int = 100,
    n_steps: int = 100,
    batch_size: int = 50,
    seed: int = 0,
    device: Optional[str] = None,
) -> List[Tuple[List[str], np.ndarray]]:
    """Generate molecules; return a list of ``(symbols, coords)`` tuples."""
    model, ckpt = load_model(output_dir, device)
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    sizes = sample_sizes(ckpt["size_histogram"], n_samples, rng)
    results: List[Tuple[List[str], np.ndarray]] = []

    for start in range(0, n_samples, batch_size):
        chunk = sizes[start : start + batch_size]
        max_n = int(chunk.max())
        node_mask = torch.zeros(len(chunk), max_n, device=device)
        for b, n in enumerate(chunk):
            node_mask[b, : int(n)] = 1.0
        coords, type_logits = model.sample(node_mask, n_steps=n_steps, generator=gen)
        types = type_logits.argmax(dim=-1).cpu().numpy()
        coords = coords.cpu().numpy()
        for b, n in enumerate(chunk):
            n = int(n)
            symbols = types_to_symbols(types[b, :n])
            results.append((symbols, coords[b, :n]))
    return results


def write_outputs(results, out_dir: str) -> Tuple[List[str], int]:
    """Write .xyz files and (if RDKit present) a SMILES list. Returns SMILES."""
    os.makedirs(out_dir, exist_ok=True)
    xyz_dir = os.path.join(out_dir, "xyz")
    os.makedirs(xyz_dir, exist_ok=True)

    from denovo.structure.chem import mol_to_smiles

    smiles: List[str] = []
    n_valid = 0
    for i, (symbols, coords) in enumerate(results):
        write_xyz(os.path.join(xyz_dir, f"mol_{i:04d}.xyz"), symbols, coords)
        smi = mol_to_smiles(symbols, coords)
        if smi:
            n_valid += 1
            smiles.append(smi)
    smi_path = os.path.join(out_dir, "generated_smiles.txt")
    with open(smi_path, "w", encoding="utf-8") as fh:
        for s in smiles:
            fh.write(s + "\n")
    return smiles, n_valid
