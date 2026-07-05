"""Training loop for the equivariant flow-matching molecule generator."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from denovo.structure.chem import ATOM_SYMBOLS
from denovo.structure.config import StructureConfig
from denovo.structure.data import (
    SDFDataset,
    ToyMoleculeDataset,
    atom_count_histogram,
    collate,
)
from denovo.structure.model import EquivariantFlowModel


def build_dataset(cfg: StructureConfig):
    if cfg.data.dataset == "toy":
        return ToyMoleculeDataset(
            n_samples=cfg.data.n_toy,
            min_atoms=cfg.data.min_atoms,
            max_atoms=cfg.data.max_atoms,
            seed=cfg.train.seed,
        )
    if cfg.data.dataset == "sdf":
        if not cfg.data.sdf_path:
            raise ValueError("data.sdf_path is required when data.dataset == 'sdf'.")
        return SDFDataset(cfg.data.sdf_path, max_atoms=cfg.data.max_atoms)
    raise ValueError(f"Unknown dataset {cfg.data.dataset!r} (use 'toy' or 'sdf').")


def train(cfg: StructureConfig) -> str:
    torch.manual_seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = build_dataset(cfg)
    histogram = atom_count_histogram(dataset)
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, len(ATOM_SYMBOLS)),
    )

    model = EquivariantFlowModel(
        n_atom_types=len(ATOM_SYMBOLS), hidden=cfg.model.hidden, n_layers=cfg.model.n_layers
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    print(f"Model: {model.num_parameters():,} params | device={device} | "
          f"{len(dataset)} molecules")

    os.makedirs(cfg.train.output_dir, exist_ok=True)
    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        running = 0.0
        for coords, onehot, node_mask in loader:
            coords, onehot, node_mask = coords.to(device), onehot.to(device), node_mask.to(device)
            loss = model.compute_loss(coords, onehot, node_mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            running += loss.item()
            step += 1
            if step % cfg.train.log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        if len(loader):
            print(f"[epoch {epoch}] mean loss {running / len(loader):.4f}")

    ckpt = os.path.join(cfg.train.output_dir, "model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hidden": cfg.model.hidden,
            "n_layers": cfg.model.n_layers,
            "n_atom_types": len(ATOM_SYMBOLS),
            "atom_symbols": ATOM_SYMBOLS,
            "size_histogram": histogram,
        },
        ckpt,
    )
    with open(os.path.join(cfg.train.output_dir, "structure_config.json"), "w") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)
    print(f"Saved model to {ckpt}")
    return cfg.train.output_dir
