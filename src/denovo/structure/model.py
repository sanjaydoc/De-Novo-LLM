"""SE(3)/E(3)-equivariant flow-matching model for 3D molecule generation.

Wraps the EGNN with an atom-feature embedding, a time embedding, and two
output heads producing the flow velocity for coordinates (equivariant) and for
atom-type features (invariant).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from denovo.structure.egnn import EGNNLayer, build_masks
from denovo.structure.flow import (
    masked_mse,
    remove_mean,
    sample_com_free_noise,
)


class SinusoidalTime(nn.Module):
    """Standard sinusoidal embedding of the scalar flow time t in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device) * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
        )
        args = t.view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:  # pad if dim is odd
            emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=-1)
        return emb


class EquivariantFlowModel(nn.Module):
    """EGNN-based velocity field for coordinate + atom-type flow matching.

    Parameters
    ----------
    n_atom_types:
        Number of atom categories (e.g. 5 for H,C,N,O,F).
    hidden:
        Hidden width of the EGNN.
    n_layers:
        Number of EGNN message-passing layers.
    """

    def __init__(self, n_atom_types: int, hidden: int = 128, n_layers: int = 4):
        super().__init__()
        self.n_atom_types = n_atom_types
        self.hidden = hidden

        self.embed = nn.Linear(n_atom_types, hidden)
        self.time_embed = nn.Sequential(
            SinusoidalTime(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.layers = nn.ModuleList([EGNNLayer(hidden) for _ in range(n_layers)])
        self.type_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, n_atom_types)
        )

    def forward(
        self, x: torch.Tensor, h: torch.Tensor, t: torch.Tensor, node_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(v_x, v_h)`` -- coordinate and atom-type velocities.

        ``v_x`` is E(3)-equivariant and zero-CoM; ``v_h`` is E(3)-invariant.
        """
        edge_mask = build_masks(node_mask)
        feat = self.embed(h) + self.time_embed(t).unsqueeze(1)   # (B,N,hidden)
        feat = feat * node_mask.unsqueeze(-1)

        x_in = x
        coords = x
        for layer in self.layers:
            feat, coords = layer(feat, coords, node_mask, edge_mask)

        v_x = remove_mean(coords - x_in, node_mask)              # equivariant, zero-CoM
        v_h = self.type_head(feat) * node_mask.unsqueeze(-1)     # invariant
        return v_x, v_h

    # -- flow-matching training loss ------------------------------------
    def compute_loss(self, x1: torch.Tensor, h1: torch.Tensor, node_mask: torch.Tensor,
                     generator=None) -> torch.Tensor:
        """Conditional flow-matching loss for a batch of molecules.

        ``x1`` (B,N,3) coords, ``h1`` (B,N,K) one-hot atom types, ``node_mask``
        (B,N).
        """
        B, N, _ = x1.shape
        device = x1.device
        x1 = remove_mean(x1, node_mask)

        # Priors: zero-CoM Gaussian for coords, standard Gaussian for types.
        z_x = sample_com_free_noise((B, N, 3), node_mask, generator=generator, device=device)
        z_h = torch.randn(B, N, self.n_atom_types, generator=generator, device=device)
        z_h = z_h * node_mask.unsqueeze(-1)

        t = torch.rand(B, 1, generator=generator, device=device)
        t_b = t.view(B, 1, 1)
        x_t = (1 - t_b) * z_x + t_b * x1
        h_t = (1 - t_b) * z_h + t_b * h1

        u_x = remove_mean(x1 - z_x, node_mask)
        u_h = (h1 - z_h) * node_mask.unsqueeze(-1)

        v_x, v_h = self.forward(x_t, h_t, t.view(B), node_mask)
        return masked_mse(v_x, u_x, node_mask) + masked_mse(v_h, u_h, node_mask)

    # -- sampling via ODE integration -----------------------------------
    @torch.no_grad()
    def sample(self, node_mask: torch.Tensor, n_steps: int = 100, generator=None):
        """Integrate the flow from prior (t=0) to data (t=1) with Euler steps.

        Returns ``(coords, type_logits)`` where ``coords`` is (B,N,3) and
        ``type_logits`` is (B,N,K); apply argmax over the last dim for atom types.
        """
        device = node_mask.device
        B, N = node_mask.shape
        x = sample_com_free_noise((B, N, 3), node_mask, generator=generator, device=device)
        h = torch.randn(B, N, self.n_atom_types, generator=generator, device=device)
        h = h * node_mask.unsqueeze(-1)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v_x, v_h = self.forward(x, h, t, node_mask)
            x = remove_mean(x + dt * v_x, node_mask)
            h = h + dt * v_h
        return x, h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
