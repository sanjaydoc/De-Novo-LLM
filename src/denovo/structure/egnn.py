"""E(3)-equivariant graph neural network (EGNN), dense/masked implementation.

Follows Satorras, Hoogeboom & Welling, "E(n) Equivariant Graph Neural
Networks" (2021). Invariant node features ``h`` and equivariant coordinates
``x`` are updated jointly:

    m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2)
    x_i  <- x_i + sum_j (x_i - x_j) * phi_x(m_ij)     # equivariant
    h_i  <- h_i + phi_h(h_i, sum_j m_ij)              # invariant

Because coordinate updates depend only on **differences** and invariant edge
scalars, the map is equivariant to rotations/reflections and invariant to
translations -- i.e. E(3)-equivariant. The implementation is dense (B, N, N)
with masks, so it needs no torch-scatter and runs comfortably for the small
graphs typical of drug-like molecules.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EGNNLayer(nn.Module):
    def __init__(self, hidden: int, edge_extra: int = 0, act=nn.SiLU):
        super().__init__()
        # edge input: h_i, h_j, dist^2 (+ optional extra edge features)
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1 + edge_extra, hidden),
            act(),
            nn.Linear(hidden, hidden),
            act(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            act(),
            nn.Linear(hidden, hidden),
        )
        # Coordinate MLP ends in a single scalar gate per edge.
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            act(),
            nn.Linear(hidden, 1, bias=False),
        )
        # Small init on the last coord layer keeps early training stable.
        nn.init.xavier_uniform_(self.coord_mlp[-1].weight, gain=0.01)

    def forward(self, h, x, node_mask, edge_mask, edge_extra=None):
        B, N, H = h.shape
        h_i = h.unsqueeze(2).expand(B, N, N, H)
        h_j = h.unsqueeze(1).expand(B, N, N, H)

        coord_diff = x.unsqueeze(2) - x.unsqueeze(1)          # (B,N,N,3)
        dist2 = (coord_diff ** 2).sum(-1, keepdim=True)        # (B,N,N,1)

        parts = [h_i, h_j, dist2]
        if edge_extra is not None:
            parts.append(edge_extra)
        edge_in = torch.cat(parts, dim=-1)
        m = self.edge_mlp(edge_in) * edge_mask.unsqueeze(-1)   # (B,N,N,H)

        # --- invariant node update ---
        agg = m.sum(dim=2)                                     # (B,N,H)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        h = h * node_mask.unsqueeze(-1)

        # --- equivariant coordinate update ---
        # Normalise the direction vector so updates don't blow up with distance.
        coeff = self.coord_mlp(m)                              # (B,N,N,1)
        norm = torch.sqrt(dist2 + 1e-8) + 1.0
        trans = (coord_diff / norm) * coeff * edge_mask.unsqueeze(-1)
        n_neighbors = edge_mask.sum(dim=2, keepdim=True).clamp(min=1)  # (B,N,1)
        x = x + trans.sum(dim=2) / n_neighbors
        x = x * node_mask.unsqueeze(-1)
        return h, x


def build_masks(node_mask: torch.Tensor):
    """From (B, N) node mask build the (B, N, N) edge mask (no self-loops)."""
    B, N = node_mask.shape
    edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)      # (B,N,N)
    eye = torch.eye(N, device=node_mask.device).unsqueeze(0)
    edge_mask = edge_mask * (1.0 - eye)
    return edge_mask
