"""Conditional flow-matching utilities for 3D point clouds.

The generative model learns a velocity field that transports a simple prior to
the data distribution. For 3D molecules the coordinate part must live in the
**zero centre-of-mass (CoM) subspace** so the whole thing is translation
invariant; the network's coordinate output is E(3)-equivariant, giving an
SE(3)/E(3)-equivariant generative model.

Interpolation (independent-coupling / OT path, sigma_min -> 0)::

    z ~ prior,  x1 = data
    x_t = (1 - t) * z + t * x1
    target velocity  u = x1 - z          (constant along the path)

The model regresses ``u``; sampling integrates ``dx/dt = v_theta(x_t, t)`` from
t=0 (prior) to t=1 (data).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def remove_mean(x: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Subtract the per-molecule centre of mass over *real* atoms.

    Parameters
    ----------
    x:          (B, N, 3) coordinates.
    node_mask:  (B, N) or (B, N, 1) with 1 for real atoms.
    """
    if node_mask.dim() == 2:
        node_mask = node_mask.unsqueeze(-1)
    n = node_mask.sum(dim=1, keepdim=True).clamp(min=1)      # (B,1,1)
    mean = (x * node_mask).sum(dim=1, keepdim=True) / n       # (B,1,3)
    return (x - mean) * node_mask


def sample_com_free_noise(
    shape: Tuple[int, int, int], node_mask: torch.Tensor, generator=None, device=None
) -> torch.Tensor:
    """Gaussian noise projected onto the zero-CoM subspace and masked."""
    device = device or node_mask.device
    x = torch.randn(*shape, generator=generator, device=device)
    return remove_mean(x, node_mask)


def masked_mse(pred: torch.Tensor, target: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error averaged over real atoms and feature dims."""
    if node_mask.dim() == 2:
        node_mask = node_mask.unsqueeze(-1)
    diff2 = ((pred - target) ** 2) * node_mask
    denom = node_mask.sum().clamp(min=1) * pred.shape[-1]
    return diff2.sum() / denom


def assert_mean_zero(x: torch.Tensor, node_mask: torch.Tensor, atol: float = 1e-4) -> bool:
    if node_mask.dim() == 2:
        node_mask = node_mask.unsqueeze(-1)
    n = node_mask.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (x * node_mask).sum(dim=1, keepdim=True) / n
    return bool(mean.abs().max() < atol)
