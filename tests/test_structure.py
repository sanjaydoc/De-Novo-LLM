"""Tests for the SE(3)-equivariant flow-matching molecule generator.

The headline test is empirical equivariance: rotating/translating the input
must rotate the coordinate velocity and leave the atom-type velocity unchanged.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from denovo.structure.chem import get_bond_order, molecule_stability  # noqa: E402
from denovo.structure.data import ToyMoleculeDataset, collate  # noqa: E402
from denovo.structure.model import EquivariantFlowModel  # noqa: E402


def _random_rotation():
    a = torch.randn(3, 3)
    q, _ = torch.linalg.qr(a)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def test_model_is_e3_equivariant():
    torch.manual_seed(0)
    B, N, K = 2, 6, 5
    model = EquivariantFlowModel(K, hidden=32, n_layers=3).eval()
    node_mask = torch.ones(B, N)
    node_mask[0, 5] = 0.0  # ragged batch
    x = torch.randn(B, N, 3) * node_mask.unsqueeze(-1)
    h = torch.randn(B, N, K) * node_mask.unsqueeze(-1)
    t = torch.rand(B)

    vx, vh = model(x, h, t, node_mask)

    R = _random_rotation()
    shift = torch.randn(1, 1, 3)
    x2 = torch.einsum("bni,ij->bnj", x, R) + shift * node_mask.unsqueeze(-1)
    vx2, vh2 = model(x2, h, t, node_mask)

    # Coordinate velocity is equivariant; type velocity is invariant.
    vx_expected = torch.einsum("bni,ij->bnj", vx, R)
    assert torch.allclose(vx2, vx_expected, atol=1e-4)
    assert torch.allclose(vh2, vh, atol=1e-4)
    # Coordinate velocity lives in the zero-CoM subspace.
    com = (vx * node_mask.unsqueeze(-1)).sum(1).abs().max()
    assert com < 1e-4


def test_loss_decreases_on_toy_data():
    torch.manual_seed(0)
    ds = ToyMoleculeDataset(n_samples=64, min_atoms=4, max_atoms=8, seed=0)
    batch = collate([ds[i] for i in range(16)])
    coords, onehot, node_mask = batch
    model = EquivariantFlowModel(5, hidden=32, n_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    torch.manual_seed(0)
    first = model.compute_loss(coords, onehot, node_mask).item()
    for _ in range(30):
        opt.zero_grad()
        loss = model.compute_loss(coords, onehot, node_mask)
        loss.backward()
        opt.step()
    torch.manual_seed(0)
    last = model.compute_loss(coords, onehot, node_mask).item()
    assert last < first


def test_sample_shapes():
    model = EquivariantFlowModel(5, hidden=16, n_layers=2).eval()
    node_mask = torch.ones(3, 7)
    node_mask[0, 6] = 0.0
    coords, logits = model.sample(node_mask, n_steps=5)
    assert coords.shape == (3, 7, 3)
    assert logits.shape == (3, 7, 5)


def test_bond_order_and_stability():
    # C-C single ~1.54 A (154 pm); C#N triple ~1.16 A.
    assert get_bond_order("C", "C", 154) == 1
    assert get_bond_order("C", "N", 116) == 3
    assert get_bond_order("C", "C", 500) == 0
    # Methane-like: 1 carbon + 4 hydrogens at ~1.09 A should be atom-stable.
    coords = np.array([[0, 0, 0], [1.09, 0, 0], [-1.09, 0, 0], [0, 1.09, 0], [0, -1.09, 0]])
    syms = ["C", "H", "H", "H", "H"]
    frac, stable = molecule_stability(syms, coords)
    assert stable and frac == 1.0
