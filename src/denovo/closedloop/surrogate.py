"""Uncertainty-aware surrogate models.

Bayesian optimization needs a *predictive distribution*, not just a point
estimate -- the uncertainty is what drives exploration. Two surrogates are
provided:

* :class:`DeepEnsemble` -- an ensemble of small MLPs (pure PyTorch). Epistemic
  uncertainty comes from disagreement across members. Robust default for the
  sparse/noisy regime and has no extra dependencies.
* :class:`GPSurrogate` -- a Gaussian process (scikit-learn), the classic
  low-data BO surrogate. Used when scikit-learn is installed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class Surrogate:
    """Interface: ``fit(X, y)`` then ``predict(X) -> (mean, std)``."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Surrogate":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class DeepEnsemble(Surrogate):
    """Ensemble of small MLP regressors with input/target standardisation."""

    def __init__(
        self,
        n_models: int = 5,
        hidden: int = 64,
        depth: int = 2,
        epochs: int = 200,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
        dropout: float = 0.0,
        seed: int = 0,
    ):
        self.n_models = n_models
        self.hidden = hidden
        self.depth = depth
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.seed = seed
        self.models = []
        self._x_mean = self._x_std = None
        self._y_mean = self._y_std = 0.0, 1.0

    def _build(self, in_dim: int):
        import torch.nn as nn

        layers = []
        d = in_dim
        for _ in range(self.depth):
            layers += [nn.Linear(d, self.hidden), nn.SiLU()]
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            d = self.hidden
        layers.append(nn.Linear(d, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepEnsemble":
        import torch

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        self._x_mean = X.mean(0, keepdims=True)
        self._x_std = X.std(0, keepdims=True) + 1e-6
        self._y_mean = float(y.mean())
        self._y_std = float(y.std() + 1e-6)

        Xn = torch.tensor((X - self._x_mean) / self._x_std)
        yn = torch.tensor((y - self._y_mean) / self._y_std).unsqueeze(1)

        self.models = []
        for m in range(self.n_models):
            torch.manual_seed(self.seed + m)
            net = self._build(X.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            loss_fn = torch.nn.MSELoss()
            # Bootstrap resample so ensemble members differ meaningfully.
            g = torch.Generator().manual_seed(self.seed + m)
            idx = torch.randint(0, len(Xn), (len(Xn),), generator=g)
            xb, yb = Xn[idx], yn[idx]
            net.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
            net.eval()
            self.models.append(net)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        import torch

        if not self.models:
            raise RuntimeError("DeepEnsemble.predict called before fit().")
        X = np.asarray(X, dtype=np.float32)
        Xn = torch.tensor((X - self._x_mean) / self._x_std)
        with torch.no_grad():
            preds = np.stack([m(Xn).squeeze(1).numpy() for m in self.models], axis=0)
        mean = preds.mean(0) * self._y_std + self._y_mean
        # Epistemic std across members (in original target units).
        std = preds.std(0) * self._y_std
        return mean, np.maximum(std, 1e-6)


class GPSurrogate(Surrogate):
    """Gaussian-process surrogate (scikit-learn), the classic low-data choice."""

    def __init__(self, seed: int = 0, alpha: float = 1e-6):
        self.seed = seed
        self.alpha = alpha
        self.gp = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GPSurrogate":
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "GPSurrogate needs scikit-learn: pip install scikit-learn"
            ) from exc
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-2)
        self.gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.alpha, normalize_y=True, random_state=self.seed
        )
        self.gp.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float).reshape(-1))
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.gp is None:
            raise RuntimeError("GPSurrogate.predict called before fit().")
        mean, std = self.gp.predict(np.asarray(X, dtype=float), return_std=True)
        return mean, np.maximum(std, 1e-6)
