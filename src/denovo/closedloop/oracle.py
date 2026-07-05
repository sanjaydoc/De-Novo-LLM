"""Simulated experimental oracle.

Real de novo campaigns query a wet lab: measurements are **noisy**, **costly**
and **budgeted**. The oracle models exactly that so the whole closed loop can
be developed and benchmarked offline, then swapped for a real assay by
implementing the same :class:`Oracle` interface.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence

import numpy as np


class Oracle:
    """Base class. Subclasses implement :meth:`_measure`.

    Tracks the number of queries against an optional budget and records the
    best (highest) *true* value seen, which the loop uses to report regret.
    """

    def __init__(self, budget: Optional[int] = None, seed: int = 0):
        self.budget = budget
        self.n_queries = 0
        self.best_value = -np.inf
        self._rng = np.random.default_rng(seed)

    # -- interface -------------------------------------------------------
    def _measure(self, candidates: Sequence[Any]) -> np.ndarray:
        """Return the *true* (noise-free) property for each candidate."""
        raise NotImplementedError

    def evaluate(self, candidates: Sequence[Any]) -> np.ndarray:
        """Query the oracle for a batch, returning noisy measurements."""
        n = len(candidates)
        if self.budget is not None and self.n_queries + n > self.budget:
            raise RuntimeError(
                f"Oracle budget exceeded: {self.n_queries}+{n} > {self.budget}."
            )
        true = np.asarray(self._measure(candidates), dtype=float)
        self.n_queries += n
        self.best_value = max(self.best_value, float(true.max()) if n else self.best_value)
        return true + self._noise(n)

    def _noise(self, n: int) -> np.ndarray:
        return np.zeros(n)

    @property
    def remaining(self) -> Optional[int]:
        return None if self.budget is None else max(0, self.budget - self.n_queries)


class SyntheticOracle(Oracle):
    """Wrap a ground-truth function over feature vectors, with Gaussian noise.

    Parameters
    ----------
    ground_truth:
        ``feature_vector -> float`` (higher is better).
    featurizer:
        Maps candidate objects to feature vectors. If ``None``, candidates are
        assumed to already be array-like feature vectors.
    noise_std:
        Standard deviation of additive measurement noise.
    """

    def __init__(
        self,
        ground_truth: Callable[[np.ndarray], float],
        featurizer: Optional[Callable[[Sequence[Any]], np.ndarray]] = None,
        noise_std: float = 0.0,
        budget: Optional[int] = None,
        seed: int = 0,
    ):
        super().__init__(budget=budget, seed=seed)
        self.ground_truth = ground_truth
        self.featurizer = featurizer
        self.noise_std = noise_std

    def _features(self, candidates: Sequence[Any]) -> np.ndarray:
        if self.featurizer is not None:
            return np.asarray(self.featurizer(candidates), dtype=float)
        return np.asarray(candidates, dtype=float)

    def _measure(self, candidates: Sequence[Any]) -> np.ndarray:
        X = self._features(candidates)
        return np.array([float(self.ground_truth(x)) for x in X])

    def _noise(self, n: int) -> np.ndarray:
        if self.noise_std <= 0:
            return np.zeros(n)
        return self._rng.normal(0.0, self.noise_std, size=n)


# ---------------------------------------------------------------------------
# A couple of standard benchmark landscapes (as maximisation problems)
# ---------------------------------------------------------------------------


def negative_ackley(x: np.ndarray, a: float = 20.0, b: float = 0.2, c: float = 2 * np.pi) -> float:
    """Ackley function negated so the global optimum (at 0) is a maximum."""
    x = np.asarray(x, dtype=float)
    d = x.size
    s1 = np.sum(x**2)
    s2 = np.sum(np.cos(c * x))
    val = -a * np.exp(-b * np.sqrt(s1 / d)) - np.exp(s2 / d) + a + np.e
    return -float(val)


def negative_rastrigin(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return -float(10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


BENCHMARKS = {
    "ackley": negative_ackley,
    "rastrigin": negative_rastrigin,
}
