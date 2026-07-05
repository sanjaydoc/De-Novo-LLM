"""Acquisition functions and batch selection for Bayesian optimization.

All functions assume **maximisation** and take the surrogate's predictive
``mean`` and ``std`` arrays. They return a per-candidate acquisition score;
higher = more worth measuring next.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _norm_pdf(z):
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def _norm_cdf(z):
    from math import erf, sqrt

    # Vectorised standard-normal CDF without SciPy.
    vfunc = np.vectorize(lambda v: 0.5 * (1.0 + erf(v / sqrt(2.0))))
    return vfunc(z)


def expected_improvement(mean, std, best, xi: float = 0.01):
    """EI: expected amount by which a candidate beats the current best."""
    std = np.maximum(std, 1e-9)
    imp = mean - best - xi
    z = imp / std
    return imp * _norm_cdf(z) + std * _norm_pdf(z)


def probability_of_improvement(mean, std, best, xi: float = 0.01):
    std = np.maximum(std, 1e-9)
    z = (mean - best - xi) / std
    return _norm_cdf(z)


def upper_confidence_bound(mean, std, best=None, beta: float = 2.0):
    """UCB: mean + beta * std. ``best`` is ignored (kept for a uniform API)."""
    return mean + beta * std


def thompson(mean, std, best=None, rng: Optional[np.random.Generator] = None):
    """One Thompson draw: sample from each predictive posterior."""
    rng = rng or np.random.default_rng()
    return rng.normal(mean, np.maximum(std, 1e-9))


def greedy(mean, std=None, best=None):
    """Pure exploitation: score = predicted mean."""
    return np.asarray(mean, dtype=float)


ACQUISITIONS = {
    "ei": expected_improvement,
    "pi": probability_of_improvement,
    "ucb": upper_confidence_bound,
    "thompson": thompson,
    "greedy": greedy,
}


def select_batch(
    scores: np.ndarray,
    k: int,
    *,
    features: Optional[np.ndarray] = None,
    diversity: float = 0.0,
) -> np.ndarray:
    """Pick ``k`` indices by score, optionally penalising near-duplicates.

    With ``diversity == 0`` this is plain top-k. With ``diversity > 0`` it is a
    greedy selection that down-weights candidates close (in feature space) to
    those already picked -- a cheap stand-in for batch-BO that avoids spending a
    whole round on one region.
    """
    scores = np.asarray(scores, dtype=float)
    k = min(k, len(scores))
    if diversity <= 0 or features is None:
        return np.argsort(-scores)[:k]

    features = np.asarray(features, dtype=float)
    fmin, fmax = features.min(0), features.max(0)
    span = np.where((fmax - fmin) > 0, fmax - fmin, 1.0)
    fn = (features - fmin) / span

    chosen = []
    remaining = set(range(len(scores)))
    adj = scores.copy()
    for _ in range(k):
        i = max(remaining, key=lambda j: adj[j])
        chosen.append(i)
        remaining.discard(i)
        if not remaining:
            break
        d = np.linalg.norm(fn[list(remaining)] - fn[i], axis=1)
        for r, dist in zip(list(remaining), d):
            adj[r] -= diversity * np.exp(-dist)
    return np.array(chosen)
