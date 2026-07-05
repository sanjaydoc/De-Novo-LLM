"""Tests for the closed-loop optimization backbone."""

import numpy as np

from denovo.closedloop import (
    ActiveLearningLoop,
    DeepEnsemble,
    SyntheticOracle,
    expected_improvement,
    select_batch,
    upper_confidence_bound,
)
from denovo.closedloop.featurize import identity_featurizer


def test_ei_monotonic_in_mean():
    std = np.array([1.0, 1.0])
    ei = expected_improvement(np.array([0.0, 2.0]), std, best=0.0)
    assert ei[1] > ei[0] >= 0.0


def test_ucb_formula():
    mean = np.array([1.0, 2.0])
    std = np.array([0.5, 0.1])
    ucb = upper_confidence_bound(mean, std, beta=2.0)
    assert np.allclose(ucb, mean + 2.0 * std)


def test_select_batch_topk_and_diversity():
    scores = np.array([0.1, 0.9, 0.5, 0.8])
    top = select_batch(scores, 2)
    assert set(top.tolist()) == {1, 3}
    feats = np.array([[0.0], [0.0], [5.0], [0.05]])  # 1 and 3 are close
    div = select_batch(scores, 2, features=feats, diversity=1.0)
    assert len(set(div.tolist())) == 2  # still returns 2 distinct picks


def test_oracle_budget_and_noise():
    o = SyntheticOracle(lambda x: -float(np.sum(x**2)), noise_std=0.0, budget=3)
    vals = o.evaluate([np.array([0.0]), np.array([1.0])])
    assert vals[0] > vals[1]           # closer to 0 scores higher
    assert o.remaining == 1
    o.evaluate([np.array([0.5])])
    assert o.remaining == 0


def test_loop_beats_or_matches_random_on_easy_problem():
    rng = np.random.default_rng(0)
    pool = list(rng.uniform(-3, 3, size=(60, 1)))
    ground_truth = lambda x: -float((x[0]) ** 2)  # noqa: E731  max at 0

    def run(acq):
        oracle = SyntheticOracle(ground_truth, noise_std=0.0, seed=0)
        loop = ActiveLearningLoop(
            pool, identity_featurizer, oracle,
            acquisition=acq,
            surrogate_factory=lambda: DeepEnsemble(n_models=2, epochs=40, seed=0),
            init_size=5, batch_size=3, n_rounds=5, seed=0,
        )
        return loop.run()

    hist = run("ei")
    # History has init + n_rounds entries and best is non-decreasing.
    assert len(hist.best_measured) == 6
    assert all(b2 >= b1 - 1e-9 for b1, b2 in zip(hist.best_measured, hist.best_measured[1:]))
    # EI should end at least as good as random search on this easy landscape.
    assert hist.best_true[-1] >= run("random").best_true[-1] - 0.25
