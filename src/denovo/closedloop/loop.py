"""The Design-Build-Test-Learn active-learning loop.

Ties the pieces together: propose -> featurize -> surrogate -> acquisition ->
oracle -> refit. Works on a fixed candidate pool or, when a ``proposer`` is
given, grows the pool each round with freshly generated candidates -- this is
where a generative de novo model (LLM / SE(3) flow-matching) plugs in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from denovo.closedloop.acquisition import ACQUISITIONS, select_batch
from denovo.closedloop.oracle import Oracle
from denovo.closedloop.surrogate import DeepEnsemble, Surrogate


@dataclass
class LoopHistory:
    """Per-round trace of an optimization run."""

    rounds: List[int] = field(default_factory=list)
    n_queries: List[int] = field(default_factory=list)
    best_measured: List[float] = field(default_factory=list)
    best_true: List[float] = field(default_factory=list)
    round_best_true: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "n_queries": self.n_queries,
            "best_measured": self.best_measured,
            "best_true": self.best_true,
            "round_best_true": self.round_best_true,
        }


class ActiveLearningLoop:
    """Batch Bayesian-optimization / active-learning loop.

    Parameters
    ----------
    candidates:
        Initial candidate pool (arbitrary objects).
    featurizer:
        ``candidates -> 2D array`` of feature vectors.
    oracle:
        The (simulated) experiment.
    acquisition:
        One of ``ei``, ``pi``, ``ucb``, ``thompson``, ``greedy`` or
        ``random`` (baseline, no surrogate).
    surrogate_factory:
        Zero-arg callable returning a fresh :class:`Surrogate` each round.
    proposer:
        Optional ``n -> list[candidate]`` to add generated candidates per round.
    """

    def __init__(
        self,
        candidates: Sequence[Any],
        featurizer: Callable[[Sequence[Any]], np.ndarray],
        oracle: Oracle,
        *,
        acquisition: str = "ei",
        surrogate_factory: Optional[Callable[[], Surrogate]] = None,
        acq_kwargs: Optional[dict] = None,
        init_size: int = 10,
        batch_size: int = 5,
        n_rounds: int = 10,
        propose_per_round: int = 0,
        proposer: Optional[Callable[[int], List[Any]]] = None,
        diversity: float = 0.0,
        seed: int = 0,
    ):
        self.pool: List[Any] = list(candidates)
        self.featurizer = featurizer
        self.oracle = oracle
        self.acquisition = acquisition.lower()
        self.surrogate_factory = surrogate_factory or (lambda: DeepEnsemble(seed=seed))
        self.acq_kwargs = acq_kwargs or {}
        self.init_size = init_size
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.propose_per_round = propose_per_round
        self.proposer = proposer
        self.diversity = diversity
        self.rng = np.random.default_rng(seed)

        if self.acquisition not in ACQUISITIONS and self.acquisition != "random":
            raise ValueError(
                f"Unknown acquisition {acquisition!r}. "
                f"Choose from {sorted(ACQUISITIONS)} or 'random'."
            )

        self._features = np.asarray(self.featurizer(self.pool), dtype=float)
        self._labeled: dict = {}   # index -> measured value
        self._best_true = -np.inf

    # -- helpers ---------------------------------------------------------
    def _unlabeled(self) -> np.ndarray:
        return np.array([i for i in range(len(self.pool)) if i not in self._labeled], dtype=int)

    def _query(self, idxs: Sequence[int]) -> None:
        cands = [self.pool[i] for i in idxs]
        measured = self.oracle.evaluate(cands)
        for i, y in zip(idxs, measured):
            self._labeled[int(i)] = float(y)
        self._best_true = max(self._best_true, self.oracle.best_value)

    def _grow_pool(self) -> None:
        if not (self.proposer and self.propose_per_round > 0):
            return
        new = self.proposer(self.propose_per_round)
        if not new:
            return
        new_feats = np.asarray(self.featurizer(new), dtype=float)
        self.pool.extend(new)
        self._features = np.vstack([self._features, new_feats])

    def _best_measured(self) -> float:
        return max(self._labeled.values()) if self._labeled else -np.inf

    # -- run -------------------------------------------------------------
    def run(self, verbose: bool = False) -> LoopHistory:
        hist = LoopHistory()

        # Seed with an initial random design.
        unl = self._unlabeled()
        n0 = min(self.init_size, len(unl))
        seed_idx = self.rng.choice(unl, size=n0, replace=False)
        self._query(seed_idx)

        hist.rounds.append(0)
        hist.n_queries.append(self.oracle.n_queries)
        hist.best_measured.append(self._best_measured())
        hist.best_true.append(self._best_true)
        hist.round_best_true.append(self._best_true)
        if verbose:
            print(f"[init] n={self.oracle.n_queries} best_true={self._best_true:.4f}")

        for r in range(1, self.n_rounds + 1):
            self._grow_pool()
            unl = self._unlabeled()
            if len(unl) == 0:
                break
            if self.oracle.remaining == 0:
                break

            if self.acquisition == "random":
                pick = self.rng.choice(unl, size=min(self.batch_size, len(unl)), replace=False)
            else:
                surrogate = self.surrogate_factory()
                lab_idx = np.array(sorted(self._labeled))
                X_lab = self._features[lab_idx]
                y_lab = np.array([self._labeled[i] for i in lab_idx])
                surrogate.fit(X_lab, y_lab)

                mean, std = surrogate.predict(self._features[unl])
                acq_fn = ACQUISITIONS[self.acquisition]
                kwargs = dict(self.acq_kwargs)
                if self.acquisition in ("ei", "pi"):
                    kwargs["best"] = self._best_measured()
                if self.acquisition == "thompson":
                    kwargs["rng"] = self.rng
                scores = acq_fn(mean, std, **kwargs)

                k = min(self.batch_size, len(unl))
                if self.oracle.remaining is not None:
                    k = min(k, self.oracle.remaining)
                sel = select_batch(
                    scores, k, features=self._features[unl], diversity=self.diversity
                )
                pick = unl[sel]

            self._query(pick)

            hist.rounds.append(r)
            hist.n_queries.append(self.oracle.n_queries)
            hist.best_measured.append(self._best_measured())
            hist.best_true.append(self._best_true)
            round_true = max(
                self.oracle._measure([self.pool[i] for i in pick])
            ) if len(pick) else -np.inf
            hist.round_best_true.append(float(round_true))
            if verbose:
                print(f"[round {r}] n={self.oracle.n_queries} best_true={self._best_true:.4f}")

        return hist
