"""Closed-loop de novo optimization backbone.

The Design-Build-Test-Learn (DBTL) loop that connects (simulated) experiments
to ML models:

    generative proposer -> candidate pool
                             |
             surrogate (mean + uncertainty)
                             |
             acquisition (EI / UCB / PI / Thompson)
                             |
             oracle / experiment  (noisy, budgeted)
                             |
             labelled data  --> refit surrogate --> repeat

Everything here is modality-agnostic: candidates are arbitrary objects turned
into feature vectors by a ``featurizer``. Small models + batch active learning
make it viable on sparse, noisy, high-cost data.
"""

from denovo.closedloop.acquisition import (  # noqa: F401
    ACQUISITIONS,
    expected_improvement,
    probability_of_improvement,
    select_batch,
    upper_confidence_bound,
)
from denovo.closedloop.loop import ActiveLearningLoop, LoopHistory  # noqa: F401
from denovo.closedloop.oracle import Oracle, SyntheticOracle  # noqa: F401
from denovo.closedloop.surrogate import DeepEnsemble, GPSurrogate, Surrogate  # noqa: F401

__all__ = [
    "Oracle",
    "SyntheticOracle",
    "Surrogate",
    "DeepEnsemble",
    "GPSurrogate",
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "select_batch",
    "ACQUISITIONS",
    "ActiveLearningLoop",
    "LoopHistory",
]
