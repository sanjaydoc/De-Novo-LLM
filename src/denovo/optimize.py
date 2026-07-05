"""Bayesian hyperparameter optimization with Optuna.

Two search modes:

* ``sampling`` -- keep a *trained* model fixed and optimise decoding
  hyperparameters (temperature, top-p, top-k) to maximise a de novo quality
  score. Cheap (no retraining) and the recommended starting point.
* ``training`` -- optimise fine-tuning hyperparameters (learning rate, LoRA
  rank, batch/accumulation, epochs). Each trial retrains, so this is
  expensive; use a small budget.

Optuna's default sampler is TPE (Tree-structured Parzen Estimator), a
Bayesian optimization method that models P(params | score) and proposes
promising points -- far more sample-efficient than grid/random search, which
matters a lot when every trial costs GPU minutes.

The scalar objective combines the standard metrics::

    score = w_validity * validity
          + w_uniqueness * uniqueness
          + w_novelty * novelty
          + w_diversity * diversity

Weights are configurable; the defaults reward valid, novel, diverse output.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from denovo.config import Config
from denovo.evaluate import GenerationMetrics, evaluate_sequences


@dataclass
class ObjectiveWeights:
    validity: float = 1.0
    uniqueness: float = 0.5
    novelty: float = 1.0
    diversity: float = 0.5

    def score(self, m: GenerationMetrics) -> float:
        div = m.diversity if m.diversity is not None else 0.0
        return (
            self.validity * m.validity
            + self.uniqueness * m.uniqueness
            + self.novelty * m.novelty
            + self.diversity * div
        )


@dataclass
class SearchSpace:
    """Ranges for the tunable hyperparameters (mode-dependent)."""

    # sampling mode
    temperature: List[float] = field(default_factory=lambda: [0.7, 1.3])
    top_p: List[float] = field(default_factory=lambda: [0.85, 1.0])
    top_k: List[int] = field(default_factory=lambda: [0, 100])
    # training mode
    lr: List[float] = field(default_factory=lambda: [1e-5, 1e-3])
    lora_r: List[int] = field(default_factory=lambda: [8, 64])
    grad_accum: List[int] = field(default_factory=lambda: [1, 16])
    epochs: List[float] = field(default_factory=lambda: [1.0, 4.0])


@dataclass
class OptimizeResult:
    best_params: Dict
    best_score: float
    history: List[Dict]  # per-trial: {number, score, params, metrics}

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "best_params": self.best_params,
                    "best_score": self.best_score,
                    "history": self.history,
                },
                fh,
                indent=2,
            )


def _require_optuna():
    try:
        import optuna  # noqa: F401
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Bayesian optimization needs Optuna: pip install optuna"
        ) from exc
    import optuna

    return optuna


def optimize(
    cfg: Config,
    *,
    mode: str = "sampling",
    model_path: Optional[str] = None,
    n_trials: int = 25,
    weights: Optional[ObjectiveWeights] = None,
    space: Optional[SearchSpace] = None,
    seed: int = 42,
    eval_samples: int = 200,
    study_name: str = "denovo-bo",
    storage: Optional[str] = None,
) -> OptimizeResult:
    """Run Bayesian optimization and return the best hyperparameters.

    Parameters
    ----------
    cfg:
        Base configuration (modality, data, model, generate ...).
    mode:
        ``"sampling"`` or ``"training"``.
    model_path:
        For sampling mode, the trained checkpoint to sample from
        (defaults to ``cfg.train.output_dir``).
    n_trials:
        Optimization budget.
    eval_samples:
        Number of molecules generated per trial for scoring.
    """
    optuna = _require_optuna()
    from denovo.generate import generate

    weights = weights or ObjectiveWeights()
    space = space or SearchSpace()
    history: List[Dict] = []

    train_seqs = None  # cache training set for novelty once
    if cfg.data.train_file and os.path.exists(cfg.data.train_file):
        from denovo.data import read_sequences

        train_seqs = read_sequences(cfg.data.train_file)

    if mode == "sampling":
        mp = model_path or cfg.train.output_dir

        def objective(trial):
            trial_cfg = copy.deepcopy(cfg)
            g = trial_cfg.generate
            g.num_samples = eval_samples
            g.temperature = trial.suggest_float(
                "temperature", *space.temperature
            )
            g.top_p = trial.suggest_float("top_p", *space.top_p)
            g.top_k = trial.suggest_int("top_k", *space.top_k)
            g.do_sample = True

            seqs = generate(
                mp, g, trust_remote_code=trial_cfg.model.trust_remote_code, seed=seed
            )
            metrics = evaluate_sequences(
                seqs, trial_cfg.data.modality, training_sequences=train_seqs
            )
            score = weights.score(metrics)
            history.append(
                {
                    "number": trial.number,
                    "score": score,
                    "params": dict(trial.params),
                    "metrics": metrics.as_dict(),
                }
            )
            return score

    elif mode == "training":
        from denovo.train import train

        def objective(trial):
            trial_cfg = copy.deepcopy(cfg)
            t = trial_cfg.train
            t.lr = trial.suggest_float("lr", *space.lr, log=True)
            t.grad_accum = trial.suggest_int("grad_accum", *space.grad_accum)
            t.epochs = trial.suggest_float("epochs", *space.epochs)
            if trial_cfg.lora.use_lora:
                trial_cfg.lora.r = trial.suggest_int("lora_r", *space.lora_r)
                trial_cfg.lora.alpha = 2 * trial_cfg.lora.r
            # Keep each trial cheap: unique output dir, small sample count.
            t.output_dir = os.path.join(
                cfg.train.output_dir, "bo_trials", f"trial_{trial.number}"
            )
            out = train(trial_cfg)
            g = trial_cfg.generate
            g.num_samples = eval_samples
            seqs = generate(
                out, g, trust_remote_code=trial_cfg.model.trust_remote_code, seed=seed
            )
            metrics = evaluate_sequences(
                seqs, trial_cfg.data.modality, training_sequences=train_seqs
            )
            score = weights.score(metrics)
            history.append(
                {
                    "number": trial.number,
                    "score": score,
                    "params": dict(trial.params),
                    "metrics": metrics.as_dict(),
                }
            )
            return score

    else:
        raise ValueError(f"Unknown optimize mode {mode!r}. Use 'sampling' or 'training'.")

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=bool(storage),
    )
    study.optimize(objective, n_trials=n_trials)

    return OptimizeResult(
        best_params=dict(study.best_params),
        best_score=float(study.best_value),
        history=history,
    )
