"""Property-conditioned (property-guided) generation.

Two strategies, both using RDKit property objectives from
:mod:`denovo.properties`:

* :func:`guided_generate` -- *best-of-N* steering: oversample from the model,
  score every molecule with the objective, keep the best. No retraining; works
  with any pretrained checkpoint and runs on CPU.
* :func:`iterative_generate` -- repeat guided rounds, each time keeping the top
  molecules, to push the property distribution further (a lightweight
  generate → score → select loop, the same idea as the closed-loop backbone).

Both report the property distribution *before* vs *after* steering so the
effect is measurable.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from denovo.config import GenerateConfig
from denovo.modalities import get_modality
from denovo.properties import PropertyObjective


@dataclass
class ConditionResult:
    selected: List[Tuple[str, float]]        # (smiles, property value), best first
    baseline_values: List[float]             # property values of the raw pool
    selected_values: List[float]

    def summary(self, objective: PropertyObjective) -> str:
        b = np.array(self.baseline_values) if self.baseline_values else np.array([np.nan])
        s = np.array(self.selected_values) if self.selected_values else np.array([np.nan])
        return (
            f"  objective         : {objective.describe()}\n"
            f"  molecules kept    : {len(self.selected)}\n"
            f"  {objective.name} (unconditioned) : mean {np.nanmean(b):.3f}  "
            f"[{np.nanmin(b):.2f}, {np.nanmax(b):.2f}]\n"
            f"  {objective.name} (conditioned)   : mean {np.nanmean(s):.3f}  "
            f"[{np.nanmin(s):.2f}, {np.nanmax(s):.2f}]"
        )


def _score_pool(raw: List[str], objective: PropertyObjective, modality_name: str):
    modality = get_modality(modality_name)
    scored = []
    values = []
    seen = set()
    for s in raw:
        canon = modality.canonicalize(s)
        if canon is None or canon in seen:
            continue
        val = objective.value(canon)
        sc = objective.score(canon)
        if val is None or sc is None:
            continue
        seen.add(canon)
        values.append(val)
        scored.append((canon, val, sc))
    return scored, values


def guided_generate(
    model_path: str,
    gen_cfg: GenerateConfig,
    objective: PropertyObjective,
    *,
    modality_name: str = "smiles",
    oversample: int = 8,
    trust_remote_code: bool = False,
    seed: int = 42,
) -> ConditionResult:
    """Best-of-N property steering: oversample, score, keep the top molecules."""
    from denovo.generate import generate

    target_n = gen_cfg.num_samples
    big = copy.deepcopy(gen_cfg)
    big.num_samples = target_n * max(1, oversample)

    raw = generate(model_path, big, trust_remote_code=trust_remote_code, seed=seed)
    scored, baseline_values = _score_pool(raw, objective, modality_name)
    scored.sort(key=lambda t: t[2], reverse=True)          # by score, best first
    top = scored[:target_n]

    return ConditionResult(
        selected=[(s, v) for s, v, _ in top],
        baseline_values=baseline_values,
        selected_values=[v for _, v, _ in top],
    )


def iterative_generate(
    model_path: str,
    gen_cfg: GenerateConfig,
    objective: PropertyObjective,
    *,
    rounds: int = 3,
    modality_name: str = "smiles",
    oversample: int = 8,
    trust_remote_code: bool = False,
    seed: int = 42,
) -> ConditionResult:
    """Run several guided rounds, accumulating the best unique molecules."""
    best: dict = {}
    baseline_values: List[float] = []
    for r in range(rounds):
        res = guided_generate(
            model_path, gen_cfg, objective,
            modality_name=modality_name, oversample=oversample,
            trust_remote_code=trust_remote_code, seed=seed + r,
        )
        if r == 0:
            baseline_values = res.baseline_values
        for smi, val in res.selected:
            best[smi] = val
    ranked = sorted(best.items(), key=lambda kv: objective.score(kv[0]) or -1e9, reverse=True)
    ranked = ranked[: gen_cfg.num_samples]
    return ConditionResult(
        selected=ranked,
        baseline_values=baseline_values,
        selected_values=[v for _, v in ranked],
    )
