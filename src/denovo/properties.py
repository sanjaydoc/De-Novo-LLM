"""Molecular property calculators and optimization objectives (RDKit).

Used for **property-conditioned generation**: steer a de novo model toward
molecules with a target logP / QED / molecular weight / etc. Properties are
computed with RDKit (optional import); an :class:`PropertyObjective` turns a
property + goal (maximize / minimize / hit a target value) into a scalar score
that generation and the closed-loop backbone can optimize.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

_RDKIT = None
_WARNED = False


def _rdkit_mods():
    """Lazily import the RDKit helpers, or return ``None`` if unavailable."""
    global _RDKIT, _WARNED
    if _RDKIT is None:
        try:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import Crippen, Descriptors, QED

            RDLogger.DisableLog("rdApp.*")
            _RDKIT = {"Chem": Chem, "Crippen": Crippen, "Descriptors": Descriptors, "QED": QED}
        except Exception:  # pragma: no cover - depends on install
            _RDKIT = False
    if not _RDKIT and not _WARNED:
        warnings.warn(
            "RDKit not installed -- property calculations need `pip install rdkit`.",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED = True
    return _RDKIT or None


def _prop_fns():
    r = _rdkit_mods()
    if r is None:
        return {}
    return {
        "logp": lambda m: r["Crippen"].MolLogP(m),
        "qed": lambda m: r["QED"].qed(m),
        "mw": lambda m: r["Descriptors"].MolWt(m),
        "tpsa": lambda m: r["Descriptors"].TPSA(m),
        "hbd": lambda m: r["Descriptors"].NumHDonors(m),
        "hba": lambda m: r["Descriptors"].NumHAcceptors(m),
        "rings": lambda m: r["Descriptors"].RingCount(m),
        "rotbonds": lambda m: r["Descriptors"].NumRotatableBonds(m),
    }


#: Human-readable descriptions of the supported properties.
PROPERTIES = {
    "logp": "Crippen octanol-water logP (lipophilicity)",
    "qed": "Quantitative Estimate of Drug-likeness (0-1)",
    "mw": "Molecular weight (Da)",
    "tpsa": "Topological polar surface area",
    "hbd": "Number of H-bond donors",
    "hba": "Number of H-bond acceptors",
    "rings": "Ring count",
    "rotbonds": "Rotatable-bond count",
}


def compute_property(smiles: str, name: str) -> Optional[float]:
    """Compute a single property for a SMILES string, or ``None`` if invalid."""
    fns = _prop_fns()
    if name not in fns:
        if name not in PROPERTIES:
            raise KeyError(f"Unknown property {name!r}. Options: {sorted(PROPERTIES)}.")
        return None  # RDKit missing
    r = _rdkit_mods()
    mol = r["Chem"].MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return float(fns[name](mol))
    except Exception:
        return None


def compute_properties(smiles: str, names: Optional[List[str]] = None) -> Dict[str, Optional[float]]:
    names = names or list(PROPERTIES)
    return {n: compute_property(smiles, n) for n in names}


@dataclass
class PropertyObjective:
    """Turns a property goal into a maximization score (higher = better).

    Parameters
    ----------
    name:
        Property key (see :data:`PROPERTIES`).
    mode:
        ``"max"``, ``"min"`` or ``"target"``.
    target:
        Desired value (required for ``mode="target"``).
    tolerance:
        Width of the Gaussian reward around ``target`` (for ``mode="target"``).
    """

    name: str
    mode: str = "max"
    target: Optional[float] = None
    tolerance: float = 0.5

    def __post_init__(self):
        if self.name not in PROPERTIES:
            raise KeyError(f"Unknown property {self.name!r}. Options: {sorted(PROPERTIES)}.")
        if self.mode not in ("max", "min", "target"):
            raise ValueError("mode must be 'max', 'min' or 'target'.")
        if self.mode == "target" and self.target is None:
            raise ValueError("mode='target' requires a target value.")

    def value(self, smiles: str) -> Optional[float]:
        return compute_property(smiles, self.name)

    def transform(self, value: float) -> float:
        """Map a raw property value to a maximization score (pure, no RDKit)."""
        if self.mode == "max":
            return value
        if self.mode == "min":
            return -value
        # target: Gaussian bump peaking at 1.0 when value == target
        return math.exp(-((value - self.target) ** 2) / (2 * self.tolerance ** 2))

    def score(self, smiles: str) -> Optional[float]:
        """Scalar score to maximize; ``None`` for invalid molecules."""
        v = self.value(smiles)
        return None if v is None else self.transform(v)

    def describe(self) -> str:
        if self.mode == "target":
            return f"{self.name} → {self.target} (±{self.tolerance})"
        return f"{self.mode}imize {self.name}"


def build_objective(name: str, mode: str = "max", target: Optional[float] = None,
                    tolerance: float = 0.5) -> PropertyObjective:
    return PropertyObjective(name=name, mode=mode, target=target, tolerance=tolerance)
