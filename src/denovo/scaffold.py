"""Scaffold- / substructure-constrained generation.

Keep only generated molecules that contain a required substructure (a scaffold
or pharmacophore core), matched with RDKit. Optionally rank the survivors by a
:class:`~denovo.properties.PropertyObjective`, so you can ask for e.g.
"molecules containing this benzodiazepine core, maximizing QED".

Constraint satisfaction is *guaranteed* by post-hoc substructure matching (a
best-of-N filter), so it works with any pretrained checkpoint and is fully
measurable (it reports the match rate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from denovo.modalities import get_modality
from denovo.properties import PropertyObjective


def _rdkit_chem():
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        return Chem
    except Exception:  # pragma: no cover - depends on install
        return None


def parse_pattern(pattern: str, is_smarts: bool = False):
    """Parse a scaffold pattern (SMILES or SMARTS) into an RDKit query mol."""
    Chem = _rdkit_chem()
    if Chem is None:
        raise ImportError("Scaffold matching needs RDKit: pip install rdkit")
    mol = Chem.MolFromSmarts(pattern) if is_smarts else Chem.MolFromSmiles(pattern)
    if mol is None:
        raise ValueError(f"Could not parse scaffold pattern {pattern!r} "
                         f"({'SMARTS' if is_smarts else 'SMILES'}).")
    return mol


def contains(smiles: str, pattern_mol) -> bool:
    """True if ``smiles`` is a valid molecule containing the pattern substructure."""
    Chem = _rdkit_chem()
    if Chem is None:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return mol.HasSubstructMatch(pattern_mol)


@dataclass
class ScaffoldResult:
    matches: List[Tuple[str, Optional[float]]]  # (smiles, objective value or None)
    n_pool: int
    n_valid: int
    n_match: int

    @property
    def match_rate(self) -> float:
        return self.n_match / self.n_valid if self.n_valid else 0.0

    def summary(self, scaffold: str, objective: Optional[PropertyObjective] = None) -> str:
        lines = [
            f"  scaffold          : {scaffold}",
            f"  generated (valid) : {self.n_valid} / {self.n_pool}",
            f"  contain scaffold  : {self.n_match} ({self.match_rate:.1%} of valid)",
            f"  kept              : {len(self.matches)}",
        ]
        if objective is not None and self.matches:
            vals = [v for _, v in self.matches if v is not None]
            if vals:
                lines.append(f"  {objective.name} of kept   : mean {sum(vals)/len(vals):.3f}")
        return "\n".join(lines)


def scaffold_generate(
    model_path: str,
    gen_cfg,
    scaffold: str,
    *,
    is_smarts: bool = False,
    modality_name: str = "smiles",
    objective: Optional[PropertyObjective] = None,
    oversample: int = 20,
    trust_remote_code: bool = False,
    seed: int = 42,
) -> ScaffoldResult:
    """Generate, then keep molecules that contain ``scaffold`` (best-of-N filter)."""
    import copy

    from denovo.generate import generate

    pattern = parse_pattern(scaffold, is_smarts=is_smarts)
    modality = get_modality(modality_name)

    target_n = gen_cfg.num_samples
    big = copy.deepcopy(gen_cfg)
    big.num_samples = target_n * max(1, oversample)
    raw = generate(model_path, big, trust_remote_code=trust_remote_code, seed=seed)

    seen = set()
    n_valid = 0
    matches: List[Tuple[str, Optional[float]]] = []
    for s in raw:
        canon = modality.canonicalize(s)
        if canon is None or canon in seen:
            continue
        seen.add(canon)
        n_valid += 1
        if contains(canon, pattern):
            val = objective.value(canon) if objective is not None else None
            matches.append((canon, val))

    # Rank by objective when given (best first); else keep generation order.
    if objective is not None:
        matches.sort(key=lambda t: (t[1] if t[1] is not None else -1e9), reverse=True)

    return ScaffoldResult(
        matches=matches[:target_n],
        n_pool=len(raw),
        n_valid=n_valid,
        n_match=len(matches),
    )
