"""Standard de novo generation metrics: validity, uniqueness, novelty, diversity.

These four are the field-standard report for generative molecular models
(e.g. MOSES, GuacaMol) and translate cleanly to proteins / nucleic acids via
the modality's validator and canonicaliser.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Set

from denovo.data import read_sequences
from denovo.modalities import get_modality


@dataclass
class GenerationMetrics:
    total: int
    n_valid: int
    validity: float          # valid / total
    uniqueness: float        # unique valid / valid
    novelty: float           # valid-unique not in training / valid-unique
    diversity: Optional[float]  # mean pairwise distance (chem only), else None

    def as_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        lines = [
            f"  samples generated : {self.total}",
            f"  valid             : {self.n_valid} ({self.validity:.1%})",
            f"  uniqueness        : {self.uniqueness:.1%}",
            f"  novelty           : {self.novelty:.1%}",
        ]
        if self.diversity is not None:
            lines.append(f"  diversity         : {self.diversity:.3f}")
        return "\n".join(lines)


def _internal_diversity(canon_smiles: List[str], sample: int = 1000) -> Optional[float]:
    """Mean pairwise (1 - Tanimoto) over Morgan fingerprints, RDKit-only.

    Returns ``None`` when RDKit is unavailable or too few molecules exist.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
    except Exception:
        return None
    mols = [Chem.MolFromSmiles(s) for s in canon_smiles[:sample]]
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024)
        for m in mols
        if m is not None
    ]
    if len(fps) < 2:
        return None
    total = 0.0
    count = 0
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for s in sims:
            total += 1.0 - s
            count += 1
    return total / count if count else None


def evaluate_sequences(
    generated: List[str],
    modality_name: str,
    *,
    training_file: Optional[str] = None,
    training_sequences: Optional[List[str]] = None,
) -> GenerationMetrics:
    """Compute generation metrics for a list of raw generated strings."""
    modality = get_modality(modality_name)
    total = len(generated)

    # Validity + canonical forms.
    valid_canon: List[str] = []
    for seq in generated:
        c = modality.canonicalize(seq)
        if c is not None:
            valid_canon.append(c)
    n_valid = len(valid_canon)

    unique = set(valid_canon)
    n_unique = len(unique)

    # Novelty vs training set (canonicalised for a fair comparison).
    train_set: Set[str] = set()
    if training_sequences is None and training_file:
        training_sequences = read_sequences(training_file)
    if training_sequences:
        for seq in training_sequences:
            c = modality.canonicalize(seq)
            train_set.add(c if c is not None else seq.strip())

    novel = [c for c in unique if c not in train_set] if train_set else list(unique)

    validity = n_valid / total if total else 0.0
    uniqueness = n_unique / n_valid if n_valid else 0.0
    novelty = (len(novel) / n_unique) if n_unique else 0.0

    diversity = None
    if modality.kind == "molecule":
        diversity = _internal_diversity(list(unique))

    return GenerationMetrics(
        total=total,
        n_valid=n_valid,
        validity=validity,
        uniqueness=uniqueness,
        novelty=novelty,
        diversity=diversity,
    )
