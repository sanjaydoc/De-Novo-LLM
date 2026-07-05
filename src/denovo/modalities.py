"""Biomolecule modality registry.

A *modality* bundles everything that is specific to one kind of biomolecule:
how to validate a string, how to canonicalise it (so novelty/uniqueness are
computed on a normal form), and a sensible default pretrained model to
fine-tune.

Adding a new modality is deliberately cheap -- construct a :class:`Modality`
and register it with :func:`register`.  The training, generation and
evaluation code never hard-codes anything modality-specific; it only talks to
this registry.

Chemistry helpers (RDKit / SELFIES) are imported lazily so the core package
works without them -- validators then fall back to alphabet checks and emit a
one-time warning.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Lazy optional imports
# ---------------------------------------------------------------------------

_RDKIT = None          # cached module handle or False if unavailable
_SELFIES = None
_WARNED: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        _WARNED.add(key)


def _rdkit():
    """Return the RDKit Chem module, or ``None`` if RDKit is not installed."""
    global _RDKIT
    if _RDKIT is None:
        try:
            from rdkit import Chem  # type: ignore
            from rdkit import RDLogger  # type: ignore

            RDLogger.DisableLog("rdApp.*")  # silence parse noise
            _RDKIT = Chem
        except Exception:  # pragma: no cover - depends on install
            _RDKIT = False
    return _RDKIT or None


def _selfies():
    """Return the ``selfies`` module, or ``None`` if not installed."""
    global _SELFIES
    if _SELFIES is None:
        try:
            import selfies as sf  # type: ignore

            _SELFIES = sf
        except Exception:  # pragma: no cover - depends on install
            _SELFIES = False
    return _SELFIES or None


# ---------------------------------------------------------------------------
# Modality definition
# ---------------------------------------------------------------------------


@dataclass
class Modality:
    """Everything the pipeline needs to know about one biomolecule type.

    Attributes
    ----------
    name:
        Registry key, e.g. ``"smiles"``.
    kind:
        Coarse family (``"molecule"``, ``"protein"``, ``"nucleic_acid"``) used
        for grouping and choosing metrics.
    default_model:
        A small, laptop-friendly pretrained causal LM to fine-tune by default.
    validate:
        ``str -> bool`` -- is this a syntactically valid sequence?
    canonicalize:
        ``str -> Optional[str]`` -- a normal form for dedup/novelty, or
        ``None`` if the string is invalid.
    description:
        Human-readable one-liner.
    """

    name: str
    kind: str
    default_model: str
    validate: Callable[[str], bool]
    canonicalize: Callable[[str], Optional[str]]
    description: str = ""
    aliases: List[str] = field(default_factory=list)


_REGISTRY: Dict[str, Modality] = {}


def register(modality: Modality) -> Modality:
    """Register (or overwrite) a modality and its aliases."""
    _REGISTRY[modality.name.lower()] = modality
    for alias in modality.aliases:
        _REGISTRY[alias.lower()] = modality
    return modality


def get_modality(name: str) -> Modality:
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted({m.name for m in _REGISTRY.values()}))
        raise KeyError(
            f"Unknown modality {name!r}. Registered modalities: {available}."
        )
    return _REGISTRY[key]


def list_modalities() -> List[str]:
    return sorted({m.name for m in _REGISTRY.values()})


# ---------------------------------------------------------------------------
# Small molecules -- SMILES
# ---------------------------------------------------------------------------

# Character classes that legitimately appear in SMILES; used only for the
# RDKit-free fallback check.
_SMILES_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\%\.\*]+$")


def _smiles_canonical(s: str) -> Optional[str]:
    s = s.strip()
    if not s:
        return None
    chem = _rdkit()
    if chem is None:
        _warn_once(
            "rdkit-smiles",
            "RDKit not installed -- SMILES validation falls back to a loose "
            "character check. Install with `pip install rdkit` for real "
            "chemical validity, canonicalisation and metrics.",
        )
        return s if _SMILES_CHARS.match(s) else None
    mol = chem.MolFromSmiles(s)
    if mol is None:
        return None
    return chem.MolToSmiles(mol, canonical=True)


def _smiles_valid(s: str) -> bool:
    return _smiles_canonical(s) is not None


# ---------------------------------------------------------------------------
# Small molecules -- SELFIES
# ---------------------------------------------------------------------------


def _selfies_canonical(s: str) -> Optional[str]:
    """Canonicalise a SELFIES string via its decoded SMILES form."""
    s = s.strip().replace(" ", "")
    if not s:
        return None
    sf = _selfies()
    if sf is None:
        _warn_once(
            "selfies",
            "`selfies` not installed -- SELFIES strings cannot be decoded or "
            "validated. Install with `pip install selfies rdkit`.",
        )
        # Loose check: SELFIES tokens are bracketed, e.g. [C][=O].
        return s if re.match(r"^(\[[^\]]+\])+$", s) else None
    try:
        smiles = sf.decoder(s)
    except Exception:
        return None
    if not smiles:
        return None
    return _smiles_canonical(smiles)


def _selfies_valid(s: str) -> bool:
    return _selfies_canonical(s) is not None


# ---------------------------------------------------------------------------
# Sequence modalities -- proteins & nucleic acids (alphabet based)
# ---------------------------------------------------------------------------

_AA = set("ACDEFGHIKLMNPQRSTVWY")          # 20 standard amino acids
_AA_EXTRA = set("BJOUXZ*")                 # ambiguous / non-standard, tolerated
_DNA = set("ACGT")
_RNA = set("ACGU")
_NUC_EXTRA = set("N")                       # ambiguous base


def _make_alphabet_modality(allowed: set, tolerated: set):
    full = allowed | tolerated

    def _canon(s: str) -> Optional[str]:
        s = s.strip().upper().replace(" ", "")
        if not s:
            return None
        return s if all(c in full for c in s) else None

    def _valid(s: str) -> bool:
        return _canon(s) is not None

    return _valid, _canon


_protein_valid, _protein_canon = _make_alphabet_modality(_AA, _AA_EXTRA)
_dna_valid, _dna_canon = _make_alphabet_modality(_DNA, _NUC_EXTRA)
_rna_valid, _rna_canon = _make_alphabet_modality(_RNA, _NUC_EXTRA)


# ---------------------------------------------------------------------------
# Register the built-in modalities
# ---------------------------------------------------------------------------

register(
    Modality(
        name="smiles",
        kind="molecule",
        default_model="entropy/gpt2_zinc_87m",
        validate=_smiles_valid,
        canonicalize=_smiles_canonical,
        description="Small molecules as SMILES strings (RDKit-canonicalised).",
        aliases=["mol", "molecule"],
    )
)

register(
    Modality(
        name="selfies",
        kind="molecule",
        default_model="ncfrey/ChemGPT-4.7M",
        validate=_selfies_valid,
        canonicalize=_selfies_canonical,
        description="Small molecules as SELFIES strings (always decodable).",
    )
)

register(
    Modality(
        name="protein",
        kind="protein",
        default_model="nferruz/ProtGPT2",
        validate=_protein_valid,
        canonicalize=_protein_canon,
        description="Protein / peptide amino-acid sequences.",
        aliases=["peptide", "aa"],
    )
)

register(
    Modality(
        name="dna",
        kind="nucleic_acid",
        default_model="gpt2",  # char-level fine-tune / from-scratch friendly
        validate=_dna_valid,
        canonicalize=_dna_canon,
        description="DNA nucleotide sequences (A/C/G/T).",
    )
)

register(
    Modality(
        name="rna",
        kind="nucleic_acid",
        default_model="gpt2",
        validate=_rna_valid,
        canonicalize=_rna_canon,
        description="RNA nucleotide sequences (A/C/G/U).",
    )
)
