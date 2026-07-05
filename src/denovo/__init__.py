"""De-Novo-LLM: fine-tune language models to generate novel biomolecules.

The package is organised around a *modality registry* (:mod:`denovo.modalities`)
so the same training / generation / evaluation machinery works for small
molecules (SMILES / SELFIES), proteins and nucleic acids.
"""

__version__ = "0.1.0"

from denovo.config import Config, load_config  # noqa: E402
from denovo.modalities import get_modality, list_modalities  # noqa: E402

__all__ = [
    "Config",
    "load_config",
    "get_modality",
    "list_modalities",
    "__version__",
]
