"""SE(3)/E(3)-equivariant flow-matching model for de novo 3D molecule design.

This is the structure-generation track: it invents molecules as 3D atomic point
clouds (atom types + coordinates) rather than as strings. The generative model
is E(3)-equivariant (via EGNN) and trained with conditional flow matching.
"""

from denovo.structure.model import EquivariantFlowModel  # noqa: F401

__all__ = ["EquivariantFlowModel"]
