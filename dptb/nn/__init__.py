"""Neural tight-binding models.

``build_model`` constructs a complete model (NNENV, NNSK, MIX or DFTBSK) from
``model_options`` and ``common_options``, or restores one from a checkpoint.
``load_checkpoint`` reads a checkpoint file, including one written by an
earlier release. The embedding selected by ``model_options.embedding.method``
is constructed through :class:`dptb.nn.embedding.Embedding`.
"""
from .checkpoint import load_checkpoint
from .build import build_model
from .deeptb import NNENV, MIX
from .nnsk import NNSK
from .dftbsk import DFTBSK
from .hamiltonian import E3Hamiltonian, SKHamiltonian
from .hr2hk import HR2HK, HR2HK_Gamma_Only
from .energy import Eigenvalues, Eigh

__all__ = [
    "build_model",
    "load_checkpoint",
    "NNENV",
    "MIX",
    "NNSK",
    "DFTBSK",
    "E3Hamiltonian",
    "SKHamiltonian",
    "HR2HK",
    "HR2HK_Gamma_Only",
    "Eigenvalues",
    "Eigh",
]
