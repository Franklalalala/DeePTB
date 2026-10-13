"""Embedding implementations; importing this package registers every method."""
from .registry import Embedding
from .identity import Identity
from .se2 import SE2Descriptor
from .baseline import BASELINE
from .mpnn import MPNN
from .deephe3 import E3DeePH
from .e3baseline_local6 import E3BaseLineModel6
from .e3baseline_nonlocal import E3BaseLineModelNonLocal
from .trinity import Trinity
from .lem import Lem
from .slem import Slem
from .lem_prior import LemPrior
from .slem_prior import SlemPrior
from .unitb import UniTB
from .legacy.graph import LemMoEV3
from .legacy.edge import LemMoEV3Edge, LemMoEV3EdgeH0
from .legacy.h0 import LemMoEV3H0
from .legacy.prior import LemMoEV3Prior
from .legacy.prior_2b import LemMoEV3Prior2b

__all__ = [
    "Embedding",
    "Identity",
    "SE2Descriptor",
    "BASELINE",
    "MPNN",
    "E3DeePH",
    "E3BaseLineModel6",
    "E3BaseLineModelNonLocal",
    "Trinity",
    "Lem",
    "Slem",
    "LemPrior",
    "SlemPrior",
    "UniTB",
    "LemMoEV3",
    "LemMoEV3Edge",
    "LemMoEV3EdgeH0",
    "LemMoEV3H0",
    "LemMoEV3Prior",
    "LemMoEV3Prior2b",
]
