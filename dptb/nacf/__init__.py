"""Geometry-only NACF inference with source-bound, device-resident tables.

Offline building and benchmark helpers are deliberately not imported here.
CUDA compilation is lazy and never required for the CPU reference backend.
"""
from .assembly import NACFTableBank
from .inference import NACFGeometryPredictor, load_predictor, prepare_geometry
from .overlap import OverlapTableStore
from .radial import TorchRadialBlockTable
from .soc import SOCProjectorStore

__all__ = ['NACFTableBank', 'NACFGeometryPredictor', 'load_predictor',
           'OverlapTableStore', 'TorchRadialBlockTable', 'SOCProjectorStore', 'prepare_geometry']
