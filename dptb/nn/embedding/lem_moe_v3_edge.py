"""Compatibility constructors for historical edge-routed embeddings."""
import logging
from .emb import Embedding
from .unitb import UniTB
from .unitb_edge import UniTBEdge
from .prior_common import H0InitLayer
from dptb.nn.route_drop import sample_structure_routes

log = logging.getLogger(__name__)


@Embedding.register("lem_moe_v3_edge")
class LemMoEV3Edge(UniTBEdge):
    """Historical geometry-initialized edge model with unchanged defaults."""

    @staticmethod
    def _sample_routes(data, probability):
        return sample_structure_routes(data, probability)

    def __init__(self, **kwargs):
        log.info("Compatibility embedding lem_moe_v3_edge uses the UniTB edge backbone.")
        super().__init__(**kwargs)


@Embedding.register("lem_moe_v3_edge_h0")
class LemMoEV3EdgeH0(UniTB, LemMoEV3Edge):
    """Translate the historical method while retaining its constructor defaults."""

    @staticmethod
    def _h0_layer_type():
        return H0InitLayer

    def __init__(self, **kwargs):
        log.info("Compatibility embedding lem_moe_v3_edge_h0 translated to unitb.")
        super().__init__(_legacy=True, **kwargs)
