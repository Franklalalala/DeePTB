"""``lem_moe_v3``: graph-level routing over the shared UniTB backbone."""
from dptb.nn.embedding.registry import Embedding
from dptb.nn.embedding.unitb.backbone import UniTBBackbone


@Embedding.register("lem_moe_v3")
class LemMoEV3(UniTBBackbone):
    """Graph-level routing with the historical positional and state contracts."""
