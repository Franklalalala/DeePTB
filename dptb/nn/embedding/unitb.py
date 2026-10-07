"""UniTB: dense and H0-routed shared-basis PDQ-MoE embeddings."""


from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from dptb.nn.embedding.emb import Embedding


from .unitb_edge import UniTBEdge
from .unitb_prior import UniTBPrior
from .unitb_options import unitb_options


@Embedding.register("unitb")
class UniTB(UniTBPrior, UniTBEdge):
    """Unified prior-conditioned embedding; parameters retain their original paths."""

    def __init__(self, *, _legacy=False, **options):
        super().__init__(**unitb_options(options, legacy=_legacy))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        edge_one_hot = self.edge_one_hot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()

        num_nodes_total = node_one_hot.shape[0]
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self._initialize_prior(
            data, edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot
        )
        return self._finish_edge_routed_forward(
            data=data,
            edge_index=edge_index,
            edge_vector=edge_vector,
            node_one_hot=node_one_hot,
            atom_type=atom_type,
            bond_type=bond_type,
            latents=latents,
            node_features=node_features,
            edge_features=edge_features,
            cutoff_coeffs=cutoff_coeffs,
            active_edges=active_edges,
            edge_one_hot=edge_one_hot,
            num_nodes_total=num_nodes_total,
        )


