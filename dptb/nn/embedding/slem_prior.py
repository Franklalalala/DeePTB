"""SLEM baseline with the shared H0/P initialization interface."""
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.AtomicDataDict import with_batch, with_edge_vectors
from .emb import Embedding
from .prior_inputs import PriorInputs
from .slem import Slem


@Embedding.register("slem_prior")
class SlemPrior(Slem):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prior_inputs = PriorInputs.from_options(self.init_layer, kwargs)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if self.prior_inputs is None:
            return super().forward(data)
        data = with_batch(with_edge_vectors(data, with_lengths=True))
        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(edge_vector[:, [1, 2, 0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]
        data = self.onehot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        latents, node_features, edge_features, hidden_features, cutoff_coeffs, active_edges = self.init_layer(
            edge_index, atom_type, bond_type, edge_sh, edge_length, node_one_hot,
        )
        node_features, edge_features = self.prior_inputs(
            data, node_features, edge_features, atom_type, bond_type,
            edge_index, edge_length, active_edges,
        )
        data[_keys.EDGE_OVERLAP_KEY] = latents
        for layer in self.layers:
            latents, hidden_features, node_features, edge_features = layer(
                latents, node_features, hidden_features, edge_features,
                node_one_hot, edge_index, edge_vector, atom_type,
                cutoff_coeffs, active_edges,
            )
        data[_keys.NODE_FEATURES_KEY] = self.out_node(node_features)
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(
            torch.zeros(edge_index.shape[1], self.idp.orbpair_irreps.dim,
                        dtype=self.dtype, device=self.device),
            0, active_edges, self.out_edge(edge_features),
        )
        return data
