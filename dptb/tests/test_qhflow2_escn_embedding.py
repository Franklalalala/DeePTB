import pytest
import torch
from torch import nn

from dptb.data import _keys
from dptb.nn.embedding.qhflow2_escn import QHFlow2ESCNEmbedding
from dptb.utils.argcheck import qhflow2_escn


def _bare_embedding(**attrs):
    embedding = QHFlow2ESCNEmbedding.__new__(QHFlow2ESCNEmbedding)
    nn.Module.__init__(embedding)
    embedding.dtype = torch.float32
    embedding.rme_dim = 1
    embedding.matrix_l = 1
    embedding.hidden_size = 2
    embedding.ham_context_mode = "features"
    embedding.flow_time_key = "flow_time"
    embedding.allow_missing_flow_time = False
    for key, value in attrs.items():
        setattr(embedding, key, value)
    return embedding


def test_qhflow2_schema_is_dedicated_and_defaults_to_time_conditioning():
    fields = {field.name: field for field in qhflow2_escn()}

    assert "hidden_size" in fields
    assert "num_ham_gnn_layers" in fields
    assert "qhflow2_src" in fields
    assert "irreps_hidden" not in fields
    assert fields["use_flow_time_embedding"].default is True
    assert fields["allow_missing_flow_time"].default is False


def test_ham_context_prefers_h_t_and_respects_expert_masks():
    embedding = _bare_embedding()
    embedding.context_proj = nn.Identity()
    batch = torch.tensor([0, 0, 1])
    edge_index = torch.tensor([[0, 2], [1, 2]])
    data = {
        _keys.NODE_H0_KEY: torch.tensor([[1.0], [3.0], [5.0]]),
        _keys.EDGE_H0_KEY: torch.tensor([[2.0], [4.0]]),
        _keys.NODE_FEATURES_KEY: torch.full((3, 1), 100.0),
        _keys.EDGE_FEATURES_KEY: torch.full((2, 1), 100.0),
        "expert_node_mask": torch.tensor([True, False, True]),
        "expert_edge_mask": torch.tensor([False, True]),
    }

    context = embedding._ham_context(data, batch, edge_index, num_graphs=2)

    assert torch.equal(context[:, 0, :], torch.tensor([[1.0, 0.0], [5.0, 4.0]]))


def test_flow_time_rejects_missing_or_misaligned_values():
    embedding = _bare_embedding()
    batch = torch.tensor([0, 0, 1])

    with pytest.raises(KeyError, match="requires one"):
        embedding._flow_time({}, batch, num_graphs=2)
    with pytest.raises(ValueError, match="must be scalar"):
        embedding._flow_time({"flow_time": torch.tensor([0.0, 0.1, 0.2, 0.3])}, batch, 2)

    assert torch.equal(
        embedding._flow_time({"flow_time": torch.tensor([0.25, 0.75])}, batch, 2),
        torch.tensor([0.25, 0.75]),
    )


class _RecordingBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, data, matrix):
        self.seen = data
        return {
            "node_embedding": torch.zeros(data["pos"].shape[0], 1),
            "xy_embedding": torch.zeros(data["edge_index"].shape[1], 1),
        }


def test_forward_uses_pbc_aware_dptb_edge_vector_with_qhflow2_sign():
    backbone = _RecordingBackbone()
    embedding = _bare_embedding(
        hidden_size=1,
        ham_context_mode="zero",
        backbone=backbone,
        node_head=nn.Identity(),
        edge_head=nn.Identity(),
    )
    embedding.register_buffer("_type_to_z", torch.tensor([1]), persistent=False)
    data = {
        _keys.POSITIONS_KEY: torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0], [1]]),
        _keys.EDGE_CELL_SHIFT_KEY: torch.tensor([[1.0, 0.0, 0.0]]),
        _keys.CELL_KEY: torch.diag(torch.tensor([10.0, 10.0, 10.0])),
        _keys.ATOM_TYPE_KEY: torch.zeros(2, 1, dtype=torch.long),
        "flow_time": torch.zeros(1),
    }

    embedding(data)

    assert torch.equal(backbone.seen["edge_distance_vec"], torch.tensor([[-11.0, 0.0, 0.0]]))
    assert torch.equal(backbone.seen["edge_distance"], torch.tensor([11.0]))
