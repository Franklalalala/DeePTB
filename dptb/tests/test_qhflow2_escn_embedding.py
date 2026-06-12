import os

import pytest
import torch
from torch import nn

from dptb.data import _keys
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.qhflow2_escn import QHFlow2ESCNEmbedding
from dptb.nn.build import _align_qhflow2_escn_with_flow_options
from dptb.utils.argcheck import qhflow2_escn, slem_h0


def _bare_embedding(**attrs):
    embedding = QHFlow2ESCNEmbedding.__new__(QHFlow2ESCNEmbedding)
    nn.Module.__init__(embedding)
    embedding.dtype = torch.float32
    embedding.rme_dim = 1
    embedding.matrix_l = 1
    embedding.hidden_size = 2
    embedding.ham_context_mode = "features"
    embedding.h0_node_key = _keys.NODE_H0_KEY
    embedding.h0_edge_key = _keys.EDGE_H0_KEY
    embedding.fallback_node_key = _keys.NODE_FEATURES_KEY
    embedding.fallback_edge_key = _keys.EDGE_FEATURES_KEY
    embedding.strict_ham_context_h0 = True
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
    assert "h0_node_key" in fields
    assert "h0_edge_key" in fields
    assert "strict_ham_context_h0" in fields
    assert fields["use_flow_time_embedding"].default is True
    assert fields["allow_missing_flow_time"].default is False


def test_slem_h0_schema_does_not_expose_qhflow2_context_mode():
    fields = {field.name: field for field in slem_h0()}

    assert "ham_context_mode" not in fields


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


def test_ham_context_follows_custom_h0_keys_and_rejects_target_fallback():
    embedding = _bare_embedding(
        h0_node_key="node_ht",
        h0_edge_key="edge_ht",
        fallback_node_key="clean_node",
        fallback_edge_key="clean_edge",
    )
    embedding.context_proj = nn.Identity()
    batch = torch.tensor([0, 0])
    edge_index = torch.tensor([[0], [1]])

    data = {
        "node_ht": torch.tensor([[1.0], [3.0]]),
        "edge_ht": torch.tensor([[2.0]]),
        "clean_node": torch.full((2, 1), 100.0),
        "clean_edge": torch.full((1, 1), 100.0),
    }
    context = embedding._ham_context(data, batch, edge_index, num_graphs=1)
    assert torch.equal(context[:, 0, :], torch.tensor([[2.0, 2.0]]))

    missing_h0 = {
        "clean_node": torch.full((2, 1), 100.0),
        "clean_edge": torch.full((1, 1), 100.0),
    }
    with pytest.raises(KeyError, match="node_ht"):
        embedding._ham_context(missing_h0, batch, edge_index, num_graphs=1)


def test_ham_context_target_fallback_requires_explicit_opt_in():
    embedding = _bare_embedding(strict_ham_context_h0=False)
    embedding.context_proj = nn.Identity()

    context = embedding._ham_context(
        {
            _keys.NODE_FEATURES_KEY: torch.tensor([[1.0], [3.0]]),
            _keys.EDGE_FEATURES_KEY: torch.tensor([[5.0]]),
        },
        torch.tensor([0, 0]),
        torch.tensor([[0], [1]]),
        num_graphs=1,
    )

    assert torch.equal(context[:, 0, :], torch.tensor([[2.0, 5.0]]))


def test_qhflow2_flow_options_inject_context_keys_without_overriding_explicit_values():
    model_options = {
        "embedding": {
            "method": "qhflow2_escn",
            "h0_node_key": "explicit_node",
        },
        "prediction": {"method": "e3tb"},
    }
    train_options = {
        "flow_options": {
            "enabled": True,
            "node_h0_key": "node_ht",
            "edge_h0_key": "edge_ht",
            "node_target_key": "node_target",
            "edge_target_key": "edge_target",
            "flow_time_key": "tau",
            "strict_ham_context_h0": False,
        }
    }

    patched = _align_qhflow2_escn_with_flow_options(model_options, train_options)

    assert patched["embedding"]["h0_node_key"] == "explicit_node"
    assert patched["embedding"]["h0_edge_key"] == "edge_ht"
    assert patched["embedding"]["fallback_node_key"] == "node_target"
    assert patched["embedding"]["fallback_edge_key"] == "edge_target"
    assert patched["embedding"]["flow_time_key"] == "tau"
    assert patched["embedding"]["strict_ham_context_h0"] is False
    assert "h0_edge_key" not in model_options["embedding"]


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


def test_real_qhflow2_init_forward_water_basis_smoke():
    qhflow2_src = os.environ.get("QHFLOW2_SRC")
    if not qhflow2_src:
        pytest.skip("set QHFLOW2_SRC to run the optional real QHFlow2 smoke")

    idp = OrbitalMapper(
        basis={"H": "2s1p", "O": "3s2p1d"},
        method="e3tb",
        device=torch.device("cpu"),
    )
    idp.get_irreps(no_parity=False)
    idp.get_orbpair_maps()
    assert idp.orbpair_irreps.dim == 121

    embedding = QHFlow2ESCNEmbedding(
        idp=idp,
        device=torch.device("cpu"),
        hidden_size=128,
        sh_lmax=4,
        num_gnn_layers=1,
        num_ham_gnn_layers=1,
        matrix_l=6,
        qhflow2_src=qhflow2_src,
    )
    edge_index = torch.tensor(
        [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
        dtype=torch.long,
    )
    data = {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0000, 0.0000, 0.0000], [0.9572, 0.0000, 0.0000], [-0.2390, 0.9270, 0.0000]],
            dtype=torch.float32,
        ),
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.ATOM_TYPE_KEY: torch.tensor([[1], [0], [0]], dtype=torch.long),
        _keys.NODE_H0_KEY: torch.zeros(3, 121),
        _keys.EDGE_H0_KEY: torch.zeros(edge_index.shape[1], 121),
        "flow_time": torch.zeros(1),
    }

    with torch.no_grad():
        out = embedding(data)

    assert out[_keys.NODE_FEATURES_KEY].shape == (3, 121)
    assert out[_keys.EDGE_FEATURES_KEY].shape == (6, 121)
    assert out[_keys.NODE_FEATURES_KEY].dtype == torch.float32
    assert torch.isfinite(out[_keys.NODE_FEATURES_KEY]).all()
    assert torch.isfinite(out[_keys.EDGE_FEATURES_KEY]).all()
