from __future__ import annotations

from copy import deepcopy
import pickle
from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch
import yaml

from dptb.data import _keys
from dptb.data.build import DatasetBuilder
from dptb.data.dataset.lmdb_dataset import (
    LMDBDataset,
    validate_non_soc_p2_blocks,
    validate_non_soc_p2_block_tensors,
    validate_p2_feature_pair,
)
from dptb.data.interfaces.blockwise_tensor import (
    EDGE_DELTA_HAMIL_BLOCKS_KEY,
    EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    EDGE_P2_BLOCKS_KEY,
    EDGE_PRED_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    NODE_P2_BLOCKS_KEY,
    NODE_PRED_HAMIL_BLOCKS_KEY,
)
from dptb.nn.blockwise_hamiltonian import (
    attach_full_hamiltonian_from_h0,
    attach_full_hamiltonian_from_prior,
)
from dptb.nn.build import build_model
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from dptb.utils.argcheck import normalize


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_strict_p2_full_h_config_and_semantic_guards():
    payload = yaml.safe_load(
        (REPO_ROOT / "configs" / "p2_prior_non_soc_full_h_smoke.yaml").read_text(
            encoding="utf-8"
        )
    )
    normalized = normalize(deepcopy(payload))
    assert normalized["data_options"]["train"]["residual_hamiltonian"] is False
    assert normalized["model_options"]["embedding"]["method"] == "lem_moe_v3_prior"
    assert normalized["train_options"]["loss_options"]["train"][
        "pred_node_block_key"
    ] == "node_full_hamil_blocks"

    residual = deepcopy(payload)
    residual["data_options"]["train"]["residual_hamiltonian"] = True
    with pytest.raises(ValueError, match="absolute Full H"):
        normalize(residual)

    correction_loss = deepcopy(payload)
    correction_loss["train_options"]["loss_options"]["train"][
        "pred_node_block_key"
    ] = "node_hamil_blocks"
    with pytest.raises(ValueError, match="reconstructed Full-H"):
        normalize(correction_loss)


def test_p2_feature_and_block_contract_fails_closed():
    node = torch.zeros(2, 4)
    edge = torch.zeros(3, 4)
    got_node, got_edge = validate_p2_feature_pair(
        node,
        edge,
        num_nodes=2,
        num_edges=3,
        feature_dim=4,
    )
    assert got_node is node and got_edge is edge

    with pytest.raises(ValueError, match="both node_p2 and edge_p2"):
        validate_p2_feature_pair(node, None)
    with pytest.raises(TypeError, match="floating-point"):
        validate_p2_feature_pair(node.to(torch.long), edge.to(torch.long))
    with pytest.raises(ValueError, match="mapper RME width"):
        validate_p2_feature_pair(node, edge, feature_dim=5)
    with pytest.raises(NotImplementedError, match="non-SOC"):
        validate_p2_feature_pair(node.to(torch.complex64), edge.to(torch.complex64))
    with pytest.raises(ValueError, match="NaN"):
        validate_p2_feature_pair(node.fill_(torch.nan), edge)

    validate_non_soc_p2_blocks({"0_0_0_0_0": torch.eye(2).numpy()})
    with pytest.raises(NotImplementedError, match="non-SOC"):
        validate_non_soc_p2_blocks(
            {"0_0_0_0_0": torch.eye(2, dtype=torch.complex64).numpy()}
        )
    with pytest.raises(ValueError, match="invalid AO-block key"):
        validate_non_soc_p2_blocks({"bad-key": torch.eye(2).numpy()})

    validate_non_soc_p2_block_tensors(
        torch.zeros(2, 3, 3),
        torch.zeros(3, 3, 3),
        torch.tensor([[1, 1], [3, 3]]),
        torch.tensor([[1, 1], [2, 2], [3, 3]]),
        num_nodes=2,
        num_edges=3,
    )
    with pytest.raises(ValueError, match="exceeds packed canvas"):
        validate_non_soc_p2_block_tensors(
            torch.zeros(2, 3, 3),
            torch.zeros(3, 3, 3),
            torch.tensor([[1, 1], [4, 3]]),
            torch.tensor([[1, 1], [2, 2], [3, 3]]),
            num_nodes=2,
            num_edges=3,
        )


def test_full_h_reconstruction_uses_explicit_output_and_preserves_h0_soc():
    pred_node = torch.randn(2, 3, 3, requires_grad=True)
    pred_edge = torch.randn(4, 3, 3, requires_grad=True)
    p2_node = torch.randn_like(pred_node)
    p2_edge = torch.randn_like(pred_edge)
    data = {
        NODE_PRED_HAMIL_BLOCKS_KEY: pred_node,
        EDGE_PRED_HAMIL_BLOCKS_KEY: pred_edge,
        NODE_P2_BLOCKS_KEY: p2_node,
        EDGE_P2_BLOCKS_KEY: p2_edge,
    }
    attach_full_hamiltonian_from_prior(
        data,
        prior_node_field=NODE_P2_BLOCKS_KEY,
        prior_edge_field=EDGE_P2_BLOCKS_KEY,
        prior_label="P2",
        non_soc_only=True,
    )
    torch.testing.assert_close(data["node_full_hamil_blocks"], p2_node + pred_node)
    torch.testing.assert_close(data["edge_full_hamil_blocks"], p2_edge + pred_edge)
    assert data[NODE_PRED_HAMIL_BLOCKS_KEY] is pred_node
    assert data[EDGE_PRED_HAMIL_BLOCKS_KEY] is pred_edge
    (data["node_full_hamil_blocks"].sum() + data["edge_full_hamil_blocks"].sum()).backward()
    assert torch.equal(pred_node.grad, torch.ones_like(pred_node))
    assert torch.equal(pred_edge.grad, torch.ones_like(pred_edge))

    complex_data = dict(data)
    complex_data[NODE_P2_BLOCKS_KEY] = p2_node.to(torch.complex64)
    complex_data[EDGE_P2_BLOCKS_KEY] = p2_edge.to(torch.complex64)
    with pytest.raises(TypeError, match="non-SOC"):
        attach_full_hamiltonian_from_prior(
            complex_data,
            prior_node_field=NODE_P2_BLOCKS_KEY,
            prior_edge_field=EDGE_P2_BLOCKS_KEY,
            non_soc_only=True,
        )

    # The generic refactor must not narrow the historical H0/SOC route.
    h0_data = {
        NODE_PRED_HAMIL_BLOCKS_KEY: pred_node.detach().to(torch.complex64),
        EDGE_PRED_HAMIL_BLOCKS_KEY: pred_edge.detach().to(torch.complex64),
        "node_h0_blocks": p2_node.to(torch.complex64) * (1 + 1j),
        "edge_h0_blocks": p2_edge.to(torch.complex64) * (1 + 1j),
    }
    attach_full_hamiltonian_from_h0(h0_data)
    assert torch.is_complex(h0_data["node_full_hamil_blocks"])


def test_lmdb_raw_full_h_plus_p2_contract_end_to_end(tmp_path):
    lmdb_name = "p2-smoke.lmdb"
    lmdb_path = tmp_path / lmdb_name
    h_blocks = {
        "0_0_0_0_0": np.asarray([[1.2]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[1.4]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[0.3]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[0.3]], dtype=np.float32),
    }
    p2_blocks = {
        "0_0_0_0_0": np.asarray([[1.0]], dtype=np.float32),
        "1_1_0_0_0": np.asarray([[1.1]], dtype=np.float32),
        "0_1_0_0_0": np.asarray([[0.2]], dtype=np.float32),
        "1_0_0_0_0": np.asarray([[0.2]], dtype=np.float32),
    }
    record = {
        _keys.CELL_KEY: np.eye(3, dtype=np.float32) * 8.0,
        _keys.POSITIONS_KEY: np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
        ),
        _keys.ATOMIC_NUMBERS_KEY: np.asarray([1, 1], dtype=np.int64),
        _keys.PBC_KEY: np.asarray([False, False, False]),
        _keys.EDGE_INDEX_KEY: np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        _keys.EDGE_CELL_SHIFT_KEY: np.zeros((2, 3), dtype=np.float32),
        _keys.NODE_FEATURES_KEY: np.asarray([[1.2], [1.4]], dtype=np.float32),
        _keys.EDGE_FEATURES_KEY: np.asarray([[0.3], [0.3]], dtype=np.float32),
        _keys.NODE_P2_KEY: np.asarray([[1.0], [1.1]], dtype=np.float32),
        _keys.EDGE_P2_KEY: np.asarray([[0.2], [0.2]], dtype=np.float32),
        "hamiltonian": h_blocks,
        "hamiltonian_p2": p2_blocks,
    }
    env = lmdb.open(str(lmdb_path), map_size=1 << 20, subdir=True)
    with env.begin(write=True) as txn:
        txn.put((0).to_bytes(4, "big"), pickle.dumps(record))
    env.close()

    dataset = DatasetBuilder()(
        root=str(tmp_path),
        r_max=2.0,
        type="LMDBDataset",
        prefix="p2-smoke",
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_P2=True,
        residual_hamiltonian=False,
    )
    assert isinstance(dataset, LMDBDataset)
    sample = dataset.get(0)

    torch.testing.assert_close(sample[_keys.NODE_P2_KEY], torch.tensor([[1.0], [1.1]]))
    torch.testing.assert_close(sample[_keys.EDGE_P2_KEY], torch.tensor([[0.2], [0.2]]))
    assert sample[NODE_P2_BLOCKS_KEY].shape == (2, 1, 1)
    assert sample[EDGE_P2_BLOCKS_KEY].shape == (2, 1, 1)
    # residual_hamiltonian=false means the historically named target slots
    # contain absolute Full-H blocks.
    torch.testing.assert_close(
        sample[NODE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([1.2, 1.4])
    )
    torch.testing.assert_close(
        sample[EDGE_DELTA_HAMIL_BLOCKS_KEY].flatten(), torch.tensor([0.3, 0.3])
    )

    raw_dataset = DatasetBuilder()(
        root=str(tmp_path),
        r_max=2.0,
        type="LMDBDataset",
        prefix="p2-smoke",
        separator=".",
        basis={"H": "1s"},
        get_Hamiltonian=True,
        get_P2=True,
        prefer_precomputed_p2=False,
        residual_hamiltonian=False,
    )
    raw_sample = raw_dataset.get(0)
    torch.testing.assert_close(
        raw_sample[_keys.NODE_P2_KEY], torch.tensor([[1.0], [1.1]])
    )
    torch.testing.assert_close(
        raw_sample[_keys.EDGE_P2_KEY], torch.tensor([[0.2], [0.2]])
    )


def _prior_model(device: str = "cpu"):
    return build_model(
        common_options={
            "basis": {"H": "1s", "O": "1s1p"},
            "overlap": False,
            "dtype": "float32",
            "device": device,
        },
        model_options={
            "embedding": {
                "method": "lem_moe_v3_prior",
                "output_route": "h_b0",
                "n_layers": 1,
                "avg_num_neighbors": 2.0,
                "r_max": 4.0,
                "irreps_hidden": "4x0e+4x1o+4x1e+4x2e",
                "env_embed_multiplicity": 4,
                "latent_dim": 8,
                "latent_channels": [8],
                "edge_one_hot_dim": 4,
                "num_experts": 1,
                "num_shared_experts": 1,
                "top_k": 1,
                "universal": True,
                "use_layer_onehot_tp": False,
                "use_out_onehot_tp": False,
                "use_interpolation_out": False,
                "tp_radial_emb": False,
                "mole_linear_mode": "indexed_ref",
                "so2_fusion_mode": "streamed_m_major_ref",
                "rme_fusion_rank": 4,
                "rme_fusion_init": 0.0,
                "prior_kind": "p2",
                "use_soft_edge_memory": True,
                "soft_edge_memory_num_slots": 8,
                "soft_edge_memory_num_heads": 2,
                "soft_edge_memory_head_dim": 4,
                "soft_edge_memory_gate_mode": "deepseek",
            },
            "prediction": {
                "method": "block_native",
                "scale_type": "no_scale",
                "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True,
                "add_prior": True,
            },
        },
        train_options={},
        no_check=False,
    )


def _prior_data(model):
    device = next(model.parameters()).device
    h = model.idp.chemical_symbol_to_type["H"]
    o = model.idp.chemical_symbol_to_type["O"]
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    edge_type = torch.tensor(
        [model.idp.bond_to_type["H-O"], model.idp.bond_to_type["O-H"]],
        dtype=torch.long,
        device=device,
    )
    rme = model.idp.orbpair_irreps.dim
    max_norb = model.idp.full_basis_norb
    return {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1]],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        ),
        _keys.EDGE_INDEX_KEY: edge_index,
        _keys.ATOM_TYPE_KEY: torch.tensor(
            [[h], [o]], dtype=torch.long, device=device
        ),
        _keys.EDGE_TYPE_KEY: edge_type,
        _keys.NODE_P2_KEY: torch.randn(2, rme, device=device),
        _keys.EDGE_P2_KEY: torch.randn(2, rme, device=device),
        NODE_P2_BLOCKS_KEY: torch.randn(
            2, max_norb, max_norb, device=device
        ),
        EDGE_P2_BLOCKS_KEY: torch.randn(
            2, max_norb, max_norb, device=device
        ),
    }


def test_prior_embedding_model_full_h_loss_backward_checkpoint_and_inference(tmp_path):
    torch.manual_seed(19)
    model = _prior_model()
    data = _prior_data(model)
    p2_node = data[NODE_P2_BLOCKS_KEY].clone()
    p2_edge = data[EDGE_P2_BLOCKS_KEY].clone()
    output = model(data)

    correction_node = output[NODE_PRED_HAMIL_BLOCKS_KEY]
    correction_edge = output[EDGE_PRED_HAMIL_BLOCKS_KEY]
    torch.testing.assert_close(
        output["node_full_hamil_blocks"], p2_node + correction_node
    )
    torch.testing.assert_close(
        output["edge_full_hamil_blocks"], p2_edge + correction_edge
    )
    for key in (
        "soft_edge_memory_attention_entropy",
        "soft_edge_memory_attention_max_probability",
        "soft_edge_memory_gate_mean",
    ):
        assert key in output and torch.isfinite(output[key])

    output[NODE_DELTA_HAMIL_BLOCKS_KEY] = (
        output["node_full_hamil_blocks"].detach() + 0.01
    )
    output[EDGE_DELTA_HAMIL_BLOCKS_KEY] = (
        output["edge_full_hamil_blocks"].detach() - 0.01
    )
    output[NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = output[
        "node_hamil_block_shape"
    ].clone()
    output[EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = output[
        "edge_hamil_block_shape"
    ].clone()
    loss = HamilBlockwiseNexTHamLoss(
        idp=model.idp,
        pred_node_block_key="node_full_hamil_blocks",
        pred_edge_block_key="edge_full_hamil_blocks",
    )(output)
    assert loss.item() > 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert data[_keys.POSITIONS_KEY].grad is not None
    assert torch.isfinite(data[_keys.POSITIONS_KEY].grad).all()
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    optimizer.step()

    checkpoint_path = tmp_path / "p2_prior_smoke.pt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    restored = _prior_model()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    restored.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    restored.eval()

    inference_source = _prior_data(model)

    def _clone_input():
        return {
            key: value.detach().clone() if isinstance(value, torch.Tensor) else value
            for key, value in inference_source.items()
        }

    with torch.no_grad():
        expected = model(_clone_input())
        actual = restored(_clone_input())
    torch.testing.assert_close(
        actual["node_full_hamil_blocks"], expected["node_full_hamil_blocks"]
    )
    torch.testing.assert_close(
        actual["edge_full_hamil_blocks"], expected["edge_full_hamil_blocks"]
    )

    missing = _prior_data(model)
    del missing[_keys.EDGE_P2_KEY]
    with pytest.raises(KeyError, match="edge P2 field"):
        model(missing)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_prior_embedding_cuda_forward_backward_smoke():
    torch.manual_seed(23)
    model = _prior_model("cuda")
    data = _prior_data(model)
    output = model(data)
    loss = (
        output["node_full_hamil_blocks"].square().mean()
        + output["edge_full_hamil_blocks"].square().mean()
    )
    loss.backward()
    torch.cuda.synchronize()
    assert torch.isfinite(loss)
    assert data[_keys.POSITIONS_KEY].grad is not None
    assert torch.isfinite(data[_keys.POSITIONS_KEY].grad).all()
