from __future__ import annotations

import pytest
import torch

from dptb.data.interfaces.blockwise_tensor import (
    EDGE_DELTA_HAMIL_BLOCKS_KEY,
    EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    EDGE_PRED_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    NODE_PRED_HAMIL_BLOCKS_KEY,
    block_tensors_to_feature_tensors,
    ensure_non_soc_mapper,
    ensure_spatial_block_mapper,
    feature_tensors_to_block_tensors,
)
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from tools.convert_feature_lmdb_to_blockwise import (
    _convert_record,
    _full_soc_mapper_for,
    _project_full_soc_to_uureal,
)


BASIS = {"H": "1s", "C": "1s1p"}


def _data() -> dict[str, torch.Tensor]:
    return {
        "atomic_numbers": torch.tensor([1, 6]),
        "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        "edge_cell_shift": torch.zeros(2, 3),
    }


def _uureal_mapper() -> OrbitalMapper:
    return OrbitalMapper(
        BASIS,
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=True,
        full_soc_prediction=False,
    )


def test_soc_uureal_feature_block_roundtrip_and_loss_contract():
    mapper = _uureal_mapper()
    assert mapper.full_basis_norb == 4
    assert mapper.reduced_matrix_element == 16

    data = _data()
    node_blocks = torch.zeros(2, 4, 4)
    node_blocks[0, 0, 0] = 1.25
    carbon = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 10.0
    node_blocks[1] = carbon

    edge_blocks = torch.zeros(2, 4, 4)
    edge_blocks[0, :1, :4] = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    edge_blocks[1, :4, :1] = torch.tensor([[6.0], [7.0], [8.0], [9.0]])

    node_features, edge_features = block_tensors_to_feature_tensors(
        data,
        mapper,
        node_blocks=node_blocks,
        edge_blocks=edge_blocks,
    )
    packed = feature_tensors_to_block_tensors(
        data,
        mapper,
        node_features=node_features,
        edge_features=edge_features,
        complete_edges=True,
        strict_complete_edges=True,
    )

    assert torch.equal(packed.node_blocks, node_blocks)
    assert torch.equal(packed.edge_blocks, edge_blocks)
    assert torch.equal(packed.node_shapes, torch.tensor([[1, 1], [4, 4]]))
    assert torch.equal(packed.edge_shapes, torch.tensor([[1, 4], [4, 1]]))

    loss_data = {
        **data,
        NODE_PRED_HAMIL_BLOCKS_KEY: packed.node_blocks.clone().requires_grad_(),
        EDGE_PRED_HAMIL_BLOCKS_KEY: packed.edge_blocks.clone().requires_grad_(),
        NODE_DELTA_HAMIL_BLOCKS_KEY: node_blocks,
        EDGE_DELTA_HAMIL_BLOCKS_KEY: edge_blocks,
        NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY: packed.node_shapes,
        EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY: packed.edge_shapes,
    }
    loss_fn = HamilBlockwiseNexTHamLoss(idp=mapper)
    loss = loss_fn(loss_data)
    assert loss.item() == pytest.approx(0.0, abs=1e-12)
    # Block-native heads keep endpoint accounting in block space by default;
    # converting to RME is opt-in because it adds avoidable hot-path work.
    assert loss_fn.last_endpoint_metric_space == "block"
    assert loss_fn.last_block_count.item() == pytest.approx(25.0)
    assert loss_fn.last_feature_count is None
    loss.backward()
    assert torch.isfinite(loss_data[NODE_PRED_HAMIL_BLOCKS_KEY].grad).all()
    assert torch.isfinite(loss_data[EDGE_PRED_HAMIL_BLOCKS_KEY].grad).all()


def test_soc_uureal_loss_can_build_its_own_mapper():
    loss_fn = HamilBlockwiseNexTHamLoss(
        basis=BASIS,
        has_soc=True,
        nextham_uureal_mask=True,
        full_soc_prediction=False,
    )
    assert loss_fn.idp.reduced_matrix_element == 16


def test_full_spinor_soc_remains_fail_closed():
    full_soc_mapper = OrbitalMapper(
        BASIS,
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=False,
    )
    with pytest.raises(NotImplementedError, match="Full spinor SOC"):
        ensure_spatial_block_mapper(full_soc_mapper)
    with pytest.raises(NotImplementedError, match="Full spinor SOC"):
        HamilBlockwiseNexTHamLoss(idp=full_soc_mapper)


def test_strict_non_soc_guard_remains_closed_for_reduced_soc():
    mapper = _uureal_mapper()
    assert ensure_spatial_block_mapper(mapper) is mapper
    with pytest.raises(NotImplementedError, match="strictly non-SOC"):
        ensure_non_soc_mapper(mapper)


def test_full_soc_feature_record_projects_first_uureal_channel_before_packing():
    mapper = _uureal_mapper()
    full_mapper = _full_soc_mapper_for(mapper)
    assert full_mapper.reduced_matrix_element == 8 * mapper.reduced_matrix_element

    data = _data()
    generator = torch.Generator().manual_seed(17)
    full_node = torch.randn(2, full_mapper.reduced_matrix_element, generator=generator)
    full_edge = torch.randn(2, full_mapper.reduced_matrix_element, generator=generator)
    full_node_h0 = torch.randn(2, full_mapper.reduced_matrix_element, generator=generator)
    full_edge_h0 = torch.randn(2, full_mapper.reduced_matrix_element, generator=generator)
    expected_node = _project_full_soc_to_uureal(full_node, mapper, full_mapper)
    expected_edge = _project_full_soc_to_uureal(full_edge, mapper, full_mapper)

    record = {
        **data,
        "node_features": full_node.numpy(),
        "edge_features": full_edge.numpy(),
        "node_h0": full_node_h0.numpy(),
        "edge_h0": full_edge_h0.numpy(),
        "full_soc_feature_width": full_mapper.reduced_matrix_element,
        "soc_real_channel_order": [
            "uu_re",
            "uu_im",
            "ud_re",
            "ud_im",
            "du_re",
            "du_im",
            "dd_re",
            "dd_im",
        ],
    }
    converted, maximum = _convert_record(
        record,
        mapper,
        target_mode="already-delta",
        include_h0_blocks=True,
        replace_existing=False,
    )
    assert maximum == 0.0
    assert converted["blockwise_source_target_feature_width"] == 128
    assert converted["blockwise_source_h0_feature_width"] == 128

    back_node, back_edge = block_tensors_to_feature_tensors(
        converted,
        mapper,
        node_blocks=torch.as_tensor(converted[NODE_DELTA_HAMIL_BLOCKS_KEY]),
        edge_blocks=torch.as_tensor(converted[EDGE_DELTA_HAMIL_BLOCKS_KEY]),
    )
    node_types = torch.tensor([mapper.chemical_symbol_to_type["H"], mapper.chemical_symbol_to_type["C"]])
    edge_types = torch.tensor(
        [mapper.bond_to_type["H-C"], mapper.bond_to_type["C-H"]]
    )
    assert torch.equal(
        back_node[mapper.mask_to_nrme[node_types]],
        expected_node[mapper.mask_to_nrme[node_types]],
    )
    assert torch.equal(
        back_edge[mapper.mask_to_erme[edge_types]],
        expected_edge[mapper.mask_to_erme[edge_types]],
    )
