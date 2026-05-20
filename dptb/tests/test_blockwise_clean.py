import sys
from pathlib import Path

import torch

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

from dptb.data.interfaces.blockwise_tensor import (
    EDGE_DELTA_HAMIL_BLOCKS_KEY,
    EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    EDGE_PRED_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCKS_KEY,
    NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY,
    NODE_PRED_HAMIL_BLOCKS_KEY,
    block_components,
    block_dict_to_ordered_tensors,
    feature_components_from_blocks,
    feature_tensors_to_block_tensors,
    l1_rmse_from_components,
    mae_from_components,
)
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss


class FakeLiMapper:
    has_soc = False

    def __init__(self):
        self.basis = {"Li": ["s0", "s1", "s2", "s3", "p0"]}
        self.norbs = {"Li": 7}
        self.full_basis = ["s0", "s1", "s2", "s3", "p0"]
        self.orbital_maps = {
            "Li": {
                "s0": slice(0, 1),
                "s1": slice(1, 2),
                "s2": slice(2, 3),
                "s3": slice(3, 4),
                "p0": slice(4, 7),
            }
        }
        self.basis_to_full_basis = {"Li": {b: b for b in self.basis["Li"]}}
        self.orbpair_maps = {}
        start = 0
        for i, bi in enumerate(self.full_basis):
            ni = self.orbital_maps["Li"][bi].stop - self.orbital_maps["Li"][bi].start
            for bj in self.full_basis[i:]:
                nj = self.orbital_maps["Li"][bj].stop - self.orbital_maps["Li"][bj].start
                self.orbpair_maps[f"{bi}-{bj}"] = slice(start, start + ni * nj)
                start += ni * nj
        self.reduced_matrix_element = start
        self.chemical_symbol_to_type = {"Li": 0}
        self.bond_types = ["Li-Li"]
        self.bond_to_type = {"Li-Li": 0}
        self.mask_to_nrme = torch.ones((1, start), dtype=torch.bool)
        self.mask_to_erme = torch.ones((1, start), dtype=torch.bool)

    def get_orbital_maps(self):
        return self.orbital_maps

    def get_orbpair_maps(self):
        return self.orbpair_maps

    def transform_bond(self, src, dst):
        return torch.zeros_like(src, dtype=torch.long)


def one_edge_data():
    return {
        "atomic_numbers": torch.tensor([[3]], dtype=torch.long),
        "edge_index": torch.tensor([[0], [0]], dtype=torch.long),
        "edge_cell_shift": torch.tensor([[1, 0, 0]], dtype=torch.float32),
    }


def reverse_pair_data():
    return {
        "atomic_numbers": torch.tensor([[3]], dtype=torch.long),
        "edge_index": torch.tensor([[0, 0], [0, 0]], dtype=torch.long),
        "edge_cell_shift": torch.tensor([[1, 0, 0], [-1, 0, 0]], dtype=torch.float32),
    }


def test_li_feature_count_31_block_count_49():
    idp = FakeLiMapper()
    data = one_edge_data()
    pred_node = torch.ones((1, 7, 7))
    pred_edge = torch.ones((1, 7, 7))
    tgt_node = torch.zeros_like(pred_node)
    tgt_edge = torch.zeros_like(pred_edge)

    ncomp, ecomp = feature_components_from_blocks(data, idp, pred_node, tgt_node, pred_edge, tgt_edge)
    assert ncomp.count.item() == 31
    assert ecomp.count.item() == 31
    assert l1_rmse_from_components(ncomp).item() == 1.0

    block_comp = block_components(pred_node, tgt_node, torch.tensor([[7, 7]]))
    assert block_comp.count.item() == 49
    assert mae_from_components(block_comp).item() == 1.0


def test_feature_compatible_uses_canonical_mapper_slices():
    idp = FakeLiMapper()
    data = one_edge_data()
    diff = torch.zeros((1, 7, 7))
    for row, col in [(slice(0, 1), slice(0, 1)), (slice(0, 1), slice(1, 2)), (slice(0, 1), slice(4, 7)), (slice(4, 7), slice(4, 7))]:
        diff[:, row, col] = 1.0
    diff[:, 1:4, 0:1] = 9.0

    ncomp, _ = feature_components_from_blocks(data, idp, diff, torch.zeros_like(diff), None, None)
    assert ncomp.count.item() == 31
    assert ncomp.abs_sum.item() < diff.abs().sum().item()


def test_block_dict_to_ordered_tensors_reverse_fallback_and_completion():
    idp = FakeLiMapper()
    data = one_edge_data()
    rev = torch.arange(49, dtype=torch.float32).reshape(7, 7)
    blocks = {
        "0_0_0_0_0": torch.zeros(7, 7),
        "0_0_-1_0_0": rev,
    }
    packed = block_dict_to_ordered_tensors(data, idp, blocks, start_id=0, complete_edges=False)
    assert packed.node_blocks.shape == (1, 7, 7)
    assert packed.edge_blocks.shape == (1, 7, 7)
    assert torch.equal(packed.edge_blocks[0], rev.T)


def test_feature_to_block_materializer_is_differentiable_and_completes_edges():
    idp = FakeLiMapper()
    data = reverse_pair_data()
    node_features = torch.zeros((1, idp.reduced_matrix_element), requires_grad=True)
    edge_base = torch.zeros((2, idp.reduced_matrix_element), dtype=torch.float32)
    edge_base[0].fill_(1.0)
    edge_base[1].fill_(2.0)
    edge_features = edge_base.requires_grad_()

    packed = feature_tensors_to_block_tensors(data, idp, node_features=node_features, edge_features=edge_features, complete_edges=True)
    assert packed.node_blocks.shape == (1, 7, 7)
    assert packed.edge_blocks.shape == (2, 7, 7)
    assert packed.edge_blocks[0, 0, 1].item() == 1.0      # direct canonical edge 0
    assert packed.edge_blocks[0, 1, 0].item() == 2.0      # reverse edge 1 transpose

    pair_slice = idp.orbpair_maps["s0-s1"]
    loss = packed.edge_blocks[0, 0, 1] + packed.edge_blocks[0, 1, 0]
    loss.backward()
    assert edge_features.grad[0, pair_slice].abs().sum() > 0
    assert edge_features.grad[1, pair_slice].abs().sum() > 0


def test_blockwise_loss_backprop_and_feature_logs():
    idp = FakeLiMapper()
    ref = one_edge_data()
    ref[NODE_DELTA_HAMIL_BLOCKS_KEY] = torch.zeros((1, 7, 7))
    ref[EDGE_DELTA_HAMIL_BLOCKS_KEY] = torch.zeros((1, 7, 7))
    ref[NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = torch.tensor([[7, 7]])
    ref[EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = torch.tensor([[7, 7]])

    data = dict(ref)
    data[NODE_PRED_HAMIL_BLOCKS_KEY] = torch.ones((1, 7, 7), requires_grad=True)
    data[EDGE_PRED_HAMIL_BLOCKS_KEY] = torch.ones((1, 7, 7), requires_grad=True)

    loss_fn = HamilBlockwiseNexTHamLoss(idp=idp, optimization="block_mae", log_feature_compatible=True)
    loss = loss_fn(data, ref)
    loss.backward()

    assert loss.item() == 1.0
    assert data[NODE_PRED_HAMIL_BLOCKS_KEY].grad.abs().sum() > 0
    assert loss_fn.last_onsite_loss.item() == 1.0
    assert loss_fn.last_hopping_loss.item() == 1.0
    assert loss_fn.last_feature_count.item() == 62
    assert loss_fn.last_block_count.item() == 98


def test_strict_edge_completion_rejects_missing_reverse_entries():
    idp = FakeLiMapper()
    data = one_edge_data()
    edge_features = torch.ones((1, idp.reduced_matrix_element), dtype=torch.float32)
    try:
        feature_tensors_to_block_tensors(
            data,
            idp,
            edge_features=edge_features,
            complete_edges=True,
            strict_complete_edges=True,
        )
    except RuntimeError as exc:
        assert "Hermitian edge completion left unresolved" in str(exc)
    else:
        raise AssertionError("strict_complete_edges should reject missing reverse edge")


def test_loss_exposes_raw_component_stats():
    idp = FakeLiMapper()
    ref = one_edge_data()
    ref[NODE_DELTA_HAMIL_BLOCKS_KEY] = torch.zeros((1, 7, 7))
    ref[EDGE_DELTA_HAMIL_BLOCKS_KEY] = torch.zeros((1, 7, 7))
    ref[NODE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = torch.tensor([[7, 7]])
    ref[EDGE_DELTA_HAMIL_BLOCK_SHAPE_KEY] = torch.tensor([[7, 7]])

    data = dict(ref)
    data[NODE_PRED_HAMIL_BLOCKS_KEY] = torch.ones((1, 7, 7), requires_grad=True)
    data[EDGE_PRED_HAMIL_BLOCKS_KEY] = torch.ones((1, 7, 7), requires_grad=True)

    loss_fn = HamilBlockwiseNexTHamLoss(idp=idp, optimization="block_mae", log_feature_compatible=True)
    loss = loss_fn(data, ref)
    assert loss.item() == 1.0
    stats = loss_fn.last_component_stats
    assert stats["feature_onsite_count"].item() == 31
    assert stats["feature_hopping_count"].item() == 31
    assert stats["block_total_count"].item() == 98
