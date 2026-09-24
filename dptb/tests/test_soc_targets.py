"""SOC uu-real targets: mapper widths, compact feature expansion, block round trip, record gates."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import dptb.data.build as data_build
from dptb.data.build import DatasetBuilder
from dptb.data.dataset.lmdb_dataset import _expand_soc_uureal_compact
from dptb.data.dataset.record_pipeline import RecordSchemaValidator, _host
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
from dptb.nnops import trainer as trainer_mod
from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss
from dptb.utils.soc_target import resolve_nextham_uureal_mask
from tools.convert_feature_lmdb_to_blockwise import (
    _convert_record,
    _full_soc_mapper_for,
    _project_full_soc_to_uureal,
)


BASIS = {"H": "1s", "C": "1s1p"}
BASIS_0603_SOC = {"C": "4s2p2d1f"}


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


# --------------------------------------------------------------------------
# target width
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("uureal_mask", "full_soc", "width"),
    [(True, False, 729), (False, False, 5832), (True, True, 5832)],
    ids=["uureal", "full_soc", "full_soc_overrides_mask"],
)
def test_soc_target_width_follows_uureal_mask_and_full_soc_override(uureal_mask, full_soc, width):
    """The uu-real target keeps one directed real block (27x27); full SOC keeps all eight."""
    compact = width == 729
    assert resolve_nextham_uureal_mask(
        nextham_uureal_mask=uureal_mask, full_soc_prediction=full_soc
    ) is compact

    mapper = OrbitalMapper(
        BASIS_0603_SOC,
        method="e3tb",
        has_soc=True,
        nextham_uureal_mask=uureal_mask,
        full_soc_prediction=full_soc,
    )
    assert mapper.full_basis_norb == 27
    assert mapper.nextham_uureal_mask is compact
    assert mapper.soc_uureal_target is compact
    assert mapper.reduced_matrix_element == width
    assert mapper.get_irreps(no_parity=False).dim == width
    if compact:
        assert mapper.mask_uureal.numel() == 729
        assert bool(mapper.mask_uureal.all())


_KEEP = [True, False, True, False]


@pytest.mark.parametrize(
    ("features", "keep", "keep_mask", "expected"),
    [
        ([[1.0, 2.0], [3.0, 4.0]], _KEEP, None, [[1.0, 0.0, 2.0, 0.0], [3.0, 0.0, 4.0, 0.0]]),
        # a keep count defers the channel mask to the type mapper
        ([[5.0, 6.0]], 2, _KEEP, [[5.0, 0.0, 6.0, 0.0]]),
        ([[1.0, 2.0, 3.0, 4.0]], _KEEP, None, "unchanged"),
        # a reduced (uu-real) mapper keeps compact rows compact
        ([[1.0, 2.0], [3.0, 4.0]], 2, [True, True], "unchanged"),
        ([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], _KEEP, None, ValueError),
    ],
    ids=["fill_full_channels", "keep_count_uses_mapper_mask", "full_width_passthrough",
         "reduced_target_stays_compact", "inconsistent_width"],
)
def test_compact_soc_features_expand_to_the_mapper_width(features, keep, keep_mask, expected):
    features = torch.tensor(features)
    meta = {
        "soc_uureal_compact": True,
        "soc_uureal_keep": torch.tensor(keep) if isinstance(keep, list) else keep,
        "soc_uureal_full_rme": 4,
    }
    kwargs = {} if keep_mask is None else {"keep_mask": torch.tensor(keep_mask)}
    if expected is ValueError:
        with pytest.raises(ValueError, match="node_h0"):
            _expand_soc_uureal_compact(features, meta, field_name="node_h0", **kwargs)
        return
    actual = _expand_soc_uureal_compact(features, meta, field_name="node_h0", **kwargs)
    if expected == "unchanged":
        assert actual is features
    else:
        assert torch.equal(actual, torch.tensor(expected))


# --------------------------------------------------------------------------
# blockwise route
# --------------------------------------------------------------------------
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
        data, mapper, node_blocks=node_blocks, edge_blocks=edge_blocks
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
    # endpoint accounting stays in block space unless RME logging is requested
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
    full_soc_mapper = OrbitalMapper(BASIS, method="e3tb", has_soc=True, nextham_uureal_mask=False)
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
            "uu_re", "uu_im", "ud_re", "ud_im", "du_re", "du_im", "dd_re", "dd_im",
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
    edge_types = torch.tensor([mapper.bond_to_type["H-C"], mapper.bond_to_type["C-H"]])
    assert torch.equal(
        back_node[mapper.mask_to_nrme[node_types]],
        expected_node[mapper.mask_to_nrme[node_types]],
    )
    assert torch.equal(
        back_edge[mapper.mask_to_erme[edge_types]],
        expected_edge[mapper.mask_to_erme[edge_types]],
    )


# --------------------------------------------------------------------------
# records, datasets and trainer wiring
# --------------------------------------------------------------------------
_SOC_TOKEN = "nextham_uureal_729_orbpair_v1"
_SOC_SCHEMA = "deeptb.soc_uureal_named_slots_rme_training_sample/v1"
# the canonical uu-real mapper fingerprint (a deliberate compatibility value)
_SOC_DIGEST = "dcc39e397efb11b1cfb4d56e64720964791418dbf3c06544c02cee0a6758ecce"


def _soc_dataset():
    mapper = SimpleNamespace(has_soc=True, nextham_uureal_mask=True, reduced_matrix_element=729)
    return SimpleNamespace(type_mapper=mapper)


def test_soc_named_basis_requires_exact_mapper_and_canonical_fingerprint(monkeypatch):
    dataset = _soc_dataset()
    record = {"basis_fingerprint": _SOC_TOKEN, "hamiltonian_schema": _SOC_SCHEMA}
    monkeypatch.setattr(_host, "mapper_basis_fingerprint", lambda mapper: _SOC_DIGEST)
    validator = RecordSchemaValidator()
    assert validator.validate_schema_and_basis(dataset, record) == (_SOC_DIGEST, _SOC_DIGEST)

    dataset.type_mapper.nextham_uureal_mask = False
    with pytest.raises(ValueError, match="mismatch"):
        validator.validate_schema_and_basis(dataset, record)

    dataset.type_mapper.nextham_uureal_mask = True
    monkeypatch.setattr(_host, "mapper_basis_fingerprint", lambda mapper: "a" * 64)
    with pytest.raises(ValueError, match="canonical"):
        validator.validate_schema_and_basis(dataset, record)


def test_soc_named_basis_rejects_wrong_schema():
    record = {"basis_fingerprint": _SOC_TOKEN, "hamiltonian_schema": "wrong"}
    with pytest.raises(ValueError, match="mismatch"):
        RecordSchemaValidator().validate_schema_and_basis(_soc_dataset(), record)


def test_dataset_builder_forwards_uureal_mask_to_orbital_mapper(monkeypatch, tmp_path):
    captured = {}

    class FakeOrbitalMapper:
        def __init__(self, **kwargs):
            captured["mapper"] = kwargs

    class FakeLMDBDataset:
        def __init__(self, **kwargs):
            captured["dataset"] = kwargs

    monkeypatch.setattr(data_build, "OrbitalMapper", FakeOrbitalMapper)
    monkeypatch.setattr(data_build, "LMDBDataset", FakeLMDBDataset)
    data_dir = tmp_path / "data.0000"
    data_dir.mkdir()
    (data_dir / "data.mdb").write_bytes(b"")

    dataset = DatasetBuilder()(
        root=str(tmp_path),
        prefix="data",
        type="LMDBDataset",
        r_max=5.0,
        basis={"H": "1s"},
        has_soc=True,
        nextham_uureal_mask=True,
    )

    assert isinstance(dataset, FakeLMDBDataset)
    assert captured["mapper"]["has_soc"] is True
    assert captured["mapper"]["nextham_uureal_mask"] is True
    assert captured["dataset"]["type_mapper"] is not None


def test_trainer_builds_flow_with_loss_idp_when_loss_uses_compressed_layout(monkeypatch):
    model_idp = object()
    loss_idp = object()
    captured = {}

    class MinimalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.hamiltonian = SimpleNamespace(idp=model_idp)

    class MinimalDataset:
        get_Hamiltonian = True
        get_DM = False

    def fake_loss(**kwargs):
        assert kwargs["idp"] is model_idp
        return SimpleNamespace(idp=loss_idp)

    def fake_flow(options, *, idp, dtype, device):
        captured["idp"] = idp
        return SimpleNamespace(enabled=False)

    monkeypatch.setattr(trainer_mod, "DataLoader", lambda **kwargs: [])
    monkeypatch.setattr(trainer_mod, "Loss", fake_loss)
    monkeypatch.setattr(trainer_mod, "build_hamiltonian_flow", fake_flow)
    monkeypatch.setattr(
        trainer_mod, "get_optimizer", lambda **kwargs: SimpleNamespace(param_groups=[{"lr": 0.1}])
    )
    monkeypatch.setattr(trainer_mod, "get_lr_scheduler", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(trainer_mod, "configure_activation_recompute", lambda *args, **kwargs: {})

    trainer_mod.Trainer(
        train_options={
            "optimizer": {"type": "Adam", "lr": 0.1},
            "lr_scheduler": {"type": "exp", "gamma": 1.0},
            "update_lr_per_iter": False,
            "clip_grad": 1.0,
            "batch_size": 1,
            "loss_options": {"train": {"method": "hamil_abs"}},
            "flow_options": {"enabled": True},
        },
        common_options={
            "dtype": "float32",
            "device": "cpu",
            "basis": {"Si": ["3s", "3p"]},
            "has_soc": True,
            "nextham_uureal_mask": True,
        },
        model=MinimalModel(),
        train_datasets=MinimalDataset(),
    )

    assert captured["idp"] is loss_idp
