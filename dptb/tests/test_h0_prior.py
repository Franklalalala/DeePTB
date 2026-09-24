"""H0InitLayer prior contract: uu-real RME sort projector, checkpoint compatibility across layout/version
changes, Saver metadata plumbing, the training-time target-fallback guard, and the physical-H0 sidecar
adapter (order-sensitive hashing, complex rejection, nested-root and train_options schema checks)."""
from __future__ import annotations

import pytest
import torch
import yaml
from pathlib import Path

import numpy as np

from dptb.data import AtomicDataDict, _keys
from dptb.data.interfaces.h0rebuild_adapter import PHYSICAL_H0_SIDECAR_SCHEMA, array_sha256, build_physical_h0_meta
from dptb.data.transforms import OrbitalMapper
from dptb.data.transforms_upper_triangle import OrbitalMapper as UpperTriangleOrbitalMapper
from dptb.nn.embedding.lem_moe_v3_h0_helpers import H0InitLayer
from dptb.utils.argcheck import flow_options
from tools.materialize_h0rebuild_lmdb import _validated_roots


class _DummyInit(torch.nn.Module):
    def __init__(self, idp: OrbitalMapper) -> None:
        super().__init__()
        self.idp = idp
        self.irreps_out = idp.get_irreps().sort()[0].simplify()


def _h0_layer(basis, *, has_soc: bool = False, use_uureal_residual_block_input: bool = False) -> H0InitLayer:
    mapper = OrbitalMapper(basis, method="e3tb", has_soc=has_soc, nextham_uureal_mask=has_soc, full_soc_prediction=False)
    return H0InitLayer(_DummyInit(mapper), use_uureal_residual_block_input=use_uureal_residual_block_input,
                       dtype=torch.float64, device="cpu").to(dtype=torch.float64)


def _mark_state_as_legacy(state):
    assert state._metadata[""]["version"] == H0InitLayer._version
    state._metadata[""]["version"] = 1
    return state


def test_uureal_projection_is_bit_exact_with_legacy_sort_path():
    """G-FIX3: the already-correct uu_real projector boundary is unchanged."""
    torch.manual_seed(20260724)
    layer = _h0_layer({"H": "1s", "C": "1s1p"}, has_soc=True, use_uureal_residual_block_input=True)
    mapper = layer.idp

    atom_type = torch.tensor([[mapper.chemical_symbol_to_type["H"]], [mapper.chemical_symbol_to_type["C"]]])
    bond_type = torch.tensor([[mapper.bond_to_type["H-C"]], [mapper.bond_to_type["C-H"]]])
    raw_node = torch.randn(2, layer.h0_dim, dtype=torch.float64)
    raw_edge = torch.randn(2, layer.h0_dim, dtype=torch.float64)
    masked_node = layer._mask_node_source(raw_node, atom_type)
    masked_edge = layer._mask_edge_source(raw_edge, bond_type)

    legacy_node = masked_node.index_select(1, layer._uureal_h0_sort_index)
    fixed_node = masked_node.index_select(1, layer._h0_sort_index)
    legacy_edge = masked_edge.index_select(1, layer._uureal_h0_sort_index)
    fixed_edge = masked_edge.index_select(1, layer._h0_sort_index)

    assert torch.equal(fixed_node, legacy_node)
    assert torch.equal(fixed_edge, legacy_edge)
    assert torch.equal(layer.node_projector(fixed_node), layer.node_projector(legacy_node))
    assert torch.equal(layer.edge_projector(fixed_edge), layer.edge_projector(legacy_edge))
    state = layer.state_dict()
    assert "_uureal_h0_sort_index" in state
    assert "_h0_sort_index" not in state


def test_scalar_only_h0_sort_index_is_identity():
    """G-FIX4: a pure-l=0 (scalar-only) layout has nothing to sort."""
    scalar_layer = _h0_layer({"H": "1s"})
    identity = torch.arange(scalar_layer.h0_dim)
    assert torch.equal(scalar_layer._h0_sort_index.cpu(), identity)


LOAD_CASES = [
    pytest.param(dict(basis={"H": "1s", "C": "1s1p"}), "current", None, id="current_layout_strict"),
    pytest.param(dict(basis={"H": "1s", "C": "1s1p"}), "legacy", "predates the H0 raw-to-sorted RME layout fix",
                id="legacy_highl_non_uureal_fails_closed"),
    pytest.param(dict(basis={"H": "1s", "C": "1s1p"}), "stripped", None, id="stripped_metadata_loadable"),
    pytest.param(dict(basis={"H": "1s"}), "legacy", None, id="legacy_scalar_only_loadable"),
    pytest.param(dict(basis={"H": "1s", "C": "1s1p"}, has_soc=True, use_uureal_residual_block_input=True),
                "legacy", None, id="legacy_uureal_loadable"),
]


@pytest.mark.parametrize("kwargs,transform,error_match", LOAD_CASES)
def test_checkpoint_load_across_layouts_and_versions(kwargs, transform, error_match):
    source = _h0_layer(**kwargs)
    target = _h0_layer(**kwargs)
    state = source.state_dict()
    if transform == "legacy":
        state = _mark_state_as_legacy(state)
    elif transform == "stripped":
        # An ensemble saver used to flatten state_dict into a plain dict, dropping _metadata; those
        # tensors were still trained under the current contract and must stay loadable.
        state = dict(state)
    if error_match is None:
        target.load_state_dict(state, strict=True)
    else:
        with pytest.raises(RuntimeError, match=error_match):
            target.load_state_dict(state, strict=True)


def test_to_cpu_obj_preserves_h0_version_metadata():
    from dptb.plugins.saver import Saver

    source = _h0_layer({"H": "1s", "C": "1s1p"})
    state = source.state_dict()
    assert state._metadata[""]["version"] == H0InitLayer._version
    saver = Saver(interval=1)
    copied = saver._to_cpu_obj(state)
    assert copied._metadata[""]["version"] == H0InitLayer._version


def test_assemble_full_model_state_prefixes_h0_version_metadata():
    from dptb.plugins.saver import Saver

    class _Expert(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Module()
            self.embedding.init_layer = _h0_layer({"H": "1s", "C": "1s1p"})

    class _Ensemble(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.ModuleList([_Expert(), _Expert()])

    class _Trainer:
        def __init__(self, model):
            self.model = model

    model = _Ensemble()
    saver = Saver(interval=1)
    saver.trainer = _Trainer(model)
    expert_states = [expert.state_dict() for expert in model.experts]
    full = saver._assemble_full_model_state(expert_states)
    assert full._metadata["experts.0.embedding.init_layer"]["version"] == H0InitLayer._version
    assert full._metadata["experts.1.embedding.init_layer"]["version"] == H0InitLayer._version


# --------------------------------------------------------------------------- training-time target fallback
def _case(device, dtype):
    idp = UpperTriangleOrbitalMapper({"H": ["1s"], "C": ["2s", "2p"]}, method="e3tb", device=device)
    node = torch.zeros(2, idp.reduced_matrix_element, device=device, dtype=dtype)
    edge = torch.zeros(2, idp.reduced_matrix_element, device=device, dtype=dtype)
    data = {
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.long),
        _keys.BATCH_KEY: torch.zeros(2, device=device, dtype=torch.long),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(
            [idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]], device=device, dtype=torch.long),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor(
            [idp.bond_to_type["H-C"], idp.bond_to_type["C-H"]], device=device, dtype=torch.long),
        _keys.NODE_FEATURES_KEY: node.clone(),
        _keys.EDGE_FEATURES_KEY: edge.clone(),
    }
    ref = {_keys.NODE_FEATURES_KEY: node.clone(), _keys.EDGE_FEATURES_KEY: edge.clone()}
    return idp, data, ref


class _BaseInit(torch.nn.Module):
    def __init__(self, idp):
        super().__init__()
        self.idp = idp
        if getattr(idp, "orbpair_irreps", None) is None:
            idp.get_irreps()
        self.irreps_out = idp.orbpair_irreps.sort()[0].simplify()

    def forward(self, edge_index, atom_type, bond_type, edge_sh, edge_length, edge_one_hot,
                active_edges=None, cutoff_coeffs=None):
        n_edge = edge_index.shape[1]
        dim = self.irreps_out.dim
        active = torch.arange(n_edge) if active_edges is None else active_edges
        return (torch.zeros(n_edge, 8), torch.zeros(atom_type.numel(), dim), torch.zeros(active.numel(), dim),
                torch.ones(n_edge) if cutoff_coeffs is None else cutoff_coeffs, active)


def test_h0init_training_target_fallback_fails_loud():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    layer = H0InitLayer(base_init=_BaseInit(idp))
    layer.train()

    edge_index = data[_keys.EDGE_INDEX_KEY]
    atom_type = data[AtomicDataDict.ATOM_TYPE_KEY]
    bond_type = data[AtomicDataDict.EDGE_TYPE_KEY]
    # target features present, node_h0/edge_h0 absent -> the fallback would feed
    # the label; in training mode this must fail loud.
    batch = {
        _keys.NODE_FEATURES_KEY: torch.randn(2, idp.reduced_matrix_element),
        _keys.EDGE_FEATURES_KEY: torch.randn(2, idp.reduced_matrix_element),
    }
    with pytest.raises(RuntimeError, match="label leak|allow_target_fallback_in_training"):
        layer(batch, edge_index, atom_type, bond_type, edge_sh=torch.zeros(2, 1), edge_length=torch.ones(2),
             edge_one_hot=torch.zeros(2, 4))

    # eval mode keeps the historical surrogate behavior (no raise).
    layer.eval()
    out = layer(batch, edge_index, atom_type, bond_type, edge_sh=torch.zeros(2, 1), edge_length=torch.ones(2),
               edge_one_hot=torch.zeros(2, 4))
    assert len(out) == 5

    # explicit opt-in restores the old training behavior.
    layer2 = H0InitLayer(base_init=_BaseInit(idp), allow_target_fallback_in_training=True)
    layer2.train()
    out2 = layer2(batch, edge_index, atom_type, bond_type, edge_sh=torch.zeros(2, 1), edge_length=torch.ones(2),
                 edge_one_hot=torch.zeros(2, 4))
    assert len(out2) == 5


def test_h0init_can_keep_native_node_init_while_replacing_edges():
    device = torch.device("cpu")
    dtype = torch.float32
    idp, data, _ref = _case(device, dtype)
    layer = H0InitLayer(base_init=_BaseInit(idp), use_h0_node_init=False, use_h0_edge_init=True,
                        fallback_to_hamiltonian=False, dtype=dtype, device=device)
    layer.eval()
    edge_index = data[_keys.EDGE_INDEX_KEY]
    atom_type = data[AtomicDataDict.ATOM_TYPE_KEY]
    bond_type = data[AtomicDataDict.EDGE_TYPE_KEY]
    batch = {
        _keys.NODE_H0_KEY: torch.randn(atom_type.numel(), idp.reduced_matrix_element),
        _keys.EDGE_H0_KEY: torch.randn(edge_index.shape[1], idp.reduced_matrix_element),
    }
    _latents, node_features, edge_features, _cutoff, _active = layer(
        batch, edge_index, atom_type, bond_type, edge_sh=torch.zeros(edge_index.shape[1], 1),
        edge_length=torch.ones(edge_index.shape[1]), edge_one_hot=torch.zeros(edge_index.shape[1], 4))
    torch.testing.assert_close(node_features, torch.zeros_like(node_features))
    assert edge_features.shape[0] == edge_index.shape[1]


# --------------------------------------------------------------------------- physical-H0 sidecar adapter
def _adapter_record():
    return {
        "atomic_numbers": np.array([14, 14], dtype=np.int64),
        "pos": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64),
        "cell": np.eye(3, dtype=np.float64) * 5.0,
        "edge_index": np.array([[0, 1], [1, 0]], dtype=np.int64),
        "edge_cell_shift": np.zeros((2, 3), dtype=np.float64),
        "node_physical_h0": np.zeros((2, 8), dtype=np.float64),
        "edge_physical_h0": np.zeros((2, 8), dtype=np.float64),
    }


def test_physical_h0_meta_is_order_sensitive_and_hashed():
    record = _adapter_record()
    meta = build_physical_h0_meta(record, energy_unit="eV")
    assert meta["schema"] == PHYSICAL_H0_SIDECAR_SCHEMA
    assert meta["node_sha256"] == array_sha256(record["node_physical_h0"])
    changed = _adapter_record()
    changed["edge_index"] = changed["edge_index"][:, ::-1]
    changed["edge_cell_shift"] = changed["edge_cell_shift"][::-1]
    changed_meta = build_physical_h0_meta(changed, energy_unit="eV")
    assert changed_meta["edge_signature"] != meta["edge_signature"]


def test_physical_h0_meta_rejects_raw_complex_features():
    record = _adapter_record()
    record["edge_physical_h0"] = record["edge_physical_h0"].astype(np.complex128)
    with pytest.raises(TypeError, match="real floating rank-2"):
        build_physical_h0_meta(record)


def test_materializer_rejects_nested_input_output_roots(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()

    with pytest.raises(ValueError, match="non-nested"):
        _validated_roots(str(input_root), str(input_root / "output"))
    with pytest.raises(ValueError, match="non-nested"):
        _validated_roots(str(input_root), str(tmp_path))


def test_physical_h0_overlay_uses_train_options_schema():
    repo_root = Path(__file__).resolve().parents[2]
    overlay = yaml.safe_load((repo_root / "configs" / "physical_h0_flow_overlay.yaml").read_text(encoding="utf-8"))

    assert "flow_options" not in overlay
    flow = overlay["train_options"]["flow_options"]
    assert flow["node_h0_key"] == "node_physical_h0"
    assert flow["edge_h0_key"] == "edge_physical_h0"
    normalized = flow_options().normalize_value(flow)
    flow_options().check_value(normalized, strict=True)

    # P0 wiring fix: the H0-init embedding must read exactly the keys the flow
    # overwrites with the interpolated state x_t.  If the embedding is left at
    # the stored-h0 defaults (node_h0/edge_h0) while the flow points at the
    # physical keys, x_t never reaches the network and the prior is silently
    # deactivated.  The overlay must therefore repoint the embedding keys too,
    # and they must stay aligned with the flow keys.
    embedding = overlay["model_options"]["embedding"]
    assert embedding["h0_node_key"] == flow["node_h0_key"] == "node_physical_h0"
    assert embedding["h0_edge_key"] == flow["edge_h0_key"] == "edge_physical_h0"
