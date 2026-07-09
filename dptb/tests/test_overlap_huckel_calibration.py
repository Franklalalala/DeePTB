from __future__ import annotations

"""Smoke tests for the Hueckel-v2 prior upgrades:

* huckel_energy_mode='orbital_pair' (per-orbital-pair Wolfsberg-Helmholz energies),
* huckel_scale_mode='global'/'pair_block' (+ prior_calibration artifacts, fail-closed),
* basis_onsite_mode='calibrated' (data-calibrated onsite table),
* prior_node/prior_edge split priors (hybrid oracle; missing-H0 fails loud),
* default behavior unchanged, and the H0InitLayer training-time target-fallback guard.
"""

import pytest
import torch

from dptb.data import AtomicDataDict, _keys
from dptb.data.transforms_upper_triangle import OrbitalMapper
from dptb.nn.sktb.onsiteDB import onsite_energy_database
from dptb.nnops import prior_calibration, prior_physical
from dptb.nnops.flow import HamiltonianCFM
from dptb.utils.argcheck import flow_options


def _case(device, dtype):
    idp = OrbitalMapper(
        {"H": ["1s"], "C": ["2s", "2p"]},
        method="e3tb",
        device=device,
    )
    node = torch.zeros(2, idp.reduced_matrix_element, device=device, dtype=dtype)
    edge = torch.zeros(2, idp.reduced_matrix_element, device=device, dtype=dtype)
    data = {
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]], device=device, dtype=torch.long),
        _keys.BATCH_KEY: torch.zeros(2, device=device, dtype=torch.long),
        AtomicDataDict.ATOM_TYPE_KEY: torch.tensor(
            [idp.chemical_symbol_to_type["H"], idp.chemical_symbol_to_type["C"]],
            device=device,
            dtype=torch.long,
        ),
        AtomicDataDict.EDGE_TYPE_KEY: torch.tensor(
            [idp.bond_to_type["H-C"], idp.bond_to_type["C-H"]],
            device=device,
            dtype=torch.long,
        ),
        _keys.NODE_FEATURES_KEY: node.clone(),
        _keys.EDGE_FEATURES_KEY: edge.clone(),
    }
    ref = {
        _keys.NODE_FEATURES_KEY: node.clone(),
        _keys.EDGE_FEATURES_KEY: edge.clone(),
    }
    return idp, data, ref


def _flow(idp, device, dtype, **extra):
    opts = {
        "enabled": True,
        "mode": "residual",
        "prior": "overlap_huckel",
        "strict_h0": False,
        "warn_missing_h0": False,
        "detach_interpolated_h0": False,
    }
    opts.update(extra)
    return HamiltonianCFM(opts, idp=idp, device=device, dtype=dtype)


def _write_artifact(tmp_path, idp, *, edge_scale=None, node_table=None, fingerprint=None):
    rme = int(idp.reduced_matrix_element)
    signature = prior_calibration.make_signature(idp, rme_dim=rme)
    if fingerprint is not None:
        signature["basis_fingerprint"] = fingerprint
    artifact = {
        "version": prior_calibration.CALIBRATION_VERSION,
        "signature": signature,
        "edge_scale": edge_scale,
        "node_table": node_table,
    }
    path = tmp_path / "calib.pt"
    torch.save(artifact, str(path))
    return str(path)


def test_defaults_unchanged_without_new_keys():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    edge_overlap = torch.linspace(-0.3, 0.4, steps=ref[_keys.EDGE_FEATURES_KEY].numel(),
                                  device=device, dtype=dtype).reshape_as(ref[_keys.EDGE_FEATURES_KEY])
    data[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    flow = _flow(idp, device, dtype, huckel_k=2.0)

    assert flow.prior_family.huckel_energy_mode == "type_mean"
    assert flow.prior_family.huckel_scale_mode == "none"

    _out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))
    h_mean = torch.tensor(onsite_energy_database["H"]["1s"], device=device, dtype=dtype)
    c_mean = torch.tensor(
        (onsite_energy_database["C"]["2s"] + 3.0 * onsite_energy_database["C"]["2p"]) / 4.0,
        device=device,
        dtype=dtype,
    )
    expected = 2.0 * 0.5 * (h_mean + c_mean) * edge_overlap
    expected = expected * flow._prior_mask(data, expected, "edge").to(dtype=dtype)
    torch.testing.assert_close(ctx.edge_prior, expected)


def test_orbital_pair_energy_mode_uses_per_slice_wh_energies():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    edge_overlap = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    data[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    flow = _flow(idp, device, dtype, huckel_k=1.75, huckel_energy_mode="orbital_pair")

    _out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    pair_table = prior_physical.huckel_pair_energy_table(idp, device=device, dtype=dtype)
    assert pair_table is not None
    edge_types = data[AtomicDataDict.EDGE_TYPE_KEY]
    expected = 1.75 * pair_table.index_select(0, edge_types) * edge_overlap
    expected = expected * flow._prior_mask(data, expected, "edge").to(dtype=dtype)
    torch.testing.assert_close(ctx.edge_prior, expected)

    # hand value: the H(1s)-C(2s) orbpair slice of the H-C bond carries
    # 0.5*(eps_H(1s)+eps_C(2s)), not the type-mean.
    full_h = idp.basis_to_full_basis["H"]["1s"]
    full_c = idp.basis_to_full_basis["C"]["2s"]
    blk = idp.orbpair_maps.get(f"{full_h}-{full_c}") or idp.orbpair_maps.get(f"{full_c}-{full_h}")
    assert blk is not None
    expected_energy = 0.5 * (
        onsite_energy_database["H"]["1s"] + onsite_energy_database["C"]["2s"]
    )
    bond_hc = idp.bond_to_type["H-C"]
    got = pair_table[bond_hc, blk.start:blk.stop]
    torch.testing.assert_close(
        got, torch.full_like(got, expected_energy)
    )


def test_orbital_pair_strict_requires_edge_type():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    data[_keys.EDGE_OVERLAP_KEY] = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    del data[AtomicDataDict.EDGE_TYPE_KEY]
    flow = _flow(idp, device, dtype, huckel_energy_mode="orbital_pair")

    with pytest.raises(KeyError, match="edge_type"):
        flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))


def test_global_scale_mode_multiplies_prior():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    edge_overlap = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    data[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    base = _flow(idp, device, dtype)
    scaled = _flow(idp, device, dtype, huckel_scale_mode="global", huckel_scale_global=-0.25)

    _o1, _r1, ctx_base = base.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))
    _o2, _r2, ctx_scaled = scaled.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    torch.testing.assert_close(ctx_scaled.edge_prior, -0.25 * ctx_base.edge_prior)


def test_edge_channel_scale_multiplies_hopping_prior():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    data[_keys.EDGE_OVERLAP_KEY] = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    scale = torch.linspace(
        0.5, 1.5, steps=ref[_keys.EDGE_FEATURES_KEY].shape[-1],
        device=device, dtype=dtype,
    )
    base = _flow(idp, device, dtype)
    scaled = _flow(idp, device, dtype, huckel_edge_channel_scale=scale.cpu().tolist())

    _o1, _r1, ctx_base = base.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))
    _o2, _r2, ctx_scaled = scaled.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    torch.testing.assert_close(ctx_scaled.edge_prior, ctx_base.edge_prior * scale.reshape(1, -1))
    assert ctx_scaled.edge_prior.dtype == dtype
    assert ctx_scaled.edge_prior.device == ref[_keys.EDGE_FEATURES_KEY].device


def test_edge_channel_scale_width_mismatch_raises():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    data[_keys.EDGE_OVERLAP_KEY] = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    flow = _flow(idp, device, dtype, overlap_huckel_edge_channel_scale=[1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="huckel_edge_channel_scale"):
        flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))


def test_pair_block_scale_applies_calibration_table(tmp_path):
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    edge_overlap = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    data[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    n_bond = max(idp.bond_to_type.values()) + 1
    rme = int(idp.reduced_matrix_element)
    torch.manual_seed(3)
    edge_scale = torch.randn(n_bond, rme, dtype=torch.float32)
    path = _write_artifact(tmp_path, idp, edge_scale=edge_scale)

    base = _flow(idp, device, dtype)
    calib = _flow(idp, device, dtype, huckel_scale_mode="pair_block", prior_calibration=path)

    _o1, _r1, ctx_base = base.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))
    _o2, _r2, ctx_calib = calib.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    scale_rows = edge_scale.to(dtype)[data[AtomicDataDict.EDGE_TYPE_KEY]]
    torch.testing.assert_close(ctx_calib.edge_prior, ctx_base.edge_prior * scale_rows)
    assert ctx_calib.edge_prior.dtype == dtype
    assert ctx_calib.edge_prior.device == ref[_keys.EDGE_FEATURES_KEY].device


def test_pair_block_scale_without_artifact_fails():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    data[_keys.EDGE_OVERLAP_KEY] = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    flow = _flow(idp, device, dtype, huckel_scale_mode="pair_block")

    with pytest.raises(ValueError, match="prior_calibration"):
        flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))


def test_calibration_basis_mismatch_fails_closed(tmp_path):
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    data[_keys.EDGE_OVERLAP_KEY] = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    n_bond = max(idp.bond_to_type.values()) + 1
    path = _write_artifact(
        tmp_path,
        idp,
        edge_scale=torch.ones(n_bond, int(idp.reduced_matrix_element)),
        fingerprint="deadbeef",
    )
    flow = _flow(idp, device, dtype, huckel_scale_mode="pair_block", prior_calibration=path)

    with pytest.raises(ValueError, match="basis_fingerprint|different basis"):
        flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))


def test_basis_onsite_calibrated_mode_uses_node_table(tmp_path):
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    data[_keys.EDGE_OVERLAP_KEY] = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.2)
    rme = int(idp.reduced_matrix_element)
    n_type = len(idp.chemical_symbol_to_type)
    torch.manual_seed(11)
    node_table = torch.randn(n_type, rme, dtype=torch.float32)
    path = _write_artifact(tmp_path, idp, node_table=node_table)
    flow = _flow(idp, device, dtype, basis_onsite_mode="calibrated", prior_calibration=path)

    _out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    expected = node_table.to(dtype)[data[AtomicDataDict.ATOM_TYPE_KEY]]
    expected = expected * flow._prior_mask(data, expected, "node").to(dtype=dtype)
    torch.testing.assert_close(ctx.node_prior, expected)


def test_split_prior_hybrid_uses_h0_edges_and_basis_onsite_nodes():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    edge_h0 = torch.full_like(ref[_keys.EDGE_FEATURES_KEY], 0.375)
    data["edge_h0_external"] = edge_h0
    flow = _flow(
        idp,
        device,
        dtype,
        prior="external",
        mode="full",
        prior_node="basis_onsite",
        prior_edge="external",
        prior_edge_key="edge_h0_external",
    )

    _out, _ref, ctx = flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))

    expected_node = torch.zeros_like(ref[_keys.NODE_FEATURES_KEY])
    for sym, orb in (("H", "1s"), ("C", "2s"), ("C", "2p")):
        full = idp.basis_to_full_basis[sym][orb]
        blk = idp.orbpair_maps[f"{full}-{full}"]
        degree = {"s": 0, "p": 1}[full[-1]]
        width = 2 * degree + 1
        diag = torch.arange(width, device=device, dtype=torch.long)
        rows = int(blk.start) + diag * width + diag
        expected_node[idp.chemical_symbol_to_type[sym], rows] = onsite_energy_database[sym][orb]
    torch.testing.assert_close(ctx.node_prior, expected_node)
    torch.testing.assert_close(ctx.edge_prior, edge_h0)


def test_split_prior_missing_h0_fails_loud():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    flow = _flow(
        idp,
        device,
        dtype,
        prior="external",
        mode="full",
        prior_node="basis_onsite",
        prior_edge="external",
        prior_edge_key="edge_h0_external",
    )

    with pytest.raises(KeyError, match="Split prior|produced no absolute"):
        flow.prepare_batch(data, ref, t=torch.zeros(1, device=device, dtype=dtype))


def test_split_prior_rejects_one_sided_and_nonphysical_names():
    device = torch.device("cpu")
    dtype = torch.float64
    idp, _data, _ref = _case(device, dtype)

    with pytest.raises(ValueError, match="set together"):
        _flow(idp, device, dtype, prior="external", prior_node="basis_onsite")

    with pytest.raises(ValueError, match="splittable|haar"):
        _flow(
            idp, device, dtype,
            prior="external", prior_node="haar_dm", prior_edge="external",
        )

    with pytest.raises(ValueError, match="physical absolute"):
        _flow(
            idp, device, dtype,
            prior="zero", prior_node="basis_onsite", prior_edge="external",
        )


def test_argcheck_accepts_new_keys_and_defaults_are_inert():
    schema = flow_options()
    value = schema.normalize_value(
        {
            "enabled": True,
            "prior": "overlap_huckel",
            "huckel_energy_mode": "orbital_pair",
            "huckel_scale_mode": "pair_block",
            "huckel_scale_global": 0.5,
            "prior_calibration": "/tmp/calib.pt",
            "basis_onsite_mode": "calibrated",
            "prior_node": "basis_onsite",
            "prior_edge": "external",
        }
    )
    schema.check_value(value, strict=True)
    assert value["huckel_energy_mode"] == "orbital_pair"
    assert value["prior_edge"] == "external"

    defaults = schema.normalize_value({"enabled": False})
    schema.check_value(defaults, strict=True)
    assert defaults["huckel_energy_mode"] == "type_mean"
    assert defaults["huckel_scale_mode"] == "none"
    assert defaults["basis_onsite_mode"] == "table"
    assert defaults["prior_node"] == "" and defaults["prior_edge"] == ""


def test_calibration_tool_end_to_end(tmp_path):
    """tools/calibrate_huckel_scales.py on a tiny synthetic LMDB: the fitted
    artifact must drive the flow prior to reproduce the planted per-channel
    scale exactly (H is constructed as alpha * huckel_prior per bond type)."""
    import json as _json
    import pickle as _pickle
    import runpy
    import sys

    import lmdb as _lmdb

    device = torch.device("cpu")
    dtype = torch.float64
    idp, data, ref = _case(device, dtype)
    rme = int(idp.reduced_matrix_element)
    torch.manual_seed(5)
    edge_overlap = torch.randn(2, rme, dtype=dtype)
    # target = planted_scale * (K * typemean-energy * S), one scale per bond type
    flow_probe = _flow(idp, device, dtype, huckel_k=1.75)
    data_probe = dict(data)
    data_probe[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    _o, _r, ctx = flow_probe.prepare_batch(
        data_probe, ref, t=torch.zeros(1, device=device, dtype=dtype)
    )
    planted = torch.tensor([2.0, -0.5], dtype=dtype).reshape(2, 1)
    edge_features = ctx.edge_prior * planted
    node_features = torch.randn(2, rme, dtype=dtype)

    record = {
        "atomic_numbers": torch.tensor([1, 6]).numpy(),
        "edge_index": data[_keys.EDGE_INDEX_KEY].numpy(),
        "node_features": node_features.numpy(),
        "edge_features": edge_features.numpy(),
        "edge_overlap": edge_overlap.numpy(),
        "node_overlap": torch.zeros(2, rme, dtype=dtype).numpy(),
    }
    lmdb_dir = tmp_path / "toy.lmdb"
    env = _lmdb.open(str(lmdb_dir), map_size=1 << 24, subdir=True)
    with env.begin(write=True) as txn:
        txn.put((0).to_bytes(4, "big"), _pickle.dumps(record))
    env.close()

    basis_json = tmp_path / "basis.json"
    basis_json.write_text(_json.dumps({"H": ["1s"], "C": ["2s", "2p"]}))
    out_pt = tmp_path / "calib.pt"

    argv = [
        "calibrate_huckel_scales.py",
        "--lmdb", str(lmdb_dir),
        "--basis-json", str(basis_json),
        "--huckel-k", "1.75",
        "--min-count", "1",
        "--out", str(out_pt),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        runpy.run_path("tools/calibrate_huckel_scales.py", run_name="__main__")
    finally:
        sys.argv = old_argv

    artifact = torch.load(str(out_pt), weights_only=False)
    assert artifact["report"]["R2_L1_pair_block_calibrated"] == pytest.approx(1.0, abs=1e-9)

    calibrated = _flow(
        idp, device, dtype, huckel_k=1.75,
        huckel_scale_mode="pair_block", prior_calibration=str(out_pt),
    )
    data2 = dict(data)
    data2[_keys.EDGE_OVERLAP_KEY] = edge_overlap
    _o2, _r2, ctx2 = calibrated.prepare_batch(
        data2, ref, t=torch.zeros(1, device=device, dtype=dtype)
    )
    torch.testing.assert_close(ctx2.edge_prior, edge_features, atol=1e-6, rtol=1e-6)


def test_h0init_training_target_fallback_fails_loud():
    from dptb.nn.embedding.lem_moe_v3_h0_helpers import H0InitLayer

    class _BaseInit(torch.nn.Module):
        def __init__(self, idp):
            super().__init__()
            self.idp = idp
            if getattr(idp, "orbpair_irreps", None) is None:
                idp.get_irreps()
            self.irreps_out = idp.orbpair_irreps.sort()[0].simplify()

        def forward(self, edge_index, atom_type, bond_type, edge_sh, edge_length,
                    edge_one_hot, active_edges=None, cutoff_coeffs=None):
            n_edge = edge_index.shape[1]
            dim = self.irreps_out.dim
            active = torch.arange(n_edge) if active_edges is None else active_edges
            return (
                torch.zeros(n_edge, 8),
                torch.zeros(atom_type.numel(), dim),
                torch.zeros(active.numel(), dim),
                torch.ones(n_edge) if cutoff_coeffs is None else cutoff_coeffs,
                active,
            )

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
        layer(
            batch,
            edge_index,
            atom_type,
            bond_type,
            edge_sh=torch.zeros(2, 1),
            edge_length=torch.ones(2),
            edge_one_hot=torch.zeros(2, 4),
        )

    # eval mode keeps the historical surrogate behavior (no raise).
    layer.eval()
    out = layer(
        batch,
        edge_index,
        atom_type,
        bond_type,
        edge_sh=torch.zeros(2, 1),
        edge_length=torch.ones(2),
        edge_one_hot=torch.zeros(2, 4),
    )
    assert len(out) == 5

    # explicit opt-in restores the old training behavior.
    layer2 = H0InitLayer(base_init=_BaseInit(idp), allow_target_fallback_in_training=True)
    layer2.train()
    out2 = layer2(
        batch,
        edge_index,
        atom_type,
        bond_type,
        edge_sh=torch.zeros(2, 1),
        edge_length=torch.ones(2),
        edge_one_hot=torch.zeros(2, 4),
    )
    assert len(out2) == 5
