from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dptb.data.interfaces.p2_table import (
    P2_TABLE_SCHEMA,
    P2TableStore,
    canonical_species_source_identity,
    p2_base_component_contract,
    species_source_identity_sha256,
)
from tools import build_nonsoc_p2_tables as p2_builder
from tools.audit_nonsoc_p2_tables import audit, parse_args as parse_audit_args
from tools.build_nonsoc_p2_tables import (
    _assert_resume_code_identity,
    _assert_resume_species_identity,
    _code_identity_sha256,
)
from tools.smoke_nonsoc_p2_table import _hermiticity, _numerical_gate


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _synthetic_code_identity(
    *, gate1: bytes = b"gate1", builder: bytes = b"builder"
) -> dict[str, str]:
    return {
        "schema": p2_builder.P2_TABLE_CODE_IDENTITY_SCHEMA,
        "gate1_script_sha256": _sha256_bytes(gate1),
        "builder_script_sha256": _sha256_bytes(builder),
    }


def _identity_manifest(tmp_path):
    orb_bytes = b"synthetic orbital source\n"
    upf_bytes = b"synthetic pseudopotential source\n"
    orb_path = tmp_path / "X.orb"
    upf_path = tmp_path / "X.upf"
    orb_path.write_bytes(orb_bytes)
    upf_path.write_bytes(upf_bytes)
    species = {
        "X": {
            # Four AOs, deliberately represented as 1s+1p.  A four-s-shell
            # sample has the same AO count and must still be rejected.
            "orbital_shells": [0, 1],
            "orbital_norb": 4,
            "orbital_cutoff_bohr": 4.0,
            "projector_shells": [],
            "projector_norb": 0,
            "projector_cutoffs_bohr": [],
            "projector_max_cutoff_bohr": 0.0,
            "orbital_sha256": _sha256_bytes(orb_bytes),
            "upf_sha256": _sha256_bytes(upf_bytes),
            "upf_has_so": False,
            "array_path": "species/X.npz",
        }
    }
    identity = canonical_species_source_identity(species)
    manifest = {
        "schema": P2_TABLE_SCHEMA,
        "length_unit": "bohr",
        "energy_unit": "Ry",
        "complete": True,
        "species": species,
        "species_source_identity": identity,
        "species_source_identity_sha256": species_source_identity_sha256(identity),
        "base_tables": {"X|X": {"path": "base/X__X.npz"}},
        "projector_tables": {},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest, orb_path, upf_path


def _write_zero_audit_table(root):
    (root / "base").mkdir()
    (root / "species").mkdir()
    species = {}
    for symbol in ("X", "Y"):
        orb_bytes = f"{symbol} orbital\n".encode()
        upf_bytes = f"{symbol} upf\n".encode()
        (root / f"{symbol}.orb").write_bytes(orb_bytes)
        (root / f"{symbol}.upf").write_bytes(upf_bytes)
        array_path = root / "species" / f"{symbol}.npz"
        np.savez(
            array_path,
            d_eff=np.zeros((0, 0)),
            onsite_base=np.zeros((1, 1)),
        )
        species[symbol] = {
            "orbital_shells": [0],
            "orbital_norb": 1,
            "orbital_cutoff_bohr": 0.5,
            "projector_shells": [],
            "projector_norb": 0,
            "projector_cutoffs_bohr": [],
            "projector_max_cutoff_bohr": 0.0,
            "orbital_sha256": _sha256_bytes(orb_bytes),
            "upf_sha256": _sha256_bytes(upf_bytes),
            "upf_has_so": False,
            "array_path": f"species/{symbol}.npz",
            "array_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
        }
    base_tables = {}
    for left in species:
        for right in species:
            path = root / "base" / f"{left}__{right}.npz"
            np.savez(
                path,
                distances=np.asarray([0.0, 1.0]),
                values=np.zeros((2, 1, 1), dtype=np.float32),
                left_shells=np.asarray([0], dtype=np.int16),
                right_shells=np.asarray([0], dtype=np.int16),
                support_bohr=np.asarray(1.0),
            )
            base_tables[f"{left}|{right}"] = {
                "path": f"base/{left}__{right}.npz",
                "interpolation": "linear",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    identity = canonical_species_source_identity(species)
    manifest = {
        "schema": P2_TABLE_SCHEMA,
        "length_unit": "bohr",
        "energy_unit": "Ry",
        "complete": True,
        "species": species,
        "species_source_identity": identity,
        "species_source_identity_sha256": species_source_identity_sha256(identity),
        "base_tables": base_tables,
        "projector_tables": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_component_reciprocity_table(root, *, tamper_vna_cross=False):
    (root / "base").mkdir()
    (root / "species").mkdir()
    contract = p2_base_component_contract("all")
    base_arrays = {
        name: spec["array"]
        for name, spec in contract["base_shard"].items()
        if spec["available"]
    }
    onsite_arrays = {
        name: spec["array"]
        for name, spec in contract["species_shard"].items()
        if spec["available"]
    }
    species = {}
    for symbol in ("X", "Y"):
        orb_bytes = f"{symbol} orbital\n".encode()
        upf_bytes = f"{symbol} upf\n".encode()
        (root / f"{symbol}.orb").write_bytes(orb_bytes)
        (root / f"{symbol}.upf").write_bytes(upf_bytes)
        array_path = root / "species" / f"{symbol}.npz"
        np.savez(
            array_path,
            d_eff=np.zeros((0, 0)),
            onsite_base=np.asarray([[3.0]]),
            onsite_overlap=np.asarray([[1.0]]),
            onsite_kinetic=np.asarray([[1.0]]),
            onsite_vna_endpoint=np.asarray([[2.0]]),
        )
        species[symbol] = {
            "orbital_shells": [0],
            "orbital_norb": 1,
            "orbital_cutoff_bohr": 0.5,
            "projector_shells": [],
            "projector_norb": 0,
            "projector_cutoffs_bohr": [],
            "projector_max_cutoff_bohr": 0.0,
            "orbital_sha256": _sha256_bytes(orb_bytes),
            "upf_sha256": _sha256_bytes(upf_bytes),
            "upf_has_so": False,
            "array_path": f"species/{symbol}.npz",
            "array_sha256": hashlib.sha256(array_path.read_bytes()).hexdigest(),
            "onsite_component_arrays": onsite_arrays,
        }

    distances = np.asarray([0.0, 0.5, 1.0], dtype=np.float64)

    def radial(value):
        output = np.full((len(distances), 1, 1), value, dtype=np.float32)
        output[-1] = 0.0
        return output

    base_tables = {}
    for left in species:
        for right in species:
            if left == right:
                vna_left, vna_right = 2.0, 2.0
            elif left == "X" and right == "Y":
                vna_left, vna_right = (2.2, 2.8) if tamper_vna_cross else (2.0, 3.0)
            else:
                vna_left, vna_right = 3.0, 2.0
            kinetic = 1.0
            path = root / "base" / f"{left}__{right}.npz"
            np.savez(
                path,
                distances=distances,
                values=radial(kinetic + vna_left + vna_right),
                overlap_values=radial(0.7),
                kinetic_values=radial(kinetic),
                vna_left_values=radial(vna_left),
                vna_right_values=radial(vna_right),
                left_shells=np.asarray([0], dtype=np.int16),
                right_shells=np.asarray([0], dtype=np.int16),
                support_bohr=np.asarray(1.0),
            )
            base_tables[f"{left}|{right}"] = {
                "path": f"base/{left}__{right}.npz",
                "interpolation": "linear",
                "component_arrays": base_arrays,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    identity = canonical_species_source_identity(species)
    manifest = {
        "schema": P2_TABLE_SCHEMA,
        "length_unit": "bohr",
        "energy_unit": "Ry",
        "complete": True,
        "base_component_contract": contract,
        "species": species,
        "species_source_identity": identity,
        "species_source_identity_sha256": species_source_identity_sha256(identity),
        "base_tables": base_tables,
        "projector_tables": {},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _production_store(root):
    return P2TableStore(
        root,
        verify_checksums=False,
        require_explicit_source_identity=True,
    )


# ---------------------------------------------------------------------------
# Store-side fail-closed: identity/shard/sample tampers (collapsed permutations)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, match",
    [
        ("species_identity_sha256", "identity checksum mismatch"),
        ("legacy_missing_identity", "lacks an explicit species source identity"),
        ("base_code_identity", r"base:X\|X code identity"),
        ("shard_shell", "shell contract mismatch"),
        ("shard_support", "support contract mismatch"),
        ("sample_shell_sequence", "shell sequence"),
        ("sample_upf_checksum", "UPF checksum mismatch"),
    ],
)
def test_store_fails_closed_on_tampered_identity_shard_or_sample(tmp_path, kind, match):
    manifest, orb_path, upf_path = _identity_manifest(tmp_path)

    if kind == "species_identity_sha256":
        manifest["species_source_identity_sha256"] = "0" * 64
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            _production_store(tmp_path)
        return

    if kind == "legacy_missing_identity":
        manifest.pop("species_source_identity")
        manifest.pop("species_source_identity_sha256")
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            _production_store(tmp_path)
        return

    if kind == "base_code_identity":
        identity = _synthetic_code_identity()
        identity_sha256 = _code_identity_sha256(identity)
        manifest["source"] = {
            "code_identity_schema": identity["schema"],
            "gate1_script_sha256": identity["gate1_script_sha256"],
            "builder_script_sha256": identity["builder_script_sha256"],
            "code_identity_sha256": identity_sha256,
        }
        manifest["species"]["X"]["code_identity_sha256"] = identity_sha256
        manifest["base_tables"]["X|X"].update(
            {"code_identity_sha256": "f" * 64, "support_bohr": 8.0}
        )
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            _production_store(tmp_path)
        return

    if kind in ("shard_shell", "shard_support"):
        shard_shells = [0, 0, 0, 0] if kind == "shard_shell" else [0, 1]
        shard_support = 8.0 if kind == "shard_shell" else 7.5
        (tmp_path / "base").mkdir()
        np.savez(
            tmp_path / "base" / "X__X.npz",
            distances=np.asarray([0.0, shard_support]),
            values=np.zeros((2, 4, 4), dtype=np.float32),
            left_shells=np.asarray(shard_shells, dtype=np.int16),
            right_shells=np.asarray(shard_shells, dtype=np.int16),
            support_bohr=np.asarray(shard_support),
        )
        manifest["base_tables"]["X|X"]["support_bohr"] = shard_support
        (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        store = _production_store(tmp_path)
        with pytest.raises(ValueError, match=match):
            store.base("X", "X")
        return

    # Sample-contract validation tampers -- a correct contract validates first.
    store = _production_store(tmp_path)
    source_files = {"X": {"orbital": orb_path, "upf": upf_path}}
    ok = store.validate_sample_contract(
        symbols=["X"],
        atom_shells=[[0, 1]],
        source_files=source_files,
        require_source_files=True,
    )
    assert ok["shell_sequence_verified"] is True
    assert ok["source_files_verified"] is True

    if kind == "sample_shell_sequence":
        with pytest.raises(ValueError, match=match):
            store.validate_sample_contract(
                symbols=["X"],
                atom_shells=[[0, 0, 0, 0]],
                source_files=source_files,
                require_source_files=True,
            )
    else:  # sample_upf_checksum
        upf_path.write_bytes(b"changed pseudopotential source\n")
        with pytest.raises(ValueError, match=match):
            store.validate_sample_contract(
                symbols=["X"],
                atom_shells=[[0, 1]],
                source_files=source_files,
                require_source_files=True,
            )


# ---------------------------------------------------------------------------
# Builder-resume fail-closed: species + code identity tampers (collapsed)
# ---------------------------------------------------------------------------


def test_builder_resume_rejects_changed_species_or_code_identity(tmp_path):
    manifest, _, _ = _identity_manifest(tmp_path)

    # Species identity: the recorded identity is reusable; a changed orbital
    # checksum refuses stale shard reuse.
    identity = manifest["species_source_identity"]
    identity_sha256 = manifest["species_source_identity_sha256"]
    _assert_resume_species_identity(manifest, identity, identity_sha256)

    changed_species = json.loads(json.dumps(manifest["species"]))
    changed_species["X"]["orbital_sha256"] = "f" * 64
    changed_identity = canonical_species_source_identity(changed_species)
    with pytest.raises(ValueError, match="refusing stale shard reuse"):
        _assert_resume_species_identity(
            manifest,
            changed_identity,
            species_source_identity_sha256(changed_identity),
        )

    # Code identity: a consistent gate1/builder pair is reusable; a missing
    # builder hash, an inconsistent recorded sha, or a changed builder all fail.
    code = _synthetic_code_identity()
    code_sha = _code_identity_sha256(code)
    source = {
        "gate1_script_sha256": code["gate1_script_sha256"],
        "builder_script_sha256": code["builder_script_sha256"],
        "code_identity_sha256": code_sha,
    }
    _assert_resume_code_identity({"source": source}, code, code_sha)

    missing_builder = {"source": dict(source)}
    missing_builder["source"].pop("builder_script_sha256")
    with pytest.raises(ValueError, match="lacks Gate1/builder code identity"):
        _assert_resume_code_identity(missing_builder, code, code_sha)

    inconsistent = {"source": dict(source)}
    inconsistent["source"]["code_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="code_identity_sha256 is inconsistent"):
        _assert_resume_code_identity(inconsistent, code, code_sha)

    changed = _synthetic_code_identity(builder=b"changed builder")
    with pytest.raises(ValueError, match="code identity changed"):
        _assert_resume_code_identity(
            {"source": source}, changed, _code_identity_sha256(changed)
        )


# ---------------------------------------------------------------------------
# All-components roundtrip (anchor): builder writes recomposable shard, resumes
# ---------------------------------------------------------------------------


def test_builder_all_components_writes_recomposable_shard_and_resumes(
    tmp_path, monkeypatch
):
    dataset_root = tmp_path / "dataset"
    case = dataset_root / "cases" / "case0"
    case.mkdir(parents=True)
    pp_orb = tmp_path / "pp_orb"
    pp_orb.mkdir()
    orb_path = pp_orb / "X.orb"
    upf_path = pp_orb / "X.upf"
    gate1_path = tmp_path / "gate1.py"
    orb_path.write_text("orbital", encoding="utf-8")
    upf_path.write_text("upf", encoding="utf-8")
    gate1_path.write_text("# synthetic", encoding="utf-8")

    radial_grid = np.asarray([0.0, 0.5, 1.0])
    channel = p2_builder.Channel(
        l=0,
        radial=np.ones_like(radial_grid),
        cutoff_bohr=1.0,
        source_index=0,
    )
    orbital = p2_builder.OrbitalLike(
        symbol="X",
        r=radial_grid,
        channels=(channel,),
        source=orb_path,
    )
    weighted = p2_builder.OrbitalLike(
        symbol="X",
        r=radial_grid,
        channels=(channel,),
        source=orb_path,
    )
    projector = p2_builder.OrbitalLike(
        symbol="X",
        r=radial_grid,
        channels=(),
        source=upf_path,
    )
    species_data = {
        "X": {
            "orbital": orbital,
            "weighted": weighted,
            "projector": projector,
            "upf": SimpleNamespace(has_so=False),
            "orb_path": orb_path,
            "upf_path": upf_path,
            "repair": {},
        }
    }
    monkeypatch.setattr(p2_builder, "_load_gate1", lambda path: object())
    monkeypatch.setattr(
        p2_builder,
        "_load_species",
        lambda gate1, cases, root, requested: ({}, species_data),
    )
    monkeypatch.setattr(
        p2_builder,
        "_effective_d",
        lambda upf: np.zeros((0, 0), dtype=np.float64),
    )

    def fake_context(left, right, **kwargs):
        return SimpleNamespace(left=left, right=right)

    def fake_values(context, distances, *, kinetic=False, distance_chunk=128):
        if kinetic:
            value = 1.0
        elif context.left is orbital and context.right is orbital:
            value = 0.7
        elif context.left is weighted:
            value = 2.0
        elif context.right is weighted:
            value = 2.0
        else:  # pragma: no cover - protects the synthetic routing contract
            raise AssertionError("unexpected synthetic SBT context")
        output = np.full((len(distances), 1, 1), value, dtype=np.float64)
        support = context.left.rcut + context.right.rcut
        output[np.asarray(distances) >= support - 1.0e-12] = 0.0
        return output

    monkeypatch.setattr(p2_builder, "_sbt_context", fake_context)
    monkeypatch.setattr(p2_builder, "_build_values", fake_values)
    args = SimpleNamespace(
        dataset_root=dataset_root,
        pp_orb=pp_orb,
        gate1_script=gate1_path,
        output=tmp_path / "table",
        case=[],
        species="",
        distance_step=0.5,
        kmax=10.0,
        n_k=8,
        n_mu=4,
        n_phi=8,
        distance_chunk=4,
        interpolation="linear",
        base_components="all",
        overwrite=False,
    )
    first = p2_builder.build(args)
    assert first["base_component_contract"]["mode"] == "all"
    code_hash = first["source"]["code_identity_sha256"]
    assert first["source"]["gate1_script_sha256"] == _sha256_bytes(
        gate1_path.read_bytes()
    )
    assert len(first["source"]["builder_script_sha256"]) == 64
    assert first["component_table_semantics"]["scope"] == (
        "component_table_diagnostic_oracle_only"
    )
    assert first["component_table_semantics"]["materialized_lmdb_side_channels"] == [
        "node_p2",
        "edge_p2",
    ]
    assert "node_vnl" in first["component_table_semantics"][
        "not_materialized_lmdb_side_channels"
    ]
    assert first["base_tables"]["X|X"]["code_identity_sha256"] == code_hash
    assert first["species"]["X"]["code_identity_sha256"] == code_hash
    with np.load(args.output / "base" / "X__X.npz", allow_pickle=False) as payload:
        assert {
            "values",
            "overlap_values",
            "kinetic_values",
            "vna_left_values",
            "vna_right_values",
            "onsite_base",
            "onsite_overlap",
            "onsite_kinetic",
            "onsite_vna_endpoint",
        }.issubset(payload.files)
        np.testing.assert_allclose(payload["values"][0], [[5.0]])
        np.testing.assert_allclose(payload["overlap_values"][0], [[0.7]])
        np.testing.assert_allclose(payload["onsite_base"], [[3.0]])
        np.testing.assert_allclose(payload["onsite_vna_endpoint"], [[2.0]])

    store = P2TableStore(
        args.output,
        require_explicit_source_identity=True,
    )
    np.testing.assert_allclose(
        store.base_component("X", "X", "kinetic").values[0], [[1.0]]
    )
    np.testing.assert_allclose(
        store.onsite_component("X", "vna_endpoint"), [[2.0]]
    )

    second = p2_builder.build(args)
    assert second["built_shards_this_run"] == 0
    assert second["skipped_shards_this_run"] == 1

    manifest_path = args.output / "manifest.json"
    missing_code_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_code_manifest["source"].pop("builder_script_sha256")
    manifest_path.write_text(json.dumps(missing_code_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks Gate1/builder code identity"):
        p2_builder.build(args)

    legacy_args = SimpleNamespace(**vars(args))
    legacy_args.output = tmp_path / "legacy_none_table"
    legacy_args.base_components = "none"
    p2_builder.build(legacy_args)
    legacy_manifest_path = legacy_args.output / "manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["build_settings"].pop("base_components")
    legacy_manifest.pop("base_component_contract")
    for metadata in legacy_manifest["base_tables"].values():
        metadata.pop("component_arrays")
    for metadata in legacy_manifest["species"].values():
        metadata.pop("onsite_component_arrays")
    legacy_manifest_path.write_text(
        json.dumps(legacy_manifest), encoding="utf-8"
    )
    resumed_legacy = p2_builder.build(legacy_args)
    assert resumed_legacy["skipped_shards_this_run"] == 1
    assert resumed_legacy["base_component_contract"]["mode"] == "none"
    assert resumed_legacy["base_tables"]["X|X"]["component_arrays"] == {
        "p2_base": "values"
    }


# ---------------------------------------------------------------------------
# Audit: one full-scope production-gate smoke (with component reciprocity)
# ---------------------------------------------------------------------------


def test_audit_full_scope_and_component_reciprocity(tmp_path):
    # CLI defaults to full-shard loading.
    default_args = parse_audit_args(
        ["--table-root", str(tmp_path), "--output", str(tmp_path / "defaults.json")]
    )
    assert default_args.base_samples == 0
    assert default_args.projector_samples == 0

    _write_zero_audit_table(tmp_path)
    full = audit(
        parse_audit_args(
            ["--table-root", str(tmp_path), "--output", str(tmp_path / "full.json")]
        )
    )
    assert full["audit_mode"] == "full_production"
    assert full["production_eligible"] is True
    assert len(full["base_samples"]) == 4
    assert full["verified_file_checksums"] == 6  # 4 base + 2 species

    sampled = audit(
        parse_audit_args(
            [
                "--table-root",
                str(tmp_path),
                "--output",
                str(tmp_path / "sampled.json"),
                "--base-samples",
                "1",
            ]
        )
    )
    assert sampled["audit_mode"] == "sampled_non_production"
    assert sampled["production_eligible"] is False
    assert sampled["status"] == "pass_non_production"
    assert len(sampled["base_samples"]) < 4

    # Component reconstruction + cross reciprocity; a tampered cross table
    # fails the vna_left->vna_right reciprocity guard.
    good = tmp_path / "good"
    good.mkdir()
    _write_component_reciprocity_table(good)
    report = audit(
        parse_audit_args(
            ["--table-root", str(good), "--output", str(good / "audit.json")]
        )
    )
    gate = report["component_numerical_gate"]
    assert gate["status"] == "pass"
    assert gate["reconstruction"]["combined"]["max_abs"] == 0.0
    assert gate["reciprocity_by_component"]["vna_left"]["summary"]["max_abs"] == 0.0

    bad = tmp_path / "bad"
    bad.mkdir()
    _write_component_reciprocity_table(bad, tamper_vna_cross=True)
    with pytest.raises(ValueError, match="vna_left->vna_right reciprocity"):
        audit(
            parse_audit_args(
                ["--table-root", str(bad), "--output", str(bad / "audit.json")]
            )
        )


# ---------------------------------------------------------------------------
# Smoke: one production numerical-gate smoke (with hermiticity closure)
# ---------------------------------------------------------------------------


def test_smoke_numerical_gate_and_hermiticity_closure():
    thresholds = {
        "max_active_mae_ry": 1.0e-3,
        "max_active_rmse_ry": 1.0e-3,
        "max_active_p99_abs_ry": 1.0e-3,
        "max_active_abs_ry": 1.0e-3,
        "min_active_r2": 0.99,
        "max_table_hermiticity_ry": 1.0e-6,
        "max_oracle_hermiticity_ry": 1.0e-6,
    }

    # The production numerical gate passes, then fails on a threshold violation.
    metrics = {
        "blocks": {
            "active_total": {
                "mae_ry": 1.0e-4,
                "rmse_ry": 2.0e-4,
                "p99_abs_ry": 3.0e-4,
                "max_abs_ry": 4.0e-4,
                "r2": 0.9999,
            }
        },
        "table_hermiticity": {
            "max_abs_ry": 1.0e-8,
            "paired_r_keys": 2,
            "nonzero_paired_r_key_pairs": 1,
            "missing_reverse_r_keys": 0,
            "reverse_closed": True,
        },
        "oracle_hermiticity": {
            "max_abs_ry": 1.0e-8,
            "paired_r_keys": 2,
            "nonzero_paired_r_key_pairs": 1,
            "missing_reverse_r_keys": 0,
            "reverse_closed": True,
        },
    }
    passed = _numerical_gate(metrics, enabled=True, thresholds=thresholds)
    assert passed["status"] == "pass"
    assert passed["production_eligible"] is True

    metrics["blocks"]["active_total"]["mae_ry"] = 2.0e-3
    failed = _numerical_gate(metrics, enabled=True, thresholds=thresholds)
    assert failed["status"] == "fail"
    assert failed["production_eligible"] is False
    assert any(
        check["metric"] == "active_total.mae_ry" and not check["pass"]
        for check in failed["checks"]
    )

    # Hermiticity closure requires nonzero reverse-closed r-keys; an open
    # oracle reverse set fails the gate on the closure metrics.
    incomplete = _hermiticity(
        np.zeros((2, 1, 1), dtype=np.float64),
        np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64),
    )
    assert incomplete["reverse_closed"] is False
    assert incomplete["missing_reverse_r_keys"] == 1
    assert incomplete["nonzero_paired_r_key_pairs"] == 0

    complete = _hermiticity(
        np.zeros((3, 1, 1), dtype=np.float64),
        np.asarray([[0, 0, 0], [1, 0, 0], [-1, 0, 0]], dtype=np.int64),
    )
    assert complete["reverse_closed"] is True
    assert complete["missing_reverse_r_keys"] == 0
    assert complete["nonzero_paired_r_key_pairs"] == 1

    herm_metrics = {
        "blocks": {
            "active_total": {
                "mae_ry": 1.0e-4,
                "rmse_ry": 2.0e-4,
                "p99_abs_ry": 3.0e-4,
                "max_abs_ry": 4.0e-4,
                "r2": 0.9999,
            }
        },
        "table_hermiticity": complete,
        "oracle_hermiticity": incomplete,
    }
    gate = _numerical_gate(herm_metrics, enabled=True, thresholds=thresholds)
    assert gate["status"] == "fail"
    failed_metrics = {
        check["metric"] for check in gate["checks"] if not check["pass"]
    }
    assert "oracle_hermiticity.reverse_closed" in failed_metrics
    assert "oracle_hermiticity.nonzero_paired_r_key_pairs" in failed_metrics
