"""Regression tests for the fail-closed build identity of tools/build_nacf_envxc_tables.py.

Written by Claude Fable 5.1, 2026-09-20 after Codex reproduced that ``--resume`` silently reused rank-1
species/tables under a rank-2 (or changed-source) manifest. Every case below asserts that a mismatched
resume changes nothing on disk and exits fail-closed (return code 2), and that a same-identity resume
reuses artifacts verbatim.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from dptb.nacf.envxc import EnvXCStore
from dptb.nacf.envxc_tables import AtomicSource, sha256_file

ROOT = Path(__file__).resolve().parents[2]
_fixture_spec = importlib.util.spec_from_file_location("test_nacf_envxc_fixture", Path(__file__).with_name("test_nacf_envxc.py"))
_fixture = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture)
SPECIES, synthetic_source = _fixture.SPECIES, _fixture.synthetic_source
spec = importlib.util.spec_from_file_location("build_nacf_envxc_tables", ROOT / "tools" / "build_nacf_envxc_tables.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

BASE = ["--kinds", "envnorm", "envfac", "--radial-rank", "1", "--l-buffer", "0", "--tail-seeds", "0",
        "--density-threshold", "1e-5", "--distance-step", "2.0", "--order", "8", "--workers", "1"]


def write_sources(directory: Path, decay_scale=1.0):
    directory.mkdir(parents=True, exist_ok=True)
    for symbol, kw in SPECIES.items():
        kw = dict(kw)
        kw["decay"] = kw["decay"] * decay_scale
        synthetic_source(symbol, **kw).save(directory / f"{symbol}.npz")
    structures = directory / "structures.json"
    structures.write_text(json.dumps([sorted(SPECIES)]))
    return structures


def snapshot(root: Path):
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


def build(sources: Path, structures: Path, output: Path, *extra):
    return cli.main(["--sources", str(sources), "--structures", str(structures), "--output", str(output), *BASE, *extra])


@pytest.fixture()
def case(tmp_path):
    sources = tmp_path / "sources"
    structures = write_sources(sources)
    output = tmp_path / "root"
    assert build(sources, structures, output) == 0
    return sources, structures, output


def test_fresh_build_writes_identity_and_same_resume_reuses(case):
    sources, structures, output = case
    identity = json.loads((output / cli.IDENTITY_FILE).read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and manifest["build_identity"] == identity["build_identity"]
    assert len(identity["build_identity"]) == 64
    for row in manifest["species"].values():
        assert row["identity"]["build_identity"] == identity["build_identity"]
    before = snapshot(output)
    lines_before = (output / "progress.jsonl").read_text().splitlines()
    assert build(sources, structures, output, "--resume") == 0
    after = snapshot(output)
    changed = {k for k in before if before[k] != after.get(k)}
    assert changed <= {"manifest.json"}                       # only the timing fields of the manifest may differ
    assert (output / "progress.jsonl").read_text().splitlines() == lines_before
    store = EnvXCStore(output)                                 # identified root loads and every artifact identity matches
    assert store.table("envnorm", "Xa", "Yb").values.shape[0] > 2
    assert store.epsilon("Xa").size > 0


def test_changed_rank_or_settings_rejected_without_touching_root(case):
    sources, structures, output = case
    before = snapshot(output)
    for extra in (["--radial-rank", "2"], ["--order", "12"], ["--distance-step", "1.0"], ["--density-threshold", "1e-6"], ["--l-buffer", "1"]):
        args = list(BASE)
        args[args.index(extra[0]) + 1] = extra[1]              # replace the base value of the changed option
        rc = cli.main(["--sources", str(sources), "--structures", str(structures), "--output", str(output), *args, "--resume"])
        assert rc == 2, extra
        assert snapshot(output) == before, extra
    # a changed code identity (different builder source) must also be rejected
    saved = cli.sha256_file
    try:
        cli.sha256_file = lambda p, _orig=saved: ("0" * 64 if str(p).endswith("build_nacf_envxc_tables.py") else _orig(p))
        assert build(sources, structures, output, "--resume") == 2
    finally:
        cli.sha256_file = saved
    assert snapshot(output) == before


def test_changed_source_rejected(case):
    sources, structures, output = case
    before = snapshot(output)
    write_sources(sources, decay_scale=1.1)                   # same species names, different atomic data
    assert build(sources, structures, output, "--resume") == 2
    assert snapshot(output) == before


def test_corrupted_table_and_species_rejected(case):
    sources, structures, output = case
    table = next(output.glob("envnorm/*.npz"))
    original = table.read_bytes()
    table.write_bytes(original + b"corrupt")
    before = snapshot(output)
    assert build(sources, structures, output, "--resume") == 2
    assert snapshot(output) == before
    table.write_bytes(original)
    species = output / "species" / "Xa.npz"
    saved = species.read_bytes()
    # replace the species artifact by one built with another rank: identity mismatch must be caught
    other = AtomicSource.load(sources / "Xa.npz")
    from dptb.nacf.envxc_tables import build_density_projectors, save_species
    proj = build_density_projectors(other, radial_rank=2, l_buffer=0, tail_seeds=0, density_threshold=1e-5)
    save_species(species, proj, identity={"build_identity": "0" * 64, "sources": {}})
    assert build(sources, structures, output, "--resume") == 2
    species.write_bytes(saved)
    assert build(sources, structures, output, "--resume") == 0


def test_legacy_or_unidentified_root_rejected(case):
    sources, structures, output = case
    (output / cli.IDENTITY_FILE).unlink()
    before = snapshot(output)
    assert build(sources, structures, output, "--resume") == 2
    assert build(sources, structures, output) == 2
    assert snapshot(output) == before
    manifest = json.loads((output / "manifest.json").read_text())
    manifest.pop("build_identity")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        EnvXCStore(output)


def test_non_resume_into_identified_root_rejected(case):
    sources, structures, output = case
    before = snapshot(output)
    assert build(sources, structures, output) == 2
    assert snapshot(output) == before


def test_store_rejects_artifacts_of_another_identity(case):
    sources, structures, output = case
    manifest = json.loads((output / "manifest.json").read_text())
    manifest["build_identity"] = "f" * 64
    (output / "manifest.json").write_text(json.dumps(manifest))
    store = EnvXCStore(output)
    with pytest.raises(ValueError):
        store.table("envnorm", "Xa", "Yb")
    with pytest.raises(ValueError):
        store.epsilon("Xa")


def test_coverage_extension_with_same_identity_is_allowed(case, tmp_path):
    sources, structures, output = case
    extended = tmp_path / "structures_ext.json"
    extended.write_text(json.dumps([["Xa"], ["Yb"], ["Xa", "Yb"]]))
    before = snapshot(output)
    assert build(sources, extended, output, "--resume") == 0
    after = snapshot(output)
    assert all(after[k] == v for k, v in before.items() if k != "manifest.json" and k != "progress.jsonl")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"]
