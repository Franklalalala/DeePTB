import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.prior import build as B
from dptb.prior.common.store import TableStore
from dptb.prior.gf.build_moments import second_moment
from dptb.prior.gf.tables import _manifest
from dptb.tests._requires import requires_cuda


def test_store_never_replaces_files_or_rows(tmp_path):
    store = TableStore(tmp_path / "p2", "deeptb.p2_radial_table/v1", length_unit="bohr")
    row = store.write_npz("A_B.npz", dict(values=np.zeros(3)))
    store.add("base_tables", "A|B", row)
    store.flush()
    with pytest.raises(FileExistsError):
        store.write_npz("A_B.npz", dict(values=np.ones(3)))
    with pytest.raises(FileExistsError):
        store.add("base_tables", "A|B", row)
    with pytest.raises(ValueError):
        TableStore(tmp_path / "p2", "deeptb.p2_radial_table/v1", length_unit="angstrom")
    assert TableStore(tmp_path / "p2", "deeptb.p2_radial_table/v1").has("base_tables", "A|B")


def test_keys_of_pair_and_species_families():
    assert B.keys_for("overlap", pairs=["Te|Pb"]) == ["Te|Pb"]          # ordered family
    assert B.keys_for("h0", pairs=["Te|Pb"]) == ["Pb|Te"]               # unordered family
    assert B.keys_for("moments", pairs=["Te"], element="Bi", species=["Te"]) == ["Te", "Bi"]
    keys = B.keys_for("p23", element="Bi", species=["Pb", "Te"])
    assert set(keys) == {"Bi|Bi", "Bi|Pb", "Pb|Bi", "Bi|Te", "Te|Bi"}


def test_legacy_schema_ids_are_read_not_rewritten(tmp_path):
    text = json.dumps(dict(schema="nacf-envxc/v1", complete=True))
    (tmp_path / "manifest.json").write_text(text)
    assert _manifest(tmp_path, "deeptb.envxc_table/v1")["schema"] == "deeptb.envxc_table/v1"
    assert (tmp_path / "manifest.json").read_text() == text


def test_second_moment_of_a_normalised_exponential_density():
    r = np.linspace(0.0, 40.0, 4001)
    q = 4 * np.pi * r ** 2 * np.exp(-2 * r)              # hydrogen 1s shape, any normalisation
    upf = SimpleNamespace(r=r, rab=np.full_like(r, r[1]), rhoatom_q=q, z_valence=2.0)
    assert second_moment(upf) == pytest.approx(2.0 * 3.0, rel=1e-8)    # Z <r^2>, <r^2> = 3 bohr^2


def _shell_m(shells):
    return [m for l in shells for m in [0] + [k for j in range(1, l + 1) for k in (j, -j)]]


@requires_cuda
@pytest.mark.parametrize("left", [(0, 1, 2), (3, 3, 3, 3, 2, 1)])     # square kernel; row-tiled kernel (nL = 36)
def test_fused_radial_kernel_matches_the_per_degree_path(left):
    pytest.importorskip("triton")
    from dptb.prior.common.harmonics import rotation_matrices
    from dptb.prior.common.radial import RadialTable, rotation_pack, rotation_pack_of
    g = np.random.default_rng(0)
    knots = np.linspace(0.0, 10.0, 201)
    right = (0, 1)
    ml, mr = _shell_m(left), _shell_m(right)
    vals = np.zeros((len(knots), len(ml), len(mr)))
    for a in range(len(ml)):                              # bond-frame blocks: diagonal in m
        for b in range(len(mr)):
            if ml[a] == mr[b]:
                vals[:, a, b] = np.exp(-knots / g.uniform(1, 3)) * g.normal()
    t = RadialTable(knots, vals, left, right, support=9.5)
    lmax = max(left)
    v = torch.as_tensor(g.normal(size=(500, 3)) * 4, device="cuda")
    v[0] = 0.0
    for flip in (False, True):
        rot = rotation_matrices(-v if flip else v, range(lmax + 1))
        ref = t(-v if flip else v, rot=rot)
        ref = ref.transpose(1, 2) if flip else ref
        pack = rotation_pack_of(v, lmax, -1.0 if flip else 1.0)
        assert torch.allclose(pack, rotation_pack(rot, lmax), rtol=0, atol=1e-13)
        out = torch.zeros((len(v), 40, 40), dtype=torch.float64, device="cuda")
        t.into(out, torch.arange(len(v), device="cuda"), v, pack, flip)
        n1, n2 = ref.shape[1:]
        assert torch.allclose(out[:, :n1, :n2], ref, rtol=0, atol=1e-13)
        assert not out[:, n1:].any() and not out[:, :, n2:].any()
