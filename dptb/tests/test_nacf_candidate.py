"""CandidatePriorPlan: explicit composition, recipe identity and fail-closed provenance (CPU, synthetic tables).

The synthetic world reuses the envxc fixture species (Xa: shells s,s,p rcut 5.0; Yb: s,p,d rcut 5.5) and builds
fake P2/P23 stores, pair-XC tables, atomic moments and a reference-engine onsite evaluator with the same declared
source hashes, so every cross-family check can pass and every single mismatch can be provoked.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dptb.data.interfaces.p2_table import RadialBlockTable
from dptb.nacf.assembly import NACFTableBank
from dptb.nacf.candidate import (XC_KEY, AtomicMoments, CandidateIdentityError, CandidateInputError, CandidatePriorPlan, CandidateRecipe,
                                 ConvergenceOrderPolicy, FixedOrderPolicy, FusionSettings, OrderPolicy, PairXCTables, normalize_sources)
from dptb.nacf.envxc import EnvXCBank, EnvXCStore
from dptb.nacf.onsite import OnsiteXCEvaluator, SplineDensity
from dptb.tests.test_nacf_envxc import SPECIES, build_root, edge_overlap, make_structure

torch.set_default_dtype(torch.float64)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("candidate")
    sources, proj = build_root(root)
    store = EnvXCStore(root)
    src = {s: store.manifest["sources"][s] for s in SPECIES}            # {'source_sha256': ..., 'synthetic': True}
    species = {}
    for s, kw in SPECIES.items():
        norb = sum(2 * l + 1 for l in kw["shells"])
        species[s] = {"orbital_norb": norb, "orbital_cutoff_bohr": kw["rcut"], "orbital_shells": list(kw["shells"]),
                      "projector_norb": 0, "projector_max_cutoff_bohr": -1.0, "projector_shells": [], "projector_cutoffs_bohr": [],
                      "vna_cutoff_bohr": 3.0, "vna_projector_norb": 1, "source_sha256": src[s]["source_sha256"]}

    def radial(value, support, shells_a, shells_b, seed):
        r = np.linspace(0.0, support, 41)
        na, nb = sum(2 * l + 1 for l in shells_a), sum(2 * l + 1 for l in shells_b)
        rng = np.random.default_rng(seed)
        profile = value * (1 - r / support) ** 2
        return RadialBlockTable(r, profile[:, None, None] * rng.normal(size=(na, nb))[None], tuple(shells_a), tuple(shells_b), support)

    p2 = SimpleNamespace(species=species)
    p2.onsite_component = lambda s, k: np.eye(species[s]["orbital_norb"]) * (2.0 if k == "p2_base" else 1.0)
    p2.base_component = lambda a, b, k: radial(1.0 if k == "p2_base" else 0.4, 10.5, species[a]["orbital_shells"], species[b]["orbital_shells"], hash((a, b, k)) % 1000)
    p2.projector = lambda a, b: None
    p2.d_eff = lambda s: np.zeros((0, 0))
    p23 = SimpleNamespace(species=species, manifest_sha256="synthetic-p23", manifest={"base_p2_table_manifest_sha256": None})
    p23.has_factor = lambda a, b: True
    p23.factor = lambda a, b: radial(0.3, 8.5, (0,), species[b]["orbital_shells"], hash((a, b)) % 1000)
    p23.epsilon = lambda s: np.array([1.5 if s == "Xa" else 2.0])
    bank = NACFTableBank(p2, p23, device="cpu", backend="torch")
    xbank = EnvXCBank(store, device="cpu", backend="torch", overlap_bank=bank)
    pair_xc = PairXCTables({("Xa", "Xa"): store.table("pairmom", "Xa", "Xa"), ("Xa", "Yb"): store.table("pairmom", "Xa", "Yb"),
                            ("Yb", "Yb"): store.table("pairmom", "Yb", "Yb")}, sources=src, manifest_sha256="synthetic-pair", device="cpu", backend="torch")
    moments = AtomicMoments({"Xa": 2.0, "Yb": 3.5}, sources=src, manifest_sha256="synthetic-m2")
    # onsite: synthetic quadrature objects and clamped-cubic densities with the accepted call contract
    rng = np.random.default_rng(5)
    quads = {}

    def qgrid(s, order):
        key = (s, tuple(order))
        if key not in quads:
            n = species[s]["orbital_norb"]; points = 40 * order[0]
            quads[key] = SimpleNamespace(xyz=torch.tensor(rng.normal(size=(points, 3)) * 1.5), basis=torch.tensor(rng.normal(size=(points, n)) * 0.1))
        return quads[key]
    from scipy.interpolate import CubicSpline
    knots = np.linspace(0.0, 6.0, 25)
    density = {s: SplineDensity(torch.tensor(knots), [torch.tensor(CubicSpline(knots, np.exp(-(0.7 if s == "Xa" else 0.5) * knots)).c)]) for s in SPECIES}
    onsite = OnsiteXCEvaluator(qgrid, density, radius=27.0, engine="reference", device="cpu")
    onsite_identity = {"species_sources": src, "potential": "in-repo lda_pz81_v_dv_torch", "density_definition": "normalized neutral valence (r=0 repair) + unscaled NLCC"}
    return dict(root=root, sources=sources, store=store, bank=bank, xbank=xbank, pair_xc=pair_xc, moments=moments, onsite=onsite,
                onsite_identity=onsite_identity, src=src, species=species)


def recipe(**kw):
    base = dict(envxc_arm="mcweda", stabilization="v2", order_policy=FixedOrderPolicy((2, 2, 2)))
    base.update(kw)
    return CandidateRecipe(**base)


def make_plan(w, **kw):
    chosen = kw.pop("recipe", None) or recipe()
    args = dict(table_bank=w["bank"], envxc_bank=w["xbank"], onsite=w["onsite"], onsite_identity=w["onsite_identity"], pair_xc=w["pair_xc"], atomic_moments=w["moments"])
    args.update(kw)
    return CandidatePriorPlan(chosen, **args)


def dimer(cell=60.0):
    """Two atoms, one directed edge each way, no periodic images inside the cutoffs."""
    return make_structure(["Xa", "Yb"], [[0.0, 0.0, 0.0], [0.0, 0.0, 3.1]], np.eye(3) * cell, {s: kw["rcut"] for s, kw in SPECIES.items()})


def test_recipe_requires_explicit_supported_choices():
    for field, value in (("xc_functional", "PBE"), ("density_definition", "valence only"), ("zero_point", "none")):
        with pytest.raises(CandidateIdentityError, match="unsupported"):
            recipe(**{field: value})
    with pytest.raises(CandidateIdentityError):
        recipe(envxc_arm="d1")
    with pytest.raises(CandidateIdentityError):
        recipe(stabilization="v3")
    with pytest.raises(CandidateIdentityError):
        recipe(fusion={"radial": True, "contraction": True})
    with pytest.raises(CandidateIdentityError, match="unknown fusion"):
        recipe(fusion={"radials": True})
    with pytest.raises(CandidateIdentityError):
        recipe(order_policy="fixed")
    ident = recipe().identity()
    assert ident["schema"] == "nacf-candidate-prior/v1" and ident["envxc_arm"] == "mcweda" and ident["stabilization"] == "v2"
    assert ident["fusion"] == {"radial": False, "contraction": False}
    assert json.dumps(ident)   # serializable receipt
    # the canonical XC key is an alias of the implemented label and does not change the recipe identity
    assert recipe(xc_functional=XC_KEY).identity() == ident


def test_identity_is_validated_across_families_and_fails_closed(world):
    w = world
    plan = make_plan(w)
    fam = plan.identity["families"]
    assert set(fam) >= {"p2", "p23", "envxc", "pair_xc", "atomic_moments", "onsite"}
    assert plan.identity["validated_species"] == ["Xa", "Yb"] and len(plan.identity_sha256) == 64
    # a single differing source hash in any family is a hard error
    bad_sources = {**w["src"], "Yb": {"source_sha256": "0" * 64}}
    with pytest.raises(CandidateIdentityError, match="source_sha256 differs"):
        make_plan(w, atomic_moments=AtomicMoments({"Xa": 2.0, "Yb": 3.5}, sources=bad_sources))
    with pytest.raises(CandidateIdentityError, match="differs across families"):
        make_plan(w, onsite_identity={**w["onsite_identity"], "species_sources": bad_sources})
    # a family that declares no provenance for a species is rejected, not trusted
    with pytest.raises(CandidateIdentityError, match="declares no source hash"):
        make_plan(w, atomic_moments=AtomicMoments({"Xa": 2.0, "Yb": 3.5}, sources={"Xa": w["src"]["Xa"], "Yb": {}}))
    with pytest.raises(CandidateIdentityError, match="onsite_identity must declare"):
        make_plan(w, onsite_identity={"species_sources": w["src"]})
    # recipe/evaluator disagreements
    other = OnsiteXCEvaluator(w["onsite"].qgrid, w["onsite"].density_bank, radius=23.0, engine="reference", device="cpu")
    with pytest.raises(CandidateIdentityError, match="radius"):
        make_plan(w, onsite=other)
    with pytest.raises(CandidateIdentityError, match="density definition"):
        make_plan(w, onsite_identity={**w["onsite_identity"], "density_definition": "valence only"})
    # D2 arms need background layers; the fixture root has them, a store without them is refused
    store2 = EnvXCStore(w["root"]); store2.background_nodes = None
    with pytest.raises(CandidateIdentityError, match="background"):
        make_plan(w, envxc_bank=EnvXCBank(store2, device="cpu", backend="torch", overlap_bank=w["bank"]), recipe=recipe(envxc_arm="d2"))
    # AO shell disagreement between P2 and the envxc tables
    species = {s: dict(v) for s, v in w["species"].items()}; species["Xa"] = {**species["Xa"], "orbital_shells": [0, 1, 1], "orbital_norb": 7}
    p2 = SimpleNamespace(**{k: getattr(w["bank"].p2, k) for k in ("onsite_component", "base_component", "projector", "d_eff")}, species=species)
    with pytest.raises(CandidateIdentityError, match="AO shells"):
        make_plan(w, table_bank=NACFTableBank(p2, w["bank"].p23, device="cpu", backend="torch"))


def test_pair_xc_shells_must_match_p2_even_with_equal_ao_count(world):
    """Reviewer counterexample (R2): P2 Xa is s,s,p (five AOs); a pair table whose left side is one d shell is also
    five AOs, every matrix shape agrees and the source hashes are the same, yet the two gauges must not be added."""
    w = world
    rr = np.linspace(0.0, 10.5, 5); vv = np.zeros((5, 5, 9)); vv[:-1] = 1.0
    wrong = RadialBlockTable(rr, vv, (2,), (0, 1, 2), 10.5)
    tables = dict(w["pair_xc"].tables); tables[("Xa", "Yb")] = wrong
    with pytest.raises(CandidateIdentityError, match=r"Xa\|Yb left shells \(2,\) vs P2 Xa \(0, 0, 1\)"):
        make_plan(w, pair_xc=PairXCTables(tables, sources=w["src"], device="cpu", backend="torch"))
    # the correct decomposition with the same AO counts is accepted (the fixture tables)
    assert make_plan(w).identity["validated_species"] == ["Xa", "Yb"]


def test_onsite_xc_declaration_is_checked_against_the_recipe(world):
    """Reviewer R4: the potential label is provenance, not a free pass. Recognizable contradictions are refused,
    the canonical key is the contract, and a label that does not name the functional needs the explicit key."""
    w = world
    base = w["onsite_identity"]
    for label in ("PBE", "zero potential", "LDA PW92"):
        with pytest.raises(CandidateIdentityError, match="contradicts"):
            make_plan(w, onsite_identity={**base, "potential": label})
    with pytest.raises(CandidateIdentityError, match="must declare xc_functional"):
        make_plan(w, onsite_identity={**base, "potential": "accepted v_and_dv callable"})
    with pytest.raises(CandidateIdentityError, match="declares xc_functional"):
        make_plan(w, onsite_identity={**base, "xc_functional": "pbe"})
    with pytest.raises(CandidateIdentityError, match="contradicts"):
        make_plan(w, onsite_identity={**base, "potential": "PBE", "xc_functional": XC_KEY})
    explicit = make_plan(w, onsite_identity={**base, "potential": "accepted v_and_dv callable", "xc_functional": XC_KEY})
    assert explicit.identity["families"]["onsite"]["xc_functional"] == XC_KEY
    assert make_plan(w).identity["families"]["onsite"]["xc_functional"] == XC_KEY      # the fixture label names pz81


def test_structure_coverage_and_periodicity_are_required(world):
    w = world
    plan = make_plan(w)
    g = dimer(7.0)
    with pytest.raises(CandidateIdentityError, match="fully periodic"):
        plan.prepare({**g, "pbc": (True, True, False)})
    with pytest.raises(CandidateIdentityError, match="not covered"):
        plan.prepare({**g, "symbols": ["Xa", "Zc"]})
    partial = PairXCTables({("Xa", "Xa"): w["pair_xc"].tables[("Xa", "Xa")]}, sources=w["src"], device="cpu", backend="torch")
    with pytest.raises(CandidateIdentityError, match="pair XC table: Xa\\|Yb"):
        make_plan(w, pair_xc=partial).prepare(g)


def test_graph_arrays_are_validated_on_raw_values_before_casting(world):
    """Reviewer counterexample (R1): fractional cell shifts (+0.25/-0.25) used to be truncated to 0/0 and another graph
    was evaluated. Raw integrality, finiteness, shape and range are checked first; integral floats stay legal."""
    w = world
    plan = make_plan(w)
    g = dimer()
    assert g["edge_index"].shape == (2, 2)
    reference = plan.prepare(g)()
    as_float = {**g, "edge_index": g["edge_index"].astype(np.float64), "edge_cell_shift": g["edge_cell_shift"].astype(np.float64)}
    prepared = plan.prepare(as_float)
    for key in ("edge_index", "edge_cell_shift"):
        assert prepared.geometry[key].dtype == np.int64
        np.testing.assert_array_equal(prepared.geometry[key], g[key])          # same rows, same order
    out = prepared()
    torch.testing.assert_close(out["edge_ao_ev"], reference["edge_ao_ev"], atol=1e-12, rtol=0)
    torch.testing.assert_close(out["node_ao_ev"], reference["node_ao_ev"], atol=1e-12, rtol=0)
    fractional = dict(as_float); fractional["edge_cell_shift"] = as_float["edge_cell_shift"].copy()
    fractional["edge_cell_shift"][0, 0] = 0.25; fractional["edge_cell_shift"][1, 0] = -0.25
    with pytest.raises(CandidateInputError, match="exact integers"):
        plan.prepare(fractional)
    nonfinite = dict(as_float); nonfinite["edge_cell_shift"] = as_float["edge_cell_shift"].copy(); nonfinite["edge_cell_shift"][0, 1] = np.nan
    with pytest.raises(CandidateInputError, match="finite"):
        plan.prepare(nonfinite)
    with pytest.raises(CandidateInputError, match="outside the structure"):
        plan.prepare({**g, "edge_index": np.array([[0, 2], [2, 0]])})
    with pytest.raises(CandidateInputError, match=r"\[2, 3\]"):
        plan.prepare({**g, "edge_cell_shift": np.zeros((3, 3), dtype=np.int64)})
    with pytest.raises(CandidateInputError, match=r"\[2, E\]"):
        plan.prepare({**g, "edge_index": g["edge_index"][0]})
    with pytest.raises(CandidateInputError, match="magnitude"):
        plan.prepare({**g, "edge_cell_shift": np.array([[2**31, 0, 0], [-2**31, 0, 0]])})
    assert issubclass(CandidateInputError, ValueError) and not issubclass(CandidateInputError, CandidateIdentityError)


def test_forward_composes_exactly_the_accepted_recipe(world):
    w = world
    plan = make_plan(w)
    cut = {s: kw["rcut"] for s, kw in SPECIES.items()}
    g = make_structure(["Xa", "Yb", "Xa"], [[0.3, 0.2, 0.1], [2.9, 0.4, 1.7], [0.8, 3.2, 2.6]], np.diag([7.0, 7.5, 8.0]), cut)
    prepared = plan.prepare(g)
    out = prepared()
    E, n, wdt = g["edge_index"].shape[1], 3, prepared.width
    assert out["node_ao_ev"].shape == (n, wdt, wdt) and out["edge_ao_ev"].shape == (E, wdt, wdt)
    assert torch.isfinite(out["node_ao_ev"]).all() and torch.isfinite(out["edge_ao_ev"]).all()
    # independent recomposition from the public plans
    b = prepared.assembly()
    vna = w["bank"].prepare_edge_vna(**g)()["edge_vna_ao_ev"]
    x = w["xbank"].prepare(**g, arms=("mcweda",), topology="python", stabilization="v2")(edge_overlap_ao=b["edge_overlap_ao"])["mcweda"]
    site = w["onsite"](g, wdt, [(2, 2, 2)] * n)
    pair = torch.zeros((E, wdt, wdt))
    pos, cell, ei, sh = g["positions_bohr"], g["cell_bohr"], g["edge_index"], g["edge_cell_shift"]
    for k in range(E):
        a, bb = g["symbols"][ei[0, k]], g["symbols"][ei[1, k]]
        vec = torch.tensor(pos[ei[1, k]] + sh[k] @ cell - pos[ei[0, k]])[None]
        table = w["pair_xc"].table(a, bb)
        block = table(vec)[0] if a <= bb else table(-vec)[0].T
        pair[k, :block.shape[0], :block.shape[1]] = block
    volume = abs(np.linalg.det(cell)); c = 4 * np.pi * (2.0 + 3.5 + 2.0) * w["bank"].ry_to_ev / (3 * volume)
    assert abs(prepared.c_ev - c) < 1e-12 * c
    torch.testing.assert_close(out["node_ao_ev"], b["node_p23_ao_ev"] + site + c * b["node_overlap_ao"], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(out["edge_ao_ev"], b["edge_p2_ao_ev"] + vna + pair + c * b["edge_overlap_ao"] + x, atol=1e-12, rtol=1e-12)
    # sorted-pair convention of the accepted xc_edges: a reversed edge is the transposed block of the reversed vector, so
    # mixed pairs are exact transposes and same-species edges agree to the rounding of the harmonic rotation
    reverse = out["components"]["pair_xc_ao_ev"][prepared.assembly.reverse].transpose(-1, -2)
    torch.testing.assert_close(out["components"]["pair_xc_ao_ev"], reverse, atol=1e-12, rtol=0)
    d = out["diagnostics"]
    assert d["orders"] == [[2, 2, 2]] * n and d["envxc"]["stabilization"] == "v2" and d["recipe_identity_sha256"] == plan.identity_sha256
    # a fixed order is a choice, not a convergence proof (reviewer R5): no check was run and none is claimed
    assert all(chk["selected_order"] == (2, 2, 2) and chk["convergence_checked"] is False and chk["converged"] is None and "passed" not in chk
               for chk in d["order_checks"])
    assert d["convergence_checked"] is False and d["converged"] is None and d["onsite_bank_rebuilds"] == 0
    # the v1 recipe is a different, explicitly stated identity with a different environment term
    plan_v1 = make_plan(w, recipe=recipe(stabilization="v1"))
    out_v1 = plan_v1.prepare(g)()
    assert plan_v1.identity_sha256 != plan.identity_sha256
    x_v1 = w["xbank"].prepare(**g, arms=("mcweda",), topology="python", stabilization="v1")(edge_overlap_ao=b["edge_overlap_ao"])["mcweda"]
    torch.testing.assert_close(out_v1["components"]["envxc_ao_ev"], x_v1, atol=1e-12, rtol=1e-12)


def test_convergence_order_policy_is_geometry_only(world):
    w = world
    g = dimer(7.0)
    loose = ConvergenceOrderPolicy((2, 2, 2), (3, 3, 3), (4, 4, 4), tolerance_eV=1e9)
    plan = make_plan(w, recipe=recipe(order_policy=loose))
    prepared = plan.prepare(g)
    assert prepared.orders == [(2, 2, 2), (2, 2, 2)]
    assert all(c["convergence_checked"] and c["converged"] and c["tolerance_eV"] == 1e9 and c["medium_fine_eV"] >= 0 for c in prepared.order_checks)
    assert prepared.convergence_summary() == (True, True)
    tight = ConvergenceOrderPolicy((2, 2, 2), (3, 3, 3), (4, 4, 4), tolerance_eV=1e-30, tail_radius_bohr=23.0)
    prepared = make_plan(w, recipe=recipe(order_policy=tight)).prepare(g)
    assert prepared.orders == [(3, 3, 3), (3, 3, 3)]
    assert all(c["fine_finer_eV"] is not None and "environment_tail_eV" in c and c["convergence_checked"] and c["converged"] is False
               for c in prepared.order_checks)
    assert prepared.convergence_summary() == (True, False)          # measured and failed: recorded, not a pass
    with pytest.raises(RuntimeError, match="did not converge"):
        make_plan(w, recipe=recipe(order_policy=ConvergenceOrderPolicy((2, 2, 2), (3, 3, 3), (4, 4, 4), tolerance_eV=1e-30, strict=True))).prepare(g)
    assert loose.identity()["policy"] == "convergence" and FixedOrderPolicy((128, 24, 48)).identity() == {"policy": "fixed", "order": [128, 24, 48]}


def test_bound_recipe_and_order_policy_cannot_drift_from_the_hashed_identity(world):
    """Reviewer R6: built-in policies, the recipe and its fusion settings are frozen; a custom mutable policy is
    snapshotted at binding and the plan refuses to prepare once its identity differs from the hashed one."""
    w = world
    policy = FixedOrderPolicy((2, 2, 2))
    with pytest.raises(AttributeError):
        policy.order = (3, 3, 3)
    rec = recipe(order_policy=policy, fusion={"radial": False})
    assert isinstance(rec.fusion, FusionSettings)
    with pytest.raises(AttributeError):
        rec.fusion.radial = True
    with pytest.raises(AttributeError):
        rec.envxc_arm = "d2"
    with pytest.raises(AttributeError):
        ConvergenceOrderPolicy().medium = (1, 1, 1)

    class Mutable(OrderPolicy):
        def __init__(self, order):
            self.order = tuple(order)

        def identity(self):
            return {"policy": "mutable-test", "order": list(self.order)}

        def select(self, evaluator, g, width):
            return FixedOrderPolicy(self.order).select(evaluator, g, width)

    custom = Mutable((2, 2, 2))
    plan = make_plan(w, recipe=recipe(order_policy=custom))
    g = dimer()
    assert plan.prepare(g).orders == [(2, 2, 2)] * 2 and plan.identity["recipe"]["order_policy"]["order"] == [2, 2, 2]
    custom.order = (3, 3, 3)
    with pytest.raises(CandidateIdentityError, match="changed after the plan was constructed"):
        plan.prepare(g)
    # the caller's onsite_identity mapping is copied at binding
    ident = dict(w["onsite_identity"])
    plan = make_plan(w, onsite_identity=ident)
    ident["potential"] = "PBE"
    assert plan.identity["families"]["onsite"]["potential"] == w["onsite_identity"]["potential"]
    assert plan.prepare(g).orders == [(2, 2, 2)] * 2


def test_missing_family_and_disjoint_sources_are_not_same_source(world):
    import copy
    for sources in ({}, {s: {'upf_sha256': '0' * 64} for s in SPECIES}):
        pair = copy.copy(world['pair_xc']); pair.sources = sources
        with pytest.raises(CandidateIdentityError, match='source'):
            make_plan(world, pair_xc=pair)


def test_injected_content_changes_the_candidate_identity(world):
    import copy
    w = world
    first = AtomicMoments({'Xa': 2.0, 'Yb': 3.5}, sources=w['src'])
    second = AtomicMoments({'Xa': 2.1, 'Yb': 3.5}, sources=w['src'])
    assert make_plan(w, atomic_moments=first).identity_sha256 != make_plan(w, atomic_moments=second).identity_sha256
    tables = copy.deepcopy(w['pair_xc'].tables)
    pair_a = PairXCTables(tables, sources=w['src'], device='cpu')
    tables[('Xa', 'Yb')].coefficients.add_(0.01)
    pair_b = PairXCTables(tables, sources=w['src'], device='cpu')
    assert make_plan(w, pair_xc=pair_a).identity_sha256 != make_plan(w, pair_xc=pair_b).identity_sha256


def test_pair_coverage_uses_actual_edges_and_geometry_is_snapshotted(world):
    w = world
    pair = PairXCTables({('Xa', 'Yb'): w['pair_xc'].table('Xa', 'Yb')}, sources=w['src'], device='cpu')
    g = dimer()
    prepared = make_plan(w, pair_xc=pair).prepare(g)
    before = {k: v.copy() for k, v in prepared.geometry.items() if isinstance(v, np.ndarray)}
    for key in before:
        g[key].flat[0] += 1
        np.testing.assert_array_equal(prepared.geometry[key], before[key])


def test_source_normalization_and_manifest_adapters(tmp_path, world):
    assert normalize_sources({"sha256": {"upf": "a" * 64, "orbital": "b" * 64}}) == {"upf_sha256": "a" * 64, "orbital_sha256": "b" * 64}
    assert normalize_sources({"upf": "a" * 64, "orbital": "b" * 64, "source_sha256": "c" * 64}) == {"upf_sha256": "a" * 64, "orbital_sha256": "b" * 64, "source_sha256": "c" * 64}
    assert normalize_sources(None) == {}
    rows = {"Xa": {"M2_bohr2": 2.0, "upf_sha256": "a" * 64, "orbital_sha256": "b" * 64, "approximation": "LDA"}}
    path = tmp_path / "atomic.json"; path.write_text(json.dumps(rows))
    m = AtomicMoments.from_json(path)
    assert m.m2_bohr2("Xa") == 2.0 and m.identity()["species_sources"]["Xa"]["upf_sha256"] == "a" * 64 and m.identity()["approximation"] == "LDA"
    with pytest.raises(ValueError):
        AtomicMoments({"Xa": -1.0}, sources={})
    # pair manifest adapter: file/sha/passed/atomic_sources layout, sorted keys, checksum verified
    table = world["store"].table("pairmom", "Xa", "Yb")
    npz = tmp_path / "Xa_Yb.npz"
    np.savez(npz, distances=table.distances, values_eV=table.values, left_shells=np.array(table.left_shells), right_shells=np.array(table.right_shells), support_bohr=table.support_bohr)
    import hashlib
    manifest = {"pairs": {"Xa|Yb": {"file": str(npz), "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(), "passed": True,
                                    "atomic_sources": {"Xa": {"upf": "a" * 64, "orbital": "b" * 64}, "Yb": {"upf": "c" * 64, "orbital": "d" * 64}}}}}
    mpath = tmp_path / "pairs.json"; mpath.write_text(json.dumps(manifest))
    pairs = PairXCTables.from_manifest(mpath, device="cpu", backend="torch")
    assert pairs.has("Yb", "Xa") and pairs.identity()["pairs"] == ["Xa|Yb"] and pairs.sources["Yb"]["upf_sha256"] == "c" * 64
    manifest["pairs"]["Xa|Yb"]["sha256"] = "0" * 64; mpath.write_text(json.dumps(manifest))
    with pytest.raises(CandidateIdentityError, match="checksum"):
        PairXCTables.from_manifest(mpath, device="cpu", backend="torch")
    with pytest.raises(ValueError):
        PairXCTables({("Yb", "Xa"): table}, sources={}, device="cpu", backend="torch")
