"""Compare shared baseline prior inputs with the fixed production contract.

The production reference is read from an immutable Git object into pytest's
temporary directory. Reference source and checkpoints are not vendored.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.data.transforms import OrbitalMapper
from dptb.nn.embedding.lem_prior import InitLayer
from dptb.nn.embedding.prior_inputs import PriorInputs


PRIOR_KEYS = {
    "h0": (_keys.NODE_H0_KEY, _keys.EDGE_H0_KEY),
    "p": (_keys.NODE_P23_KEY, _keys.EDGE_P2_KEY),
}
REFERENCE_REVISION = "49cacb07cbda202d3ec209ded9dec625797dcdb2"
REFERENCE_FILE = "dptb/nn/embedding/lem_moe_v3_h0_helpers.py"
REFERENCE_SHA256 = "9950c2fea7c14dcaad317a442a6818e309db15bd6b51f294155d1146832abd1b"


@pytest.fixture(autouse=True)
def fp64_default():
    previous = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(2026)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


@pytest.fixture(scope="module")
def production_layer(tmp_path_factory):
    repo = Path(__file__).resolve().parents[2]
    try:
        source = subprocess.check_output(
            ["git", "show", f"{REFERENCE_REVISION}:{REFERENCE_FILE}"],
            cwd=repo, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("production comparison requires the fixed baseline Git object")
    assert hashlib.sha256(source).hexdigest() == REFERENCE_SHA256
    path = tmp_path_factory.mktemp("production_prior") / "h0_reference.py"
    path.write_bytes(source)
    name = "dptb.nn.embedding._fixed_prior_contract_reference"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module.H0InitLayer
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class _GeometryWithCaches(torch.nn.Module):
    """Adapt only the legacy geometry call's two optional cache arguments."""

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.idp, self.irreps_out = geometry.idp, geometry.irreps_out

    def forward(self, *args):
        assert len(args) == 8 and args[-2:] == (None, None)
        return self.geometry(*args[:6])


def _pair(production_layer, *, kind="h0", self_edges=False, **options):
    mapper = OrbitalMapper({"C": "1s1p", "H": "1s"}, method="e3tb")
    mapper.get_orbpair_maps()
    mapper.get_irreps()
    irreps_sh = o3.Irreps.spherical_harmonics(2)
    geometry = InitLayer(
        idp=mapper, num_types=2, n_radial_basis=4, r_max=4.0,
        avg_num_neighbors=2.0, irreps_sh=irreps_sh,
        env_embed_multiplicity=2, two_body_latent_channels=[8], latent_dim=8,
        dtype=torch.float64, device="cpu",
    )
    node_key, edge_key = PRIOR_KEYS[kind]
    adapter_options = dict(h0_node_key=node_key, h0_edge_key=edge_key, **options)
    production_options = dict(adapter_options)
    for new_name, old_name in (
        ("use_node", "use_h0_node_init"), ("use_edge", "use_h0_edge_init"),
        ("h0_merge_mode", "merge_mode"), ("h0_self_edge_tol", "self_edge_tol"),
    ):
        if new_name in production_options:
            production_options[old_name] = production_options.pop(new_name)
    adapter = PriorInputs(geometry, **adapter_options)
    reference = production_layer(
        _GeometryWithCaches(copy.deepcopy(geometry)), dtype=torch.float64,
        device="cpu", **production_options,
    )
    for name in ("node_projector", "edge_projector"):
        getattr(reference, name).load_state_dict(getattr(adapter, name).state_dict(), strict=True)
    atom_type = torch.tensor([mapper.chemical_symbol_to_type[s] for s in ("C", "H", "C")])
    if self_edges:
        edge_index = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0]])
        # The legacy geometric Bessel basis is singular at exact zero. These
        # finite self-edge lengths lie within the production self-edge tolerance.
        edge_length = torch.tensor([1e-9, 1e-9, 1e-9, 1.2, 5.0, 1.8])
    else:
        edge_index = torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]])
        edge_length = torch.tensor([1.2, 5.0, 1.8, 1.8, 5.0, 1.2])
    names = mapper.type_names
    bond_type = torch.tensor([
        mapper.bond_to_type[f"{names[atom_type[src]]}-{names[atom_type[dst]]}"]
        for src, dst in edge_index.T
    ])
    edge_sh = o3.spherical_harmonics(irreps_sh, torch.randn(6, 3), normalize=True)
    one_hot = torch.nn.functional.one_hot(atom_type, num_classes=2).to(torch.float64)
    args = (edge_index, atom_type, bond_type, edge_sh, edge_length, one_hot)
    data = {
        node_key: torch.randn(3, mapper.reduced_matrix_element),
        edge_key: torch.randn(6, mapper.reduced_matrix_element),
    }
    return adapter, reference, geometry, args, data


def _run_adapter(adapter, geometry, args, data):
    _, nodes, edges, _, active_edges = geometry(*args)
    edge_index, atom_type, bond_type, _, edge_length, _ = args
    return adapter(
        data, node_features=nodes, edge_features=edges, atom_type=atom_type,
        bond_type=bond_type, edge_index=edge_index, edge_length=edge_length,
        active_edges=active_edges,
    )


def _assert_contract(adapter, reference, geometry, args, data):
    actual = _run_adapter(adapter, geometry, args, data)
    expected = reference(data, *args)[1:3]
    for observed, target in zip(actual, expected):
        assert torch.isfinite(observed).all() and torch.isfinite(target).all()
        assert torch.equal(observed, target)
    return actual


@pytest.mark.parametrize("kind", ["h0", "p"])
@pytest.mark.parametrize("merge", ["replace", "add"])
@pytest.mark.parametrize("coupled", [False, True], ids=["ao_product", "coupled_rme"])
def test_complete_prior_outputs_and_gradients_match_production(production_layer, kind, merge, coupled, record_property):
    adapter, reference, geometry, args, data = _pair(
        production_layer, kind=kind, h0_merge_mode=merge, fallback_to_hamiltonian=False,
    )
    data[_keys.H0_COUPLED_RME_KEY] = coupled

    def evaluate(layer, base, production):
        inputs = {key: value.detach().clone().requires_grad_() for key, value in data.items() if torch.is_tensor(value)}
        inputs[_keys.H0_COUPLED_RME_KEY] = coupled
        edge_sh, lengths = (value.detach().clone().requires_grad_() for value in args[3:5])
        call_args = (*args[:3], edge_sh, lengths, args[5])
        output = layer(inputs, *call_args)[1:3] if production else _run_adapter(layer, base, call_args, inputs)
        parameters = (*base.parameters(), *layer.node_projector.parameters(), *layer.edge_projector.parameters())
        grads = torch.autograd.grad(
            sum(value.square().mean() for value in output),
            (*(inputs[key] for key in PRIOR_KEYS[kind]), edge_sh, lengths, *parameters), allow_unused=True,
        )
        return output, grads

    expected, expected_grads = evaluate(reference, reference.base_init.geometry, True)
    actual, actual_grads = evaluate(adapter, geometry, False)
    for observed, target in zip(actual, expected):
        assert torch.isfinite(observed).all() and torch.isfinite(target).all()
        assert torch.equal(observed, target)
    assert len(actual_grads) == len(expected_grads)
    max_gradient_error = 0.0
    for observed, target in zip(actual_grads, expected_grads):
        if target is None:
            assert observed is None
        else:
            assert torch.isfinite(observed).all()
            max_gradient_error = max(max_gradient_error, float((observed - target).abs().max()))
            # The production fast Linear and e3nn Linear accumulate parameter
            # derivatives in different orders even when forwards are bitwise.
            torch.testing.assert_close(observed, target, rtol=1e-13, atol=1e-15)
    record_property("gradient_max_abs_error", max_gradient_error)
    # Both prior inputs must influence the result; inactive edge rows cannot.
    for gradient in actual_grads[:2]:
        assert torch.count_nonzero(gradient)
    assert all(torch.count_nonzero(gradient) for gradient in actual_grads[-4:])
    active = geometry(*args)[-1]
    inactive = torch.ones(6, dtype=torch.bool)
    inactive[active] = False
    assert inactive.any() and not torch.count_nonzero(actual_grads[1][inactive])


@pytest.mark.parametrize("kind", ["h0", "p"])
@pytest.mark.parametrize("missing", ["node", "edge"])
def test_missing_prior_matches_production_boundary(production_layer, kind, missing):
    adapter, reference, geometry, args, data = _pair(production_layer, kind=kind, fallback_to_hamiltonian=False)
    data.pop(PRIOR_KEYS[kind][{"node": 0, "edge": 1}[missing]])
    nodes, edges = _assert_contract(adapter, reference, geometry, args, data)
    base = geometry(*args)[1:3]
    assert torch.equal(nodes, base[0])
    assert torch.equal(edges, base[1]) == (missing == "edge")


@pytest.mark.parametrize("kind", ["h0", "p"])
@pytest.mark.parametrize("scope", ["node", "edge"])
def test_scope_and_active_edges_match_production(production_layer, kind, scope):
    adapter, reference, geometry, args, data = _pair(
        production_layer, kind=kind, use_node=scope == "node", use_edge=scope == "edge",
        fallback_to_hamiltonian=False,
    )
    baseline = geometry(*args)
    output = _assert_contract(adapter, reference, geometry, args, data)
    assert output[1].shape[0] == baseline[-1].numel() < data[PRIOR_KEYS[kind][1]].shape[0]
    untouched = int(scope == "node")
    assert torch.equal(output[untouched], baseline[1 + untouched])
    data.pop(PRIOR_KEYS[kind][{"node": 0, "edge": 1}[scope]])
    for observed, target in zip(_assert_contract(adapter, reference, geometry, args, data), baseline[1:3]):
        assert torch.equal(observed, target)


@pytest.mark.parametrize("fallback", ["hamiltonian", "features", "custom"])
def test_default_fallback_and_training_guard_match_production(production_layer, fallback):
    keys = {
        "hamiltonian": (_keys.NODE_HAMILTONIAN_KEY, _keys.EDGE_HAMILTONIAN_KEY),
        "features": (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY),
        "custom": ("alternate_node_prior", "alternate_edge_prior"),
    }[fallback]
    options = dict(fallback_node_key=keys[0], fallback_edge_key=keys[1]) if fallback == "custom" else {}
    adapter, reference, geometry, args, primary = _pair(production_layer, **options)
    fallback_data = dict(zip(keys, primary.values()))
    # Both guards reject the default fallback during training.
    for layer, call in (
        (adapter, lambda: _run_adapter(adapter, geometry, args, fallback_data)),
        (reference, lambda: reference(fallback_data, *args)),
    ):
        layer.train()
        with pytest.raises(RuntimeError):
            call()
    adapter.eval()
    reference.eval()
    output = _assert_contract(adapter, reference, geometry, args, fallback_data)
    for observed, target in zip(output, _assert_contract(adapter, reference, geometry, args, primary)):
        assert torch.equal(observed, target)
    # The explicit training opt-in preserves the same projection arithmetic.
    permitted, permitted_reference, permitted_geometry, permitted_args, permitted_primary = _pair(
        production_layer, allow_target_fallback_in_training=True, **options,
    )
    _assert_contract(
        permitted, permitted_reference, permitted_geometry, permitted_args,
        dict(zip(keys, permitted_primary.values())),
    )


def test_primary_then_hamiltonian_then_feature_fallback_priority(production_layer):
    adapter, reference, geometry, args, data = _pair(production_layer)
    adapter.eval()
    reference.eval()
    primary = _assert_contract(adapter, reference, geometry, args, data)
    fallback_keys = (
        (_keys.NODE_HAMILTONIAN_KEY, _keys.EDGE_HAMILTONIAN_KEY),
        (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY),
    )
    for keys in fallback_keys:
        data.update({key: value * 2 for key, value in zip(keys, tuple(data.values())[:2])})
    for observed, target in zip(_assert_contract(adapter, reference, geometry, args, data), primary):
        assert torch.equal(observed, target)
    for key in PRIOR_KEYS["h0"]:
        data.pop(key)
    hamiltonian = _assert_contract(adapter, reference, geometry, args, data)
    for key in fallback_keys[1]:
        data[key] *= 3
    for observed, target in zip(_assert_contract(adapter, reference, geometry, args, data), hamiltonian):
        assert torch.equal(observed, target)


@pytest.mark.parametrize("self_edges", [False, True], ids=["no_self_edges", "self_edges"])
@pytest.mark.parametrize("merge", ["replace", "add"])
def test_self_edge_and_direct_fallback_match_production(production_layer, self_edges, merge):
    adapter, reference, geometry, args, data = _pair(
        production_layer, self_edges=self_edges, h0_node_mode="self_edge",
        h0_merge_mode=merge, fallback_to_hamiltonian=False,
    )
    initial = _assert_contract(adapter, reference, geometry, args, data)
    data[_keys.NODE_H0_KEY] *= 2
    changed = _assert_contract(adapter, reference, geometry, args, data)
    assert torch.equal(initial[0], changed[0]) == self_edges


@pytest.mark.parametrize("self_edges", [False, True], ids=["no_self_edges", "self_edges"])
def test_self_edge_evaluates_node_fallback_guard(production_layer, self_edges):
    adapter, reference, geometry, args, data = _pair(
        production_layer, self_edges=self_edges, h0_node_mode="self_edge",
    )
    data[_keys.NODE_HAMILTONIAN_KEY] = data.pop(_keys.NODE_H0_KEY)
    with pytest.raises(RuntimeError):
        _run_adapter(adapter, geometry, args, data)
    with pytest.raises(RuntimeError):
        reference(data, *args)
    adapter.allow_target_fallback_in_training = True
    reference.allow_target_fallback_in_training = True
    _assert_contract(adapter, reference, geometry, args, data)


def test_absent_prior_options_leave_geometry_module_unchanged(production_layer):
    _, _, geometry, _, _ = _pair(production_layer)
    assert PriorInputs.from_options(geometry, {}) is None
    assert PriorInputs.from_options(geometry, {"h0_init_scope": "none"}) is None
