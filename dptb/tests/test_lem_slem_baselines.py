"""Observable behavior of the upstream and prior LEM/SLEM baselines.

The optional legacy comparison reads an existing source file through
``DPTB_LEGACY_LEM_SOURCE``; reference code and checkpoints are not vendored.
"""
from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import sys

import pytest
import torch
from e3nn import o3

from dptb.data import _keys
from dptb.nn.build import build_model
from dptb.nn.embedding.emb import Embedding
from dptb.nn.embedding.lem import Lem
from dptb.nn.embedding.lem_prior import LemPrior
from dptb.nn.embedding.prior_inputs import PriorInputs
from dptb.nn.embedding.slem import Slem
from dptb.nn.embedding.slem_prior import SlemPrior
from dptb.nn.tensor_product import SO2LinearCached


OUTPUT_KEYS = (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY)
PRIOR_KEYS = {
    "h0": (_keys.NODE_H0_KEY, _keys.EDGE_H0_KEY),
    "p": (_keys.NODE_P23_KEY, _keys.EDGE_P2_KEY),
}


@pytest.fixture(autouse=True)
def fp64_default():
    """The upstream SO(2) implementation allocates with the default dtype."""
    previous = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(2026)
        try:
            yield
        finally:
            torch.set_default_dtype(previous)


def _options(**extra):
    return {
        "basis": {"C": "1s1p"},
        "n_layers": 1,
        "n_radial_basis": 4,
        "r_max": 4.0,
        "irreps_hidden": "4x0e+4x1o+4x1e+4x2e",
        "avg_num_neighbors": 2.0,
        "env_embed_multiplicity": 2,
        "latent_channels": [8],
        "latent_dim": 8,
        "tp_radial_emb": False,
        "dtype": torch.float64,
        "device": "cpu",
        **extra,
    }


def _graph(model):
    mapper = model.idp
    atom_type = mapper.chemical_symbol_to_type["C"]
    bond_type = mapper.bond_to_type["C-C"]
    return {
        _keys.POSITIONS_KEY: torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, -0.1], [-0.3, 1.2, 0.4]],
        ),
        _keys.EDGE_INDEX_KEY: torch.tensor(
            [[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]], dtype=torch.long,
        ),
        _keys.ATOM_TYPE_KEY: torch.full((3, 1), atom_type, dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.full((6,), bond_type, dtype=torch.long),
    }


def _clone(data):
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}


def _outputs(model, data):
    result = model(_clone(data))
    return {key: result[key] for key in OUTPUT_KEYS}


def _assert_equal(reference, actual):
    for key in OUTPUT_KEYS:
        assert torch.equal(reference[key], actual[key]), key


def _with_priors(model, kind):
    data = _graph(model)
    dimension = int(model.idp.reduced_matrix_element)
    node_key, edge_key = PRIOR_KEYS[kind]
    data[node_key] = torch.randn(3, dimension)
    data[edge_key] = torch.randn(6, dimension)
    return data


@pytest.mark.parametrize("baseline", [Lem, Slem], ids=["lem", "slem"])
def test_upstream_baseline_forward_and_backward_are_finite(baseline):
    model = baseline(**_options())
    data = _graph(model)
    data[_keys.POSITIONS_KEY].requires_grad_()
    outputs = _outputs(model, data)
    for key, rows in zip(OUTPUT_KEYS, (3, 6)):
        assert outputs[key].shape == (rows, model.idp.reduced_matrix_element)
        assert torch.isfinite(outputs[key]).all()
    sum(value.square().mean() for value in outputs.values()).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)
    assert torch.isfinite(data[_keys.POSITIONS_KEY].grad).all()


@pytest.mark.parametrize("prior_options", [{}, {"h0_init_scope": "none"}], ids=["default", "disabled"])
def test_slem_prior_without_prior_fields_matches_upstream_bitwise(prior_options):
    rng_state = torch.get_rng_state()
    original = Slem(**_options())
    torch.set_rng_state(rng_state)
    prior = SlemPrior(**_options(**prior_options))
    assert original.state_dict().keys() == prior.state_dict().keys()
    for key, value in original.state_dict().items():
        assert torch.equal(value, prior.state_dict()[key]), key
    _assert_equal(_outputs(original, _graph(original)), _outputs(prior, _graph(prior)))
    prior.load_state_dict(original.state_dict(), strict=True)
    _assert_equal(_outputs(original, _graph(original)), _outputs(prior, _graph(prior)))


@pytest.mark.parametrize("baseline", [LemPrior, SlemPrior], ids=["lem_prior", "slem_prior"])
@pytest.mark.parametrize("kind", ["h0", "p"])
def test_each_named_prior_changes_output_and_receives_gradient(baseline, kind):
    node_key, edge_key = PRIOR_KEYS[kind]
    model = baseline(**_options(h0_node_key=node_key, h0_edge_key=edge_key, h0_merge_mode="add"))
    priors = _with_priors(model, kind)
    reference = _outputs(model, priors)
    for key in (node_key, edge_key):
        data = _clone(priors)
        data[key] = (2 * priors[key]).requires_grad_()
        actual = _outputs(model, data)
        assert any(not torch.equal(reference[name], actual[name]) for name in OUTPUT_KEYS), key
        model.zero_grad(set_to_none=True)
        sum(value.square().mean() for value in actual.values()).backward()
        assert data[key].grad is not None
        assert torch.isfinite(data[key].grad).all()
        assert torch.count_nonzero(data[key].grad), key


def _rotate_packed_ao(source, mapper, rotation):
    """Independent shell-wise AO transform, before any coupled-RME projection."""
    transformed = torch.empty_like(source)
    angular_momenta = {"s": 0, "p": 1}
    for pair, section in mapper.orbpair_maps.items():
        left, right = pair.split("-")
        l_left = angular_momenta[left[-1]]
        l_right = angular_momenta[right[-1]]
        left_rotation = o3.Irrep(l_left, (-1) ** l_left).D_from_matrix(rotation)
        right_rotation = o3.Irrep(l_right, (-1) ** l_right).D_from_matrix(rotation)
        block = source[:, section].reshape(-1, 2 * l_left + 1, 2 * l_right + 1)
        transformed[:, section] = (left_rotation @ block @ right_rotation.T).flatten(1)
    return transformed


@pytest.mark.parametrize("baseline", [LemPrior, SlemPrior], ids=["lem_prior", "slem_prior"])
@pytest.mark.parametrize("kind", ["h0", "p"])
def test_prior_baseline_is_equivariant_under_random_proper_rotation(baseline, kind, record_property):
    node_key, edge_key = PRIOR_KEYS[kind]
    model = baseline(**_options(h0_node_key=node_key, h0_edge_key=edge_key, h0_merge_mode="add"))
    data = _with_priors(model, kind)
    rotation = o3.rand_matrix(dtype=torch.float64)
    assert torch.linalg.det(rotation) > 0
    # Spherical harmonics and SO(2) rotations use the cyclic y,z,x convention.
    permutation = torch.tensor([1, 2, 0])
    spherical_rotation = rotation[permutation][:, permutation]
    rotated_data = _clone(data)
    rotated_data[_keys.POSITIONS_KEY] = data[_keys.POSITIONS_KEY] @ rotation.T
    for key in (node_key, edge_key):
        rotated_data[key] = _rotate_packed_ao(data[key], model.idp, spherical_rotation)
    representation = model.idp.orbpair_irreps.D_from_matrix(spherical_rotation)
    with torch.no_grad():
        reference = _outputs(model, data)
        actual = _outputs(model, rotated_data)
    for key in OUTPUT_KEYS:
        expected = reference[key] @ representation.T
        error = float((actual[key] - expected).abs().max())
        record_property(f"{key}_rotation_max_abs_error", error)
        assert error <= 1e-10, f"{key}: maximum FP64 rotation error {error:.3e}"


@pytest.mark.parametrize("prior_options", [{}, {"h0_init_scope": "none"}], ids=["default", "disabled"])
@pytest.mark.parametrize("capture_shift_features", [False, True])
def test_renamed_lem_preserves_external_legacy_initialization_and_checkpoint(monkeypatch, prior_options, capture_shift_features):
    source = os.environ.get("DPTB_LEGACY_LEM_SOURCE")
    if source is None:
        pytest.skip("set DPTB_LEGACY_LEM_SOURCE to an existing legacy lem.py")
    source_path = Path(source)
    assert source_path.is_file(), "DPTB_LEGACY_LEM_SOURCE must name a readable source file"
    module_name = "dptb.nn.embedding._legacy_baseline_reference"
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    # Import the reference without modifying the active public embedding registry.
    with monkeypatch.context() as isolated:
        isolated.setattr(Embedding, "register", staticmethod(lambda name: lambda cls: cls))
        spec.loader.exec_module(module)
    rng_state = torch.get_rng_state()
    legacy = module.Lem(**_options())
    torch.set_rng_state(rng_state)
    renamed = LemPrior(**_options(**prior_options))
    legacy.capture_shift_features = renamed.capture_shift_features = capture_shift_features
    legacy_state, renamed_state = legacy.state_dict(), renamed.state_dict()
    assert legacy_state.keys() == renamed_state.keys()
    for key in legacy_state:
        assert torch.equal(legacy_state[key], renamed_state[key]), key
    data = _graph(legacy)
    expected = _outputs(legacy, data)
    _assert_equal(expected, _outputs(renamed, data))
    if capture_shift_features:
        legacy_capture, renamed_capture = legacy(_clone(data)), renamed(_clone(data))
        for key in ("_shift_node_features", "_shift_active_edges"):
            assert torch.equal(legacy_capture[key], renamed_capture[key]), key
    checkpoint = io.BytesIO()
    torch.save({"state_dict": legacy_state}, checkpoint)
    checkpoint.seek(0)
    restored = LemPrior(**_options(**prior_options))
    restored.load_state_dict(torch.load(checkpoint, weights_only=True)["state_dict"], strict=True)
    _assert_equal(expected, _outputs(restored, data))


def _build_options(method="lem", **extra):
    embedding = _options()
    basis = embedding.pop("basis")
    embedding.pop("dtype")
    embedding.pop("device")
    embedding.update(method=method, h0_init_scope="both", h0_merge_mode="add")
    embedding.update(extra)
    common = {"basis": basis, "overlap": False, "dtype": "float64", "device": "cpu"}
    return common, {"embedding": embedding, "prediction": {"method": "e3tb", "scale_type": "no_scale"}}


@pytest.mark.parametrize("method", ["lem_prior", "slem_prior"])
def test_normalization_preserves_unconfigured_geometry_state(method):
    from dptb.nn.embedding.prior_inputs import PRIOR_INPUT_KEYS
    from dptb.utils.argcheck import normalize

    common, options = _build_options(method)
    options["embedding"] = {
        key: value for key, value in options["embedding"].items()
        if key not in PRIOR_INPUT_KEYS
    }
    original = build_model(common_options=common, model_options=options, train_options={})
    config = normalize({
        "common_options": common, "model_options": options,
        "train_options": {"num_epoch": 1, "loss_options": {"train": {"method": "hamil_abs"}}},
        "data_options": {"train": {"root": "."}},
    })
    restored = build_model(common_options=config["common_options"],
                           model_options=config["model_options"], train_options={})
    assert original.embedding.prior_inputs is None
    assert restored.embedding.prior_inputs is None
    restored.load_state_dict(original.state_dict(), strict=True)
    data = _with_priors(original, "h0")
    _assert_equal(_outputs(original, data), _outputs(restored, data))


def test_build_model_maps_legacy_lem_with_prior_options_and_warns():
    common, options = _build_options()
    with pytest.warns(FutureWarning, match="lem_prior"):
        model = build_model(
            common_options=common,
            model_options=options,
            train_options={},
            no_check=False,
        )
    assert isinstance(model.embedding, LemPrior)
    result = model(_with_priors(model, "h0"))
    assert torch.isfinite(result[_keys.NODE_FEATURES_KEY]).all()
    assert torch.isfinite(result[_keys.EDGE_FEATURES_KEY]).all()


@pytest.mark.parametrize("method, baseline", [("lem", LemPrior), ("slem", SlemPrior)])
def test_build_model_restores_checkpoint_with_legacy_prior_method(tmp_path, method, baseline):
    common, options = _build_options(method + "_prior")
    original = build_model(common_options=common, model_options=options, train_options={})
    data = _with_priors(original, "h0")
    with torch.no_grad():
        expected = _outputs(original, data)
    options["embedding"]["method"] = method
    checkpoint = tmp_path / "legacy_prior.pth"
    torch.save({
        "config": {"common_options": common, "model_options": options, "train_options": {}},
        "model_state_dict": original.state_dict(),
    }, checkpoint)
    with pytest.warns(FutureWarning, match=method + "_prior"):
        restored = build_model(checkpoint=str(checkpoint))
    assert isinstance(restored.embedding, baseline)
    with torch.no_grad():
        _assert_equal(expected, _outputs(restored, data))


@pytest.mark.parametrize("method, baseline", [("lem", LemPrior), ("slem", SlemPrior)])
def test_build_model_restores_legacy_geometry_checkpoint_that_ignored_prior_options(tmp_path, method, baseline):
    common, options = _build_options(method + "_prior", h0_init_scope="none")
    original = build_model(common_options=common, model_options=options, train_options={})
    data = _with_priors(original, "h0")
    with torch.no_grad():
        expected = _outputs(original, data)
    # The historical embedding accepted these options through **kwargs but
    # ignored them, leaving a geometry-only checkpoint.
    options["embedding"].update(method=method, h0_init_scope="both")
    checkpoint = tmp_path / "legacy_geometry.pth"
    torch.save({
        "config": {"common_options": common, "model_options": options, "train_options": {}},
        "model_state_dict": original.state_dict(),
    }, checkpoint)
    with pytest.warns(FutureWarning, match=method + "_prior"):
        restored = build_model(checkpoint=str(checkpoint))
    assert isinstance(restored.embedding, baseline)
    with torch.no_grad():
        _assert_equal(expected, _outputs(restored, data))
        _assert_equal(expected, _outputs(restored, _graph(restored)))
    _, override = _build_options(method, h0_merge_mode="replace")
    with pytest.warns(FutureWarning, match=method + "_prior"), pytest.raises(RuntimeError):
        build_model(checkpoint=str(checkpoint), model_options=override)
    # An explicit prior method must not hide a damaged new checkpoint.
    options["embedding"]["method"] = method + "_prior"
    torch.save({
        "config": {"common_options": common, "model_options": options, "train_options": {}},
        "model_state_dict": original.state_dict(),
    }, checkpoint)
    with pytest.raises(RuntimeError):
        build_model(checkpoint=str(checkpoint))


@pytest.mark.parametrize("baseline", [LemPrior, SlemPrior], ids=["lem_prior", "slem_prior"])
@pytest.mark.parametrize("scope", ["none", "node", "edge"])
def test_prior_scope_consumes_only_selected_fields(baseline, scope):
    model = baseline(**_options(h0_init_scope=scope, h0_merge_mode="add"))
    data = _with_priors(model, "h0")
    reference = _outputs(model, _graph(model))
    for selected_scope, key in zip(("node", "edge"), PRIOR_KEYS["h0"]):
        one_prior = _graph(model)
        one_prior[key] = data[key]
        actual = _outputs(model, one_prior)
        if scope == selected_scope:
            assert any(not torch.equal(reference[name], actual[name]) for name in OUTPUT_KEYS)
        else:
            _assert_equal(reference, actual)


def _initial_features(model):
    graph = _graph(model)
    edge_index = graph[_keys.EDGE_INDEX_KEY]
    dimension = model.init_layer.irreps_out.dim
    return {
        "node_features": torch.randn(3, dimension),
        "edge_features": torch.randn(6, dimension),
        "atom_type": graph[_keys.ATOM_TYPE_KEY].flatten(),
        "bond_type": graph[_keys.EDGE_TYPE_KEY].flatten(),
        "edge_index": edge_index,
        "edge_length": (graph[_keys.POSITIONS_KEY][edge_index[0]] - graph[_keys.POSITIONS_KEY][edge_index[1]]).norm(dim=1),
        "active_edges": torch.arange(6),
    }


def test_shared_prior_addition_is_base_plus_replacement():
    model = LemPrior(**_options(h0_init_scope="none"))
    replacement = PriorInputs(model.init_layer, h0_merge_mode="replace")
    addition = PriorInputs(model.init_layer, h0_merge_mode="add")
    addition.load_state_dict(replacement.state_dict(), strict=True)
    data, features = _with_priors(model, "h0"), _initial_features(model)
    replaced = replacement(data, **features)
    added = addition(data, **features)
    for base, projected, merged in zip((features["node_features"], features["edge_features"]), replaced, added):
        torch.testing.assert_close(merged, base + projected, rtol=0, atol=0)


def test_shared_prior_self_edge_initializes_nodes_and_falls_back_to_direct():
    model = LemPrior(**_options(h0_init_scope="none"))
    self_edge = PriorInputs(model.init_layer, h0_node_mode="self_edge")
    direct = PriorInputs(model.init_layer, h0_node_mode="direct")
    direct.load_state_dict(self_edge.state_dict(), strict=True)
    data, features = _with_priors(model, "h0"), _initial_features(model)
    features["edge_index"] = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 1, 2, 0]])
    features["edge_length"] = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    nodes, edges = self_edge(data, **features)
    assert torch.equal(nodes, edges[:3])
    changed_node_prior = _clone(data)
    changed_node_prior[_keys.NODE_H0_KEY] *= 2
    assert torch.equal(nodes, self_edge(changed_node_prior, **features)[0])
    features["edge_index"] = _graph(model)[_keys.EDGE_INDEX_KEY]
    features["edge_length"] = torch.ones(6)
    fallback = self_edge(data, **features)
    expected = direct(data, **features)
    for actual, reference in zip(fallback, expected):
        assert torch.equal(actual, reference)


def test_shared_prior_missing_fields_preserve_initialization_and_bad_rows_fail():
    model = LemPrior(**_options(h0_init_scope="none"))
    adapter = PriorInputs(model.init_layer)
    features = _initial_features(model)
    nodes, edges = adapter(_graph(model), **features)
    assert torch.equal(nodes, features["node_features"])
    assert torch.equal(edges, features["edge_features"])
    data = _with_priors(model, "h0")
    data[_keys.NODE_H0_KEY] = data[_keys.NODE_H0_KEY][:-1]
    with pytest.raises(ValueError):
        adapter(data, **features)


def test_shared_prior_target_fallback_is_guarded_during_training():
    model = LemPrior(**_options(h0_init_scope="none"))
    guarded = PriorInputs(model.init_layer, fallback_to_hamiltonian=True)
    permitted = PriorInputs(model.init_layer, fallback_to_hamiltonian=True, allow_target_fallback_in_training=True)
    permitted.load_state_dict(guarded.state_dict(), strict=True)
    data, features = _with_priors(model, "h0"), _initial_features(model)
    data[_keys.NODE_HAMILTONIAN_KEY] = data.pop(_keys.NODE_H0_KEY)
    data[_keys.EDGE_HAMILTONIAN_KEY] = data.pop(_keys.EDGE_H0_KEY)
    with pytest.raises(RuntimeError):
        guarded(data, **features)
    allowed = permitted(data, **features)
    guarded.eval()
    for actual, expected in zip(guarded(data, **features), allowed):
        assert torch.equal(actual, expected)


def test_baseline_and_cached_so2_classes_preserve_math():
    from dptb.nn.embedding.lem import SO2_Linear as BaselineSO2Linear

    irreps = o3.Irreps("2x0e+2x1o+2x2e")
    original = BaselineSO2Linear(irreps, irreps, latent_dim=8)
    extended = SO2LinearCached(irreps, irreps, latent_dim=8)
    extended.load_state_dict(original.state_dict(), strict=True)
    features = torch.randn(6, irreps.dim, requires_grad=True)
    vectors = torch.randn(6, 3, requires_grad=True)
    latents = torch.randn(6, 8)

    tensor_result = original(features, vectors, latents)
    cached_result, cache = extended(features, vectors, latents)
    assert torch.is_tensor(tensor_result) and cache is not None
    assert torch.equal(tensor_result, cached_result)
    gradients = [torch.autograd.grad(result.square().sum(),
                 (features, vectors, *layer.parameters()), allow_unused=True)
                 for result, layer in ((tensor_result, original), (cached_result, extended))]
    for expected, actual in zip(*gradients):
        if expected is None:
            assert actual is None
        else:
            assert torch.isfinite(actual).all() and torch.equal(expected, actual)
