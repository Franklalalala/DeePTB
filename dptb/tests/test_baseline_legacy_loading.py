"""Restore legacy baseline checkpoints through raw and normalized options.

The release LEM reference is opt-in via ``DPTB_LEGACY_LEM_SOURCE``. Tests use
the actual reference class; neither reference sources nor weights are vendored.
"""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import sys

import pytest
import torch

from dptb.data import _keys
from dptb.nn.build import build_model
from dptb.nn.embedding.emb import Embedding
from dptb.nn.embedding.slem import Slem
from dptb.utils.argcheck import normalize


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


@pytest.fixture(params=["lem", "slem"])
def legacy_checkpoint(request, monkeypatch, tmp_path):
    method = request.param
    legacy_class = Slem
    if method == "lem":
        source = os.environ.get("DPTB_LEGACY_LEM_SOURCE")
        if source is None:
            pytest.skip("set DPTB_LEGACY_LEM_SOURCE to an existing release lem.py")
        source = Path(source)
        assert source.is_file(), "DPTB_LEGACY_LEM_SOURCE must be a readable file"
        module_name = "dptb.nn.embedding._legacy_checkpoint_reference"
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        with monkeypatch.context() as isolated:
            isolated.setattr(Embedding, "register", staticmethod(lambda name: lambda cls: cls))
            spec.loader.exec_module(module)
        legacy_class = module.Lem

    common, options = _model_options(method)
    # Only registry dispatch is isolated. The historical class itself receives
    # the prior kwargs and creates its real geometry-only state.
    with monkeypatch.context() as isolated:
        registry = dict(Embedding._register.items())
        registry[method + "_prior"] = legacy_class
        isolated.setattr(Embedding, "_register", registry)
        with pytest.warns(FutureWarning):
            original = build_model(common_options=common, model_options=options)
    assert isinstance(original.embedding, legacy_class)
    original.eval()
    state = original.state_dict()
    assert not any(".prior_inputs." in key for key in state)
    checkpoint = tmp_path / "legacy_geometry.pth"
    torch.save({
        "config": {"common_options": common, "model_options": options, "train_options": {}},
        "model_state_dict": state,
    }, checkpoint)
    return method, original, common, options, checkpoint


def _model_options(method):
    common = {"basis": {"C": "1s1p"}, "overlap": False, "dtype": "float64", "device": "cpu"}
    options = {
        "embedding": {
            "method": method, "n_layers": 1, "n_radial_basis": 4, "r_max": 4.0,
            "irreps_hidden": "4x0e+4x1o+4x1e+4x2e", "avg_num_neighbors": 2.0,
            "env_embed_multiplicity": 2, "latent_channels": [8], "latent_dim": 8,
            "tp_radial_emb": False, "h0_init_scope": "both", "h0_merge_mode": "add",
            "h0_node_key": _keys.NODE_P23_KEY, "h0_edge_key": _keys.EDGE_P2_KEY,
            "fallback_to_hamiltonian": False,
        },
        "prediction": {"method": "e3tb", "scale_type": "no_scale"},
    }
    return common, options


def _normalize(common, options):
    return normalize({
        "common_options": copy.deepcopy(common),
        "model_options": copy.deepcopy(options),
        "train_options": {"num_epoch": 1, "loss_options": {"train": {"method": "eigvals"}}},
        "data_options": {"train": {"root": ".", "prefix": "toy", "get_Hamiltonian": True}},
    })


def _graph(model, *, prior=True):
    mapper = model.idp
    data = {
        _keys.POSITIONS_KEY: torch.tensor([[0., 0., 0.], [1.1, .2, -.1], [-.3, 1.2, .4]]),
        _keys.EDGE_INDEX_KEY: torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        _keys.ATOM_TYPE_KEY: torch.full((3, 1), mapper.chemical_symbol_to_type["C"], dtype=torch.long),
        _keys.EDGE_TYPE_KEY: torch.full((6,), mapper.bond_to_type["C-C"], dtype=torch.long),
    }
    if prior:
        dimension = int(mapper.reduced_matrix_element)
        data[_keys.NODE_P23_KEY] = torch.arange(3 * dimension).reshape(3, dimension) / 10.0
        data[_keys.EDGE_P2_KEY] = torch.arange(6 * dimension).reshape(6, dimension) / 7.0
    return data


def _outputs(model, data):
    with torch.no_grad():
        result = model({key: value.clone() for key, value in data.items()})
    return {key: result[key] for key in (_keys.NODE_FEATURES_KEY, _keys.EDGE_FEATURES_KEY)}


@pytest.mark.parametrize("options_form", ["checkpoint", "raw", "aliases", "normalized"])
def test_legacy_geometry_checkpoint_equivalent_options_preserve_state_and_outputs(
        legacy_checkpoint, options_form):
    method, original, common, options, checkpoint = legacy_checkpoint
    override = {}
    if options_form != "checkpoint":
        override["model_options"] = copy.deepcopy(options)
    if options_form == "aliases":
        embedding = override["model_options"]["embedding"]
        embedding.pop("h0_init_scope")
        embedding.update(use_h0_init=True, use_h0_node_init=True, use_h0_edge_init=True)
        embedding["h0_fallback_to_hamiltonian"] = embedding.pop("fallback_to_hamiltonian")
        embedding["method"] = method + "_prior"
    elif options_form == "normalized":
        with pytest.warns(FutureWarning):
            normalized = _normalize(common, options)
        assert normalized["model_options"]["embedding"]["method"] == method + "_prior"
        override.update(model_options=normalized["model_options"], common_options=normalized["common_options"])
    caller_options = copy.deepcopy(override.get("model_options"))
    with pytest.warns(FutureWarning, match="geometry-only"):
        restored = build_model(checkpoint=str(checkpoint), **override)
    restored.eval()
    assert override.get("model_options") == caller_options
    assert restored.model_options["embedding"]["h0_init_scope"] == "none"
    assert original.state_dict().keys() == restored.state_dict().keys()
    for key, expected in original.state_dict().items():
        assert torch.equal(expected, restored.state_dict()[key]), key
    expected = _outputs(original, _graph(original))
    for data in (_graph(restored), _graph(restored, prior=False)):
        actual = _outputs(restored, data)
        for key in expected:
            assert torch.equal(expected[key], actual[key]), key


@pytest.mark.parametrize("changed_option, value", [("h0_merge_mode", "replace"), ("r_max", 3.5)])
def test_legacy_checkpoint_changed_options_keep_strict_loading(legacy_checkpoint, changed_option, value):
    _, _, common, options, checkpoint = legacy_checkpoint
    with pytest.warns(FutureWarning):
        override = _normalize(common, options)["model_options"]
    override["embedding"][changed_option] = value
    with pytest.raises(RuntimeError, match="Missing key"):
        build_model(checkpoint=str(checkpoint), model_options=override)


@pytest.mark.parametrize("method", ["lem_prior", "slem_prior"])
@pytest.mark.parametrize("damage", ["one_weight", "all_prior_state"])
def test_new_prior_checkpoint_missing_adapter_state_is_an_error(tmp_path, method, damage):
    common, options = _model_options(method)
    model = build_model(common_options=common, model_options=options)
    state = model.state_dict()
    if damage == "one_weight":
        del state["embedding.prior_inputs.node_projector.weight"]
    else:
        state = {key: value for key, value in state.items() if ".prior_inputs." not in key}
    checkpoint = tmp_path / "damaged_prior.pth"
    torch.save({
        "config": {"common_options": common, "model_options": options, "train_options": {}},
        "model_state_dict": state,
    }, checkpoint)
    normalized = _normalize(common, options)
    with pytest.raises(RuntimeError, match="Missing key"):
        build_model(checkpoint=str(checkpoint), model_options=normalized["model_options"])
