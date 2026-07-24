from __future__ import annotations

import copy
import weakref

import pytest
import torch

from dptb.data import _keys
from dptb.nn.embedding.lem_pair import PairInitLayer, PairLayer

from test_lem_pair_common import fp64_default, model, molecule_data


_OUTPUT_KEYS = (
    _keys.NODE_HAMILTONIAN_KEY,
    _keys.EDGE_HAMILTONIAN_KEY,
    _keys.EDGE_OVERLAP_KEY,
)


@pytest.fixture(scope="module")
def lifecycle_model():
    with fp64_default():
        return model(mp_cutoff=1.0, pair_refine_enable=True)


def _outputs(pair_model):
    with torch.no_grad():
        result = pair_model(molecule_data(pair_model))
    return {key: result[key].detach().clone() for key in _OUTPUT_KEYS}


def _assert_equal_outputs(reference, actual):
    for key in _OUTPUT_KEYS:
        assert torch.equal(reference[key], actual[key])


def _assert_no_reference_to_original(value, original_ids, visited):
    value_id = id(value)
    if value_id in visited:
        return
    visited.add(value_id)
    if isinstance(value, weakref.ReferenceType):
        target = value()
        assert target is None or id(target) not in original_ids
        return
    if isinstance(value, torch.nn.Module):
        assert value_id not in original_ids
        for nested in vars(value).values():
            _assert_no_reference_to_original(nested, original_ids, visited)
        return
    if isinstance(value, dict):
        for nested in value.values():
            _assert_no_reference_to_original(nested, original_ids, visited)
        return
    if isinstance(value, (tuple, list)):
        for nested in value:
            _assert_no_reference_to_original(nested, original_ids, visited)


def test_deepcopy_is_independent_and_pair_forward_has_no_mutable_context(
    lifecycle_model,
):
    pair_modules = [
        module
        for module in lifecycle_model.modules()
        if isinstance(module, (PairInitLayer, PairLayer))
    ]
    attribute_keys = [set(vars(module)) for module in pair_modules]
    buffer_values = [
        {
            name: value.detach().clone()
            for name, value in module.named_buffers(recurse=False)
            if value is not None
        }
        for module in pair_modules
    ]

    clone = copy.deepcopy(lifecycle_model)
    reference = _outputs(lifecycle_model)
    actual = _outputs(clone)
    _assert_equal_outputs(reference, actual)

    for module, keys, buffers in zip(pair_modules, attribute_keys, buffer_values):
        assert set(vars(module)) == keys
        for name, value in buffers.items():
            assert torch.equal(module.get_buffer(name), value)

    original_ids = {
        id(value)
        for module in lifecycle_model.modules()
        for value in (module, *module.parameters(recurse=False), *module.buffers(recurse=False))
    }
    _assert_no_reference_to_original(clone, original_ids, set())

    original_parameter = next(lifecycle_model.parameters()).detach().clone()
    with torch.no_grad():
        next(clone.parameters()).add_(1.0)
    assert torch.equal(next(lifecycle_model.parameters()), original_parameter)
    _assert_equal_outputs(reference, _outputs(lifecycle_model))


def test_state_dict_round_trip_is_strict_and_exact(lifecycle_model):
    with fp64_default():
        target = model(
            seed=20260724,
            mp_cutoff=1.0,
            pair_refine_enable=True,
        )
    target.load_state_dict(lifecycle_model.state_dict(), strict=True)
    _assert_equal_outputs(_outputs(lifecycle_model), _outputs(target))


def test_whole_model_torch_save_load_round_trip_is_exact(
    lifecycle_model, tmp_path
):
    path = tmp_path / "lem_pair_whole_model.pt"
    reference = _outputs(lifecycle_model)
    torch.save(lifecycle_model, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    _assert_equal_outputs(reference, _outputs(restored))
