"""DatasetBuilder.check_cutoffs must actually run.

It used to raise KeyError: 'model_options' on its very first statement
(`collect_cutoffs(model.model_options)` while collect_cutoffs indexes
`jdata["model_options"]`), and it had zero callers repo-wide.
"""

from __future__ import annotations

import inspect

import pytest

from dptb.data.build import DatasetBuilder, build_dataset


class _FakeModel:
    def __init__(self, r_max):
        self.model_options = {
            "embedding": {"method": "lem_moe_v3", "r_max": r_max},
            "prediction": {"method": "e3tb"},
        }


def _builder(r_max):
    builder = DatasetBuilder()
    builder.r_max = r_max
    builder.er_max = None
    builder.oer_max = None
    builder.if_check_cutoffs = False
    return builder


def test_check_cutoffs_no_longer_raises_keyerror():
    builder = _builder(6.0)
    builder.check_cutoffs(model=_FakeModel(5.0))
    assert builder.if_check_cutoffs is True


def test_dataset_cutoff_smaller_than_model_is_rejected():
    builder = _builder(4.0)
    with pytest.raises(ValueError, match="smaller than model"):
        builder.check_cutoffs(model=_FakeModel(5.0))


def test_dict_cutoffs_are_compared_per_key():
    builder = _builder({"Si": 6.0, "O": 4.0})
    builder.check_cutoffs(model=_FakeModel({"Si": 5.0, "O": 4.0}))

    builder = _builder({"Si": 6.0, "O": 3.0})
    with pytest.raises(ValueError, match="offending values"):
        builder.check_cutoffs(model=_FakeModel({"Si": 5.0, "O": 4.0}))


def test_missing_model_disables_the_check():
    builder = _builder(6.0)
    builder.check_cutoffs(model=None)
    assert builder.if_check_cutoffs is False


@pytest.mark.parametrize("module_name", ["dptb.entrypoints.train", "dptb.entrypoints.test"])
def test_entrypoints_call_check_cutoffs(module_name):
    import importlib

    module = importlib.import_module(module_name)
    source = inspect.getsource(module)
    assert "build_dataset.check_cutoffs(" in source


def test_build_dataset_singleton_exposes_the_check():
    assert callable(build_dataset.check_cutoffs)
