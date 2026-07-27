"""The `.json` model route is retired and must fail with a clear message.

build_model used to branch on a `.json` suffix, parse the file with j_loader,
and then hand the SAME path to torch.load, which raised
`_pickle.UnpicklingError: invalid load key, '{'`. The route belonged to the
SK/nnsk models this branch removed.
"""

from __future__ import annotations

import json

import pytest

from dptb.nn.build import build_model


def _write_json_model(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "common_options": {"basis": {"Si": ["3s", "3p"]}},
                "model_options": {"nnsk": {"onsite": {"method": "uniform"}}},
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def test_build_model_rejects_json_checkpoint_with_an_explicit_message(tmp_path):
    with pytest.raises(ValueError, match="retired SK route"):
        build_model(checkpoint=_write_json_model(tmp_path), common_options={})


def test_train_entrypoint_rejects_json_init_model(tmp_path):
    from dptb.entrypoints.train import _reject_retired_json_model

    with pytest.raises(ValueError, match="retired SK route"):
        _reject_retired_json_model(_write_json_model(tmp_path))


def test_pth_checkpoints_are_not_rejected_by_the_guard():
    from dptb.nn.build import _reject_retired_json_model

    _reject_retired_json_model("checkpoint/nnenv.latest.pth")
    _reject_retired_json_model(None)


def test_run_entrypoint_no_longer_advertises_json_models():
    import inspect

    from dptb.entrypoints import run as run_mod

    source = inspect.getsource(run_mod)
    # INPUT.endswith(".json") is the *config* file and must stay.
    assert 'init_model.endswith(".json")' not in source
