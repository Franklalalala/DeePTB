"""End-to-end config/registration contract for the manifold-alignment losses.

The dense-math smoke tests construct the loss directly and therefore never exercised
``argcheck.normalize()`` -- which is exactly why the missing ``grassmann_p_align``
argcheck variant slipped through until an actual ``dptb train`` failed.  These tests
run the config through ``normalize()`` and then build the loss via the ``Loss``
registry, locking three contracts:

1. the loss ``method`` is a registered argcheck ``Variant`` (config validation passes);
2. the ``Loss`` registry resolves the method (lazy-import shim works);
3. the constructed loss exposes ``last_scalar_state`` (the telemetry dict the trainer
   now surfaces).
"""
import pytest

from dptb.utils.argcheck import loss_options
from dptb.nnops.loss import Loss


def _normalize_train_block(block):
    """Run a single loss block through the real argcheck loss_options schema."""
    arg = loss_options()
    normalized = arg.normalize_value({"train": dict(block)}, trim_pattern="_*")
    return normalized["train"]


@pytest.mark.parametrize(
    "block, expect_cls",
    [
        ({"method": "grassmann_p_align", "n_occ": 5, "chordal_normalize": True,
          "all_electron": True, "min_gap": 0.05}, "GrassmannPAlignLoss"),
        ({"method": "p_regression", "from_density": True}, "GrassmannPAlignLoss"),
        ({"method": "riemannian_align", "lambda_pp": 1.0, "max_kpoints": 4}, "RiemannianAlignmentLoss"),
        ({"method": "fermi_projector_align"}, "RiemannianAlignmentLoss"),
    ],
)
def test_config_normalizes_and_constructs(block, expect_cls):
    # 1. config validation passes and fills defaults
    norm = _normalize_train_block(block)
    assert norm["method"] == block["method"]
    # a schema default that the caller did not supply must be present
    assert "coeff_align" in norm

    # 2. registry resolves the method (lazy import) and 3. telemetry contract exists
    loss = Loss(**norm, idp=None)
    assert type(loss).__name__ == expect_cls
    assert hasattr(loss, "last_scalar_state")
    assert isinstance(loss.last_scalar_state, dict)


def test_unknown_method_is_rejected():
    with pytest.raises(Exception):
        _normalize_train_block({"method": "definitely_not_a_registered_loss"})


def test_grassmann_all_kwargs_have_schema():
    """Every user-facing GrassmannPAlignLoss kwarg must be declared in argcheck.

    Guards against the __init__ <-> argcheck 1:1 mirror drifting (a new __init__ kwarg
    added without a matching Argument would be silently dropped by normalize()).
    """
    import inspect
    from dptb.nnops.grassmann import GrassmannPAlignLoss

    sig = inspect.signature(GrassmannPAlignLoss.__init__)
    framework = {"self", "basis", "idp", "overlap", "dtype", "device", "kwargs"}

    # supply every user kwarg (at its own default) and require normalize() to accept
    # them all -- an undeclared kwarg would trip dargs' "extra key" check.
    full = {"method": "grassmann_p_align"}
    for name, p in sig.parameters.items():
        if name in framework:
            continue
        if p.default is not inspect._empty and p.default is not None:
            full[name] = p.default
    norm = _normalize_train_block(full)
    assert norm["method"] == "grassmann_p_align"


if __name__ == "__main__":
    import sys
    raise SystemExit(pytest.main([__file__, "-q"]))
