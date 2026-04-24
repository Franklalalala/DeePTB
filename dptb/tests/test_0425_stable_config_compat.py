import pytest


def test_argcheck_accepts_standard_legacy_route_keys():
    pytest.importorskip("dargs")
    from dptb.utils.argcheck import slem

    names = {arg.name for arg in slem()}

    assert "so2_m_linear_mode" in names
    assert "mole_linear_m0_mode" in names


def test_stable_legacy_route_key_guard_accepts_only_standard():
    pytest.importorskip("torch")
    pytest.importorskip("e3nn.o3")
    pytest.importorskip("torch_scatter")
    pytest.importorskip("torch_runstats")
    from dptb.nn.embedding.lem_moe_v3 import _normalize_stable_standard_compat_mode

    assert _normalize_stable_standard_compat_mode("so2_m_linear_mode", None) == "standard"
    assert _normalize_stable_standard_compat_mode("so2_m_linear_mode", "") == "standard"
    assert _normalize_stable_standard_compat_mode("so2_m_linear_mode", "standard") == "standard"

    with pytest.raises(ValueError, match="Triton experiment branch"):
        _normalize_stable_standard_compat_mode(
            "so2_m_linear_mode",
            "triton_complex_exact_grouped_linear",
        )
