import json
from pathlib import Path

import pytest
import torch
from e3nn import o3

from dptb.nn.build import build_model
from dptb.nn.embedding.rme_nocg_fusion_head import normalize_rme_head_mode


ORDINARY_HIDDEN = "4x0e+4x1o+4x1e+4x2e+4x2o+4x3o+4x3e+4x4e"


def test_route_matrix_modes_are_normalized():
    expected = {
        "late_rme_expansion_nocg": "late_rme_expansion_nocg",
        "late_nocg": "late_rme_expansion_nocg",
        "late_rme_cartesian_hybrid": "late_rme_cartesian_hybrid",
        "late_rme_ict_hybrid": "late_rme_cartesian_hybrid",
        "late_block_expansion_cg": "late_block_expansion_cg",
        "late_block_cartesian_projector": "late_block_cartesian_projector",
        "late_block_ict_projector": "late_block_cartesian_projector",
        "direct_ao_projector": "direct_ao_projector",
        "ao_projector": "direct_ao_projector",
    }
    for value, normalized in expected.items():
        assert normalize_rme_head_mode(value) == normalized


def test_ordinary_hidden_rme_heads_keep_rme_contract():
    from dptb.nn.embedding.late_rme_cartesian_hybrid import (
        LateRMECartesianHybridHead,
    )
    from dptb.nn.embedding.late_rme_expansion_nocg import (
        LateRMEExpansionNoCGHead,
    )

    irreps_in = o3.Irreps("4x0e+3x1o+3x1e+2x2e")
    irreps_out = o3.Irreps("2x0e+1x1o+1x1e+1x2e")
    x = torch.randn(5, irreps_in.dim, dtype=torch.float64)

    nocg = LateRMEExpansionNoCGHead(
        irreps_in, irreps_out, rank=4, init=0.0, dtype=torch.float64
    )
    ict = LateRMECartesianHybridHead(
        irreps_in,
        irreps_out,
        rank=4,
        init=0.0,
        product_scope="missing_only",
        dtype=torch.float64,
    )

    assert nocg.output_contract == "rme"
    assert nocg.performs_angular_coupling is False
    assert ict.output_contract == "rme"
    assert ict.performs_angular_coupling is True
    assert nocg(x).shape == (5, irreps_out.dim)
    assert ict(x).shape == (5, irreps_out.dim)


def test_ordinary_hidden_block_heads_bypass_rme_and_e3hamiltonian():
    from dptb.nn.embedding.late_block_cartesian_projector import (
        LateBlockCartesianProjectorHead,
    )
    from dptb.nn.embedding.late_block_expansion_cg import (
        LateBlockExpansionCGHead,
    )

    irreps_in = o3.Irreps("4x0e+4x1o+4x1e+4x2e")
    full_basis = ("s", "p")
    x = torch.randn(6, irreps_in.dim, dtype=torch.float64)

    wigner = LateBlockExpansionCGHead(
        irreps_in, full_basis, symmetrize=True, rank=4, dtype=torch.float64
    )
    ict = LateBlockCartesianProjectorHead(
        irreps_in,
        full_basis,
        symmetrize=False,
        rank=4,
        product_scope="missing_only",
        dtype=torch.float64,
    )

    assert wigner.output_contract == "ao_block"
    assert wigner.bypasses_rme
    assert wigner.uses_ict is False
    assert ict.output_contract == "ao_block"
    assert ict.bypasses_rme
    assert ict.uses_ict is True
    assert wigner(x).shape == (6, 4, 4)
    assert ict(x).shape == (6, 4, 4)


def test_ao_pair_recontract_supports_reference_and_precomputed_backends(tmp_path):
    from dptb.nn.embedding.ao_angular_projector import AOAngularProjectorHead
    from dptb.nn.embedding.ao_projector_bank import (
        build_ao_decoder_irreps,
        export_projector_bank,
    )

    full_basis = ("s", "p")
    irreps = build_ao_decoder_irreps(full_basis, channels=0)
    assert irreps.dim == 16
    x = torch.randn(3, irreps.dim, dtype=torch.float64)

    reference = AOAngularProjectorHead(
        irreps,
        full_basis,
        symmetrize=False,
        projector_backend="reference_wigner",
        rank=4,
        dtype=torch.float64,
    )
    bank_path = export_projector_bank(
        tmp_path / "ict_projectors.json", full_basis, source="ict_cartesian"
    )
    payload = json.loads(Path(bank_path).read_text(encoding="utf-8"))
    assert payload["source"] == "ict_cartesian"

    precomputed = AOAngularProjectorHead(
        irreps,
        full_basis,
        symmetrize=False,
        projector_backend="precomputed",
        projector_bank_path=bank_path,
        rank=4,
        dtype=torch.float64,
    )

    assert reference.output_contract == "ao_block"
    assert reference.uses_ict is False
    assert precomputed.output_contract == "ao_block"
    assert precomputed.uses_ict is True
    assert reference(x).shape == (3, 4, 4)
    assert precomputed(x).shape == (3, 4, 4)


def test_ao_pair_decoder_requires_complete_irreps():
    from dptb.nn.embedding.ao_angular_projector import AOAngularProjectorHead

    with pytest.raises(ValueError, match="do not cover shell pairs"):
        AOAngularProjectorHead("1x0e", ("s", "p"), symmetrize=False)


def _base_model_options(mode, prediction, extra_embedding=None):
    embedding = {
        "method": "lem_moe_v3",
        "n_layers": 1,
        "avg_num_neighbors": 2.0,
        "r_max": 4.0,
        "irreps_hidden": ORDINARY_HIDDEN,
        "env_embed_multiplicity": 4,
        "latent_dim": 8,
        "latent_channels": [8],
        "edge_one_hot_dim": 4,
        "num_experts": 1,
        "num_shared_experts": 1,
        "top_k": 1,
        "universal": True,
        "use_layer_onehot_tp": False,
        "use_out_onehot_tp": True,
        "use_interpolation_out": False,
        "tp_radial_emb": False,
        "mole_linear_mode": "indexed_ref",
        "so2_fusion_mode": "streamed_m_major_ref",
        "rme_head_mode": mode,
        "rme_fusion_rank": 4,
        "rme_fusion_init": 0.0,
    }
    if extra_embedding:
        embedding.update(extra_embedding)
    return {
        "embedding": embedding,
        "prediction": prediction,
    }


def _build_route(mode, prediction, extra_embedding=None):
    return build_model(
        common_options={
            "basis": {"H": "2s1p", "O": "3s2p1d"},
            "overlap": False,
            "dtype": "float32",
            "device": "cpu",
        },
        model_options=_base_model_options(mode, prediction, extra_embedding),
        train_options={},
        no_check=True,
    )


@pytest.mark.parametrize(
    ("mode", "head_name", "uses_ict", "prediction"),
    [
        (
            "late_rme_expansion_nocg",
            "LateRMEExpansionNoCGHead",
            False,
            {"method": "e3tb", "scale_type": "no_scale"},
        ),
        (
            "late_rme_cartesian_hybrid",
            "LateRMECartesianHybridHead",
            True,
            {"method": "e3tb", "scale_type": "no_scale"},
        ),
        (
            "late_block_expansion_cg",
            "LateBlockExpansionCGHead",
            False,
            {
                "method": "block_native",
                "block_decoder": "expansion_cg",
                "blockwise_hamiltonian": True,
                "scale_type": "no_scale",
            },
        ),
        (
            "late_block_cartesian_projector",
            "LateBlockCartesianProjectorHead",
            True,
            {
                "method": "block_native",
                "block_decoder": "cartesian_projector",
                "blockwise_hamiltonian": True,
                "scale_type": "no_scale",
            },
        ),
    ],
)
def test_ordinary_hidden_model_routes_keep_final_hidden_contract(
    mode, head_name, uses_ict, prediction
):
    model = _build_route(mode, prediction)
    embedding = model.embedding

    assert embedding.layers[-1].irreps_out == o3.Irreps(ORDINARY_HIDDEN)
    assert type(embedding.out_node).__name__ == head_name
    assert getattr(embedding.out_node, "uses_ict") is uses_ict
    if prediction["method"] == "block_native":
        assert not hasattr(model, "hamiltonian")
        assert embedding.out_node.output_contract == "ao_block"
    else:
        assert hasattr(model, "hamiltonian")
        assert embedding.out_node.output_contract == "rme"


@pytest.mark.parametrize(
    ("backend", "uses_ict"),
    [("reference_wigner", False), ("precomputed", True)],
)
def test_ao_pair_recontract_model_routes_change_final_layer_to_ao_pair_irreps(
    tmp_path, backend, uses_ict
):
    extra = {
        "rme_head_mode": "direct_ao_projector",
        "ao_projector_backend": backend,
    }
    if backend == "precomputed":
        from dptb.nn.embedding.ao_projector_bank import export_projector_bank

        bank = export_projector_bank(
            tmp_path / "ict_projectors.json", ("s", "s", "s", "p", "p", "d")
        )
        extra["ao_projector_bank_path"] = str(bank)
    model = _build_route(
        "direct_ao_projector",
        {
            "method": "block_native",
            "block_decoder": "ao_projector",
            "blockwise_hamiltonian": True,
            "scale_type": "no_scale",
        },
        extra,
    )

    embedding = model.embedding
    assert not hasattr(model, "hamiltonian")
    assert embedding.layers[-1].irreps_out.dim == 14 * 14
    assert type(embedding.out_node).__name__ == "AOAngularProjectorHead"
    assert embedding.out_node.uses_ict is uses_ict
