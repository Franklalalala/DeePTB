import pytest


def test_oeq_get_feasible_tp_unit_paths_match_existing_instruction_shape():
    o3 = pytest.importorskip("e3nn.o3")

    from dptb.nn.embedding.oeq_tp import get_feasible_tp

    irreps_mid, instructions = get_feasible_tp(
        o3.Irreps("2x0e + 1x1o"),
        o3.Irreps("3x0e"),
        o3.Irreps("4x0e + 1x1o"),
        tp_mode="uvw",
        trainable=True,
        path_normalization="unit",
        sort_irreps=False,
    )

    assert irreps_mid == o3.Irreps("4x0e + 1x1o")
    assert instructions == [
        (0, 0, 0, "uvw", True, 1.0),
        (1, 0, 1, "uvw", True, 1.0),
    ]


def test_oeq_get_feasible_tp_path_normalization_sorts_outputs():
    o3 = pytest.importorskip("e3nn.o3")

    from dptb.nn.embedding.oeq_tp import get_feasible_tp

    irreps_mid, instructions = get_feasible_tp(
        o3.Irreps("1x1o + 2x0e"),
        o3.Irreps("1x0e"),
        o3.Irreps("2x0e + 1x1o"),
        tp_mode="uvw",
        trainable=True,
        path_normalization="e3nn",
        sort_irreps=True,
    )

    assert irreps_mid == o3.Irreps("2x0e + 1x1o")
    assert len(instructions) == 2
    assert instructions[0][:5] == (0, 0, 1, "uvw", True)
    assert instructions[1][:5] == (1, 0, 0, "uvw", True)
    assert all(path_weight > 0 for *_, path_weight in instructions)


def test_scalar_side_direct_tp_matches_e3nn_reference():
    torch = pytest.importorskip("torch")
    o3 = pytest.importorskip("e3nn.o3")

    from dptb.nn.embedding.oeq_tp import OEQTensorProduct, get_feasible_tp

    irreps_in1 = o3.Irreps("2x0e + 1x1o")
    irreps_in2 = o3.Irreps("3x0e")
    irreps_out = o3.Irreps("4x0e + 2x1o")
    irreps_mid, instructions = get_feasible_tp(
        irreps_in1,
        irreps_in2,
        irreps_out,
        tp_mode="uvw",
        trainable=True,
        path_normalization="unit",
        sort_irreps=False,
    )
    direct = OEQTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        internal_weights=False,
        backend="scalar_direct",
    )
    reference = o3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_mid,
        instructions,
        internal_weights=False,
        shared_weights=True,
    )

    torch.manual_seed(0)
    x = torch.randn(5, irreps_in1.dim)
    y = torch.randn(5, irreps_in2.dim)
    oeq_weight = torch.randn(direct.weight_numel)
    e3nn_weights = []
    offset = 0
    for i_in1, i_in2, i_out, mode, _, _ in instructions:
        assert mode == "uvw"
        mul_1 = irreps_in1[i_in1].mul
        mul_2 = irreps_in2[i_in2].mul
        mul_out = irreps_mid[i_out].mul
        block_numel = mul_1 * mul_2 * mul_out
        e3nn_weights.append(
            oeq_weight[offset : offset + block_numel]
            .reshape(mul_2, mul_1, mul_out)
            .permute(1, 0, 2)
            .reshape(-1)
        )
        offset += block_numel
    e3nn_weight = torch.cat(e3nn_weights)

    assert direct.weight_numel == reference.weight_numel
    torch.testing.assert_close(direct(x, y, oeq_weight), reference(x, y, e3nn_weight), atol=1e-6, rtol=1e-6)


def test_scalar_side_direct_tp_matches_oeq_when_available():
    torch = pytest.importorskip("torch")
    pytest.importorskip("openequivariance")
    o3 = pytest.importorskip("e3nn.o3")
    if not torch.cuda.is_available():
        pytest.skip("OpenEquivariance comparison needs CUDA")

    from dptb.nn.embedding.oeq_tp import OEQTensorProduct

    irreps_in1 = o3.Irreps("2x0e + 1x1o")
    irreps_in2 = o3.Irreps("3x0e")
    irreps_out = o3.Irreps("4x0e + 2x1o")
    direct = OEQTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        internal_weights=False,
        backend="scalar_direct",
    ).cuda()
    oeq_tp = OEQTensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        internal_weights=False,
        backend="oeq",
    ).cuda()

    torch.manual_seed(1)
    x = torch.randn(7, irreps_in1.dim, device="cuda")
    y = torch.randn(7, irreps_in2.dim, device="cuda")
    weight = torch.randn(direct.weight_numel, device="cuda")

    assert direct.weight_numel == oeq_tp.weight_numel
    torch.testing.assert_close(direct(x, y, weight), oeq_tp(x, y, weight), atol=1e-5, rtol=1e-5)
