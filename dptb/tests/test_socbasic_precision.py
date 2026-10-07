"""SOC constants are generated at the requested precision, independent of defaults."""
import pytest
import torch

from dptb.nn.sktb.socbasic import get_soc_matrix_cubic_basis, lm2cubic_mat


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("orbital,l", [("s", 0), ("p", 1), ("d", 2)])
def test_soc_precision_and_angular_momentum_spectrum(default_dtype, dtype, orbital, l):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        matrix = get_soc_matrix_cubic_basis(orbital, dtype=dtype)
        assert matrix.dtype == (torch.complex128 if dtype == torch.float64 else torch.complex64)
        assert matrix.device.type == "cpu"
        # L.S has j=l+1/2 and j=l-1/2 multiplets. This catches silent FP32
        # constants upcast into a nominally complex128 output.
        expected = torch.tensor([-(l + 1) / 2] * (2 * l) + [l / 2] * (2 * l + 2), dtype=dtype)
        tol = 1e-14 if dtype == torch.float64 else 1e-6
        torch.testing.assert_close(matrix, matrix.mH, rtol=tol, atol=tol)
        torch.testing.assert_close(torch.linalg.eigvalsh(matrix), expected, rtol=tol, atol=tol)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("function,args", [(get_soc_matrix_cubic_basis, ("p",)),
                                          (lm2cubic_mat, ([-1, 0, 1], [-1, 0, 1]))])
def test_soc_rejects_unsupported_precision(function, args):
    with pytest.raises(ValueError, match="float32.*float64"):
        function(*args, dtype=torch.float16)
