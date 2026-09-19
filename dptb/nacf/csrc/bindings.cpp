#include <torch/extension.h>

torch::Tensor nacf_radial_cuda(
    torch::Tensor vectors, torch::Tensor knots, torch::Tensor coefficients,
    torch::Tensor degrees, torch::Tensor directions, torch::Tensor inverse,
    torch::Tensor scales, torch::Tensor ptr, torch::Tensor terms,
    torch::Tensor canonical, double support);

void nacf_pack_cuda_out(torch::Tensor blocks, torch::Tensor rows,
    torch::Tensor indices, torch::Tensor signs, torch::Tensor imaginary,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("radial", &nacf_radial_cuda,
        "Fused cubic radial evaluation and ABACUS real-harmonic rotation (CUDA)");
  m.def("pack_out", &nacf_pack_cuda_out,
        "Pack AO blocks using shared species-pair templates (CUDA inference)");
}
