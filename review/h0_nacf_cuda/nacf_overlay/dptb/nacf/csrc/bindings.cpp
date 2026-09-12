#include <torch/extension.h>

torch::Tensor nacf_radial_cuda(
    torch::Tensor vectors, torch::Tensor knots, torch::Tensor coefficients,
    torch::Tensor degrees, torch::Tensor directions, torch::Tensor inverse,
    torch::Tensor scales, torch::Tensor ptr, torch::Tensor terms,
    torch::Tensor canonical, double support);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("radial", &nacf_radial_cuda,
        "Fused cubic radial evaluation and ABACUS real-harmonic rotation (CUDA)");
}
