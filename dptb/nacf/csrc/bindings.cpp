#include <torch/extension.h>

torch::Tensor nacf_radial_cuda(
    torch::Tensor vectors, torch::Tensor knots, torch::Tensor coefficients,
    torch::Tensor degrees, torch::Tensor directions, torch::Tensor inverse,
    torch::Tensor scales, torch::Tensor ptr, torch::Tensor terms,
    torch::Tensor canonical, double support);

void nacf_pack_cuda_out(torch::Tensor blocks, torch::Tensor rows,
    torch::Tensor indices, torch::Tensor signs, torch::Tensor imaginary,
    torch::Tensor output);

torch::Tensor nacf_radial_multi_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);
void nacf_contract_add(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
void nacf_density_add(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
torch::Tensor nacf_onsite_density(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("radial_multi", &nacf_radial_multi_cuda);
  m.def("contract_add", &nacf_contract_add);
  m.def("density_add", &nacf_density_add);
  m.def("onsite_density", &nacf_onsite_density,
        "Batched onsite density of species-segmented neighbour lists on one quadrature grid (CUDA)");
  m.def("radial", &nacf_radial_cuda,
        "Fused cubic radial evaluation and ABACUS real-harmonic rotation (CUDA)");
  m.def("pack_out", &nacf_pack_cuda_out,
        "Pack AO blocks using shared species-pair templates (CUDA inference)");
}
