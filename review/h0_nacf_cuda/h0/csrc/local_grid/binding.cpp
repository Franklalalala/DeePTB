#include <torch/extension.h>
#include <vector>

// Forward declarations from support_kernel.cu
std::vector<torch::Tensor> build_atom_support_cuda(
    torch::Tensor center,
    double rcut,
    torch::Tensor nmin,
    torch::Tensor counts,
    torch::Tensor shape,
    torch::Tensor cell,
    double dr,
    int n_intervals,
    int num_channels,
    torch::Tensor spline_coeffs,
    int norb,
    torch::Tensor descriptors,
    torch::Tensor field_values,
    bool has_spin_z,
    torch::Tensor spin_z_values
);

// Forward declarations from pair_kernel.cu
std::vector<torch::Tensor> contract_single_pair_cuda(
    torch::Tensor ga_lo,
    torch::Tensor ga_hi,
    torch::Tensor ga_lookup,
    torch::Tensor ga_values,
    torch::Tensor ga_potential,
    bool has_spin_z,
    torch::Tensor ga_spin_z_potential,
    torch::Tensor gb_lo,
    torch::Tensor gb_hi,
    torch::Tensor gb_lookup,
    torch::Tensor gb_values,
    torch::Tensor shift,
    double grid_weight,
    int norb_i,
    int norb_j
);

std::vector<std::vector<torch::Tensor>> contract_pairs_batch_cuda(
    const std::vector<torch::Tensor>& anchors_lo,
    const std::vector<torch::Tensor>& anchors_hi,
    const std::vector<torch::Tensor>& anchors_lookup,
    const std::vector<torch::Tensor>& anchors_values,
    const std::vector<torch::Tensor>& anchors_potential,
    bool has_spin_z,
    const std::vector<torch::Tensor>& anchors_spin_z_potential,
    torch::Tensor pair_i,
    torch::Tensor pair_j,
    torch::Tensor pair_R,
    torch::Tensor grid_shape,
    double grid_weight,
    const std::vector<int>& atom_norbs
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_atom_support_cuda", &build_atom_support_cuda, "Build atom support on GPU");
    m.def("contract_single_pair_cuda", &contract_single_pair_cuda, "Contract single pair on GPU");
    m.def("contract_pairs_batch_cuda", &contract_pairs_batch_cuda, "Contract all pairs in batch on GPU");
}
