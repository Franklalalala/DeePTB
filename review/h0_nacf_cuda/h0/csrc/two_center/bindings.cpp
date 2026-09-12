#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "two_center_tables.h"
#include "two_center_cuda.cuh"

namespace py = pybind11;

namespace two_center {

// Python wrapper for eval_two_center_batch_cuda
std::pair<torch::Tensor, torch::Tensor> eval_two_center_batch(
    const torch::Tensor& displacements,
    const torch::Tensor& pair_s1,
    const torch::Tensor& pair_s2,
    const torch::Tensor& S_coeffs,
    const torch::Tensor& T_coeffs,
    const torch::Tensor& S_index_map,
    const torch::Tensor& T_index_map,
    const std::vector<int64_t>& index_map_strides,
    const torch::Tensor& gaunt_table,
    const std::vector<int64_t>& gaunt_dims,
    const torch::Tensor& orb_l,
    const torch::Tensor& orb_zeta,
    const torch::Tensor& orb_m,
    const torch::Tensor& species_orb_offsets,
    double dr,
    double rmax,
    int nr,
    int max_norb
) {
    int n_pairs = displacements.size(0);
    auto out_S = torch::zeros({n_pairs, max_norb, max_norb}, displacements.options());
    auto out_T = torch::zeros({n_pairs, max_norb, max_norb}, displacements.options());
    if (n_pairs > 0) {
        launch_eval_two_center_batch_cuda(
            displacements, pair_s1, pair_s2,
            S_coeffs, T_coeffs,
            S_index_map, T_index_map,
            index_map_strides,
            gaunt_table, gaunt_dims,
            orb_l, orb_zeta, orb_m,
            species_orb_offsets,
            dr, rmax, nr, max_norb,
            out_S, out_T
        );
    }
    return {out_S, out_T};
}

// Python wrapper for eval_projector_overlap_batch_cuda
torch::Tensor eval_projector_overlap_batch(
    const torch::Tensor& displacements,
    const torch::Tensor& proj_species,
    const torch::Tensor& orb_species,
    const torch::Tensor& Q_coeffs,
    const torch::Tensor& Q_index_map,
    const std::vector<int64_t>& index_map_strides,
    const torch::Tensor& gaunt_table,
    const std::vector<int64_t>& gaunt_dims,
    const torch::Tensor& proj_l,
    const torch::Tensor& proj_zeta,
    const torch::Tensor& proj_m,
    const torch::Tensor& species_proj_offsets,
    const torch::Tensor& orb_l,
    const torch::Tensor& orb_zeta,
    const torch::Tensor& orb_m,
    const torch::Tensor& species_orb_offsets,
    double dr,
    double rmax,
    int nr,
    int max_nproj,
    int max_norb
) {
    int n_pairs = displacements.size(0);
    auto out_Q = torch::zeros({n_pairs, max_nproj, max_norb}, displacements.options());
    if (n_pairs > 0) {
        launch_eval_projector_overlap_batch_cuda(
            displacements,
            proj_species, orb_species,
            Q_coeffs, Q_index_map,
            index_map_strides,
            gaunt_table, gaunt_dims,
            proj_l, proj_zeta, proj_m,
            species_proj_offsets,
            orb_l, orb_zeta, orb_m,
            species_orb_offsets,
            dr, rmax, nr,
            max_nproj, max_norb,
            out_Q
        );
    }
    return out_Q;
}

// Python wrapper for assemble_nonlocal_candidates_cuda
torch::Tensor assemble_nonlocal_candidates(
    const torch::Tensor& Q_i,
    const torch::Tensor& Q_j,
    const torch::Tensor& D_matrices,
    const torch::Tensor& cand_active,
    const torch::Tensor& cand_nproj,
    int norb_i,
    int norb_j,
    int nspin
) {
    int mult = (nspin == 4) ? 2 : 1;
    auto out_dtype = (nspin == 4) ? torch::kComplexDouble : torch::kFloat64;
    auto out_Vnl = torch::zeros({mult * norb_i, mult * norb_j}, Q_i.options().dtype(out_dtype));
    if (Q_i.size(0) > 0) {
        launch_assemble_nonlocal_candidates_cuda(
            Q_i, Q_j, D_matrices,
            cand_active, cand_nproj,
            norb_i, norb_j, nspin,
            out_Vnl
        );
    }
    return out_Vnl;
}

} // namespace two_center

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CUDA Two-Center and Nonlocal Integrator Extension";
    m.def("extract_radial_table", &two_center::extract_radial_table, "Extract radial table from TwoCenterIntegrator");
    m.def("extract_gaunt_table", &two_center::extract_gaunt_table, "Extract real Gaunt table tensor");
    m.def("eval_two_center_batch", &two_center::eval_two_center_batch, "Batched evaluation of S and T on GPU");
    m.def("eval_projector_overlap_batch", &two_center::eval_projector_overlap_batch, "Batched evaluation of Q on GPU");
    m.def("assemble_nonlocal_candidates", &two_center::assemble_nonlocal_candidates, "Assembly of nonlocal Vnl from Q and D on GPU");
}
