#ifndef TWO_CENTER_CUDA_CUH_
#define TWO_CENTER_CUDA_CUH_

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

namespace two_center {

// Evaluates S and T matrices for a batch of orbital pairs on GPU
// displacements: [N_pairs, 3] (float64)
// pair_s1: [N_pairs] (int32)
// pair_s2: [N_pairs] (int32)
// S_coeffs, T_coeffs: [ntab, nr - 1, 4] (float64)
// S_index_map, T_index_map: flattened 7D index maps (int32)
// index_map_strides: [7] (int64)
// gaunt_table: flattened 3D Gaunt table (float64)
// gaunt_dims: [3] (int64)
// orb_l, orb_zeta, orb_m: flattened orbital channel arrays (int32)
// species_orb_offsets: [ntype + 1] (int32)
// dr: radial grid step (double)
// rmax: radial cutoff (double)
// nr: number of radial points (int)
// max_norb: maximum number of orbitals across species (int)
void launch_eval_two_center_batch_cuda(
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
    int max_norb,
    torch::Tensor& out_S,
    torch::Tensor& out_T
);

// Evaluates Q (projector-orbital overlap) matrices on GPU
void launch_eval_projector_overlap_batch_cuda(
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
    int max_norb,
    torch::Tensor& out_Q
);

// Assembles nonlocal block V_nl = sum_P Q_Pi^T D_P Q_Pj for a batch of candidate projectors on GPU
// Supports scalar (nspin=1) and spinor SOC (nspin=4)
void launch_assemble_nonlocal_candidates_cuda(
    const torch::Tensor& Q_i, // [N_cand, max_nproj, norb_i] (float64)
    const torch::Tensor& Q_j, // [N_cand, max_nproj, norb_j] (float64)
    const torch::Tensor& D_matrices, // [N_cand, 2*max_nproj, 2*max_nproj] (complex128 or float64)
    const torch::Tensor& cand_active, // [N_cand] (bool/uint8)
    const torch::Tensor& cand_nproj, // [N_cand] (int32)
    int norb_i,
    int norb_j,
    int nspin,
    torch::Tensor& out_Vnl // [nspin_mult * norb_i, nspin_mult * norb_j]
);

} // namespace two_center

#endif // TWO_CENTER_CUDA_CUH_