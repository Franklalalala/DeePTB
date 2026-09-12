#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "two_center_tables.h"
#include "two_center_cuda.cuh"

namespace py = pybind11;

namespace two_center {

void check_tensor(const torch::Tensor& t, const torch::Tensor& anchor, torch::ScalarType dtype, int dim) {
    TORCH_CHECK(t.is_cuda() && t.device()==anchor.device(), "all inputs must be CUDA tensors on the same device");
    TORCH_CHECK(t.scalar_type()==dtype && t.dim()==dim && t.is_contiguous(), "invalid tensor dtype, rank or contiguous layout");
}
void check_range(const torch::Tensor& t, int64_t low, int64_t high) {
    if(t.numel()) TORCH_CHECK(t.min().item<int64_t>()>=low && t.max().item<int64_t>()<high, "index out of range");
}
void check_displacements(const torch::Tensor& t) {
    TORCH_CHECK(t.is_cuda(), "displacements must be CUDA");
    check_tensor(t,t,torch::kFloat64,2);
    TORCH_CHECK(t.size(1)==3 && torch::isfinite(t).all().item<bool>(), "displacements must be finite Nx3");
}
void check_layout(const torch::Tensor& anchor, const torch::Tensor& ls, const torch::Tensor& zs,
    const torch::Tensor& ms, const torch::Tensor& offsets, int maximum) {
    for(const auto& t:{ls,zs,ms,offsets}) check_tensor(t,anchor,torch::kInt32,1);
    TORCH_CHECK(ls.numel()==zs.numel() && ls.numel()==ms.numel() && offsets.numel()>=2 && maximum>0, "invalid descriptor sizes");
    check_range(ls,0,5);check_range(zs,0,INT32_MAX);
    TORCH_CHECK((ms.abs()<=ls).all().item<bool>(), "invalid harmonic m");
    auto counts=offsets.slice(0,1)-offsets.slice(0,0,-1);
    TORCH_CHECK(offsets[0].item<int>()==0 && offsets[-1].item<int>()==ls.numel() &&
        (counts>=0).all().item<bool>() && (counts<=maximum).all().item<bool>(), "invalid species offsets");
}
void check_table(const torch::Tensor& anchor, const torch::Tensor& coeffs, const torch::Tensor& map,
    const std::vector<int64_t>& strides, int nr) {
    check_tensor(coeffs,anchor,torch::kFloat64,3);check_tensor(map,anchor,torch::kInt32,7);
    TORCH_CHECK(nr>=2 && coeffs.size(1)==nr-1 && coeffs.size(2)==4 && strides.size()==7, "invalid radial table dimensions");
    for(int a=0;a<7;++a) TORCH_CHECK(strides[a]==map.stride(a), "invalid map strides");
    check_range(map,-1,coeffs.size(0));
}
void check_map_layout(const torch::Tensor& map, const torch::Tensor& left_l, const torch::Tensor& left_z,
    const torch::Tensor& left_offsets,const torch::Tensor& right_l,const torch::Tensor& right_z,const torch::Tensor& right_offsets) {
    TORCH_CHECK(map.size(0)>=left_offsets.numel()-1 && map.size(3)>=right_offsets.numel()-1, "species outside table map");
    check_range(left_l,0,map.size(1));check_range(left_z,0,map.size(2));
    check_range(right_l,0,map.size(4));check_range(right_z,0,map.size(5));
    if(left_l.numel() && right_l.numel()) TORCH_CHECK(map.size(6)>left_l.max().item<int>()+right_l.max().item<int>(), "angular product outside map");
}
void check_gaunt(const torch::Tensor& anchor,const torch::Tensor& gaunt,const std::vector<int64_t>& dims) {
    check_tensor(gaunt,anchor,torch::kFloat64,3);
    TORCH_CHECK(dims.size()==3, "invalid Gaunt dimensions");
    for(int a=0;a<3;++a) TORCH_CHECK(dims[a]==gaunt.size(a), "Gaunt dimension mismatch");
    TORCH_CHECK(dims[0]>=25 && dims[1]>=25 && dims[2]>=81, "Gaunt table must cover l<=4 products");
}


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
    check_displacements(displacements);
    const c10::cuda::CUDAGuard guard(displacements.device());
    TORCH_CHECK(std::isfinite(dr) && dr>0 && std::isfinite(rmax) && rmax>0, "invalid radial grid");
    check_layout(displacements,orb_l,orb_zeta,orb_m,species_orb_offsets,max_norb);
    check_table(displacements,S_coeffs,S_index_map,index_map_strides,nr);
    check_table(displacements,T_coeffs,T_index_map,index_map_strides,nr);
    check_map_layout(S_index_map,orb_l,orb_zeta,species_orb_offsets,orb_l,orb_zeta,species_orb_offsets);
    check_map_layout(T_index_map,orb_l,orb_zeta,species_orb_offsets,orb_l,orb_zeta,species_orb_offsets);
    check_gaunt(displacements,gaunt_table,gaunt_dims);
    for(const auto& pairs:{pair_s1,pair_s2}) {
        check_tensor(pairs,displacements,torch::kInt32,1);
        TORCH_CHECK(pairs.numel()==displacements.size(0), "pair count mismatch");
        check_range(pairs,0,species_orb_offsets.numel()-1);
    }
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
    check_displacements(displacements);
    const c10::cuda::CUDAGuard guard(displacements.device());
    TORCH_CHECK(std::isfinite(dr) && dr>0 && std::isfinite(rmax) && rmax>0, "invalid radial grid");
    check_layout(displacements,orb_l,orb_zeta,orb_m,species_orb_offsets,max_norb);
    check_layout(displacements,proj_l,proj_zeta,proj_m,species_proj_offsets,max_nproj);
    check_table(displacements,Q_coeffs,Q_index_map,index_map_strides,nr);
    check_map_layout(Q_index_map,proj_l,proj_zeta,species_proj_offsets,orb_l,orb_zeta,species_orb_offsets);
    check_gaunt(displacements,gaunt_table,gaunt_dims);
    check_tensor(proj_species,displacements,torch::kInt32,1);
    check_tensor(orb_species,displacements,torch::kInt32,1);
    TORCH_CHECK(proj_species.numel()==displacements.size(0) && orb_species.numel()==displacements.size(0), "pair count mismatch");
    check_range(proj_species,0,species_proj_offsets.numel()-1);
    check_range(orb_species,0,species_orb_offsets.numel()-1);
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
    TORCH_CHECK(Q_i.is_cuda(), "Q must be CUDA");
    const c10::cuda::CUDAGuard guard(Q_i.device());
    TORCH_CHECK(nspin==1 || nspin==4, "nspin must be 1 or 4");
    check_tensor(Q_i,Q_i,torch::kFloat64,3);check_tensor(Q_j,Q_i,torch::kFloat64,3);
    check_tensor(cand_active,Q_i,torch::kBool,1);check_tensor(cand_nproj,Q_i,torch::kInt32,1);
    int mult = (nspin == 4) ? 2 : 1;
    check_tensor(D_matrices,Q_i,nspin==4?torch::kComplexDouble:torch::kFloat64,3);
    TORCH_CHECK(norb_i>0 && norb_j>0 && Q_i.size(2)==norb_i && Q_j.size(2)==norb_j &&
        Q_i.size(0)==Q_j.size(0) && Q_i.size(1)==Q_j.size(1) &&
        cand_active.numel()==Q_i.size(0) && cand_nproj.numel()==Q_i.size(0) &&
        D_matrices.size(0)==Q_i.size(0) && D_matrices.size(1)==mult*Q_i.size(1) &&
        D_matrices.size(2)==mult*Q_i.size(1), "nonlocal candidate shape mismatch");
    check_range(cand_nproj,0,Q_i.size(1)+1);
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
