#include "two_center_cuda.cuh"
#include "sph_harm_cuda.cuh"
#include <c10/cuda/CUDAStream.h>

namespace two_center {

__global__ void eval_two_center_batch_kernel(
    int n_pairs,
    const double* __restrict__ displacements,
    const int* __restrict__ pair_s1,
    const int* __restrict__ pair_s2,
    const double* __restrict__ S_coeffs,
    const double* __restrict__ T_coeffs,
    const int* __restrict__ S_index_map,
    const int* __restrict__ T_index_map,
    int64_t st0, int64_t st1, int64_t st2, int64_t st3, int64_t st4, int64_t st5, int64_t st6,
    const double* __restrict__ gaunt_table,
    int64_t g_dim2, int64_t g_dim3,
    const int* __restrict__ orb_l,
    const int* __restrict__ orb_zeta,
    const int* __restrict__ orb_m,
    const int* __restrict__ species_orb_offsets,
    double dr,
    double inv_dr,
    double rmax,
    int nr,
    int max_norb,
    double* __restrict__ out_S,
    double* __restrict__ out_T
) {
    int k = blockIdx.x;
    if (k >= n_pairs) return;

    int s1 = pair_s1[k];
    int s2 = pair_s2[k];
    int orb_start1 = species_orb_offsets[s1];
    int orb_count1 = species_orb_offsets[s1 + 1] - orb_start1;
    int orb_start2 = species_orb_offsets[s2];
    int orb_count2 = species_orb_offsets[s2 + 1] - orb_start2;

    double dx = displacements[k * 3 + 0];
    double dy = displacements[k * 3 + 1];
    double dz = displacements[k * 3 + 2];
    double R = sqrt(dx * dx + dy * dy + dz * dz);

    __shared__ double s_rly[121]; // up to Lmax = 10: (10+1)^2 = 121

    if (threadIdx.x == 0) {
        if (R <= rmax) {
            compute_rl_sph_harm_device(8, dx, dy, dz, s_rly);
        }
    }
    __syncthreads();

    int total_orbs = orb_count1 * orb_count2;
    double* out_S_k = out_S + k * max_norb * max_norb;
    double* out_T_k = out_T + k * max_norb * max_norb;

    if (R > rmax) {
        for (int idx = threadIdx.x; idx < total_orbs; idx += blockDim.x) {
            int mu1 = idx / orb_count2;
            int mu2 = idx % orb_count2;
            out_S_k[mu1 * max_norb + mu2] = 0.0;
            out_T_k[mu1 * max_norb + mu2] = 0.0;
        }
        return;
    }

    int p = (int)(R * inv_dr);
    if (p < 0) p = 0;
    if (p >= nr - 1) p = nr - 2;
    double w = R - p * dr;

    for (int idx = threadIdx.x; idx < total_orbs; idx += blockDim.x) {
        int mu1 = idx / orb_count2;
        int mu2 = idx % orb_count2;

        int l1 = orb_l[orb_start1 + mu1];
        int z1 = orb_zeta[orb_start1 + mu1];
        int m1 = orb_m[orb_start1 + mu1];

        int l2 = orb_l[orb_start2 + mu2];
        int z2 = orb_zeta[orb_start2 + mu2];
        int m2 = orb_m[orb_start2 + mu2];

        double total_S = 0.0;
        double total_T = 0.0;

        int sign = (l1 - l2 - abs(l1 - l2)) % 4 == 0 ? 1 : -1;
        int idx1 = l1 * (l1 + 1) + m1;
        int idx2 = l2 * (l2 + 1) + m2;
        int64_t base_idx = s1 * st0 + l1 * st1 + z1 * st2 + s2 * st3 + l2 * st4 + z2 * st5;

        for (int l = abs(l1 - l2); l <= l1 + l2; l += 2) {
            int64_t map_idx = base_idx + l * st6;
            int itab_S = S_index_map[map_idx];
            int itab_T = T_index_map[map_idx];

            if (itab_S >= 0 || itab_T >= 0) {
                double ang_sum = 0.0;
                for (int m = -l; m <= l; ++m) {
                    int idx3 = l * (l + 1) + m;
                    double G = gaunt_table[idx1 * g_dim2 * g_dim3 + idx2 * g_dim3 + idx3];
                    int y_idx = ylm_index(l, m);
                    ang_sum += G * s_rly[y_idx];
                }

                if (itab_S >= 0) {
                    const double* c = S_coeffs + (itab_S * (nr - 1) + p) * 4;
                    double s_by_rl = ((c[3] * w + c[2]) * w + c[1]) * w + c[0];
                    total_S += sign * s_by_rl * ang_sum;
                }
                if (itab_T >= 0) {
                    const double* c = T_coeffs + (itab_T * (nr - 1) + p) * 4;
                    double t_by_rl = ((c[3] * w + c[2]) * w + c[1]) * w + c[0];
                    total_T += sign * t_by_rl * ang_sum;
                }
            }
            sign = -sign;
        }

        out_S_k[mu1 * max_norb + mu2] = total_S;
        out_T_k[mu1 * max_norb + mu2] = total_T;
    }
}

__global__ void eval_projector_overlap_batch_kernel(
    int n_pairs,
    const double* __restrict__ displacements,
    const int* __restrict__ proj_species,
    const int* __restrict__ orb_species,
    const double* __restrict__ Q_coeffs,
    const int* __restrict__ Q_index_map,
    int64_t st0, int64_t st1, int64_t st2, int64_t st3, int64_t st4, int64_t st5, int64_t st6,
    const double* __restrict__ gaunt_table,
    int64_t g_dim2, int64_t g_dim3,
    const int* __restrict__ proj_l,
    const int* __restrict__ proj_zeta,
    const int* __restrict__ proj_m,
    const int* __restrict__ species_proj_offsets,
    const int* __restrict__ orb_l,
    const int* __restrict__ orb_zeta,
    const int* __restrict__ orb_m,
    const int* __restrict__ species_orb_offsets,
    double dr,
    double inv_dr,
    double rmax,
    int nr,
    int max_nproj,
    int max_norb,
    double* __restrict__ out_Q
) {
    int k = blockIdx.x;
    if (k >= n_pairs) return;

    int sp = proj_species[k];
    int so = orb_species[k];
    int p_start = species_proj_offsets[sp];
    int p_count = species_proj_offsets[sp + 1] - p_start;
    int o_start = species_orb_offsets[so];
    int o_count = species_orb_offsets[so + 1] - o_start;

    double dx = displacements[k * 3 + 0];
    double dy = displacements[k * 3 + 1];
    double dz = displacements[k * 3 + 2];
    double R = sqrt(dx * dx + dy * dy + dz * dz);

    __shared__ double s_rly[121];

    if (threadIdx.x == 0) {
        if (R <= rmax) {
            compute_rl_sph_harm_device(8, dx, dy, dz, s_rly);
        }
    }
    __syncthreads();

    int total_elements = p_count * o_count;
    double* out_Q_k = out_Q + k * max_nproj * max_norb;

    if (R > rmax) {
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            int ip = idx / o_count;
            int io = idx % o_count;
            out_Q_k[ip * max_norb + io] = 0.0;
        }
        return;
    }

    int p_seg = (int)(R * inv_dr);
    if (p_seg < 0) p_seg = 0;
    if (p_seg >= nr - 1) p_seg = nr - 2;
    double w = R - p_seg * dr;

    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        int ip = idx / o_count;
        int io = idx % o_count;

        int l1 = proj_l[p_start + ip];
        int z1 = proj_zeta[p_start + ip];
        int m1 = proj_m[p_start + ip];

        int l2 = orb_l[o_start + io];
        int z2 = orb_zeta[o_start + io];
        int m2 = orb_m[o_start + io];

        double total_Q = 0.0;
        int sign = (l1 - l2 - abs(l1 - l2)) % 4 == 0 ? 1 : -1;
        int idx1 = l1 * (l1 + 1) + m1;
        int idx2 = l2 * (l2 + 1) + m2;
        int64_t base_idx = sp * st0 + l1 * st1 + z1 * st2 + so * st3 + l2 * st4 + z2 * st5;

        for (int l = abs(l1 - l2); l <= l1 + l2; l += 2) {
            int64_t map_idx = base_idx + l * st6;
            int itab_Q = Q_index_map[map_idx];

            if (itab_Q >= 0) {
                double ang_sum = 0.0;
                for (int m = -l; m <= l; ++m) {
                    int idx3 = l * (l + 1) + m;
                    double G = gaunt_table[idx1 * g_dim2 * g_dim3 + idx2 * g_dim3 + idx3];
                    int y_idx = ylm_index(l, m);
                    ang_sum += G * s_rly[y_idx];
                }

                const double* c = Q_coeffs + (itab_Q * (nr - 1) + p_seg) * 4;
                double q_by_rl = ((c[3] * w + c[2]) * w + c[1]) * w + c[0];
                total_Q += sign * q_by_rl * ang_sum;
            }
            sign = -sign;
        }

        out_Q_k[ip * max_norb + io] = total_Q;
    }
}

// Scalar nonlocal assembly kernel: V_nl[a, b] += sum_cand Q_i[c]^T * D[c] * Q_j[c]
__global__ void assemble_nonlocal_scalar_kernel(
    int n_cand,
    const double* __restrict__ Q_i, // [n_cand, max_nproj, norb_i]
    const double* __restrict__ Q_j, // [n_cand, max_nproj, norb_j]
    const double* __restrict__ D_matrices, // [n_cand, max_nproj, max_nproj]
    const uint8_t* __restrict__ cand_active, // [n_cand]
    const int* __restrict__ cand_nproj, // [n_cand]
    int max_nproj,
    int norb_i,
    int norb_j,
    double* __restrict__ out_Vnl // [norb_i, norb_j]
) {
    int a = blockIdx.y * blockDim.y + threadIdx.y; // index in norb_i
    int b = blockIdx.x * blockDim.x + threadIdx.x; // index in norb_j

    if (a >= norb_i || b >= norb_j) return;

    double sum = 0.0;

    for (int c = 0; c < n_cand; ++c) {
        if (!cand_active[c]) continue;

        int np = cand_nproj[c];
        const double* Qi_c = Q_i + c * max_nproj * norb_i;
        const double* Qj_c = Q_j + c * max_nproj * norb_j;
        const double* D_c = D_matrices + c * max_nproj * max_nproj;

        // Contract Qi[:, a]^T * D * Qj[:, b]
        for (int p = 0; p < np; ++p) {
            double qi_pa = Qi_c[p * norb_i + a];
            if (qi_pa == 0.0) continue;
            for (int q = 0; q < np; ++q) {
                double qj_qb = Qj_c[q * norb_j + b];
                if (qj_qb == 0.0) continue;
                sum += qi_pa * D_c[p * max_nproj + q] * qj_qb;
            }
        }
    }

    out_Vnl[a * norb_j + b] = sum;
}

// Spinor SOC nonlocal assembly kernel: V_nl[s*norb_i + a, t*norb_j + b] += sum_cand Qi^T * D_{st} * Qj
__global__ void assemble_nonlocal_spinor_kernel(
    int n_cand,
    const double* __restrict__ Q_i, // [n_cand, max_nproj, norb_i]
    const double* __restrict__ Q_j, // [n_cand, max_nproj, norb_j]
    const double* __restrict__ D_real, // [n_cand, 2 * max_nproj, 2 * max_nproj]
    const double* __restrict__ D_imag, // [n_cand, 2 * max_nproj, 2 * max_nproj]
    const uint8_t* __restrict__ cand_active, // [n_cand]
    const int* __restrict__ cand_nproj, // [n_cand]
    int max_nproj,
    int norb_i,
    int norb_j,
    double* __restrict__ out_Vnl_real, // [2 * norb_i, 2 * norb_j]
    double* __restrict__ out_Vnl_imag
) {
    int a = blockIdx.y * blockDim.y + threadIdx.y; // [0, 2 * norb_i - 1]
    int b = blockIdx.x * blockDim.x + threadIdx.x; // [0, 2 * norb_j - 1]

    int full_ni = 2 * norb_i;
    int full_nj = 2 * norb_j;
    if (a >= full_ni || b >= full_nj) return;

    int spin_i = a / norb_i; // 0 or 1
    int spat_a = a % norb_i; // [0, norb_i - 1]
    int spin_j = b / norb_j; // 0 or 1
    int spat_b = b % norb_j; // [0, norb_j - 1]

    double sum_r = 0.0;
    double sum_i = 0.0;

    int d_stride = 2 * max_nproj;

    for (int c = 0; c < n_cand; ++c) {
        if (!cand_active[c]) continue;

        int np = cand_nproj[c];
        const double* Qi_c = Q_i + c * max_nproj * norb_i;
        const double* Qj_c = Q_j + c * max_nproj * norb_j;
        const double* Dr_c = D_real + c * d_stride * d_stride;
        const double* Di_c = D_imag + c * d_stride * d_stride;

        int p_offset = spin_i * max_nproj;
        int q_offset = spin_j * max_nproj;

        for (int p = 0; p < np; ++p) {
            double qi_pa = Qi_c[p * norb_i + spat_a];
            if (qi_pa == 0.0) continue;
            for (int q = 0; q < np; ++q) {
                double qj_qb = Qj_c[q * norb_j + spat_b];
                if (qj_qb == 0.0) continue;
                int d_idx = (p_offset + p) * d_stride + (q_offset + q);
                double factor = qi_pa * qj_qb;
                sum_r += factor * Dr_c[d_idx];
                sum_i += factor * Di_c[d_idx];
            }
        }
    }

    out_Vnl_real[a * full_nj + b] = sum_r;
    out_Vnl_imag[a * full_nj + b] = sum_i;
}

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
) {
    int n_pairs = displacements.size(0);
    if (n_pairs == 0) return;

    dim3 blocks(n_pairs);
    dim3 threads(128);

    auto stream = c10::cuda::getCurrentCUDAStream();

    eval_two_center_batch_kernel<<<blocks, threads, 0, stream>>>(
        n_pairs,
        displacements.data_ptr<double>(),
        pair_s1.data_ptr<int>(),
        pair_s2.data_ptr<int>(),
        S_coeffs.data_ptr<double>(),
        T_coeffs.data_ptr<double>(),
        S_index_map.data_ptr<int>(),
        T_index_map.data_ptr<int>(),
        index_map_strides[0], index_map_strides[1], index_map_strides[2],
        index_map_strides[3], index_map_strides[4], index_map_strides[5],
        index_map_strides[6],
        gaunt_table.data_ptr<double>(),
        gaunt_dims[1], gaunt_dims[2],
        orb_l.data_ptr<int>(),
        orb_zeta.data_ptr<int>(),
        orb_m.data_ptr<int>(),
        species_orb_offsets.data_ptr<int>(),
        dr,
        1.0 / dr,
        rmax,
        nr,
        max_norb,
        out_S.data_ptr<double>(),
        out_T.data_ptr<double>()
    );
}

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
) {
    int n_pairs = displacements.size(0);
    if (n_pairs == 0) return;

    dim3 blocks(n_pairs);
    dim3 threads(128);

    auto stream = c10::cuda::getCurrentCUDAStream();

    eval_projector_overlap_batch_kernel<<<blocks, threads, 0, stream>>>(
        n_pairs,
        displacements.data_ptr<double>(),
        proj_species.data_ptr<int>(),
        orb_species.data_ptr<int>(),
        Q_coeffs.data_ptr<double>(),
        Q_index_map.data_ptr<int>(),
        index_map_strides[0], index_map_strides[1], index_map_strides[2],
        index_map_strides[3], index_map_strides[4], index_map_strides[5],
        index_map_strides[6],
        gaunt_table.data_ptr<double>(),
        gaunt_dims[1], gaunt_dims[2],
        proj_l.data_ptr<int>(),
        proj_zeta.data_ptr<int>(),
        proj_m.data_ptr<int>(),
        species_proj_offsets.data_ptr<int>(),
        orb_l.data_ptr<int>(),
        orb_zeta.data_ptr<int>(),
        orb_m.data_ptr<int>(),
        species_orb_offsets.data_ptr<int>(),
        dr,
        1.0 / dr,
        rmax,
        nr,
        max_nproj,
        max_norb,
        out_Q.data_ptr<double>()
    );
}

void launch_assemble_nonlocal_candidates_cuda(
    const torch::Tensor& Q_i,
    const torch::Tensor& Q_j,
    const torch::Tensor& D_matrices,
    const torch::Tensor& cand_active,
    const torch::Tensor& cand_nproj,
    int norb_i,
    int norb_j,
    int nspin,
    torch::Tensor& out_Vnl
) {
    int n_cand = Q_i.size(0);
    int max_nproj = Q_i.size(1);
    auto stream = c10::cuda::getCurrentCUDAStream();

    if (nspin == 1) {
        dim3 threads(16, 16);
        dim3 blocks((norb_j + 15) / 16, (norb_i + 15) / 16);
        assemble_nonlocal_scalar_kernel<<<blocks, threads, 0, stream>>>(
            n_cand,
            Q_i.data_ptr<double>(),
            Q_j.data_ptr<double>(),
            D_matrices.data_ptr<double>(),
            reinterpret_cast<const uint8_t*>(cand_active.data_ptr<bool>()),
            cand_nproj.data_ptr<int>(),
            max_nproj,
            norb_i,
            norb_j,
            out_Vnl.data_ptr<double>()
        );
    } else {
        int full_ni = 2 * norb_i;
        int full_nj = 2 * norb_j;
        dim3 threads(16, 16);
        dim3 blocks((full_nj + 15) / 16, (full_ni + 15) / 16);
        
        // D_matrices is complex128
        auto D_real = torch::real(D_matrices).contiguous();
        auto D_imag = torch::imag(D_matrices).contiguous();
        
        // out_Vnl is complex128: [2*norb_i, 2*norb_j]
        auto out_real = torch::zeros({full_ni, full_nj}, torch::TensorOptions().dtype(torch::kFloat64).device(out_Vnl.device()));
        auto out_imag = torch::zeros({full_ni, full_nj}, torch::TensorOptions().dtype(torch::kFloat64).device(out_Vnl.device()));
        
        assemble_nonlocal_spinor_kernel<<<blocks, threads, 0, stream>>>(
            n_cand,
            Q_i.data_ptr<double>(),
            Q_j.data_ptr<double>(),
            D_real.data_ptr<double>(),
            D_imag.data_ptr<double>(),
            reinterpret_cast<const uint8_t*>(cand_active.data_ptr<bool>()),
            cand_nproj.data_ptr<int>(),
            max_nproj,
            norb_i,
            norb_j,
            out_real.data_ptr<double>(),
            out_imag.data_ptr<double>()
        );
        
        out_Vnl.copy_(torch::complex(out_real, out_imag));
    }
}

} // namespace two_center