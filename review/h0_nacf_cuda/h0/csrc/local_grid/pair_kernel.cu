#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

// Fast GPU pair intersection and contraction for a single pair
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
    torch::Tensor shift, // [3] int64
    double grid_weight,
    int norb_i,
    int norb_j
) {
    auto device = ga_values.device();
    auto opt_f64 = torch::dtype(torch::kFloat64).device(device);

    auto ga_lo_cpu = ga_lo.to(torch::kCPU).contiguous();
    auto ga_hi_cpu = ga_hi.to(torch::kCPU).contiguous();
    auto gb_lo_cpu = gb_lo.to(torch::kCPU).contiguous();
    auto gb_hi_cpu = gb_hi.to(torch::kCPU).contiguous();
    auto shift_cpu = shift.to(torch::kCPU).contiguous();

    auto ga_lo_a = ga_lo_cpu.accessor<int64_t, 1>();
    auto ga_hi_a = ga_hi_cpu.accessor<int64_t, 1>();
    auto gb_lo_a = gb_lo_cpu.accessor<int64_t, 1>();
    auto gb_hi_a = gb_hi_cpu.accessor<int64_t, 1>();
    auto shift_a = shift_cpu.accessor<int64_t, 1>();

    int64_t lo0 = std::max(ga_lo_a[0], gb_lo_a[0] + shift_a[0]);
    int64_t hi0 = std::min(ga_hi_a[0], gb_hi_a[0] + shift_a[0]);
    int64_t lo1 = std::max(ga_lo_a[1], gb_lo_a[1] + shift_a[1]);
    int64_t hi1 = std::min(ga_hi_a[1], gb_hi_a[1] + shift_a[1]);
    int64_t lo2 = std::max(ga_lo_a[2], gb_lo_a[2] + shift_a[2]);
    int64_t hi2 = std::min(ga_hi_a[2], gb_hi_a[2] + shift_a[2]);

    if (hi0 < lo0 || hi1 < lo1 || hi2 < lo2) {
        auto out = torch::zeros({norb_i, norb_j}, opt_f64);
        auto out_z = has_spin_z ? torch::zeros({norb_i, norb_j}, opt_f64) : torch::empty({0}, opt_f64);
        return {out, out_z};
    }

    // Slices for lookup
    int64_t sa0_start = lo0 - ga_lo_a[0];
    int64_t sa0_end   = hi0 - ga_lo_a[0] + 1;
    int64_t sa1_start = lo1 - ga_lo_a[1];
    int64_t sa1_end   = hi1 - ga_lo_a[1] + 1;
    int64_t sa2_start = lo2 - ga_lo_a[2];
    int64_t sa2_end   = hi2 - ga_lo_a[2] + 1;

    int64_t sb0_start = lo0 - shift_a[0] - gb_lo_a[0];
    int64_t sb0_end   = hi0 - shift_a[0] - gb_lo_a[0] + 1;
    int64_t sb1_start = lo1 - shift_a[1] - gb_lo_a[1];
    int64_t sb1_end   = hi1 - shift_a[1] - gb_lo_a[1] + 1;
    int64_t sb2_start = lo2 - shift_a[2] - gb_lo_a[2];
    int64_t sb2_end   = hi2 - shift_a[2] - gb_lo_a[2] + 1;

    auto sub_a = ga_lookup.slice(0, sa0_start, sa0_end)
                          .slice(1, sa1_start, sa1_end)
                          .slice(2, sa2_start, sa2_end);

    auto sub_b = gb_lookup.slice(0, sb0_start, sb0_end)
                          .slice(1, sb1_start, sb1_end)
                          .slice(2, sb2_start, sb2_end);

    auto mask = (sub_a >= 0) & (sub_b >= 0);
    auto ia = sub_a.masked_select(mask).to(torch::kInt64);
    auto ib = sub_b.masked_select(mask).to(torch::kInt64);

    if (ia.size(0) == 0) {
        auto out = torch::zeros({norb_i, norb_j}, opt_f64);
        auto out_z = has_spin_z ? torch::zeros({norb_i, norb_j}, opt_f64) : torch::empty({0}, opt_f64);
        return {out, out_z};
    }

    auto left = ga_values.index_select(0, ia);
    auto right = gb_values.index_select(0, ib);
    auto pot = ga_potential.index_select(0, ia).unsqueeze(1);

    auto out = torch::mm(left.t(), pot * right) * grid_weight;
    torch::Tensor out_z;
    if (has_spin_z) {
        auto pot_z = ga_spin_z_potential.index_select(0, ia).unsqueeze(1);
        out_z = torch::mm(left.t(), pot_z * right) * grid_weight;
    } else {
        out_z = torch::empty({0}, opt_f64);
    }

    return {out, out_z};
}

// Batched GPU pair contraction across all pairs in a structure
// Returns vector of tensors for unpolarized and spin-z
std::vector<std::vector<torch::Tensor>> contract_pairs_batch_cuda(
    const std::vector<torch::Tensor>& anchors_lo,
    const std::vector<torch::Tensor>& anchors_hi,
    const std::vector<torch::Tensor>& anchors_lookup,
    const std::vector<torch::Tensor>& anchors_values,
    const std::vector<torch::Tensor>& anchors_potential,
    bool has_spin_z,
    const std::vector<torch::Tensor>& anchors_spin_z_potential,
    torch::Tensor pair_i,      // [M] int32
    torch::Tensor pair_j,      // [M] int32
    torch::Tensor pair_R,      // [M, 3] int64
    torch::Tensor grid_shape,  // [3] int64
    double grid_weight,
    const std::vector<int>& atom_norbs
) {
    int M = pair_i.size(0);
    auto device = pair_i.device();
    auto opt_f64 = torch::dtype(torch::kFloat64).device(device);

    auto pair_i_cpu = pair_i.to(torch::kCPU).contiguous();
    auto pair_j_cpu = pair_j.to(torch::kCPU).contiguous();
    auto pair_i_a = pair_i_cpu.accessor<int, 1>();
    auto pair_j_a = pair_j_cpu.accessor<int, 1>();

    std::vector<torch::Tensor> v_blocks;
    std::vector<torch::Tensor> vz_blocks;
    v_blocks.reserve(M);
    if (has_spin_z) vz_blocks.reserve(M);

    for (int p = 0; p < M; ++p) {
        int i = pair_i_a[p];
        int j = pair_j_a[p];
        auto R_p = pair_R[p];
        auto shift_p = R_p * grid_shape;

        int norb_i = atom_norbs[i];
        int norb_j = atom_norbs[j];

        auto res = contract_single_pair_cuda(
            anchors_lo[i],
            anchors_hi[i],
            anchors_lookup[i],
            anchors_values[i],
            anchors_potential[i],
            has_spin_z,
            has_spin_z ? anchors_spin_z_potential[i] : torch::empty({0}, opt_f64),
            anchors_lo[j],
            anchors_hi[j],
            anchors_lookup[j],
            anchors_values[j],
            shift_p,
            grid_weight,
            norb_i,
            norb_j
        );
        v_blocks.push_back(res[0]);
        if (has_spin_z) vz_blocks.push_back(res[1]);
    }

    return {v_blocks, vz_blocks};
}
