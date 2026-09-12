#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdint>
#include <vector>
#include "harmonics_cuda.cuh"
#include "spline_cuda.cuh"

__global__ void find_active_nodes_kernel(
    int total_candidates,
    int64_t nmin0, int64_t nmin1, int64_t nmin2,
    int count1, int count2,
    int64_t shape0, int64_t shape1, int64_t shape2,
    double c0, double c1, double c2,
    double cell0, double cell1, double cell2,
    double cell3, double cell4, double cell5,
    double cell6, double cell7, double cell8,
    double rcut2,
    int* __restrict__ mask
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total_candidates) return;

    int iz = idx % count2;
    int rem = idx / count2;
    int iy = rem % count1;
    int ix = rem / count1;

    int64_t gx = nmin0 + ix;
    int64_t gy = nmin1 + iy;
    int64_t gz = nmin2 + iz;

    double fx = (double)gx / (double)shape0;
    double fy = (double)gy / (double)shape1;
    double fz = (double)gz / (double)shape2;

    double rx = fx * cell0 + fy * cell3 + fz * cell6;
    double ry = fx * cell1 + fy * cell4 + fz * cell7;
    double rz = fx * cell2 + fy * cell5 + fz * cell8;

    double dx = rx - c0;
    double dy = ry - c1;
    double dz = rz - c2;
    double dist2 = dx * dx + dy * dy + dz * dz;

    mask[idx] = (dist2 <= rcut2) ? 1 : 0;
}

__global__ void evaluate_ao_nodes_kernel(
    int npoints,
    const int64_t* __restrict__ active_indices,
    int64_t nmin0, int64_t nmin1, int64_t nmin2,
    int count1, int count2,
    int64_t shape0, int64_t shape1, int64_t shape2,
    double c0, double c1, double c2,
    double cell0, double cell1, double cell2,
    double cell3, double cell4, double cell5,
    double cell6, double cell7, double cell8,
    double dr, double rcut, int n_intervals, int num_channels,
    const double* __restrict__ spline_coeffs,
    int norb,
    const int* __restrict__ descriptors,
    const double* __restrict__ field_values,
    bool has_spin_z,
    const double* __restrict__ spin_z_values,
    int64_t* __restrict__ integer_indices,
    double* __restrict__ values,
    double* __restrict__ potential,
    double* __restrict__ spin_z_potential
) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= npoints) return;

    int64_t idx = active_indices[k];
    int iz = idx % count2;
    int rem = idx / count2;
    int iy = rem % count1;
    int ix = rem / count1;

    int64_t gx = nmin0 + ix;
    int64_t gy = nmin1 + iy;
    int64_t gz = nmin2 + iz;

    integer_indices[k * 3 + 0] = gx;
    integer_indices[k * 3 + 1] = gy;
    integer_indices[k * 3 + 2] = gz;

    // Modulo potential index
    int64_t wx = (gx % shape0 + shape0) % shape0;
    int64_t wy = (gy % shape1 + shape1) % shape1;
    int64_t wz = (gz % shape2 + shape2) % shape2;
    int64_t pot_idx = wx * (shape1 * shape2) + wy * shape2 + wz;
    potential[k] = field_values[pot_idx];
    if (has_spin_z) {
        spin_z_potential[k] = spin_z_values[pot_idx];
    }

    double fx = (double)gx / (double)shape0;
    double fy = (double)gy / (double)shape1;
    double fz = (double)gz / (double)shape2;

    double rx = fx * cell0 + fy * cell3 + fz * cell6;
    double ry = fx * cell1 + fy * cell4 + fz * cell7;
    double rz = fx * cell2 + fy * cell5 + fz * cell8;

    double dx = rx - c0;
    double dy = ry - c1;
    double dz = rz - c2;
    double dist = sqrt(dx * dx + dy * dy + dz * dz);

    double r_vals[16];
    int max_ch = (num_channels < 16) ? num_channels : 16;
    for (int ch = 0; ch < max_ch; ++ch) {
        const double* c_ptr = spline_coeffs + ch * (4 * n_intervals);
        r_vals[ch] = eval_uniform_cubic_spline(dist, dr, rcut, n_intervals, c_ptr);
    }

    for (int mu = 0; mu < norb; ++mu) {
        int ch = descriptors[mu * 3 + 0];
        int l  = descriptors[mu * 3 + 1];
        int m  = descriptors[mu * 3 + 2];
        double ylm = compute_real_ylm_abacus(l, m, dx, dy, dz);
        values[k * norb + mu] = r_vals[ch] * ylm;
    }
}

__global__ void populate_lookup_kernel(
    int npoints,
    const int64_t* __restrict__ integer_indices,
    int64_t lo0, int64_t lo1, int64_t lo2,
    int64_t stride0, int64_t stride1,
    int* __restrict__ lookup
) {
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k >= npoints) return;

    int64_t rx = integer_indices[k * 3 + 0] - lo0;
    int64_t ry = integer_indices[k * 3 + 1] - lo1;
    int64_t rz = integer_indices[k * 3 + 2] - lo2;

    int64_t idx = rx * stride0 + ry * stride1 + rz;
    lookup[idx] = k;
}

// Host C++ wrapper to build an atom's support entirely on GPU
std::vector<torch::Tensor> build_atom_support_cuda(
    torch::Tensor center,         // [3] float64
    double rcut,
    torch::Tensor nmin,           // [3] int64
    torch::Tensor counts,         // [3] int32
    torch::Tensor shape,          // [3] int64
    torch::Tensor cell,           // [3, 3] float64
    double dr,
    int n_intervals,
    int num_channels,
    torch::Tensor spline_coeffs,  // [num_channels, 4, n_intervals] float64
    int norb,
    torch::Tensor descriptors,    // [norb, 3] int32
    torch::Tensor field_values,   // [N0, N1, N2] float64
    bool has_spin_z,
    torch::Tensor spin_z_values   // [N0, N1, N2] float64 (optional)
) {
    auto device = field_values.device();
    TORCH_CHECK(device.is_cuda(), "build_atom_support_cuda requires CUDA tensor");
    const c10::cuda::CUDAGuard device_guard(device);
    const auto stream = c10::cuda::getCurrentCUDAStream(device.index());

    auto nmin_cpu = nmin.to(torch::kCPU).contiguous();
    auto counts_cpu = counts.to(torch::kCPU).contiguous();
    auto shape_cpu = shape.to(torch::kCPU).contiguous();
    auto center_cpu = center.to(torch::kCPU).contiguous();
    auto cell_cpu = cell.to(torch::kCPU).contiguous();

    auto nmin_a = nmin_cpu.accessor<int64_t, 1>();
    auto counts_a = counts_cpu.accessor<int, 1>();
    auto shape_a = shape_cpu.accessor<int64_t, 1>();
    auto center_a = center_cpu.accessor<double, 1>();
    auto cell_a = cell_cpu.accessor<double, 2>();

    int64_t total_candidates = (int64_t)counts_a[0] * counts_a[1] * counts_a[2];
    auto mask = torch::empty({total_candidates}, torch::dtype(torch::kInt32).device(device));

    int threads = 256;
    int blocks = (int)((total_candidates + threads - 1) / threads);

    find_active_nodes_kernel<<<blocks, threads, 0, stream>>>(
        (int)total_candidates,
        nmin_a[0], nmin_a[1], nmin_a[2],
        counts_a[1], counts_a[2],
        shape_a[0], shape_a[1], shape_a[2],
        center_a[0], center_a[1], center_a[2],
        cell_a[0][0], cell_a[0][1], cell_a[0][2],
        cell_a[1][0], cell_a[1][1], cell_a[1][2],
        cell_a[2][0], cell_a[2][1], cell_a[2][2],
        rcut * rcut + 1e-12,
        mask.data_ptr<int>()
    );

    auto active_indices = torch::nonzero(mask).squeeze(1);
    int npoints = (int)active_indices.size(0);

    auto opt_f64 = torch::dtype(torch::kFloat64).device(device);
    auto opt_i64 = torch::dtype(torch::kInt64).device(device);
    auto opt_i32 = torch::dtype(torch::kInt32).device(device);

    auto integer_indices = torch::empty({npoints, 3}, opt_i64);
    auto values = torch::empty({npoints, norb}, opt_f64);
    auto potential = torch::empty({npoints}, opt_f64);
    auto spin_z_pot = has_spin_z ? torch::empty({npoints}, opt_f64) : torch::empty({0}, opt_f64);

    if (npoints > 0) {
        int eval_blocks = (npoints + threads - 1) / threads;
        evaluate_ao_nodes_kernel<<<eval_blocks, threads, 0, stream>>>(
            npoints,
            active_indices.data_ptr<int64_t>(),
            nmin_a[0], nmin_a[1], nmin_a[2],
            counts_a[1], counts_a[2],
            shape_a[0], shape_a[1], shape_a[2],
            center_a[0], center_a[1], center_a[2],
            cell_a[0][0], cell_a[0][1], cell_a[0][2],
            cell_a[1][0], cell_a[1][1], cell_a[1][2],
            cell_a[2][0], cell_a[2][1], cell_a[2][2],
            dr, rcut, n_intervals, num_channels,
            spline_coeffs.data_ptr<double>(),
            norb,
            descriptors.data_ptr<int>(),
            field_values.data_ptr<double>(),
            has_spin_z,
            has_spin_z ? spin_z_values.data_ptr<double>() : nullptr,
            integer_indices.data_ptr<int64_t>(),
            values.data_ptr<double>(),
            potential.data_ptr<double>(),
            has_spin_z ? spin_z_pot.data_ptr<double>() : nullptr
        );
    }

    // Compute lo, hi, and build 3D lookup tensor
    torch::Tensor lo, hi, lookup;
    if (npoints > 0) {
        lo = std::get<0>(integer_indices.min(0));
        hi = std::get<0>(integer_indices.max(0));
        auto lo_cpu = lo.to(torch::kCPU).contiguous();
        auto hi_cpu = hi.to(torch::kCPU).contiguous();
        auto lo_a = lo_cpu.accessor<int64_t, 1>();
        auto hi_a = hi_cpu.accessor<int64_t, 1>();

        int64_t box0 = hi_a[0] - lo_a[0] + 1;
        int64_t box1 = hi_a[1] - lo_a[1] + 1;
        int64_t box2 = hi_a[2] - lo_a[2] + 1;
        lookup = torch::full({box0, box1, box2}, -1, opt_i32);

        int64_t stride1 = box2;
        int64_t stride0 = box1 * stride1;

        int pop_blocks = (npoints + threads - 1) / threads;
        populate_lookup_kernel<<<pop_blocks, threads, 0, stream>>>(
            npoints,
            integer_indices.data_ptr<int64_t>(),
            lo_a[0], lo_a[1], lo_a[2],
            stride0, stride1,
            lookup.data_ptr<int>()
        );
    } else {
        lo = torch::zeros({3}, torch::dtype(torch::kInt64).device(device));
        hi = torch::zeros({3}, torch::dtype(torch::kInt64).device(device));
        lookup = torch::full({0, 0, 0}, -1, opt_i32);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {integer_indices, values, potential, spin_z_pot, lo, hi, lookup};
}
