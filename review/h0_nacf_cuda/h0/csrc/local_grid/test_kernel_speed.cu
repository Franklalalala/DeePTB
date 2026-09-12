#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void test_pair_contraction_atomic_shared(
    int total_points,
    int norb_i,
    int norb_j,
    const double* __restrict__ values_i,
    const double* __restrict__ values_j,
    const double* __restrict__ potential_i,
    double grid_weight,
    double* __restrict__ out_mat
) {
    extern __shared__ double s_mat[];
    int mat_size = norb_i * norb_j;
    for (int k = threadIdx.x; k < mat_size; k += blockDim.x) {
        s_mat[k] = 0.0;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < total_points; idx += blockDim.x) {
        double pot = potential_i[idx] * grid_weight;
        const double* left = values_i + idx * norb_i;
        const double* right = values_j + idx * norb_j;
        for (int mu = 0; mu < norb_i; ++mu) {
            double l_val = left[mu] * pot;
            for (int nu = 0; nu < norb_j; ++nu) {
                atomicAdd(&s_mat[mu * norb_j + nu], l_val * right[nu]);
            }
        }
    }
    __syncthreads();

    for (int k = threadIdx.x; k < mat_size; k += blockDim.x) {
        out_mat[k] = s_mat[k];
    }
}

torch::Tensor run_benchmark(torch::Tensor v_i, torch::Tensor v_j, torch::Tensor pot, double weight) {
    int total_points = v_i.size(0);
    int norb_i = v_i.size(1);
    int norb_j = v_j.size(1);
    auto out = torch::zeros({norb_i, norb_j}, v_i.options());
    
    int threads = 256;
    size_t shared_bytes = norb_i * norb_j * sizeof(double);
    test_pair_contraction_atomic_shared<<<1, threads, shared_bytes>>>(
        total_points, norb_i, norb_j,
        v_i.data_ptr<double>(),
        v_j.data_ptr<double>(),
        pot.data_ptr<double>(),
        weight,
        out.data_ptr<double>()
    );
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_benchmark", &run_benchmark, "Benchmark pair contraction kernel");
}
