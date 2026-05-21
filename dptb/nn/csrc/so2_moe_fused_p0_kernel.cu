#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#ifdef DPTB_SO2_MOE_FUSED_P0_CUTLASS
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#endif

namespace {

constexpr int kThreads = 128;

__device__ __forceinline__ float load_wigner_value(
    const float* __restrict__ wigner,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    int64_t edge,
    int l,
    int row,
    int col,
    int64_t dense_stride,
    int64_t compact_stride,
    int wigner_mode) {
  if (wigner_mode == 1) {
    const int64_t off = offsets[l];
    return wigner[(edge * dense_stride + off + row) * dense_stride + off + col];
  }
  if (wigner_mode == 2) {
    const int dim = 2 * l + 1;
    const int64_t off = compact_offsets[l];
    return wigner[edge * compact_stride + off + row * dim + col];
  }
  return row == col ? 1.0f : 0.0f;
}

__device__ __forceinline__ float load_pair_value(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    int64_t edge,
    int64_t channel,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t compact_stride,
    int wigner_mode,
    int m,
    int pair,
    bool rotate_in) {
  const int64_t base = in_base[channel];
  const int l = static_cast<int>(in_l[channel]);
  const int row = l + (pair == 0 ? -m : m);
  if (!rotate_in) {
    return x[edge * in_dim + base + row];
  }

  const int dim = 2 * l + 1;
  float acc = 0.0f;
  for (int d = 0; d < dim; ++d) {
    const float x_val = x[edge * in_dim + base + d];
    const float d_val = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row, dense_stride, compact_stride, wigner_mode);
    acc += x_val * d_val;
  }
  return acc;
}

__device__ __forceinline__ void scatter_pair_grad_x(
    float* __restrict__ grad_x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    int64_t edge,
    int64_t channel,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t compact_stride,
    int wigner_mode,
    int m,
    float grad0,
    float grad1,
    bool rotate_in) {
  const int64_t base = in_base[channel];
  const int l = static_cast<int>(in_l[channel]);
  const int row0 = l - m;
  const int row1 = l + m;
  float* __restrict__ grad_edge = grad_x + edge * in_dim;
  if (!rotate_in) {
    atomicAdd(grad_edge + base + row0, grad0);
    atomicAdd(grad_edge + base + row1, grad1);
    return;
  }

  const int dim = 2 * l + 1;
  for (int d = 0; d < dim; ++d) {
    const float d0 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row0, dense_stride, compact_stride, wigner_mode);
    const float d1 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row1, dense_stride, compact_stride, wigner_mode);
    atomicAdd(grad_edge + base + d, grad0 * d0 + grad1 * d1);
  }
}

__global__ void fused_pair_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ graph_index,
    const float* __restrict__ mixed_weight,
    const float* __restrict__ radial,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ out_base,
    const int64_t* __restrict__ out_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ out,
    int64_t n_edges,
    int64_t in_dim,
    int64_t out_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t n_routes,
    int64_t cin,
    int64_t cout,
    int m,
    int wigner_mode,
    bool rotate_in,
    bool rotate_out,
    bool has_radial,
    bool radial_on_input) {
  const int64_t edge = static_cast<int64_t>(blockIdx.x);
  const int64_t out_channel = static_cast<int64_t>(blockIdx.y);
  const int tid = threadIdx.x;
  if (edge >= n_edges || out_channel >= cout) {
    return;
  }

  const int64_t route = graph_index[edge];
  if (route < 0 || route >= n_routes) {
    return;
  }

  const float* __restrict__ weight =
      mixed_weight + route * (2 * cout * cin);

  float rr0 = 0.0f;
  float ii0 = 0.0f;
  float rr1 = 0.0f;
  float ii1 = 0.0f;
  for (int64_t ci = tid; ci < cin; ci += blockDim.x) {
    float x0 = load_pair_value(
        x, wigner, in_base, in_l, offsets, compact_offsets,
        edge, ci, in_dim, dense_stride, wigner_stride, wigner_mode, m, 0, rotate_in);
    float x1 = load_pair_value(
        x, wigner, in_base, in_l, offsets, compact_offsets,
        edge, ci, in_dim, dense_stride, wigner_stride, wigner_mode, m, 1, rotate_in);

    if (has_radial && radial_on_input) {
      const float r = radial[edge * cin + ci];
      x0 *= r;
      x1 *= r;
    }

    const float wr = weight[out_channel * cin + ci];
    const float wi = weight[(cout + out_channel) * cin + ci];
    rr0 += x0 * wr;
    ii0 += x0 * wi;
    rr1 += x1 * wr;
    ii1 += x1 * wi;
  }

  extern __shared__ float smem[];
  float* s_rr0 = smem;
  float* s_ii0 = s_rr0 + blockDim.x;
  float* s_rr1 = s_ii0 + blockDim.x;
  float* s_ii1 = s_rr1 + blockDim.x;
  s_rr0[tid] = rr0;
  s_ii0[tid] = ii0;
  s_rr1[tid] = rr1;
  s_ii1[tid] = ii1;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_rr0[tid] += s_rr0[tid + stride];
      s_ii0[tid] += s_ii0[tid + stride];
      s_rr1[tid] += s_rr1[tid + stride];
      s_ii1[tid] += s_ii1[tid + stride];
    }
    __syncthreads();
  }

  if (tid != 0) {
    return;
  }

  float y0 = s_rr0[0] - s_ii1[0];
  float y1 = s_rr1[0] + s_ii0[0];

  if (has_radial && !radial_on_input) {
    const float r = radial[edge * cout + out_channel];
    y0 *= r;
    y1 *= r;
  }

  const int l = static_cast<int>(out_l[out_channel]);
  const int dim = 2 * l + 1;
  const int row0 = l - m;
  const int row1 = l + m;
  const int64_t base = out_base[out_channel];
  float* __restrict__ out_edge = out + edge * out_dim;

  if (!rotate_out) {
    out_edge[base + row0] += y0;
    out_edge[base + row1] += y1;
    return;
  }

  for (int d = 0; d < dim; ++d) {
    const float d0 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row0, dense_stride, wigner_stride, wigner_mode);
    const float d1 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row1, dense_stride, wigner_stride, wigner_mode);
    out_edge[base + d] += y0 * d0 + y1 * d1;
  }
}

__global__ void fused_pair_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ graph_index,
    const float* __restrict__ mixed_weight,
    const float* __restrict__ radial,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ out_base,
    const int64_t* __restrict__ out_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ grad_x,
    float* __restrict__ grad_mixed_weight,
    float* __restrict__ grad_radial,
    int64_t n_edges,
    int64_t in_dim,
    int64_t out_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t n_routes,
    int64_t cin,
    int64_t cout,
    int m,
    int wigner_mode,
    bool rotate_in,
    bool rotate_out,
    bool has_radial,
    bool radial_on_input) {
  const int64_t edge = static_cast<int64_t>(blockIdx.x);
  const int64_t out_channel = static_cast<int64_t>(blockIdx.y);
  const int tid = threadIdx.x;
  if (edge >= n_edges || out_channel >= cout) {
    return;
  }

  const int64_t route = graph_index[edge];
  if (route < 0 || route >= n_routes) {
    return;
  }

  const int l = static_cast<int>(out_l[out_channel]);
  const int dim = 2 * l + 1;
  const int row0 = l - m;
  const int row1 = l + m;
  const int64_t base = out_base[out_channel];
  const float* __restrict__ grad_out_edge = grad_out + edge * out_dim;

  float grad_y0 = 0.0f;
  float grad_y1 = 0.0f;
  if (!rotate_out) {
    grad_y0 = grad_out_edge[base + row0];
    grad_y1 = grad_out_edge[base + row1];
  } else {
    for (int d = 0; d < dim; ++d) {
      const float go = grad_out_edge[base + d];
      const float d0 = load_wigner_value(
          wigner, offsets, compact_offsets,
          edge, l, d, row0, dense_stride, wigner_stride, wigner_mode);
      const float d1 = load_wigner_value(
          wigner, offsets, compact_offsets,
          edge, l, d, row1, dense_stride, wigner_stride, wigner_mode);
      grad_y0 += go * d0;
      grad_y1 += go * d1;
    }
  }

  const float* __restrict__ weight =
      mixed_weight + route * (2 * cout * cin);
  float rr0 = 0.0f;
  float ii0 = 0.0f;
  float rr1 = 0.0f;
  float ii1 = 0.0f;
  for (int64_t ci = tid; ci < cin; ci += blockDim.x) {
    float x0 = load_pair_value(
        x, wigner, in_base, in_l, offsets, compact_offsets,
        edge, ci, in_dim, dense_stride, wigner_stride, wigner_mode, m, 0, rotate_in);
    float x1 = load_pair_value(
        x, wigner, in_base, in_l, offsets, compact_offsets,
        edge, ci, in_dim, dense_stride, wigner_stride, wigner_mode, m, 1, rotate_in);
    if (has_radial && radial_on_input) {
      const float r = radial[edge * cin + ci];
      x0 *= r;
      x1 *= r;
    }
    const float wr = weight[out_channel * cin + ci];
    const float wi = weight[(cout + out_channel) * cin + ci];
    rr0 += x0 * wr;
    ii0 += x0 * wi;
    rr1 += x1 * wr;
    ii1 += x1 * wi;
  }

  extern __shared__ float smem[];
  float* s_rr0 = smem;
  float* s_ii0 = s_rr0 + blockDim.x;
  float* s_rr1 = s_ii0 + blockDim.x;
  float* s_ii1 = s_rr1 + blockDim.x;
  s_rr0[tid] = rr0;
  s_ii0[tid] = ii0;
  s_rr1[tid] = rr1;
  s_ii1[tid] = ii1;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_rr0[tid] += s_rr0[tid + stride];
      s_ii0[tid] += s_ii0[tid + stride];
      s_rr1[tid] += s_rr1[tid + stride];
      s_ii1[tid] += s_ii1[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    if (has_radial && !radial_on_input) {
      const float y0_pre = s_rr0[0] - s_ii1[0];
      const float y1_pre = s_rr1[0] + s_ii0[0];
      const float r = radial[edge * cout + out_channel];
      atomicAdd(grad_radial + edge * cout + out_channel, grad_y0 * y0_pre + grad_y1 * y1_pre);
      grad_y0 *= r;
      grad_y1 *= r;
    }
    s_rr0[0] = grad_y0;
    s_ii0[0] = grad_y1;
  }
  __syncthreads();
  grad_y0 = s_rr0[0];
  grad_y1 = s_ii0[0];

  const float grad_rr0 = grad_y0;
  const float grad_ii0 = grad_y1;
  const float grad_rr1 = grad_y1;
  const float grad_ii1 = -grad_y0;

  float* __restrict__ grad_weight =
      grad_mixed_weight + route * (2 * cout * cin);
  for (int64_t ci = tid; ci < cin; ci += blockDim.x) {
    const float x0_no_radial = load_pair_value(
        x, wigner, in_base, in_l, offsets, compact_offsets,
        edge, ci, in_dim, dense_stride, wigner_stride, wigner_mode, m, 0, rotate_in);
    const float x1_no_radial = load_pair_value(
        x, wigner, in_base, in_l, offsets, compact_offsets,
        edge, ci, in_dim, dense_stride, wigner_stride, wigner_mode, m, 1, rotate_in);
    float x0_eff = x0_no_radial;
    float x1_eff = x1_no_radial;
    if (has_radial && radial_on_input) {
      const float r = radial[edge * cin + ci];
      x0_eff *= r;
      x1_eff *= r;
    }

    const float wr = weight[out_channel * cin + ci];
    const float wi = weight[(cout + out_channel) * cin + ci];

    atomicAdd(grad_weight + out_channel * cin + ci, x0_eff * grad_rr0 + x1_eff * grad_rr1);
    atomicAdd(grad_weight + (cout + out_channel) * cin + ci, x0_eff * grad_ii0 + x1_eff * grad_ii1);

    float grad_x0_eff = wr * grad_rr0 + wi * grad_ii0;
    float grad_x1_eff = wr * grad_rr1 + wi * grad_ii1;
    if (has_radial && radial_on_input) {
      const float r = radial[edge * cin + ci];
      atomicAdd(grad_radial + edge * cin + ci, x0_no_radial * grad_x0_eff + x1_no_radial * grad_x1_eff);
      grad_x0_eff *= r;
      grad_x1_eff *= r;
    }

    scatter_pair_grad_x(
        grad_x, wigner, in_base, in_l, offsets, compact_offsets,
        edge, ci, in_dim, dense_stride, wigner_stride, wigner_mode,
        m, grad_x0_eff, grad_x1_eff, rotate_in);
  }
}

}  // namespace

#ifdef DPTB_SO2_MOE_FUSED_P0_CUTLASS
namespace {

__global__ void cutlass_cute_probe_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int64_t n,
    int64_t k) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int tid = threadIdx.x;
  if (row >= n) {
    return;
  }

  using namespace cute;
  Tensor tensor_a = make_tensor(make_gmem_ptr(a + row * k), make_shape(k));
  Tensor tensor_b = make_tensor(make_gmem_ptr(b + row * k), make_shape(k));

  float acc = 0.0f;
  for (int64_t col = tid; col < k; col += blockDim.x) {
    acc += tensor_a(col) * tensor_b(col);
  }

  extern __shared__ float smem[];
  smem[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[row] = smem[0];
  }
}

}  // namespace

torch::Tensor cutlass_cute_probe_cuda(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "a and b must be CUDA");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "a and b must be contiguous");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "a and b must be fp32");
  TORCH_CHECK(a.dim() == 2 && b.sizes() == a.sizes(), "a and b must be same-shape [N,K]");
  auto out = torch::empty({a.size(0)}, a.options());
  if (a.size(0) == 0) {
    return out;
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cutlass_cute_probe_kernel<<<a.size(0), kThreads, kThreads * sizeof(float), stream>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      out.data_ptr<float>(),
      a.size(0),
      a.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
#endif

torch::Tensor fused_pair_forward_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor graph_index,
    torch::Tensor mixed_weight,
    torch::Tensor radial,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = x.size(0);
  const int64_t in_dim = x.size(1);
  const int64_t n_routes = mixed_weight.size(0);
  const int64_t cout = out_base.numel();
  const int64_t cin = in_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto out = torch::zeros({n_edges, out_dim}, x.options());
  if (n_edges == 0 || cin == 0 || cout == 0) {
    return out;
  }

  const bool has_radial = radial.numel() != 0;
  dim3 grid(n_edges, cout);
  const size_t smem_bytes = 4 * kThreads * sizeof(float);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_pair_forward_kernel<<<grid, kThreads, smem_bytes, stream>>>(
      x.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      graph_index.data_ptr<int64_t>(),
      mixed_weight.data_ptr<float>(),
      radial.numel() == 0 ? nullptr : radial.data_ptr<float>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      n_edges,
      in_dim,
      out_dim,
      dense_stride,
      wigner_stride,
      n_routes,
      cin,
      cout,
      static_cast<int>(m),
      static_cast<int>(wigner_mode),
      rotate_in,
      rotate_out,
      has_radial,
      radial_on_input);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> fused_pair_backward_fp32_cuda(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor graph_index,
    torch::Tensor mixed_weight,
    torch::Tensor radial,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = x.size(0);
  const int64_t in_dim = x.size(1);
  const int64_t n_routes = mixed_weight.size(0);
  const int64_t cout = out_base.numel();
  const int64_t cin = in_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto grad_x = torch::zeros_like(x);
  auto grad_mixed_weight = torch::zeros_like(mixed_weight);
  auto grad_radial = radial.numel() == 0 ? radial.new_empty({0}) : torch::zeros_like(radial);
  if (n_edges == 0 || cin == 0 || cout == 0) {
    return {grad_x, grad_mixed_weight, grad_radial};
  }

  const bool has_radial = radial.numel() != 0;
  dim3 grid(n_edges, cout);
  const size_t smem_bytes = 4 * kThreads * sizeof(float);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_pair_backward_kernel<<<grid, kThreads, smem_bytes, stream>>>(
      grad_out.data_ptr<float>(),
      x.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      graph_index.data_ptr<int64_t>(),
      mixed_weight.data_ptr<float>(),
      radial.numel() == 0 ? nullptr : radial.data_ptr<float>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      grad_x.data_ptr<float>(),
      grad_mixed_weight.data_ptr<float>(),
      grad_radial.numel() == 0 ? nullptr : grad_radial.data_ptr<float>(),
      n_edges,
      in_dim,
      out_dim,
      dense_stride,
      wigner_stride,
      n_routes,
      cin,
      cout,
      static_cast<int>(m),
      static_cast<int>(wigner_mode),
      rotate_in,
      rotate_out,
      has_radial,
      radial_on_input);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_mixed_weight, grad_radial};
}
