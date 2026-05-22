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

__device__ __forceinline__ float load_m0_value(
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
    bool rotate_in) {
  const int64_t base = in_base[channel];
  const int l = static_cast<int>(in_l[channel]);
  if (!rotate_in || l == 0) {
    return x[edge * in_dim + base + l];
  }

  const int dim = 2 * l + 1;
  float acc = 0.0f;
  for (int d = 0; d < dim; ++d) {
    const float x_val = x[edge * in_dim + base + d];
    const float d_val = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, l, dense_stride, compact_stride, wigner_mode);
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

__global__ void pack_pair_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ pair,
    int64_t n_edges,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cin,
    int m,
    int wigner_mode,
    bool rotate_in) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cin;
  if (linear >= total) {
    return;
  }
  const int64_t edge = linear / cin;
  const int64_t channel = linear - edge * cin;
  pair[(edge * 2) * cin + channel] = load_pair_value(
      x, wigner, in_base, in_l, offsets, compact_offsets,
      edge, channel, in_dim, dense_stride, wigner_stride, wigner_mode, m, 0, rotate_in);
  pair[(edge * 2 + 1) * cin + channel] = load_pair_value(
      x, wigner, in_base, in_l, offsets, compact_offsets,
      edge, channel, in_dim, dense_stride, wigner_stride, wigner_mode, m, 1, rotate_in);
}

__global__ void pack_pairs_multi_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* const* __restrict__ in_base_ptrs,
    const int64_t* const* __restrict__ in_l_ptrs,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    const int64_t* __restrict__ cin_prefix,
    const int64_t* __restrict__ m_values,
    float* __restrict__ pair_flat,
    int64_t n_edges,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t n_m,
    int64_t total_cin,
    int wigner_mode,
    bool rotate_in) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * total_cin;
  if (idx >= total) {
    return;
  }
  const int64_t edge = idx / total_cin;
  const int64_t local = idx - edge * total_cin;

  int64_t m_idx = 0;
  while (m_idx + 1 < n_m && local >= cin_prefix[m_idx + 1]) {
    ++m_idx;
  }
  const int64_t cin = cin_prefix[m_idx + 1] - cin_prefix[m_idx];
  const int64_t channel = local - cin_prefix[m_idx];
  const int m = static_cast<int>(m_values[m_idx]);
  const int64_t* __restrict__ in_base = in_base_ptrs[m_idx];
  const int64_t* __restrict__ in_l = in_l_ptrs[m_idx];

  pair_flat[(edge * 2) * total_cin + local] = load_pair_value(
      x, wigner, in_base, in_l, offsets, compact_offsets,
      edge, channel, in_dim, dense_stride, wigner_stride, wigner_mode, m, 0, rotate_in);
  pair_flat[(edge * 2 + 1) * total_cin + local] = load_pair_value(
      x, wigner, in_base, in_l, offsets, compact_offsets,
      edge, channel, in_dim, dense_stride, wigner_stride, wigner_mode, m, 1, rotate_in);
}

__global__ void output_pair_grad_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ out_base,
    const int64_t* __restrict__ out_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ grad_pair,
    int64_t n_edges,
    int64_t out_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cout,
    int m,
    int wigner_mode,
    bool rotate_out) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cout;
  if (linear >= total) {
    return;
  }
  const int64_t edge = linear / cout;
  const int64_t channel = linear - edge * cout;
  const int l = static_cast<int>(out_l[channel]);
  const int row0 = l - m;
  const int row1 = l + m;
  const int64_t base = out_base[channel];
  const float* __restrict__ grad_edge = grad_out + edge * out_dim;

  float grad0 = 0.0f;
  float grad1 = 0.0f;
  if (!rotate_out) {
    grad0 = grad_edge[base + row0];
    grad1 = grad_edge[base + row1];
  } else {
    const int dim = 2 * l + 1;
    for (int d = 0; d < dim; ++d) {
      const float go = grad_edge[base + d];
      const float d0 = load_wigner_value(
          wigner, offsets, compact_offsets,
          edge, l, d, row0, dense_stride, wigner_stride, wigner_mode);
      const float d1 = load_wigner_value(
          wigner, offsets, compact_offsets,
          edge, l, d, row1, dense_stride, wigner_stride, wigner_mode);
      grad0 += go * d0;
      grad1 += go * d1;
    }
  }
  grad_pair[(edge * 2) * cout + channel] = grad0;
  grad_pair[(edge * 2 + 1) * cout + channel] = grad1;
}

__global__ void scatter_pair_forward_kernel(
    const float* __restrict__ pair_out,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ out_base,
    const int64_t* __restrict__ out_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ out,
    int64_t n_edges,
    int64_t out_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cout,
    int m,
    int wigner_mode,
    bool rotate_out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cout;
  if (idx >= total) {
    return;
  }
  const int64_t edge = idx / cout;
  const int64_t channel = idx - edge * cout;
  const int l = static_cast<int>(out_l[channel]);
  const int row0 = l - m;
  const int row1 = l + m;
  const int64_t base = out_base[channel];
  const float y0 = pair_out[(edge * 2) * cout + channel];
  const float y1 = pair_out[(edge * 2 + 1) * cout + channel];
  float* __restrict__ out_edge = out + edge * out_dim;

  if (!rotate_out) {
    atomicAdd(out_edge + base + row0, y0);
    atomicAdd(out_edge + base + row1, y1);
    return;
  }

  const int dim = 2 * l + 1;
  for (int d = 0; d < dim; ++d) {
    const float d0 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row0, dense_stride, wigner_stride, wigner_mode);
    const float d1 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row1, dense_stride, wigner_stride, wigner_mode);
    atomicAdd(out_edge + base + d, y0 * d0 + y1 * d1);
  }
}

__global__ void scatter_raw_pair_forward_kernel(
    const float* __restrict__ raw,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ out_base,
    const int64_t* __restrict__ out_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ out,
    int64_t n_edges,
    int64_t out_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cout,
    int m,
    int wigner_mode,
    bool rotate_out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cout;
  if (idx >= total) {
    return;
  }
  const int64_t edge = idx / cout;
  const int64_t channel = idx - edge * cout;
  const int64_t out2 = 2 * cout;
  const float rr0 = raw[(edge * 2) * out2 + channel];
  const float ii0 = raw[(edge * 2) * out2 + cout + channel];
  const float rr1 = raw[(edge * 2 + 1) * out2 + channel];
  const float ii1 = raw[(edge * 2 + 1) * out2 + cout + channel];
  const float y0 = rr0 - ii1;
  const float y1 = rr1 + ii0;

  const int l = static_cast<int>(out_l[channel]);
  const int row0 = l - m;
  const int row1 = l + m;
  const int64_t base = out_base[channel];
  float* __restrict__ out_edge = out + edge * out_dim;

  if (!rotate_out) {
    out_edge[base + row0] += y0;
    out_edge[base + row1] += y1;
    return;
  }

  const int dim = 2 * l + 1;
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

__global__ void scatter_raw_pairs_multi_forward_kernel(
    const float* const* __restrict__ raw_ptrs,
    const int64_t* const* __restrict__ out_base_ptrs,
    const int64_t* const* __restrict__ out_l_ptrs,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    const int64_t* __restrict__ cout_prefix,
    const int64_t* __restrict__ m_values,
    float* __restrict__ out,
    int64_t n_edges,
    int64_t out_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t n_m,
    int64_t total_cout,
    int wigner_mode,
    bool rotate_out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * total_cout;
  if (idx >= total) {
    return;
  }
  const int64_t edge = idx / total_cout;
  const int64_t local = idx - edge * total_cout;

  int64_t m_idx = 0;
  while (m_idx + 1 < n_m && local >= cout_prefix[m_idx + 1]) {
    ++m_idx;
  }
  const int64_t cout = cout_prefix[m_idx + 1] - cout_prefix[m_idx];
  const int64_t channel = local - cout_prefix[m_idx];
  const int m = static_cast<int>(m_values[m_idx]);
  const float* __restrict__ raw = raw_ptrs[m_idx];
  const int64_t* __restrict__ out_base = out_base_ptrs[m_idx];
  const int64_t* __restrict__ out_l = out_l_ptrs[m_idx];

  const int64_t out2 = 2 * cout;
  const float rr0 = raw[(edge * 2) * out2 + channel];
  const float ii0 = raw[(edge * 2) * out2 + cout + channel];
  const float rr1 = raw[(edge * 2 + 1) * out2 + channel];
  const float ii1 = raw[(edge * 2 + 1) * out2 + cout + channel];
  const float y0 = rr0 - ii1;
  const float y1 = rr1 + ii0;

  const int l = static_cast<int>(out_l[channel]);
  const int row0 = l - m;
  const int row1 = l + m;
  const int64_t base = out_base[channel];
  float* __restrict__ out_edge = out + edge * out_dim;

  if (!rotate_out) {
    out_edge[base + row0] += y0;
    out_edge[base + row1] += y1;
    return;
  }

  const int dim = 2 * l + 1;
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

__global__ void scatter_raw_pairs_multi_output_major_forward_kernel(
    const float* const* __restrict__ raw_ptrs,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    const int64_t* __restrict__ cout_prefix,
    const int64_t* __restrict__ m_values,
    const int64_t* __restrict__ entry_offsets,
    const int64_t* __restrict__ entry_m,
    const int64_t* __restrict__ entry_channel,
    const int64_t* __restrict__ entry_d,
    const int64_t* __restrict__ entry_l,
    float* __restrict__ out,
    int64_t n_edges,
    int64_t out_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int wigner_mode,
    bool rotate_out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * out_dim;
  if (idx >= total) {
    return;
  }
  const int64_t edge = idx / out_dim;
  const int64_t feature = idx - edge * out_dim;

  float acc = 0.0f;
  const int64_t begin = entry_offsets[feature];
  const int64_t end = entry_offsets[feature + 1];
  for (int64_t ei = begin; ei < end; ++ei) {
    const int64_t m_idx = entry_m[ei];
    const int64_t channel = entry_channel[ei];
    const int d = static_cast<int>(entry_d[ei]);
    const int l = static_cast<int>(entry_l[ei]);
    const int m = static_cast<int>(m_values[m_idx]);
    const int64_t cout = cout_prefix[m_idx + 1] - cout_prefix[m_idx];
    const int64_t out2 = 2 * cout;
    const float* __restrict__ raw = raw_ptrs[m_idx];

    const float rr0 = raw[(edge * 2) * out2 + channel];
    const float ii0 = raw[(edge * 2) * out2 + cout + channel];
    const float rr1 = raw[(edge * 2 + 1) * out2 + channel];
    const float ii1 = raw[(edge * 2 + 1) * out2 + cout + channel];
    const float y0 = rr0 - ii1;
    const float y1 = rr1 + ii0;
    const int row0 = l - m;
    const int row1 = l + m;

    if (!rotate_out) {
      if (d == row0) {
        acc += y0;
      }
      if (d == row1) {
        acc += y1;
      }
      continue;
    }

    const float d0 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row0, dense_stride, wigner_stride, wigner_mode);
    const float d1 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row1, dense_stride, wigner_stride, wigner_mode);
    acc += y0 * d0 + y1 * d1;
  }
  out[idx] = acc;
}

__global__ void raw_pair_output_grad_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ out_base,
    const int64_t* __restrict__ out_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ grad_raw,
    int64_t n_edges,
    int64_t out_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cout,
    int m,
    int wigner_mode,
    bool rotate_out) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cout;
  if (idx >= total) {
    return;
  }
  const int64_t edge = idx / cout;
  const int64_t channel = idx - edge * cout;
  const int l = static_cast<int>(out_l[channel]);
  const int row0 = l - m;
  const int row1 = l + m;
  const int64_t base = out_base[channel];
  const float* __restrict__ grad_out_edge = grad_out + edge * out_dim;

  float grad0 = 0.0f;
  float grad1 = 0.0f;
  if (!rotate_out) {
    grad0 = grad_out_edge[base + row0];
    grad1 = grad_out_edge[base + row1];
  } else {
    const int dim = 2 * l + 1;
    for (int d = 0; d < dim; ++d) {
      const float go = grad_out_edge[base + d];
      const float d0 = load_wigner_value(
          wigner, offsets, compact_offsets,
          edge, l, d, row0, dense_stride, wigner_stride, wigner_mode);
      const float d1 = load_wigner_value(
          wigner, offsets, compact_offsets,
          edge, l, d, row1, dense_stride, wigner_stride, wigner_mode);
      grad0 += go * d0;
      grad1 += go * d1;
    }
  }

  const int64_t out2 = 2 * cout;
  grad_raw[(edge * 2) * out2 + channel] = grad0;
  grad_raw[(edge * 2) * out2 + cout + channel] = grad1;
  grad_raw[(edge * 2 + 1) * out2 + channel] = grad1;
  grad_raw[(edge * 2 + 1) * out2 + cout + channel] = -grad0;
}

__global__ void scatter_pair_grad_kernel(
    const float* __restrict__ grad_pair,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ grad_x,
    int64_t n_edges,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cin,
    int m,
    int wigner_mode,
    bool rotate_in) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cin;
  if (linear >= total) {
    return;
  }
  const int64_t edge = linear / cin;
  const int64_t channel = linear - edge * cin;
  const float grad0 = grad_pair[(edge * 2) * cin + channel];
  const float grad1 = grad_pair[(edge * 2 + 1) * cin + channel];
  const int64_t base = in_base[channel];
  const int l = static_cast<int>(in_l[channel]);
  const int row0 = l - m;
  const int row1 = l + m;
  float* __restrict__ grad_edge = grad_x + edge * in_dim;

  if (!rotate_in) {
    grad_edge[base + row0] = grad0;
    grad_edge[base + row1] = grad1;
    return;
  }

  const int dim = 2 * l + 1;
  for (int d = 0; d < dim; ++d) {
    const float d0 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row0, dense_stride, wigner_stride, wigner_mode);
    const float d1 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row1, dense_stride, wigner_stride, wigner_mode);
    grad_edge[base + d] = grad0 * d0 + grad1 * d1;
  }
}

__global__ void scatter_pairs_multi_grad_kernel(
    const float* __restrict__ grad_packed,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ in_base_all,
    const int64_t* __restrict__ in_l_all,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    const int64_t* __restrict__ cin_prefix,
    const int64_t* __restrict__ m_values,
    float* __restrict__ grad_x,
    int64_t n_edges,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t n_m,
    int64_t total_cin,
    int wigner_mode,
    bool rotate_in) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * total_cin;
  if (idx >= total) {
    return;
  }
  const int64_t edge = idx / total_cin;
  const int64_t local = idx - edge * total_cin;

  int64_t m_idx = 0;
  while (m_idx + 1 < n_m && local >= cin_prefix[m_idx + 1]) {
    ++m_idx;
  }
  const int64_t channel = local - cin_prefix[m_idx];
  const int m = static_cast<int>(m_values[m_idx]);
  const float grad0 = grad_packed[(edge * 2) * total_cin + local];
  const float grad1 = grad_packed[(edge * 2 + 1) * total_cin + local];

  scatter_pair_grad_x(
      grad_x,
      wigner,
      in_base_all + cin_prefix[m_idx],
      in_l_all + cin_prefix[m_idx],
      offsets,
      compact_offsets,
      edge,
      channel,
      in_dim,
      dense_stride,
      wigner_stride,
      wigner_mode,
      m,
      grad0,
      grad1,
      rotate_in);
}

__global__ void scatter_pair_grad_radial_input_kernel(
    const float* __restrict__ grad_pair_eff,
    const float* __restrict__ pair_no_radial,
    const float* __restrict__ radial,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ grad_x,
    float* __restrict__ grad_radial,
    int64_t n_edges,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cin,
    int m,
    int wigner_mode,
    bool rotate_in) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cin;
  if (linear >= total) {
    return;
  }
  const int64_t edge = linear / cin;
  const int64_t channel = linear - edge * cin;
  const int64_t pair0 = (edge * 2) * cin + channel;
  const int64_t pair1 = (edge * 2 + 1) * cin + channel;
  const float grad0_eff = grad_pair_eff[pair0];
  const float grad1_eff = grad_pair_eff[pair1];
  const float x0_no_radial = pair_no_radial[pair0];
  const float x1_no_radial = pair_no_radial[pair1];
  const float r = radial[edge * cin + channel];
  grad_radial[edge * cin + channel] = grad0_eff * x0_no_radial + grad1_eff * x1_no_radial;

  const float grad0 = grad0_eff * r;
  const float grad1 = grad1_eff * r;
  const int64_t base = in_base[channel];
  const int l = static_cast<int>(in_l[channel]);
  const int row0 = l - m;
  const int row1 = l + m;
  float* __restrict__ grad_edge = grad_x + edge * in_dim;

  if (!rotate_in) {
    grad_edge[base + row0] = grad0;
    grad_edge[base + row1] = grad1;
    return;
  }

  const int dim = 2 * l + 1;
  for (int d = 0; d < dim; ++d) {
    const float d0 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row0, dense_stride, wigner_stride, wigner_mode);
    const float d1 = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, row1, dense_stride, wigner_stride, wigner_mode);
    grad_edge[base + d] = grad0 * d0 + grad1 * d1;
  }
}

__global__ void fused_m0_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ graph_index,
    const float* __restrict__ mixed_weight,
    const float* __restrict__ mixed_bias,
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
    int wigner_mode,
    bool rotate_in,
    bool rotate_out,
    bool has_bias,
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
  const float* __restrict__ weight = mixed_weight + route * (cout * cin);
  float acc = 0.0f;
  for (int64_t ci = tid; ci < cin; ci += blockDim.x) {
    float x0 = load_m0_value(
        x, wigner, in_base, in_l, offsets, compact_offsets,
        edge, ci, in_dim, dense_stride, wigner_stride, wigner_mode, rotate_in);
    if (has_radial && radial_on_input) {
      x0 *= radial[edge * cin + ci];
    }
    acc += x0 * weight[out_channel * cin + ci];
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

  if (tid != 0) {
    return;
  }

  float y = smem[0];
  if (has_bias) {
    y += mixed_bias[route * cout + out_channel];
  }
  if (has_radial && !radial_on_input) {
    y *= radial[edge * cout + out_channel];
  }

  const int l = static_cast<int>(out_l[out_channel]);
  const int dim = 2 * l + 1;
  const int64_t base = out_base[out_channel];
  float* __restrict__ out_edge = out + edge * out_dim;
  if (!rotate_out || l == 0) {
    out_edge[base + l] += y;
    return;
  }
  for (int d = 0; d < dim; ++d) {
    const float dv = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, l, dense_stride, wigner_stride, wigner_mode);
    out_edge[base + d] += y * dv;
  }
}

__global__ void pack_m0_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ packed,
    int64_t n_edges,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cin,
    int wigner_mode,
    bool rotate_in) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cin;
  if (linear >= total) {
    return;
  }
  const int64_t edge = linear / cin;
  const int64_t channel = linear - edge * cin;
  packed[edge * cin + channel] = load_m0_value(
      x, wigner, in_base, in_l, offsets, compact_offsets,
      edge, channel, in_dim, dense_stride, wigner_stride, wigner_mode, rotate_in);
}

__global__ void output_m0_grad_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ out_base,
    const int64_t* __restrict__ out_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ grad_m0,
    int64_t n_edges,
    int64_t out_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cout,
    int wigner_mode,
    bool rotate_out) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cout;
  if (linear >= total) {
    return;
  }
  const int64_t edge = linear / cout;
  const int64_t channel = linear - edge * cout;
  const int l = static_cast<int>(out_l[channel]);
  const int64_t base = out_base[channel];
  const float* __restrict__ grad_edge = grad_out + edge * out_dim;
  float grad = 0.0f;
  if (!rotate_out || l == 0) {
    grad = grad_edge[base + l];
  } else {
    const int dim = 2 * l + 1;
    for (int d = 0; d < dim; ++d) {
      const float go = grad_edge[base + d];
      const float dv = load_wigner_value(
          wigner, offsets, compact_offsets,
          edge, l, d, l, dense_stride, wigner_stride, wigner_mode);
      grad += go * dv;
    }
  }
  grad_m0[edge * cout + channel] = grad;
}

__global__ void scatter_m0_grad_kernel(
    const float* __restrict__ grad_m0,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ grad_x,
    int64_t n_edges,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cin,
    int wigner_mode,
    bool rotate_in) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cin;
  if (linear >= total) {
    return;
  }
  const int64_t edge = linear / cin;
  const int64_t channel = linear - edge * cin;
  const float grad = grad_m0[edge * cin + channel];
  const int64_t base = in_base[channel];
  const int l = static_cast<int>(in_l[channel]);
  float* __restrict__ grad_edge = grad_x + edge * in_dim;
  if (!rotate_in || l == 0) {
    grad_edge[base + l] = grad;
    return;
  }
  const int dim = 2 * l + 1;
  for (int d = 0; d < dim; ++d) {
    const float dv = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, l, dense_stride, wigner_stride, wigner_mode);
    grad_edge[base + d] = grad * dv;
  }
}

__global__ void scatter_m0_grad_radial_input_kernel(
    const float* __restrict__ grad_eff,
    const float* __restrict__ m0_no_radial,
    const float* __restrict__ radial,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    float* __restrict__ grad_x,
    float* __restrict__ grad_radial,
    int64_t n_edges,
    int64_t in_dim,
    int64_t dense_stride,
    int64_t wigner_stride,
    int64_t cin,
    int wigner_mode,
    bool rotate_in) {
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = n_edges * cin;
  if (linear >= total) {
    return;
  }
  const int64_t edge = linear / cin;
  const int64_t channel = linear - edge * cin;
  const float grad_e = grad_eff[edge * cin + channel];
  const float x0 = m0_no_radial[edge * cin + channel];
  const float r = radial[edge * cin + channel];
  grad_radial[edge * cin + channel] = grad_e * x0;
  const float grad = grad_e * r;
  const int64_t base = in_base[channel];
  const int l = static_cast<int>(in_l[channel]);
  float* __restrict__ grad_edge = grad_x + edge * in_dim;
  if (!rotate_in || l == 0) {
    grad_edge[base + l] = grad;
    return;
  }
  const int dim = 2 * l + 1;
  for (int d = 0; d < dim; ++d) {
    const float dv = load_wigner_value(
        wigner, offsets, compact_offsets,
        edge, l, d, l, dense_stride, wigner_stride, wigner_mode);
    grad_edge[base + d] = grad * dv;
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

template <int TileOut>
__global__ void fused_pair_forward_tiled_kernel(
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
  const int64_t out_base_tile = static_cast<int64_t>(blockIdx.y) * TileOut;
  const int tid = threadIdx.x;
  if (edge >= n_edges) {
    return;
  }

  const int64_t route = graph_index[edge];
  if (route < 0 || route >= n_routes) {
    return;
  }

  const float* __restrict__ weight = mixed_weight + route * (2 * cout * cin);
  float rr0[TileOut];
  float ii0[TileOut];
  float rr1[TileOut];
  float ii1[TileOut];
#pragma unroll
  for (int t = 0; t < TileOut; ++t) {
    rr0[t] = 0.0f;
    ii0[t] = 0.0f;
    rr1[t] = 0.0f;
    ii1[t] = 0.0f;
  }

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

#pragma unroll
    for (int t = 0; t < TileOut; ++t) {
      const int64_t out_channel = out_base_tile + t;
      if (out_channel >= cout) {
        continue;
      }
      const float wr = weight[out_channel * cin + ci];
      const float wi = weight[(cout + out_channel) * cin + ci];
      rr0[t] += x0 * wr;
      ii0[t] += x0 * wi;
      rr1[t] += x1 * wr;
      ii1[t] += x1 * wi;
    }
  }

  extern __shared__ float smem[];
  auto acc_layout = cute::make_layout(
      cute::make_shape(cute::Int<4>{}, cute::Int<TileOut>{}, cute::Int<kThreads>{}));
#pragma unroll
  for (int t = 0; t < TileOut; ++t) {
    smem[acc_layout(0, t, tid)] = rr0[t];
    smem[acc_layout(1, t, tid)] = ii0[t];
    smem[acc_layout(2, t, tid)] = rr1[t];
    smem[acc_layout(3, t, tid)] = ii1[t];
  }
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
#pragma unroll
      for (int t = 0; t < TileOut; ++t) {
        smem[acc_layout(0, t, tid)] += smem[acc_layout(0, t, tid + stride)];
        smem[acc_layout(1, t, tid)] += smem[acc_layout(1, t, tid + stride)];
        smem[acc_layout(2, t, tid)] += smem[acc_layout(2, t, tid + stride)];
        smem[acc_layout(3, t, tid)] += smem[acc_layout(3, t, tid + stride)];
      }
    }
    __syncthreads();
  }

  if (tid >= TileOut) {
    return;
  }
  const int64_t out_channel = out_base_tile + tid;
  if (out_channel >= cout) {
    return;
  }

  float y0 = smem[acc_layout(0, tid, 0)] - smem[acc_layout(3, tid, 0)];
  float y1 = smem[acc_layout(2, tid, 0)] + smem[acc_layout(1, tid, 0)];
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

template <int TileOut>
torch::Tensor fused_pair_forward_tiled_fp32_cuda_impl(
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
  dim3 grid(n_edges, (cout + TileOut - 1) / TileOut);
  const size_t smem_bytes = 4 * TileOut * kThreads * sizeof(float);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_pair_forward_tiled_kernel<TileOut><<<grid, kThreads, smem_bytes, stream>>>(
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

torch::Tensor fused_pair_forward_tiled2_fp32_cuda(
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
  return fused_pair_forward_tiled_fp32_cuda_impl<2>(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}

torch::Tensor fused_pair_forward_tiled3_fp32_cuda(
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
  return fused_pair_forward_tiled_fp32_cuda_impl<3>(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}

torch::Tensor fused_pair_forward_tiled4_fp32_cuda(
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
  return fused_pair_forward_tiled_fp32_cuda_impl<4>(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}

torch::Tensor fused_pair_forward_tiled8_fp32_cuda(
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
  return fused_pair_forward_tiled_fp32_cuda_impl<8>(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
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

torch::Tensor fused_m0_forward_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor graph_index,
    torch::Tensor mixed_weight,
    torch::Tensor mixed_bias,
    torch::Tensor radial,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
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

  const bool has_bias = mixed_bias.numel() != 0;
  const bool has_radial = radial.numel() != 0;
  dim3 grid(n_edges, cout);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fused_m0_forward_kernel<<<grid, kThreads, kThreads * sizeof(float), stream>>>(
      x.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      graph_index.data_ptr<int64_t>(),
      mixed_weight.data_ptr<float>(),
      mixed_bias.numel() == 0 ? nullptr : mixed_bias.data_ptr<float>(),
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
      static_cast<int>(wigner_mode),
      rotate_in,
      rotate_out,
      has_bias,
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

torch::Tensor pack_m0_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = x.size(0);
  const int64_t in_dim = x.size(1);
  const int64_t cin = in_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto packed = torch::empty({n_edges, cin}, x.options());
  if (n_edges == 0 || cin == 0) {
    return packed;
  }
  const int threads = 256;
  const int64_t total = n_edges * cin;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  pack_m0_kernel<<<grid, threads, 0, stream>>>(
      x.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      packed.data_ptr<float>(),
      n_edges,
      in_dim,
      dense_stride,
      wigner_stride,
      cin,
      static_cast<int>(wigner_mode),
      rotate_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return packed;
}

torch::Tensor output_m0_grad_fp32_cuda(
    torch::Tensor grad_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = grad_out.size(0);
  const int64_t out_dim = grad_out.size(1);
  const int64_t cout = out_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto grad_m0 = torch::empty({n_edges, cout}, grad_out.options());
  if (n_edges == 0 || cout == 0) {
    return grad_m0;
  }
  const int threads = 256;
  const int64_t total = n_edges * cout;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  output_m0_grad_kernel<<<grid, threads, 0, stream>>>(
      grad_out.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      grad_m0.data_ptr<float>(),
      n_edges,
      out_dim,
      dense_stride,
      wigner_stride,
      cout,
      static_cast<int>(wigner_mode),
      rotate_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_m0;
}

torch::Tensor scatter_m0_grad_fp32_cuda(
    torch::Tensor grad_m0,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = grad_m0.size(0);
  const int64_t cin = in_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto grad_x = torch::zeros({n_edges, in_dim}, grad_m0.options());
  if (n_edges == 0 || cin == 0) {
    return grad_x;
  }
  const int threads = 256;
  const int64_t total = n_edges * cin;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  scatter_m0_grad_kernel<<<grid, threads, 0, stream>>>(
      grad_m0.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      grad_x.data_ptr<float>(),
      n_edges,
      in_dim,
      dense_stride,
      wigner_stride,
      cin,
      static_cast<int>(wigner_mode),
      rotate_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_x;
}

std::vector<torch::Tensor> scatter_m0_grad_radial_input_fp32_cuda(
    torch::Tensor grad_eff,
    torch::Tensor m0_no_radial,
    torch::Tensor radial,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = grad_eff.size(0);
  const int64_t cin = in_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto grad_x = torch::zeros({n_edges, in_dim}, grad_eff.options());
  auto grad_radial = torch::empty_like(radial);
  if (n_edges == 0 || cin == 0) {
    return {grad_x, grad_radial};
  }
  const int threads = 256;
  const int64_t total = n_edges * cin;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  scatter_m0_grad_radial_input_kernel<<<grid, threads, 0, stream>>>(
      grad_eff.data_ptr<float>(),
      m0_no_radial.data_ptr<float>(),
      radial.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      grad_x.data_ptr<float>(),
      grad_radial.data_ptr<float>(),
      n_edges,
      in_dim,
      dense_stride,
      wigner_stride,
      cin,
      static_cast<int>(wigner_mode),
      rotate_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_radial};
}

torch::Tensor pack_pair_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t m,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = x.size(0);
  const int64_t in_dim = x.size(1);
  const int64_t cin = in_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto pair = torch::empty({n_edges, 2, cin}, x.options());
  if (n_edges == 0 || cin == 0) {
    return pair;
  }

  const int threads = 256;
  const int64_t total = n_edges * cin;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  pack_pair_kernel<<<grid, threads, 0, stream>>>(
      x.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      pair.data_ptr<float>(),
      n_edges,
      in_dim,
      dense_stride,
      wigner_stride,
      cin,
      static_cast<int>(m),
      static_cast<int>(wigner_mode),
      rotate_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return pair;
}

torch::Tensor pack_pairs_multi_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    std::vector<torch::Tensor> in_bases,
    std::vector<torch::Tensor> in_ls,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cin_prefix,
    torch::Tensor m_values,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_m = static_cast<int64_t>(in_bases.size());
  TORCH_CHECK(n_m > 0, "in_bases must be non-empty");
  TORCH_CHECK(static_cast<int64_t>(in_ls.size()) == n_m, "in_ls length mismatch");
  const int64_t n_edges = x.size(0);
  const int64_t in_dim = x.size(1);
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  const int64_t total_cin = cin_prefix[cin_prefix.numel() - 1].item<int64_t>();
  auto pair_flat = torch::empty({n_edges, 2, total_cin}, x.options());
  if (n_edges == 0 || total_cin == 0) {
    return pair_flat;
  }

  std::vector<int64_t> in_base_ptr_host;
  std::vector<int64_t> in_l_ptr_host;
  in_base_ptr_host.reserve(n_m);
  in_l_ptr_host.reserve(n_m);
  for (int64_t i = 0; i < n_m; ++i) {
    TORCH_CHECK(in_bases[i].is_cuda() && in_bases[i].is_contiguous(), "in_base tensors must be contiguous CUDA");
    TORCH_CHECK(in_ls[i].is_cuda() && in_ls[i].is_contiguous(), "in_l tensors must be contiguous CUDA");
    TORCH_CHECK(in_bases[i].scalar_type() == torch::kInt64, "in_base tensors must be int64");
    TORCH_CHECK(in_ls[i].scalar_type() == torch::kInt64, "in_l tensors must be int64");
    TORCH_CHECK(in_bases[i].numel() == in_ls[i].numel(), "in_base/in_l shape mismatch");
    in_base_ptr_host.push_back(reinterpret_cast<int64_t>(in_bases[i].data_ptr<int64_t>()));
    in_l_ptr_host.push_back(reinterpret_cast<int64_t>(in_ls[i].data_ptr<int64_t>()));
  }

  auto ptr_options = x.options().dtype(torch::kInt64);
  auto in_base_ptrs = torch::empty({n_m}, ptr_options);
  auto in_l_ptrs = torch::empty({n_m}, ptr_options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(in_base_ptrs.data_ptr<int64_t>(), in_base_ptr_host.data(), n_m * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(in_l_ptrs.data_ptr<int64_t>(), in_l_ptr_host.data(), n_m * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

  const int threads = 256;
  const int64_t total = n_edges * total_cin;
  const dim3 grid((total + threads - 1) / threads);
  pack_pairs_multi_kernel<<<grid, threads, 0, stream>>>(
      x.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      reinterpret_cast<const int64_t* const*>(in_base_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const int64_t* const*>(in_l_ptrs.data_ptr<int64_t>()),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      cin_prefix.data_ptr<int64_t>(),
      m_values.data_ptr<int64_t>(),
      pair_flat.data_ptr<float>(),
      n_edges,
      in_dim,
      dense_stride,
      wigner_stride,
      n_m,
      total_cin,
      static_cast<int>(wigner_mode),
      rotate_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return pair_flat;
}

torch::Tensor output_pair_grad_fp32_cuda(
    torch::Tensor grad_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = grad_out.size(0);
  const int64_t out_dim = grad_out.size(1);
  const int64_t cout = out_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto grad_pair = torch::empty({n_edges, 2, cout}, grad_out.options());
  if (n_edges == 0 || cout == 0) {
    return grad_pair;
  }

  const int threads = 256;
  const int64_t total = n_edges * cout;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  output_pair_grad_kernel<<<grid, threads, 0, stream>>>(
      grad_out.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      grad_pair.data_ptr<float>(),
      n_edges,
      out_dim,
      dense_stride,
      wigner_stride,
      cout,
      static_cast<int>(m),
      static_cast<int>(wigner_mode),
      rotate_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_pair;
}

torch::Tensor scatter_pair_forward_fp32_cuda(
    torch::Tensor pair_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = pair_out.size(0);
  const int64_t cout = out_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto out = torch::zeros({n_edges, out_dim}, pair_out.options());
  if (n_edges == 0 || cout == 0) {
    return out;
  }

  const int threads = 256;
  const int64_t total = n_edges * cout;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  scatter_pair_forward_kernel<<<grid, threads, 0, stream>>>(
      pair_out.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      n_edges,
      out_dim,
      dense_stride,
      wigner_stride,
      cout,
      static_cast<int>(m),
      static_cast<int>(wigner_mode),
      rotate_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor scatter_raw_pair_forward_fp32_cuda(
    torch::Tensor raw,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = raw.size(0);
  const int64_t cout = out_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto out = torch::zeros({n_edges, out_dim}, raw.options());
  if (n_edges == 0 || cout == 0) {
    return out;
  }

  const int threads = 256;
  const int64_t total = n_edges * cout;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  scatter_raw_pair_forward_kernel<<<grid, threads, 0, stream>>>(
      raw.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      n_edges,
      out_dim,
      dense_stride,
      wigner_stride,
      cout,
      static_cast<int>(m),
      static_cast<int>(wigner_mode),
      rotate_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor scatter_raw_pairs_multi_forward_fp32_cuda(
    std::vector<torch::Tensor> raws,
    torch::Tensor wigner,
    std::vector<torch::Tensor> out_bases,
    std::vector<torch::Tensor> out_ls,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cout_prefix,
    torch::Tensor m_values,
    int64_t out_dim,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_m = static_cast<int64_t>(raws.size());
  TORCH_CHECK(n_m > 0, "raws must be non-empty");
  TORCH_CHECK(static_cast<int64_t>(out_bases.size()) == n_m, "out_bases length mismatch");
  TORCH_CHECK(static_cast<int64_t>(out_ls.size()) == n_m, "out_ls length mismatch");
  const int64_t n_edges = raws[0].size(0);
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  const int64_t total_cout = cout_prefix[cout_prefix.numel() - 1].item<int64_t>();
  auto out = torch::zeros({n_edges, out_dim}, raws[0].options());
  if (n_edges == 0 || total_cout == 0) {
    return out;
  }

  std::vector<int64_t> raw_ptr_host;
  std::vector<int64_t> out_base_ptr_host;
  std::vector<int64_t> out_l_ptr_host;
  raw_ptr_host.reserve(n_m);
  out_base_ptr_host.reserve(n_m);
  out_l_ptr_host.reserve(n_m);
  for (int64_t i = 0; i < n_m; ++i) {
    const int64_t cout = out_bases[i].numel();
    TORCH_CHECK(raws[i].is_cuda() && raws[i].is_contiguous(), "raw tensors must be contiguous CUDA");
    TORCH_CHECK(out_bases[i].is_cuda() && out_bases[i].is_contiguous(), "out_base tensors must be contiguous CUDA");
    TORCH_CHECK(out_ls[i].is_cuda() && out_ls[i].is_contiguous(), "out_l tensors must be contiguous CUDA");
    TORCH_CHECK(raws[i].scalar_type() == torch::kFloat32, "raw tensors must be fp32");
    TORCH_CHECK(out_bases[i].scalar_type() == torch::kInt64, "out_base tensors must be int64");
    TORCH_CHECK(out_ls[i].scalar_type() == torch::kInt64, "out_l tensors must be int64");
    TORCH_CHECK(raws[i].dim() == 3 && raws[i].size(0) == n_edges && raws[i].size(1) == 2 && raws[i].size(2) == 2 * cout,
                "raw tensor shape must be [N, 2, 2*Cout]");
    TORCH_CHECK(out_ls[i].numel() == cout, "out_l/out_base shape mismatch");
    raw_ptr_host.push_back(reinterpret_cast<int64_t>(raws[i].data_ptr<float>()));
    out_base_ptr_host.push_back(reinterpret_cast<int64_t>(out_bases[i].data_ptr<int64_t>()));
    out_l_ptr_host.push_back(reinterpret_cast<int64_t>(out_ls[i].data_ptr<int64_t>()));
  }

  auto ptr_options = raws[0].options().dtype(torch::kInt64);
  auto raw_ptrs = torch::empty({n_m}, ptr_options);
  auto out_base_ptrs = torch::empty({n_m}, ptr_options);
  auto out_l_ptrs = torch::empty({n_m}, ptr_options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(raw_ptrs.data_ptr<int64_t>(), raw_ptr_host.data(), n_m * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(out_base_ptrs.data_ptr<int64_t>(), out_base_ptr_host.data(), n_m * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(out_l_ptrs.data_ptr<int64_t>(), out_l_ptr_host.data(), n_m * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

  const int threads = 256;
  const int64_t total = n_edges * total_cout;
  const dim3 grid((total + threads - 1) / threads);
  scatter_raw_pairs_multi_forward_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const float* const*>(raw_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const int64_t* const*>(out_base_ptrs.data_ptr<int64_t>()),
      reinterpret_cast<const int64_t* const*>(out_l_ptrs.data_ptr<int64_t>()),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      cout_prefix.data_ptr<int64_t>(),
      m_values.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      n_edges,
      out_dim,
      dense_stride,
      wigner_stride,
      n_m,
      total_cout,
      static_cast<int>(wigner_mode),
      rotate_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor scatter_raw_pairs_multi_output_major_forward_fp32_cuda(
    std::vector<torch::Tensor> raws,
    torch::Tensor wigner,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cout_prefix,
    torch::Tensor m_values,
    torch::Tensor entry_offsets,
    torch::Tensor entry_m,
    torch::Tensor entry_channel,
    torch::Tensor entry_d,
    torch::Tensor entry_l,
    int64_t out_dim,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_m = static_cast<int64_t>(raws.size());
  TORCH_CHECK(n_m > 0, "raws must be non-empty");
  const int64_t n_edges = raws[0].size(0);
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto out = torch::empty({n_edges, out_dim}, raws[0].options());
  if (n_edges == 0 || out_dim == 0) {
    return out;
  }

  std::vector<int64_t> raw_ptr_host;
  raw_ptr_host.reserve(n_m);
  for (int64_t i = 0; i < n_m; ++i) {
    const int64_t cout = cout_prefix[i + 1].item<int64_t>() - cout_prefix[i].item<int64_t>();
    TORCH_CHECK(raws[i].is_cuda() && raws[i].is_contiguous(), "raw tensors must be contiguous CUDA");
    TORCH_CHECK(raws[i].scalar_type() == torch::kFloat32, "raw tensors must be fp32");
    TORCH_CHECK(raws[i].dim() == 3 && raws[i].size(0) == n_edges && raws[i].size(1) == 2 && raws[i].size(2) == 2 * cout,
                "raw tensor shape must be [N, 2, 2*Cout]");
    raw_ptr_host.push_back(reinterpret_cast<int64_t>(raws[i].data_ptr<float>()));
  }

  auto ptr_options = raws[0].options().dtype(torch::kInt64);
  auto raw_ptrs = torch::empty({n_m}, ptr_options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudaMemcpyAsync(raw_ptrs.data_ptr<int64_t>(), raw_ptr_host.data(), n_m * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

  const int threads = 256;
  const int64_t total = n_edges * out_dim;
  const dim3 grid((total + threads - 1) / threads);
  scatter_raw_pairs_multi_output_major_forward_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const float* const*>(raw_ptrs.data_ptr<int64_t>()),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      cout_prefix.data_ptr<int64_t>(),
      m_values.data_ptr<int64_t>(),
      entry_offsets.data_ptr<int64_t>(),
      entry_m.data_ptr<int64_t>(),
      entry_channel.data_ptr<int64_t>(),
      entry_d.data_ptr<int64_t>(),
      entry_l.data_ptr<int64_t>(),
      out.data_ptr<float>(),
      n_edges,
      out_dim,
      dense_stride,
      wigner_stride,
      static_cast<int>(wigner_mode),
      rotate_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor raw_pair_output_grad_fp32_cuda(
    torch::Tensor grad_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = grad_out.size(0);
  const int64_t out_dim = grad_out.size(1);
  const int64_t cout = out_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto grad_raw = torch::empty({n_edges, 2, 2 * cout}, grad_out.options());
  if (n_edges == 0 || cout == 0) {
    return grad_raw;
  }

  const int threads = 256;
  const int64_t total = n_edges * cout;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  raw_pair_output_grad_kernel<<<grid, threads, 0, stream>>>(
      grad_out.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      grad_raw.data_ptr<float>(),
      n_edges,
      out_dim,
      dense_stride,
      wigner_stride,
      cout,
      static_cast<int>(m),
      static_cast<int>(wigner_mode),
      rotate_out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_raw;
}

torch::Tensor scatter_pair_grad_fp32_cuda(
    torch::Tensor grad_pair,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    int64_t m,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = grad_pair.size(0);
  const int64_t cin = in_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto grad_x = torch::zeros({n_edges, in_dim}, grad_pair.options());
  if (n_edges == 0 || cin == 0) {
    return grad_x;
  }

  const int threads = 256;
  const int64_t total = n_edges * cin;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  scatter_pair_grad_kernel<<<grid, threads, 0, stream>>>(
      grad_pair.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      grad_x.data_ptr<float>(),
      n_edges,
      in_dim,
      dense_stride,
      wigner_stride,
      cin,
      static_cast<int>(m),
      static_cast<int>(wigner_mode),
      rotate_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_x;
}

torch::Tensor scatter_pairs_multi_grad_fp32_cuda(
    torch::Tensor grad_packed,
    torch::Tensor wigner,
    torch::Tensor in_base_all,
    torch::Tensor in_l_all,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cin_prefix,
    torch::Tensor m_values,
    int64_t in_dim,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_m = m_values.numel();
  TORCH_CHECK(n_m > 0, "m_values must be non-empty");
  TORCH_CHECK(cin_prefix.numel() == n_m + 1, "cin_prefix length must be n_m + 1");
  const int64_t n_edges = grad_packed.size(0);
  const int64_t total_cin = cin_prefix[cin_prefix.numel() - 1].item<int64_t>();
  TORCH_CHECK(in_base_all.numel() == total_cin && in_l_all.numel() == total_cin,
              "flat input maps must have total_cin entries");
  TORCH_CHECK(grad_packed.dim() == 3 && grad_packed.size(1) == 2 && grad_packed.size(2) == total_cin,
              "grad_packed must be [N, 2, total_cin]");
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto grad_x = torch::zeros({n_edges, in_dim}, grad_packed.options());
  if (n_edges == 0 || total_cin == 0) {
    return grad_x;
  }

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int threads = 256;
  const int64_t total = n_edges * total_cin;
  const dim3 grid((total + threads - 1) / threads);
  scatter_pairs_multi_grad_kernel<<<grid, threads, 0, stream>>>(
      grad_packed.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      in_base_all.data_ptr<int64_t>(),
      in_l_all.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      cin_prefix.data_ptr<int64_t>(),
      m_values.data_ptr<int64_t>(),
      grad_x.data_ptr<float>(),
      n_edges,
      in_dim,
      dense_stride,
      wigner_stride,
      n_m,
      total_cin,
      static_cast<int>(wigner_mode),
      rotate_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return grad_x;
}

std::vector<torch::Tensor> scatter_pair_grad_radial_input_fp32_cuda(
    torch::Tensor grad_pair_eff,
    torch::Tensor pair_no_radial,
    torch::Tensor radial,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    int64_t m,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  const int64_t n_edges = grad_pair_eff.size(0);
  const int64_t cin = in_base.numel();
  const int64_t dense_stride = wigner_mode == 1 ? wigner.size(1) : 0;
  auto grad_x = torch::zeros({n_edges, in_dim}, grad_pair_eff.options());
  auto grad_radial = torch::empty_like(radial);
  if (n_edges == 0 || cin == 0) {
    return {grad_x, grad_radial};
  }

  const int threads = 256;
  const int64_t total = n_edges * cin;
  const dim3 grid((total + threads - 1) / threads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  scatter_pair_grad_radial_input_kernel<<<grid, threads, 0, stream>>>(
      grad_pair_eff.data_ptr<float>(),
      pair_no_radial.data_ptr<float>(),
      radial.data_ptr<float>(),
      wigner.numel() == 0 ? nullptr : wigner.data_ptr<float>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() == 0 ? nullptr : compact_offsets.data_ptr<int64_t>(),
      grad_x.data_ptr<float>(),
      grad_radial.data_ptr<float>(),
      n_edges,
      in_dim,
      dense_stride,
      wigner_stride,
      cin,
      static_cast<int>(m),
      static_cast<int>(wigner_mode),
      rotate_in);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {grad_x, grad_radial};
}
