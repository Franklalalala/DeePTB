#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace {

__device__ __forceinline__ int64_t ceil_div_i64(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ int64_t find_problem_for_tile(
    const int64_t* __restrict__ prefix,
    int64_t n_problems,
    int64_t tile) {
  int64_t lo = 0;
  int64_t hi = n_problems;
  while (lo + 1 < hi) {
    int64_t mid = (lo + hi) >> 1;
    if (prefix[mid] <= tile) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ __forceinline__ float load_wigner_value(
    const float* __restrict__ wigner,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    int64_t edge,
    int64_t l,
    int64_t row,
    int64_t col,
    int64_t dense_stride,
    int64_t compact_stride,
    int64_t wigner_mode) {
  if (wigner_mode == 0) {
    return row == col ? 1.0f : 0.0f;
  }
  if (wigner_mode == 1) {
    const int64_t base = edge * dense_stride * dense_stride + offsets[l] * dense_stride + offsets[l];
    return wigner[base + row * dense_stride + col];
  }
  const int64_t dim = 2 * l + 1;
  const int64_t base = edge * compact_stride + compact_offsets[l];
  return wigner[base + row * dim + col];
}

__device__ __forceinline__ float load_scalar_component(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    int64_t edge,
    int64_t in_dim,
    int64_t base,
    int64_t l,
    int64_t dense_stride,
    int64_t compact_stride,
    int64_t wigner_mode,
    bool rotate_in) {
  const int64_t center = l;
  if (!rotate_in || wigner_mode == 0) {
    return x[edge * in_dim + base + center];
  }
  const int64_t dim = 2 * l + 1;
  float acc = 0.0f;
  for (int64_t d = 0; d < dim; ++d) {
    acc += x[edge * in_dim + base + d] *
           load_wigner_value(wigner, offsets, compact_offsets, edge, l, d, center,
                             dense_stride, compact_stride, wigner_mode);
  }
  return acc;
}

__device__ __forceinline__ float load_pair_component(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    int64_t edge,
    int64_t in_dim,
    int64_t base,
    int64_t l,
    int64_t row0,
    int64_t dense_stride,
    int64_t compact_stride,
    int64_t wigner_mode,
    bool rotate_in) {
  if (!rotate_in || wigner_mode == 0) {
    return x[edge * in_dim + base + row0];
  }
  const int64_t dim = 2 * l + 1;
  float acc = 0.0f;
  for (int64_t d = 0; d < dim; ++d) {
    acc += x[edge * in_dim + base + d] *
           load_wigner_value(wigner, offsets, compact_offsets, edge, l, d, row0,
                             dense_stride, compact_stride, wigner_mode);
  }
  return acc;
}

__device__ __forceinline__ void store_scalar_component(
    float* __restrict__ out,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    int64_t edge,
    int64_t out_dim,
    int64_t base,
    int64_t l,
    float value,
    int64_t dense_stride,
    int64_t compact_stride,
    int64_t wigner_mode,
    bool rotate_out) {
  const int64_t center = l;
  if (!rotate_out || wigner_mode == 0) {
    atomicAdd(out + edge * out_dim + base + center, value);
    return;
  }
  const int64_t dim = 2 * l + 1;
  for (int64_t d = 0; d < dim; ++d) {
    const float coeff = load_wigner_value(wigner, offsets, compact_offsets, edge, l,
                                          d, center, dense_stride, compact_stride,
                                          wigner_mode);
    atomicAdd(out + edge * out_dim + base + d, value * coeff);
  }
}

__device__ __forceinline__ void store_pair_component(
    float* __restrict__ out,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    int64_t edge,
    int64_t out_dim,
    int64_t base,
    int64_t l,
    int64_t row0,
    int64_t row1,
    float v0,
    float v1,
    int64_t dense_stride,
    int64_t compact_stride,
    int64_t wigner_mode,
    bool rotate_out) {
  if (!rotate_out || wigner_mode == 0) {
    atomicAdd(out + edge * out_dim + base + row0, v0);
    atomicAdd(out + edge * out_dim + base + row1, v1);
    return;
  }
  const int64_t dim = 2 * l + 1;
  for (int64_t d = 0; d < dim; ++d) {
    const float c0 = load_wigner_value(wigner, offsets, compact_offsets, edge, l,
                                       d, row0, dense_stride, compact_stride,
                                       wigner_mode);
    const float c1 = load_wigner_value(wigner, offsets, compact_offsets, edge, l,
                                       d, row1, dense_stride, compact_stride,
                                       wigner_mode);
    atomicAdd(out + edge * out_dim + base + d, v0 * c0 + v1 * c1);
  }
}

__global__ void persistent_grouped_forward_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ edge_order,
    const int64_t* __restrict__ route_ptr,
    const int64_t* __restrict__ problem_tile_prefix,
    const float* __restrict__ weight_flat,
    const int64_t* __restrict__ weight_offsets,
    const float* __restrict__ bias_flat,
    const int64_t* __restrict__ bias_offsets,
    const int64_t* __restrict__ m_values,
    const int64_t* __restrict__ in_ptr,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ out_ptr,
    const int64_t* __restrict__ out_base,
    const int64_t* __restrict__ out_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    const float* __restrict__ radial_all,
    const int64_t* __restrict__ m_in_index,
    float* __restrict__ out,
    unsigned long long* __restrict__ counter,
    int64_t n_edges,
    int64_t in_dim,
    int64_t out_dim,
    int64_t n_routes,
    int64_t n_m,
    int64_t n_problems,
    int64_t total_tiles,
    int64_t radial_dim,
    int64_t dense_stride,
    int64_t compact_stride,
    int64_t wigner_mode,
    int64_t block_m,
    int64_t block_n,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    bool has_radial) {
  __shared__ unsigned long long tile_shared;
  while (true) {
    if (threadIdx.x == 0) {
      tile_shared = atomicAdd(counter, 1ULL);
    }
    __syncthreads();
    const unsigned long long tile_ull = tile_shared;
    if (tile_ull >= static_cast<unsigned long long>(total_tiles)) {
      return;
    }
    const int64_t tile = static_cast<int64_t>(tile_ull);
    const int64_t problem = find_problem_for_tile(problem_tile_prefix, n_problems, tile);
    const int64_t route = problem / n_m;
    const int64_t m_idx = problem - route * n_m;
    const int64_t m = m_values[m_idx];

    const int64_t route_begin = route_ptr[route];
    const int64_t route_end = route_ptr[route + 1];
    const int64_t rows = route_end - route_begin;
    if (rows <= 0) {
      continue;
    }

    const int64_t in_begin = in_ptr[m_idx];
    const int64_t in_end = in_ptr[m_idx + 1];
    const int64_t cin = in_end - in_begin;
    const int64_t out_begin = out_ptr[m_idx];
    const int64_t out_end = out_ptr[m_idx + 1];
    const int64_t cout = out_end - out_begin;
    if (cin <= 0 || cout <= 0) {
      continue;
    }

    const int64_t col_tiles = ceil_div_i64(cout, block_n);
    const int64_t local = tile - problem_tile_prefix[problem];
    const int64_t row_tile = local / col_tiles;
    const int64_t col_tile = local - row_tile * col_tiles;
    const int64_t row0 = row_tile * block_m;
    const int64_t col0 = col_tile * block_n;
    const int64_t row_count = (m == 0) ? cout : 2 * cout;
    const int64_t w_off = weight_offsets[m_idx] + route * row_count * cin;
    const int64_t b_off_raw = bias_offsets[m_idx];
    const int64_t b_off = b_off_raw >= 0 ? b_off_raw + route * row_count : -1;
    const int64_t radial_begin = has_radial ? m_in_index[m] : 0;

    const int64_t work_items = block_m * block_n;
    for (int64_t linear = threadIdx.x; linear < work_items; linear += blockDim.x) {
      const int64_t local_row = linear / block_n;
      const int64_t local_col = linear - local_row * block_n;
      const int64_t sorted_row = row0 + local_row;
      const int64_t oc = col0 + local_col;
      if (sorted_row >= rows || oc >= cout) {
        continue;
      }
      const int64_t edge = edge_order[route_begin + sorted_row];
      if (edge < 0 || edge >= n_edges) {
        continue;
      }

      if (m == 0) {
        float acc = (b_off >= 0) ? bias_flat[b_off + oc] : 0.0f;
        for (int64_t ci = 0; ci < cin; ++ci) {
          const int64_t src = in_begin + ci;
          const int64_t l = in_l[src];
          float xv = load_scalar_component(x, wigner, offsets, compact_offsets,
                                           edge, in_dim, in_base[src], l,
                                           dense_stride, compact_stride,
                                           wigner_mode, rotate_in);
          if (has_radial && radial_on_input) {
            xv *= radial_all[edge * radial_dim + radial_begin + ci];
          }
          acc += xv * weight_flat[w_off + oc * cin + ci];
        }
        if (has_radial && !radial_on_input) {
          acc *= radial_all[edge * radial_dim + radial_begin + oc];
        }
        const int64_t dst = out_begin + oc;
        store_scalar_component(out, wigner, offsets, compact_offsets, edge, out_dim,
                               out_base[dst], out_l[dst], acc, dense_stride,
                               compact_stride, wigner_mode, rotate_out);
      } else {
        float acc0 = (b_off >= 0) ? bias_flat[b_off + oc] : 0.0f;
        float acc1 = (b_off >= 0) ? bias_flat[b_off + cout + oc] : 0.0f;
        for (int64_t ci = 0; ci < cin; ++ci) {
          const int64_t src = in_begin + ci;
          const int64_t l = in_l[src];
          const int64_t r0 = l - m;
          const int64_t r1 = l + m;
          float x0 = load_pair_component(x, wigner, offsets, compact_offsets,
                                         edge, in_dim, in_base[src], l, r0,
                                         dense_stride, compact_stride,
                                         wigner_mode, rotate_in);
          float x1 = load_pair_component(x, wigner, offsets, compact_offsets,
                                         edge, in_dim, in_base[src], l, r1,
                                         dense_stride, compact_stride,
                                         wigner_mode, rotate_in);
          if (has_radial && radial_on_input) {
            const float rv = radial_all[edge * radial_dim + radial_begin + ci];
            x0 *= rv;
            x1 *= rv;
          }
          const float wr = weight_flat[w_off + oc * cin + ci];
          const float wi = weight_flat[w_off + (cout + oc) * cin + ci];
          acc0 += x0 * wr - x1 * wi;
          acc1 += x1 * wr + x0 * wi;
        }
        if (has_radial && !radial_on_input) {
          const float rv = radial_all[edge * radial_dim + radial_begin + oc];
          acc0 *= rv;
          acc1 *= rv;
        }
        const int64_t dst = out_begin + oc;
        const int64_t l = out_l[dst];
        store_pair_component(out, wigner, offsets, compact_offsets, edge, out_dim,
                             out_base[dst], l, l - m, l + m, acc0, acc1,
                             dense_stride, compact_stride, wigner_mode, rotate_out);
      }
    }
  }
}

constexpr int kWarpTileNMax = 16;

__device__ __forceinline__ float warp_sum(float value) {
  unsigned mask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

__global__ void persistent_grouped_forward_warp_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ edge_order,
    const int64_t* __restrict__ route_ptr,
    const int64_t* __restrict__ problem_tile_prefix,
    const float* __restrict__ weight_flat,
    const int64_t* __restrict__ weight_offsets,
    const float* __restrict__ bias_flat,
    const int64_t* __restrict__ bias_offsets,
    const int64_t* __restrict__ m_values,
    const int64_t* __restrict__ in_ptr,
    const int64_t* __restrict__ in_base,
    const int64_t* __restrict__ in_l,
    const int64_t* __restrict__ out_ptr,
    const int64_t* __restrict__ out_base,
    const int64_t* __restrict__ out_l,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ compact_offsets,
    const float* __restrict__ radial_all,
    const int64_t* __restrict__ m_in_index,
    float* __restrict__ out,
    unsigned long long* __restrict__ counter,
    int64_t n_edges,
    int64_t in_dim,
    int64_t out_dim,
    int64_t n_routes,
    int64_t n_m,
    int64_t n_problems,
    int64_t total_tiles,
    int64_t radial_dim,
    int64_t dense_stride,
    int64_t compact_stride,
    int64_t wigner_mode,
    int64_t block_m,
    int64_t block_n,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    bool has_radial) {
  __shared__ unsigned long long tile_shared;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;

  while (true) {
    if (threadIdx.x == 0) {
      tile_shared = atomicAdd(counter, 1ULL);
    }
    __syncthreads();
    const unsigned long long tile_ull = tile_shared;
    if (tile_ull >= static_cast<unsigned long long>(total_tiles)) {
      return;
    }
    if (warp >= block_m || warp >= warps_per_block) {
      continue;
    }

    const int64_t tile = static_cast<int64_t>(tile_ull);
    const int64_t problem = find_problem_for_tile(problem_tile_prefix, n_problems, tile);
    const int64_t route = problem / n_m;
    const int64_t m_idx = problem - route * n_m;
    const int64_t m = m_values[m_idx];

    const int64_t route_begin = route_ptr[route];
    const int64_t route_end = route_ptr[route + 1];
    const int64_t rows = route_end - route_begin;
    if (rows <= 0) {
      continue;
    }

    const int64_t in_begin = in_ptr[m_idx];
    const int64_t in_end = in_ptr[m_idx + 1];
    const int64_t cin = in_end - in_begin;
    const int64_t out_begin = out_ptr[m_idx];
    const int64_t out_end = out_ptr[m_idx + 1];
    const int64_t cout = out_end - out_begin;
    if (cin <= 0 || cout <= 0) {
      continue;
    }

    const int64_t col_tiles = ceil_div_i64(cout, block_n);
    const int64_t local = tile - problem_tile_prefix[problem];
    const int64_t row_tile = local / col_tiles;
    const int64_t col_tile = local - row_tile * col_tiles;
    const int64_t sorted_row = row_tile * block_m + warp;
    const int64_t col0 = col_tile * block_n;
    if (sorted_row >= rows) {
      continue;
    }

    const int64_t edge = edge_order[route_begin + sorted_row];
    if (edge < 0 || edge >= n_edges) {
      continue;
    }

    const int64_t row_count = (m == 0) ? cout : 2 * cout;
    const int64_t w_off = weight_offsets[m_idx] + route * row_count * cin;
    const int64_t b_off_raw = bias_offsets[m_idx];
    const int64_t b_off = b_off_raw >= 0 ? b_off_raw + route * row_count : -1;
    const int64_t radial_begin = has_radial ? m_in_index[m] : 0;

    float acc0[kWarpTileNMax];
    float acc1[kWarpTileNMax];
#pragma unroll
    for (int t = 0; t < kWarpTileNMax; ++t) {
      acc0[t] = 0.0f;
      acc1[t] = 0.0f;
    }

    if (m == 0) {
      for (int64_t ci = lane; ci < cin; ci += 32) {
        const int64_t src = in_begin + ci;
        const int64_t l = in_l[src];
        float xv = load_scalar_component(x, wigner, offsets, compact_offsets,
                                         edge, in_dim, in_base[src], l,
                                         dense_stride, compact_stride,
                                         wigner_mode, rotate_in);
        if (has_radial && radial_on_input) {
          xv *= radial_all[edge * radial_dim + radial_begin + ci];
        }
#pragma unroll
        for (int t = 0; t < kWarpTileNMax; ++t) {
          if (t >= block_n) {
            continue;
          }
          const int64_t oc = col0 + t;
          if (oc >= cout) {
            continue;
          }
          acc0[t] += xv * weight_flat[w_off + oc * cin + ci];
        }
      }
#pragma unroll
      for (int t = 0; t < kWarpTileNMax; ++t) {
        if (t >= block_n) {
          continue;
        }
        float value = warp_sum(acc0[t]);
        if (lane == 0) {
          const int64_t oc = col0 + t;
          if (oc >= cout) {
            continue;
          }
          if (b_off >= 0) {
            value += bias_flat[b_off + oc];
          }
          if (has_radial && !radial_on_input) {
            value *= radial_all[edge * radial_dim + radial_begin + oc];
          }
          const int64_t dst = out_begin + oc;
          store_scalar_component(out, wigner, offsets, compact_offsets, edge, out_dim,
                                 out_base[dst], out_l[dst], value, dense_stride,
                                 compact_stride, wigner_mode, rotate_out);
        }
      }
    } else {
      for (int64_t ci = lane; ci < cin; ci += 32) {
        const int64_t src = in_begin + ci;
        const int64_t l = in_l[src];
        const int64_t r0 = l - m;
        const int64_t r1 = l + m;
        float x0 = load_pair_component(x, wigner, offsets, compact_offsets,
                                       edge, in_dim, in_base[src], l, r0,
                                       dense_stride, compact_stride,
                                       wigner_mode, rotate_in);
        float x1 = load_pair_component(x, wigner, offsets, compact_offsets,
                                       edge, in_dim, in_base[src], l, r1,
                                       dense_stride, compact_stride,
                                       wigner_mode, rotate_in);
        if (has_radial && radial_on_input) {
          const float rv = radial_all[edge * radial_dim + radial_begin + ci];
          x0 *= rv;
          x1 *= rv;
        }
#pragma unroll
        for (int t = 0; t < kWarpTileNMax; ++t) {
          if (t >= block_n) {
            continue;
          }
          const int64_t oc = col0 + t;
          if (oc >= cout) {
            continue;
          }
          const float wr = weight_flat[w_off + oc * cin + ci];
          const float wi = weight_flat[w_off + (cout + oc) * cin + ci];
          acc0[t] += x0 * wr - x1 * wi;
          acc1[t] += x1 * wr + x0 * wi;
        }
      }
#pragma unroll
      for (int t = 0; t < kWarpTileNMax; ++t) {
        if (t >= block_n) {
          continue;
        }
        float value0 = warp_sum(acc0[t]);
        float value1 = warp_sum(acc1[t]);
        if (lane == 0) {
          const int64_t oc = col0 + t;
          if (oc >= cout) {
            continue;
          }
          if (b_off >= 0) {
            value0 += bias_flat[b_off + oc];
            value1 += bias_flat[b_off + cout + oc];
          }
          if (has_radial && !radial_on_input) {
            const float rv = radial_all[edge * radial_dim + radial_begin + oc];
            value0 *= rv;
            value1 *= rv;
          }
          const int64_t dst = out_begin + oc;
          const int64_t l = out_l[dst];
          store_pair_component(out, wigner, offsets, compact_offsets, edge, out_dim,
                               out_base[dst], l, l - m, l + m, value0, value1,
                               dense_stride, compact_stride, wigner_mode, rotate_out);
        }
      }
    }
  }
}

}  // namespace

at::Tensor persistent_grouped_forward_fp32_cuda(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  at::Tensor out = at::zeros({x.size(0), out_dim}, x.options());
  if (x.size(0) == 0 || out_dim == 0 || problem_tile_prefix.numel() == 0) {
    return out;
  }

  const int64_t n_m = m_values.numel();
  const int64_t n_problems = n_routes * n_m;
  if (n_problems == 0) {
    return out;
  }

  const int64_t total_tiles = problem_tile_prefix.narrow(0, problem_tile_prefix.numel() - 1, 1).item<int64_t>();
  if (total_tiles <= 0) {
    return out;
  }

  cudaDeviceProp props{};
  int dev = -1;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&props, dev);
  const int64_t sm_count = std::max(1, props.multiProcessorCount);
  int64_t blocks_i64 = active_blocks > 0 ? active_blocks : std::min<int64_t>(std::max<int64_t>(1, total_tiles), sm_count * 4);
  blocks_i64 = std::min<int64_t>(blocks_i64, 65535);
  const dim3 grid(static_cast<unsigned int>(blocks_i64));
  const dim3 threads(256);

  at::Tensor counter = at::zeros({1}, at::TensorOptions().device(x.device()).dtype(at::kLong));
  const bool has_radial = radial_all.numel() > 0;
  const int64_t radial_dim = has_radial ? radial_all.size(1) : 0;
  const int64_t compact_stride = (wigner_mode == 2) ? wigner_stride : 0;
  const int64_t dense_stride = (wigner_mode == 1) ? wigner_stride : 0;

  persistent_grouped_forward_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(),
      wigner.numel() > 0 ? wigner.data_ptr<float>() : nullptr,
      edge_order.data_ptr<int64_t>(),
      route_ptr.data_ptr<int64_t>(),
      problem_tile_prefix.data_ptr<int64_t>(),
      weight_flat.data_ptr<float>(),
      weight_offsets.data_ptr<int64_t>(),
      bias_flat.numel() > 0 ? bias_flat.data_ptr<float>() : nullptr,
      bias_offsets.data_ptr<int64_t>(),
      m_values.data_ptr<int64_t>(),
      in_ptr.data_ptr<int64_t>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      out_ptr.data_ptr<int64_t>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() > 0 ? compact_offsets.data_ptr<int64_t>() : nullptr,
      has_radial ? radial_all.data_ptr<float>() : nullptr,
      m_in_index.numel() > 0 ? m_in_index.data_ptr<int64_t>() : nullptr,
      out.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(counter.data_ptr<int64_t>()),
      x.size(0),
      x.size(1),
      out_dim,
      n_routes,
      n_m,
      n_problems,
      total_tiles,
      radial_dim,
      dense_stride,
      compact_stride,
      wigner_mode,
      block_m,
      block_n,
      rotate_in,
      rotate_out,
      radial_on_input,
      has_radial);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor persistent_grouped_forward_warp_fp32_cuda(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks) {
  const c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(block_n <= kWarpTileNMax, "warp collective block_n must be <= 16");
  at::Tensor out = at::zeros({x.size(0), out_dim}, x.options());
  if (x.size(0) == 0 || out_dim == 0 || problem_tile_prefix.numel() == 0) {
    return out;
  }

  const int64_t n_m = m_values.numel();
  const int64_t n_problems = n_routes * n_m;
  if (n_problems == 0) {
    return out;
  }

  const int64_t total_tiles = problem_tile_prefix.narrow(0, problem_tile_prefix.numel() - 1, 1).item<int64_t>();
  if (total_tiles <= 0) {
    return out;
  }

  cudaDeviceProp props{};
  int dev = -1;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&props, dev);
  const int64_t sm_count = std::max(1, props.multiProcessorCount);
  int64_t blocks_i64 = active_blocks > 0 ? active_blocks : std::min<int64_t>(std::max<int64_t>(1, total_tiles), sm_count * 4);
  blocks_i64 = std::min<int64_t>(blocks_i64, 65535);
  const dim3 grid(static_cast<unsigned int>(blocks_i64));
  const dim3 threads(256);

  at::Tensor counter = at::zeros({1}, at::TensorOptions().device(x.device()).dtype(at::kLong));
  const bool has_radial = radial_all.numel() > 0;
  const int64_t radial_dim = has_radial ? radial_all.size(1) : 0;
  const int64_t compact_stride = (wigner_mode == 2) ? wigner_stride : 0;
  const int64_t dense_stride = (wigner_mode == 1) ? wigner_stride : 0;

  persistent_grouped_forward_warp_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(),
      wigner.numel() > 0 ? wigner.data_ptr<float>() : nullptr,
      edge_order.data_ptr<int64_t>(),
      route_ptr.data_ptr<int64_t>(),
      problem_tile_prefix.data_ptr<int64_t>(),
      weight_flat.data_ptr<float>(),
      weight_offsets.data_ptr<int64_t>(),
      bias_flat.numel() > 0 ? bias_flat.data_ptr<float>() : nullptr,
      bias_offsets.data_ptr<int64_t>(),
      m_values.data_ptr<int64_t>(),
      in_ptr.data_ptr<int64_t>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      out_ptr.data_ptr<int64_t>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() > 0 ? compact_offsets.data_ptr<int64_t>() : nullptr,
      has_radial ? radial_all.data_ptr<float>() : nullptr,
      m_in_index.numel() > 0 ? m_in_index.data_ptr<int64_t>() : nullptr,
      out.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(counter.data_ptr<int64_t>()),
      x.size(0),
      x.size(1),
      out_dim,
      n_routes,
      n_m,
      n_problems,
      total_tiles,
      radial_dim,
      dense_stride,
      compact_stride,
      wigner_mode,
      block_m,
      block_n,
      rotate_in,
      rotate_out,
      radial_on_input,
      has_radial);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
