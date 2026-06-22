#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#ifdef DPTB_SO2_MOE_PERSISTENT_P1_CUTE
#include <cute/tensor.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/matrix.h"
#endif

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

#ifdef DPTB_SO2_MOE_PERSISTENT_P1_CUTE

constexpr int kCuteTileMMax = 16;
constexpr int kCuteTileNMax = 32;
constexpr int kCuteTileK = 32;
constexpr int kCuteThreads = 256;

__global__ void persistent_grouped_forward_cute_tiled_kernel(
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
  extern __shared__ float smem[];
  auto a_layout = cute::make_layout(
      cute::make_shape(cute::Int<2>{}, cute::Int<kCuteTileMMax>{}, cute::Int<kCuteTileK>{}));
  auto b_layout = cute::make_layout(
      cute::make_shape(cute::Int<2>{}, cute::Int<kCuteTileNMax>{}, cute::Int<kCuteTileK>{}));
  float* smem_a = smem;
  float* smem_b = smem_a + 2 * kCuteTileMMax * kCuteTileK;

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
    const int64_t sorted_row0 = row_tile * block_m;
    const int64_t col0 = col_tile * block_n;
    const int64_t row_count = (m == 0) ? cout : 2 * cout;
    const int64_t w_off = weight_offsets[m_idx] + route * row_count * cin;
    const int64_t b_off_raw = bias_offsets[m_idx];
    const int64_t b_off = b_off_raw >= 0 ? b_off_raw + route * row_count : -1;
    const int64_t radial_begin = has_radial ? m_in_index[m] : 0;

    const int64_t out_linear = threadIdx.x;
    const int64_t local_row = out_linear / block_n;
    const int64_t local_col = out_linear - local_row * block_n;
    const bool owns_output = out_linear < block_m * block_n &&
                             local_row < block_m &&
                             local_col < block_n &&
                             sorted_row0 + local_row < rows &&
                             col0 + local_col < cout;
    const int64_t edge = owns_output ? edge_order[route_begin + sorted_row0 + local_row] : -1;
    const int64_t oc = col0 + local_col;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int64_t k0 = 0; k0 < cin; k0 += kCuteTileK) {
      for (int64_t idx = threadIdx.x; idx < block_m * kCuteTileK; idx += blockDim.x) {
        const int64_t lr = idx / kCuteTileK;
        const int64_t kk = idx - lr * kCuteTileK;
        const int64_t ci = k0 + kk;
        float a0 = 0.0f;
        float a1 = 0.0f;
        if (lr < block_m && sorted_row0 + lr < rows && ci < cin) {
          const int64_t e = edge_order[route_begin + sorted_row0 + lr];
          const int64_t src = in_begin + ci;
          const int64_t l = in_l[src];
          if (m == 0) {
            a0 = load_scalar_component(x, wigner, offsets, compact_offsets,
                                       e, in_dim, in_base[src], l,
                                       dense_stride, compact_stride,
                                       wigner_mode, rotate_in);
          } else {
            a0 = load_pair_component(x, wigner, offsets, compact_offsets,
                                     e, in_dim, in_base[src], l, l - m,
                                     dense_stride, compact_stride,
                                     wigner_mode, rotate_in);
            a1 = load_pair_component(x, wigner, offsets, compact_offsets,
                                     e, in_dim, in_base[src], l, l + m,
                                     dense_stride, compact_stride,
                                     wigner_mode, rotate_in);
          }
          if (has_radial && radial_on_input) {
            const float rv = radial_all[e * radial_dim + radial_begin + ci];
            a0 *= rv;
            a1 *= rv;
          }
        }
        smem_a[a_layout(0, lr, kk)] = a0;
        smem_a[a_layout(1, lr, kk)] = a1;
      }

      for (int64_t idx = threadIdx.x; idx < block_n * kCuteTileK; idx += blockDim.x) {
        const int64_t lc = idx / kCuteTileK;
        const int64_t kk = idx - lc * kCuteTileK;
        const int64_t ci = k0 + kk;
        const int64_t oc_i = col0 + lc;
        float wr = 0.0f;
        float wi = 0.0f;
        if (lc < block_n && oc_i < cout && ci < cin) {
          wr = weight_flat[w_off + oc_i * cin + ci];
          if (m != 0) {
            wi = weight_flat[w_off + (cout + oc_i) * cin + ci];
          }
        }
        smem_b[b_layout(0, lc, kk)] = wr;
        smem_b[b_layout(1, lc, kk)] = wi;
      }
      __syncthreads();

      if (owns_output && edge >= 0 && edge < n_edges) {
        const int64_t remaining_k = cin - k0;
        const int64_t valid_k = remaining_k < static_cast<int64_t>(kCuteTileK)
                                     ? remaining_k
                                     : static_cast<int64_t>(kCuteTileK);
        for (int64_t kk = 0; kk < valid_k; ++kk) {
          const float a0 = smem_a[a_layout(0, local_row, kk)];
          const float a1 = smem_a[a_layout(1, local_row, kk)];
          const float wr = smem_b[b_layout(0, local_col, kk)];
          const float wi = smem_b[b_layout(1, local_col, kk)];
          if (m == 0) {
            acc0 += a0 * wr;
          } else {
            acc0 += a0 * wr - a1 * wi;
            acc1 += a1 * wr + a0 * wi;
          }
        }
      }
      __syncthreads();
    }

    if (!owns_output || edge < 0 || edge >= n_edges) {
      continue;
    }

    if (m == 0) {
      if (b_off >= 0) {
        acc0 += bias_flat[b_off + oc];
      }
      if (has_radial && !radial_on_input) {
        acc0 *= radial_all[edge * radial_dim + radial_begin + oc];
      }
      const int64_t dst = out_begin + oc;
      store_scalar_component(out, wigner, offsets, compact_offsets, edge, out_dim,
                             out_base[dst], out_l[dst], acc0, dense_stride,
                             compact_stride, wigner_mode, rotate_out);
    } else {
      if (b_off >= 0) {
        acc0 += bias_flat[b_off + oc];
        acc1 += bias_flat[b_off + cout + oc];
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

using NativeElement = float;
using NativeLayoutA = cutlass::layout::RowMajor;
using NativeLayoutB = cutlass::layout::ColumnMajor;
using NativeLayoutC = cutlass::layout::RowMajor;
using NativeThreadblockShape = cutlass::gemm::GemmShape<64, 32, 8>;
using NativeWarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
using NativeInstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
using NativeEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    NativeElement,
    1,
    NativeElement,
    NativeElement>;

using NativeBaseGemm = cutlass::gemm::device::GemmUniversal<
    NativeElement,
    NativeLayoutA,
    NativeElement,
    NativeLayoutB,
    NativeElement,
    NativeLayoutC,
    NativeElement,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    NativeThreadblockShape,
    NativeWarpShape,
    NativeInstructionShape,
    NativeEpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    1,
    1>;

template <typename BaseIterator_>
class PersistentSo2AIterator {
 public:
  using BaseIterator = BaseIterator_;
  using Shape = typename BaseIterator::Shape;
  using Element = typename BaseIterator::Element;
  using Layout = typename BaseIterator::Layout;
  static int const kAdvanceRank = BaseIterator::kAdvanceRank;
  using ThreadMap = typename BaseIterator::ThreadMap;
  using Index = typename BaseIterator::Index;
  using LongIndex = typename BaseIterator::LongIndex;
  using TensorCoord = typename BaseIterator::TensorCoord;
  using Pointer = typename BaseIterator::Pointer;
  using AccessType = typename BaseIterator::AccessType;
  using Fragment = typename BaseIterator::Fragment;
  using Mask = typename BaseIterator::Mask;

  struct Params {
    const float* x;
    const float* wigner;
    const int64_t* edge_order;
    const int64_t* route_ptr;
    const int64_t* in_base;
    const int64_t* in_l;
    const int64_t* offsets;
    const int64_t* compact_offsets;
    const float* radial_all;
    const int64_t* m_in_index;
    int64_t route;
    int64_t m;
    int64_t in_begin;
    int64_t cin;
    int64_t in_dim;
    int64_t radial_dim;
    int64_t dense_stride;
    int64_t compact_stride;
    int64_t wigner_mode;
    bool rotate_in;
    bool radial_on_input;
    bool has_radial;

    CUTLASS_HOST_DEVICE
    Params() = default;

    CUTLASS_HOST_DEVICE
    Params(Layout const& /*layout*/) {}
  };

 private:
  Params params_;
  TensorCoord extent_;
  cutlass::layout::PitchLinearCoord thread_offset_;

 public:
  PersistentSo2AIterator() = default;

  CUTLASS_HOST_DEVICE
  PersistentSo2AIterator(
      Params const& params,
      Pointer /*pointer*/,
      TensorCoord extent,
      int thread_id,
      TensorCoord const& threadblock_offset,
      int const* /*indices*/ = nullptr)
      : params_(params),
        extent_(extent),
        thread_offset_(threadblock_offset.column(), threadblock_offset.row()) {
    thread_offset_ += ThreadMap::initial_offset(thread_id);
  }

  CUTLASS_HOST_DEVICE void add_pointer_offset(LongIndex /*pointer_offset*/) {}

  CUTLASS_HOST_DEVICE
  PersistentSo2AIterator& operator++() {
    thread_offset_ += cutlass::layout::PitchLinearCoord(Shape::kColumn, 0);
    return *this;
  }

  CUTLASS_HOST_DEVICE
  PersistentSo2AIterator operator++(int) {
    PersistentSo2AIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool /*enable*/ = true) {}
  CUTLASS_HOST_DEVICE void enable_mask() {}
  CUTLASS_HOST_DEVICE void set_mask(Mask const& /*mask*/) {}
  CUTLASS_HOST_DEVICE void get_mask(Mask& /*mask*/) const {}

 private:
  CUTLASS_DEVICE
  float load_one(int matrix_row, int k2) const {
    if (matrix_row < 0 || k2 < 0 || matrix_row >= extent_.row() || k2 >= extent_.column()) {
      return 0.0f;
    }
    const int64_t sorted_row = matrix_row;
    const int64_t ci = k2 >> 1;
    const int64_t pair_row = k2 & 1;
    const int64_t edge = params_.edge_order[params_.route_ptr[params_.route] + sorted_row];
    const int64_t src = params_.in_begin + ci;
    const int64_t l = params_.in_l[src];
    const int64_t row = pair_row ? (l + params_.m) : (l - params_.m);
    float value = load_pair_component(
        params_.x,
        params_.wigner,
        params_.offsets,
        params_.compact_offsets,
        edge,
        params_.in_dim,
        params_.in_base[src],
        l,
        row,
        params_.dense_stride,
        params_.compact_stride,
        params_.wigner_mode,
        params_.rotate_in);
    if (params_.has_radial && params_.radial_on_input) {
      const int64_t radial_begin = params_.m_in_index[params_.m];
      value *= params_.radial_all[edge * params_.radial_dim + radial_begin + ci];
    }
    return value;
  }

 public:
  CUTLASS_DEVICE
  void load(Fragment& frag) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < ThreadMap::kElementsPerAccess; ++v) {
          const int idx = v + ThreadMap::kElementsPerAccess *
              (c + s * ThreadMap::Iterations::kContiguous);
          const int matrix_col = thread_offset_.contiguous() +
              c * ThreadMap::Delta::kContiguous + v;
          const int matrix_row = thread_offset_.strided() +
              s * ThreadMap::Delta::kStrided;
          frag[idx] = load_one(matrix_row, matrix_col);
        }
      }
    }
  }

  CUTLASS_DEVICE void load_with_pointer_offset(Fragment& frag, Index /*pointer_offset*/) { load(frag); }
  CUTLASS_DEVICE void load_with_byte_offset(Fragment& frag, LongIndex /*byte_offset*/) { load(frag); }
};

template <typename OutputTileIterator_, typename ThreadblockShape_>
class PersistentSo2PairCallbacks {
 public:
  using OutputTileIterator = OutputTileIterator_;
  using ThreadblockShape = ThreadblockShape_;
  using ElementAccumulator = float;

  struct Arguments {
    float* out;
    const float* wigner;
    const int64_t* edge_order;
    const int64_t* route_ptr;
    const int64_t* out_base;
    const int64_t* out_l;
    const int64_t* out_ptr;
    const int64_t* offsets;
    const int64_t* compact_offsets;
    const float* radial_all;
    const int64_t* m_in_index;
    int64_t route;
    int64_t m;
    int64_t out_begin;
    int64_t cout;
    int64_t raw_m;
    int64_t raw_n;
    int64_t out_dim;
    int64_t radial_dim;
    int64_t dense_stride;
    int64_t compact_stride;
    int64_t wigner_mode;
    bool rotate_out;
    bool radial_on_input;
    bool has_radial;
  };

  struct Params : Arguments {
    typename OutputTileIterator::Params params_raw;
    float* ptr_raw;

    CUTLASS_HOST_DEVICE
    Params() : params_raw(NativeLayoutC(0)), ptr_raw(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args)
        : Arguments(args),
          params_raw(NativeLayoutC(args.raw_n)),
          ptr_raw(args.out) {}
  };

  struct SharedStorage {};

  CUTLASS_HOST_DEVICE
  static Params to_underlying_arguments(
      cutlass::gemm::GemmCoord const& /*problem_size*/,
      Arguments const& args,
      void* /*workspace*/) {
    return Params(args);
  }

 private:
  Params params_;

 public:
  CUTLASS_DEVICE
  PersistentSo2PairCallbacks(Params const& params, SharedStorage& /*shared_storage*/)
      : params_(params) {}

  class RuntimeCallbacks {
   private:
    Params params_;
    OutputTileIterator iterator_raw_;
    cutlass::MatrixCoord extent_;

   public:
    CUTLASS_DEVICE
    RuntimeCallbacks(Params const& params, OutputTileIterator iterator_raw, cutlass::MatrixCoord extent)
        : params_(params), iterator_raw_(iterator_raw), extent_(extent) {}

    CUTLASS_DEVICE void begin_epilogue() {}
    CUTLASS_DEVICE void begin_step(int /*step_idx*/) {}
    CUTLASS_DEVICE void begin_row(int /*row_idx*/) {}
    CUTLASS_DEVICE void end_row(int /*row_idx*/) {}

    template <typename ElementAccumulator_, int FragmentSize>
    CUTLASS_DEVICE void visit(
        int /*iter_idx*/,
        int /*row_idx*/,
        int /*column_idx*/,
        int frag_idx,
        cutlass::Array<ElementAccumulator_, FragmentSize> const& accum) {
      cutlass::MatrixCoord base =
          iterator_raw_.thread_start() + OutputTileIterator::ThreadMap::iteration_offset(frag_idx);

      CUTLASS_PRAGMA_UNROLL
      for (int v = 0; v < FragmentSize; ++v) {
        const int raw_row = base.row();
        const int raw_col = base.column() + v;
        if (raw_row < 0 || raw_col < 0 || raw_row >= extent_.row() || raw_col >= extent_.column()) {
          continue;
        }
        const int64_t sorted_row = raw_row;
        const int64_t edge = params_.edge_order[params_.route_ptr[params_.route] + sorted_row];
        const int64_t cout = params_.cout;
        if (raw_col >= 2 * cout) {
          continue;
        }
        if ((raw_col & 1) && v > 0 && (base.column() + v - 1) == (raw_col - 1)) {
          continue;
        }
        const int64_t channel = raw_col >> 1;
        const bool imag_col = (raw_col & 1) != 0;
        const int64_t dst = params_.out_begin + channel;
        const int64_t l = params_.out_l[dst];
        const bool has_pair = !imag_col && (v + 1 < FragmentSize) && (raw_col + 1 < extent_.column());

        float value = static_cast<float>(accum[v]);
        float value_pair = has_pair ? static_cast<float>(accum[v + 1]) : 0.0f;
        if (params_.has_radial && !params_.radial_on_input) {
          const int64_t radial_begin = params_.m_in_index[params_.m];
          const float r = params_.radial_all[edge * params_.radial_dim + radial_begin + channel];
          value *= r;
          value_pair *= r;
        }
        const int64_t dim = 2 * l + 1;
        const int64_t base_out = params_.out_base[dst];
        if (!params_.rotate_out || params_.wigner_mode == 0) {
          if (has_pair) {
            atomicAdd(params_.out + edge * params_.out_dim + base_out + l - params_.m, value);
            atomicAdd(params_.out + edge * params_.out_dim + base_out + l + params_.m, value_pair);
          } else {
            atomicAdd(
                params_.out + edge * params_.out_dim + base_out + (imag_col ? l + params_.m : l - params_.m),
                value);
          }
          continue;
        }
        const int64_t wigner_col = imag_col ? (l + params_.m) : (l - params_.m);
        const int64_t wigner_col_pair = l + params_.m;
        for (int64_t d = 0; d < dim; ++d) {
          const float coeff = load_wigner_value(
              params_.wigner,
              params_.offsets,
              params_.compact_offsets,
              edge,
              l,
              d,
              wigner_col,
              params_.dense_stride,
              params_.compact_stride,
              params_.wigner_mode);
          float update = value * coeff;
          if (has_pair) {
            const float coeff_pair = load_wigner_value(
                params_.wigner,
                params_.offsets,
                params_.compact_offsets,
                edge,
                l,
                d,
                wigner_col_pair,
                params_.dense_stride,
                params_.compact_stride,
                params_.wigner_mode);
            update += value_pair * coeff_pair;
          }
          atomicAdd(params_.out + edge * params_.out_dim + base_out + d, update);
        }
      }
    }

    CUTLASS_DEVICE void end_step(int /*step_idx*/) { ++iterator_raw_; }
    CUTLASS_DEVICE void end_epilogue() {}
  };

  template <class ProblemShape>
  CUTLASS_DEVICE RuntimeCallbacks get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset,
      int thread_idx,
      ProblemShape /*problem_shape*/) {
    cutlass::MatrixCoord extent(params_.raw_m, params_.raw_n);
    cutlass::MatrixCoord tb_offset(
        threadblock_tile_offset.m() * ThreadblockShape::kM,
        threadblock_tile_offset.n() * ThreadblockShape::kN);
    OutputTileIterator iterator_raw(params_.params_raw, params_.ptr_raw, extent, thread_idx, tb_offset);
    return RuntimeCallbacks(params_, iterator_raw, extent);
  }
};

template <typename BaseIterator_>
class PersistentSo2BIterator {
 public:
  using BaseIterator = BaseIterator_;
  using Shape = typename BaseIterator::Shape;
  using Element = typename BaseIterator::Element;
  using Layout = typename BaseIterator::Layout;
  static int const kAdvanceRank = BaseIterator::kAdvanceRank;
  using ThreadMap = typename BaseIterator::ThreadMap;
  using Index = typename BaseIterator::Index;
  using LongIndex = typename BaseIterator::LongIndex;
  using TensorCoord = typename BaseIterator::TensorCoord;
  using Pointer = typename BaseIterator::Pointer;
  using AccessType = typename BaseIterator::AccessType;
  using Fragment = typename BaseIterator::Fragment;
  using Mask = typename BaseIterator::Mask;

  struct Params {
    const float* weight;
    int64_t cin;
    int64_t cout;

    CUTLASS_HOST_DEVICE Params() : weight(nullptr), cin(0), cout(0) {}
    CUTLASS_HOST_DEVICE Params(Layout const& /*layout*/) : weight(nullptr), cin(0), cout(0) {}
    CUTLASS_HOST_DEVICE Params(Layout const& /*layout*/, const float* weight_, int64_t cin_, int64_t cout_)
        : weight(weight_), cin(cin_), cout(cout_) {}
  };

 private:
  Params params_;
  TensorCoord extent_;
  cutlass::layout::PitchLinearCoord thread_offset_;

 public:
  PersistentSo2BIterator() = default;

  CUTLASS_HOST_DEVICE
  PersistentSo2BIterator(
      Params const& params,
      Pointer /*pointer*/,
      TensorCoord extent,
      int thread_id,
      TensorCoord const& threadblock_offset,
      int const* /*indices*/ = nullptr)
      : params_(params),
        extent_(extent),
        thread_offset_(threadblock_offset.row(), threadblock_offset.column()) {
    thread_offset_ += ThreadMap::initial_offset(thread_id);
  }

  CUTLASS_HOST_DEVICE void add_pointer_offset(LongIndex /*pointer_offset*/) {}

  CUTLASS_HOST_DEVICE
  PersistentSo2BIterator& operator++() {
    thread_offset_ += cutlass::layout::PitchLinearCoord(Shape::kRow, 0);
    return *this;
  }

  CUTLASS_HOST_DEVICE
  PersistentSo2BIterator operator++(int) {
    PersistentSo2BIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE void clear_mask(bool /*enable*/ = true) {}
  CUTLASS_HOST_DEVICE void enable_mask() {}
  CUTLASS_HOST_DEVICE void set_mask(Mask const& /*mask*/) {}
  CUTLASS_HOST_DEVICE void get_mask(Mask& /*mask*/) const {}

 private:
  CUTLASS_DEVICE
  float load_one(int k2, int raw_col) const {
    if (!params_.weight || k2 < 0 || raw_col < 0 || k2 >= extent_.row() || raw_col >= extent_.column()) {
      return 0.0f;
    }
    const int64_t ci = k2 >> 1;
    if (ci >= params_.cin) {
      return 0.0f;
    }
    const bool x1 = (k2 & 1) != 0;
    const int64_t channel = raw_col >> 1;
    const bool imag_out = (raw_col & 1) != 0;
    if (channel < 0 || channel >= params_.cout) {
      return 0.0f;
    }
    const float wr = params_.weight[channel * params_.cin + ci];
    const float wi = params_.weight[(params_.cout + channel) * params_.cin + ci];
    if (!imag_out) {
      return x1 ? -wi : wr;
    }
    return x1 ? wr : wi;
  }

 public:
  CUTLASS_DEVICE
  void load(Fragment& frag) {
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < ThreadMap::Iterations::kStrided; ++s) {
      CUTLASS_PRAGMA_UNROLL
      for (int c = 0; c < ThreadMap::Iterations::kContiguous; ++c) {
        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < ThreadMap::kElementsPerAccess; ++v) {
          const int idx = v + ThreadMap::kElementsPerAccess *
              (c + s * ThreadMap::Iterations::kContiguous);
          const int matrix_k = thread_offset_.contiguous() +
              c * ThreadMap::Delta::kContiguous + v;
          const int matrix_n = thread_offset_.strided() +
              s * ThreadMap::Delta::kStrided;
          frag[idx] = load_one(matrix_k, matrix_n);
        }
      }
    }
  }

  CUTLASS_DEVICE void load_with_pointer_offset(Fragment& frag, Index /*pointer_offset*/) { load(frag); }
  CUTLASS_DEVICE void load_with_byte_offset(Fragment& frag, LongIndex /*byte_offset*/) { load(frag); }
};

using NativeAIterator = PersistentSo2AIterator<typename NativeBaseGemm::GemmKernel::Mma::IteratorA>;
using NativeBIterator = PersistentSo2BIterator<typename NativeBaseGemm::GemmKernel::Mma::IteratorB>;
using NativeMma = cutlass::gemm::threadblock::MmaPipelined<
    NativeThreadblockShape,
    NativeAIterator,
    typename NativeBaseGemm::GemmKernel::Mma::SmemIteratorA,
    NativeBIterator,
    typename NativeBaseGemm::GemmKernel::Mma::SmemIteratorB,
    typename NativeBaseGemm::GemmKernel::Mma::ElementC,
    typename NativeBaseGemm::GemmKernel::Mma::LayoutC,
    typename NativeBaseGemm::GemmKernel::Mma::Policy>;
using NativeCallbacks = PersistentSo2PairCallbacks<
    typename NativeBaseGemm::GemmKernel::Epilogue::OutputTileIterator,
    NativeThreadblockShape>;
using NativeEpilogue = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
    typename NativeBaseGemm::GemmKernel::Epilogue,
    NativeCallbacks,
    1>;

union NativeSharedStorage {
  typename NativeMma::SharedStorage main_loop;
  typename NativeEpilogue::SharedStorage epilogue;
};

__global__ void persistent_grouped_forward_cutlass_native_kernel(
    const float* __restrict__ x,
    const float* __restrict__ wigner,
    const int64_t* __restrict__ edge_order,
    const int64_t* __restrict__ route_ptr,
    const int64_t* __restrict__ problem_tile_prefix,
    const float* __restrict__ weight_flat,
    const int64_t* __restrict__ weight_offsets,
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
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    bool has_radial) {
  __shared__ NativeSharedStorage shared_storage;
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
    if (m <= 0) {
      continue;
    }

    const int64_t route_begin = route_ptr[route];
    const int64_t route_end = route_ptr[route + 1];
    const int64_t rows = route_end - route_begin;
    if (rows <= 0) {
      continue;
    }

    const int64_t in_begin = in_ptr[m_idx];
    const int64_t cin = in_ptr[m_idx + 1] - in_begin;
    const int64_t out_begin = out_ptr[m_idx];
    const int64_t cout = out_ptr[m_idx + 1] - out_begin;
    if (cin <= 0 || cout <= 0) {
      continue;
    }

    const int64_t raw_m = rows;
    const int64_t raw_n = 2 * cout;
    const int64_t gemm_k = 2 * cin;
    const int64_t col_tiles = ceil_div_i64(raw_n, NativeThreadblockShape::kN);
    const int64_t local = tile - problem_tile_prefix[problem];
    const int64_t tile_m = local / col_tiles;
    const int64_t tile_n = local - tile_m * col_tiles;
    const int64_t row_count = 2 * cout;
    const int64_t w_off = weight_offsets[m_idx] + route * row_count * cin;

    const int thread_idx = static_cast<int>(threadIdx.x);
    const int warp_idx = thread_idx / 32;
    const int lane_idx = thread_idx & 31;
    cutlass::MatrixCoord tb_offset_A(
        static_cast<int>(tile_m * NativeThreadblockShape::kM),
        0);
    cutlass::MatrixCoord tb_offset_B(
        0,
        static_cast<int>(tile_n * NativeThreadblockShape::kN));

    typename NativeAIterator::Params params_A;
    params_A.x = x;
    params_A.wigner = wigner;
    params_A.edge_order = edge_order;
    params_A.route_ptr = route_ptr;
    params_A.in_base = in_base;
    params_A.in_l = in_l;
    params_A.offsets = offsets;
    params_A.compact_offsets = compact_offsets;
    params_A.radial_all = radial_all;
    params_A.m_in_index = m_in_index;
    params_A.route = route;
    params_A.m = m;
    params_A.in_begin = in_begin;
    params_A.cin = cin;
    params_A.in_dim = in_dim;
    params_A.radial_dim = radial_dim;
    params_A.dense_stride = dense_stride;
    params_A.compact_stride = compact_stride;
    params_A.wigner_mode = wigner_mode;
    params_A.rotate_in = rotate_in;
    params_A.radial_on_input = radial_on_input;
    params_A.has_radial = has_radial;

    NativeAIterator iterator_A(
        params_A,
        const_cast<float*>(x),
        cutlass::MatrixCoord(static_cast<int>(raw_m), static_cast<int>(gemm_k)),
        thread_idx,
        tb_offset_A,
        nullptr);

    typename NativeBIterator::Params params_B{NativeLayoutB(gemm_k), weight_flat + w_off, cin, cout};
    NativeBIterator iterator_B(
        params_B,
        const_cast<float*>(weight_flat + w_off),
        cutlass::MatrixCoord(static_cast<int>(gemm_k), static_cast<int>(raw_n)),
        thread_idx,
        tb_offset_B,
        nullptr);

    NativeMma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename NativeMma::FragmentC accumulators;
    accumulators.clear();
    const int gemm_k_iterations =
        static_cast<int>((gemm_k + NativeThreadblockShape::kK - 1) / NativeThreadblockShape::kK);
    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

    typename NativeCallbacks::Arguments epilogue_args{
        out,
        wigner,
        edge_order,
        route_ptr,
        out_base,
        out_l,
        out_ptr,
        offsets,
        compact_offsets,
        radial_all,
        m_in_index,
        route,
        m,
        out_begin,
        cout,
        raw_m,
        raw_n,
        out_dim,
        radial_dim,
        dense_stride,
        compact_stride,
        wigner_mode,
        rotate_out,
        radial_on_input,
        has_radial};
    typename NativeCallbacks::Params callback_params(epilogue_args);
    NativeEpilogue epilogue(callback_params, shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
    epilogue(
        accumulators,
        cutlass::gemm::GemmCoord(static_cast<int>(tile_m), static_cast<int>(tile_n), 0),
        cutlass::gemm::GemmCoord(static_cast<int>(raw_m), static_cast<int>(raw_n), 1),
        thread_idx);
    __syncthreads();
  }
}

#endif  // DPTB_SO2_MOE_PERSISTENT_P1_CUTE

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

at::Tensor persistent_grouped_forward_cute_tiled_fp32_cuda(
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
#ifndef DPTB_SO2_MOE_PERSISTENT_P1_CUTE
  TORCH_CHECK(false, "persistent_grouped_forward_cute_tiled_fp32 requires building with DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT or DPTB_CUTLASS_ROOT");
#else
  const c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(block_m <= kCuteTileMMax, "cute_tiled block_m must be <= 16");
  TORCH_CHECK(block_n <= kCuteTileNMax, "cute_tiled block_n must be <= 32");
  TORCH_CHECK(block_m * block_n <= kCuteThreads, "cute_tiled block_m * block_n must be <= 256");
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
  int64_t blocks_i64 = active_blocks > 0 ? active_blocks : std::min<int64_t>(std::max<int64_t>(1, total_tiles), sm_count * 3);
  blocks_i64 = std::min<int64_t>(blocks_i64, 65535);
  const dim3 grid(static_cast<unsigned int>(blocks_i64));
  const dim3 threads(kCuteThreads);

  at::Tensor counter = at::zeros({1}, at::TensorOptions().device(x.device()).dtype(at::kLong));
  const bool has_radial = radial_all.numel() > 0;
  const int64_t radial_dim = has_radial ? radial_all.size(1) : 0;
  const int64_t compact_stride = (wigner_mode == 2) ? wigner_stride : 0;
  const int64_t dense_stride = (wigner_mode == 1) ? wigner_stride : 0;
  const size_t smem_bytes =
      (2 * kCuteTileMMax * kCuteTileK + 2 * kCuteTileNMax * kCuteTileK) * sizeof(float);

  persistent_grouped_forward_cute_tiled_kernel<<<grid, threads, smem_bytes, at::cuda::getCurrentCUDAStream()>>>(
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
#endif
}

at::Tensor persistent_grouped_forward_cutlass_native_fp32_cuda(
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
#ifndef DPTB_SO2_MOE_PERSISTENT_P1_CUTE
  TORCH_CHECK(false, "persistent_grouped_forward_cutlass_native_fp32 requires building with DPTB_SO2_MOE_PERSISTENT_P1_CUTLASS_ROOT or DPTB_CUTLASS_ROOT");
#else
  const c10::cuda::CUDAGuard device_guard(x.device());
  TORCH_CHECK(block_m == 64 && block_n == 32, "cutlass_native uses fixed 64x32 raw GEMM tiles");
  at::Tensor out = at::zeros({x.size(0), out_dim}, x.options());
  if (x.size(0) == 0 || out_dim == 0 || problem_tile_prefix.numel() == 0) {
    return out;
  }

  at::Tensor counter = at::zeros({1}, x.options().dtype(at::kLong));
  const int64_t total_tiles = problem_tile_prefix.narrow(0, problem_tile_prefix.numel() - 1, 1).item<int64_t>();
  if (total_tiles <= 0) {
    return out;
  }

  int dev = 0;
  cudaGetDevice(&dev);
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, dev);
  const int64_t sm_count = std::max(1, props.multiProcessorCount);
  int64_t blocks_i64 = active_blocks > 0 ? active_blocks : std::min<int64_t>(std::max<int64_t>(1, total_tiles), sm_count * 2);
  blocks_i64 = std::min<int64_t>(blocks_i64, 65535);
  const dim3 grid(static_cast<unsigned int>(blocks_i64));
  const dim3 threads(static_cast<unsigned int>(NativeBaseGemm::GemmKernel::kThreadCount));

  const int64_t radial_dim = radial_all.numel() > 0 ? radial_all.size(1) : 0;
  const bool has_radial = radial_all.numel() > 0;
  const int64_t compact_stride = (wigner_mode == 2) ? wigner_stride : 0;
  const int64_t dense_stride = (wigner_mode == 1) ? wigner_stride : 0;

  persistent_grouped_forward_cutlass_native_kernel<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<float>(),
      wigner.numel() > 0 ? wigner.data_ptr<float>() : nullptr,
      edge_order.data_ptr<int64_t>(),
      route_ptr.data_ptr<int64_t>(),
      problem_tile_prefix.data_ptr<int64_t>(),
      weight_flat.data_ptr<float>(),
      weight_offsets.data_ptr<int64_t>(),
      m_values.data_ptr<int64_t>(),
      in_ptr.data_ptr<int64_t>(),
      in_base.data_ptr<int64_t>(),
      in_l.data_ptr<int64_t>(),
      out_ptr.data_ptr<int64_t>(),
      out_base.data_ptr<int64_t>(),
      out_l.data_ptr<int64_t>(),
      offsets.data_ptr<int64_t>(),
      compact_offsets.numel() > 0 ? compact_offsets.data_ptr<int64_t>() : nullptr,
      radial_all.numel() > 0 ? radial_all.data_ptr<float>() : nullptr,
      m_in_index.numel() > 0 ? m_in_index.data_ptr<int64_t>() : nullptr,
      out.data_ptr<float>(),
      reinterpret_cast<unsigned long long*>(counter.data_ptr<int64_t>()),
      x.size(0),
      x.size(1),
      out_dim,
      n_routes,
      m_values.numel(),
      n_routes * m_values.numel(),
      total_tiles,
      radial_dim,
      dense_stride,
      compact_stride,
      wigner_mode,
      rotate_in,
      rotate_out,
      radial_on_input,
      has_radial);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
#endif
}
