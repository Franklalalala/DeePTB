// Fused crystal Reynolds projection for packed real AO-product rows.
//
//   out[r] = 1/|G_s| * sum_{g < |G_s|} R_{s,g}( x[ src_g(r) ], transposed if flagged )
//
// One CTA owns a tile of up to kTileRows output rows of one structure, loops over the structure's own group
// operations (no padded operations), and keeps the running sum in registers.  The per-block rotation
// A X B^T (A = D_la, B = D_lb) is evaluated separably through shared memory, one output element per thread.
// A reversed edge is rotated after reading its source row transposed (out(a,b) = D_la X(b,a)^T D_lb^T), which
// equals the reference "rotate, then permute the result" without a second pass.  Owner-computes with an inverse
// gather: no atomics, the summation order is fixed, so the result is deterministic.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdint>

namespace {

constexpr int kThreads = 256;
constexpr int kTileRows = 4;
constexpr int kDSize = 84;            // D_0 (1) + D_1 (9) + D_2 (25) + D_3 (49), each row-major
constexpr int kInfoStride = 8;        // int64 per structure: inv, rev, dmat, order, rows, row_off, tile_off, unused

__host__ __device__ constexpr int d_offset(int l) { return l == 0 ? 0 : (l == 1 ? 1 : (l == 2 ? 10 : 35)); }

template <typename T, int N>
__device__ __forceinline__ T dot_strided(const T* __restrict__ a, const T* __restrict__ x, int stride) {
    T s = T(0);
#pragma unroll
    for (int j = 0; j < N; ++j) s = fma(a[j], x[j * stride], s);
    return s;
}

template <typename T, int N>
__device__ __forceinline__ T dot_contig(const T* __restrict__ a, const T* __restrict__ b) {
    T s = T(0);
#pragma unroll
    for (int j = 0; j < N; ++j) s = fma(a[j], b[j], s);
    return s;
}

// info layout per structure: see kInfoStride.  pos_info packs (la | lb << 2 | i << 4 | m << 7 | base << 10).
template <typename T, int EPT>
__global__ void __launch_bounds__(kThreads, 4)
reynolds_kernel(const T* __restrict__ x, T* __restrict__ out, const int64_t* __restrict__ info, int n_struct,
                const int32_t* __restrict__ row_list, const int32_t* __restrict__ pos_info,
                const int32_t* __restrict__ transpose, int width) {
    extern __shared__ unsigned char smem_raw[];
    T* sh_x = reinterpret_cast<T*>(smem_raw);      // [kTileRows][width] gathered (maybe transposed) source rows
    T* sh_u = sh_x + kTileRows * width;            // [kTileRows][width] left-rotated blocks
    T* sh_d = sh_u + kTileRows * width;            // [kDSize] rotations of the current operation
    __shared__ int sh_src[kTileRows];
    __shared__ int sh_rev[kTileRows];

    const int tid = threadIdx.x;
    const int tile = blockIdx.x;
    int lo = 0, hi = n_struct - 1;
    while (lo < hi) {                              // last structure whose first tile is <= tile
        const int mid = (lo + hi + 1) >> 1;
        if (info[mid * kInfoStride + 6] <= tile) lo = mid; else hi = mid - 1;
    }
    const int64_t* si = info + static_cast<int64_t>(lo) * kInfoStride;
    const int32_t* inv = reinterpret_cast<const int32_t*>(si[0]);
    const uint8_t* rev = reinterpret_cast<const uint8_t*>(si[1]);
    const T* dmat = reinterpret_cast<const T*>(si[2]);
    const int order = static_cast<int>(si[3]);
    const int rows = static_cast<int>(si[4]);
    const int row_off = static_cast<int>(si[5]);
    const int t0 = (tile - static_cast<int>(si[6])) * kTileRows;
    const int nvalid = min(kTileRows, rows - t0);

    int pinfo[EPT], tpos[EPT];
    T acc[kTileRows][EPT];
#pragma unroll
    for (int k = 0; k < EPT; ++k) {
        const int p = tid + k * kThreads;
        pinfo[k] = p < width ? pos_info[p] : 0;
        tpos[k] = p < width ? transpose[p] : 0;
#pragma unroll
        for (int rr = 0; rr < kTileRows; ++rr) acc[rr][k] = T(0);
    }

    for (int g = 0; g < order; ++g) {
        if (tid < kTileRows) {
            int src = -1, rv = 0;
            if (tid < nvalid) {
                const int64_t at = static_cast<int64_t>(g) * rows + t0 + tid;
                src = row_list[row_off + inv[at]];
                rv = rev != nullptr ? static_cast<int>(rev[at]) : 0;
            }
            sh_src[tid] = src;
            sh_rev[tid] = rv;
        } else if (tid < kTileRows + kDSize) {
            sh_d[tid - kTileRows] = dmat[static_cast<int64_t>(g) * kDSize + (tid - kTileRows)];
        }
        __syncthreads();
#pragma unroll
        for (int rr = 0; rr < kTileRows; ++rr) {
            if (rr >= nvalid) break;
            const T* xr = x + static_cast<int64_t>(sh_src[rr]) * width;
            const bool flip = sh_rev[rr] != 0;
#pragma unroll
            for (int k = 0; k < EPT; ++k) {
                const int p = tid + k * kThreads;
                if (p < width) sh_x[rr * width + p] = xr[flip ? tpos[k] : p];
            }
        }
        __syncthreads();
#pragma unroll
        for (int rr = 0; rr < kTileRows; ++rr) {
            if (rr >= nvalid) break;
            const T* xs = sh_x + rr * width;
#pragma unroll
            for (int k = 0; k < EPT; ++k) {
                const int p = tid + k * kThreads;
                if (p < width) {
                    const int pi = pinfo[k];
                    const int la = pi & 3, lb = (pi >> 2) & 3, i = (pi >> 4) & 7, m = (pi >> 7) & 7, base = pi >> 10;
                    const int da = 2 * la + 1, db = 2 * lb + 1;
                    const T* a = sh_d + d_offset(la) + i * da;
                    const T* col = xs + base + m;
                    T u;
                    switch (la) {
                        case 0: u = dot_strided<T, 1>(a, col, db); break;
                        case 1: u = dot_strided<T, 3>(a, col, db); break;
                        case 2: u = dot_strided<T, 5>(a, col, db); break;
                        default: u = dot_strided<T, 7>(a, col, db); break;
                    }
                    sh_u[rr * width + p] = u;
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int rr = 0; rr < kTileRows; ++rr) {
            if (rr >= nvalid) break;
            const T* us = sh_u + rr * width;
#pragma unroll
            for (int k = 0; k < EPT; ++k) {
                const int p = tid + k * kThreads;
                if (p < width) {
                    const int pi = pinfo[k];
                    const int lb = (pi >> 2) & 3, i = (pi >> 4) & 7, m = (pi >> 7) & 7, base = pi >> 10;
                    const int db = 2 * lb + 1;
                    const T* urow = us + base + i * db;
                    const T* b = sh_d + d_offset(lb) + m * db;
                    T s;
                    switch (lb) {
                        case 0: s = dot_contig<T, 1>(urow, b); break;
                        case 1: s = dot_contig<T, 3>(urow, b); break;
                        case 2: s = dot_contig<T, 5>(urow, b); break;
                        default: s = dot_contig<T, 7>(urow, b); break;
                    }
                    acc[rr][k] += s;
                }
            }
        }
        __syncthreads();
    }

    const T weight = T(1) / static_cast<T>(order);
#pragma unroll
    for (int rr = 0; rr < kTileRows; ++rr) {
        if (rr >= nvalid) break;
        T* orow = out + static_cast<int64_t>(row_list[row_off + t0 + rr]) * width;
#pragma unroll
        for (int k = 0; k < EPT; ++k) {
            const int p = tid + k * kThreads;
            if (p < width) orow[p] = acc[rr][k] * weight;
        }
    }
}

template <typename T, int EPT>
void launch(const torch::Tensor& x, torch::Tensor& out, const torch::Tensor& info, const torch::Tensor& row_list,
            const torch::Tensor& pos_info, const torch::Tensor& transpose, int64_t n_tiles) {
    const int width = static_cast<int>(x.size(1));
    const size_t smem = (2 * static_cast<size_t>(kTileRows) * width + kDSize) * sizeof(T);
    auto kernel = reynolds_kernel<T, EPT>;
    if (smem > 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)));
    }
    kernel<<<static_cast<unsigned int>(n_tiles), kThreads, smem, at::cuda::getCurrentCUDAStream()>>>(
        x.data_ptr<T>(), out.data_ptr<T>(), info.data_ptr<int64_t>(), static_cast<int>(info.size(0)),
        row_list.data_ptr<int32_t>(), pos_info.data_ptr<int32_t>(), transpose.data_ptr<int32_t>(), width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void dispatch_width(const torch::Tensor& x, torch::Tensor& out, const torch::Tensor& info, const torch::Tensor& row_list,
                    const torch::Tensor& pos_info, const torch::Tensor& transpose, int64_t n_tiles) {
    const int64_t width = x.size(1);
    if (width <= kThreads) launch<T, 1>(x, out, info, row_list, pos_info, transpose, n_tiles);
    else if (width <= 2 * kThreads) launch<T, 2>(x, out, info, row_list, pos_info, transpose, n_tiles);
    else if (width <= 3 * kThreads) launch<T, 3>(x, out, info, row_list, pos_info, transpose, n_tiles);
    else if (width <= 4 * kThreads) launch<T, 4>(x, out, info, row_list, pos_info, transpose, n_tiles);
    else TORCH_CHECK(false, "sym_projection_fused: feature width ", width, " exceeds ", 4 * kThreads);
}

}  // namespace

torch::Tensor reynolds_apply(torch::Tensor x, torch::Tensor info, torch::Tensor row_list, torch::Tensor pos_info,
                             torch::Tensor transpose, int64_t n_tiles) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.is_contiguous(), "x must be a contiguous CUDA matrix");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble, "x must be float32 or float64");
    TORCH_CHECK(info.is_cuda() && info.scalar_type() == at::kLong && info.dim() == 2 && info.size(1) == kInfoStride,
                "info must be a CUDA int64 [S, 8] table");
    TORCH_CHECK(row_list.is_cuda() && row_list.scalar_type() == at::kInt, "row_list must be CUDA int32");
    TORCH_CHECK(pos_info.is_cuda() && pos_info.scalar_type() == at::kInt && pos_info.numel() == x.size(1),
                "pos_info must be CUDA int32 [width]");
    TORCH_CHECK(transpose.is_cuda() && transpose.scalar_type() == at::kInt && transpose.numel() == x.size(1),
                "transpose must be CUDA int32 [width]");
    c10::cuda::CUDAGuard guard(x.device());
    auto out = torch::empty_like(x);
    if (x.size(0) == 0 || n_tiles == 0) return out.zero_();
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "sym_projection_fused", [&] {
        dispatch_width<scalar_t>(x, out, info, row_list, pos_info, transpose, n_tiles);
    });
    return out;
}

int64_t tile_rows() { return kTileRows; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("reynolds_apply", &reynolds_apply, "Fused gather-rotate-sum crystal projection (forward == adjoint)");
    m.def("tile_rows", &tile_rows, "Output rows per CTA tile");
}
