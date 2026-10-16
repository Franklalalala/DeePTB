"""FP64 tensor-core (DMMA) kernel of the three-centre row sums, built on first use.

``pair_sum`` computes what ``assembly._pair_sum`` computes: for every output row r (block), out[row_r] += sum over the
row's terms t of X[q1_t]^T Z[q2_t] (X, Z [Q, NQ, W] FP64, W <= 32). Warp w of a thread block owns output rows
[8w, 8w + 8) and accumulates its 8 x W strip in ``mma.sync.m8n8k4.f64`` fragments over the terms and channels (four
channels per step). With enough rows to fill the device (``splits``) there is one thread block per row, which sums the
row in registers and writes it once. With fewer rows (small cells: hundreds of terms per row) the terms of a row are
split over NS thread blocks; each writes its partial block and a second kernel adds the NS partials to the output in
split order, so the result is deterministic. Only the summation order differs from the Triton kernel (rounding
level); NS = 1 is the single-pass kernel. The extension is compiled for the local GPU by ``torch.utils.cpp_extension`` (nvcc)
into ``DPTB_PRIOR_BUILD_DIR`` (default: the torch extensions directory); without a CUDA toolkit, or on a device below
sm_80, ``available()`` is False and the caller keeps the Triton kernel.
"""
import os

import torch

_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ void dmma(double& c0, double& c1, double a, double b) {
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%0, %1}, {%2}, {%3}, {%0, %1};\n"
               : "+d"(c0), "+d"(c1) : "d"(a), "d"(b));
}

template <int TN>
__global__ void __launch_bounds__(128) pair_sum_kernel(
    double* __restrict__ out, double* __restrict__ part, const double* __restrict__ X, const double* __restrict__ Z,
    const int64_t* __restrict__ q1, const int64_t* __restrict__ q2, const int64_t* __restrict__ rows,
    const int64_t* __restrict__ starts, const int64_t* __restrict__ counts, int NQ, int W, int64_t RS, int64_t ES,
    int NS, int64_t NR) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, k4 = lane & 3;
  const int64_t r = blockIdx.x / NS;
  const int s = blockIdx.x % NS;
  const int64_t row = rows[r], st = starts[r], n = counts[r];
  const int64_t chunk = (n + NS - 1) / NS;
  const int64_t t0 = st + s * chunk;
  const int64_t t1 = (t0 + chunk < st + n) ? t0 + chunk : st + n;
  const int am = warp * 8 + g;
  const bool am_ok = am < W;
  bool bn_ok[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) bn_ok[j] = j * 8 + g < W;
  double c[TN][2];
#pragma unroll
  for (int j = 0; j < TN; ++j) c[j][0] = c[j][1] = 0.0;
  const int64_t NW = (int64_t)NQ * W;
  for (int64_t t = t0; t < t1; ++t) {
    const double* x = X + q1[t] * NW + am;
    const double* z = Z + q2[t] * NW + g;
#pragma unroll 4
    for (int p0 = 0; p0 < NQ; p0 += 4) {
      const int p = p0 + k4;
      const bool pk = p < NQ;
      const double a = (pk && am_ok) ? __ldg(x + (int64_t)p * W) : 0.0;
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const double b = (pk && bn_ok[j]) ? __ldg(z + (int64_t)p * W + j * 8) : 0.0;
        dmma(c[j][0], c[j][1], a, b);
      }
    }
  }
#pragma unroll
  for (int j = 0; j < TN; ++j) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int b = j * 8 + 2 * k4 + h;
      if (am_ok && b < W) {
        if (NS == 1) {
          double* o = out + row * RS + ((int64_t)am * W + b) * ES;
          *o += c[j][h];
        } else {
          part[((int64_t)s * NR + r) * W * W + (int64_t)am * W + b] = c[j][h];
        }
      }
    }
  }
}

__global__ void pair_sum_reduce(double* __restrict__ out, const double* __restrict__ part,
                                const int64_t* __restrict__ rows, int64_t NR, int NS, int W, int64_t RS, int64_t ES) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t WW = (int64_t)W * W;
  if (i >= NR * WW) return;
  const int64_t r = i / WW, e = i % WW;
  double acc = part[r * WW + e];
  for (int s = 1; s < NS; ++s) acc += part[((int64_t)s * NR + r) * WW + e];
  out[rows[r] * RS + e * ES] += acc;
}

void pair_sum(torch::Tensor out, torch::Tensor X, torch::Tensor Z, torch::Tensor q1, torch::Tensor q2,
              torch::Tensor rows, torch::Tensor starts, torch::Tensor counts, int64_t NQ, int64_t W, int64_t RS,
              int64_t ES, int64_t NS, torch::Tensor part) {
  const int64_t nr = rows.numel();
  if (nr == 0) return;
  const int tiles = (int)((W + 7) / 8);
  dim3 block(32 * tiles);
  auto stream = at::cuda::getCurrentCUDAStream();
  double* pp = NS > 1 ? part.data_ptr<double>() : nullptr;
#define LAUNCH(TN) pair_sum_kernel<TN><<<nr * NS, block, 0, stream>>>(out.data_ptr<double>(), pp,     \
      X.data_ptr<double>(), Z.data_ptr<double>(), q1.data_ptr<int64_t>(), q2.data_ptr<int64_t>(),             \
      rows.data_ptr<int64_t>(), starts.data_ptr<int64_t>(), counts.data_ptr<int64_t>(), (int)NQ, (int)W, RS, ES, \
      (int)NS, nr)
  switch (tiles) {
    case 1: LAUNCH(1); break;
    case 2: LAUNCH(2); break;
    case 3: LAUNCH(3); break;
    default: LAUNCH(4); break;
  }
#undef LAUNCH
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (NS > 1) {
    const int64_t total = nr * W * W;
    pair_sum_reduce<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(out.data_ptr<double>(), pp,
        rows.data_ptr<int64_t>(), nr, (int)NS, (int)W, RS, ES);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
"""

_EXT = None
_FAILED = False


def _ext():
    global _EXT, _FAILED
    if _EXT is None and not _FAILED:
        try:
            from torch.utils.cpp_extension import load_inline
            major, minor = torch.cuda.get_device_capability()
            name = f'dptb_gf_dmma_split_sm{major}{minor}'
            kw = {}
            if os.environ.get('DPTB_PRIOR_BUILD_DIR'):     # one subdirectory per extension (sources are file-named)
                kw['build_directory'] = os.path.join(os.environ['DPTB_PRIOR_BUILD_DIR'], name)
                os.makedirs(kw['build_directory'], exist_ok=True)
            _EXT = load_inline(name=name, cpp_sources='',
                               cuda_sources=_SOURCE + '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) '
                                                      '{ m.def("pair_sum", &pair_sum); }',
                               extra_cuda_cflags=['-O3', f'-gencode=arch=compute_{major}{minor},code=sm_{major}{minor}'],
                               verbose=False, **kw)
        except Exception as e:          # noqa: BLE001 - no toolkit or compiler: keep the Triton kernel
            _FAILED = True
            _EXT = None
            print(f'dptb.prior.gf.dmma: extension unavailable ({type(e).__name__}: {str(e)[:200]})', flush=True)
    return _EXT


def available(device=None):
    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability(device)[0] < 8:
        return False
    return _ext() is not None


SPLIT_TARGET, SPLIT_CAP = 4096, 16      # thread blocks to aim for; most blocks per row


def splits(rows):
    """Thread blocks per row: 1 once the rows fill the device, else up to SPLIT_CAP."""
    return 1 if rows >= SPLIT_TARGET else max(1, min(SPLIT_CAP, SPLIT_TARGET // max(rows, 1)))


def pair_sum(out, X, Z, q1, q2, rows, starts, counts, NQ, W, RS, ES):
    """out[rows[r]] += sum_t X[q1_t]^T Z[q2_t] over terms t in [starts[r], starts[r] + counts[r]) (see module)."""
    ns = splits(len(rows))
    part = torch.empty((ns, len(rows), W, W) if ns > 1 else (1,), dtype=torch.float64, device=X.device)
    _ext().pair_sum(out, X, Z, q1, q2, rows, starts, counts, int(NQ), int(W), int(RS), int(ES), ns, part)
