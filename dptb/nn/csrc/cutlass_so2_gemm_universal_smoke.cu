#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal_with_visitor.h"
#include "cutlass/gemm/threadblock/mma_pipelined.h"
#include "cutlass/layout/matrix.h"

namespace {

void check_cutlass(cutlass::Status status) {
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GemmUniversal error: ", static_cast<int>(status));
}

using Element = float;
using ElementAccumulator = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    Element,
    1,
    ElementAccumulator,
    ElementAccumulator>;

using GemmUniversalFp32 = cutlass::gemm::device::GemmUniversal<
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    1,
    1>;

template <typename OutputTileIterator_, typename ThreadblockShape_>
class So2PairScatterCallbacks {
 public:
  using OutputTileIterator = OutputTileIterator_;
  using ThreadblockShape = ThreadblockShape_;
  using ElementAccumulator = float;
  using AccumulatorAccessType = cutlass::Array<ElementAccumulator, OutputTileIterator::kElementsPerAccess>;

  struct Arguments {
    float* out;
    float const* wigner;
    int out_dim;
    int cout;
    int raw_m;
    int raw_n;
    int raw_stride;
  };

  struct Params {
    float* out;
    float const* wigner;
    int out_dim;
    int cout;
    int raw_m;
    int raw_n;
    int raw_stride;
    typename OutputTileIterator::Params params_raw;
    float* ptr_raw;

    CUTLASS_HOST_DEVICE
    Params()
        : out(nullptr),
          wigner(nullptr),
          out_dim(0),
          cout(0),
          raw_m(0),
          raw_n(0),
          raw_stride(0),
          params_raw(LayoutC(0)),
          ptr_raw(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args)
        : out(args.out),
          wigner(args.wigner),
          out_dim(args.out_dim),
          cout(args.cout),
          raw_m(args.raw_m),
          raw_n(args.raw_n),
          raw_stride(args.raw_stride),
          params_raw(LayoutC(args.raw_stride)),
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
  So2PairScatterCallbacks(Params const& params, SharedStorage& /*shared_storage*/)
      : params_(params) {}

  class RuntimeCallbacks {
   private:
    Params params_;
    OutputTileIterator iterator_raw_;
    cutlass::MatrixCoord extent_;

   public:
    CUTLASS_DEVICE
    RuntimeCallbacks(
        Params const& params,
        OutputTileIterator iterator_raw,
        cutlass::MatrixCoord extent)
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

        const int edge = raw_row >> 1;
        const int pair_row = raw_row & 1;
        const int cout = params_.cout;
        if (cout <= 0 || raw_col >= 2 * cout) {
          continue;
        }
        const int channel = raw_col < cout ? raw_col : raw_col - cout;
        const bool imag_col = raw_col >= cout;

        int wigner_col = 2;
        float sign = 1.0f;
        if (pair_row == 0 && !imag_col) {
          wigner_col = 0;   // rr0 contributes to y0
        } else if (pair_row == 0 && imag_col) {
          wigner_col = 2;   // ii0 contributes to y1
        } else if (pair_row == 1 && !imag_col) {
          wigner_col = 2;   // rr1 contributes to y1
        } else {
          wigner_col = 0;   // ii1 subtracts from y0
          sign = -1.0f;
        }

        const float value = sign * static_cast<float>(accum[v]);
        float* out_edge = params_.out + static_cast<int64_t>(edge) * params_.out_dim + channel * 3;
        const float* w = params_.wigner + static_cast<int64_t>(edge) * 9 + wigner_col;
        atomicAdd(out_edge + 0, value * w[0]);
        atomicAdd(out_edge + 1, value * w[3]);
        atomicAdd(out_edge + 2, value * w[6]);
      }
    }

    CUTLASS_DEVICE void end_step(int /*step_idx*/) {
      ++iterator_raw_;
    }
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
    OutputTileIterator iterator_raw(
        params_.params_raw,
        params_.ptr_raw,
        extent,
        thread_idx,
        tb_offset);
    return RuntimeCallbacks(params_, iterator_raw, extent);
  }
};

using PairThreadblockShape = cutlass::gemm::GemmShape<64, 64, 8>;
using PairWarpShape = cutlass::gemm::GemmShape<32, 32, 8>;
using PairInstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

struct So2RawAParamsDevice {
  float const* wigner;
  int in_dim;
  int cin;
  int n_edges;
};

template <typename BaseIterator_>
class So2RawPairAIterator {
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

  class Params {
   public:
    float const* wigner;
    int in_dim;
    int cin;
    int n_edges;

    CUTLASS_HOST_DEVICE
    Params(Layout const& /*layout*/)
        : wigner(nullptr), in_dim(0), cin(0), n_edges(0) {}

    CUTLASS_HOST_DEVICE
    Params(Layout const& /*layout*/, float const* wigner_, int in_dim_, int cin_, int n_edges_)
        : wigner(wigner_), in_dim(in_dim_), cin(cin_), n_edges(n_edges_) {}

    CUTLASS_HOST_DEVICE
    Params() : wigner(nullptr), in_dim(0), cin(0), n_edges(0) {}
  };

 private:
  Pointer x_;
  float const* wigner_;
  int in_dim_;
  int cin_;
  int n_edges_;
  TensorCoord extent_;
  cutlass::layout::PitchLinearCoord thread_offset_;

 public:
  So2RawPairAIterator() = default;

  CUTLASS_HOST_DEVICE
  So2RawPairAIterator(
      Params const& params,
      Pointer pointer,
      TensorCoord extent,
      int thread_id,
      TensorCoord const& threadblock_offset,
      int const* indices = nullptr)
      : x_(pointer),
        wigner_(params.wigner),
        in_dim_(params.in_dim),
        cin_(params.cin),
        n_edges_(params.n_edges),
        extent_(extent),
        thread_offset_(threadblock_offset.column(), threadblock_offset.row()) {
    if (indices) {
      auto const* meta = reinterpret_cast<So2RawAParamsDevice const*>(indices);
      wigner_ = meta->wigner;
      in_dim_ = meta->in_dim;
      cin_ = meta->cin;
      n_edges_ = meta->n_edges;
    }
    thread_offset_ += ThreadMap::initial_offset(thread_id);
  }

  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex /*pointer_offset*/) {}

  CUTLASS_HOST_DEVICE
  So2RawPairAIterator& operator++() {
    thread_offset_ += cutlass::layout::PitchLinearCoord(Shape::kColumn, 0);
    return *this;
  }

  CUTLASS_HOST_DEVICE
  So2RawPairAIterator operator++(int) {
    So2RawPairAIterator self(*this);
    operator++();
    return self;
  }

  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const& tile_offset) {
    thread_offset_ += cutlass::layout::PitchLinearCoord(
        tile_offset.column() * Shape::kColumn,
        tile_offset.row() * Shape::kRow);
  }

  CUTLASS_HOST_DEVICE
  void clear_mask(bool /*enable*/ = true) {}
  CUTLASS_HOST_DEVICE
  void enable_mask() {}
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const& /*mask*/) {}
  CUTLASS_HOST_DEVICE
  void get_mask(Mask& /*mask*/) const {}

 private:
  CUTLASS_DEVICE
  float load_one(int matrix_row, int matrix_col) const {
    if (!wigner_ || matrix_row < 0 || matrix_col < 0 ||
        matrix_row >= extent_.row() || matrix_col >= extent_.column()) {
      return 0.0f;
    }
    const int edge = matrix_row >> 1;
    if (edge < 0 || edge >= n_edges_ || matrix_col >= cin_) {
      return 0.0f;
    }
    const int pair_row = matrix_row & 1;
    const int wigner_col = pair_row ? 2 : 0;
    const float* x_edge = x_ + static_cast<int64_t>(edge) * in_dim_ + matrix_col * 3;
    const float* w = wigner_ + static_cast<int64_t>(edge) * 9 + wigner_col;
    return x_edge[0] * w[0] + x_edge[1] * w[3] + x_edge[2] * w[6];
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

  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment& frag, Index /*pointer_offset*/) {
    load(frag);
  }

  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment& frag, LongIndex /*byte_offset*/) {
    load(frag);
  }
};

using PairBaseGemm = cutlass::gemm::device::GemmUniversal<
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    PairThreadblockShape,
    PairWarpShape,
    PairInstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    2,
    1,
    1>;
using PairCallbacks = So2PairScatterCallbacks<
    typename PairBaseGemm::GemmKernel::Epilogue::OutputTileIterator,
    PairThreadblockShape>;
using PairEpilogue = cutlass::epilogue::threadblock::EpilogueWithVisitorCallbacks<
    typename PairBaseGemm::GemmKernel::Epilogue,
    PairCallbacks,
    1>;
using PairKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
    typename PairBaseGemm::GemmKernel::Mma,
    PairEpilogue,
    typename PairBaseGemm::GemmKernel::ThreadblockSwizzle>;
using PairEpilogueGemm = cutlass::gemm::device::GemmUniversalAdapter<PairKernel>;
using PairRawAIterator = So2RawPairAIterator<typename PairBaseGemm::GemmKernel::Mma::IteratorA>;
using PairRawAMma = cutlass::gemm::threadblock::MmaPipelined<
    PairThreadblockShape,
    PairRawAIterator,
    typename PairBaseGemm::GemmKernel::Mma::SmemIteratorA,
    typename PairBaseGemm::GemmKernel::Mma::IteratorB,
    typename PairBaseGemm::GemmKernel::Mma::SmemIteratorB,
    typename PairBaseGemm::GemmKernel::Mma::ElementC,
    typename PairBaseGemm::GemmKernel::Mma::LayoutC,
    typename PairBaseGemm::GemmKernel::Mma::Policy>;
using PairRawAKernel = cutlass::gemm::kernel::GemmWithEpilogueVisitor<
    PairRawAMma,
    PairEpilogue,
    typename PairBaseGemm::GemmKernel::ThreadblockSwizzle>;
using PairRawAEpilogueGemm = cutlass::gemm::device::GemmUniversalAdapter<PairRawAKernel>;

union PairRawADirectSharedStorage {
  typename PairRawAMma::SharedStorage main_loop;
  typename PairEpilogue::SharedStorage epilogue;
};

__global__ void pair_raw_a_direct_kernel(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ wigner,
    float* __restrict__ out,
    int raw_m,
    int raw_n,
    int cin,
    int out_dim,
    int n_edges,
    int in_dim) {
  __shared__ PairRawADirectSharedStorage shared_storage;

  const int tile_m = static_cast<int>(blockIdx.x);
  const int tile_n = static_cast<int>(blockIdx.y);
  const int thread_idx = static_cast<int>(threadIdx.x);
  const int warp_idx = thread_idx / 32;
  const int lane_idx = thread_idx & 31;

  cutlass::MatrixCoord tb_offset_A(tile_m * PairThreadblockShape::kM, 0);
  cutlass::MatrixCoord tb_offset_B(0, tile_n * PairThreadblockShape::kN);

  typename PairRawAIterator::Params params_A{LayoutA(cin), wigner, in_dim, cin, n_edges};
  PairRawAIterator iterator_A(
      params_A,
      const_cast<float*>(x),
      cutlass::MatrixCoord(raw_m, cin),
      thread_idx,
      tb_offset_A,
      nullptr);

  typename PairBaseGemm::GemmKernel::Mma::IteratorB::Params params_B{LayoutB(cin)};
  typename PairBaseGemm::GemmKernel::Mma::IteratorB iterator_B(
      params_B,
      const_cast<float*>(weight),
      cutlass::MatrixCoord(cin, raw_n),
      thread_idx,
      tb_offset_B,
      nullptr);

  PairRawAMma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
  typename PairRawAMma::FragmentC accumulators;
  accumulators.clear();
  const int gemm_k_iterations = (cin + PairThreadblockShape::kK - 1) / PairThreadblockShape::kK;
  mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

  typename PairCallbacks::Arguments epilogue_args{
      out,
      wigner,
      out_dim,
      raw_n / 2,
      raw_m,
      raw_n,
      raw_n};
  typename PairCallbacks::Params callback_params(epilogue_args);
  PairEpilogue epilogue(callback_params, shared_storage.epilogue, thread_idx, warp_idx, lane_idx);
  epilogue(
      accumulators,
      cutlass::gemm::GemmCoord(tile_m, tile_n, 0),
      cutlass::gemm::GemmCoord(raw_m, raw_n, 1),
      thread_idx);
}

}  // namespace

torch::Tensor gemm_universal_smoke_fp32(torch::Tensor a, torch::Tensor b_row_major_nk) {
  TORCH_CHECK(a.is_cuda(), "a must be CUDA");
  TORCH_CHECK(b_row_major_nk.is_cuda(), "b must be CUDA");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be fp32");
  TORCH_CHECK(b_row_major_nk.scalar_type() == torch::kFloat32, "b must be fp32");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b_row_major_nk.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(a.dim() == 2, "a must be [M,K]");
  TORCH_CHECK(b_row_major_nk.dim() == 2, "b must be [N,K]");

  const int64_t m64 = a.size(0);
  const int64_t k64 = a.size(1);
  const int64_t n64 = b_row_major_nk.size(0);
  TORCH_CHECK(b_row_major_nk.size(1) == k64, "b K must match a K");
  TORCH_CHECK(m64 <= std::numeric_limits<int>::max(), "M too large for CUTLASS GemmCoord");
  TORCH_CHECK(n64 <= std::numeric_limits<int>::max(), "N too large for CUTLASS GemmCoord");
  TORCH_CHECK(k64 <= std::numeric_limits<int>::max(), "K too large for CUTLASS GemmCoord");

  auto out = torch::empty({m64, n64}, a.options());
  if (m64 == 0 || n64 == 0 || k64 == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(a.device());
  cutlass::gemm::GemmCoord problem_size(
      static_cast<int>(m64),
      static_cast<int>(n64),
      static_cast<int>(k64));

  typename GemmUniversalFp32::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      1,
      typename EpilogueOp::Params(1.0f, 0.0f),
      a.data_ptr<float>(),
      b_row_major_nk.data_ptr<float>(),
      out.data_ptr<float>(),
      out.data_ptr<float>(),
      m64 * k64,
      n64 * k64,
      m64 * n64,
      m64 * n64,
      k64,
      k64,
      n64,
      n64);

  GemmUniversalFp32 gemm;
  const size_t workspace_size = GemmUniversalFp32::get_workspace_size(args);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, a.options().dtype(torch::kUInt8));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  check_cutlass(gemm.can_implement(args));
  check_cutlass(gemm.initialize(args, workspace.data_ptr(), stream));
  check_cutlass(gemm.run(stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor gemm_universal_pair_epilogue_smoke_fp32(
    torch::Tensor pair,
    torch::Tensor weight,
    torch::Tensor wigner) {
  TORCH_CHECK(pair.is_cuda(), "pair must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(wigner.is_cuda(), "wigner must be CUDA");
  TORCH_CHECK(pair.scalar_type() == torch::kFloat32, "pair must be fp32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be fp32");
  TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  TORCH_CHECK(pair.is_contiguous(), "pair must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(wigner.is_contiguous(), "wigner must be contiguous");
  TORCH_CHECK(pair.dim() == 3 && pair.size(1) == 2, "pair must be [E,2,Cin]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) % 2 == 0, "weight must be [2*Cout,Cin]");
  TORCH_CHECK(weight.size(1) == pair.size(2), "weight Cin must match pair Cin");
  TORCH_CHECK(wigner.dim() == 3 && wigner.size(0) == pair.size(0) && wigner.size(1) == 3 && wigner.size(2) == 3,
              "wigner must be [E,3,3]");

  const int64_t n_edges = pair.size(0);
  const int64_t cin = pair.size(2);
  const int64_t raw_m = n_edges * 2;
  const int64_t raw_n = weight.size(0);
  const int64_t cout = raw_n / 2;
  TORCH_CHECK(raw_m <= std::numeric_limits<int>::max(), "raw M too large for CUTLASS");
  TORCH_CHECK(raw_n <= std::numeric_limits<int>::max(), "raw N too large for CUTLASS");
  TORCH_CHECK(cin <= std::numeric_limits<int>::max(), "raw K too large for CUTLASS");
  TORCH_CHECK(cout <= std::numeric_limits<int>::max(), "Cout too large for epilogue smoke");

  auto out = torch::zeros({n_edges, 3 * cout}, pair.options());
  if (n_edges == 0 || cin == 0 || cout == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(pair.device());
  cutlass::gemm::GemmCoord problem_size(
      static_cast<int>(raw_m),
      static_cast<int>(raw_n),
      static_cast<int>(cin));

  typename PairCallbacks::Arguments epilogue_args{
      out.data_ptr<float>(),
      wigner.data_ptr<float>(),
      static_cast<int>(3 * cout),
      static_cast<int>(cout),
      static_cast<int>(raw_m),
      static_cast<int>(raw_n),
      static_cast<int>(raw_n)};

  typename PairEpilogueGemm::Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_size,
      1,
      epilogue_args,
      pair.data_ptr<float>(),
      weight.data_ptr<float>(),
      out.data_ptr<float>(),
      out.data_ptr<float>(),
      raw_m * cin,
      raw_n * cin,
      raw_m * raw_n,
      raw_m * raw_n,
      cin,
      cin,
      raw_n,
      raw_n);

  PairEpilogueGemm gemm;
  const size_t workspace_size = PairEpilogueGemm::get_workspace_size(args);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, pair.options().dtype(torch::kUInt8));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  check_cutlass(PairEpilogueGemm::can_implement(args));
  check_cutlass(gemm.initialize(args, workspace.data_ptr(), stream));
  check_cutlass(gemm.run(stream));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

torch::Tensor gemm_universal_raw_a_loader_pair_epilogue_smoke_fp32(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor wigner) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(wigner.is_cuda(), "wigner must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be fp32");
  TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(wigner.is_contiguous(), "wigner must be contiguous");
  TORCH_CHECK(x.dim() == 2 && x.size(1) % 3 == 0, "x must be [E,3*Cin]");
  TORCH_CHECK(weight.dim() == 2 && weight.size(0) % 2 == 0, "weight must be [2*Cout,Cin]");
  TORCH_CHECK(weight.size(1) == x.size(1) / 3, "weight Cin must match x Cin");
  TORCH_CHECK(wigner.dim() == 3 && wigner.size(0) == x.size(0) && wigner.size(1) == 3 && wigner.size(2) == 3,
              "wigner must be [E,3,3]");

  const int64_t n_edges = x.size(0);
  const int64_t cin = x.size(1) / 3;
  const int64_t raw_m = n_edges * 2;
  const int64_t raw_n = weight.size(0);
  const int64_t cout = raw_n / 2;
  TORCH_CHECK(cin % PairThreadblockShape::kK == 0,
              "raw-A smoke currently expects Cin to be a multiple of the CUTLASS K tile");
  TORCH_CHECK(raw_m <= std::numeric_limits<int>::max(), "raw M too large for CUTLASS");
  TORCH_CHECK(raw_n <= std::numeric_limits<int>::max(), "raw N too large for CUTLASS");
  TORCH_CHECK(cin <= std::numeric_limits<int>::max(), "raw K too large for CUTLASS");
  TORCH_CHECK(cout <= std::numeric_limits<int>::max(), "Cout too large for epilogue smoke");
  TORCH_CHECK(x.size(1) <= std::numeric_limits<int>::max(), "input dim too large for raw-A metadata");

  auto out = torch::zeros({n_edges, 3 * cout}, x.options());
  if (n_edges == 0 || cin == 0 || cout == 0) {
    return out;
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(
      static_cast<unsigned int>((raw_m + PairThreadblockShape::kM - 1) / PairThreadblockShape::kM),
      static_cast<unsigned int>((raw_n + PairThreadblockShape::kN - 1) / PairThreadblockShape::kN));
  const dim3 threads(static_cast<unsigned int>(PairBaseGemm::GemmKernel::kThreadCount));
  pair_raw_a_direct_kernel<<<grid, threads, 0, stream>>>(
      x.data_ptr<float>(),
      weight.data_ptr<float>(),
      wigner.data_ptr<float>(),
      out.data_ptr<float>(),
      static_cast<int>(raw_m),
      static_cast<int>(raw_n),
      static_cast<int>(cin),
      static_cast<int>(3 * cout),
      static_cast<int>(n_edges),
      static_cast<int>(x.size(1)));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_universal_smoke_fp32", &gemm_universal_smoke_fp32, "CUTLASS GemmUniversal fp32 smoke");
  m.def(
      "gemm_universal_pair_epilogue_smoke_fp32",
      &gemm_universal_pair_epilogue_smoke_fp32,
      "CUTLASS GemmUniversal fp32 smoke with SO2 pair epilogue scatter");
  m.def(
      "gemm_universal_raw_a_loader_pair_epilogue_smoke_fp32",
      &gemm_universal_raw_a_loader_pair_epilogue_smoke_fp32,
      "CUTLASS GemmUniversal fp32 smoke with SO2 raw-A loader and pair epilogue scatter");
}
