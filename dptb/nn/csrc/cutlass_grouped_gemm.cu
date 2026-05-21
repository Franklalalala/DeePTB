#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace {

void check_cuda(cudaError_t status) {
  TORCH_CHECK(status == cudaSuccess, "CUDA error: ", cudaGetErrorString(status));
}

void check_cutlass(cutlass::Status status) {
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS error: ", static_cast<int>(status));
}

template <typename T>
torch::Tensor copy_vector_to_device(const std::vector<T>& host, const torch::Tensor& like, torch::Dtype dtype) {
  auto out = torch::empty({static_cast<int64_t>(host.size())}, like.options().dtype(dtype));
  if (!host.empty()) {
    check_cuda(cudaMemcpyAsync(
        out.data_ptr(),
        host.data(),
        host.size() * sizeof(T),
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()));
  }
  return out;
}

torch::Tensor copy_problem_sizes_to_device(
    const std::vector<cutlass::gemm::GemmCoord>& host,
    const torch::Tensor& like) {
  auto out = torch::empty({static_cast<int64_t>(host.size() * 3)}, like.options().dtype(torch::kInt32));
  if (!host.empty()) {
    check_cuda(cudaMemcpyAsync(
        out.data_ptr<int>(),
        host.data(),
        host.size() * sizeof(cutlass::gemm::GemmCoord),
        cudaMemcpyHostToDevice,
        at::cuda::getCurrentCUDAStream()));
  }
  return out;
}

template <typename LayoutA, typename LayoutB, typename LayoutC>
struct SimtGroupedGemm {
  using Element = float;
  using ElementAccumulator = float;
  using ElementOutput = float;
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
      Element,
      LayoutA,
      cutlass::ComplexTransform::kNone,
      1,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      1,
      ElementOutput,
      LayoutC,
      ElementAccumulator,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm80,
      cutlass::gemm::GemmShape<128, 128, 8>,
      cutlass::gemm::GemmShape<32, 64, 8>,
      cutlass::gemm::GemmShape<1, 1, 1>,
      cutlass::epilogue::thread::LinearCombination<ElementOutput, 1, ElementAccumulator, ElementAccumulator>,
      cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
      2,
      cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
  using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;
};

template <typename LayoutA, typename LayoutB, typename LayoutC>
void run_grouped_gemm(
    const std::vector<cutlass::gemm::GemmCoord>& host_problem_sizes,
    const std::vector<int64_t>& host_a_ptrs,
    const std::vector<int64_t>& host_b_ptrs,
    const std::vector<int64_t>& host_c_ptrs,
    const std::vector<int64_t>& host_d_ptrs,
    const std::vector<int64_t>& host_lda,
    const std::vector<int64_t>& host_ldb,
    const std::vector<int64_t>& host_ldc,
    const std::vector<int64_t>& host_ldd,
    const torch::Tensor& like) {
  using Gemm = typename SimtGroupedGemm<LayoutA, LayoutB, LayoutC>::Gemm;
  const int problem_count = static_cast<int>(host_problem_sizes.size());
  if (problem_count == 0) {
    return;
  }

  auto problem_sizes = copy_problem_sizes_to_device(host_problem_sizes, like);
  auto ptr_a = copy_vector_to_device<int64_t>(host_a_ptrs, like, torch::kInt64);
  auto ptr_b = copy_vector_to_device<int64_t>(host_b_ptrs, like, torch::kInt64);
  auto ptr_c = copy_vector_to_device<int64_t>(host_c_ptrs, like, torch::kInt64);
  auto ptr_d = copy_vector_to_device<int64_t>(host_d_ptrs, like, torch::kInt64);
  auto lda = copy_vector_to_device<int64_t>(host_lda, like, torch::kInt64);
  auto ldb = copy_vector_to_device<int64_t>(host_ldb, like, torch::kInt64);
  auto ldc = copy_vector_to_device<int64_t>(host_ldc, like, torch::kInt64);
  auto ldd = copy_vector_to_device<int64_t>(host_ldd, like, torch::kInt64);

  int threadblock_count = Gemm::sufficient(host_problem_sizes.data(), problem_count);
  TORCH_CHECK(threadblock_count > 0, "CUTLASS grouped GEMM found no runnable threadblocks");

  typename Gemm::EpilogueOutputOp::Params epilogue_op(1.0f, 0.0f);
  typename Gemm::Arguments args(
      reinterpret_cast<cutlass::gemm::GemmCoord*>(problem_sizes.data_ptr<int>()),
      problem_count,
      threadblock_count,
      epilogue_op,
      reinterpret_cast<float**>(ptr_a.data_ptr<int64_t>()),
      reinterpret_cast<float**>(ptr_b.data_ptr<int64_t>()),
      reinterpret_cast<float**>(ptr_c.data_ptr<int64_t>()),
      reinterpret_cast<float**>(ptr_d.data_ptr<int64_t>()),
      lda.data_ptr<int64_t>(),
      ldb.data_ptr<int64_t>(),
      ldc.data_ptr<int64_t>(),
      ldd.data_ptr<int64_t>(),
      const_cast<cutlass::gemm::GemmCoord*>(host_problem_sizes.data()));

  Gemm gemm;
  size_t workspace_size = gemm.get_workspace_size(args);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)}, like.options().dtype(torch::kUInt8));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  check_cutlass(gemm.initialize(args, workspace.data_ptr(), stream));
  check_cutlass(gemm.run(stream));
}

void check_common(torch::Tensor ptr) {
  TORCH_CHECK(!ptr.is_cuda(), "ptr must be CPU int64");
  TORCH_CHECK(ptr.scalar_type() == torch::kInt64, "ptr must be int64");
  TORCH_CHECK(ptr.is_contiguous(), "ptr must be contiguous");
}

}  // namespace

torch::Tensor grouped_gemm_forward_fp32(
    torch::Tensor x,
    torch::Tensor ptr,
    torch::Tensor weight) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be fp32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  check_common(ptr);

  const int64_t n_rows = x.size(0);
  const int64_t in_features = x.size(1);
  const int64_t groups = weight.size(0);
  const int64_t out_features = weight.size(1);
  TORCH_CHECK(weight.size(2) == in_features, "weight shape must be [G, O, I]");
  TORCH_CHECK(ptr.numel() == groups + 1, "ptr must have G + 1 entries");

  auto y = torch::empty({n_rows, out_features}, x.options());
  if (n_rows == 0 || groups == 0) {
    return y;
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t* ptr_data = ptr.data_ptr<int64_t>();
  const float* x_base = x.data_ptr<float>();
  const float* w_base = weight.data_ptr<float>();
  float* y_base = y.data_ptr<float>();

  std::vector<cutlass::gemm::GemmCoord> problems;
  std::vector<int64_t> a_ptrs, b_ptrs, c_ptrs, d_ptrs, lda, ldb, ldc, ldd;
  for (int64_t g = 0; g < groups; ++g) {
    const int64_t start = ptr_data[g];
    const int64_t end = ptr_data[g + 1];
    TORCH_CHECK(end >= start, "ptr must be non-decreasing");
    TORCH_CHECK(start >= 0 && end <= n_rows, "ptr out of range");
    const int64_t rows = end - start;
    if (rows == 0) {
      continue;
    }
    problems.emplace_back(static_cast<int>(rows), static_cast<int>(out_features), static_cast<int>(in_features));
    a_ptrs.push_back(reinterpret_cast<int64_t>(x_base + start * in_features));
    b_ptrs.push_back(reinterpret_cast<int64_t>(w_base + g * out_features * in_features));
    c_ptrs.push_back(reinterpret_cast<int64_t>(y_base + start * out_features));
    d_ptrs.push_back(reinterpret_cast<int64_t>(y_base + start * out_features));
    lda.push_back(in_features);
    ldb.push_back(in_features);
    ldc.push_back(out_features);
    ldd.push_back(out_features);
  }

  run_grouped_gemm<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(
      problems, a_ptrs, b_ptrs, c_ptrs, d_ptrs, lda, ldb, ldc, ldd, x);
  return y;
}

torch::Tensor grouped_gemm_grad_x_fp32(
    torch::Tensor grad_out,
    torch::Tensor ptr,
    torch::Tensor weight) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be fp32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be fp32");
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  check_common(ptr);

  const int64_t n_rows = grad_out.size(0);
  const int64_t groups = weight.size(0);
  const int64_t out_features = weight.size(1);
  const int64_t in_features = weight.size(2);
  TORCH_CHECK(grad_out.size(1) == out_features, "grad_out O must match weight");
  TORCH_CHECK(ptr.numel() == groups + 1, "ptr must have G + 1 entries");

  auto grad_x = torch::empty({n_rows, in_features}, grad_out.options());
  if (n_rows == 0 || groups == 0) {
    return grad_x;
  }

  c10::cuda::CUDAGuard device_guard(grad_out.device());
  const int64_t* ptr_data = ptr.data_ptr<int64_t>();
  const float* go_base = grad_out.data_ptr<float>();
  const float* w_base = weight.data_ptr<float>();
  float* gx_base = grad_x.data_ptr<float>();

  std::vector<cutlass::gemm::GemmCoord> problems;
  std::vector<int64_t> a_ptrs, b_ptrs, c_ptrs, d_ptrs, lda, ldb, ldc, ldd;
  for (int64_t g = 0; g < groups; ++g) {
    const int64_t start = ptr_data[g];
    const int64_t end = ptr_data[g + 1];
    const int64_t rows = end - start;
    if (rows == 0) {
      continue;
    }
    problems.emplace_back(static_cast<int>(rows), static_cast<int>(in_features), static_cast<int>(out_features));
    a_ptrs.push_back(reinterpret_cast<int64_t>(go_base + start * out_features));
    b_ptrs.push_back(reinterpret_cast<int64_t>(w_base + g * out_features * in_features));
    c_ptrs.push_back(reinterpret_cast<int64_t>(gx_base + start * in_features));
    d_ptrs.push_back(reinterpret_cast<int64_t>(gx_base + start * in_features));
    lda.push_back(out_features);
    ldb.push_back(in_features);
    ldc.push_back(in_features);
    ldd.push_back(in_features);
  }

  run_grouped_gemm<cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(
      problems, a_ptrs, b_ptrs, c_ptrs, d_ptrs, lda, ldb, ldc, ldd, grad_out);
  return grad_x;
}

torch::Tensor grouped_gemm_backward_weight_fp32(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor ptr,
    int64_t groups) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be fp32");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(grad_out.size(0) == x.size(0), "grad_out and x row counts differ");
  check_common(ptr);

  const int64_t n_rows = x.size(0);
  const int64_t out_features = grad_out.size(1);
  const int64_t in_features = x.size(1);
  TORCH_CHECK(ptr.numel() == groups + 1, "ptr must have G + 1 entries");

  auto grad_weight = torch::zeros({groups, out_features, in_features}, x.options());
  if (n_rows == 0 || groups == 0) {
    return grad_weight;
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  const int64_t* ptr_data = ptr.data_ptr<int64_t>();
  const float* x_base = x.data_ptr<float>();
  const float* go_base = grad_out.data_ptr<float>();
  float* gw_base = grad_weight.data_ptr<float>();

  std::vector<cutlass::gemm::GemmCoord> problems;
  std::vector<int64_t> a_ptrs, b_ptrs, c_ptrs, d_ptrs, lda, ldb, ldc, ldd;
  for (int64_t g = 0; g < groups; ++g) {
    const int64_t start = ptr_data[g];
    const int64_t end = ptr_data[g + 1];
    TORCH_CHECK(end >= start, "ptr must be non-decreasing");
    TORCH_CHECK(start >= 0 && end <= n_rows, "ptr out of range");
    const int64_t rows = end - start;
    if (rows == 0) {
      continue;
    }
    problems.emplace_back(static_cast<int>(out_features), static_cast<int>(in_features), static_cast<int>(rows));
    a_ptrs.push_back(reinterpret_cast<int64_t>(go_base + start * out_features));
    b_ptrs.push_back(reinterpret_cast<int64_t>(x_base + start * in_features));
    c_ptrs.push_back(reinterpret_cast<int64_t>(gw_base + g * out_features * in_features));
    d_ptrs.push_back(reinterpret_cast<int64_t>(gw_base + g * out_features * in_features));
    lda.push_back(out_features);
    ldb.push_back(in_features);
    ldc.push_back(in_features);
    ldd.push_back(in_features);
  }

  run_grouped_gemm<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(
      problems, a_ptrs, b_ptrs, c_ptrs, d_ptrs, lda, ldb, ldc, ldd, x);
  return grad_weight;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_gemm_forward_fp32", &grouped_gemm_forward_fp32, "CUTLASS grouped GEMM forward fp32");
  m.def("grouped_gemm_grad_x_fp32", &grouped_gemm_grad_x_fp32, "CUTLASS grouped GEMM grad-x fp32");
  m.def("grouped_gemm_backward_weight_fp32", &grouped_gemm_backward_weight_fp32, "CUTLASS grouped GEMM grad-weight fp32");
}
