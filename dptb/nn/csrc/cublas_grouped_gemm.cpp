#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <vector>

static void check_cublas(cublasStatus_t status) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cuBLAS error: ", static_cast<int>(status));
}

static void check_cuda(cudaError_t status) {
  TORCH_CHECK(status == cudaSuccess, "CUDA error: ", cudaGetErrorString(status));
}

static torch::Tensor copy_pointer_array_to_device(
    const std::vector<int64_t>& host_ptrs,
    const torch::Tensor& like) {
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(like.device());
  auto out = torch::empty({static_cast<int64_t>(host_ptrs.size())}, options);
  if (!host_ptrs.empty()) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    check_cuda(cudaMemcpyAsync(
        out.data_ptr<int64_t>(),
        host_ptrs.data(),
        host_ptrs.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice,
        stream));
  }
  return out;
}

static void configure_math(cublasHandle_t handle, bool fast_tf32) {
  check_cublas(cublasSetStream(handle, at::cuda::getCurrentCUDAStream()));
  check_cublas(cublasSetMathMode(
      handle, fast_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH));
}

torch::Tensor grouped_gemm_forward_fp32(
    torch::Tensor x,
    torch::Tensor ptr,
    torch::Tensor weight,
    bool fast_tf32) {
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(!ptr.is_cuda(), "ptr must be CPU int64");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be fp32");
  TORCH_CHECK(ptr.scalar_type() == torch::kInt64, "ptr must be int64");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

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
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  configure_math(handle, fast_tf32);

  const int64_t* ptr_data = ptr.data_ptr<int64_t>();
  const float* x_base = x.data_ptr<float>();
  const float* w_base = weight.data_ptr<float>();
  float* y_base = y.data_ptr<float>();
  const int in_i = static_cast<int>(in_features);
  const int out_i = static_cast<int>(out_features);

  std::vector<cublasOperation_t> transa;
  std::vector<cublasOperation_t> transb;
  std::vector<int> m, n, k, lda, ldb, ldc, group_size;
  std::vector<int64_t> a_array, b_array, c_array;
  std::vector<float> alpha, beta;

  for (int64_t g = 0; g < groups; ++g) {
    const int64_t start = ptr_data[g];
    const int64_t end = ptr_data[g + 1];
    TORCH_CHECK(end >= start, "ptr must be non-decreasing");
    TORCH_CHECK(start >= 0 && end <= n_rows, "ptr out of range");
    const int64_t rows = end - start;
    if (rows == 0) {
      continue;
    }
    transa.push_back(CUBLAS_OP_T);
    transb.push_back(CUBLAS_OP_N);
    m.push_back(out_i);
    n.push_back(static_cast<int>(rows));
    k.push_back(in_i);
    lda.push_back(in_i);
    ldb.push_back(in_i);
    ldc.push_back(out_i);
    group_size.push_back(1);
    alpha.push_back(1.0f);
    beta.push_back(0.0f);
    a_array.push_back(reinterpret_cast<int64_t>(w_base + g * out_features * in_features));
    b_array.push_back(reinterpret_cast<int64_t>(x_base + start * in_features));
    c_array.push_back(reinterpret_cast<int64_t>(y_base + start * out_features));
  }

  const int active_groups = static_cast<int>(group_size.size());
  if (active_groups == 0) {
    return y;
  }

  auto a_dev = copy_pointer_array_to_device(a_array, x);
  auto b_dev = copy_pointer_array_to_device(b_array, x);
  auto c_dev = copy_pointer_array_to_device(c_array, x);
  const cublasComputeType_t compute_type =
      fast_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  check_cublas(cublasGemmGroupedBatchedEx(
      handle,
      transa.data(),
      transb.data(),
      m.data(),
      n.data(),
      k.data(),
      static_cast<const void*>(alpha.data()),
      reinterpret_cast<const void* const*>(a_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      lda.data(),
      reinterpret_cast<const void* const*>(b_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      ldb.data(),
      static_cast<const void*>(beta.data()),
      reinterpret_cast<void* const*>(c_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      ldc.data(),
      active_groups,
      group_size.data(),
      compute_type));

  return y;
}

torch::Tensor grouped_gemm_backward_weight_fp32(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor ptr,
    int64_t groups,
    bool fast_tf32) {
  TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
  TORCH_CHECK(x.is_cuda(), "x must be CUDA");
  TORCH_CHECK(!ptr.is_cuda(), "ptr must be CPU int64");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be fp32");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(ptr.scalar_type() == torch::kInt64, "ptr must be int64");
  TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(grad_out.size(0) == x.size(0), "grad_out and x row counts differ");
  TORCH_CHECK(ptr.numel() == groups + 1, "ptr must have G + 1 entries");

  const int64_t n_rows = x.size(0);
  const int64_t out_features = grad_out.size(1);
  const int64_t in_features = x.size(1);
  auto grad_weight = torch::zeros({groups, out_features, in_features}, x.options());
  if (n_rows == 0 || groups == 0) {
    return grad_weight;
  }

  c10::cuda::CUDAGuard device_guard(x.device());
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  configure_math(handle, fast_tf32);

  const int64_t* ptr_data = ptr.data_ptr<int64_t>();
  const float* x_base = x.data_ptr<float>();
  const float* go_base = grad_out.data_ptr<float>();
  float* gw_base = grad_weight.data_ptr<float>();
  const int in_i = static_cast<int>(in_features);
  const int out_i = static_cast<int>(out_features);

  std::vector<cublasOperation_t> transa;
  std::vector<cublasOperation_t> transb;
  std::vector<int> m, n, k, lda, ldb, ldc, group_size;
  std::vector<int64_t> a_array, b_array, c_array;
  std::vector<float> alpha, beta;

  for (int64_t g = 0; g < groups; ++g) {
    const int64_t start = ptr_data[g];
    const int64_t end = ptr_data[g + 1];
    TORCH_CHECK(end >= start, "ptr must be non-decreasing");
    TORCH_CHECK(start >= 0 && end <= n_rows, "ptr out of range");
    const int64_t rows = end - start;
    if (rows == 0) {
      continue;
    }
    transa.push_back(CUBLAS_OP_N);
    transb.push_back(CUBLAS_OP_T);
    m.push_back(in_i);
    n.push_back(out_i);
    k.push_back(static_cast<int>(rows));
    lda.push_back(in_i);
    ldb.push_back(out_i);
    ldc.push_back(in_i);
    group_size.push_back(1);
    alpha.push_back(1.0f);
    beta.push_back(0.0f);
    a_array.push_back(reinterpret_cast<int64_t>(x_base + start * in_features));
    b_array.push_back(reinterpret_cast<int64_t>(go_base + start * out_features));
    c_array.push_back(reinterpret_cast<int64_t>(gw_base + g * out_features * in_features));
  }

  const int active_groups = static_cast<int>(group_size.size());
  if (active_groups == 0) {
    return grad_weight;
  }

  auto a_dev = copy_pointer_array_to_device(a_array, x);
  auto b_dev = copy_pointer_array_to_device(b_array, x);
  auto c_dev = copy_pointer_array_to_device(c_array, x);
  const cublasComputeType_t compute_type =
      fast_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  check_cublas(cublasGemmGroupedBatchedEx(
      handle,
      transa.data(),
      transb.data(),
      m.data(),
      n.data(),
      k.data(),
      static_cast<const void*>(alpha.data()),
      reinterpret_cast<const void* const*>(a_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      lda.data(),
      reinterpret_cast<const void* const*>(b_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      ldb.data(),
      static_cast<const void*>(beta.data()),
      reinterpret_cast<void* const*>(c_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      ldc.data(),
      active_groups,
      group_size.data(),
      compute_type));

  return grad_weight;
}

std::vector<torch::Tensor> grouped_gemm_multi_forward_fp32(
    std::vector<torch::Tensor> xs,
    std::vector<torch::Tensor> ptrs,
    std::vector<torch::Tensor> weights,
    bool fast_tf32) {
  const int64_t problems = static_cast<int64_t>(xs.size());
  TORCH_CHECK(problems == static_cast<int64_t>(ptrs.size()), "xs and ptrs must have the same length");
  TORCH_CHECK(problems == static_cast<int64_t>(weights.size()), "xs and weights must have the same length");
  TORCH_CHECK(problems > 0, "at least one GEMM problem is required");

  std::vector<torch::Tensor> outputs;
  outputs.reserve(problems);
  int64_t active_total = 0;
  for (int64_t p = 0; p < problems; ++p) {
    auto x = xs[p];
    auto ptr = ptrs[p];
    auto weight = weights[p];
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(!ptr.is_cuda(), "ptr must be CPU int64");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "weight must be fp32");
    TORCH_CHECK(ptr.scalar_type() == torch::kInt64, "ptr must be int64");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(x.device() == xs[0].device(), "all x tensors must be on the same device");
    TORCH_CHECK(weight.device() == xs[0].device(), "all weight tensors must be on the x device");
    TORCH_CHECK(weight.size(2) == x.size(1), "weight shape must be [G, O, I]");
    TORCH_CHECK(ptr.numel() == weight.size(0) + 1, "ptr must have G + 1 entries");

    const int64_t* ptr_data = ptr.data_ptr<int64_t>();
    const int64_t groups = weight.size(0);
    for (int64_t g = 0; g < groups; ++g) {
      const int64_t start = ptr_data[g];
      const int64_t end = ptr_data[g + 1];
      TORCH_CHECK(end >= start, "ptr must be non-decreasing");
      TORCH_CHECK(start >= 0 && end <= x.size(0), "ptr out of range");
      if (end > start) {
        active_total += 1;
      }
    }
    outputs.push_back(torch::empty({x.size(0), weight.size(1)}, x.options()));
  }
  if (active_total == 0) {
    return outputs;
  }

  c10::cuda::CUDAGuard device_guard(xs[0].device());
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  configure_math(handle, fast_tf32);

  std::vector<cublasOperation_t> transa;
  std::vector<cublasOperation_t> transb;
  std::vector<int> m, n, k, lda, ldb, ldc, group_size;
  std::vector<int64_t> a_array, b_array, c_array;
  std::vector<float> alpha, beta;
  transa.reserve(active_total);
  transb.reserve(active_total);
  m.reserve(active_total);
  n.reserve(active_total);
  k.reserve(active_total);
  lda.reserve(active_total);
  ldb.reserve(active_total);
  ldc.reserve(active_total);
  group_size.reserve(active_total);
  a_array.reserve(active_total);
  b_array.reserve(active_total);
  c_array.reserve(active_total);
  alpha.reserve(active_total);
  beta.reserve(active_total);

  for (int64_t p = 0; p < problems; ++p) {
    const auto x = xs[p];
    const auto ptr = ptrs[p];
    const auto weight = weights[p];
    const int64_t* ptr_data = ptr.data_ptr<int64_t>();
    const float* x_base = x.data_ptr<float>();
    const float* w_base = weight.data_ptr<float>();
    float* y_base = outputs[p].data_ptr<float>();
    const int in_i = static_cast<int>(x.size(1));
    const int out_i = static_cast<int>(weight.size(1));
    const int64_t groups = weight.size(0);

    for (int64_t g = 0; g < groups; ++g) {
      const int64_t start = ptr_data[g];
      const int64_t end = ptr_data[g + 1];
      const int64_t rows = end - start;
      if (rows == 0) {
        continue;
      }
      transa.push_back(CUBLAS_OP_T);
      transb.push_back(CUBLAS_OP_N);
      m.push_back(out_i);
      n.push_back(static_cast<int>(rows));
      k.push_back(in_i);
      lda.push_back(in_i);
      ldb.push_back(in_i);
      ldc.push_back(out_i);
      group_size.push_back(1);
      alpha.push_back(1.0f);
      beta.push_back(0.0f);
      a_array.push_back(reinterpret_cast<int64_t>(w_base + g * weight.size(1) * weight.size(2)));
      b_array.push_back(reinterpret_cast<int64_t>(x_base + start * x.size(1)));
      c_array.push_back(reinterpret_cast<int64_t>(y_base + start * weight.size(1)));
    }
  }

  auto a_dev = copy_pointer_array_to_device(a_array, xs[0]);
  auto b_dev = copy_pointer_array_to_device(b_array, xs[0]);
  auto c_dev = copy_pointer_array_to_device(c_array, xs[0]);
  const cublasComputeType_t compute_type =
      fast_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  check_cublas(cublasGemmGroupedBatchedEx(
      handle,
      transa.data(),
      transb.data(),
      m.data(),
      n.data(),
      k.data(),
      static_cast<const void*>(alpha.data()),
      reinterpret_cast<const void* const*>(a_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      lda.data(),
      reinterpret_cast<const void* const*>(b_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      ldb.data(),
      static_cast<const void*>(beta.data()),
      reinterpret_cast<void* const*>(c_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      ldc.data(),
      static_cast<int>(group_size.size()),
      group_size.data(),
      compute_type));

  return outputs;
}

std::vector<torch::Tensor> grouped_gemm_multi_backward_weight_fp32(
    std::vector<torch::Tensor> grad_outs,
    std::vector<torch::Tensor> xs,
    std::vector<torch::Tensor> ptrs,
    bool fast_tf32) {
  const int64_t problems = static_cast<int64_t>(xs.size());
  TORCH_CHECK(problems == static_cast<int64_t>(ptrs.size()), "xs and ptrs must have the same length");
  TORCH_CHECK(problems == static_cast<int64_t>(grad_outs.size()), "xs and grad_outs must have the same length");
  TORCH_CHECK(problems > 0, "at least one GEMM problem is required");

  std::vector<torch::Tensor> grad_weights;
  grad_weights.reserve(problems);
  int64_t active_total = 0;
  for (int64_t p = 0; p < problems; ++p) {
    auto x = xs[p];
    auto grad_out = grad_outs[p];
    auto ptr = ptrs[p];
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    TORCH_CHECK(!ptr.is_cuda(), "ptr must be CPU int64");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
    TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be fp32");
    TORCH_CHECK(ptr.scalar_type() == torch::kInt64, "ptr must be int64");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(grad_out.is_contiguous(), "grad_out must be contiguous");
    TORCH_CHECK(x.size(0) == grad_out.size(0), "x and grad_out row counts differ");
    TORCH_CHECK(x.device() == xs[0].device(), "all x tensors must be on the same device");
    TORCH_CHECK(grad_out.device() == xs[0].device(), "all grad_out tensors must be on the x device");

    const int64_t groups = ptr.numel() - 1;
    const int64_t* ptr_data = ptr.data_ptr<int64_t>();
    for (int64_t g = 0; g < groups; ++g) {
      const int64_t start = ptr_data[g];
      const int64_t end = ptr_data[g + 1];
      TORCH_CHECK(end >= start, "ptr must be non-decreasing");
      TORCH_CHECK(start >= 0 && end <= x.size(0), "ptr out of range");
      if (end > start) {
        active_total += 1;
      }
    }
    grad_weights.push_back(torch::zeros({groups, grad_out.size(1), x.size(1)}, x.options()));
  }
  if (active_total == 0) {
    return grad_weights;
  }

  c10::cuda::CUDAGuard device_guard(xs[0].device());
  auto handle = at::cuda::getCurrentCUDABlasHandle();
  configure_math(handle, fast_tf32);

  std::vector<cublasOperation_t> transa;
  std::vector<cublasOperation_t> transb;
  std::vector<int> m, n, k, lda, ldb, ldc, group_size;
  std::vector<int64_t> a_array, b_array, c_array;
  std::vector<float> alpha, beta;
  transa.reserve(active_total);
  transb.reserve(active_total);
  m.reserve(active_total);
  n.reserve(active_total);
  k.reserve(active_total);
  lda.reserve(active_total);
  ldb.reserve(active_total);
  ldc.reserve(active_total);
  group_size.reserve(active_total);
  a_array.reserve(active_total);
  b_array.reserve(active_total);
  c_array.reserve(active_total);
  alpha.reserve(active_total);
  beta.reserve(active_total);

  for (int64_t p = 0; p < problems; ++p) {
    const auto x = xs[p];
    const auto grad_out = grad_outs[p];
    const auto ptr = ptrs[p];
    const int64_t groups = ptr.numel() - 1;
    const int64_t* ptr_data = ptr.data_ptr<int64_t>();
    const float* x_base = x.data_ptr<float>();
    const float* go_base = grad_out.data_ptr<float>();
    float* gw_base = grad_weights[p].data_ptr<float>();
    const int in_i = static_cast<int>(x.size(1));
    const int out_i = static_cast<int>(grad_out.size(1));

    for (int64_t g = 0; g < groups; ++g) {
      const int64_t start = ptr_data[g];
      const int64_t end = ptr_data[g + 1];
      const int64_t rows = end - start;
      if (rows == 0) {
        continue;
      }
      transa.push_back(CUBLAS_OP_N);
      transb.push_back(CUBLAS_OP_T);
      m.push_back(in_i);
      n.push_back(out_i);
      k.push_back(static_cast<int>(rows));
      lda.push_back(in_i);
      ldb.push_back(out_i);
      ldc.push_back(in_i);
      group_size.push_back(1);
      alpha.push_back(1.0f);
      beta.push_back(0.0f);
      a_array.push_back(reinterpret_cast<int64_t>(x_base + start * x.size(1)));
      b_array.push_back(reinterpret_cast<int64_t>(go_base + start * grad_out.size(1)));
      c_array.push_back(reinterpret_cast<int64_t>(gw_base + g * grad_out.size(1) * x.size(1)));
    }
  }

  auto a_dev = copy_pointer_array_to_device(a_array, xs[0]);
  auto b_dev = copy_pointer_array_to_device(b_array, xs[0]);
  auto c_dev = copy_pointer_array_to_device(c_array, xs[0]);
  const cublasComputeType_t compute_type =
      fast_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

  check_cublas(cublasGemmGroupedBatchedEx(
      handle,
      transa.data(),
      transb.data(),
      m.data(),
      n.data(),
      k.data(),
      static_cast<const void*>(alpha.data()),
      reinterpret_cast<const void* const*>(a_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      lda.data(),
      reinterpret_cast<const void* const*>(b_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      ldb.data(),
      static_cast<const void*>(beta.data()),
      reinterpret_cast<void* const*>(c_dev.data_ptr<int64_t>()),
      CUDA_R_32F,
      ldc.data(),
      static_cast<int>(group_size.size()),
      group_size.data(),
      compute_type));

  return grad_weights;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grouped_gemm_forward_fp32", &grouped_gemm_forward_fp32, "cuBLAS grouped GEMM forward fp32");
  m.def("grouped_gemm_backward_weight_fp32", &grouped_gemm_backward_weight_fp32, "cuBLAS grouped GEMM grad weight fp32");
  m.def("grouped_gemm_multi_forward_fp32", &grouped_gemm_multi_forward_fp32, "cuBLAS multi-problem grouped GEMM forward fp32");
  m.def("grouped_gemm_multi_backward_weight_fp32", &grouped_gemm_multi_backward_weight_fp32, "cuBLAS multi-problem grouped GEMM grad weight fp32");
}
