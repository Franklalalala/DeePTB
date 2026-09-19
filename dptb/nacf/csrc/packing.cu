#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <type_traits>

namespace {
template <typename T> struct Parts {
  using real_type = T;
  __device__ static T real(T x) { return x; }
  __device__ static T imag(T) { return T(0); }
};
template <typename T> struct Parts<c10::complex<T>> {
  using real_type = T;
  __device__ static T real(c10::complex<T> x) { return x.real(); }
  __device__ static T imag(c10::complex<T> x) { return x.imag(); }
};

template <typename In, typename Out>
__global__ void pack_kernel(const In* blocks, const int64_t* rows,
    const int64_t* indices, const int8_t* signs, const bool* imaginary,
    int64_t count, int64_t features, int64_t block_size, Out* output) {
  for (int64_t t = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       t < count; t += int64_t(gridDim.x)*blockDim.x) {
    const int64_t row = t/features, col = t%features;
    const int64_t map = rows[row]*features+col;
    const In value = blocks[row*block_size+indices[map]] * typename Parts<In>::real_type(signs[map]);
    if constexpr (std::is_same<Out,c10::complex<float>>::value || std::is_same<Out,c10::complex<double>>::value) {
      using Real = typename Parts<Out>::real_type;
      output[t] = Out(Real(Parts<In>::real(value)), Real(Parts<In>::imag(value)));
    } else {
      output[t] = Out(imaginary[map] ? Parts<In>::imag(value) : Parts<In>::real(value));
    }
  }
}
}

void nacf_pack_cuda_out(at::Tensor blocks, at::Tensor rows,
    at::Tensor indices, at::Tensor signs, at::Tensor imaginary, at::Tensor output) {
  TORCH_CHECK(blocks.is_cuda() && blocks.dim()==3 && blocks.size(1)==blocks.size(2),
              "AO blocks must be CUDA [blocks,width,width]");
  const c10::cuda::CUDAGuard guard(blocks.device());
  for (auto t : {blocks,rows,indices,signs,imaginary,output})
    TORCH_CHECK(t.device()==blocks.device() && t.is_contiguous(), "packing buffers must be contiguous on the same CUDA device");
  TORCH_CHECK(rows.scalar_type()==at::kLong && indices.scalar_type()==at::kLong &&
              signs.scalar_type()==at::kChar && imaginary.scalar_type()==at::kBool,
              "invalid packing metadata dtypes");
  TORCH_CHECK(rows.dim()==1 && rows.size(0)==blocks.size(0) && indices.dim()==2 &&
              signs.sizes()==indices.sizes() && imaginary.sizes()==indices.sizes() &&
              output.dim()==2 && output.size(0)==blocks.size(0) && output.size(1)==indices.size(1),
              "invalid packing metadata shapes");
  if (!output.numel()) return;
  TORCH_CHECK(indices.size(0)>0 && blocks.size(1)>0, "empty packing templates or AO width");
  const int grid = int(std::min<int64_t>((output.numel()+255)/256, 65535));
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(blocks.scalar_type(), "nacf_pack_input", [&] {
    using In = scalar_t;
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(output.scalar_type(), "nacf_pack_output", [&] {
      pack_kernel<In,scalar_t><<<grid,256,0,at::cuda::getCurrentCUDAStream()>>>(
          blocks.data_ptr<In>(), rows.data_ptr<int64_t>(), indices.data_ptr<int64_t>(),
          signs.data_ptr<int8_t>(), imaginary.data_ptr<bool>(), output.numel(),
          output.size(1), blocks.size(1)*blocks.size(2), output.data_ptr<scalar_t>());
    });
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
