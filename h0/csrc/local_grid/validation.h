#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
#include <cmath>

namespace local_check {
constexpr int64_t capacity = INT32_MAX - 256;
constexpr int64_t coordinate_limit = capacity / 4;
inline void tensor(const torch::Tensor& t, c10::Device dev, torch::ScalarType dtype, int rank) {
    TORCH_CHECK(t.is_cuda() && t.device()==dev && t.scalar_type()==dtype && t.dim()==rank && t.is_contiguous(),
        "invalid local-grid device, dtype, rank or contiguous layout");
    TORCH_CHECK(t.numel()<=capacity, "local-grid tensor exceeds int32 capacity");
    for (auto size:t.sizes()) TORCH_CHECK(size<=capacity, "local-grid dimension exceeds int32 capacity");
}
inline void vector3(const torch::Tensor& t, c10::Device dev, torch::ScalarType dtype) {
    tensor(t,dev,dtype,1); TORCH_CHECK(t.numel()==3, "local-grid vector must have three entries");
}
inline void finite(const torch::Tensor& t) {
    TORCH_CHECK(torch::isfinite(t).all().item<bool>(), "nonfinite local-grid data");
}
inline int64_t product(int64_t a,int64_t b) {
    TORCH_CHECK(a>=0 && b>=0 && (b==0 || a<=capacity/b), "local-grid count exceeds int32 capacity");
    return a*b;
}
inline void candidate_capacity(int64_t candidates) {
    // Before scanning: bound both candidate traversal and the worst possible
    // [npoints,3] coordinate output. AO storage is checked after compaction.
    product(candidates,3);
}
inline void support_capacity(int64_t points,int64_t norb) {
    TORCH_CHECK(norb>0, "invalid support orbital count");
    product(points,norb);
    // Both AO evaluation and lookup population index integer_indices[3*k+a]
    // with int arithmetic. Check the full output before any allocation.
    product(points,3);
}
inline void coordinates(const torch::Tensor& t) {
    TORCH_CHECK(((t>=-coordinate_limit)&(t<=coordinate_limit)).all().item<bool>(), "local-grid coordinates exceed supported range");
}
inline void anchor(const torch::Tensor& lo,const torch::Tensor& hi,const torch::Tensor& lookup,
                   const torch::Tensor& values,c10::Device dev,int norb) {
    vector3(lo,dev,torch::kInt64);vector3(hi,dev,torch::kInt64);
    coordinates(lo);coordinates(hi);
    tensor(lookup,dev,torch::kInt32,3);tensor(values,dev,torch::kFloat64,2);
    TORCH_CHECK(norb>0 && values.size(1)==norb, "invalid anchor orbital shape");
    finite(values);
    if (!lookup.numel()) {
        TORCH_CHECK(values.size(0)==0 && lookup.size(0)==0 && lookup.size(1)==0 && lookup.size(2)==0, "invalid empty anchor");
        return;
    }
    auto extent=(hi-lo+1).to(torch::kCPU);auto a=extent.accessor<int64_t,1>();
    for(int k=0;k<3;++k) TORCH_CHECK(a[k]>0 && lookup.size(k)==a[k], "anchor lookup extent mismatch");
    TORCH_CHECK(lookup.min().item<int>()>=-1 && lookup.max().item<int64_t>()<values.size(0), "anchor lookup index out of range");
}
}
