#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include <limits>

// Coefficients retain the source cubic order [interval, power(3..0), channel].
// Binary search preserves nonuniform knots and a shortened final interval.
template <typename T>
__device__ T spline(const T* c, int interval, int channels, int channel, T x) {
  const T* row = c + static_cast<int64_t>(interval) * 4 * channels + channel;
  return ((row[0] * x + row[channels]) * x + row[2*channels]) * x + row[3*channels];
}

template <typename T>
__device__ T harmonic(int l, int a, T x, T y, T z, T scale) {
  const T norm = sqrt(x*x + y*y + z*z);
  x /= norm; y /= norm; z /= norm;
  const int m = (a + 1) / 2;
  T re = 1, im = 0, q = 1;
  for (int k = 1; k <= m; ++k) {
    const T next = re*x - im*y;
    im = im*x + re*y; re = next;
    q *= -(2*k - 1);
  }
  if (l > m) {
    T prev = q;
    q *= (2*m + 1)*z;
    for (int k = m+2; k <= l; ++k) {
      const T next = ((2*k-1)*z*q - (k+m-1)*prev)/(k-m);
      prev = q; q = next;
    }
  }
  return scale*q*(a == 0 ? T(1) : (a % 2 ? re : im));
}

template <typename T>
__global__ void radial_kernel(
    const T* vectors, const T* knots, const T* coefficients,
    const int64_t* degrees, const T* directions, const T* inverse,
    const T* scales, const int64_t* ptr, const int64_t* terms,
    const int64_t* canonical, int nk, int channels, int ngroups,
    int nrot, int width, T support, T* output) {
  extern __shared__ double storage[];
  T* ys = reinterpret_cast<T*>(storage);
  T* rot = ys + nrot;
  T* cart = rot + nrot;
  T* radius = cart + 9;
  T* delta = radius + 1;
  __shared__ int interval;
  const int query = blockIdx.x;
  if (threadIdx.x == 0) {
    const int64_t vbase=static_cast<int64_t>(query)*3;
    const T vx=vectors[vbase], vy=vectors[vbase+1], vz=vectors[vbase+2];
    *radius = sqrt(vx*vx + vy*vy + vz*vz);
    int lo=0, hi=nk;
    while (lo < hi) {
      const int mid=(lo+hi)/2;
      if (*radius < knots[mid]) hi=mid; else lo=mid+1;
    }
    interval = min(max(lo, 1), nk-1)-1;
    *delta = *radius-knots[interval];
    const T denom = *radius > T(1e-30) ? *radius : T(1e-30);
    const T x=vx/denom, y=vy/denom, z=vz/denom;
    const T xy=x*x+y*y;
    const T divisor=z < 0 ? xy : 1+z;
    const T factor=(z < 0 ? 1-z : 1)/(divisor > T(1e-30) ? divisor : T(1e-30));
    cart[0]=1-x*x*factor; cart[1]=-x*y*factor; cart[2]=x;
    cart[3]=-x*y*factor; cart[4]=1-y*y*factor; cart[5]=y;
    cart[6]=-x; cart[7]=-y; cart[8]=1-xy*factor;
    if (z <= T(-1+1e-14) || z >= T(1-1e-14) || *radius <= T(1e-14)) {
      for (int k=0;k<9;++k) cart[k]=0;
      cart[0]=1;
      cart[4]=cart[8]=(z <= T(-1+1e-14) && *radius > T(1e-14)) ? -1 : 1;
    }
  }
  __syncthreads();
  if (*radius >= support - T(1e-12) || *radius <= T(1e-14)) {
    for (int o=threadIdx.x;o<width;o+=blockDim.x) {
      const int c=canonical[o];
      output[static_cast<int64_t>(query)*width+o]=
          (*radius >= support-T(1e-12) || c < 0) ? T(0) : spline(coefficients,interval,channels,c,*delta);
    }
    return;
  }
  // Evaluate Y_lm(R directions) once, then recover the small rotation matrices.
  for (int g=0;g<ngroups;++g) {
    const int l=degrees[3*g], db=degrees[3*g+1], rb=degrees[3*g+2], d=2*l+1;
    for (int t=threadIdx.x;t<d*d;t+=blockDim.x) {
      const int s=t/d, a=t%d;
      const T* v=directions+3*(db+s);
      ys[rb+t]=harmonic(l,a,cart[0]*v[0]+cart[1]*v[1]+cart[2]*v[2],
          cart[3]*v[0]+cart[4]*v[1]+cart[5]*v[2],
          cart[6]*v[0]+cart[7]*v[1]+cart[8]*v[2],scales[db+a]);
    }
  }
  __syncthreads();
  for (int g=0;g<ngroups;++g) {
    const int l=degrees[3*g], rb=degrees[3*g+2], d=2*l+1;
    for (int t=threadIdx.x;t<d*d;t+=blockDim.x) {
      const int a=t/d, b=t%d;
      T sum=0;
      for (int s=0;s<d;++s) sum+=inverse[rb+b*d+s]*ys[rb+s*d+a];
      rot[rb+t]=sum;
    }
  }
  __syncthreads();
  // Sparse canonical channels and shell selection were compiled on the host.
  for (int o=threadIdx.x;o<width;o+=blockDim.x) {
    T sum=0;
    for (int64_t t=ptr[o];t<ptr[o+1];++t) {
      const int64_t* term=terms+3*t;
      sum+=rot[term[1]]*spline(coefficients,interval,channels,term[0],*delta)*rot[term[2]];
    }
    output[static_cast<int64_t>(query)*width+o]=sum;
  }
}

at::Tensor nacf_radial_cuda(
    at::Tensor vectors, at::Tensor knots, at::Tensor coefficients,
    at::Tensor degrees, at::Tensor directions, at::Tensor inverse,
    at::Tensor scales, at::Tensor ptr, at::Tensor terms,
    at::Tensor canonical, double support) {
  TORCH_CHECK(vectors.is_cuda() && vectors.dim()==2 && vectors.size(1)==3,
              "vectors must be CUDA [queries,3]");
  const c10::cuda::CUDAGuard guard(vectors.device());
  TORCH_CHECK(vectors.scalar_type()==at::kFloat || vectors.scalar_type()==at::kDouble,
              "NACF CUDA supports float32/float64 only");
  for (auto t : {vectors,knots,coefficients,directions,inverse,scales})
    TORCH_CHECK(t.device()==vectors.device() && t.scalar_type()==vectors.scalar_type() && t.is_contiguous(),
                "floating buffers must match device/dtype and be contiguous");
  for (auto t : {degrees,ptr,terms,canonical})
    TORCH_CHECK(t.device()==vectors.device() && t.scalar_type()==at::kLong && t.is_contiguous(),
                "index buffers must be contiguous int64 on the same CUDA device");
  TORCH_CHECK(knots.dim()==1 && knots.numel()>=2 && coefficients.dim()==3 &&
              coefficients.size(0)==knots.numel()-1 && coefficients.size(1)==4,
              "invalid cubic coefficient layout");
  TORCH_CHECK(degrees.dim()==2 && degrees.size(1)==3 && directions.dim()==2 && directions.size(1)==3 &&
              inverse.dim()==1 && scales.dim()==1 && scales.numel()==directions.size(0) &&
              canonical.dim()==1 && ptr.dim()==1 && ptr.numel()==canonical.numel()+1 &&
              terms.dim()==2 && terms.size(1)==3, "invalid compiled angular metadata");
  TORCH_CHECK(std::isfinite(support) && support>0, "invalid radial support");
  TORCH_CHECK(vectors.size(0)<=std::numeric_limits<int>::max() && canonical.numel()<=std::numeric_limits<int>::max() &&
              knots.numel()<=std::numeric_limits<int>::max(), "query batch, knots or AO block exceeds CUDA indexing range");
  const int nrot=inverse.numel();
  const size_t shared=(2*nrot+11)*vectors.element_size();
  TORCH_CHECK(shared <= 48*1024, "angular basis exceeds fused kernel shared-memory capacity");
  auto output=at::empty({vectors.size(0),canonical.numel()},vectors.options());
  if (output.numel()==0) return output;
  AT_DISPATCH_FLOATING_TYPES(vectors.scalar_type(), "nacf_radial_cuda", [&] {
    radial_kernel<scalar_t><<<vectors.size(0),128,shared,at::cuda::getCurrentCUDAStream()>>>(
        vectors.data_ptr<scalar_t>(),knots.data_ptr<scalar_t>(),coefficients.data_ptr<scalar_t>(),
        degrees.data_ptr<int64_t>(),directions.data_ptr<scalar_t>(),inverse.data_ptr<scalar_t>(),
        scales.data_ptr<scalar_t>(),ptr.data_ptr<int64_t>(),terms.data_ptr<int64_t>(),canonical.data_ptr<int64_t>(),
        knots.numel(),coefficients.size(2),degrees.size(0),nrot,canonical.numel(),static_cast<scalar_t>(support),
        output.data_ptr<scalar_t>());
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
