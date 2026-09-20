#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Batched onsite density on one shared quadrature grid: rho[a,p] for every atom a of a
// (species, order) group and every point p of the group's quadrature. One thread owns one
// (atom, point) pair and walks that atom's neighbour list, which is a sequence of species
// segments [begin,end) into the concatenated displacement buffer. Every segment is summed in
// 16-neighbour chunks (chunk partial sum, then added to the running density) exactly like the
// accepted per-species Torch loop. Per-channel positivity, end clamping and support clipping
// follow AtomicDensity: rr=clamp(r,k0,kK), interval=searchsorted(right)-1 clamped, per-channel
// cubic max(0,.), and zero contribution for r>kK. No distance or spline intermediate is stored.
// The knot interval is located with a float32 binary search (the FP64 pipe of consumer GPUs runs
// at 1/64 rate) and then corrected against the FP64 knots, so the interval and every FP64 value
// are identical to a pure FP64 searchsorted.
template<typename T> __global__ void onsite_density_kernel(
 const T* __restrict__ xyz,const T* __restrict__ pos,const int64_t* __restrict__ seg_ptr,
 const int64_t* __restrict__ segments,const T* __restrict__ knots,const float* __restrict__ knots_f,
 const int64_t* __restrict__ knot_ptr,const T* __restrict__ coeff,const int64_t* __restrict__ coeff_ptr,
 const int64_t* __restrict__ channels,T* __restrict__ rho,int64_t points) {
 const int64_t p=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
 const int a=blockIdx.y;
 if(p>=points)return;
 const T x0=xyz[3*p],y0=xyz[3*p+1],z0=xyz[3*p+2];
 T total=0;
 for(int64_t s=seg_ptr[a];s<seg_ptr[a+1];++s) {
  const int64_t sp=segments[3*s],begin=segments[3*s+1],end=segments[3*s+2];
  const T* kn=knots+knot_ptr[sp];const float* knf=knots_f+knot_ptr[sp];const int nk=static_cast<int>(knot_ptr[sp+1]-knot_ptr[sp]);const int m=nk-1;
  const T* cf=coeff+coeff_ptr[sp];const int nch=static_cast<int>(channels[sp]);
  const T kfirst=kn[0],klast=kn[m];
  for(int64_t start=begin;start<end;start+=16) {
   const int64_t stop=min(start+16,end);T sum=0;
   for(int64_t k=start;k<stop;++k) {
    const T dx=x0-pos[3*k],dy=y0-pos[3*k+1],dz=z0-pos[3*k+2];
    const T r=sqrt(dx*dx+dy*dy+dz*dz);
    if(r>klast)continue;
    const T rr=max(kfirst,min(r,klast));
    const float rf=static_cast<float>(rr);
    int lo=0,hi=nk;
    while(lo<hi){const int mid=(lo+hi)>>1;if(rf<knf[mid])hi=mid;else lo=mid+1;}
    int i=min(max(lo-1,0),m-1);
    while(i>0 && rr<kn[i])--i;
    while(i<m-1 && rr>=kn[i+1])++i;
    const T d=rr-kn[i];
    T v=0;
    for(int c=0;c<nch;++c){const T* q=cf+static_cast<int64_t>(c)*4*m+i;v+=max(T(0),((q[0]*d+q[m])*d+q[2*m])*d+q[3*m]);}
    sum+=v;
   }
   total+=sum;
  }
 }
 rho[static_cast<int64_t>(a)*points+p]=total;
}

at::Tensor nacf_onsite_density(at::Tensor xyz,at::Tensor pos,at::Tensor seg_ptr,at::Tensor segments,
 at::Tensor knots,at::Tensor knots_f,at::Tensor knot_ptr,at::Tensor coeff,at::Tensor coeff_ptr,at::Tensor channels) {
 TORCH_CHECK(xyz.is_cuda(),"onsite density requires CUDA");
 const c10::cuda::CUDAGuard guard(xyz.device());
 for(auto t:{xyz,pos,knots,coeff})TORCH_CHECK(t.device()==xyz.device() && t.scalar_type()==xyz.scalar_type() && t.is_contiguous(),"onsite density buffers must be contiguous and share device and dtype");
 for(auto t:{seg_ptr,segments,knot_ptr,coeff_ptr,channels})TORCH_CHECK(t.device()==xyz.device() && t.scalar_type()==at::kLong && t.is_contiguous(),"onsite density index buffers must be contiguous int64 on the same device");
 TORCH_CHECK(knots_f.device()==xyz.device() && knots_f.scalar_type()==at::kFloat && knots_f.is_contiguous() && knots_f.numel()==knots.numel(),"onsite density needs a float32 copy of the knots for the interval search");
 TORCH_CHECK(xyz.dim()==2 && xyz.size(1)==3 && pos.dim()==2 && pos.size(1)==3,"xyz and displacements must be [n,3]");
 TORCH_CHECK(seg_ptr.dim()==1 && seg_ptr.numel()>=1 && segments.dim()==2 && segments.size(1)==3,"invalid onsite segment layout");
 TORCH_CHECK(knot_ptr.dim()==1 && coeff_ptr.dim()==1 && channels.dim()==1 && knot_ptr.numel()==coeff_ptr.numel() && knot_ptr.numel()==channels.numel()+1 && channels.numel()>=1,"invalid onsite species layout");
 const int64_t atoms=seg_ptr.numel()-1,points=xyz.size(0);
 TORCH_CHECK(atoms<=65535,"onsite density supports at most 65535 atoms per launch");
 auto rho=at::zeros({atoms,points},xyz.options());
 if(!atoms || !points)return rho;
 const dim3 grid(static_cast<unsigned>((points+255)/256),static_cast<unsigned>(atoms));
 AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(),"nacf_onsite_density",[&]{
  onsite_density_kernel<scalar_t><<<grid,256,0,at::cuda::getCurrentCUDAStream()>>>(
   xyz.data_ptr<scalar_t>(),pos.data_ptr<scalar_t>(),seg_ptr.data_ptr<int64_t>(),segments.data_ptr<int64_t>(),
   knots.data_ptr<scalar_t>(),knots_f.data_ptr<float>(),knot_ptr.data_ptr<int64_t>(),coeff.data_ptr<scalar_t>(),coeff_ptr.data_ptr<int64_t>(),
   channels.data_ptr<int64_t>(),rho.data_ptr<scalar_t>(),points);
 });
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 return rho;
}
