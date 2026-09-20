#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

template<typename T> __global__ void density_kernel(const T* xyz,const T* pos,
 const T* knots,const T* coeff,T* rho,int points,int neighbors,int nk,int channels) {
 const int p=blockIdx.x*blockDim.x+threadIdx.x;
 if(p>=points)return;
 T total=rho[p];
 // Retain the accepted 16-neighbour chunking and species accumulation order.
 for(int start=0;start<neighbors;start+=16) {
  T sum=0;
  for(int k=start;k<min(start+16,neighbors);++k) {
   T x=xyz[3*p]-pos[3*k],y=xyz[3*p+1]-pos[3*k+1],z=xyz[3*p+2]-pos[3*k+2];
   T r=sqrt(x*x+y*y+z*z);
   if(r>knots[nk-1])continue;
   r=max(knots[0],min(r,knots[nk-1]));
   int lo=0,hi=nk;
   while(lo<hi){int mid=(lo+hi)/2;if(r<knots[mid])hi=mid;else lo=mid+1;}
   int i=min(max(lo-1,0),nk-2);T d=r-knots[i],v=0;
   for(int c=0;c<channels;++c){
    const T* a=coeff+c*4*(nk-1)+i;
    v+=max(T(0),((a[0]*d+a[nk-1])*d+a[2*(nk-1)])*d+a[3*(nk-1)]);
   }
   sum+=v;
  }
  total+=sum;
 }
 rho[p]=total;
}

void nacf_density_add(at::Tensor xyz,at::Tensor pos,at::Tensor knots,at::Tensor coeff,at::Tensor rho) {
 TORCH_CHECK(xyz.is_cuda(),"density requires CUDA");
 const c10::cuda::CUDAGuard guard(xyz.device());
 for(auto t:{xyz,pos,knots,coeff,rho})TORCH_CHECK(t.device()==xyz.device() && t.scalar_type()==xyz.scalar_type() && t.is_contiguous(),"density buffers must match");
 TORCH_CHECK(xyz.dim()==2 && xyz.size(1)==3 && pos.dim()==2 && pos.size(1)==3 && knots.dim()==1 && knots.numel()>=2 && coeff.dim()==3 && coeff.size(1)==4 && coeff.size(2)==knots.numel()-1 && rho.dim()==1 && rho.numel()==xyz.size(0),"invalid density shapes");
 if(!xyz.size(0))return;
 AT_DISPATCH_FLOATING_TYPES(xyz.scalar_type(),"nacf_density_add",[&]{
  density_kernel<scalar_t><<<(xyz.size(0)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(xyz.data_ptr<scalar_t>(),pos.data_ptr<scalar_t>(),knots.data_ptr<scalar_t>(),coeff.data_ptr<scalar_t>(),rho.data_ptr<scalar_t>(),xyz.size(0),pos.size(0),knots.numel(),coeff.size(0));
 });
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
