#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cassert>

template<typename T> __global__ void contract_kernel(const T* a,const T* m,const T* b,
 const int64_t* rows,T* out,int64_t terms,int ni,int nj,int rank_a,int rank_b,
 int width,bool diagonal,int64_t na,int64_t nb,int64_t blocks) {
 const int64_t t=blockIdx.x;
 const int64_t dest=rows[3*t],left=rows[3*t+1],right=rows[3*t+2];
 assert(dest>=0 && dest<blocks && left>=0 && left<na && right>=0 && right<nb);
 extern __shared__ double storage[];
 T* weighted=reinterpret_cast<T*>(storage);
 if(!diagonal) {
  for(int index=threadIdx.x;index<ni*rank_b;index+=blockDim.x) {
   const int k=index/ni,i=index%ni;T value=0;
   for(int h=0;h<rank_a;++h)value+=a[(left*rank_a+h)*ni+i]*m[h*rank_b+k];
   weighted[index]=value;
  }
  __syncthreads();
 }
 for(int o=threadIdx.x;o<ni*nj;o+=blockDim.x) {
  const int i=o/nj,j=o%nj;T sum=0;
  // Preserve the two-stage contraction: (A^T M) first, then B.
  for(int k=0;k<rank_b;++k) {
   const T value=diagonal?a[(left*rank_a+k)*ni+i]*m[k]:weighted[k*ni+i];
   sum+=value*b[(right*rank_b+k)*nj+j];
  }
  atomicAdd(out+(dest*width+i)*width+j,sum);
 }
}

void nacf_contract_add(at::Tensor a,at::Tensor m,at::Tensor b,at::Tensor rows,at::Tensor out) {
 TORCH_CHECK(a.is_cuda(),"contraction requires CUDA");
 const c10::cuda::CUDAGuard guard(a.device());
 for(auto t:{a,m,b,out})TORCH_CHECK(t.device()==a.device() && t.scalar_type()==a.scalar_type() && t.is_contiguous(),"contraction requires contiguous matching buffers");
 TORCH_CHECK(a.dim()==3 && b.dim()==3 && out.dim()==3 && out.size(1)==out.size(2) && a.size(2)<=out.size(1) && b.size(2)<=out.size(2),"invalid contraction blocks");
 TORCH_CHECK(rows.device()==a.device() && rows.scalar_type()==at::kLong && rows.is_contiguous() && rows.dim()==2 && rows.size(1)==3,"invalid contraction rows");
 TORCH_CHECK((m.dim()==1 && a.size(1)==b.size(1) && m.numel()==a.size(1)) || (m.dim()==2 && m.size(0)==a.size(1) && m.size(1)==b.size(1)),"invalid contraction matrix");
 if(rows.size(0)==0)return;
 const size_t shared=m.dim()==1?0:a.size(2)*b.size(1)*a.element_size();
 TORCH_CHECK(shared<=48*1024,"contraction shared memory limit exceeded");
 AT_DISPATCH_FLOATING_TYPES(a.scalar_type(),"nacf_contract_add",[&]{
  contract_kernel<scalar_t><<<rows.size(0),128,shared,at::cuda::getCurrentCUDAStream()>>>(a.data_ptr<scalar_t>(),m.data_ptr<scalar_t>(),b.data_ptr<scalar_t>(),rows.data_ptr<int64_t>(),out.data_ptr<scalar_t>(),rows.size(0),a.size(2),b.size(2),a.size(1),b.size(1),out.size(1),m.dim()==1,a.size(0),b.size(0),out.size(0));
 });
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
