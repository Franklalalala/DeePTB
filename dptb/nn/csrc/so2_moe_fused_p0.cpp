#include <torch/extension.h>

#include <vector>

torch::Tensor fused_pair_forward_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor graph_index,
    torch::Tensor mixed_weight,
    torch::Tensor radial,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride);

std::vector<torch::Tensor> fused_pair_backward_fp32_cuda(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor graph_index,
    torch::Tensor mixed_weight,
    torch::Tensor radial,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor pack_pair_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t m,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor output_pair_grad_fp32_cuda(
    torch::Tensor grad_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor scatter_pair_grad_fp32_cuda(
    torch::Tensor grad_pair,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    int64_t m,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride);

#ifdef DPTB_SO2_MOE_FUSED_P0_CUTLASS
torch::Tensor cutlass_cute_probe_cuda(torch::Tensor a, torch::Tensor b);
#endif

static void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

static void check_common(
    const torch::Tensor& x,
    const torch::Tensor& wigner,
    const torch::Tensor& graph_index,
    const torch::Tensor& mixed_weight,
    const torch::Tensor& radial,
    const torch::Tensor& in_base,
    const torch::Tensor& in_l,
    const torch::Tensor& out_base,
    const torch::Tensor& out_l,
    const torch::Tensor& offsets,
    const torch::Tensor& compact_offsets,
    int64_t out_dim,
    int64_t wigner_mode,
    int64_t wigner_stride,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(graph_index, "graph_index");
  check_cuda_contiguous(mixed_weight, "mixed_weight");
  check_cuda_contiguous(radial, "radial");
  check_cuda_contiguous(in_base, "in_base");
  check_cuda_contiguous(in_l, "in_l");
  check_cuda_contiguous(out_base, "out_base");
  check_cuda_contiguous(out_l, "out_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");

  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(mixed_weight.scalar_type() == torch::kFloat32, "mixed_weight must be fp32");
  TORCH_CHECK(radial.numel() == 0 || radial.scalar_type() == torch::kFloat32, "radial must be fp32");
  TORCH_CHECK(graph_index.scalar_type() == torch::kInt64, "graph_index must be int64");
  TORCH_CHECK(in_base.scalar_type() == torch::kInt64, "in_base must be int64");
  TORCH_CHECK(in_l.scalar_type() == torch::kInt64, "in_l must be int64");
  TORCH_CHECK(out_base.scalar_type() == torch::kInt64, "out_base must be int64");
  TORCH_CHECK(out_l.scalar_type() == torch::kInt64, "out_l must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");

  TORCH_CHECK(x.dim() == 2, "x must be [N, in_dim]");
  TORCH_CHECK(graph_index.dim() == 1 && graph_index.numel() == x.size(0), "graph_index must be [N]");
  TORCH_CHECK(mixed_weight.dim() == 3, "mixed_weight must be [routes, 2*Cout, Cin]");
  TORCH_CHECK(in_base.dim() == 1 && in_l.dim() == 1 && in_base.numel() == in_l.numel(), "input maps must be 1D and aligned");
  TORCH_CHECK(out_base.dim() == 1 && out_l.dim() == 1 && out_base.numel() == out_l.numel(), "output maps must be 1D and aligned");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");
  TORCH_CHECK(mixed_weight.size(1) == 2 * out_base.numel(), "mixed_weight O does not match out maps");
  TORCH_CHECK(mixed_weight.size(2) == in_base.numel(), "mixed_weight I does not match in maps");
  TORCH_CHECK(out_dim > 0, "out_dim must be positive");

  if (rotate_in || rotate_out) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
    TORCH_CHECK(wigner_mode == 1 || wigner_mode == 2, "wigner_mode must be dense(1) or compact(2) when rotation is active");
    if (wigner_mode == 1) {
      TORCH_CHECK(wigner.dim() == 3 && wigner.size(0) == x.size(0) && wigner.size(1) == wigner.size(2),
                  "dense wigner must be [N,D,D]");
      TORCH_CHECK(wigner_stride == wigner.size(1), "dense wigner_stride mismatch");
    } else {
      TORCH_CHECK(wigner.dim() == 2 && wigner.size(0) == x.size(0), "compact wigner must be [N,flat]");
      TORCH_CHECK(wigner_stride == wigner.size(1), "compact wigner_stride mismatch");
      TORCH_CHECK(compact_offsets.dim() == 1 && compact_offsets.numel() > 0, "compact_offsets must be [L]");
    }
  } else {
    TORCH_CHECK(wigner_mode == 0 || wigner_mode == 1 || wigner_mode == 2, "invalid wigner_mode");
  }

  if (radial.numel() != 0) {
    TORCH_CHECK(radial.dim() == 2 && radial.size(0) == x.size(0), "radial must be [N,C]");
    TORCH_CHECK(radial.size(1) == (radial_on_input ? in_base.numel() : out_base.numel()),
                "radial channel count mismatch");
  }
}

torch::Tensor fused_pair_forward_fp32(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor graph_index,
    torch::Tensor mixed_weight,
    torch::Tensor radial,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_common(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, wigner_mode, wigner_stride, rotate_in, rotate_out, radial_on_input);
  return fused_pair_forward_fp32_cuda(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}

std::vector<torch::Tensor> fused_pair_backward_fp32(
    torch::Tensor grad_out,
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor graph_index,
    torch::Tensor mixed_weight,
    torch::Tensor radial,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(grad_out, "grad_out");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be fp32");
  TORCH_CHECK(grad_out.dim() == 2 && grad_out.size(0) == x.size(0) && grad_out.size(1) == out_dim,
              "grad_out must be [N,out_dim]");
  check_common(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, wigner_mode, wigner_stride, rotate_in, rotate_out, radial_on_input);
  return fused_pair_backward_fp32_cuda(
      grad_out, x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}

torch::Tensor pack_pair_fp32(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t m,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(in_base, "in_base");
  check_cuda_contiguous(in_l, "in_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(in_base.scalar_type() == torch::kInt64, "in_base must be int64");
  TORCH_CHECK(in_l.scalar_type() == torch::kInt64, "in_l must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(x.dim() == 2, "x must be [N, in_dim]");
  TORCH_CHECK(in_base.dim() == 1 && in_l.dim() == 1 && in_base.numel() == in_l.numel(),
              "input maps must be 1D and aligned");
  if (rotate_in) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return pack_pair_fp32_cuda(
      x, wigner, in_base, in_l, offsets, compact_offsets,
      m, rotate_in, wigner_mode, wigner_stride);
}

torch::Tensor output_pair_grad_fp32(
    torch::Tensor grad_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(grad_out, "grad_out");
  check_cuda_contiguous(out_base, "out_base");
  check_cuda_contiguous(out_l, "out_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  TORCH_CHECK(grad_out.scalar_type() == torch::kFloat32, "grad_out must be fp32");
  TORCH_CHECK(out_base.scalar_type() == torch::kInt64, "out_base must be int64");
  TORCH_CHECK(out_l.scalar_type() == torch::kInt64, "out_l must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(grad_out.dim() == 2, "grad_out must be [N, out_dim]");
  TORCH_CHECK(out_base.dim() == 1 && out_l.dim() == 1 && out_base.numel() == out_l.numel(),
              "output maps must be 1D and aligned");
  if (rotate_out) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return output_pair_grad_fp32_cuda(
      grad_out, wigner, out_base, out_l, offsets, compact_offsets,
      m, rotate_out, wigner_mode, wigner_stride);
}

torch::Tensor scatter_pair_grad_fp32(
    torch::Tensor grad_pair,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    int64_t m,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(grad_pair, "grad_pair");
  check_cuda_contiguous(in_base, "in_base");
  check_cuda_contiguous(in_l, "in_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  TORCH_CHECK(grad_pair.scalar_type() == torch::kFloat32, "grad_pair must be fp32");
  TORCH_CHECK(in_base.scalar_type() == torch::kInt64, "in_base must be int64");
  TORCH_CHECK(in_l.scalar_type() == torch::kInt64, "in_l must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(grad_pair.dim() == 3 && grad_pair.size(1) == 2,
              "grad_pair must be [N, 2, Cin]");
  TORCH_CHECK(in_base.dim() == 1 && in_l.dim() == 1 && in_base.numel() == in_l.numel(),
              "input maps must be 1D and aligned");
  TORCH_CHECK(grad_pair.size(2) == in_base.numel(), "grad_pair Cin mismatch");
  if (rotate_in) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return scatter_pair_grad_fp32_cuda(
      grad_pair, wigner, in_base, in_l, offsets, compact_offsets,
      in_dim, m, rotate_in, wigner_mode, wigner_stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_pair_forward_fp32", &fused_pair_forward_fp32, "SO2 MoE fused P0 pair forward fp32");
  m.def("fused_pair_backward_fp32", &fused_pair_backward_fp32, "SO2 MoE fused P0 pair backward fp32");
  m.def("pack_pair_fp32", &pack_pair_fp32, "SO2 MoE fused P0 pack pair fp32");
  m.def("output_pair_grad_fp32", &output_pair_grad_fp32, "SO2 MoE fused P0 output pair grad fp32");
  m.def("scatter_pair_grad_fp32", &scatter_pair_grad_fp32, "SO2 MoE fused P0 scatter pair grad fp32");
#ifdef DPTB_SO2_MOE_FUSED_P0_CUTLASS
  m.def("cutlass_cute_probe", &cutlass_cute_probe_cuda, "Compile-only CuTe dot-mainloop probe");
#endif
}
