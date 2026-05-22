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

torch::Tensor pack_pairs_multi_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    std::vector<torch::Tensor> in_bases,
    std::vector<torch::Tensor> in_ls,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cin_prefix,
    torch::Tensor m_values,
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

torch::Tensor scatter_pair_forward_fp32_cuda(
    torch::Tensor pair_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor scatter_raw_pair_forward_fp32_cuda(
    torch::Tensor raw,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor scatter_raw_pairs_multi_forward_fp32_cuda(
    std::vector<torch::Tensor> raws,
    torch::Tensor wigner,
    std::vector<torch::Tensor> out_bases,
    std::vector<torch::Tensor> out_ls,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cout_prefix,
    torch::Tensor m_values,
    int64_t out_dim,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor scatter_raw_pairs_multi_output_major_forward_fp32_cuda(
    std::vector<torch::Tensor> raws,
    torch::Tensor wigner,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cout_prefix,
    torch::Tensor m_values,
    torch::Tensor entry_offsets,
    torch::Tensor entry_m,
    torch::Tensor entry_channel,
    torch::Tensor entry_d,
    torch::Tensor entry_l,
    int64_t out_dim,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor raw_pair_output_grad_fp32_cuda(
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

torch::Tensor scatter_pairs_multi_grad_fp32_cuda(
    torch::Tensor grad_packed,
    torch::Tensor wigner,
    torch::Tensor in_base_all,
    torch::Tensor in_l_all,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cin_prefix,
    torch::Tensor m_values,
    int64_t in_dim,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride);

std::vector<torch::Tensor> scatter_pair_grad_radial_input_fp32_cuda(
    torch::Tensor grad_pair_eff,
    torch::Tensor pair_no_radial,
    torch::Tensor radial,
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

torch::Tensor fused_m0_forward_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor graph_index,
    torch::Tensor mixed_weight,
    torch::Tensor mixed_bias,
    torch::Tensor radial,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor pack_m0_fp32_cuda(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor output_m0_grad_fp32_cuda(
    torch::Tensor grad_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride);

torch::Tensor scatter_m0_grad_fp32_cuda(
    torch::Tensor grad_m0,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride);

std::vector<torch::Tensor> scatter_m0_grad_radial_input_fp32_cuda(
    torch::Tensor grad_eff,
    torch::Tensor m0_no_radial,
    torch::Tensor radial,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride);

#ifdef DPTB_SO2_MOE_FUSED_P0_CUTLASS
torch::Tensor cutlass_cute_probe_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor fused_pair_forward_tiled2_fp32_cuda(
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

torch::Tensor fused_pair_forward_tiled3_fp32_cuda(
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

torch::Tensor fused_pair_forward_tiled4_fp32_cuda(
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

torch::Tensor fused_pair_forward_tiled8_fp32_cuda(
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

static void check_m0_common(
    const torch::Tensor& x,
    const torch::Tensor& wigner,
    const torch::Tensor& graph_index,
    const torch::Tensor& mixed_weight,
    const torch::Tensor& mixed_bias,
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
  check_cuda_contiguous(mixed_bias, "mixed_bias");
  check_cuda_contiguous(radial, "radial");
  check_cuda_contiguous(in_base, "in_base");
  check_cuda_contiguous(in_l, "in_l");
  check_cuda_contiguous(out_base, "out_base");
  check_cuda_contiguous(out_l, "out_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");

  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(mixed_weight.scalar_type() == torch::kFloat32, "mixed_weight must be fp32");
  TORCH_CHECK(mixed_bias.numel() == 0 || mixed_bias.scalar_type() == torch::kFloat32, "mixed_bias must be fp32");
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
  TORCH_CHECK(mixed_weight.dim() == 3, "mixed_weight must be [routes, Cout, Cin]");
  TORCH_CHECK(in_base.dim() == 1 && in_l.dim() == 1 && in_base.numel() == in_l.numel(), "input maps must be 1D and aligned");
  TORCH_CHECK(out_base.dim() == 1 && out_l.dim() == 1 && out_base.numel() == out_l.numel(), "output maps must be 1D and aligned");
  TORCH_CHECK(offsets.dim() == 1, "offsets must be 1D");
  TORCH_CHECK(mixed_weight.size(1) == out_base.numel(), "mixed_weight O does not match out maps");
  TORCH_CHECK(mixed_weight.size(2) == in_base.numel(), "mixed_weight I does not match in maps");
  TORCH_CHECK(mixed_bias.numel() == 0 || (mixed_bias.dim() == 2 && mixed_bias.size(0) == mixed_weight.size(0) && mixed_bias.size(1) == mixed_weight.size(1)),
              "mixed_bias must be empty or [routes, Cout]");
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

torch::Tensor fused_m0_forward_fp32(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor graph_index,
    torch::Tensor mixed_weight,
    torch::Tensor mixed_bias,
    torch::Tensor radial,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_m0_common(
      x, wigner, graph_index, mixed_weight, mixed_bias, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, wigner_mode, wigner_stride, rotate_in, rotate_out, radial_on_input);
  return fused_m0_forward_fp32_cuda(
      x, wigner, graph_index, mixed_weight, mixed_bias, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}

#ifdef DPTB_SO2_MOE_FUSED_P0_CUTLASS
torch::Tensor fused_pair_forward_tiled2_fp32(
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
  return fused_pair_forward_tiled2_fp32_cuda(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}

torch::Tensor fused_pair_forward_tiled3_fp32(
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
  return fused_pair_forward_tiled3_fp32_cuda(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}

torch::Tensor fused_pair_forward_tiled4_fp32(
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
  return fused_pair_forward_tiled4_fp32_cuda(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}

torch::Tensor fused_pair_forward_tiled8_fp32(
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
  return fused_pair_forward_tiled8_fp32_cuda(
      x, wigner, graph_index, mixed_weight, radial,
      in_base, in_l, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_in, rotate_out, radial_on_input, wigner_mode, wigner_stride);
}
#endif

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

torch::Tensor pack_pairs_multi_fp32(
    torch::Tensor x,
    torch::Tensor wigner,
    std::vector<torch::Tensor> in_bases,
    std::vector<torch::Tensor> in_ls,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cin_prefix,
    torch::Tensor m_values,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(x, "x");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  check_cuda_contiguous(cin_prefix, "cin_prefix");
  check_cuda_contiguous(m_values, "m_values");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be fp32");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(cin_prefix.scalar_type() == torch::kInt64, "cin_prefix must be int64");
  TORCH_CHECK(m_values.scalar_type() == torch::kInt64, "m_values must be int64");
  TORCH_CHECK(cin_prefix.numel() == static_cast<int64_t>(in_bases.size()) + 1,
              "cin_prefix must have m_count + 1 entries");
  TORCH_CHECK(m_values.numel() == static_cast<int64_t>(in_bases.size()),
              "m_values must have m_count entries");
  if (rotate_in) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return pack_pairs_multi_fp32_cuda(
      x, wigner, in_bases, in_ls, offsets, compact_offsets,
      cin_prefix, m_values, rotate_in, wigner_mode, wigner_stride);
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

torch::Tensor scatter_pair_forward_fp32(
    torch::Tensor pair_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(pair_out, "pair_out");
  check_cuda_contiguous(out_base, "out_base");
  check_cuda_contiguous(out_l, "out_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  TORCH_CHECK(pair_out.scalar_type() == torch::kFloat32, "pair_out must be fp32");
  TORCH_CHECK(out_base.scalar_type() == torch::kInt64, "out_base must be int64");
  TORCH_CHECK(out_l.scalar_type() == torch::kInt64, "out_l must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(pair_out.dim() == 3 && pair_out.size(1) == 2, "pair_out must be [N, 2, Cout]");
  TORCH_CHECK(out_base.dim() == 1 && out_l.dim() == 1 && out_base.numel() == out_l.numel(),
              "output maps must be 1D and aligned");
  TORCH_CHECK(pair_out.size(2) == out_base.numel(), "pair_out Cout mismatch");
  TORCH_CHECK(out_dim > 0, "out_dim must be positive");
  if (rotate_out) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return scatter_pair_forward_fp32_cuda(
      pair_out, wigner, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_out, wigner_mode, wigner_stride);
}

torch::Tensor scatter_raw_pair_forward_fp32(
    torch::Tensor raw,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t out_dim,
    int64_t m,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(raw, "raw");
  check_cuda_contiguous(out_base, "out_base");
  check_cuda_contiguous(out_l, "out_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  TORCH_CHECK(raw.scalar_type() == torch::kFloat32, "raw must be fp32");
  TORCH_CHECK(out_base.scalar_type() == torch::kInt64, "out_base must be int64");
  TORCH_CHECK(out_l.scalar_type() == torch::kInt64, "out_l must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(raw.dim() == 3 && raw.size(1) == 2, "raw must be [N, 2, 2*Cout]");
  TORCH_CHECK(out_base.dim() == 1 && out_l.dim() == 1 && out_base.numel() == out_l.numel(),
              "output maps must be 1D and aligned");
  TORCH_CHECK(raw.size(2) == 2 * out_base.numel(), "raw 2*Cout mismatch");
  TORCH_CHECK(out_dim > 0, "out_dim must be positive");
  if (rotate_out) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return scatter_raw_pair_forward_fp32_cuda(
      raw, wigner, out_base, out_l, offsets, compact_offsets,
      out_dim, m, rotate_out, wigner_mode, wigner_stride);
}

torch::Tensor scatter_raw_pairs_multi_forward_fp32(
    std::vector<torch::Tensor> raws,
    torch::Tensor wigner,
    std::vector<torch::Tensor> out_bases,
    std::vector<torch::Tensor> out_ls,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cout_prefix,
    torch::Tensor m_values,
    int64_t out_dim,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  TORCH_CHECK(!raws.empty(), "raws must be non-empty");
  TORCH_CHECK(raws.size() == out_bases.size() && raws.size() == out_ls.size(),
              "raws/out_bases/out_ls length mismatch");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  check_cuda_contiguous(cout_prefix, "cout_prefix");
  check_cuda_contiguous(m_values, "m_values");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(cout_prefix.scalar_type() == torch::kInt64, "cout_prefix must be int64");
  TORCH_CHECK(m_values.scalar_type() == torch::kInt64, "m_values must be int64");
  TORCH_CHECK(cout_prefix.numel() == static_cast<int64_t>(raws.size()) + 1,
              "cout_prefix must have raw_count + 1 entries");
  TORCH_CHECK(m_values.numel() == static_cast<int64_t>(raws.size()),
              "m_values must have raw_count entries");
  TORCH_CHECK(out_dim > 0, "out_dim must be positive");
  if (rotate_out) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return scatter_raw_pairs_multi_forward_fp32_cuda(
      raws, wigner, out_bases, out_ls, offsets, compact_offsets,
      cout_prefix, m_values, out_dim, rotate_out, wigner_mode, wigner_stride);
}

torch::Tensor scatter_raw_pairs_multi_output_major_forward_fp32(
    std::vector<torch::Tensor> raws,
    torch::Tensor wigner,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cout_prefix,
    torch::Tensor m_values,
    torch::Tensor entry_offsets,
    torch::Tensor entry_m,
    torch::Tensor entry_channel,
    torch::Tensor entry_d,
    torch::Tensor entry_l,
    int64_t out_dim,
    bool rotate_out,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  TORCH_CHECK(!raws.empty(), "raws must be non-empty");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  check_cuda_contiguous(cout_prefix, "cout_prefix");
  check_cuda_contiguous(m_values, "m_values");
  check_cuda_contiguous(entry_offsets, "entry_offsets");
  check_cuda_contiguous(entry_m, "entry_m");
  check_cuda_contiguous(entry_channel, "entry_channel");
  check_cuda_contiguous(entry_d, "entry_d");
  check_cuda_contiguous(entry_l, "entry_l");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(cout_prefix.scalar_type() == torch::kInt64, "cout_prefix must be int64");
  TORCH_CHECK(m_values.scalar_type() == torch::kInt64, "m_values must be int64");
  TORCH_CHECK(entry_offsets.scalar_type() == torch::kInt64, "entry_offsets must be int64");
  TORCH_CHECK(entry_m.scalar_type() == torch::kInt64, "entry_m must be int64");
  TORCH_CHECK(entry_channel.scalar_type() == torch::kInt64, "entry_channel must be int64");
  TORCH_CHECK(entry_d.scalar_type() == torch::kInt64, "entry_d must be int64");
  TORCH_CHECK(entry_l.scalar_type() == torch::kInt64, "entry_l must be int64");
  TORCH_CHECK(cout_prefix.numel() == static_cast<int64_t>(raws.size()) + 1,
              "cout_prefix must have raw_count + 1 entries");
  TORCH_CHECK(m_values.numel() == static_cast<int64_t>(raws.size()),
              "m_values must have raw_count entries");
  TORCH_CHECK(entry_offsets.numel() == out_dim + 1, "entry_offsets must be [out_dim + 1]");
  TORCH_CHECK(entry_m.numel() == entry_channel.numel() &&
              entry_m.numel() == entry_d.numel() &&
              entry_m.numel() == entry_l.numel(),
              "entry arrays must have the same length");
  TORCH_CHECK(out_dim > 0, "out_dim must be positive");
  if (rotate_out) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return scatter_raw_pairs_multi_output_major_forward_fp32_cuda(
      raws, wigner, offsets, compact_offsets, cout_prefix, m_values,
      entry_offsets, entry_m, entry_channel, entry_d, entry_l,
      out_dim, rotate_out, wigner_mode, wigner_stride);
}

torch::Tensor raw_pair_output_grad_fp32(
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
  return raw_pair_output_grad_fp32_cuda(
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

torch::Tensor scatter_pairs_multi_grad_fp32(
    torch::Tensor grad_packed,
    torch::Tensor wigner,
    torch::Tensor in_base_all,
    torch::Tensor in_l_all,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    torch::Tensor cin_prefix,
    torch::Tensor m_values,
    int64_t in_dim,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(grad_packed, "grad_packed");
  check_cuda_contiguous(in_base_all, "in_base_all");
  check_cuda_contiguous(in_l_all, "in_l_all");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  check_cuda_contiguous(cin_prefix, "cin_prefix");
  check_cuda_contiguous(m_values, "m_values");
  TORCH_CHECK(grad_packed.scalar_type() == torch::kFloat32, "grad_packed must be fp32");
  TORCH_CHECK(in_base_all.scalar_type() == torch::kInt64, "in_base_all must be int64");
  TORCH_CHECK(in_l_all.scalar_type() == torch::kInt64, "in_l_all must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(cin_prefix.scalar_type() == torch::kInt64, "cin_prefix must be int64");
  TORCH_CHECK(m_values.scalar_type() == torch::kInt64, "m_values must be int64");
  TORCH_CHECK(cin_prefix.numel() == m_values.numel() + 1,
              "cin_prefix must have m_count + 1 entries");
  TORCH_CHECK(in_base_all.numel() == in_l_all.numel(), "flat input maps must be aligned");
  if (rotate_in) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return scatter_pairs_multi_grad_fp32_cuda(
      grad_packed, wigner, in_base_all, in_l_all, offsets, compact_offsets,
      cin_prefix, m_values, in_dim, rotate_in, wigner_mode, wigner_stride);
}

std::vector<torch::Tensor> scatter_pair_grad_radial_input_fp32(
    torch::Tensor grad_pair_eff,
    torch::Tensor pair_no_radial,
    torch::Tensor radial,
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
  check_cuda_contiguous(grad_pair_eff, "grad_pair_eff");
  check_cuda_contiguous(pair_no_radial, "pair_no_radial");
  check_cuda_contiguous(radial, "radial");
  check_cuda_contiguous(in_base, "in_base");
  check_cuda_contiguous(in_l, "in_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  TORCH_CHECK(grad_pair_eff.scalar_type() == torch::kFloat32, "grad_pair_eff must be fp32");
  TORCH_CHECK(pair_no_radial.scalar_type() == torch::kFloat32, "pair_no_radial must be fp32");
  TORCH_CHECK(radial.scalar_type() == torch::kFloat32, "radial must be fp32");
  TORCH_CHECK(in_base.scalar_type() == torch::kInt64, "in_base must be int64");
  TORCH_CHECK(in_l.scalar_type() == torch::kInt64, "in_l must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(grad_pair_eff.dim() == 3 && grad_pair_eff.size(1) == 2,
              "grad_pair_eff must be [N, 2, Cin]");
  TORCH_CHECK(pair_no_radial.sizes() == grad_pair_eff.sizes(),
              "pair_no_radial shape must match grad_pair_eff");
  TORCH_CHECK(radial.dim() == 2 && radial.size(0) == grad_pair_eff.size(0) &&
              radial.size(1) == grad_pair_eff.size(2),
              "radial must be [N, Cin]");
  TORCH_CHECK(in_base.dim() == 1 && in_l.dim() == 1 && in_base.numel() == in_l.numel(),
              "input maps must be 1D and aligned");
  TORCH_CHECK(grad_pair_eff.size(2) == in_base.numel(), "grad_pair_eff Cin mismatch");
  if (rotate_in) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return scatter_pair_grad_radial_input_fp32_cuda(
      grad_pair_eff, pair_no_radial, radial, wigner,
      in_base, in_l, offsets, compact_offsets,
      in_dim, m, rotate_in, wigner_mode, wigner_stride);
}

torch::Tensor pack_m0_fp32(
    torch::Tensor x,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
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
  return pack_m0_fp32_cuda(
      x, wigner, in_base, in_l, offsets, compact_offsets,
      rotate_in, wigner_mode, wigner_stride);
}

torch::Tensor output_m0_grad_fp32(
    torch::Tensor grad_out,
    torch::Tensor wigner,
    torch::Tensor out_base,
    torch::Tensor out_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
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
  return output_m0_grad_fp32_cuda(
      grad_out, wigner, out_base, out_l, offsets, compact_offsets,
      rotate_out, wigner_mode, wigner_stride);
}

torch::Tensor scatter_m0_grad_fp32(
    torch::Tensor grad_m0,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(grad_m0, "grad_m0");
  check_cuda_contiguous(in_base, "in_base");
  check_cuda_contiguous(in_l, "in_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  TORCH_CHECK(grad_m0.scalar_type() == torch::kFloat32, "grad_m0 must be fp32");
  TORCH_CHECK(in_base.scalar_type() == torch::kInt64, "in_base must be int64");
  TORCH_CHECK(in_l.scalar_type() == torch::kInt64, "in_l must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(grad_m0.dim() == 2, "grad_m0 must be [N, Cin]");
  TORCH_CHECK(in_base.dim() == 1 && in_l.dim() == 1 && in_base.numel() == in_l.numel(),
              "input maps must be 1D and aligned");
  TORCH_CHECK(grad_m0.size(1) == in_base.numel(), "grad_m0 Cin mismatch");
  if (rotate_in) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return scatter_m0_grad_fp32_cuda(
      grad_m0, wigner, in_base, in_l, offsets, compact_offsets,
      in_dim, rotate_in, wigner_mode, wigner_stride);
}

std::vector<torch::Tensor> scatter_m0_grad_radial_input_fp32(
    torch::Tensor grad_eff,
    torch::Tensor m0_no_radial,
    torch::Tensor radial,
    torch::Tensor wigner,
    torch::Tensor in_base,
    torch::Tensor in_l,
    torch::Tensor offsets,
    torch::Tensor compact_offsets,
    int64_t in_dim,
    bool rotate_in,
    int64_t wigner_mode,
    int64_t wigner_stride) {
  check_cuda_contiguous(grad_eff, "grad_eff");
  check_cuda_contiguous(m0_no_radial, "m0_no_radial");
  check_cuda_contiguous(radial, "radial");
  check_cuda_contiguous(in_base, "in_base");
  check_cuda_contiguous(in_l, "in_l");
  check_cuda_contiguous(offsets, "offsets");
  check_cuda_contiguous(compact_offsets, "compact_offsets");
  TORCH_CHECK(grad_eff.scalar_type() == torch::kFloat32, "grad_eff must be fp32");
  TORCH_CHECK(m0_no_radial.scalar_type() == torch::kFloat32, "m0_no_radial must be fp32");
  TORCH_CHECK(radial.scalar_type() == torch::kFloat32, "radial must be fp32");
  TORCH_CHECK(in_base.scalar_type() == torch::kInt64, "in_base must be int64");
  TORCH_CHECK(in_l.scalar_type() == torch::kInt64, "in_l must be int64");
  TORCH_CHECK(offsets.scalar_type() == torch::kInt64, "offsets must be int64");
  TORCH_CHECK(compact_offsets.scalar_type() == torch::kInt64, "compact_offsets must be int64");
  TORCH_CHECK(grad_eff.dim() == 2, "grad_eff must be [N, Cin]");
  TORCH_CHECK(m0_no_radial.sizes() == grad_eff.sizes(), "m0_no_radial shape must match grad_eff");
  TORCH_CHECK(radial.sizes() == grad_eff.sizes(), "radial shape must match grad_eff");
  TORCH_CHECK(in_base.dim() == 1 && in_l.dim() == 1 && in_base.numel() == in_l.numel(),
              "input maps must be 1D and aligned");
  TORCH_CHECK(grad_eff.size(1) == in_base.numel(), "grad_eff Cin mismatch");
  if (rotate_in) {
    check_cuda_contiguous(wigner, "wigner");
    TORCH_CHECK(wigner.scalar_type() == torch::kFloat32, "wigner must be fp32");
  }
  return scatter_m0_grad_radial_input_fp32_cuda(
      grad_eff, m0_no_radial, radial, wigner,
      in_base, in_l, offsets, compact_offsets,
      in_dim, rotate_in, wigner_mode, wigner_stride);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_pair_forward_fp32", &fused_pair_forward_fp32, "SO2 MoE fused P0 pair forward fp32");
  m.def("fused_m0_forward_fp32", &fused_m0_forward_fp32, "SO2 MoE fused P0 m0 forward fp32");
  m.def("fused_pair_backward_fp32", &fused_pair_backward_fp32, "SO2 MoE fused P0 pair backward fp32");
  m.def("pack_m0_fp32", &pack_m0_fp32, "SO2 MoE fused P0 pack m0 fp32");
  m.def("output_m0_grad_fp32", &output_m0_grad_fp32, "SO2 MoE fused P0 output m0 grad fp32");
  m.def("scatter_m0_grad_fp32", &scatter_m0_grad_fp32, "SO2 MoE fused P0 scatter m0 grad fp32");
  m.def("scatter_m0_grad_radial_input_fp32", &scatter_m0_grad_radial_input_fp32, "SO2 MoE fused P0 radial-input scatter m0 grad fp32");
  m.def("pack_pair_fp32", &pack_pair_fp32, "SO2 MoE fused P0 pack pair fp32");
  m.def("pack_pairs_multi_fp32", &pack_pairs_multi_fp32, "SO2 MoE fused P0 grouped pack pair fp32");
  m.def("output_pair_grad_fp32", &output_pair_grad_fp32, "SO2 MoE fused P0 output pair grad fp32");
  m.def("scatter_pair_forward_fp32", &scatter_pair_forward_fp32, "SO2 MoE fused P0 scatter pair forward fp32");
  m.def("scatter_raw_pair_forward_fp32", &scatter_raw_pair_forward_fp32, "SO2 MoE fused P0 scatter raw pair forward fp32");
  m.def("scatter_raw_pairs_multi_forward_fp32", &scatter_raw_pairs_multi_forward_fp32, "SO2 MoE fused P0 grouped raw pair epilogue forward fp32");
  m.def("scatter_raw_pairs_multi_output_major_forward_fp32", &scatter_raw_pairs_multi_output_major_forward_fp32, "SO2 MoE fused P0 output-major grouped raw pair epilogue forward fp32");
  m.def("raw_pair_output_grad_fp32", &raw_pair_output_grad_fp32, "SO2 MoE fused P0 raw pair output grad fp32");
  m.def("scatter_pair_grad_fp32", &scatter_pair_grad_fp32, "SO2 MoE fused P0 scatter pair grad fp32");
  m.def("scatter_pairs_multi_grad_fp32", &scatter_pairs_multi_grad_fp32, "SO2 MoE fused P0 grouped scatter pair grad fp32");
  m.def("scatter_pair_grad_radial_input_fp32", &scatter_pair_grad_radial_input_fp32, "SO2 MoE fused P0 radial-input scatter pair grad fp32");
#ifdef DPTB_SO2_MOE_FUSED_P0_CUTLASS
  m.def("cutlass_cute_probe", &cutlass_cute_probe_cuda, "Compile-only CuTe dot-mainloop probe");
  m.def("fused_pair_forward_tiled2_fp32", &fused_pair_forward_tiled2_fp32, "CuTe-indexed tiled SO2 MoE fused P0 pair forward fp32, tile2");
  m.def("fused_pair_forward_tiled3_fp32", &fused_pair_forward_tiled3_fp32, "CuTe-indexed tiled SO2 MoE fused P0 pair forward fp32, tile3");
  m.def("fused_pair_forward_tiled4_fp32", &fused_pair_forward_tiled4_fp32, "CuTe-indexed tiled SO2 MoE fused P0 pair forward fp32, tile4");
  m.def("fused_pair_forward_tiled8_fp32", &fused_pair_forward_tiled8_fp32, "CuTe-indexed tiled SO2 MoE fused P0 pair forward fp32, tile8");
#endif
}
