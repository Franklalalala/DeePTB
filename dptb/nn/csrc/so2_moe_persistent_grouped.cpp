#include <torch/extension.h>

#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Float, #x " must be float32")
#define CHECK_LONG(x) TORCH_CHECK((x).scalar_type() == at::ScalarType::Long, #x " must be int64")
#define CHECK_CUDA_CONTIGUOUS(x) \
  do {                         \
    CHECK_CUDA(x);             \
    CHECK_CONTIGUOUS(x);       \
  } while (false)

at::Tensor persistent_grouped_forward_fp32_cuda(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks);

at::Tensor persistent_grouped_forward_warp_fp32_cuda(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks);

at::Tensor persistent_grouped_forward_cute_tiled_fp32_cuda(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks);

at::Tensor persistent_grouped_forward_cutlass_native_fp32_cuda(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks);

at::Tensor persistent_grouped_forward_fp32(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks) {
  CHECK_CUDA_CONTIGUOUS(x);
  CHECK_FLOAT(x);
  TORCH_CHECK(x.dim() == 2, "x must be [E, in_dim]");

  CHECK_CUDA_CONTIGUOUS(edge_order);
  CHECK_CUDA_CONTIGUOUS(route_ptr);
  CHECK_CUDA_CONTIGUOUS(problem_tile_prefix);
  CHECK_CUDA_CONTIGUOUS(weight_flat);
  CHECK_CUDA_CONTIGUOUS(weight_offsets);
  CHECK_CUDA_CONTIGUOUS(bias_flat);
  CHECK_CUDA_CONTIGUOUS(bias_offsets);
  CHECK_CUDA_CONTIGUOUS(m_values);
  CHECK_CUDA_CONTIGUOUS(in_ptr);
  CHECK_CUDA_CONTIGUOUS(in_base);
  CHECK_CUDA_CONTIGUOUS(in_l);
  CHECK_CUDA_CONTIGUOUS(out_ptr);
  CHECK_CUDA_CONTIGUOUS(out_base);
  CHECK_CUDA_CONTIGUOUS(out_l);
  CHECK_CUDA_CONTIGUOUS(offsets);
  CHECK_CUDA_CONTIGUOUS(compact_offsets);
  CHECK_CUDA_CONTIGUOUS(radial_all);
  CHECK_CUDA_CONTIGUOUS(m_in_index);

  CHECK_LONG(edge_order);
  CHECK_LONG(route_ptr);
  CHECK_LONG(problem_tile_prefix);
  CHECK_LONG(weight_offsets);
  CHECK_LONG(bias_offsets);
  CHECK_LONG(m_values);
  CHECK_LONG(in_ptr);
  CHECK_LONG(in_base);
  CHECK_LONG(in_l);
  CHECK_LONG(out_ptr);
  CHECK_LONG(out_base);
  CHECK_LONG(out_l);
  CHECK_LONG(offsets);
  CHECK_LONG(compact_offsets);
  if (m_in_index.numel() > 0) CHECK_LONG(m_in_index);

  CHECK_FLOAT(weight_flat);
  if (bias_flat.numel() > 0) CHECK_FLOAT(bias_flat);
  if (wigner.numel() > 0) {
    CHECK_CUDA_CONTIGUOUS(wigner);
    CHECK_FLOAT(wigner);
  }
  if (radial_all.numel() > 0) {
    CHECK_FLOAT(radial_all);
    TORCH_CHECK(radial_all.dim() == 2, "radial_all must be [E, radial_dim] when non-empty");
  }

  TORCH_CHECK(out_dim >= 0, "out_dim must be non-negative");
  TORCH_CHECK(n_routes >= 0, "n_routes must be non-negative");
  TORCH_CHECK(block_m > 0 && block_n > 0, "block_m/block_n must be positive");
  TORCH_CHECK(wigner_mode >= 0 && wigner_mode <= 2, "wigner_mode must be 0, 1, or 2");
  TORCH_CHECK(route_ptr.numel() == n_routes + 1, "route_ptr length must be n_routes + 1");
  TORCH_CHECK(m_values.numel() + 1 == in_ptr.numel(), "in_ptr length must be m_count + 1");
  TORCH_CHECK(m_values.numel() + 1 == out_ptr.numel(), "out_ptr length must be m_count + 1");
  TORCH_CHECK(weight_offsets.numel() == m_values.numel(), "weight_offsets length must be m_count");
  TORCH_CHECK(bias_offsets.numel() == m_values.numel(), "bias_offsets length must be m_count");
  TORCH_CHECK(problem_tile_prefix.numel() == n_routes * m_values.numel() + 1,
              "problem_tile_prefix length must be n_routes * m_count + 1");

  return persistent_grouped_forward_fp32_cuda(
      x, wigner, edge_order, route_ptr, problem_tile_prefix, weight_flat,
      weight_offsets, bias_flat, bias_offsets, m_values, in_ptr, in_base, in_l, out_ptr, out_base,
      out_l, offsets, compact_offsets, radial_all, m_in_index, out_dim,
      n_routes, rotate_in, rotate_out, radial_on_input, wigner_mode,
      wigner_stride, block_m, block_n, active_blocks);
}

at::Tensor persistent_grouped_forward_warp_fp32(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks) {
  CHECK_CUDA_CONTIGUOUS(x);
  CHECK_FLOAT(x);
  TORCH_CHECK(x.dim() == 2, "x must be [E, in_dim]");

  CHECK_CUDA_CONTIGUOUS(edge_order);
  CHECK_CUDA_CONTIGUOUS(route_ptr);
  CHECK_CUDA_CONTIGUOUS(problem_tile_prefix);
  CHECK_CUDA_CONTIGUOUS(weight_flat);
  CHECK_CUDA_CONTIGUOUS(weight_offsets);
  CHECK_CUDA_CONTIGUOUS(bias_flat);
  CHECK_CUDA_CONTIGUOUS(bias_offsets);
  CHECK_CUDA_CONTIGUOUS(m_values);
  CHECK_CUDA_CONTIGUOUS(in_ptr);
  CHECK_CUDA_CONTIGUOUS(in_base);
  CHECK_CUDA_CONTIGUOUS(in_l);
  CHECK_CUDA_CONTIGUOUS(out_ptr);
  CHECK_CUDA_CONTIGUOUS(out_base);
  CHECK_CUDA_CONTIGUOUS(out_l);
  CHECK_CUDA_CONTIGUOUS(offsets);
  CHECK_CUDA_CONTIGUOUS(compact_offsets);
  CHECK_CUDA_CONTIGUOUS(radial_all);
  CHECK_CUDA_CONTIGUOUS(m_in_index);

  CHECK_LONG(edge_order);
  CHECK_LONG(route_ptr);
  CHECK_LONG(problem_tile_prefix);
  CHECK_LONG(weight_offsets);
  CHECK_LONG(bias_offsets);
  CHECK_LONG(m_values);
  CHECK_LONG(in_ptr);
  CHECK_LONG(in_base);
  CHECK_LONG(in_l);
  CHECK_LONG(out_ptr);
  CHECK_LONG(out_base);
  CHECK_LONG(out_l);
  CHECK_LONG(offsets);
  CHECK_LONG(compact_offsets);
  if (m_in_index.numel() > 0) CHECK_LONG(m_in_index);

  CHECK_FLOAT(weight_flat);
  if (bias_flat.numel() > 0) CHECK_FLOAT(bias_flat);
  if (wigner.numel() > 0) {
    CHECK_CUDA_CONTIGUOUS(wigner);
    CHECK_FLOAT(wigner);
  }
  if (radial_all.numel() > 0) {
    CHECK_FLOAT(radial_all);
    TORCH_CHECK(radial_all.dim() == 2, "radial_all must be [E, radial_dim] when non-empty");
  }

  TORCH_CHECK(out_dim >= 0, "out_dim must be non-negative");
  TORCH_CHECK(n_routes >= 0, "n_routes must be non-negative");
  TORCH_CHECK(block_m > 0 && block_n > 0, "block_m/block_n must be positive");
  TORCH_CHECK(block_n <= 16, "warp collective block_n must be <= 16");
  TORCH_CHECK(wigner_mode >= 0 && wigner_mode <= 2, "wigner_mode must be 0, 1, or 2");
  TORCH_CHECK(route_ptr.numel() == n_routes + 1, "route_ptr length must be n_routes + 1");
  TORCH_CHECK(m_values.numel() + 1 == in_ptr.numel(), "in_ptr length must be m_count + 1");
  TORCH_CHECK(m_values.numel() + 1 == out_ptr.numel(), "out_ptr length must be m_count + 1");
  TORCH_CHECK(weight_offsets.numel() == m_values.numel(), "weight_offsets length must be m_count");
  TORCH_CHECK(bias_offsets.numel() == m_values.numel(), "bias_offsets length must be m_count");
  TORCH_CHECK(problem_tile_prefix.numel() == n_routes * m_values.numel() + 1,
              "problem_tile_prefix length must be n_routes * m_count + 1");

  return persistent_grouped_forward_warp_fp32_cuda(
      x, wigner, edge_order, route_ptr, problem_tile_prefix, weight_flat,
      weight_offsets, bias_flat, bias_offsets, m_values, in_ptr, in_base, in_l, out_ptr, out_base,
      out_l, offsets, compact_offsets, radial_all, m_in_index, out_dim,
      n_routes, rotate_in, rotate_out, radial_on_input, wigner_mode,
      wigner_stride, block_m, block_n, active_blocks);
}

at::Tensor persistent_grouped_forward_cute_tiled_fp32(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks) {
  CHECK_CUDA_CONTIGUOUS(x);
  CHECK_FLOAT(x);
  TORCH_CHECK(x.dim() == 2, "x must be [E, in_dim]");

  CHECK_CUDA_CONTIGUOUS(edge_order);
  CHECK_CUDA_CONTIGUOUS(route_ptr);
  CHECK_CUDA_CONTIGUOUS(problem_tile_prefix);
  CHECK_CUDA_CONTIGUOUS(weight_flat);
  CHECK_CUDA_CONTIGUOUS(weight_offsets);
  CHECK_CUDA_CONTIGUOUS(bias_flat);
  CHECK_CUDA_CONTIGUOUS(bias_offsets);
  CHECK_CUDA_CONTIGUOUS(m_values);
  CHECK_CUDA_CONTIGUOUS(in_ptr);
  CHECK_CUDA_CONTIGUOUS(in_base);
  CHECK_CUDA_CONTIGUOUS(in_l);
  CHECK_CUDA_CONTIGUOUS(out_ptr);
  CHECK_CUDA_CONTIGUOUS(out_base);
  CHECK_CUDA_CONTIGUOUS(out_l);
  CHECK_CUDA_CONTIGUOUS(offsets);
  CHECK_CUDA_CONTIGUOUS(compact_offsets);
  CHECK_CUDA_CONTIGUOUS(radial_all);
  CHECK_CUDA_CONTIGUOUS(m_in_index);

  CHECK_LONG(edge_order);
  CHECK_LONG(route_ptr);
  CHECK_LONG(problem_tile_prefix);
  CHECK_LONG(weight_offsets);
  CHECK_LONG(bias_offsets);
  CHECK_LONG(m_values);
  CHECK_LONG(in_ptr);
  CHECK_LONG(in_base);
  CHECK_LONG(in_l);
  CHECK_LONG(out_ptr);
  CHECK_LONG(out_base);
  CHECK_LONG(out_l);
  CHECK_LONG(offsets);
  CHECK_LONG(compact_offsets);
  if (m_in_index.numel() > 0) CHECK_LONG(m_in_index);

  CHECK_FLOAT(weight_flat);
  if (bias_flat.numel() > 0) CHECK_FLOAT(bias_flat);
  if (wigner.numel() > 0) {
    CHECK_CUDA_CONTIGUOUS(wigner);
    CHECK_FLOAT(wigner);
  }
  if (radial_all.numel() > 0) {
    CHECK_FLOAT(radial_all);
    TORCH_CHECK(radial_all.dim() == 2, "radial_all must be [E, radial_dim] when non-empty");
  }

  TORCH_CHECK(out_dim >= 0, "out_dim must be non-negative");
  TORCH_CHECK(n_routes >= 0, "n_routes must be non-negative");
  TORCH_CHECK(block_m > 0 && block_n > 0, "block_m/block_n must be positive");
  TORCH_CHECK(block_m <= 16, "cute_tiled block_m must be <= 16");
  TORCH_CHECK(block_n <= 32, "cute_tiled block_n must be <= 32");
  TORCH_CHECK(block_m * block_n <= 256, "cute_tiled block_m * block_n must be <= 256");
  TORCH_CHECK(wigner_mode >= 0 && wigner_mode <= 2, "wigner_mode must be 0, 1, or 2");
  TORCH_CHECK(route_ptr.numel() == n_routes + 1, "route_ptr length must be n_routes + 1");
  TORCH_CHECK(m_values.numel() + 1 == in_ptr.numel(), "in_ptr length must be m_count + 1");
  TORCH_CHECK(m_values.numel() + 1 == out_ptr.numel(), "out_ptr length must be m_count + 1");
  TORCH_CHECK(weight_offsets.numel() == m_values.numel(), "weight_offsets length must be m_count");
  TORCH_CHECK(bias_offsets.numel() == m_values.numel(), "bias_offsets length must be m_count");
  TORCH_CHECK(problem_tile_prefix.numel() == n_routes * m_values.numel() + 1,
              "problem_tile_prefix length must be n_routes * m_count + 1");

  return persistent_grouped_forward_cute_tiled_fp32_cuda(
      x, wigner, edge_order, route_ptr, problem_tile_prefix, weight_flat,
      weight_offsets, bias_flat, bias_offsets, m_values, in_ptr, in_base, in_l, out_ptr, out_base,
      out_l, offsets, compact_offsets, radial_all, m_in_index, out_dim,
      n_routes, rotate_in, rotate_out, radial_on_input, wigner_mode,
      wigner_stride, block_m, block_n, active_blocks);
}

at::Tensor persistent_grouped_forward_cutlass_native_fp32(
    const at::Tensor& x,
    const at::Tensor& wigner,
    const at::Tensor& edge_order,
    const at::Tensor& route_ptr,
    const at::Tensor& problem_tile_prefix,
    const at::Tensor& weight_flat,
    const at::Tensor& weight_offsets,
    const at::Tensor& bias_flat,
    const at::Tensor& bias_offsets,
    const at::Tensor& m_values,
    const at::Tensor& in_ptr,
    const at::Tensor& in_base,
    const at::Tensor& in_l,
    const at::Tensor& out_ptr,
    const at::Tensor& out_base,
    const at::Tensor& out_l,
    const at::Tensor& offsets,
    const at::Tensor& compact_offsets,
    const at::Tensor& radial_all,
    const at::Tensor& m_in_index,
    int64_t out_dim,
    int64_t n_routes,
    bool rotate_in,
    bool rotate_out,
    bool radial_on_input,
    int64_t wigner_mode,
    int64_t wigner_stride,
    int64_t block_m,
    int64_t block_n,
    int64_t active_blocks) {
  CHECK_CUDA_CONTIGUOUS(x);
  CHECK_FLOAT(x);
  TORCH_CHECK(x.dim() == 2, "x must be [E, in_dim]");

  CHECK_CUDA_CONTIGUOUS(edge_order);
  CHECK_CUDA_CONTIGUOUS(route_ptr);
  CHECK_CUDA_CONTIGUOUS(problem_tile_prefix);
  CHECK_CUDA_CONTIGUOUS(weight_flat);
  CHECK_CUDA_CONTIGUOUS(weight_offsets);
  CHECK_CUDA_CONTIGUOUS(bias_flat);
  CHECK_CUDA_CONTIGUOUS(bias_offsets);
  CHECK_CUDA_CONTIGUOUS(m_values);
  CHECK_CUDA_CONTIGUOUS(in_ptr);
  CHECK_CUDA_CONTIGUOUS(in_base);
  CHECK_CUDA_CONTIGUOUS(in_l);
  CHECK_CUDA_CONTIGUOUS(out_ptr);
  CHECK_CUDA_CONTIGUOUS(out_base);
  CHECK_CUDA_CONTIGUOUS(out_l);
  CHECK_CUDA_CONTIGUOUS(offsets);
  CHECK_CUDA_CONTIGUOUS(compact_offsets);
  CHECK_CUDA_CONTIGUOUS(radial_all);
  CHECK_CUDA_CONTIGUOUS(m_in_index);

  CHECK_LONG(edge_order);
  CHECK_LONG(route_ptr);
  CHECK_LONG(problem_tile_prefix);
  CHECK_LONG(weight_offsets);
  CHECK_LONG(bias_offsets);
  CHECK_LONG(m_values);
  CHECK_LONG(in_ptr);
  CHECK_LONG(in_base);
  CHECK_LONG(in_l);
  CHECK_LONG(out_ptr);
  CHECK_LONG(out_base);
  CHECK_LONG(out_l);
  CHECK_LONG(offsets);
  CHECK_LONG(compact_offsets);
  if (m_in_index.numel() > 0) CHECK_LONG(m_in_index);

  CHECK_FLOAT(weight_flat);
  if (bias_flat.numel() > 0) CHECK_FLOAT(bias_flat);
  if (wigner.numel() > 0) {
    CHECK_CUDA_CONTIGUOUS(wigner);
    CHECK_FLOAT(wigner);
  }
  if (radial_all.numel() > 0) {
    CHECK_FLOAT(radial_all);
    TORCH_CHECK(radial_all.dim() == 2, "radial_all must be [E, radial_dim] when non-empty");
  }

  TORCH_CHECK(out_dim >= 0, "out_dim must be non-negative");
  TORCH_CHECK(n_routes >= 0, "n_routes must be non-negative");
  TORCH_CHECK(block_m == 64 && block_n == 32, "cutlass_native currently uses fixed 64x32 raw GEMM tiles");
  TORCH_CHECK(wigner_mode >= 0 && wigner_mode <= 2, "wigner_mode must be 0, 1, or 2");
  TORCH_CHECK(route_ptr.numel() == n_routes + 1, "route_ptr length must be n_routes + 1");
  TORCH_CHECK(m_values.numel() + 1 == in_ptr.numel(), "in_ptr length must be m_count + 1");
  TORCH_CHECK(m_values.numel() + 1 == out_ptr.numel(), "out_ptr length must be m_count + 1");
  TORCH_CHECK(weight_offsets.numel() == m_values.numel(), "weight_offsets length must be m_count");
  TORCH_CHECK(bias_offsets.numel() == m_values.numel(), "bias_offsets length must be m_count");
  TORCH_CHECK(problem_tile_prefix.numel() == n_routes * m_values.numel() + 1,
              "problem_tile_prefix length must be n_routes * m_count + 1");

  if (m_values.numel() > 0) {
    TORCH_CHECK(m_values.min().item<int64_t>() > 0, "cutlass_native currently expects m>0 problems only");
  }

  return persistent_grouped_forward_cutlass_native_fp32_cuda(
      x, wigner, edge_order, route_ptr, problem_tile_prefix, weight_flat,
      weight_offsets, bias_flat, bias_offsets, m_values, in_ptr, in_base, in_l, out_ptr, out_base,
      out_l, offsets, compact_offsets, radial_all, m_in_index, out_dim,
      n_routes, rotate_in, rotate_out, radial_on_input, wigner_mode,
      wigner_stride, block_m, block_n, active_blocks);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "persistent_grouped_forward_fp32",
      &persistent_grouped_forward_fp32,
      "Persistent grouped SO2/MoE fused forward P1 (CUDA fp32)");
  m.def(
      "persistent_grouped_forward_warp_fp32",
      &persistent_grouped_forward_warp_fp32,
      "Persistent grouped SO2/MoE fused forward P1 warp-collective mainloop (CUDA fp32)");
  m.def(
      "persistent_grouped_forward_cute_tiled_fp32",
      &persistent_grouped_forward_cute_tiled_fp32,
      "Persistent grouped SO2/MoE fused forward P1 CuTe tiled custom A-loader mainloop (CUDA fp32)");
  m.def(
      "persistent_grouped_forward_cutlass_native_fp32",
      &persistent_grouped_forward_cutlass_native_fp32,
      "Persistent grouped SO2/MoE fused forward P1 CUTLASS native custom A-loader/epilogue mainloop (CUDA fp32)");
}
