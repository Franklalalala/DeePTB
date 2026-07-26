"""DeePTB adapter for optional SO2 CUDA tensor-product kernels.

The optimized implementation lives in the separately packaged
``so2_cuda_ops`` project.  Importing DeePTB must still work on CPU-only
installations, where the regular PyTorch/cuEquivariance route remains the
fallback.
"""

from __future__ import annotations

from dptb.nn.cuda_ops.segments import repeated_segment_layout

try:
    from so2_cuda_ops import tensor_product as _backend
except ModuleNotFoundError as exc:
    _BACKEND_IMPORT_ERROR = exc

    def _pair_segment_layout(graph_index, num_routes):
        return repeated_segment_layout(
            graph_index,
            num_routes,
            repeat=2,
            cache_name="so2_moe_fused_p0_pair",
        ).as_legacy_tuple()

    def _load_extension():
        raise RuntimeError(
            "The fused SO2 CUDA route requires the optional so2-cuda-ops "
            "package. Install it before enabling streamed_m_major_fused_p0."
        ) from _BACKEND_IMPORT_ERROR

    def try_forward_so2_moe_fused_p0(*args, **kwargs):
        """Decline the optional fused route so the maintained fallback runs."""

        return None
else:
    globals().update(
        {
            name: getattr(_backend, name)
            for name in dir(_backend)
            if not name.startswith("__")
        }
    )
