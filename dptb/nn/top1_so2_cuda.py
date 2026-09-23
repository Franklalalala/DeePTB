"""Reused SO2CUDA pack/scatter for the streamed SO2 route of routed MoLE layers.

Derived from the qualified 20260912 activation CUDA adapter; no new kernels.
The Wigner rotation and m-packing run on SO2CUDA's pack/scatter autograd
kernels while the expert linear stays in MOLELinear.forward: the independent
top-1 branch (Switch) and activation-space prior_activate (per-edge top-k
mixing of expert outputs) both use it.  No parameter-space mixing and no
per-edge weights are built.
"""
import os
import torch

CALLS = 0
ACTIVATION_FALLBACKS = 0
ACTIVATION_LAST_ERROR = None
_ACTIVATION_DISABLED = False
_TOP1_IMPORT_WARNED = False

def cuda_forward(module, x, R, mole_globals, latents=None, wigner_D_all=None, *, route):
    global CALLS
    from so2_cuda_ops import tensor_product as ops
    if x.device.type != 'cuda' or x.dtype != torch.float32:
        raise RuntimeError('activation CUDA pack/scatter requires CUDA float32')
    if torch.is_tensor(R) and R.requires_grad:
        raise RuntimeError('activation CUDA pack/scatter requires fixed geometry')
    wigner_D_all = module._ensure_wigner_rotation(R, wigner_D_all)
    if ops._wigner_requires_grad(wigner_D_all):
        raise RuntimeError('activation CUDA pack/scatter does not differentiate Wigner matrices')
    info = ops._wigner_tensor_and_mode(module, wigner_D_all, x)
    if info is None:
        raise RuntimeError('unsupported Wigner layout for activation CUDA pack/scatter')
    wigner, compact_offsets, mode, stride = info
    weights = module.radial_emb(latents) if module.radial_emb else None
    out = None
    for m in range(module.m_max + 1):
        ib, il, ob, ol, offsets = ops._pair_maps(module, m, x.device)
        common = (wigner, ib, il, offsets, compact_offsets)
        if m == 0:
            inp = ops._PackM0Function.apply(x.contiguous(), *common, module.rotate_in, mode, stride)
        else:
            inp = ops._PackPairFunction.apply(x.contiguous(), *common, m, module.rotate_in, mode, stride)
        radial = None
        if weights is not None:
            radial = weights[:, module.m_in_index[m]:module.m_in_index[m + 1]]
            if m: radial = radial.unsqueeze(1)
            if module.front: inp = inp * radial
        y = module.fc_m0(inp, mole_globals) if m == 0 else module.m_linear[m-1](inp, mole_globals)
        if radial is not None and not module.front: y = y * radial
        common_out = (wigner, ob, ol, offsets, compact_offsets, module.irreps_out.dim)
        if m == 0:
            part = ops._ScatterM0OutputFunction.apply(y.contiguous(), *common_out, module.rotate_out, mode, stride)
        else:
            part = ops._ScatterPairOutputFunction.apply(y.contiguous(), *common_out, m, module.rotate_out, mode, stride)
        out = part if out is None else out + part
    CALLS += 1
    if CALLS == 1:
        print('SO2_ACTIVATION_CUDA_ACTIVE pid=%s edges=%s m_max=%s mode=%s' % (os.getpid(),x.shape[0],module.m_max,mode),flush=True)
    return out.contiguous(), wigner_D_all


def try_forward(module, x, R, mole_globals, latents=None, wigner_D_all=None, *, route):
    if getattr(mole_globals, "top1_reference_so2", False):
        return None
    if x.device.type != 'cuda' or x.dtype != torch.float32:
        return None
    if torch.is_tensor(R) and R.requires_grad:
        return None
    if torch.is_autocast_enabled():
        return None
    # This function is also the Switch fallback when fused-P0 is unavailable.
    # The optional pack/scatter package must not become mandatory here.
    global _TOP1_IMPORT_WARNED
    try:
        from so2_cuda_ops import tensor_product as ops
    except ImportError as exc:
        if not _TOP1_IMPORT_WARNED:
            _TOP1_IMPORT_WARNED = True
            print("SO2_TOP1_CUDA_UNAVAILABLE (streamed route): %s" % repr(exc)[:800], flush=True)
        return None
    wigner = module._ensure_wigner_rotation(R, wigner_D_all)
    if ops._wigner_requires_grad(wigner) or ops._wigner_tensor_and_mode(module, wigner, x) is None:
        return None
    return cuda_forward(module, x, R, mole_globals, latents, wigner, route=route)


def activation_route_enabled():
    """DPTB_SO2_ACTIVATION_CUDA=0 keeps prior_activate on the streamed SO2 route."""
    return os.environ.get("DPTB_SO2_ACTIVATION_CUDA", "1").strip().lower() not in ("0", "false", "off", "no")


def try_activation_forward(module, x, R, mole_globals, latents=None, wigner_D_all=None, *, route):
    """prior_activate (activation-space MoLE) through the same pack/scatter kernels.

    Returns None when the route does not apply (disabled, CPU, float64,
    differentiable geometry, unsupported Wigner layout, SO2CUDA not
    importable); the caller then runs the streamed route. Unexpected runtime
    failures propagate. Unsupported Wigner inputs are handled by try_forward's
    preflight and do not disable later compatible calls.
    """
    global ACTIVATION_FALLBACKS, ACTIVATION_LAST_ERROR, _ACTIVATION_DISABLED
    if _ACTIVATION_DISABLED or not activation_route_enabled():
        return None
    if x.device.type != 'cuda' or x.dtype != torch.float32:
        return None
    try:
        import so2_cuda_ops  # noqa: F401
    except ImportError as exc:
        _ACTIVATION_DISABLED = True
        ACTIVATION_LAST_ERROR = repr(exc)[:800]
        print('SO2_ACTIVATION_CUDA_UNAVAILABLE (streamed route): %s' % ACTIVATION_LAST_ERROR, flush=True)
        return None
    # Do not turn OOM, illegal memory access, or a programming error into a
    # second execution attempt. CUDA errors can surface asynchronously.
    return try_forward(module, x, R, mole_globals, latents, wigner_D_all, route=route)
