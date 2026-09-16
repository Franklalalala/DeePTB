"""Reused SO2CUDA pack/scatter, scoped to the independent top-1 branch.

Derived from the qualified 20260912 activation CUDA adapter; no new kernels.
"""
import os
import torch

CALLS = 0

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
    from so2_cuda_ops import tensor_product as ops
    wigner = module._ensure_wigner_rotation(R, wigner_D_all)
    if ops._wigner_requires_grad(wigner) or ops._wigner_tensor_and_mode(module, wigner, x) is None:
        return None
    return cuda_forward(module, x, R, mole_globals, latents, wigner, route=route)
