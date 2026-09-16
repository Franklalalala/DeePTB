"""Reusable, chemistry-independent periodic type-1 transforms.

S[k] = sum_j weights[j] exp(-2*pi*i*k.frac[j]), NumPy FFT mode order.
Production engines: FINUFFT (CPU), cuFINUFFT (CUDA, native torch tensors).
The explicit torch_gaussian engine is a portable reference/backup; it retains
our previous oversampling=3 Gaussian algorithm, not FINUFFT's tuned kernel.
No automatic engine substitution, global geometry cache, autograd, or dense
N-by-G fallback. A plan owns a coordinate snapshot; construct a new one when
geometry changes. Plans are sequential-use only (not thread safe).
"""
from __future__ import annotations

from contextlib import nullcontext
from importlib import import_module
from math import prod
import warnings

import numpy as np
from scipy.fft import next_fast_len

from .structure_factor import gaussian_nufft_parameters

BACKENDS = frozenset({"finufft", "cufinufft", "torch_gaussian"})


def _optional(name: str):
    try:
        return import_module(name)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"Requested backend requires an importable {name!r}. Install its official "
            "wheel/build matching your platform. No silent backend fallback is allowed."
        ) from exc


def _shape3(shape):
    shape = tuple(shape)
    if len(shape) != 3 or any(isinstance(n, (bool, np.bool_)) or
        not isinstance(n, (int, np.integer)) or n <= 0 for n in shape):
        raise ValueError("shape must contain three positive integers")
    return tuple(int(n) for n in shape)


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _budget(estimate, max_work_mb):
    if max_work_mb is not None:
        if not np.isfinite(max_work_mb) or max_work_mb <= 0:
            raise ValueError("max_work_mb must be finite and positive or None")
        if estimate > float(max_work_mb) * 1024**2:
            raise MemoryError(
                f"NUFFT working-set estimate {estimate/1024**2:.2f} MiB exceeds "
                f"max_work_mb={max_work_mb}; no fallback or precision reduction."
            )


def resolve_torch_device(device):
    torch = _optional("torch")
    dev = torch.device(device)
    if dev.type not in {"cpu", "cuda"}:
        raise ValueError("Only CPU and CUDA devices are supported")
    if dev.type == "cpu" and dev.index is not None:
        raise ValueError("Use device='cpu', without an index")
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch reports no available CUDA device")
        idx = torch.cuda.current_device() if dev.index is None else dev.index
        if idx < 0 or idx >= torch.cuda.device_count():
            raise ValueError("CUDA device index is out of range")
        dev = torch.device("cuda", idx)
    return torch, dev


def to_numpy(array):
    """Explicit device-to-host boundary; no implicit NumPy conversion of CUDA."""
    if isinstance(array, np.ndarray):
        return array
    return array.detach().cpu().numpy()


class Type1NUFFTPlan:
    """Fixed-geometry weighted/batched 3D periodic NUFFT, complex128 only.

    ``n_trans=1`` takes weights shape (N,) and returns ``shape``; otherwise
    weights have shape (n_trans,N), output (n_trans,*shape). ``weights=None``
    means all ones. Calling execute again reuses the upstream plan/setpts.
    NumPy arrays are returned by FINUFFT; torch tensors by the torch engines.
    ``eps`` is the requested transform accuracy, NOT a bound on H0 error.

    ``max_work_mb`` is a conservative preflight estimate, NOT an allocator cap
    or a guarantee covering opaque FFT/library workspaces or the whole process.
    CUDA work uses the current stream captured at construction. A different
    current stream at execute is rejected rather than risking a read/write race.
    Output validation and close synchronize; this inference API is not async.
    """
    def __init__(self, fractional_positions, shape, *, backend="finufft",
                 eps=1e-12, device="cpu", n_trans=1, nthreads=1,
                 max_work_mb=512.0, spread_batch_size=64):
        self._plan = None
        self._closed = False
        self._points = ()
        self._stream = None
        self.backend = str(backend).lower()
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {sorted(BACKENDS)}")
        self.shape = _shape3(shape)
        self.n_trans = _positive_int(n_trans, "n_trans")
        self.nthreads = _positive_int(nthreads, "nthreads")
        self.batch_size = _positive_int(spread_batch_size, "spread_batch_size")
        self.eps = float(eps)
        if not np.isfinite(self.eps) or not 1e-12 <= self.eps <= 1e-3:
            raise ValueError("eps must be finite and in [1e-12, 1e-3] for this double-precision adapter")
        self.torch = None
        if hasattr(fractional_positions, "is_complex"):
            if fractional_positions.is_complex():
                raise ValueError("fractional_positions must be real")
        elif np.iscomplexobj(fractional_positions):
            raise ValueError("fractional_positions must be real")
        if self.backend == "finufft":
            if str(device) != "cpu":
                raise ValueError("finufft is a CPU backend; use cufinufft for CUDA")
            self.device = "cpu"
            if hasattr(fractional_positions, "requires_grad") and fractional_positions.requires_grad:
                raise ValueError("NUFFT plan is inference-only; autograd coordinates are not accepted")
            frac = np.array(to_numpy(fractional_positions) if hasattr(fractional_positions, "detach")
                            else fractional_positions, dtype=np.float64, copy=True)
            if frac.ndim != 2 or frac.shape[1] != 3 or not np.isfinite(frac).all():
                raise ValueError("fractional_positions must have finite shape (N,3)")
        else:
            self.torch, self.device = resolve_torch_device(device)
            t = self.torch
            if self.backend == "cufinufft" and self.device.type != "cuda":
                raise ValueError("cufinufft requires a CUDA device")
            if hasattr(fractional_positions, "requires_grad") and fractional_positions.requires_grad:
                raise ValueError("NUFFT plan is inference-only; autograd coordinates are not accepted")
            # Copy: caller changes must not silently invalidate the setpts state.
            frac = t.as_tensor(fractional_positions, dtype=t.float64, device=self.device).clone().contiguous()
            if frac.ndim != 2 or frac.shape[1] != 3 or not bool(t.isfinite(frac).all()):
                raise ValueError("fractional_positions must have finite shape (N,3)")
            if self.device.type == "cuda":
                self._stream = t.cuda.current_stream(self.device)
        self.npoints = int(frac.shape[0])
        grid_size = prod(self.shape)
        self._gauss = None
        if self.backend == "torch_gaussian":
            self._gauss = gaussian_nufft_parameters(self.shape, self.eps, None)
            fine_shape = tuple(self._gauss["work_shape"])
            width = int(self._gauss["stencil_width"])
            # Complex grid/FFT + temporary indices/windows for bounded point chunks.
            estimate = (64*self.n_trans*prod(fine_shape) + 48*self.n_trans*grid_size
                        + 64*min(self.batch_size, max(1,self.npoints))*width**3
                        + (64+32*self.n_trans)*self.npoints)
        else:
            # Explicit upsampfac=2.0. Account for a minimum kernel-sized grid.
            fine_shape = tuple(int(next_fast_len(max(64,2*n))) for n in self.shape)
            estimate = (64*self.n_trans*prod(fine_shape) + 48*self.n_trans*grid_size
                        + (128+32*self.n_trans)*self.npoints)
        _budget(estimate, max_work_mb)
        self.metadata = dict(backend=self.backend, requested_eps=self.eps,
            dtype="complex128", coordinate_dtype="float64", n_trans=self.n_trans,
            atoms=self.npoints, shape=list(self.shape), device=str(self.device),
            sign=-1, modeord=1, dc_enforced_exact=True, dense_phase_elements=0,
            geometry_snapshot=True, autograd=False, execution_count=0,
            estimated_work_bytes=int(estimate), max_work_mb=max_work_mb,
            memory_estimate_is_hard_limit=False, nthreads=self.nthreads if self.backend=="finufft" else None,
            accuracy_contract="requested NUFFT tolerance; check abs(error)/sum(abs(weights)); not an H0 bound")
        if self.backend == "finufft":
            engine = _optional("finufft")
            angles = (np.remainder(frac + 0.5, 1.0) - 0.5) * (2*np.pi)
            self._points = tuple(np.ascontiguousarray(angles[:,d]) for d in range(3))
            self.metadata["library_version"] = getattr(engine,"__version__","unknown")
            if self.npoints:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # no ignored precision/option warning
                    self._plan = engine.Plan(1, self.shape, n_trans=self.n_trans,
                        eps=self.eps, isign=-1, dtype="complex128", modeord=1,
                        nthreads=self.nthreads, upsampfac=2.0)
                    self._plan.setpts(*self._points)
        elif self.backend == "cufinufft":
            # torch first, as in the upstream getting_started_torch.py example.
            engine = _optional("cufinufft")
            t = self.torch
            with t.cuda.device(self.device), t.cuda.stream(self._stream):
                angles = (t.remainder(frac+0.5,1.0)-0.5)*(2*np.pi)
                self._points = tuple(angles[:,d].contiguous() for d in range(3))
                self.metadata.update(library_version=getattr(engine,"__version__","unknown"),
                    torch_version=t.__version__, cuda_runtime=t.version.cuda,
                    gpu_name=t.cuda.get_device_name(self.device), stream_policy="captured_current_stream")
                if self.npoints:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        self._plan = engine.Plan(1,self.shape,n_trans=self.n_trans,
                            eps=self.eps,isign=-1,dtype="complex128",modeord=1,
                            upsampfac=2.0,gpu_device_id=self.device.index,
                            gpu_stream=self._stream.cuda_stream)
                        self._plan.setpts(*self._points)
        else:
            t = self.torch
            self._frac = t.remainder(frac, 1.0)
            self.metadata.update(torch_version=t.__version__, gaussian=self._gauss,
                scatter_reduction="CUDA atomics may change last bits; not bitwise deterministic")
            self._fine_shape = fine_shape
            self._offsets = t.arange(-int(self._gauss["stencil_half_width"]),
                                    int(self._gauss["stencil_half_width"])+1,device=self.device)
            self._modes = tuple(t.as_tensor(np.rint(np.fft.fftfreq(n,d=1/n)).astype(np.int64),
                                           device=self.device) for n in self.shape)
            self._deconv = tuple(t.exp(4*np.pi**2*float(self._gauss["tau_grid_units"])*(self._modes[d].to(t.float64)/fine_shape[d])**2)
                /np.sqrt(4*np.pi*float(self._gauss["tau_grid_units"])) for d in range(3))

    def _context(self):
        if self._stream is None:
            return nullcontext()
        if self.torch.cuda.current_stream(self.device).cuda_stream != self._stream.cuda_stream:
            raise RuntimeError("Plan must execute on its captured CUDA stream; create/use it inside the same stream context")
        return self.torch.cuda.device(self.device)

    def execute(self, weights=None):
        if self._closed:
            raise RuntimeError("NUFFT plan is closed")
        input_shape = (self.npoints,) if self.n_trans==1 else (self.n_trans,self.npoints)
        out_shape = self.shape if self.n_trans==1 else (self.n_trans,*self.shape)
        if hasattr(weights,"requires_grad") and weights.requires_grad:
            raise ValueError("NUFFT plan is inference-only; autograd weights are not accepted")
        with self._context():
            if self.backend == "finufft":
                if weights is not None and hasattr(weights,"detach"):
                    raise TypeError("FINUFFT weights must be CPU NumPy arrays, not torch tensors")
                c = np.ones(input_shape,np.complex128) if weights is None else np.ascontiguousarray(weights,dtype=np.complex128)
                if c.shape != input_shape or not np.isfinite(c).all():
                    raise ValueError(f"weights must be finite with shape {input_shape}")
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    out = self._plan.execute(c) if self.npoints else np.zeros(out_shape,np.complex128)
                # Some upstream single-transform interfaces retain a length-one batch.
                out = out.reshape(out_shape)
                out.reshape(self.n_trans,-1)[:,0] = c.reshape(self.n_trans,self.npoints).sum(axis=1)
                if not np.isfinite(out).all():
                    raise FloatingPointError("Non-finite FINUFFT output")
            else:
                t = self.torch
                with t.no_grad():
                    if isinstance(weights,t.Tensor) and weights.device != self.device:
                        raise ValueError(f"weights tensor must be on {self.device}; transfer explicitly")
                    c = t.ones(input_shape,dtype=t.complex128,device=self.device) if weights is None else t.as_tensor(weights,dtype=t.complex128,device=self.device).contiguous()
                    if tuple(c.shape)!=input_shape or not bool(t.isfinite(c).all()):
                        raise ValueError(f"weights must be finite with shape {input_shape}")
                    if not self.npoints:
                        out = t.zeros(out_shape,dtype=t.complex128,device=self.device)
                    elif self.backend == "cufinufft":
                        with warnings.catch_warnings():
                            warnings.simplefilter("error")
                            out = self._plan.execute(c).reshape(out_shape)
                    else:
                        out = self._execute_gaussian(c).reshape(out_shape)
                    out.reshape(self.n_trans,-1)[:,0] = c.reshape(self.n_trans,self.npoints).sum(dim=1)
                    if not bool(t.isfinite(out).all()):
                        raise FloatingPointError("Non-finite torch/CUDA NUFFT output")
            self.metadata["execution_count"] += 1
            return out

    def _execute_gaussian(self,c):
        t=self.torch; K=self._fine_shape; tau=float(self._gauss["tau_grid_units"])
        grid=t.zeros((self.n_trans,prod(K)),dtype=t.complex128,device=self.device)
        c=c.reshape(self.n_trans,self.npoints)
        kval=t.tensor(K,dtype=t.float64,device=self.device)
        for start in range(0,self.npoints,self.batch_size):
            stop=min(start+self.batch_size,self.npoints)
            u=self._frac[start:stop]*kval
            inds=[t.floor(u[:,d]).to(t.int64)[:,None]+self._offsets[None,:] for d in range(3)]
            weights=[t.exp(-(inds[d]-u[:,d,None])**2/(4*tau)) for d in range(3)]
            inds=[inds[d].remainder(K[d]) for d in range(3)]
            linear=(inds[0][:,:,None,None]*(K[1]*K[2])+inds[1][:,None,:,None]*K[2]+inds[2][:,None,None,:]).reshape(-1)
            window=weights[0][:,:,None,None]*weights[1][:,None,:,None]*weights[2][:,None,None,:]
            # One bounded scatter per transform/chunk; no Python loop per atom.
            for b in range(self.n_trans):
                vals=(c[b,start:stop,None,None,None]*window).reshape(-1)
                grid[b].scatter_add_(0,linear,vals)
        transformed=t.fft.fftn(grid.reshape(self.n_trans,*K),dim=(-3,-2,-1))
        out=transformed[:,self._modes[0][:,None,None].remainder(K[0]),
            self._modes[1][None,:,None].remainder(K[1]),self._modes[2][None,None,:].remainder(K[2])].contiguous()
        for d in range(3):
            broadcast=[1,1,1,1];broadcast[d+1]=self.shape[d]
            out*=self._deconv[d].reshape(broadcast)
        return out

    def close(self):
        if self._closed:
            return
        if self._stream is not None:
            self._stream.synchronize()
            with self.torch.cuda.device(self.device):
                self._plan = None
        else:
            self._plan = None  # upstream owner destroys the native plan
        self._points = ()
        for name in ("_frac", "_offsets", "_modes", "_deconv"):
            if hasattr(self, name):
                setattr(self, name, None)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self,*exc):
        self.close()


def library_structure_factor(fractional_positions,shape,*,backend="finufft",eps=1e-12,
                             device="cpu",nthreads=1,max_work_mb=512.0):
    """One-shot unweighted wrapper; output remains on the chosen backend/device."""
    with Type1NUFFTPlan(fractional_positions,shape,backend=backend,eps=eps,
                       device=device,nthreads=nthreads,max_work_mb=max_work_mb) as plan:
        out=plan.execute()
        meta=dict(plan.metadata)
    return out,meta
