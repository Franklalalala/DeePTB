"""Batched Hermitian eigensolver for torch CUDA tensors through ``cusolverDnXsyevBatched`` (ctypes, no compiled extension).

``torch.linalg.eigh`` / ``eigvalsh`` on CUDA loop over the batch and call ``syevd`` once per matrix; this routine solves the whole
batch with one cuSOLVER call (about 2-55x faster per matrix for complex64, depending on the matrix size, with the same accuracy).
It needs the ``libcusolver`` of cuSOLVER >= 11.7.1 (the ``nvidia-cusolver`` wheel that ships with torch 2.8 / CUDA 12.8).

Importing this module never loads the library.  The library is opened by the first ``BatchedEigh`` (or ``get_batched_eigh``)
call; when it cannot be opened ``get_batched_eigh`` raises ``RuntimeError`` and ``batched_eigh_available`` returns False
(logging the reason once), and the caller is expected to use ``torch.linalg`` instead.

Layout: a row-major ``[B, n, n]`` Hermitian tensor read column-major is ``conj(A)``, which has the same eigenvalues and the
conjugated eigenvectors, so the eigenvectors of ``A`` are the columns of ``(overwritten A).mH``.  Only the lower triangle
(column-major) is read, so the caller must pass Hermitian matrices.

Failure handling: every call reads the per-matrix ``info`` and checks that the input and the eigenvalues are finite.  When
something is wrong it raises ``CusolverBatchError`` carrying the indices of the bad matrices; the input has been overwritten by
then, so the caller keeps its own copy and recomputes with torch.  The cuSOLVER stream is set to torch's current stream on every
call and the workspaces are planned per (shape, stream).
"""
import ctypes
import functools
import glob
import logging
import os
import threading
from collections import OrderedDict
from ctypes import POINTER, byref, c_int, c_int64, c_size_t, c_void_p

import torch

log = logging.getLogger(__name__)

CUDA_R_32F, CUDA_R_64F, CUDA_C_32F, CUDA_C_64F = 0, 1, 4, 5
EIG_NOVECTOR, EIG_VECTOR = 0, 1
FILL_LOWER = 0
MIN_CUSOLVER_VERSION = 11701  # cusolverDnXsyevBatched appeared in cuSOLVER 11.7.1


class CusolverBatchError(RuntimeError):
    """cuSOLVER reported a failure (or produced / received non-finite values) for some matrices of a batch."""

    def __init__(self, msg, bad):
        super().__init__(msg)
        self.bad = bad  # LongTensor (cpu) of the batch indices whose result is unusable


def _bind(lib):
    lib.cusolverDnXsyevBatched_bufferSize.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int64, c_int, c_void_p, c_int64, c_int, c_void_p,
                                                      c_int, POINTER(c_size_t), POINTER(c_size_t), c_int64]
    lib.cusolverDnXsyevBatched.argtypes = [c_void_p, c_void_p, c_int, c_int, c_int64, c_int, c_void_p, c_int64, c_int, c_void_p, c_int,
                                           c_void_p, c_size_t, c_void_p, c_size_t, c_void_p, c_int64]
    lib.cusolverDnSetStream.argtypes = [c_void_p, c_void_p]
    lib.cusolverDnDestroy.argtypes = [c_void_p]
    lib.cusolverDnDestroyParams.argtypes = [c_void_p]


@functools.lru_cache(maxsize=None)
def _load_library():
    """(CDLL, None) or (None, reason); the result is cached for the process."""
    try:
        import nvidia.cusolver as pkg
    except ImportError as error:
        return None, f"the nvidia-cusolver wheel is not importable ({error})"
    roots = [os.path.dirname(pkg.__file__)] if getattr(pkg, "__file__", None) else list(getattr(pkg, "__path__", []))
    paths = sorted(p for root in roots for p in glob.glob(os.path.join(root, "lib", "libcusolver.so*")))
    if not paths:
        return None, "libcusolver.so* not found in the nvidia-cusolver wheel"
    try:
        lib = ctypes.CDLL(paths[-1])
        _bind(lib)
    except (OSError, AttributeError) as error:
        return None, f"cannot open {paths[-1]} ({error})"
    version = c_int()
    lib.cusolverGetVersion(byref(version))
    if version.value < MIN_CUSOLVER_VERSION:
        return None, f"cuSOLVER {version.value} < {MIN_CUSOLVER_VERSION} (cusolverDnXsyevBatched needs 11.7.1)"
    return lib, None


class BatchedEigh:
    def __init__(self, device, Bmax_hint=32, max_plans=8):
        lib, reason = _load_library()
        if lib is None:
            raise RuntimeError(f"cuSOLVER batched eigensolver unavailable: {reason}")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("BatchedEigh needs a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device()) if device.index is None else device
        self.lib = lib
        self.handle, self.params = c_void_p(), c_void_p()
        with torch.cuda.device(self.device):  # the handle belongs to the current device
            self._chk(lib.cusolverDnCreate(byref(self.handle)), "cusolverDnCreate")
            self._chk(lib.cusolverDnCreateParams(byref(self.params)), "cusolverDnCreateParams")
        version = c_int()
        lib.cusolverGetVersion(byref(version))
        self.version = version.value
        self.plans = OrderedDict()
        self.max_plans = max_plans
        self.Bmax_hint = Bmax_hint
        self.calls = 0
        self.errors = 0
        self._lock = threading.Lock()

    @staticmethod
    def _chk(status, name):
        if status != 0:
            raise RuntimeError("%s: cusolverStatus %d" % (name, status))

    def _stream(self):
        stream = torch.cuda.current_stream(self.device)
        self._chk(self.lib.cusolverDnSetStream(self.handle, c_void_p(stream.cuda_stream)), "cusolverDnSetStream")
        return stream.cuda_stream

    @staticmethod
    def _dtypes(dtype):
        if dtype == torch.complex64:
            return CUDA_C_32F, CUDA_R_32F
        if dtype == torch.complex128:
            return CUDA_C_64F, CUDA_R_64F
        if dtype == torch.float32:
            return CUDA_R_32F, CUDA_R_32F
        if dtype == torch.float64:
            return CUDA_R_64F, CUDA_R_64F
        raise TypeError(dtype)

    def _plan(self, n, Bmax, jobz, dtype, stream):
        key = (n, Bmax, jobz, dtype, stream)
        plan = self.plans.get(key)
        if plan is not None:
            self.plans.move_to_end(key)
            return plan
        tA, tW = self._dtypes(dtype)
        wdt = torch.empty(0, dtype=dtype).real.dtype
        A = torch.empty((Bmax, n, n), dtype=dtype, device=self.device)
        W = torch.empty((Bmax, n), dtype=wdt, device=self.device)
        size_dev, size_host = c_size_t(), c_size_t()
        self._chk(self.lib.cusolverDnXsyevBatched_bufferSize(self.handle, self.params, jobz, FILL_LOWER, n, tA, A.data_ptr(), n, tW, W.data_ptr(), tA,
                                                             byref(size_dev), byref(size_host), Bmax), "cusolverDnXsyevBatched_bufferSize")
        del A
        plan = dict(n=n, Bmax=Bmax, jobz=jobz, tA=tA, tW=tW, W=W, bufD=torch.empty(max(size_dev.value, 16), dtype=torch.uint8, device=self.device),
                    dA=size_dev.value, bufH=(ctypes.c_uint8 * max(size_host.value, 16))(), dH=size_host.value,
                    info=torch.zeros(Bmax, dtype=torch.int32, device=self.device))
        self.plans[key] = plan
        while len(self.plans) > self.max_plans:
            self.plans.popitem(last=False)
        return plan

    def eigh(self, A, vectors=True):
        """A: ``[B, n, n]`` Hermitian, contiguous, on this device; it is overwritten.  Returns ``(W [B, n] ascending, V [B, n, n] or None)``.

        Raises ``CusolverBatchError`` (with ``.bad``) when a matrix has non-finite entries, ``info != 0`` or non-finite eigenvalues."""
        assert A.is_contiguous() and A.dim() == 3 and A.shape[-1] == A.shape[-2]
        B, n = A.shape[0], A.shape[-1]
        jobz = EIG_VECTOR if vectors else EIG_NOVECTOR
        with self._lock, torch.cuda.device(self.device):
            stream = self._stream()
            finite_in = torch.isfinite(torch.view_as_real(A) if A.is_complex() else A).flatten(1).all(1)
            plan = self._plan(n, max(B, self.Bmax_hint), jobz, A.dtype, stream)
            self._chk(self.lib.cusolverDnXsyevBatched(self.handle, self.params, jobz, FILL_LOWER, n, plan["tA"], A.data_ptr(), n, plan["tW"],
                                                      plan["W"].data_ptr(), plan["tA"], plan["bufD"].data_ptr(), plan["dA"],
                                                      ctypes.cast(plan["bufH"], c_void_p), plan["dH"], plan["info"].data_ptr(), B),
                      "cusolverDnXsyevBatched")
            self.calls += 1
            W = plan["W"][:B].clone()
            bad = (~finite_in) | (plan["info"][:B] != 0) | (~torch.isfinite(W).all(1))
            if bool(bad.any()):  # one device sync per call; the eigensolve itself dominates the cost
                self.errors += 1
                idx = bad.nonzero().flatten().cpu()
                raise CusolverBatchError("cusolverDnXsyevBatched: %d of %d matrices failed (info/non-finite)" % (idx.numel(), B), idx)
        return W, (A.mH if vectors else None)

    def eigvalsh(self, A):
        return self.eigh(A, vectors=False)[0]

    def release(self):
        """Drop the cached workspaces (frees their device memory)."""
        self.plans.clear()

    def close(self):
        self.plans.clear()
        if self.params:
            self.lib.cusolverDnDestroyParams(self.params)
            self.params = c_void_p()
        if self.handle:
            self.lib.cusolverDnDestroy(self.handle)
            self.handle = c_void_p()

    def __del__(self):
        try:
            self.close()
        except Exception:  # interpreter shutdown, CUDA context already gone
            pass


_registry = {}
_registry_lock = threading.Lock()
_reported = set()


def get_batched_eigh(device):
    """The process-wide ``BatchedEigh`` of a CUDA device (created on first use).  Raises ``RuntimeError`` when cuSOLVER is unusable."""
    device = torch.device(device)
    index = torch.cuda.current_device() if device.index is None else device.index
    with _registry_lock:
        solver = _registry.get(index)
        if solver is None:
            solver = _registry[index] = BatchedEigh(torch.device("cuda", index))
    return solver


def batched_eigh_available(device):
    """True when the batched cuSOLVER solver can be used on ``device``; the reason is logged once otherwise."""
    try:
        get_batched_eigh(device)
        return True
    except (RuntimeError, ValueError, OSError) as error:
        if str(error) not in _reported:
            _reported.add(str(error))
            log.warning("cuSOLVER batched eigensolver not used (%s); falling back to torch.linalg", error)
        return False
