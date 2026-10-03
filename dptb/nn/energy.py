"""
The quantities module of GNN, with AtomicDataDict.Type as input and output the same class.

This version:
  - Keeps SOC (Full H) + scalar overlap (S is NxN) compatibility by expanding S -> blockdiag(S,S) implicitly.
  - Adds ill-conditioned overlap fallback via ill_threshold projection (migrated from the second script).
  - Spectral clipping of the Hamiltonian residual H - H0 (dptb/nn/band_fast.py) when H0 is given, off the autograd path.
"""

import os  # kept for compatibility with your previous iterations (even if unused)
import logging

import torch
import numpy as np
import torch.nn as nn
from dptb.nn.hr2hk import HR2HK
from typing import Union, Optional
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict

log = logging.getLogger(__name__)

_CLIP_FROM_INIT = object()  # forward(clip_b=...) default: the value given to __init__


def _blockdiag_dup(mat: torch.Tensor) -> torch.Tensor:
    """
    Build block diagonal duplication:
      mat: [B, N, N] -> out: [B, 2N, 2N] = [[mat,0],[0,mat]]
    """
    zeros = torch.zeros_like(mat)
    row1 = torch.cat([mat, zeros], dim=-1)
    row2 = torch.cat([zeros, mat], dim=-1)
    return torch.cat([row1, row2], dim=-2)


def _blockdiag_dup_vecs(vecs: torch.Tensor) -> torch.Tensor:
    """
    Build block diagonal duplication for eigenvectors:
      vecs: [B, N, N] -> out: [B, 2N, 2N] = [[vecs,0],[0,vecs]]
    """
    B, N, _ = vecs.shape
    out = torch.zeros((B, 2 * N, 2 * N), dtype=vecs.dtype, device=vecs.device)
    out[:, :N, :N] = vecs
    out[:, N:, N:] = vecs
    return out


class Eigenvalues(nn.Module):
    def __init__(
            self,
            idp: Union[OrbitalMapper, None] = None,
            h_edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            h_node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            h_out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            out_field: str = AtomicDataDict.ENERGY_EIGENVALUE_KEY,
            s_edge_field: str = None,
            s_node_field: str = None,
            s_out_field: str = None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            clip_b: Optional[float] = 10.0,
            h0_node_field: str = AtomicDataDict.NODE_H0_KEY,
            h0_edge_field: str = AtomicDataDict.EDGE_H0_KEY,
            fp32_min_lambda: float = 1e-4,
            kchunk: Union[str, int] = "auto",
            eig_backend: str = "auto"):
        """
        clip_b:          half-width (eV) of the spectral clipping of the residual H - H0 (generalized eigenvalues of (H - H0, S));
                         None switches the clipping off.  The clipped fast path (dptb/nn/band_fast.py) is taken only when
                         clip_b is not None, there is an overlap matrix, the data carry H0 (h0_node_field / h0_edge_field, the
                         same shapes as the H features) and nothing is being differentiated; every other call runs the legacy
                         code below unchanged.
        fp32_min_lambda: float32 only - k points whose smallest S eigenvalue is below this limit are solved in float64.
        kchunk:          k points per chunk of the fast path, 'auto' (memory bound, at most 128) or an int.
        eig_backend:     'auto' (batched cuSOLVER on CUDA when it can be loaded, else torch), 'cusolver' or 'torch'.
        """
        super(Eigenvalues, self).__init__()

        self.h2k = HR2HK(
            idp=idp,
            edge_field=h_edge_field,
            node_field=h_node_field,
            out_field=h_out_field,
            dtype=dtype,
            device=device,
        )

        if s_edge_field is not None:
            self.s2k = HR2HK(
                idp=idp,
                overlap=True,
                edge_field=s_edge_field,
                node_field=s_node_field,
                out_field=s_out_field,
                dtype=dtype,
                device=device,
            )
            self.overlap = True
        else:
            self.overlap = False

        self.out_field = out_field
        self.h_out_field = h_out_field
        self.s_out_field = s_out_field

        if clip_b is not None and not clip_b > 0:
            raise ValueError(f"clip_b must be positive or None, got {clip_b!r}")
        if eig_backend not in ("auto", "cusolver", "torch"):
            raise ValueError(f"eig_backend must be 'auto', 'cusolver' or 'torch', got {eig_backend!r}")
        if not (kchunk == "auto" or (isinstance(kchunk, int) and kchunk >= 1)):
            raise ValueError(f"kchunk must be 'auto' or a positive int, got {kchunk!r}")
        self.clip_b = clip_b
        self.h0_node_field = h0_node_field
        self.h0_edge_field = h0_edge_field
        self.fp32_min_lambda = fp32_min_lambda
        self.kchunk = kchunk
        self.eig_backend = eig_backend
        self.clip_diagnostics = False  # True: the fast path also records the exact residual spectrum per k (slow; for validation)
        self.last_clip_stats = None  # statistics of the last fast-path call (dict); None after a legacy-path call
        self._clip_notes = set()

    def _clip_note(self, key, message):
        """Log a message once per instance."""
        if key not in self._clip_notes:
            self._clip_notes.add(key)
            log.info(message)

    def _clip_inputs(self, data):
        """(H0 node, H0 edge) features when the clipped fast path applies to this call, else None (legacy path).

        Raises ValueError when H0 is present but not in the layout of the H features."""
        from dptb.nn import band_fast

        h_node, h_edge = data.get(self.h2k.node_field), data.get(self.h2k.edge_field)
        if h_node is None or h_edge is None or data.get(self.s2k.node_field) is None or data.get(self.s2k.edge_field) is None:
            return None
        if torch.is_grad_enabled() or h_node.requires_grad or h_edge.requires_grad:
            return None  # training / differentiation: the legacy path
        h0_node, h0_edge = data.get(self.h0_node_field), data.get(self.h0_edge_field)
        if h0_node is None or h0_edge is None:
            self._clip_note("no_h0", "no H0 in data; spectral clipping skipped")
            return None
        if h0_node.shape != h_node.shape or h0_edge.shape != h_edge.shape:
            raise ValueError(
                f"spectral clipping needs H0 in the layout of the Hamiltonian features, but {self.h0_node_field!r} / {self.h0_edge_field!r} "
                f"have shapes {tuple(h0_node.shape)} / {tuple(h0_edge.shape)} and {self.h2k.node_field!r} / {self.h2k.edge_field!r} have "
                f"{tuple(h_node.shape)} / {tuple(h_edge.shape)}; give H0 in the same (full) layout or pass clip_b=None")
        if band_fast.feature_layout(self.h2k.idp) is None:
            self._clip_note("layout", "spectral clipping does not handle this orbital layout; legacy eigensolver used")
            return None
        return h0_node, h0_edge

    def _clip_backend(self, device):
        """The batched cuSOLVER solver for a CUDA device (eig_backend 'auto' / 'cusolver'), else None (torch.linalg)."""
        if self.eig_backend == "torch" or device.type != "cuda":
            return None
        from dptb.nn.cusolver_batched import batched_eigh_available, get_batched_eigh

        if self.eig_backend == "cusolver":
            return get_batched_eigh(device)
        return get_batched_eigh(device) if batched_eigh_available(device) else None

    def _forward_clipped(self, data, nk, ill_threshold, clip_b, h0):
        """Bands with spectral clipping of the residual (see dptb/nn/band_fast.py); same output as the legacy path."""
        from dptb.nn import band_fast

        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            assert kpoints.size(0) == 1
            kpoints0 = kpoints[0]
        else:
            kpoints0 = kpoints
        idp, device = self.h2k.idp, torch.device(self.h2k.device)
        fp64 = self.h2k.dtype == torch.float64
        h_node, h_edge = data[self.h2k.node_field], data[self.h2k.edge_field]
        h0_node, h0_edge = h0
        dtype = torch.promote_types(h_node.dtype, h0_node.dtype)
        r_node, r_edge = h_node.to(device, dtype) - h0_node.to(device, dtype), h_edge.to(device, dtype) - h0_edge.to(device, dtype)
        scale = max(band_fast._absmax(x) for x in (h_node, h_edge, h0_node, h0_edge))
        kind, spin = band_fast.detect_kind(idp, (r_node, r_edge), scale, torch.finfo(dtype).eps)
        atom_types = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        # strict FP32 inside the fast path, whatever the global matmul precision is; restored on every exit
        tf32, precision = torch.backends.cuda.matmul.allow_tf32, torch.get_float32_matmul_precision()
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            solver = band_fast.SpectralBandSolver(
                idp, atom_types, data[AtomicDataDict.EDGE_INDEX_KEY], data[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
                (data[self.s2k.node_field], data[self.s2k.edge_field]), (h0_node, h0_edge), (r_node, r_edge), device,
                precision="fp64" if fp64 else "fp32", kind=kind, batched=self._clip_backend(device))
            self.last_clip_stats = dict(spin)
            try:
                levels = solver.solve(kpoints0, b=clip_b, ill_threshold=ill_threshold, fp32_min_lambda=self.fp32_min_lambda,
                                      kchunk=self.kchunk, diagnose=self.clip_diagnostics, max_chunk=nk)
            finally:
                self.last_clip_stats = {**spin, **solver.stats}
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.set_float32_matmul_precision(precision)
        data[self.out_field] = torch.nested.as_nested_tensor([levels])
        return data

    def forward(
            self,
            data: AtomicDataDict.Type,
            nk: Optional[int] = None,
            ill_threshold: Optional[float] = 1e-5,
            clip_b: Optional[float] = _CLIP_FROM_INIT
    ) -> AtomicDataDict.Type:
        """
        Compute eigenvalues along k-points.

        ill_threshold:
          - None: legacy behavior (pure Cholesky reduction of generalized eigenproblem)
          - float: robust projection for ill-conditioned overlap S
        clip_b: half-width (eV) of the spectral clipping of H - H0 (default: the value given to __init__); None = no clipping.
          The clipped fast path runs when clip_b is not None, there is an overlap matrix, data carries H0 in the layout of the H
          features and gradients are not needed; otherwise this is the legacy code, unchanged.  The fast path keeps the output
          layout (nested eigenvalues, ascending, same dtype), fills self.last_clip_stats, does not leave H(k) / S(k) in data and
          takes nk as an upper bound of its k chunk.  ill_threshold keeps its meaning (directions of S with eigenvalue <=
          ill_threshold are removed and padded with 1e4 eV; None: removal only if S is not positive definite even in float64).
        """
        self.last_clip_stats = None
        if clip_b is _CLIP_FROM_INIT:
            clip_b = self.clip_b
        if clip_b is not None and self.overlap:
            h0 = self._clip_inputs(data)
            if h0 is not None:
                return self._forward_clipped(data, nk, ill_threshold, clip_b, h0)

        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            nested = True
            assert kpoints.size(0) == 1
            kpoints0 = kpoints[0]
        else:
            nested = False
            kpoints0 = kpoints

        num_k = kpoints0.shape[0]
        eigvals_chunks = []
        if nk is None:
            nk = num_k

        for i in range(int(np.ceil(num_k / nk))):
            data[AtomicDataDict.KPOINT_KEY] = kpoints0[i * nk:(i + 1) * nk]
            data = self.h2k(data)

            H_k = data[self.h_out_field]  # [B, dimH, dimH]

            if not self.overlap:
                batch_eigvals = torch.linalg.eigvalsh(H_k)
                eigvals_chunks.append(batch_eigvals)
                continue

            # overlap branch
            data = self.s2k(data)
            S_k = data[self.s_out_field]  # [B, dimS, dimS]

            # SOC mismatch detection: H is 2N but S is N (scalar overlap)
            soc_mismatch = (H_k.shape[-1] == 2 * S_k.shape[-1])

            # print(ill_threshold)

            # --------
            # Case A: no ill fallback (legacy cholesky)
            # --------
            if ill_threshold is None:
                L = torch.linalg.cholesky(S_k)
                L_inv = torch.linalg.inv(L)  # [B,N,N]

                if soc_mismatch:
                    # Expand L_inv -> blockdiag(L_inv, L_inv), then H' = Linv_big H Linv_big^H
                    L_inv_big = _blockdiag_dup(L_inv)
                    H_k_transformed = (L_inv_big @ H_k @ L_inv_big.mH)
                else:
                    H_k_transformed = (L_inv @ H_k @ L_inv.mH)

                batch_eigvals = torch.linalg.eigvalsh(H_k_transformed)
                eigvals_chunks.append(batch_eigvals)
                continue

            # --------
            # Case B: ill-conditioned overlap fallback (projection)
            # --------
            # Eigen-decompose S in its native dimension (N×N). For SOC mismatch, we later duplicate to 2N implicitly.
            egval_S, egvec_S = torch.linalg.eigh(S_k)  # egval_S:[B,N], egvec_S:[B,N,N]
            processed_eigvals_list = []

            if soc_mismatch:
                num_orbitals = H_k.shape[-1]  # 2N
                N = S_k.shape[-1]
            else:
                num_orbitals = H_k.shape[-1]  # N
                N = S_k.shape[-1]

            B = S_k.shape[0]
            for k_idx in range(B):
                # healthy indices in S-eigen space (size N)
                healthy_mask = egval_S[k_idx] > ill_threshold
                n_healthy = int(healthy_mask.sum().item())

                if n_healthy == 0:
                    # Extreme case: everything ill -> return all padded
                    egval = torch.full(
                        (num_orbitals,),
                        1e4,
                        dtype=H_k.dtype.real_dtype if H_k.is_complex() else H_k.dtype,
                        device=H_k.device,
                    )
                    processed_eigvals_list.append(egval)
                    continue

                if healthy_mask.all():
                    # well-conditioned -> solve full generalized eig (with SOC mismatch handled)
                    S_i = S_k[k_idx]
                    H_i = H_k[k_idx]

                    L = torch.linalg.cholesky(S_i)
                    L_inv = torch.linalg.inv(L)

                    if soc_mismatch:
                        L_inv_big = _blockdiag_dup(L_inv)
                        H_transformed = L_inv_big @ H_i @ L_inv_big.mH
                    else:
                        H_transformed = L_inv @ H_i @ L_inv.mH

                    egval = torch.linalg.eigvalsh(H_transformed)
                    processed_eigvals_list.append(egval)
                    continue

                # ill-conditioned -> project
                U = egvec_S[k_idx]  # [N,N]
                evals = egval_S[k_idx]  # [N]

                U_sel = U[:, healthy_mask]  # [N, M]
                eval_sel = evals[healthy_mask]  # [M]

                if soc_mismatch:
                    # Build V = blockdiag(U_sel, U_sel): [2N, 2M]
                    # and S_proj = diag([eval_sel, eval_sel])
                    V = torch.zeros((2 * N, 2 * n_healthy), dtype=U.dtype, device=U.device)
                    V[:N, :n_healthy] = U_sel
                    V[N:, n_healthy:] = U_sel

                    # Project H: [2M,2M]
                    H_proj = V.mH @ H_k[k_idx] @ V

                    # S_proj is diagonal in this eigenbasis
                    eval_dup = torch.cat([eval_sel, eval_sel], dim=0)  # [2M]
                    S_proj = torch.diag(eval_dup).to(dtype=H_proj.dtype, device=H_proj.device)

                    L = torch.linalg.cholesky(S_proj)
                    L_inv = torch.linalg.inv(L)
                    H_transformed = L_inv @ H_proj @ L_inv.mH
                    egval_proj = torch.linalg.eigvalsh(H_transformed)

                    # pad to 2N
                    num_projected_out = num_orbitals - egval_proj.shape[0]
                    if num_projected_out > 0:
                        padding = torch.full(
                            (num_projected_out,),
                            1e4,
                            dtype=egval_proj.dtype,
                            device=egval_proj.device
                        )
                        egval = torch.cat([egval_proj, padding], dim=0)
                    else:
                        egval = egval_proj

                    processed_eigvals_list.append(egval)
                else:
                    # Non-SOC: V = U_sel: [N,M], S_proj = diag(eval_sel)
                    V = U_sel  # [N,M]
                    H_proj = V.mH @ H_k[k_idx] @ V
                    S_proj = torch.diag(eval_sel).to(dtype=H_proj.dtype, device=H_proj.device)

                    L = torch.linalg.cholesky(S_proj)
                    L_inv = torch.linalg.inv(L)
                    H_transformed = L_inv @ H_proj @ L_inv.mH
                    egval_proj = torch.linalg.eigvalsh(H_transformed)

                    # pad to N
                    num_projected_out = num_orbitals - egval_proj.shape[0]
                    if num_projected_out > 0:
                        padding = torch.full(
                            (num_projected_out,),
                            1e4,
                            dtype=egval_proj.dtype,
                            device=egval_proj.device
                        )
                        egval = torch.cat([egval_proj, padding], dim=0)
                    else:
                        egval = egval_proj

                    processed_eigvals_list.append(egval)

            batch_eigvals = torch.stack(processed_eigvals_list, dim=0)
            eigvals_chunks.append(batch_eigvals)

        # concat all chunks
        final_eigvals = torch.cat(eigvals_chunks, dim=0)

        # IMPORTANT:
        # Keep historical behavior used by ElecStruCal: store eigenvalues as nested tensor with one item.
        data[self.out_field] = torch.nested.as_nested_tensor([final_eigvals])

        # restore kpoints
        if nested:
            data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints0])
        else:
            data[AtomicDataDict.KPOINT_KEY] = kpoints0

        return data


class Eigh(nn.Module):
    """
    Keep your first-script SOC scalar-overlap (S NxN, H 2N×2N) compatibility.
    Note: ill_threshold projection is NOT migrated here (same as your second script), because it would require
          careful eigenvector padding / reconstruction semantics. If you want, I can extend Eigh similarly.
    """
    def __init__(
            self,
            idp: Union[OrbitalMapper, None] = None,
            h_edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            h_node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            h_out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            eigval_field: str = AtomicDataDict.ENERGY_EIGENVALUE_KEY,
            eigvec_field: str = AtomicDataDict.EIGENVECTOR_KEY,
            s_edge_field: str = None,
            s_node_field: str = None,
            s_out_field: str = None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu")):
        super(Eigh, self).__init__()

        self.h2k = HR2HK(
            idp=idp,
            edge_field=h_edge_field,
            node_field=h_node_field,
            out_field=h_out_field,
            dtype=dtype,
            device=device,
        )

        if s_edge_field is not None:
            self.s2k = HR2HK(
                idp=idp,
                overlap=True,
                edge_field=s_edge_field,
                node_field=s_node_field,
                out_field=s_out_field,
                dtype=dtype,
                device=device,
            )
            self.overlap = True
        else:
            self.overlap = False

        self.eigval_field = eigval_field
        self.eigvec_field = eigvec_field
        self.h_out_field = h_out_field
        self.s_out_field = s_out_field

    def forward(self, data: AtomicDataDict.Type, nk: Optional[int] = None) -> AtomicDataDict.Type:
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            nested = True
            assert kpoints.size(0) == 1
            kpoints0 = kpoints[0]
        else:
            nested = False
            kpoints0 = kpoints

        num_k = kpoints0.shape[0]
        eigvals = []
        eigvecs = []
        if nk is None:
            nk = num_k

        for i in range(int(np.ceil(num_k / nk))):
            data[AtomicDataDict.KPOINT_KEY] = kpoints0[i * nk:(i + 1) * nk]
            data = self.h2k(data)

            chklowtinv_final = None

            if self.overlap:
                data = self.s2k(data)

                H = data[self.h_out_field]  # [B, dimH, dimH]
                S = data[self.s_out_field]  # [B, dimS, dimS]

                L = torch.linalg.cholesky(S)
                L_inv = torch.linalg.inv(L)  # [B, dimS, dimS]

                # SOC mismatch: H is 2N while S is N
                if H.shape[-1] == 2 * S.shape[-1]:
                    L_inv_big = _blockdiag_dup(L_inv)
                    chklowtinv_final = L_inv_big
                    data[self.h_out_field] = (L_inv_big @ H @ L_inv_big.mH)
                else:
                    chklowtinv_final = L_inv
                    data[self.h_out_field] = (L_inv @ H @ L_inv.mH)

            # eig
            eigval, eigvec = torch.linalg.eigh(data[self.h_out_field])

            # restore eigenvectors to AO basis if overlap
            if self.overlap:
                # x = L^{-H} y  (but we stored L_inv already)
                raw_eigvec = chklowtinv_final.mH @ eigvec
                eigvecs.append(raw_eigvec.mH)  # [B, nband, norb]
            else:
                eigvecs.append(eigvec.mH)

            eigvals.append(eigval)

        data[self.eigval_field] = torch.nested.as_nested_tensor([torch.cat(eigvals, dim=0)])
        data[self.eigvec_field] = torch.cat(eigvecs, dim=0)

        if nested:
            data[AtomicDataDict.KPOINT_KEY] = torch.nested.as_nested_tensor([kpoints0])
        else:
            data[AtomicDataDict.KPOINT_KEY] = kpoints0

        return data