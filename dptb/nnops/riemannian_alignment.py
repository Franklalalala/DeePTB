# SPDX-License-Identifier: LGPL-3.0-or-later
"""Near-Fermi / Riemannian alignment losses for Hamiltonian training.

This module adds a lightweight training-time constraint for the case where a
Hamiltonian predictor gives good band structures but spends element-wise error in
irrelevant high-energy virtual directions.  Instead of repairing the predicted
Hamiltonian after inference, the loss aligns the *reference near-Fermi subspace*
and a few invariant moments during training.

The numerical core accepts dense H(k), H_ref(k), and optional S(k).  The
registered Loss wrapper can be used either directly or as a flow auxiliary loss.
It deliberately differentiates only through H_pred: the reference generalized
eigenvectors that define the P/Q projectors are computed under ``no_grad``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:  # normal DeePTB runtime
    from dptb.nnops.loss import Loss
except Exception:  # pragma: no cover - lets the dense core be unit-tested standalone
    class _DummyLoss:
        @staticmethod
        def register(_name):
            def deco(cls):
                return cls
            return deco
    Loss = _DummyLoss()

try:
    from dptb.data import AtomicDataDict, _keys
except Exception:  # pragma: no cover - standalone dense-core tests
    class _Keys:
        HAMILTONIAN_KEY = "hamiltonian"
        OVERLAP_KEY = "overlap"
        ENERGY_EIGENVALUE_KEY = "eigenvalue"
        KPOINT_KEY = "kpoint"
        NODE_FEATURES_KEY = "node_features"
        EDGE_FEATURES_KEY = "edge_features"
        NODE_OVERLAP_KEY = "node_overlap"
        EDGE_OVERLAP_KEY = "edge_overlap"
    _keys = _Keys()

    class AtomicDataDict:  # type: ignore[no-redef]
        HAMILTONIAN_KEY = _keys.HAMILTONIAN_KEY
        OVERLAP_KEY = _keys.OVERLAP_KEY
        ENERGY_EIGENVALUE_KEY = _keys.ENERGY_EIGENVALUE_KEY
        KPOINT_KEY = _keys.KPOINT_KEY
        NODE_FEATURES_KEY = _keys.NODE_FEATURES_KEY
        EDGE_FEATURES_KEY = _keys.EDGE_FEATURES_KEY
        NODE_OVERLAP_KEY = _keys.NODE_OVERLAP_KEY
        EDGE_OVERLAP_KEY = _keys.EDGE_OVERLAP_KEY

try:
    from dptb.nn.energy import Eigenvalues
except Exception:  # pragma: no cover
    Eigenvalues = None

Tensor = torch.Tensor

# Shared numerical core -- single source, no local redefinition (see _manifold_math).
from dptb.nnops._manifold_math import (
    hermitian_part,
    regularize_overlap,
    generalized_eigh,
)


def _as_tuple_of_ints(value: Union[str, int, Iterable[int], None], default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return tuple(int(v) for v in value)


def _identity_like(h: Tensor) -> Tensor:
    eye = torch.eye(h.shape[-1], device=h.device, dtype=h.dtype)
    return eye.expand(h.shape)


def fermi_from_n_occ(eigvals: Tensor, n_occ: int) -> Tensor:
    n = int(eigvals.numel())
    if n_occ <= 0 or n_occ >= n:
        raise ValueError(f"n_occ must be in [1, n_bands-1], got {n_occ} for {n} bands")
    return 0.5 * (eigvals[n_occ - 1] + eigvals[n_occ])


def near_fermi_indices(
    eigvals: Tensor,
    *,
    n_occ: int,
    e_cut_below_fermi: Optional[float] = 2.0,
    e_cut_above_fermi: Optional[float] = 2.0,
    min_states: int = 2,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Select the P subspace as a reference near-Fermi energy window."""
    n = int(eigvals.numel())
    fermi = fermi_from_n_occ(eigvals, int(n_occ))
    lower = eigvals.new_tensor(-torch.inf) if e_cut_below_fermi is None else fermi - float(e_cut_below_fermi)
    upper = eigvals.new_tensor(torch.inf) if e_cut_above_fermi is None else fermi + float(e_cut_above_fermi)
    mask = (eigvals >= lower) & (eigvals <= upper)
    p_idx = torch.nonzero(mask, as_tuple=False).reshape(-1)

    if int(p_idx.numel()) < int(min_states):
        # Always include the VBM/CBM pair, then expand symmetrically around it.
        lo = max(0, int(n_occ) - max(1, int(min_states) // 2))
        hi = min(n, lo + int(min_states))
        lo = max(0, hi - int(min_states))
        p_idx = torch.arange(lo, hi, device=eigvals.device, dtype=torch.long)

    if int(p_idx.numel()) >= n:
        # Keep Q non-empty so PQ/QQ diagnostics remain well-defined.
        drop = 0 if p_idx[0].item() != 0 else n - 1
        p_idx = p_idx[p_idx != drop]

    all_idx = torch.arange(n, device=eigvals.device, dtype=torch.long)
    q_mask = torch.ones(n, device=eigvals.device, dtype=torch.bool)
    q_mask[p_idx] = False
    q_idx = all_idx[q_mask]
    return p_idx, q_idx, fermi


def _index_block(x: Tensor, rows: Tensor, cols: Tensor) -> Tensor:
    return x.index_select(-2, rows).index_select(-1, cols)


def _mse_abs(x: Tensor) -> Tensor:
    if x.numel() == 0:
        return x.real.new_zeros(())
    return (x.abs() ** 2).mean()


def _solve_scalar_gauge(diff_t: Tensor, p_idx: Tensor, q_idx: Tensor, lambda_pp: float, lambda_qq: float, eps: float = 1e-30) -> Tensor:
    """Fit a k-local scalar shift in the reference eigenbasis."""
    diag = diff_t.diagonal().real
    num = diff_t.real.new_zeros(())
    den = 0.0
    if p_idx.numel() > 0 and lambda_pp != 0.0:
        num = num + float(lambda_pp) * diag.index_select(0, p_idx).sum()
        den += float(lambda_pp) * int(p_idx.numel())
    if q_idx.numel() > 0 and lambda_qq != 0.0:
        num = num + float(lambda_qq) * diag.index_select(0, q_idx).sum()
        den += float(lambda_qq) * int(q_idx.numel())
    if den <= eps:
        return diff_t.real.new_zeros(())
    return num / den


def _matrix_trace(x: Tensor) -> Tensor:
    return x.diagonal(dim1=-2, dim2=-1).sum(-1)


def _trace_moment_loss(pred_pp: Tensor, ref_pp: Tensor, orders: Sequence[int]) -> Tensor:
    """TraceGrad-inspired invariant moment matching inside P."""
    if pred_pp.numel() == 0 or len(orders) == 0:
        return pred_pp.real.new_zeros(())
    n = max(int(pred_pp.shape[-1]), 1)
    terms = []
    pred_h = hermitian_part(pred_pp)
    ref_h = hermitian_part(ref_pp)
    for order in orders:
        order = int(order)
        if order <= 0:
            continue
        pp = torch.matrix_power(pred_h, order)
        rr = torch.matrix_power(ref_h, order)
        scale = float(n) ** max(order, 1)
        terms.append(((_matrix_trace(pp) - _matrix_trace(rr)).abs() ** 2) / scale)
    if not terms:
        return pred_pp.real.new_zeros(())
    return torch.stack([t.real for t in terms]).mean()


@dataclass(frozen=True)
class RiemannianAlignmentResult:
    loss: Tensor
    loss_pp: Tensor
    loss_pq: Tensor
    loss_qq: Tensor
    loss_trace: Tensor
    loss_diag: Tensor
    mu: Tensor
    n_p: int
    n_q: int
    fermi: Tensor


def riemannian_alignment_loss_single(
    h_pred: Tensor,
    h_ref: Tensor,
    s: Optional[Tensor] = None,
    *,
    n_occ: int,
    e_cut_below_fermi: Optional[float] = 2.0,
    e_cut_above_fermi: Optional[float] = 2.0,
    lambda_pp: float = 1.0,
    lambda_pq: float = 1.0,
    lambda_qq: float = 0.02,
    lambda_trace: float = 0.05,
    lambda_diag: float = 0.1,
    moment_orders: Sequence[int] = (1, 2),
    gauge_mu: bool = True,
    overlap_eig_floor: float = 1.0e-10,
) -> RiemannianAlignmentResult:
    """Dense one-k loss using the reference near-Fermi Grassmann point.

    P is the near-Fermi reference subspace.  PP fits the effective Hamiltonian in
    that subspace; PQ suppresses leakage/rotation against Q; QQ is intentionally
    down-weighted; trace moments match basis-invariant scalars inside P.
    """
    with torch.no_grad():
        eps_ref, u_ref = generalized_eigh(h_ref.detach(), None if s is None else s.detach(), eig_floor=overlap_eig_floor)
        p_idx, q_idx, fermi = near_fermi_indices(
            eps_ref,
            n_occ=int(n_occ),
            e_cut_below_fermi=e_cut_below_fermi,
            e_cut_above_fermi=e_cut_above_fermi,
            min_states=2,
        )

    pred_t = u_ref.mH @ hermitian_part(h_pred) @ u_ref
    ref_t = (u_ref.mH @ hermitian_part(h_ref) @ u_ref).detach()
    diff_t = pred_t - ref_t
    if gauge_mu:
        mu = _solve_scalar_gauge(diff_t, p_idx, q_idx, lambda_pp=lambda_pp, lambda_qq=lambda_qq)
        diff_t = diff_t - mu.to(dtype=diff_t.dtype) * torch.eye(diff_t.shape[-1], device=diff_t.device, dtype=diff_t.dtype)
        pred_t = pred_t - mu.to(dtype=pred_t.dtype) * torch.eye(pred_t.shape[-1], device=pred_t.device, dtype=pred_t.dtype)
    else:
        mu = diff_t.real.new_zeros(())

    pp = _index_block(diff_t, p_idx, p_idx)
    pq = _index_block(diff_t, p_idx, q_idx)
    qq = _index_block(diff_t, q_idx, q_idx)
    pred_pp = _index_block(pred_t, p_idx, p_idx)
    ref_pp = _index_block(ref_t, p_idx, p_idx)

    loss_pp = _mse_abs(pp)
    loss_pq = _mse_abs(pq)
    loss_qq = _mse_abs(qq)
    loss_trace = _trace_moment_loss(pred_pp, ref_pp, moment_orders)
    loss_diag = _mse_abs(pp.diagonal())
    loss = (
        float(lambda_pp) * loss_pp
        + float(lambda_pq) * loss_pq
        + float(lambda_qq) * loss_qq
        + float(lambda_trace) * loss_trace
        + float(lambda_diag) * loss_diag
    )
    return RiemannianAlignmentResult(
        loss=loss,
        loss_pp=loss_pp,
        loss_pq=loss_pq,
        loss_qq=loss_qq,
        loss_trace=loss_trace,
        loss_diag=loss_diag,
        mu=mu,
        n_p=int(p_idx.numel()),
        n_q=int(q_idx.numel()),
        fermi=fermi,
    )


def _flatten_dense_mats(x: Tensor) -> Tensor:
    if x.ndim == 2:
        return x.reshape((1,) + tuple(x.shape))
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"expected dense matrices [..., n, n], got {tuple(x.shape)}")
    return x.reshape((-1,) + tuple(x.shape[-2:]))


def _flatten_optional_overlap(s: Optional[Tensor], like: Tensor) -> Optional[Tensor]:
    if s is None:
        return None
    if s.ndim == 2:
        return s.reshape((1,) + tuple(s.shape)).expand(like.shape)
    return s.reshape((-1,) + tuple(s.shape[-2:]))


def _select_matrix_indices(n_mats: int, max_kpoints: Optional[int], random_kpoints: bool, device: torch.device) -> Tensor:
    if max_kpoints is None or int(max_kpoints) <= 0 or n_mats <= int(max_kpoints):
        return torch.arange(n_mats, device=device, dtype=torch.long)
    k = int(max_kpoints)
    if random_kpoints:
        return torch.randperm(n_mats, device=device)[:k]
    return torch.linspace(0, n_mats - 1, steps=k, device=device).round().to(torch.long).unique()


def _n_occ_for_flat_index(n_occ: Union[int, Tensor], i: int, n_items: int) -> int:
    if torch.is_tensor(n_occ):
        flat = n_occ.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        if flat.numel() == 0:
            raise ValueError("empty n_occ tensor")
        if flat.numel() == 1:
            return int(flat[0].item())
        # If one n_occ per structure rather than per k point, fall back to the
        # nearest valid entry by proportional indexing.
        j = min(int(i), int(flat.numel()) - 1)
        if flat.numel() != n_items and n_items > 0:
            j = min(int(i * flat.numel() / n_items), int(flat.numel()) - 1)
        return int(flat[j].item())
    return int(n_occ)


def dense_riemannian_alignment_loss(
    h_pred: Tensor,
    h_ref: Tensor,
    s: Optional[Tensor] = None,
    *,
    n_occ: Union[int, Tensor],
    e_cut_below_fermi: Optional[float] = 2.0,
    e_cut_above_fermi: Optional[float] = 2.0,
    lambda_pp: float = 1.0,
    lambda_pq: float = 1.0,
    lambda_qq: float = 0.02,
    lambda_trace: float = 0.05,
    lambda_diag: float = 0.1,
    moment_orders: Sequence[int] = (1, 2),
    gauge_mu: bool = True,
    overlap_eig_floor: float = 1.0e-10,
    max_kpoints: Optional[int] = None,
    random_kpoints: bool = True,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    if h_pred.shape != h_ref.shape:
        raise ValueError(f"h_pred/h_ref shape mismatch: {tuple(h_pred.shape)} vs {tuple(h_ref.shape)}")
    flat_pred = _flatten_dense_mats(h_pred)
    flat_ref = _flatten_dense_mats(h_ref)
    flat_s = _flatten_optional_overlap(s, flat_pred)
    if flat_s is not None and flat_s.shape != flat_pred.shape:
        raise ValueError(f"S shape mismatch after flatten: {tuple(flat_s.shape)} vs {tuple(flat_pred.shape)}")

    indices = _select_matrix_indices(flat_pred.shape[0], max_kpoints, random_kpoints, flat_pred.device)
    results = []
    for idx in indices.tolist():
        ss = None if flat_s is None else flat_s[int(idx)]
        results.append(
            riemannian_alignment_loss_single(
                flat_pred[int(idx)],
                flat_ref[int(idx)],
                ss,
                n_occ=_n_occ_for_flat_index(n_occ, int(idx), int(flat_pred.shape[0])),
                e_cut_below_fermi=e_cut_below_fermi,
                e_cut_above_fermi=e_cut_above_fermi,
                lambda_pp=lambda_pp,
                lambda_pq=lambda_pq,
                lambda_qq=lambda_qq,
                lambda_trace=lambda_trace,
                lambda_diag=lambda_diag,
                moment_orders=moment_orders,
                gauge_mu=gauge_mu,
                overlap_eig_floor=overlap_eig_floor,
            )
        )
    if not results:
        zero = flat_pred.real.new_zeros(())
        return zero, {"riem_skipped": flat_pred.real.new_ones(())}

    loss = torch.stack([r.loss for r in results]).mean()
    stats = {
        "riem_align_loss": loss.detach(),
        "riem_pp_loss": torch.stack([r.loss_pp.detach() for r in results]).mean(),
        "riem_pq_loss": torch.stack([r.loss_pq.detach() for r in results]).mean(),
        "riem_qq_loss": torch.stack([r.loss_qq.detach() for r in results]).mean(),
        "riem_trace_loss": torch.stack([r.loss_trace.detach() for r in results]).mean(),
        "riem_diag_loss": torch.stack([r.loss_diag.detach() for r in results]).mean(),
        "riem_mu": torch.stack([r.mu.detach() for r in results]).mean(),
        "riem_rank_p": flat_pred.real.new_tensor(float(sum(r.n_p for r in results) / len(results))),
        "riem_rank_q": flat_pred.real.new_tensor(float(sum(r.n_q for r in results) / len(results))),
        "riem_fermi": torch.stack([r.fermi.detach() for r in results]).mean(),
        "riem_skipped": flat_pred.real.new_zeros(()),
    }
    return loss, stats


@Loss.register("riemannian_align")
@Loss.register("fermi_projector_align")
class RiemannianAlignmentLoss(nn.Module):
    """Registered DeePTB loss for near-Fermi subspace alignment.

    Recommended use with CFM is as ``train_options.flow_options.auxiliary_loss``
    with a small weight, because the flow objective path otherwise bypasses the
    normal ``train_lossfunc``.
    """

    def __init__(
        self,
        basis: Optional[dict] = None,
        idp: Any = None,
        overlap: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        n_occ: Optional[int] = None,
        n_occ_key: str = "n_occ",
        h_pred_key: str = _keys.HAMILTONIAN_KEY,
        h_ref_key: str = _keys.HAMILTONIAN_KEY,
        overlap_key: str = _keys.OVERLAP_KEY,
        e_cut_below_fermi: Optional[float] = 2.0,
        e_cut_above_fermi: Optional[float] = 2.0,
        lambda_pp: float = 1.0,
        lambda_pq: float = 1.0,
        lambda_qq: float = 0.02,
        lambda_trace: float = 0.05,
        lambda_diag: float = 0.1,
        moment_orders: Union[str, int, Iterable[int]] = (1, 2),
        gauge_mu: bool = True,
        overlap_eig_floor: float = 1.0e-10,
        max_kpoints: Optional[int] = 8,
        random_kpoints: bool = True,
        skip_if_missing_dense: bool = True,
        use_eigenvalue_builder: bool = True,
        coeff_align: float = 1.0,
        coeff_base: float = 0.0,
        base_loss_options: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.idp = idp
        self.overlap = bool(overlap)
        self.n_occ = None if n_occ is None else int(n_occ)
        self.n_occ_key = str(n_occ_key)
        self.h_pred_key = str(h_pred_key)
        self.h_ref_key = str(h_ref_key)
        self.overlap_key = str(overlap_key)
        self.e_cut_below_fermi = None if e_cut_below_fermi is None else float(e_cut_below_fermi)
        self.e_cut_above_fermi = None if e_cut_above_fermi is None else float(e_cut_above_fermi)
        self.lambda_pp = float(lambda_pp)
        self.lambda_pq = float(lambda_pq)
        self.lambda_qq = float(lambda_qq)
        self.lambda_trace = float(lambda_trace)
        self.lambda_diag = float(lambda_diag)
        self.moment_orders = _as_tuple_of_ints(moment_orders, default=(1, 2))
        self.gauge_mu = bool(gauge_mu)
        self.overlap_eig_floor = float(overlap_eig_floor)
        self.max_kpoints = None if max_kpoints is None else int(max_kpoints)
        self.random_kpoints = bool(random_kpoints)
        self.skip_if_missing_dense = bool(skip_if_missing_dense)
        self.coeff_align = float(coeff_align)
        self.coeff_base = float(coeff_base)
        self.last_scalar_state: Dict[str, Tensor] = {}

        self.base_loss = None
        if base_loss_options:
            opts = dict(base_loss_options)
            opts.pop("enabled", None)
            if "method" in opts:
                self.base_loss = Loss(**opts, idp=idp, basis=basis, dtype=dtype, device=device, overlap=overlap, **kwargs)

        self.eigenvalue = None
        if use_eigenvalue_builder and Eigenvalues is not None and idp is not None:
            self.eigenvalue = Eigenvalues(
                idp=idp,
                h_edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                h_node_field=AtomicDataDict.NODE_FEATURES_KEY,
                h_out_field=AtomicDataDict.HAMILTONIAN_KEY,
                out_field=AtomicDataDict.ENERGY_EIGENVALUE_KEY,
                s_edge_field=AtomicDataDict.EDGE_OVERLAP_KEY if self.overlap else None,
                s_node_field=AtomicDataDict.NODE_OVERLAP_KEY if self.overlap else None,
                s_out_field=AtomicDataDict.OVERLAP_KEY if self.overlap else None,
                dtype=dtype,
                device=device,
            )

    @staticmethod
    def _dict_get(data: Mapping[str, Any], key: str, default=None):
        try:
            return data[key]
        except Exception:
            return default

    def _with_dense(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._dict_get(data, self.h_pred_key if self.h_pred_key != _keys.HAMILTONIAN_KEY else _keys.HAMILTONIAN_KEY) is not None:
            return data
        if self.eigenvalue is None:
            return data
        return self.eigenvalue(data)  # type: ignore[misc]

    def _resolve_n_occ(self, data: Mapping[str, Any], ref_data: Mapping[str, Any]) -> Union[int, Tensor, None]:
        for src in (ref_data, data):
            val = self._dict_get(src, self.n_occ_key, None)
            if val is not None:
                return val
        return self.n_occ

    def _zero_with_state(self, like: Optional[Tensor], skipped: float = 1.0) -> Tensor:
        if like is None:
            zero = torch.zeros((), device=self.device, dtype=self.dtype)
        else:
            zero = like.real.new_zeros(())
        self.last_scalar_state = {
            "riem_align_loss": zero.detach(),
            "riem_pp_loss": zero.detach(),
            "riem_pq_loss": zero.detach(),
            "riem_qq_loss": zero.detach(),
            "riem_trace_loss": zero.detach(),
            "riem_diag_loss": zero.detach(),
            "riem_skipped": zero.detach() + float(skipped),
        }
        return zero

    def forward(self, data: Mapping[str, Any], ref_data: Optional[Mapping[str, Any]] = None) -> Tensor:
        if ref_data is None:
            ref_data = data

        pred_data = self._with_dense(data)
        ref_dense_data = self._with_dense(ref_data)
        h_pred = self._dict_get(pred_data, self.h_pred_key, None)
        h_ref = self._dict_get(ref_dense_data, self.h_ref_key, None)
        if h_pred is None or h_ref is None:
            if self.skip_if_missing_dense:
                return self._zero_with_state(h_pred if torch.is_tensor(h_pred) else h_ref)
            raise KeyError(
                f"riemannian_align requires dense `{self.h_pred_key}`/`{self.h_ref_key}`; "
                "enable use_eigenvalue_builder or emit dense k-space Hamiltonians."
            )
        if not torch.is_tensor(h_pred) or not torch.is_tensor(h_ref):
            h_pred = torch.as_tensor(h_pred, device=self.device, dtype=self.dtype)
            h_ref = torch.as_tensor(h_ref, device=self.device, dtype=self.dtype)
        h_pred = h_pred.to(device=self.device)
        h_ref = h_ref.to(device=h_pred.device, dtype=h_pred.dtype)

        n_occ = self._resolve_n_occ(pred_data, ref_dense_data)
        if n_occ is None:
            if self.skip_if_missing_dense:
                return self._zero_with_state(h_pred)
            raise KeyError(f"riemannian_align needs `{self.n_occ_key}` or an explicit n_occ option")

        s = self._dict_get(ref_dense_data, self.overlap_key, None)
        if s is not None:
            s = torch.as_tensor(s, device=h_pred.device, dtype=h_pred.dtype)

        align_loss, stats = dense_riemannian_alignment_loss(
            h_pred,
            h_ref,
            s,
            n_occ=n_occ,
            e_cut_below_fermi=self.e_cut_below_fermi,
            e_cut_above_fermi=self.e_cut_above_fermi,
            lambda_pp=self.lambda_pp,
            lambda_pq=self.lambda_pq,
            lambda_qq=self.lambda_qq,
            lambda_trace=self.lambda_trace,
            lambda_diag=self.lambda_diag,
            moment_orders=self.moment_orders,
            gauge_mu=self.gauge_mu,
            overlap_eig_floor=self.overlap_eig_floor,
            max_kpoints=self.max_kpoints,
            random_kpoints=self.random_kpoints,
        )
        self.last_scalar_state = {k: v.detach() if torch.is_tensor(v) else h_pred.real.new_tensor(float(v)) for k, v in stats.items()}

        total = self.coeff_align * align_loss
        if self.base_loss is not None and self.coeff_base != 0.0:
            base_loss = self.base_loss(data, ref_data)
            total = total + self.coeff_base * base_loss
            self.last_scalar_state["riem_base_loss"] = base_loss.detach() if torch.is_tensor(base_loss) else h_pred.real.new_tensor(float(base_loss))
        return total
