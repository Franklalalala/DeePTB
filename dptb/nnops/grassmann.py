# SPDX-License-Identifier: LGPL-3.0-or-later
"""Grassmann-manifold toolkit and occupied-projector (P) regression for DeePTB.

Motivation
----------
The 2026-07-04 audit showed that a Hamiltonian predictor's element-wise error is
organized in the spectrum: the harmful part lives in high-energy virtual states,
while the near-Fermi / occupied subspace is what band structure actually reads.
The occupied projector ``P`` (a.k.a. the S-metric density matrix) is the object
that naturally lives on a curved manifold: the set of rank-``n_occ`` idempotent
projectors is the **Grassmann manifold** ``Gr(n_occ, N)``.  This module implements
the Grassmann geometry and a training loss that regresses ``P`` instead of raw
matrix elements, so model capacity is spent on the occupied subspace.

Design
------
* All geometry is done in the **symmetrically orthogonalized basis** ``S^{1/2}``
  so the overlap becomes identity and standard (orthonormal) Grassmann formulas
  apply.  AO <-> orthogonalized transport uses ``X = S^{1/2}`` / ``X^{-1}``.
* The occupied projector of ``H`` is a **smooth** function of ``H`` when a HOMO-LUMO
  gap is present (insulators/semiconductors).  The ``eigh`` gradient scales like
  ``1/gap`` at the occupied/virtual boundary, so it is well conditioned only while
  that gap stays open.  Guards: eigensolves run in float64 (``_stable_eigh``);
  ``min_gap`` rejects metallic *reference* records; the optional ``check_pred_gap``
  rejects a metallic *predicted* spectrum (off by default -- an early-training model
  can transiently close its gap, and rejecting every such record would stall
  training); and the trainer's ``clip_grad`` bounds any residual spike.
* The reference Grassmann point is built under ``no_grad``; gradients flow only
  through the predicted Hamiltonian, exactly mirroring ``riemannian_alignment``.
* The default training distance is the **chordal** (Frobenius) distance
  ``||P_pred - P_ref||_F^2``: it is smooth everywhere *including at the optimum*.
  The **geodesic** (principal-angle) distance is provided for diagnostics and for
  the product-manifold flow story, but its gradient is singular at zero distance,
  so it is not the default loss.

This module reuses the numerically-hardened helpers from
``dptb.nnops.riemannian_alignment`` (the manifold-alignment "legacy") and adds the
Grassmann layer on top of them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# --- single framework-free numerical core (no fallback redefinition, no drift) ---
from dptb.nnops._manifold_math import (
    hermitian_part,
    regularize_overlap,
    generalized_eigh,
    stable_eigh as _stable_eigh,
    s_half_and_inv,
    occupied_projector,
    idempotency_error,
    chordal_distance_sq,
    principal_angles,
    geodesic_distance,
    grassmann_exp,
    grassmann_log,
    mcweeny_purify,
)
from dptb.nnops._manifold_loss_base import (
    flatten_dense_mats,
    flatten_optional_overlap,
    select_matrix_indices,
    n_occ_for_flat_index,
    dict_get,
    ManifoldLossModule,
)

try:
    from dptb.nnops.loss import Loss
except ImportError:  # pragma: no cover - lets the loss module import without the registry
    class _DummyLoss:
        @staticmethod
        def register(_name):
            def deco(cls):
                return cls
            return deco
    Loss = _DummyLoss()

try:
    from dptb.data import AtomicDataDict, _keys
except ImportError:  # pragma: no cover
    class _Keys:
        HAMILTONIAN_KEY = "hamiltonian"
        OVERLAP_KEY = "overlap"
        KPOINT_KEY = "kpoint"
        NODE_FEATURES_KEY = "node_features"
        EDGE_FEATURES_KEY = "edge_features"
        NODE_OVERLAP_KEY = "node_overlap"
        EDGE_OVERLAP_KEY = "edge_overlap"
    _keys = _Keys()

    class AtomicDataDict:  # type: ignore[no-redef]
        HAMILTONIAN_KEY = _keys.HAMILTONIAN_KEY
        OVERLAP_KEY = _keys.OVERLAP_KEY
        KPOINT_KEY = _keys.KPOINT_KEY
        NODE_FEATURES_KEY = _keys.NODE_FEATURES_KEY
        EDGE_FEATURES_KEY = _keys.EDGE_FEATURES_KEY
        NODE_OVERLAP_KEY = _keys.NODE_OVERLAP_KEY
        EDGE_OVERLAP_KEY = _keys.EDGE_OVERLAP_KEY

try:
    from dptb.nn.hr2hk import HR2HK
except ImportError:  # pragma: no cover
    HR2HK = None

Tensor = torch.Tensor


class SkippableRecord(Exception):
    """A record that is legitimately unusable for the Grassmann objective.

    Raised ONLY for expected physical/data conditions (metallic / near-degenerate
    gap, missing overlap for a non-orthonormal basis, unresolved ``n_occ``).  The
    registered loss swallows *only* this exception under ``skip_on_error`` -- every
    other error (shape/device/eigh-convergence/HR2HK bug) propagates, so a real bug
    never turns into a silent zero-loss no-op.
    """


# ============================================================================
# Dense P-regression loss core (unit-testable, framework-free)
# ============================================================================
@dataclass(frozen=True)
class GrassmannPResult:
    loss: Tensor
    loss_chordal: Tensor
    loss_eps: Tensor
    geo_dist: Tensor
    n_occ: int
    n_bands: int
    gap: Tensor
    pred_gap: Tensor
    gap_weight: Tensor


def grassmann_p_loss_single(
    h_pred: Tensor,
    h_ref: Tensor,
    s: Optional[Tensor] = None,
    *,
    n_occ: int,
    lambda_chordal: float = 1.0,
    lambda_eps: float = 0.0,
    eps_window: Optional[int] = None,
    gauge_mu: bool = True,
    eig_floor: float = 1.0e-10,
    from_density: bool = False,
    min_gap: float = 0.0,
    check_pred_gap: bool = False,
    soft_pred_gap: bool = True,
    chordal_normalize: bool = False,
) -> GrassmannPResult:
    """Single-k Grassmann P-regression loss.

    * ``lambda_chordal``: weight on the occupied-subspace chordal distance
      (constrains *which states are occupied* -- the Grassmann object).
    * ``lambda_eps``: optional weight on occupied (+ small virtual window)
      eigenvalue matching, which supplies band *energies* that the pure subspace
      distance does not carry.  A scalar gauge ``mu`` (Fermi shift) is removed.
      Not meaningful when ``from_density=True`` (occupations, not energies).
    * ``from_density``: treat ``h_pred``/``h_ref`` as density matrices and take the
      ``n_occ`` highest-occupation eigenvectors (DM-regression retraction).
    * ``soft_pred_gap`` (default): when a scale ``min_gap`` is set and the *predicted*
      occ/vir gap falls below it, multiply the chordal term by a detached weight
      ``(gap/min_gap)^2 in [0,1)`` so the ``eigh`` eigenvector gradient (which scales
      like ``1/gap``) stays bounded (``~ gap -> 0``) instead of spiking on an early,
      transiently metallic prediction.  The record is kept, not dropped; the eigenvalue
      term (gradient-safe, and what actually drives the gap open) keeps full weight.
      This is the smooth alternative to ``check_pred_gap`` (which hard-skips).  Ignored
      when ``check_pred_gap`` is set (strict wins) or ``min_gap == 0``.
    """
    x = x_inv = None
    if s is not None:
        with torch.no_grad():
            x, x_inv = s_half_and_inv(s.detach(), eig_floor=eig_floor)
    sel = slice(-n_occ, None) if from_density else slice(0, n_occ)

    def _project(h, grad: bool):
        h_h = hermitian_part(h)
        if x_inv is None:
            h_orth = h_h
        else:
            transport = x if from_density else x_inv  # X D X for density, X^{-1} H X^{-1} for H
            h_orth = hermitian_part(transport @ h_h @ transport)
        if grad:
            eps, q = _stable_eigh(h_orth)
        else:
            with torch.no_grad():
                eps, q = _stable_eigh(h_orth)
        u = q[:, sel]
        return u @ u.mH, u, eps

    def _boundary_gap(e: Tensor) -> Tensor:
        b = e.shape[-1] - n_occ if from_density else n_occ  # occ/vir boundary index
        return e[b] - e[b - 1]

    with torch.no_grad():
        p_ref, u_ref, eps_ref = _project(h_ref.detach(), grad=False)
    p_pred, u_pred, eps_pred = _project(h_pred, grad=True)

    with torch.no_grad():
        gap = _boundary_gap(eps_ref).detach()
        pred_gap = _boundary_gap(eps_pred).detach()

    # Guards keyed to the occ/vir boundary gap (where the eigh eigenvector gradient
    # ~1/gap lives).  A metallic *reference* target is always unusable (skip); a
    # metallic *predicted* spectrum is either hard-skipped (check_pred_gap) or, by
    # default, softly down-weighted below (soft_pred_gap) so the record is kept.
    if min_gap and float(min_gap) > 0.0:
        kind = "occupation" if from_density else "HOMO-LUMO"
        if (not torch.isfinite(gap)) or float(gap.abs()) < float(min_gap):
            raise SkippableRecord(f"reference {kind} gap {float(gap):.3g} < min_gap {float(min_gap):.3g} "
                                  "(metallic/near-degenerate: projector is ill-conditioned)")
        if check_pred_gap and ((not torch.isfinite(pred_gap)) or float(pred_gap.abs()) < float(min_gap)):
            raise SkippableRecord(f"predicted {kind} gap {float(pred_gap):.3g} < min_gap {float(min_gap):.3g}")

    loss_chordal = chordal_distance_sq(p_pred, p_ref)
    # Raw chordal = sum_i sin^2(theta_i) in [0, n_occ]; its magnitude therefore grows
    # with n_occ / system size, which distorts the effective LR when mixing systems of
    # different size (or against a fixed-coeff base loss).  Optionally normalize per
    # occupied state so the term is O(1) regardless of size.
    chordal_term = loss_chordal / max(int(n_occ), 1) if chordal_normalize else loss_chordal

    # Soft predicted-gap floor: bound the 1/gap eigenvector gradient of the chordal term
    # without dropping a transiently-metallic prediction.  w = (gap/min_gap)^2 clamped to
    # [0,1] is detached, so it scales the chordal gradient by ~gap^2 -> the effective
    # gradient magnitude ~ gap stays finite as the predicted gap closes; w=1 (no effect)
    # once the gap is healthy.
    gap_weight = h_pred.real.new_ones(())
    if soft_pred_gap and not check_pred_gap and min_gap and float(min_gap) > 0.0:
        if torch.isfinite(pred_gap):
            gap_weight = (pred_gap.abs() / float(min_gap)).clamp(max=1.0).to(gap_weight.dtype) ** 2
        else:
            gap_weight = h_pred.real.new_zeros(())
    chordal_term = chordal_term * gap_weight.detach()

    loss_eps = h_pred.real.new_zeros(())
    if lambda_eps != 0.0 and not from_density:
        n = eps_ref.shape[-1]
        if eps_window is None:
            hi = n
        else:
            hi = min(n, n_occ + int(eps_window))
        lo = 0
        de = eps_pred[lo:hi] - eps_ref[lo:hi].detach()
        if gauge_mu:
            de = de - de.mean()
        loss_eps = (de ** 2).mean()

    loss = lambda_chordal * chordal_term + lambda_eps * loss_eps
    with torch.no_grad():
        geo = geodesic_distance(u_pred, u_ref)

    return GrassmannPResult(
        loss=loss,
        loss_chordal=loss_chordal.detach(),
        loss_eps=loss_eps.detach() if torch.is_tensor(loss_eps) else loss_eps,
        geo_dist=geo,
        n_occ=int(n_occ),
        n_bands=int(eps_ref.shape[-1]),
        gap=gap,
        pred_gap=pred_gap,
        gap_weight=gap_weight.detach(),
    )


def dense_grassmann_p_loss(
    h_pred: Tensor,
    h_ref: Tensor,
    s: Optional[Tensor] = None,
    *,
    n_occ: Union[int, Sequence[int], Tensor],
    lambda_chordal: float = 1.0,
    lambda_eps: float = 0.0,
    eps_window: Optional[int] = None,
    gauge_mu: bool = True,
    eig_floor: float = 1.0e-10,
    max_kpoints: Optional[int] = None,
    random_kpoints: bool = False,
    from_density: bool = False,
    min_gap: float = 0.0,
    check_pred_gap: bool = False,
    soft_pred_gap: bool = True,
    chordal_normalize: bool = False,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Batched (over k / matrices) Grassmann P-regression loss."""
    if h_pred.shape != h_ref.shape:
        raise ValueError(f"h_pred/h_ref shape mismatch: {tuple(h_pred.shape)} vs {tuple(h_ref.shape)}")
    fp = flatten_dense_mats(h_pred)
    fr = flatten_dense_mats(h_ref)
    fs = flatten_optional_overlap(s, fp)
    if fs is not None and fs.shape != fp.shape:
        raise ValueError(f"S shape mismatch after flatten: {tuple(fs.shape)} vs {tuple(fp.shape)}")

    n_mats = fp.shape[0]
    idx = select_matrix_indices(n_mats, max_kpoints, random_kpoints, fp.device)

    results = []
    for j in idx.tolist():
        ss = None if fs is None else fs[int(j)]
        results.append(
            grassmann_p_loss_single(
                fp[int(j)], fr[int(j)], ss,
                n_occ=n_occ_for_flat_index(n_occ, int(j), n_mats),
                lambda_chordal=lambda_chordal,
                lambda_eps=lambda_eps,
                eps_window=eps_window,
                gauge_mu=gauge_mu,
                eig_floor=eig_floor,
                from_density=from_density,
                min_gap=min_gap,
                check_pred_gap=check_pred_gap,
                soft_pred_gap=soft_pred_gap,
                chordal_normalize=chordal_normalize,
            )
        )
    if not results:
        z = fp.real.new_zeros(())
        return z, {"grassmann_skipped": fp.real.new_ones(())}

    loss = torch.stack([r.loss for r in results]).mean()
    stats = {
        "grassmann_loss": loss.detach(),
        "grassmann_chordal": torch.stack([r.loss_chordal for r in results]).mean(),
        "grassmann_eps": torch.stack([torch.as_tensor(r.loss_eps) for r in results]).mean(),
        "grassmann_pred_gap": torch.stack([r.pred_gap for r in results]).mean(),
        "grassmann_gap_weight": torch.stack([r.gap_weight for r in results]).mean(),
        "grassmann_geo_dist": torch.stack([r.geo_dist for r in results]).mean(),
        "grassmann_gap": torch.stack([r.gap for r in results]).mean(),
        "grassmann_rank": fp.real.new_tensor(float(sum(r.n_occ for r in results) / len(results))),
        "grassmann_skipped": fp.real.new_zeros(()),
    }
    return loss, stats


# ============================================================================
# Registered DeePTB loss
# ============================================================================
def _valence_electron_table() -> Dict[int, int]:
    """Minimal pseudopotential valence-electron count fallback (extend as needed)."""
    return {
        1: 1, 2: 2, 3: 3, 4: 4, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8,
        11: 9, 12: 10, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
        19: 9, 20: 10, 31: 3, 32: 4, 33: 5, 34: 6, 35: 7,
        49: 3, 50: 4, 51: 5, 52: 6, 53: 7,
    }


@Loss.register("grassmann_p_align")
@Loss.register("p_regression")
class GrassmannPAlignLoss(ManifoldLossModule, nn.Module):
    """Occupied-projector (Grassmann) regression loss for non-SOC band prediction.

    Assembles dense ``H(k)``/``S(k)`` from predicted vs. reference features via
    ``HR2HK`` at a fixed k-set, forms the occupied projectors, and penalizes their
    chordal distance (plus an optional occupied-eigenvalue term for band energies).

    Use as a normal ``loss_options.train`` method::

        loss_options:
          train:
            method: grassmann_p_align
            coeff_align: 1.0
            lambda_chordal: 1.0
            lambda_eps: 0.1
            base_loss_options: {method: hamil_abs, coeff_base: 0.0}
    """

    def __init__(
        self,
        basis: Optional[dict] = None,
        idp: Any = None,
        overlap: bool = True,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        n_occ: Optional[int] = None,
        n_occ_key: str = "n_occ",
        kpoints: Optional[Sequence[Sequence[float]]] = None,
        lambda_chordal: float = 1.0,
        lambda_eps: float = 0.1,
        eps_window: Optional[int] = 8,
        gauge_mu: bool = True,
        eig_floor: float = 1.0e-10,
        max_kpoints: Optional[int] = None,
        coeff_align: float = 1.0,
        coeff_base: float = 0.0,
        base_loss_options: Optional[Mapping[str, Any]] = None,
        valence_fallback: bool = True,
        all_electron: bool = False,
        skip_on_error: bool = False,
        from_density: bool = False,
        density_key: str = "density_matrix",
        min_gap: float = 0.0,
        check_pred_gap: bool = False,
        soft_pred_gap: bool = True,
        chordal_normalize: bool = False,
        require_overlap: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.from_density = bool(from_density)
        self.density_key = str(density_key)
        self.min_gap = float(min_gap)
        self.check_pred_gap = bool(check_pred_gap)
        self.soft_pred_gap = bool(soft_pred_gap)
        self.chordal_normalize = bool(chordal_normalize)
        self.dtype = dtype
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.idp = idp
        self.overlap = bool(overlap)
        self.n_occ = None if n_occ is None else int(n_occ)
        self.n_occ_key = str(n_occ_key)
        self.kpoints = [[0.0, 0.0, 0.0]] if kpoints is None else [list(map(float, k)) for k in kpoints]
        self.lambda_chordal = float(lambda_chordal)
        self.lambda_eps = float(lambda_eps)
        self.eps_window = None if eps_window is None else int(eps_window)
        self.gauge_mu = bool(gauge_mu)
        self.eig_floor = float(eig_floor)
        self.max_kpoints = None if max_kpoints is None else int(max_kpoints)
        self.coeff_align = float(coeff_align)
        self.coeff_base = float(coeff_base)
        self.valence_fallback = bool(valence_fallback)
        self.all_electron = bool(all_electron)
        self.skip_on_error = bool(skip_on_error)
        self.require_overlap = bool(self.overlap if require_overlap is None else require_overlap)
        self.last_scalar_state: Dict[str, Tensor] = {}
        self._vtab = _valence_electron_table()

        self.base_loss = self._build_base_loss(
            base_loss_options, Loss,
            idp=idp, basis=basis, dtype=dtype, device=device, overlap=overlap, **kwargs)

        self.h2k = self.s2k = None
        if HR2HK is not None and idp is not None:
            self.h2k = HR2HK(
                idp=idp,
                edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.HAMILTONIAN_KEY,
                dtype=dtype, device=device,
            )
            # Build the S(k) assembler whenever an overlap is wanted. `self.overlap`
            # reflects whether the *model* predicts overlap; the reference S we need
            # for the Grassmann projector comes from the dataset (get_overlap) and is
            # gated by require_overlap -- so decouple from the model-overlap flag.
            if self.overlap or self.require_overlap:
                self.s2k = HR2HK(
                    idp=idp, overlap=True,
                    edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                    out_field=AtomicDataDict.OVERLAP_KEY,
                    dtype=dtype, device=device,
                )

    _get = staticmethod(dict_get)  # shared, tolerant AtomicData getter

    def _atomic_numbers(self, data, ref_data):
        """Return per-atom Z as a list[int], or None.

        Prefers an explicit ``atomic_numbers`` field; a processed AtomicData batch
        usually carries only ``atom_types`` (basis indices), so fall back to mapping
        those through the idp's ``type_names`` (element symbols) -> Z.
        """
        akey = AtomicDataDict.ATOMIC_NUMBERS_KEY if hasattr(AtomicDataDict, "ATOMIC_NUMBERS_KEY") else "atomic_numbers"
        for src in (ref_data, data):
            z = self._get(src, akey, None)
            if z is not None:
                return [int(v) for v in torch.as_tensor(z).reshape(-1).tolist()]
        tkey = AtomicDataDict.ATOM_TYPE_KEY if hasattr(AtomicDataDict, "ATOM_TYPE_KEY") else "atom_types"
        type_names = getattr(self.idp, "type_names", None)
        if type_names is not None:
            try:
                from ase.data import atomic_numbers as _AN
                for src in (ref_data, data):
                    t = self._get(src, tkey, None)
                    if t is not None:
                        return [int(_AN[type_names[int(ti)]]) for ti in torch.as_tensor(t).reshape(-1).tolist()]
            except Exception:
                return None
        return None

    def _resolve_n_occ(self, data, ref_data):
        for src in (ref_data, data):
            v = self._get(src, self.n_occ_key, None)
            if v is not None:
                return int(v.reshape(-1)[0].item()) if torch.is_tensor(v) else int(v)
        if self.n_occ is not None:
            return self.n_occ
        zs = self._atomic_numbers(data, ref_data)
        if self.all_electron and zs:
            # All-electron, neutral, closed-shell: n_occ = (sum of Z) / 2 exactly.
            # Correct for all-electron GTO/def2 data where core states are occupied
            # (the pseudopotential-valence table below would undercount).
            nel = sum(zs)
            if nel > 0 and nel % 2 == 0:
                return nel // 2
        if self.valence_fallback and zs:
            if any(zi not in self._vtab for zi in zs):
                return None  # unknown species -> force explicit n_occ, never a silent guess
            nel = sum(self._vtab[zi] for zi in zs)
            if nel > 0 and nel % 2 == 0:
                return nel // 2
        return None

    def _num_structures(self, data: Mapping[str, Any]) -> int:
        ptr = self._get(data, AtomicDataDict.BATCH_PTR_KEY if hasattr(AtomicDataDict, "BATCH_PTR_KEY") else "ptr", None)
        if torch.is_tensor(ptr) and ptr.numel() >= 2:
            return int(ptr.numel() - 1)
        batch = self._get(data, AtomicDataDict.BATCH_KEY if hasattr(AtomicDataDict, "BATCH_KEY") else "batch", None)
        if torch.is_tensor(batch) and batch.numel() > 0:
            return int(batch.detach().max().item()) + 1
        return 1

    def _build_hk(self, data, module, out_key) -> Tensor:
        kpts = torch.as_tensor(self.kpoints, device=self.device, dtype=self.dtype)
        d = dict(data)
        d[AtomicDataDict.KPOINT_KEY] = kpts
        d = module(d)
        return d[out_key]

    def _zero(self, like, skipped=1.0):
        z = self._zero_scalar(like if torch.is_tensor(like) else None)
        self.last_scalar_state = {"grassmann_loss": z.detach(), "grassmann_skipped": z.detach() + float(skipped)}
        return z

    def forward(self, data: Mapping[str, Any], ref_data: Optional[Mapping[str, Any]] = None) -> Tensor:
        if ref_data is None:
            ref_data = data
        if self.h2k is None:
            raise RuntimeError("GrassmannPAlignLoss requires HR2HK and an idp.")
        h_pred = h_ref = None
        try:
            # This dense, single-structure path assumes one system per loss call.
            # Reject PyG batches until per-structure HR2HK/n_occ handling exists.
            if self._num_structures(data) != 1 or self._num_structures(ref_data) != 1:
                raise NotImplementedError(
                    "grassmann_p_align currently assumes one structure per loss call; use "
                    "batch_size=1 (or implement per-structure HR2HK/n_occ before batch>1).")
            n_occ = self._resolve_n_occ(data, ref_data)
            if n_occ is None:
                raise SkippableRecord(
                    f"grassmann_p_align needs `{self.n_occ_key}`, an explicit n_occ, "
                    "all_electron, or a known-species valence fallback")
            if self.from_density:
                # DM route (dataset get_DM). Prefer an explicit dense density field;
                # otherwise the node/edge feature slots hold density blocks under
                # get_DM and HR2HK assembles the density the same way it assembles H.
                dp, dr = self._get(data, self.density_key), self._get(ref_data, self.density_key)
                if torch.is_tensor(dp) and torch.is_tensor(dr):
                    h_pred = dp.to(device=self.device)
                    h_ref = dr.to(device=self.device, dtype=h_pred.dtype)
                else:
                    h_pred = self._build_hk(data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
                    h_ref = self._build_hk(ref_data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
            else:
                h_pred = self._build_hk(data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
                h_ref = self._build_hk(ref_data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
            s = None
            if self.s2k is not None and self._get(ref_data, AtomicDataDict.EDGE_OVERLAP_KEY) is not None:
                s = self._build_hk(ref_data, self.s2k, AtomicDataDict.OVERLAP_KEY)
            if self.require_overlap and s is None:
                raise SkippableRecord("grassmann_p_align: require_overlap is set but no overlap was assembled "
                                      "(a non-orthogonal NAO basis must not be treated as orthonormal)")
            loss, stats = dense_grassmann_p_loss(
                h_pred, h_ref, s,
                n_occ=n_occ,
                lambda_chordal=self.lambda_chordal,
                lambda_eps=self.lambda_eps,
                eps_window=self.eps_window,
                gauge_mu=self.gauge_mu,
                eig_floor=self.eig_floor,
                max_kpoints=self.max_kpoints,
                chordal_normalize=self.chordal_normalize,
                from_density=self.from_density,
                min_gap=self.min_gap,
                check_pred_gap=self.check_pred_gap,
                soft_pred_gap=self.soft_pred_gap,
            )
        except SkippableRecord as exc:
            # Expected physical/data skip (metallic gap, missing overlap, unresolved
            # n_occ).  Only these are swallowed under skip_on_error; a genuine bug
            # (shape/device/eigh/HR2HK) is NOT a SkippableRecord and propagates, so it
            # can never masquerade as a silent zero-loss no-op.
            if self.skip_on_error:
                log.warning("grassmann_p_align skipped a record: %s", exc)
                return self._zero(h_pred if torch.is_tensor(h_pred) else h_ref)
            raise
        self.last_scalar_state = {k: (v.detach() if torch.is_tensor(v) else torch.as_tensor(float(v))) for k, v in stats.items()}
        total = self.coeff_align * loss
        if self.base_loss is not None and self.coeff_base != 0.0:
            base = self.base_loss(data, ref_data)
            total = total + self.coeff_base * base
            self.last_scalar_state["grassmann_base_loss"] = base.detach() if torch.is_tensor(base) else torch.as_tensor(float(base))
        return total


__all__ = [
    "SkippableRecord",
    "s_half_and_inv",
    "occupied_projector",
    "chordal_distance_sq",
    "principal_angles",
    "geodesic_distance",
    "grassmann_exp",
    "grassmann_log",
    "mcweeny_purify",
    "idempotency_error",
    "GrassmannPResult",
    "grassmann_p_loss_single",
    "dense_grassmann_p_loss",
    "GrassmannPAlignLoss",
]
