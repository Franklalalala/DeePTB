"""Hamiltonian-anchored spectral losses and their explicit contracts."""

import logging
import math
from typing import Dict, Union
import torch
from torch import nn
from dptb.data import AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.energy import Eigenvalues
from dptb.nn.hr2hk import HR2HK
from .loss import Loss, HamilLossAbs, _unnest

log = logging.getLogger(__name__)


@Loss.register("eig_ham_h0res")
class EigHamH0ResLoss(nn.Module):
    """Joint matrix and band loss for an H0-residual target.

    ``eig_ham`` assumes NODE/EDGE_FEATURES already hold the physical
    Hamiltonian. Residual training predicts dH against a stored H0 prior, so the
    band term has to diagonalize H0 + dH while the matrix term keeps scoring
    the residual exactly as pretraining did -- otherwise the two halves of the
    objective disagree about what the model outputs.

    total = coeff_ham * L_ham(residual) + (1 - coeff_ham) * L_eig(H0 + residual)

    The matrix term anchors residual predictions while the spectral term
    constrains the eigenvalues of the reconstructed physical Hamiltonian.
    """

    def __init__(
        self,
        basis: Dict[str, Union[str, list]] = None,
        idp: Union[OrbitalMapper, None] = None,
        band_overlap: bool = True,
        coeff_ham: float = 0.9,
        band_emin: float = None,
        band_emax: float = None,
        band_min: int = 0,
        band_max: int = None,
        eout_weight: float = 0.01,
        diff_on: bool = False,
        diff_weight: float = 0.01,
        diff_valence: dict = None,
        spin_deg: int = 2,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        super(EigHamH0ResLoss, self).__init__()
        if not 0.0 <= coeff_ham <= 1.0:
            raise ValueError(f"coeff_ham must be in [0, 1], got {coeff_ham}.")
        self.coeff_ham = float(coeff_ham)
        self.device = device

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        # Same matrix loss the pretraining run used, so the anchor term is
        # numerically comparable to the checkpoint's own train_loss.
        # Trainer merges common_options into every loss kwargs, so basis /
        # overlap / dtype / device arrive twice; drop the copies passed
        # explicitly. overlap stays False: this model predicts no S, and the
        # matrix term must score exactly what pretraining scored.
        ham_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("basis", "overlap", "dtype", "device", "idp")
        }
        self.ham_loss = HamilLossAbs(
            idp=self.idp, overlap=False, dtype=dtype, device=device, **ham_kwargs
        )
        # Direct eigensolver rather than EigLoss: see module note on the
        # Batch.from_dict/to_data_list mis-slice.
        self.eigen = Eigenvalues(
            idp=self.idp,
            h_edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            h_node_field=AtomicDataDict.NODE_FEATURES_KEY,
            h_out_field=AtomicDataDict.HAMILTONIAN_KEY,
            out_field=AtomicDataDict.ENERGY_EIGENVALUE_KEY,
            s_edge_field=AtomicDataDict.EDGE_OVERLAP_KEY if band_overlap else None,
            s_node_field=AtomicDataDict.NODE_OVERLAP_KEY if band_overlap else None,
            s_out_field=AtomicDataDict.OVERLAP_KEY if band_overlap else None,
            dtype=dtype,
            device=device,
        )
        self.eout_weight = eout_weight

        self.band_emin = band_emin
        self.band_emax = band_emax
        self.band_min = band_min
        self.band_max = band_max
        self._last_parts = {}

    def _add_h0(self, src: AtomicDataDict) -> dict:
        """Shallow copy with the physical Hamiltonian in the feature fields."""
        for key in (AtomicDataDict.NODE_H0_KEY, AtomicDataDict.EDGE_H0_KEY):
            if src.get(key, None) is None:
                raise KeyError(
                    f"eig_ham_h0res needs {key!r} to rebuild the physical "
                    "Hamiltonian; run the dataset with get_H0=true."
                )
        out = dict(src)
        out[AtomicDataDict.NODE_FEATURES_KEY] = (
            src[AtomicDataDict.NODE_FEATURES_KEY] + src[AtomicDataDict.NODE_H0_KEY]
        )
        out[AtomicDataDict.EDGE_FEATURES_KEY] = (
            src[AtomicDataDict.EDGE_FEATURES_KEY] + src[AtomicDataDict.EDGE_H0_KEY]
        )
        return out

    # Only these reach the eigensolver. A collated batch also carries nested
    # k-point/eigenvalue tensors and batch bookkeeping, which HR2HK cannot size.
    _SOLVER_FIELDS = (
        AtomicDataDict.EDGE_INDEX_KEY,
        AtomicDataDict.EDGE_CELL_SHIFT_KEY,
        AtomicDataDict.POSITIONS_KEY,
        AtomicDataDict.CELL_KEY,
        AtomicDataDict.PBC_KEY,
        AtomicDataDict.ATOM_TYPE_KEY,
        AtomicDataDict.EDGE_TYPE_KEY,
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
        AtomicDataDict.NODE_FEATURES_KEY,
        AtomicDataDict.EDGE_FEATURES_KEY,
    )

    def _solver_dict(self, src: dict, ref: dict) -> dict:
        """Assemble the eigensolver input.

        Hamiltonian RMEs come from ``src`` (the model output); the overlap and
        the k-points come from ``ref`` (the untouched batch). The model
        overwrites EDGE_OVERLAP_KEY with internal 128-dim latents during
        forward, so reading S from the prediction gives garbage of the wrong
        width.
        """
        out = {}
        for key in self._SOLVER_FIELDS:
            value = src.get(key, None)
            if value is not None:
                out[key] = _unnest(value)

        n_rme = self.idp.reduced_matrix_element
        for key in (AtomicDataDict.NODE_OVERLAP_KEY, AtomicDataDict.EDGE_OVERLAP_KEY):
            value = _unnest(ref.get(key, None))
            if value is None:
                raise KeyError(
                    f"band term needs {key!r} on the reference batch; run the "
                    "dataset with get_overlap=true."
                )
            if value.shape[-1] != n_rme:
                raise ValueError(
                    f"{key} has width {value.shape[-1]}, expected the RME width "
                    f"{n_rme}. A width of 128 means the model overwrote this "
                    "field with internal latents -- read S from the reference "
                    "batch, not the prediction."
                )
            out[key] = value

        kpoint = _unnest(ref.get(AtomicDataDict.KPOINT_KEY, None))
        if kpoint is None:
            kpoint = _unnest(src.get(AtomicDataDict.KPOINT_KEY, None))
        if kpoint is None:
            raise KeyError("band term needs k-points on the batch.")
        out[AtomicDataDict.KPOINT_KEY] = kpoint.reshape(-1, 3)
        return out

    def _band_loss(self, pred_phys: dict, ref_phys: dict):
        """Windowed MSE between predicted and reference bands, one graph.

        Follows EigLoss's conventions so the number stays comparable: each side
        is shifted by its own minimum, the window is measured in that
        bottom-relative coordinate, and bands outside it are down-weighted to
        eout_weight rather than dropped.
        """
        n_graph = 1
        ptr = pred_phys.get(AtomicDataDict.BATCH_PTR_KEY, None)
        if ptr is not None:
            n_graph = int(ptr.numel()) - 1
        if n_graph != 1:
            raise RuntimeError(
                f"eig_ham_h0res band term expects one graph per batch, got "
                f"{n_graph}. k-points and band counts are ragged across "
                f"structures, so set batch_size=1."
            )

        solver_input = self._solver_dict(pred_phys, ref_phys)
        out = self.eigen(solver_input)
        eig_pred = out[AtomicDataDict.ENERGY_EIGENVALUE_KEY]
        if eig_pred.dim() == 3:
            eig_pred = eig_pred[0]
        eig_ref = _unnest(ref_phys[AtomicDataDict.ENERGY_EIGENVALUE_KEY])
        if torch.is_tensor(eig_ref) and eig_ref.dim() == 3:
            eig_ref = eig_ref[0]
        eig_ref = eig_ref.to(device=eig_pred.device, dtype=eig_pred.dtype)

        if eig_pred.shape[0] != eig_ref.shape[0]:
            raise RuntimeError(
                f"k-point count differs: pred {eig_pred.shape[0]} vs ref "
                f"{eig_ref.shape[0]}."
            )
        nb = min(eig_pred.shape[1], eig_ref.shape[1])
        lo = int(self.band_min)
        hi = int(self.band_max) if self.band_max is not None else nb
        hi = min(hi, nb)
        if lo >= hi:
            raise RuntimeError(f"empty band window: band_min={lo} band_max={hi}.")

        p = eig_pred[:, lo:hi]
        r = eig_ref[:, lo:hi]
        p = p - p.reshape(-1).min()
        r = r - r.reshape(-1).min()

        diff2 = (p - r) ** 2
        if self.band_emin is None and self.band_emax is None:
            return diff2.mean()

        mask = torch.ones_like(r, dtype=torch.bool)
        if self.band_emin is not None:
            mask &= r > self.band_emin
        if self.band_emax is not None:
            mask &= r < self.band_emax
        n_in = int(mask.sum())
        n_out = mask.numel() - n_in
        loss = diff2.new_zeros(())
        if n_in:
            loss = loss + diff2[mask].mean()
        if n_out:
            loss = loss + self.eout_weight * diff2[~mask].mean()
        return loss

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        ham_loss = self.ham_loss(data, ref_data)

        if self.coeff_ham >= 1.0:
            self._last_parts = {"ham": float(ham_loss.detach()), "eig": 0.0}
            return ham_loss

        pred_phys = self._add_h0(data)
        ref_phys = self._add_h0(ref_data)

        # The window lives on the loss, not in the dataset: it is a training
        # knob, and putting it here keeps the LMDB records reusable.
        if self.band_emin is not None or self.band_emax is not None:
            ref_phys[AtomicDataDict.ENERGY_WINDOWS_KEY] = (
                self.band_emin,
                self.band_emax,
            )
        if self.band_max is not None:
            ref_phys[AtomicDataDict.BAND_WINDOW_KEY] = (self.band_min, self.band_max)

        eig_loss = self._band_loss(pred_phys, ref_phys)
        self._last_parts = {
            "ham": float(ham_loss.detach()),
            "eig": float(eig_loss.detach()),
        }
        return self.coeff_ham * ham_loss + (1.0 - self.coeff_ham) * eig_loss


@Loss.register("hamil_abs_gauged")
class HamilAbsGaugedLoss(nn.Module):
    """hamil_abs, with the H -> H + mu*S gauge freedom removed first.

    Arm A of the NextHAM port: no k-space term, no eigendecomposition, no band
    labels. Only the overlap is needed beyond what pretraining used.
    """

    def __init__(
        self,
        basis: Dict[str, Union[str, list]] = None,
        idp: Union[OrbitalMapper, None] = None,
        gauge: bool = True,
        gauge_clip: float = 1.0,
        **kwargs,
    ):
        super(HamilAbsGaugedLoss, self).__init__()
        if basis is not None:
            self.idp = OrbitalMapper(
                basis, method="e3tb", device=kwargs.get("device", torch.device("cpu"))
            )
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        inner = {
            k: v for k, v in kwargs.items() if k not in ("basis", "overlap", "idp")
        }
        self.ham_loss = HamilLossAbs(idp=self.idp, overlap=False, **inner)
        self.gauge = bool(gauge)
        self.gauge_clip = float(gauge_clip)
        self._last_parts = {}

    def _solve_mu(self, data, ref_data):
        """Closed-form mu on the real-space projection, detached.

        mu = <dH_pred - dH_ref, S> / <S, S>, summed over the valid RME entries
        of both node and edge blocks. Detached so no gradient flows through the
        gauge solve itself.
        """
        num = den = 0.0
        for feat_key, ovp_key in (
            (AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
            (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY),
        ):
            s = ref_data.get(ovp_key, None)
            if s is None:
                raise KeyError(
                    f"hamil_abs_gauged needs {ovp_key!r} on the reference batch; "
                    "run the dataset with get_overlap=true."
                )
            if torch.is_tensor(s) and s.is_nested:
                s = s[0]
            n_rme = self.idp.reduced_matrix_element
            if s.shape[-1] != n_rme:
                raise ValueError(
                    f"{ovp_key} has width {s.shape[-1]}, expected {n_rme}. "
                    "A width of 128 means the model overwrote this field -- "
                    "read S from the reference batch, not the prediction."
                )
            d = data[feat_key].detach() - ref_data[feat_key].detach()
            num = num + (d * s).sum()
            den = den + (s * s).sum()
        mu = num / den.clamp_min(1e-30)
        return float(mu.clamp(-self.gauge_clip, self.gauge_clip))

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        if not self.gauge:
            loss = self.ham_loss(data, ref_data)
            self._last_parts = {
                "mu": 0.0,
                "ham": float(loss.detach()),
                "ham_ungauged": float(loss.detach()),
            }
            return loss

        with torch.no_grad():
            ungauged = float(self.ham_loss(data, ref_data).detach())

        mu = self._solve_mu(data, ref_data)

        # Shift the TARGET, not the prediction: the gauge belongs to the label.
        shifted = dict(ref_data)
        for feat_key, ovp_key in (
            (AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
            (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY),
        ):
            s = ref_data[ovp_key]
            if torch.is_tensor(s) and s.is_nested:
                s = s[0]
            shifted[feat_key] = ref_data[feat_key] + mu * s

        loss = self.ham_loss(data, shifted)
        self._last_parts = {
            "mu": mu,
            "ham": float(loss.detach()),
            "ham_ungauged": ungauged,
            "gauge_gain": ungauged / max(float(loss.detach()), 1e-30),
        }
        return loss


@Loss.register("nextham_kspace")
class NextHAMKSpaceLoss(nn.Module):
    """Real-space H loss plus NextHAM's P/Q/PQ projection loss."""

    def __init__(
        self,
        basis: Dict[str, Union[str, list]] = None,
        idp: Union[OrbitalMapper, None] = None,
        w_p: float = 2e-4,
        w_q: float = 1e-4,
        w_pq: float = 1.5e-4,
        gauge: bool = True,
        gauge_clip: float = 1.0,
        band_window: float = 10.0,
        q_window: float = 30.0,
        n_kpoints: int = 1,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        super(NextHAMKSpaceLoss, self).__init__()
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        inner = {
            k: v
            for k, v in kwargs.items()
            if k not in ("basis", "overlap", "idp", "dtype", "device")
        }
        self.ham_loss = HamilLossAbs(
            idp=self.idp, overlap=False, dtype=dtype, device=device, **inner
        )
        self.w_p, self.w_q, self.w_pq = float(w_p), float(w_q), float(w_pq)
        self.w_r = 1.0 - (self.w_p + self.w_q + self.w_pq)
        if self.w_r < 0:
            raise ValueError("k-space weights sum to > 1 (w_R = %.4f < 0)." % self.w_r)
        if self.w_r == 0:
            # Deliberate: a pure-k control. Nothing holds the model to the
            # matrix it already fits, so H accuracy is free to drift and must
            # be reported separately rather than assumed.
            log.info(
                "NextHAMKSpaceLoss: w_R = 0, training on the k-space "
                "terms alone (no real-space H anchor)."
            )
        self.gauge = bool(gauge)
        self.gauge_clip = float(gauge_clip)
        self.band_window = band_window
        self.q_window = q_window
        self.pq_align = str(kwargs.pop("pq_align", "nextham"))
        if self.pq_align not in ("nextham", "fw10"):
            raise ValueError(
                "pq_align must be 'nextham' or 'fw10', got %r" % self.pq_align
            )
        self.n_kpoints = int(n_kpoints)
        self.device = device
        self._dtype = dtype
        self.l1 = nn.L1Loss(reduction="mean")
        self._last_parts = {}
        self._empty_pq = 0
        self._bad_k = 0
        self._calls = 0
        self._last_k = None
        self._last_ortho = None
        # Own RNG. If the global stream is reseeded elsewhere -- restart,
        # a dataloader worker, another plugin -- the k sampling would silently
        # collapse to one point and nothing in the logs would show it.
        self._kgen = torch.Generator(device="cpu")
        self._kgen.manual_seed(int(kwargs.get("kpoint_seed", 20260829)))
        self.assert_every = int(kwargs.get("assert_every", 50))
        self.ortho_tol = float(kwargs.get("ortho_tol", 1e-4))

        self.h2k = HR2HK(
            idp=self.idp,
            edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            node_field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.HAMILTONIAN_KEY,
            dtype=dtype,
            device=device,
        )
        self.s2k = HR2HK(
            idp=self.idp,
            overlap=True,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            dtype=dtype,
            device=device,
        )

    # ---- gauge -----------------------------------------------------------
    def _solve_mu(self, data, ref_data):
        num = den = 0.0
        for fk, ok in (
            (AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
            (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY),
        ):
            s = _unnest(ref_data.get(ok, None))
            if s is None:
                raise KeyError(
                    "nextham_kspace needs %r; run with get_overlap=true." % ok
                )
            if s.shape[-1] != self.idp.reduced_matrix_element:
                raise ValueError(
                    "%s width %d != RME width %d (128 means the model overwrote it)"
                    % (ok, s.shape[-1], self.idp.reduced_matrix_element)
                )
            d = data[fk].detach() - ref_data[fk].detach()
            num = num + (d * s).sum()
            den = den + (s * s).sum()
        return float(
            (num / den.clamp_min(1e-30)).clamp(-self.gauge_clip, self.gauge_clip)
        )

    # ---- k-space ---------------------------------------------------------
    def _base_fields(self, ref_data):
        keys = (
            AtomicDataDict.EDGE_INDEX_KEY,
            AtomicDataDict.EDGE_CELL_SHIFT_KEY,
            AtomicDataDict.POSITIONS_KEY,
            AtomicDataDict.CELL_KEY,
            AtomicDataDict.PBC_KEY,
            AtomicDataDict.ATOM_TYPE_KEY,
            AtomicDataDict.EDGE_TYPE_KEY,
            AtomicDataDict.ATOMIC_NUMBERS_KEY,
        )
        return {k: _unnest(ref_data[k]) for k in keys if k in ref_data}

    def _split_pq(self, eigvals, nelec):
        """P/Q by energy window around E_F, estimated from the electron count."""
        n_occ = max(1, int(math.ceil(float(nelec) / 2.0)))
        n_occ = min(n_occ, eigvals.numel() - 1)
        e_f = float(eigvals[n_occ - 1])
        rel = eigvals - e_f
        if self.pq_align == "fw10":
            # Occupied / virtual split inside the fw_10 window.
            p_mask = (rel >= -self.band_window) & (rel <= 0.0)
            q_mask = (rel > 0.0) & (rel <= self.band_window)
            return p_mask, q_mask
        # NextHAM's P is [band floor, E_F + ecut], not a window centred on E_F:
        # every occupied state belongs to P (outwf.py:158-161, train_val.py:279).
        p_mask = rel <= self.band_window
        if self.q_window is None:
            q_mask = rel > self.band_window
        else:
            q_mask = (rel > self.band_window) & (
                rel <= self.band_window + self.q_window
            )
        return p_mask, q_mask

    def _kspace_terms(self, data, ref_data, mu):
        # _unnest takes [0]: with more than one graph in the batch every k-space
        # term would silently be computed for the first structure only.
        _b = ref_data.get(AtomicDataDict.BATCH_KEY)
        if _b is not None and int(_b.max()) > 0:
            raise ValueError(
                "nextham_kspace requires batch_size=1 (got %d graphs); the "
                "per-graph eigenbasis cannot be shared across structures."
                % (int(_b.max()) + 1)
            )
        self._empty_pq = 0
        self._bad_k = 0
        for key in (AtomicDataDict.NODE_H0_KEY, AtomicDataDict.EDGE_H0_KEY):
            if ref_data.get(key, None) is None:
                raise KeyError(
                    "nextham_kspace needs %r to rebuild the physical "
                    "Hamiltonian; run the dataset with get_H0=true." % key
                )
        base = self._base_fields(ref_data)
        n_orb_probe = None
        acc = {"p": [], "q": [], "pq": []}

        for _ in range(self.n_kpoints):
            # Drawn on CPU from the dedicated generator, then moved: a CUDA
            # generator would not survive a restart identically.
            kpt = torch.rand(
                1, 3, generator=self._kgen, dtype=torch.get_default_dtype()
            ).to(data[AtomicDataDict.EDGE_FEATURES_KEY].device)
            d_lab = dict(base)
            d_lab[AtomicDataDict.KPOINT_KEY] = kpt
            self._last_k = kpt.reshape(-1).tolist()
            # The eigenbasis must come from the physical H = H0 + dH. H0 cancels
            # in the difference terms but not inside the diagonalisation.
            nh0 = _unnest(ref_data[AtomicDataDict.NODE_H0_KEY])
            eh0 = _unnest(ref_data[AtomicDataDict.EDGE_H0_KEY])
            d_lab[AtomicDataDict.NODE_FEATURES_KEY] = (
                _unnest(ref_data[AtomicDataDict.NODE_FEATURES_KEY]) + nh0
            )
            d_lab[AtomicDataDict.EDGE_FEATURES_KEY] = (
                _unnest(ref_data[AtomicDataDict.EDGE_FEATURES_KEY]) + eh0
            )
            d_lab[AtomicDataDict.NODE_OVERLAP_KEY] = _unnest(
                ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
            )
            d_lab[AtomicDataDict.EDGE_OVERLAP_KEY] = _unnest(
                ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            )

            with torch.no_grad():
                sk = self.s2k(dict(d_lab))[AtomicDataDict.OVERLAP_KEY][0]
                hk_lab = self.h2k(dict(d_lab))[AtomicDataDict.HAMILTONIAN_KEY][0]
                # Generalised eigenproblem in the label's own gauge.
                # float64: at float32 the same expression drifts to ~2.6e-04
                # on badly conditioned S(k), which is 30x the label-set value
                # and enough to supervise a subtly wrong subspace.
                c128 = torch.complex128 if sk.is_complex() else torch.float64
                # Hermitise before Cholesky: S(k) assembled from R-space blocks
                # carries asymmetry at the 1e-7 level, and Cholesky on a matrix
                # that is not quite Hermitian fails or returns silently wrong.
                sk64 = sk.to(c128)
                sk64 = 0.5 * (sk64 + sk64.conj().transpose(-1, -2))
                try:
                    lo = torch.linalg.cholesky(sk64)
                except Exception:
                    # Not positive definite at this k. Counted with the other
                    # rejected k-points so a systematic problem is visible as a
                    # rate rather than as a crash or as silence.
                    self._bad_k += 1
                    continue
                eye64 = torch.eye(sk64.shape[0], device=sk64.device, dtype=c128)
                lo_inv = torch.linalg.solve_triangular(lo, eye64, upper=False)
                heff = lo_inv @ hk_lab.to(c128) @ lo_inv.conj().transpose(-1, -2)
                evals, evecs = torch.linalg.eigh(heff)
                u64 = lo_inv.conj().transpose(-1, -2) @ evecs
                # Check in the precision it was computed in, then cast back.
                self._last_ortho = (
                    (u64.conj().transpose(-1, -2) @ sk64 @ u64 - eye64)
                    .abs()
                    .max()
                    .item()
                )
                u = u64.to(sk.dtype)
                evals = evals.to(torch.float64)
                if ref_data.get("nelec") is None:
                    raise KeyError(
                        "nextham_kspace needs nelec to place E_F; the old default "
                        "(all bands) put it near mid-spectrum and supervised the "
                        "wrong subspace without erroring"
                    )
                _ne = ref_data["nelec"]
                nelec = float(_ne.reshape(-1)[0] if hasattr(_ne, "reshape") else _ne)
                p_mask, q_mask = self._split_pq(evals.real, nelec)
                u_p = u[:, p_mask]
                u_q = u[:, q_mask]
                # Reported so a wrong split shows up in the log rather
                # than as a quietly plausible loss value.
                self._n_p = int(p_mask.sum())
                self._n_q = int(q_mask.sum())
                self._e_min = float(evals.real.min())
                self._e_max = float(evals.real.max())
                self._n_occ = int(math.ceil(nelec / 2.0))
            has_p = u_p.shape[1] > 0
            has_q = u_q.shape[1] > 0
            if not has_p and not has_q:
                # Neither window has states: the split (or the Hamiltonian) is
                # wrong. Counted, not silently skipped.
                self._empty_pq += 1
                continue
            if not (has_p and has_q):
                self._empty_pq += 1
            n_orb_probe = sk.shape[0]

            # U must diagonalise the label in S(k)'s metric. A near-singular
            # S(k) still yields a finite U, and every downstream term would
            # look healthy while supervising the wrong subspace.
            # A single ill-conditioned k must not end an overnight run, but it
            # must not be supervised either. Skip and count; a systematic
            # problem shows up as a large skip count, not as silence.
            if self._last_ortho is not None and self._last_ortho > self.ortho_tol:
                self._bad_k += 1
                continue

            d_pred = dict(d_lab)
            d_pred[AtomicDataDict.NODE_FEATURES_KEY] = (
                data[AtomicDataDict.NODE_FEATURES_KEY] + nh0
            )
            d_pred[AtomicDataDict.EDGE_FEATURES_KEY] = (
                data[AtomicDataDict.EDGE_FEATURES_KEY] + eh0
            )
            hk_pred = self.h2k(d_pred)[AtomicDataDict.HAMILTONIAN_KEY][0]

            eye_p = torch.eye(u_p.shape[1], device=hk_pred.device, dtype=hk_pred.dtype)
            eye_q = torch.eye(u_q.shape[1], device=hk_pred.device, dtype=hk_pred.dtype)
            terms = []
            if has_p:
                terms.append(("p", u_p, u_p, eye_p))
            if has_q:
                terms.append(("q", u_q, u_q, eye_q))
            if has_p and has_q:
                terms.append(("pq", u_p, u_q, None))
            for tag, ua, ub, eye in terms:
                a = ua.conj().transpose(-1, -2) @ hk_lab @ ub
                b = ua.conj().transpose(-1, -2) @ hk_pred @ ub
                if eye is not None:
                    a = a + mu * eye
                acc[tag].append(self.l1(a.real, b.real) + self.l1(a.imag, b.imag))

        def red(v):
            return torch.stack(v).mean() if v else torch.zeros((), device=self.device)

        return red(acc["p"]), red(acc["q"]), red(acc["pq"]), n_orb_probe

    # ---- forward ---------------------------------------------------------
    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        self._calls += 1
        mu = self._solve_mu(data, ref_data) if self.gauge else 0.0

        shifted = dict(ref_data)
        if mu:
            for fk, ok in (
                (AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
                (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY),
            ):
                shifted[fk] = ref_data[fk] + mu * _unnest(ref_data[ok])
        l_r = self.ham_loss(data, shifted)

        if self.w_p == self.w_q == self.w_pq == 0.0:
            self._last_parts = {
                "R": float(l_r.detach()),
                "wR": float(l_r.detach()),
                "mu": mu,
                "band_computed": False,
            }
            return l_r
        l_p, l_q, l_pq, n_orb = self._kspace_terms(data, ref_data, mu)
        total = self.w_r * l_r + self.w_p * l_p + self.w_q * l_q + self.w_pq * l_pq

        self._last_parts = {
            "mu": mu,
            "n_orb": n_orb,
            "empty_pq": getattr(self, "_empty_pq", 0),
            "bad_k": getattr(self, "_bad_k", 0),
            "n_P": getattr(self, "_n_p", None),
            "n_Q": getattr(self, "_n_q", None),
            "n_occ": getattr(self, "_n_occ", None),
            "e_min": getattr(self, "_e_min", None),
            "e_max": getattr(self, "_e_max", None),
            "ortho": getattr(self, "_last_ortho", None),
            "R": float(l_r.detach()),
            "P": float(l_p.detach()),
            "Q": float(l_q.detach()),
            "PQ": float(l_pq.detach()),
            "wR": self.w_r * float(l_r.detach()),
            "wP": self.w_p * float(l_p.detach()),
            "wQ": self.w_q * float(l_q.detach()),
            "wPQ": self.w_pq * float(l_pq.detach()),
        }
        return total


@Loss.register("fw10_eig")
class FW10EigLoss(EigHamH0ResLoss):
    """H-matrix loss plus a band term that is literally the fw_10 metric.

    total = coeff_ham * L_ham(residual) + (1 - coeff_ham) * fw_10(H0 + residual)
    """

    def __init__(self, band_window: float = 10.0, **kwargs):
        super(FW10EigLoss, self).__init__(**kwargs)
        self.band_window = float(band_window)
        self._last_parts = {}

    @staticmethod
    def _vbm(eig, n_occ):
        """Top of the highest occupied band, over all k. Differentiable."""
        return eig[:, n_occ - 1].max()

    def _band_loss(self, pred_phys: dict, ref_phys: dict):
        solver_input = self._solver_dict(pred_phys, ref_phys)
        eig_p = self.eigen(solver_input)[AtomicDataDict.ENERGY_EIGENVALUE_KEY]
        if eig_p.dim() == 3:
            eig_p = eig_p[0]
        eig_r = _unnest(ref_phys[AtomicDataDict.ENERGY_EIGENVALUE_KEY])
        if torch.is_tensor(eig_r) and eig_r.dim() == 3:
            eig_r = eig_r[0]
        eig_r = eig_r.to(device=eig_p.device, dtype=eig_p.dtype)
        if eig_p.shape[0] != eig_r.shape[0]:
            raise RuntimeError(
                "k-point count differs: pred %d vs ref %d"
                % (eig_p.shape[0], eig_r.shape[0])
            )
        nb = min(eig_p.shape[1], eig_r.shape[1])
        eig_p, eig_r = eig_p[:, :nb], eig_r[:, :nb]

        ne = ref_phys.get("nelec", None)
        if ne is None:
            raise KeyError(
                "fw10_eig needs nelec to locate the VBM; without it the window "
                "lands mid-spectrum, which is the error the first round made."
            )
        ne = float(ne.reshape(-1)[0] if hasattr(ne, "reshape") else ne)
        n_occ = max(1, min(int(math.ceil(ne / 2.0)), nb - 1))

        # Each side by its own VBM: an overall shift is unphysical, and the
        # metric is defined this way.
        rel_p = eig_p - self._vbm(eig_p, n_occ)
        rel_r = eig_r - self._vbm(eig_r, n_occ)

        # Mask on the reference: it is a constant, so no gradient flows through
        # the choice of which bands to score, and the set cannot drift as the
        # prediction moves.
        mask = (rel_r >= -self.band_window) & (rel_r <= self.band_window)
        n_in = int(mask.sum())
        if n_in == 0:
            raise RuntimeError(
                "empty fw_10 window: no reference band within +/-%.1f eV of the "
                "VBM. Check nelec and the eigenvalue units." % self.band_window
            )

        loss = (rel_p[mask] - rel_r[mask]).abs().mean()
        self._fw_parts = {
            "n_occ": n_occ,
            "n_bands": int(nb),
            "n_in_window": n_in,
            "frac_in_window": n_in / float(rel_r.numel()),
            "vbm_ref": float(self._vbm(eig_r, n_occ).detach()),
            "fw10": float(loss.detach()),
        }
        return loss

    def forward(self, data, ref_data):
        out = super(FW10EigLoss, self).forward(data, ref_data)
        # Parent sets _last_parts after _band_loss; merge the window
        # diagnostics back so a mis-placed window is visible, not inferred.
        self._last_parts.update(getattr(self, "_fw_parts", {}))
        return out


@Loss.register("band_stage2")
class BandStage2Loss(nn.Module):
    """HamGNN-style second-stage loss: hamil_abs + lambda * band MAE at random k.

    See patch_band_stage2.py for the literature mapping. One graph per batch.
    """

    def __init__(
        self,
        basis: Dict[str, Union[str, list]] = None,
        idp: Union[OrbitalMapper, None] = None,
        band_weight: float = 1e-2,
        n_kpoints: int = 5,
        band_window: float = 10.0,
        window_mode: str = "fermi",
        n_bands: int = None,
        align: str = "none",
        gauge: bool = False,
        gauge_clip: float = 1.0,
        ill_threshold: float = 1e-5,
        solver_float64: bool = True,
        k_seed: int = 20260902,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        super(BandStage2Loss, self).__init__()
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
        if window_mode not in ("fermi", "lowest"):
            raise ValueError("window_mode must be 'fermi' or 'lowest'")
        if align not in ("none", "vbm"):
            raise ValueError("align must be 'none' or 'vbm'")
        if window_mode == "lowest" and not n_bands:
            raise ValueError("window_mode='lowest' needs n_bands")

        inner = {
            k: v
            for k, v in kwargs.items()
            if k not in ("basis", "overlap", "idp", "dtype", "device")
        }
        self.ham_loss = HamilLossAbs(
            idp=self.idp, overlap=False, dtype=dtype, device=device, **inner
        )
        self.band_weight = float(band_weight)
        self.n_kpoints = int(n_kpoints)
        self.band_window = float(band_window)
        self.window_mode = window_mode
        self.n_bands = n_bands
        self.align = align
        self.gauge = bool(gauge)
        self.gauge_clip = float(gauge_clip)
        self.ill_threshold = ill_threshold
        self.solver_dtype = (
            torch.float64
            if solver_float64
            else (getattr(torch, dtype) if isinstance(dtype, str) else dtype)
        )
        self.device = device
        # k-points are drawn here, in the loss (main process), from a private
        # generator: a dataset-side draw would collapse to one k per structure
        # under fixed worker seeds.
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(int(k_seed))
        self.eigen = Eigenvalues(
            idp=self.idp,
            h_edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            h_node_field=AtomicDataDict.NODE_FEATURES_KEY,
            h_out_field=AtomicDataDict.HAMILTONIAN_KEY,
            out_field=AtomicDataDict.ENERGY_EIGENVALUE_KEY,
            s_edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            s_node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            s_out_field=AtomicDataDict.OVERLAP_KEY,
            dtype=self.solver_dtype,
            device=device,
        )
        self._last_parts = {}
        self._last_tensors = {}

    _SOLVER_FIELDS = (
        AtomicDataDict.EDGE_INDEX_KEY,
        AtomicDataDict.EDGE_CELL_SHIFT_KEY,
        AtomicDataDict.POSITIONS_KEY,
        AtomicDataDict.CELL_KEY,
        AtomicDataDict.PBC_KEY,
        AtomicDataDict.ATOM_TYPE_KEY,
        AtomicDataDict.EDGE_TYPE_KEY,
        AtomicDataDict.ATOMIC_NUMBERS_KEY,
    )

    # ---- helpers ---------------------------------------------------------
    def _n_graph(self, ref):
        ptr = ref.get(AtomicDataDict.BATCH_PTR_KEY, None)
        return 1 if ptr is None else int(ptr.numel()) - 1

    def _solver_dict(self, feats_node, feats_edge, ref, kpts):
        out = {}
        for key in self._SOLVER_FIELDS:
            v = ref.get(key, None)
            if v is not None:
                v = _unnest(v)
                # HR2HK forms the Bloch phase from cell / shift in its own dtype;
                # geometry must match the solver precision or torch.matmul raises.
                if torch.is_tensor(v) and v.is_floating_point():
                    v = v.to(self.solver_dtype)
                out[key] = v
        n_rme = self.idp.reduced_matrix_element
        for key in (AtomicDataDict.NODE_OVERLAP_KEY, AtomicDataDict.EDGE_OVERLAP_KEY):
            s = _unnest(ref.get(key, None))
            if s is None:
                raise KeyError(
                    "band_stage2 needs %r; run the dataset with get_overlap=true." % key
                )
            if s.shape[-1] != n_rme:
                raise ValueError(
                    "%s width %d != RME width %d (128 means the model overwrote it; "
                    "read S from the reference batch)" % (key, s.shape[-1], n_rme)
                )
            out[key] = s.to(self.solver_dtype)
        out[AtomicDataDict.NODE_FEATURES_KEY] = feats_node.to(self.solver_dtype)
        out[AtomicDataDict.EDGE_FEATURES_KEY] = feats_edge.to(self.solver_dtype)
        out[AtomicDataDict.KPOINT_KEY] = kpts.to(
            device=feats_node.device, dtype=self.solver_dtype
        )
        return out

    def _solve_mu(self, data, ref):
        num = den = 0.0
        for fk, ok in (
            (AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
            (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY),
        ):
            s = _unnest(ref[ok])
            d = data[fk].detach() - ref[fk].detach()
            num = num + (d * s).sum()
            den = den + (s * s).sum()
        return float(
            (num / den.clamp_min(1e-30)).clamp(-self.gauge_clip, self.gauge_clip)
        )

    # ---- forward ---------------------------------------------------------
    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        if self._n_graph(ref_data) != 1:
            raise RuntimeError(
                "band_stage2 expects one graph per batch (HamGNN and NextHAM both "
                "train with batch_size=1); set batch_size=1 and dynamic_batch off."
            )

        # (1) real-space anchor, optionally in the mu-gauge
        mu = 0.0
        target = ref_data
        if self.gauge:
            mu = self._solve_mu(data, ref_data)
            target = dict(ref_data)
            for fk, ok in (
                (AtomicDataDict.NODE_FEATURES_KEY, AtomicDataDict.NODE_OVERLAP_KEY),
                (AtomicDataDict.EDGE_FEATURES_KEY, AtomicDataDict.EDGE_OVERLAP_KEY),
            ):
                target[fk] = ref_data[fk] + mu * _unnest(ref_data[ok])
        l_ham = self.ham_loss(data, target)

        if self.band_weight == 0.0:
            # A pure-H control must have exactly the H gradient, including at
            # degenerate spectra where a zero-weight eigen backward can be NaN.
            self._last_parts = {
                "ham": float(l_ham.detach()),
                "wband": 0.0,
                "mu": mu,
                "band_computed": False,
            }
            return l_ham

        # (2) band term
        for key in (AtomicDataDict.NODE_H0_KEY, AtomicDataDict.EDGE_H0_KEY):
            if ref_data.get(key, None) is None:
                raise KeyError(
                    "band_stage2 needs %r to rebuild the physical H; run with get_H0=true."
                    % key
                )
        nh0 = _unnest(ref_data[AtomicDataDict.NODE_H0_KEY])
        eh0 = _unnest(ref_data[AtomicDataDict.EDGE_H0_KEY])
        ne = ref_data.get("nelec", None)
        if ne is None:
            raise KeyError(
                "band_stage2 needs nelec on the batch (record or valence table)."
            )
        ne = float(ne.reshape(-1)[0]) if hasattr(ne, "reshape") else float(ne)

        kpts = torch.rand(self.n_kpoints, 3, generator=self._gen)
        lab = self._solver_dict(
            _unnest(ref_data[AtomicDataDict.NODE_FEATURES_KEY]) + nh0,
            _unnest(ref_data[AtomicDataDict.EDGE_FEATURES_KEY]) + eh0,
            ref_data,
            kpts,
        )
        with torch.no_grad():
            eig_ref = self.eigen(lab, ill_threshold=self.ill_threshold)[
                AtomicDataDict.ENERGY_EIGENVALUE_KEY
            ]
        pred = self._solver_dict(
            data[AtomicDataDict.NODE_FEATURES_KEY] + nh0,
            data[AtomicDataDict.EDGE_FEATURES_KEY] + eh0,
            ref_data,
            kpts,
        )
        eig_pred = self.eigen(pred, ill_threshold=self.ill_threshold)[
            AtomicDataDict.ENERGY_EIGENVALUE_KEY
        ]
        if eig_ref.dim() == 3:
            eig_ref = eig_ref[0]
        if eig_pred.dim() == 3:
            eig_pred = eig_pred[0]
        n_orb = eig_ref.shape[1]
        n_occ = max(1, min(int(math.ceil(ne / 2.0)), n_orb - 1))
        occ_frac = n_occ / float(n_orb)
        if not (0.03 <= occ_frac <= 0.6):
            raise RuntimeError(
                "n_occ/n_orb = %.3f is outside [0.03, 0.6]: nelec=%s n_orb=%d. "
                "This is the half-filling fallback signature." % (occ_frac, ne, n_orb)
            )
        vbm_ref = eig_ref[:, n_occ - 1].max()
        if self.window_mode == "fermi":
            mask = ((eig_ref - vbm_ref) >= -self.band_window) & (
                (eig_ref - vbm_ref) <= self.band_window
            )
        else:
            mask = torch.zeros_like(eig_ref, dtype=torch.bool)
            mask[:, : min(int(self.n_bands), n_orb)] = True
        n_in = int(mask.sum())
        if n_in == 0:
            raise RuntimeError("empty band window; check nelec / eigenvalue units.")
        if self.align == "vbm":
            ep = eig_pred - eig_pred[:, n_occ - 1].max()
            er = eig_ref - vbm_ref
        else:
            ep, er = eig_pred, eig_ref
        l_band = (ep[mask] - er[mask]).abs().mean().to(l_ham.dtype)

        total = l_ham + self.band_weight * l_band
        self._last_tensors = {"ham": l_ham, "band": l_band}
        self._last_parts = {
            "ham": float(l_ham.detach()),
            "band": float(l_band.detach()),
            "wband": self.band_weight * float(l_band.detach()),
            "mu": mu,
            "n_occ": n_occ,
            "n_orb": int(n_orb),
            "n_in_window": n_in,
            "frac_in_window": n_in / float(mask.numel()),
            "vbm_ref": float(vbm_ref),
            "eig_min": float(eig_ref.min()),
            "eig_max": float(eig_ref.max()),
        }
        return total
