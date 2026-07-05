# SPDX-License-Identifier: LGPL-3.0-or-later
"""``product_split_flow`` objective: wire the product-manifold (Euclidean ΔH ⊕ Grassmann P)
flow into the live :class:`~dptb.nnops.flow.HamiltonianCFM` training path.

This is the "接 flow" layer.  It is a **plain static subclass** of ``HamiltonianCFM`` (the
pattern ``HamiltonianPixelMeanFlow`` already proves), selected by ``objective:
product_split_flow`` in ``flow_options``.  It reuses the CFM machinery verbatim:

* ``super().prepare_batch(..., t=t_cur)`` interpolates ``H_t`` into the model input and
  returns a :class:`~dptb.nnops.flow.CFMContext` exposing ``node_base/node_prior/
  node_target/node_current/node_t`` (and ``edge_*``);
* ``super().loss(pred, ref, ctx)`` is the **Hamiltonian endpoint (velocity) anchor**;
* ``model_in_loss = True`` routes the trainer to :meth:`loss_with_model` (no trainer edit;
  verified at ``trainer.py:254``).

On top of the anchor it adds three additive terms (the M-b algebra of the handoff):

* **H feature-space split-flow consistency** (``mu_consistency``): stepping the interpolated
  ``H_t`` toward the predicted endpoint must land on the geodesic point ``H_r``.  For the
  straight Euclidean path this is exact iff ``pred`` is the clean endpoint, so it is a pure
  self-consistency with no extra model call.  Works at any batch size / SOC (feature space).
* **Grassmann occupied-P endpoint** (``lambda_p``): chordal distance between the occupied
  projector of the *predicted* dense ``H(k)`` and that of the *reference* ``H(k)`` (reusing
  :func:`~dptb.nnops.grassmann.dense_grassmann_p_loss` + the ``GrassmannManifold``).  Scoped to
  ``batch_size==1`` (single-structure dense HR2HK) and gapped systems; metallic / undefined
  ``n_occ`` / missing-overlap records are skipped via :class:`SkippableRecord`, not crashed.
* **Conduction window** (``lambda_conduction``): the fixed-window Euclidean block penalty for
  near-Fermi virtual states the occupied projector cannot see (handoff Part 2).

Nothing here duplicates the geometry layer -- it reuses ``manifold_type.GrassmannManifold``,
``manifold_flow.{OrderedIntervalSampler,ThreePointSampler}`` and
``manifold_flow_model.conduction_window_loss`` rather than re-deriving them.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple

import torch

# Top-level import of flow.py is safe: build_hamiltonian_flow imports THIS module lazily, so
# flow.py is fully initialised before product_split_flow is ever imported (no cycle).
from dptb.nnops.flow import HamiltonianCFM, CFMContext
from dptb.nnops.layout import project_uureal_to_like
from dptb.nnops.manifold_type import GrassmannManifold
from dptb.nnops.manifold_flow import OrderedIntervalSampler, ThreePointSampler
from dptb.nnops.manifold_flow_model import conduction_window_loss
from dptb.nnops.grassmann import dense_grassmann_p_loss, SkippableRecord

try:
    from dptb.data import AtomicDataDict
except Exception:  # pragma: no cover - standalone import guard
    class AtomicDataDict:  # type: ignore[no-redef]
        NODE_FEATURES_KEY = "node_features"
        EDGE_FEATURES_KEY = "edge_features"
        NODE_OVERLAP_KEY = "node_overlap"
        EDGE_OVERLAP_KEY = "edge_overlap"
        HAMILTONIAN_KEY = "hamiltonian"
        OVERLAP_KEY = "overlap"
        KPOINT_KEY = "kpoint"

try:
    from dptb.nn.hr2hk import HR2HK
except Exception:  # pragma: no cover
    HR2HK = None

log = logging.getLogger(__name__)


@dataclass
class ProductSplitContext:
    """Bundle the base :class:`CFMContext` with the product split times."""

    cfm: CFMContext
    t: torch.Tensor
    r: torch.Tensor
    s: Optional[torch.Tensor] = None


class HamiltonianProductSplitFlow(HamiltonianCFM):
    """Product-manifold split-flow objective (Euclidean ΔH anchor + Grassmann-P + conduction).

    Static subclass of ``HamiltonianCFM``; selected by ``objective: product_split_flow``.
    ``model_in_loss = True`` -> the trainer calls :meth:`loss_with_model`.
    """

    model_in_loss = True

    def __init__(self, options: Optional[Dict[str, Any]], *, idp: Any = None,
                 dtype: Any = torch.float32, device: Any = torch.device("cpu")) -> None:
        super().__init__(options, idp=idp, dtype=dtype, device=device)
        options = dict(options or {})
        pm = dict(options.get("product_manifold", options.get("product", {})) or {})

        self.p_n_occ = pm.get("n_occ", options.get("n_occ", None))
        self.p_n_occ_key = str(pm.get("n_occ_key", "n_occ"))
        self.p_kpoints = pm.get("kpoints", [[0.0, 0.0, 0.0]])
        self.lambda_h = float(pm.get("lambda_h", 1.0))
        self.lambda_p = float(pm.get("lambda_p", 0.0))
        self.mu_consistency = float(pm.get("mu_consistency", 1.0))
        self.lambda_conduction = float(pm.get("lambda_conduction", 0.0))
        self.conduction_window = int(pm.get("conduction_window", 0))
        self.p_boundary_ratio = float(pm.get("boundary_ratio", 0.0))
        self.p_use_three_point = bool(pm.get("three_point", False))
        self.p_eig_floor = float(pm.get("eig_floor", 1.0e-10))
        self.p_require_overlap = bool(pm.get("require_overlap", False))
        self.p_chordal_normalize = bool(pm.get("chordal_normalize", False))
        self.p_min_gap = float(pm.get("min_gap", 0.0))
        self.p_from_density = bool(pm.get("from_density", False))

        # The product split times: the base HamiltonianCFM does NOT set flow_time_{t,r,h}_key
        # (only HamiltonianPixelMeanFlow does), so read them here with the argcheck defaults.
        self.flow_time_t_key = str(options.get("flow_time_t_key", "flow_time_t"))
        self.flow_time_r_key = str(options.get("flow_time_r_key", "flow_time_r"))
        self.flow_time_h_key = str(options.get("flow_time_h_key", "flow_time_h"))

        self.grassmann = GrassmannManifold()
        self._interval_sampler = OrderedIntervalSampler(boundary_ratio=self.p_boundary_ratio)
        self._triple_sampler = ThreePointSampler(boundary_ratio=self.p_boundary_ratio)
        self._p_scope_warned = False

        # HR2HK assemblers for the dense Grassmann-P / conduction terms.  Built lazily-ish:
        # only when a P/conduction weight is on and an idp is available (mirrors
        # GrassmannPAlignLoss.__init__).  The overlap assembler is gated on require_overlap.
        self.h2k = self.s2k = None
        wants_dense = (self.lambda_p != 0.0) or (self.lambda_conduction != 0.0)
        if wants_dense and HR2HK is not None and idp is not None:
            self.h2k = HR2HK(
                idp=idp,
                edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.HAMILTONIAN_KEY,
                dtype=self.dtype, device=self.device,
            )
            if self.p_require_overlap:
                self.s2k = HR2HK(
                    idp=idp, overlap=True,
                    edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                    out_field=AtomicDataDict.OVERLAP_KEY,
                    dtype=self.dtype, device=self.device,
                )

    # ------------------------------------------------------------------ #
    # time sampling + batch preparation
    # ------------------------------------------------------------------ #
    def _sample_split_times(self, num_graphs: int, *, device, dtype):
        if self.p_use_three_point:
            t, s, r = self._triple_sampler.sample(num_graphs, device=device, dtype=dtype)
            return t, r, s
        t, r = self._interval_sampler.sample(num_graphs, device=device, dtype=dtype)
        return t, r, None

    def prepare_batch(self, data, ref_data, *, r: Optional[torch.Tensor] = None,
                      t: Optional[torch.Tensor] = None):
        if not self.enabled:
            raise RuntimeError("HamiltonianProductSplitFlow.prepare_batch called while disabled")
        like = ref_data.get(self.node_target_key, ref_data.get(self.edge_target_key, None))
        if like is None:
            raise KeyError(
                "product_split_flow needs node and/or edge Hamiltonian targets in ref_data "
                f"(`{self.node_target_key}`/`{self.edge_target_key}`).")
        num_graphs = self._num_graphs(data)
        device = like.device
        dtype = like.dtype if torch.is_floating_point(like) else self.dtype

        if r is None or t is None:
            t_cur, r_next, s_mid = self._sample_split_times(num_graphs, device=device, dtype=dtype)
        else:
            # The trainer's model-in-loss validation passes (r=0, t=1) for the one-step
            # endpoint pass; canonicalise to current(t_cur) <= target(r_next) so the base
            # interpolation at t_cur is the noisy input and r_next is the consistency target.
            tt = self._normalize_t(t, num_graphs=num_graphs, device=device, dtype=dtype)
            rr = self._normalize_t(r, num_graphs=num_graphs, device=device, dtype=dtype)
            t_cur = torch.minimum(tt, rr)
            r_next = torch.maximum(tt, rr)
            s_mid = None

        data_t, ref_t, cfm_ctx = super().prepare_batch(data, ref_data, t=t_cur)
        for d in (data_t, ref_t):
            d[self.flow_time_t_key] = t_cur.detach()
            d[self.flow_time_r_key] = r_next.detach()
            d[self.flow_time_h_key] = (r_next - t_cur).detach()
        return data_t, ref_t, ProductSplitContext(cfm=cfm_ctx, t=t_cur, r=r_next, s=s_mid)

    # ------------------------------------------------------------------ #
    # loss terms
    # ------------------------------------------------------------------ #
    def _align_layout(self, pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Project a predicted RME table onto the reference layout (SOC uureal masking).

        A no-op when no idp is set (offline tests) or the basis carries no uureal mask
        (non-SOC), so it is always safe to call.
        """
        if self.idp is None:
            return pred
        aligned, _ = project_uureal_to_like(self.idp, pred, ref)
        return aligned

    def _h_consistency_loss(self, pred_data, ref_data, ctx: ProductSplitContext) -> torch.Tensor:
        """Euclidean split-flow consistency: step H_t -> r must match the interpolant H_r.

        ``step_r = H_t + (r-t)/(1-t) * (pred - H_t)`` (linear extrapolation toward the
        predicted endpoint) is compared to the detached geodesic point
        ``H_r = base + (1-r)*prior + r*res``; exact-zero iff ``pred`` is the clean endpoint,
        so it is a pure self-consistency (no extra model call) and works at any batch/SOC.
        """
        cfm = ctx.cfm
        # Resolve the per-node AND per-edge target time r in ONE call (mirrors the base's
        # single _expand_graph_times at flow.py:795): with no BATCH_KEY the fallback builds a
        # node-sized batch that edge indexing needs -- resolving the edge branch separately
        # with node_count=None would give an empty batch and break edge index_select.
        node_r, edge_r = self._expand_graph_times(
            ref_data, ctx.r,
            node_count=None if cfm.node_target is None else cfm.node_target.shape[0],
            edge_count=None if cfm.edge_target is None else cfm.edge_target.shape[0],
        )
        pieces = []
        for key, cur, base, prior, target, node_t, r_rows, weight in (
            (self.node_target_key, cfm.node_current, cfm.node_base, cfm.node_prior,
             cfm.node_target, cfm.node_t, node_r, self.node_weight),
            (self.edge_target_key, cfm.edge_current, cfm.edge_base, cfm.edge_prior,
             cfm.edge_target, cfm.edge_t, edge_r, self.edge_weight),
        ):
            if cur is None or target is None or key not in pred_data or r_rows is None:
                continue
            pred = self._align_layout(pred_data[key], target)
            pred = pred.to(device=cur.device, dtype=cur.dtype)
            res = target - base
            t_view = node_t.reshape((-1,) + (1,) * (cur.ndim - 1))
            r_view = r_rows.reshape((-1,) + (1,) * (cur.ndim - 1))
            denom = (1.0 - t_view).clamp_min(self.t_eps)
            step_r = cur + (r_view - t_view) / denom * (pred - cur)
            true_r = base + (1.0 - r_view) * prior + r_view * res
            pieces.append(float(weight) * (step_r - true_r.detach()).square().mean())
        if not pieces:
            return ctx.t.new_zeros(())
        return torch.stack(pieces).sum()

    def _build_hk(self, feats: Dict[str, torch.Tensor], template, module, out_key: str) -> torch.Tensor:
        """Assemble dense ``H(k)`` (or ``S(k)``) by overlaying ``feats`` on a template dict."""
        d = dict(template)
        d.update(feats)
        d[AtomicDataDict.KPOINT_KEY] = torch.as_tensor(self.p_kpoints, device=self.device, dtype=self.dtype)
        d = module(d)
        return d[out_key]

    def _resolve_p_n_occ(self, ref_data, data) -> Optional[int]:
        for src in (ref_data, data):
            v = src.get(self.p_n_occ_key, None)
            if v is not None:
                return int(torch.as_tensor(v).reshape(-1)[0].item())
        return None if self.p_n_occ is None else int(self.p_n_occ)

    def _grassmann_endpoint_loss(self, pred_data, ref_data, ctx: ProductSplitContext,
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Grassmann occupied-P endpoint (λ_p) + conduction window (λ_cond), batch-1 scoped.

        Builds dense ``H(k)`` from the predicted vs reference endpoint RME via HR2HK and
        reuses :func:`dense_grassmann_p_loss` for the chordal projector loss.  Restricted to
        ``batch_size==1`` (single-structure dense assembly); metallic/undefined records are
        skipped via :class:`SkippableRecord`.  Returns ``(p_loss, c_loss)`` (zeros if scoped out).
        """
        zero = ctx.t.new_zeros(())
        if (self.lambda_p == 0.0 and self.lambda_conduction == 0.0) or self.h2k is None:
            return zero, zero
        if self._num_graphs(ref_data) != 1:
            if not self._p_scope_warned:
                log.warning("product_split_flow: Grassmann-P/conduction terms need batch_size=1 "
                            "(dense per-structure HR2HK); skipping them for batch>1.")
                self._p_scope_warned = True
            return zero, zero
        n_occ = self._resolve_p_n_occ(ref_data, pred_data)
        if n_occ is None:
            if not self._p_scope_warned:
                log.warning("product_split_flow: Grassmann-P term needs `n_occ` (product_manifold.n_occ "
                            "or an n_occ field); skipping.")
                self._p_scope_warned = True
            return zero, zero
        try:
            # Predicted vs reference CLEAN endpoint RME (the model predicts the clean H;
            # ref target is the true clean H).  Align predicted layout to the ref target.
            pred_node = pred_data.get(self.node_target_key)
            pred_edge = pred_data.get(self.edge_target_key)
            feats_pred: Dict[str, torch.Tensor] = {}
            feats_ref: Dict[str, torch.Tensor] = {}
            if ctx.cfm.node_target is not None and pred_node is not None:
                feats_pred[AtomicDataDict.NODE_FEATURES_KEY] = self._align_layout(pred_node, ctx.cfm.node_target)
                feats_ref[AtomicDataDict.NODE_FEATURES_KEY] = ctx.cfm.node_target
            if ctx.cfm.edge_target is not None and pred_edge is not None:
                feats_pred[AtomicDataDict.EDGE_FEATURES_KEY] = self._align_layout(pred_edge, ctx.cfm.edge_target)
                feats_ref[AtomicDataDict.EDGE_FEATURES_KEY] = ctx.cfm.edge_target
            if not feats_pred:
                return zero, zero
            h_pred = self._build_hk(feats_pred, ref_data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
            h_ref = self._build_hk(feats_ref, ref_data, self.h2k, AtomicDataDict.HAMILTONIAN_KEY)
            s = None
            if self.s2k is not None:
                s = self._build_hk({}, ref_data, self.s2k, AtomicDataDict.OVERLAP_KEY)
            elif self.p_require_overlap:
                raise SkippableRecord("product_split_flow.require_overlap set but no overlap assembler")

            p_loss = zero
            if self.lambda_p != 0.0:
                p_loss, _ = dense_grassmann_p_loss(
                    h_pred, h_ref.detach(), s,
                    n_occ=n_occ, lambda_chordal=1.0, lambda_eps=0.0,
                    eig_floor=self.p_eig_floor, from_density=self.p_from_density,
                    min_gap=self.p_min_gap, chordal_normalize=self.p_chordal_normalize,
                )
            c_loss = zero
            if self.lambda_conduction != 0.0 and self.conduction_window > 0:
                # eigenvector window frame of the reference H, detached (fixed reference).
                fp = h_pred.reshape((-1,) + tuple(h_pred.shape[-2:]))
                fr = h_ref.reshape((-1,) + tuple(h_ref.shape[-2:]))
                pieces = []
                for i in range(fp.shape[0]):
                    with torch.no_grad():
                        _, vecs = torch.linalg.eigh(0.5 * (fr[i] + fr[i].mH))
                    hi = min(vecs.shape[-1], int(n_occ) + int(self.conduction_window))
                    c_window = vecs[:, int(n_occ):hi].detach()
                    pieces.append(conduction_window_loss(fp[i], fr[i].detach(), c_window))
                if pieces:
                    c_loss = torch.stack([p.reshape(()) for p in pieces]).mean()
            return p_loss, c_loss
        except SkippableRecord as exc:
            if not self._p_scope_warned:
                log.warning("product_split_flow: Grassmann-P term skipped (%s)", exc)
                self._p_scope_warned = True
            return zero, zero

    # ------------------------------------------------------------------ #
    # model-in-loss entry point (trainer dispatches here via model_in_loss=True)
    # ------------------------------------------------------------------ #
    def loss_with_model(self, model, data, ref_data, *, prefix: str = "train",
                        r: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None):
        data_t, ref_t, ctx = self.prepare_batch(data, ref_data, r=r, t=t)
        pred = model(data_t)
        h_loss, base_state = super().loss(pred, ref_t, ctx.cfm)  # Hamiltonian endpoint anchor
        h_consistency = self._h_consistency_loss(pred, ref_t, ctx)
        p_loss, c_loss = self._grassmann_endpoint_loss(pred, ref_t, ctx)

        total = (self.lambda_h * h_loss
                 + self.mu_consistency * h_consistency
                 + self.lambda_p * p_loss
                 + self.lambda_conduction * c_loss)

        # super().loss() hard-codes `train_*` state keys; re-namespace for non-train prefixes
        # (validation one-step) so the trainer's compatible-loss/legacy mapping lines up.
        if prefix != "train":
            base_state = {(k.replace("train_", f"{prefix}_", 1) if k.startswith("train_") else k): v
                          for k, v in base_state.items()}
        state = dict(base_state)
        state.update({
            f"{prefix}_flow_product_loss": total.detach(),
            f"{prefix}_flow_product_h_endpoint_loss": h_loss.detach(),
            f"{prefix}_flow_product_h_consistency_loss": h_consistency.detach(),
            f"{prefix}_flow_product_grassmann_loss": p_loss.detach(),
            f"{prefix}_flow_product_conduction_loss": c_loss.detach(),
            f"{prefix}_flow_product_t": ctx.t.detach().mean(),
            f"{prefix}_flow_product_r": ctx.r.detach().mean(),
            # Fill the model_in_loss canary fields the flow logger registers, so the
            # terminal columns show real values instead of a misleading constant 0.
            f"{prefix}_flow_r": ctx.r.detach().mean(),
            f"{prefix}_flow_h": (ctx.r - ctx.t).detach().mean(),
            f"{prefix}_flow_du_dt_backend_jvp": ctx.t.new_tensor(0.0),  # this objective uses no jvp
            f"{prefix}_flow_explicit_model_calls": ctx.t.new_tensor(1.0),
        })
        # super().loss() stashes a nested dict `_compatible_clean_stats` (for the
        # model_in_loss=False compatible-loss path); the model_in_loss validation metric
        # accumulator (trainer._accumulate_metric_state) sums values, so drop any
        # non-tensor entry -- pMF's loss_with_model likewise emits only tensor scalars.
        state = {k: v for k, v in state.items() if torch.is_tensor(v)}
        self.last_state = state
        return total, state

    def loss(self, pred_data, ref_data, ctx):  # pragma: no cover - guard
        raise RuntimeError("HamiltonianProductSplitFlow requires loss_with_model(model, data, ref_data).")


__all__ = ["HamiltonianProductSplitFlow", "ProductSplitContext"]
