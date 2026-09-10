"""Stepwise LoopSCF losses and explicit Trainer backward completion."""

from __future__ import annotations

import sys
from typing import Optional, Sequence
import torch
from dptb.data import AtomicData, AtomicDataDict
from .constants import K_DEFAULT
from .spectral import patch_fw10_per_graph


def _format_loop_loss_log(logs) -> str:
    msg = ["[WM-TrueDiag-LOSS]"]
    for k, lv, parts in logs:
        bit = "step=%d L=%.6g" % (k, lv)
        if "fw10" in parts:
            bit += " fw10=%.6g" % parts["fw10"]
        elif "eig" in parts:
            bit += " eig=%.6g" % parts["eig"]
        if "ham" in parts:
            bit += " ham=%.6g" % parts["ham"]
        msg.append(bit)
    return " ".join(msg)


def stepwise_train_loss(
    model, batch, ref, lossfunc, weights: Optional[Sequence[float]] = None
):
    """Interleave orig_fwd_k -> L_k.backward() so only one loop graph is live.

    WM between k and k+1 is detached (occupy is no_grad), so this matches
    0.5 L1 + 0.5 L2 up to float add order. Returns a detached weighted sum.
    """
    one_k = getattr(model, "_wm_one_k", None)
    prepare = getattr(model, "_wm_prepare", None)
    update = getattr(model, "_wm_update", None)
    if one_k is None or prepare is None or update is None:
        raise RuntimeError(
            "stepwise_train_loss needs install_working_memory_true_diag helpers"
        )
    K = int(getattr(model, "_wm_K", K_DEFAULT))
    if weights is None:
        w = [1.0 / float(K)] * K
    else:
        w = list(weights)
        if len(w) != K:
            raise ValueError("weights must have length K=%d" % K)

    state = prepare(batch)
    wm_node = None
    wm_edge = None
    acc = None
    logs = []
    for k in range(1, K + 1):
        out = one_k(batch, k, wm_node, wm_edge, state)
        lk = lossfunc(out, ref)
        parts = dict(getattr(lossfunc, "_last_parts", {}) or {})
        logs.append((k, float(lk.detach()), parts))
        term = w[k - 1] * lk
        term.backward()
        acc = term.detach() if acc is None else acc + term.detach()
        wm_node, wm_edge = update(out, state)
        del out, lk, term
    if acc is None:
        raise RuntimeError("stepwise_train_loss produced no terms")
    print(_format_loop_loss_log(logs), flush=True)
    acc._loopscf_backward_done = True
    return acc, logs


def patch_stepwise_loss(
    K: int = K_DEFAULT, weights: Optional[Sequence[float]] = None
) -> None:
    from dptb.nnops.trainer import Trainer

    node_key = AtomicDataDict.NODE_FEATURES_KEY
    edge_key = AtomicDataDict.EDGE_FEATURES_KEY
    orig = Trainer._loss_on_batch
    if weights is None:
        w = [1.0 / float(K)] * K
    else:
        w = list(weights)
        if len(w) != K:
            raise ValueError("weights must have length K=%d" % K)

    def _loss_on_batch(
        self, batch, lossfunc, *, use_flow=True, allow_self_consistency=True
    ):
        if (
            use_flow
            and getattr(self, "flow_cfm", None) is not None
            and getattr(self.flow_cfm, "enabled", False)
        ):
            return orig(
                self,
                batch,
                lossfunc,
                use_flow=use_flow,
                allow_self_consistency=allow_self_consistency,
            )
        batch = batch.to(self.device)
        batch_info = self._batch_info(batch)
        batch = AtomicData.to_AtomicDataDict(batch)
        batch_for_loss = batch.copy()
        train_stepwise = (
            bool(getattr(self.model, "training", False))
            and hasattr(self.model, "_wm_one_k")
            and not getattr(self.model, "_loopscf_full_bptt", False)
        )
        if train_stepwise:
            if allow_self_consistency and getattr(
                self, "self_consistency_enabled", False
            ):
                raise NotImplementedError(
                    "stepwise LoopSCF does not support the asynchronous self-consistency loss"
                )
            # Do not merge pyg batch_info (__slices__/__data_class__) into the
            # model dict; TorchScript with_edge_vectors cannot cast those keys.
            loss, logs = stepwise_train_loss(
                self.model, batch, batch_for_loss, lossfunc, w
            )
            self._last_loop_parts = logs
            self._last_flow_state = {}
            return loss
        batch = self.model(batch)
        batch.update(batch_info)
        batch_for_loss.update(batch_info)
        loop = batch.get("_loop_preds")
        if not loop:
            loss = lossfunc(batch, batch_for_loss)
        else:
            loss = None
            logs = []
            for k, (dH_node, dH_edge) in enumerate(loop, start=1):
                pk = dict(batch)
                pk[node_key] = dH_node
                pk[edge_key] = dH_edge
                lk = lossfunc(pk, batch_for_loss)
                parts = dict(getattr(lossfunc, "_last_parts", {}) or {})
                logs.append((k, float(lk.detach()), parts))
                term = w[k - 1] * lk
                loss = term if loss is None else loss + term
            if loss is None:
                loss = lossfunc(batch, batch_for_loss)
            self._last_loop_parts = logs
            print(_format_loop_loss_log(logs), flush=True)
        if allow_self_consistency and hasattr(self, "_apply_self_consistency_loss"):
            loss = self._apply_self_consistency_loss(loss, batch)
        self._last_flow_state = {}
        return loss

    if not hasattr(Trainer, "_backward_loss"):
        raise RuntimeError(
            "LoopSCF requires the explicit Trainer._backward_loss hook; update trainer.py"
        )
    Trainer._loss_on_batch = _loss_on_batch
    print(
        "[WM-TrueDiag] patched Trainer._loss_on_batch for interleaved stepwise K=%d w=%s"
        % (K, w),
        flush=True,
    )
    patch_fw10_per_graph()


def patch_smoke_exit(max_iter: int = 2) -> None:
    from dptb.nnops.trainer import Trainer

    orig = Trainer.iteration

    def iteration(self, batch, ref_batch=None):
        out = orig(self, batch, ref_batch=ref_batch)
        n = int(getattr(self, "iter", 0))
        print("[SMOKE] completed iteration, trainer.iter=%d" % n, flush=True)
        if n >= max_iter:
            print("[SMOKE] reached max_iter=%d, exiting cleanly" % max_iter, flush=True)
            sys.exit(0)
        return out

    Trainer.iteration = iteration
    print("[SMOKE] Trainer.iteration will exit after %d steps" % max_iter, flush=True)
