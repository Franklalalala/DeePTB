"""Zero-initialized scalar feedback adapters and trainable parameter selection."""

from __future__ import annotations

import types
from typing import List, Sequence, Tuple
import torch
from torch import nn
from .representation import AOScalarFeedback


def _irreps(obj):
    from e3nn.o3 import Irreps

    if obj is None:
        raise RuntimeError("missing irreps")
    if isinstance(obj, Irreps):
        return obj
    return Irreps(str(obj))


def _zeroe_slices(irreps) -> Tuple[int, List[slice]]:
    irreps = _irreps(irreps)
    slices = []
    n = 0
    for slc, (mul, ir) in zip(irreps.slices(), irreps):
        if int(ir.l) == 0 and int(ir.p) == 1:
            slices.append(slc)
            n += int(mul) * int(ir.dim)
    if n <= 0:
        raise RuntimeError("irreps %s have no 0e block" % irreps)
    return n, slices


def _gather_0e(feat: torch.Tensor, slices: Sequence[slice]) -> torch.Tensor:
    """Select scalars from irreps coordinates; never use on packed AO output."""
    parts = [feat[:, sl] for sl in slices]
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def _add_0e(
    feat: torch.Tensor, delta: torch.Tensor, slices: Sequence[slice]
) -> torch.Tensor:
    out = feat
    off = 0
    pieces = []
    last = 0
    for sl in slices:
        width = sl.stop - sl.start
        if sl.start > last:
            pieces.append(out[:, last : sl.start])
        pieces.append(out[:, sl] + delta[:, off : off + width])
        off += width
        last = sl.stop
    if last < out.shape[-1]:
        pieces.append(out[:, last:])
    if off != delta.shape[-1]:
        raise RuntimeError(
            "0e delta width %d != slice width %d" % (delta.shape[-1], off)
        )
    return torch.cat(pieces, dim=-1) if len(pieces) > 1 else pieces[0]


def _iter_embeddings(model: nn.Module):
    found = []
    for name, mod in model.named_modules():
        if (
            hasattr(mod, "out_node")
            and hasattr(mod, "layers")
            and hasattr(mod, "init_layer")
            and hasattr(mod, "_apply_rme_output_heads")
        ):
            found.append((name, mod))
    if not found:
        raise RuntimeError("no lem_moe embedding modules found on %s" % type(model))
    return found


class ZeroInitWM(nn.Module):
    """Maps working memory [q_features, r_0e] -> hidden 0e. Weight starts at 0."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Persist the input representation, not just a same-shaped weight.
        self.register_buffer("feedback_version", torch.tensor(3, dtype=torch.int64))
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        version = state_dict.get(prefix + "feedback_version")
        if version is None or version.numel() != 1 or int(version) != 3:
            raise RuntimeError(
                "incompatible LoopSCF feedback checkpoint: AO trace feedback v3 "
                "requires newly trained adapters; initialize from a base checkpoint"
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, wm_feat: torch.Tensor) -> torch.Tensor:
        return self.proj(wm_feat)


def _attach_adapters(emb: nn.Module, mode: str = "head") -> None:
    if mode not in ("head", "moe"):
        raise ValueError("unknown feedback injection mode")
    injection_layer = emb.init_layer if mode == "moe" else emb.layers[-1]
    hidden_ir = getattr(injection_layer, "irreps_out", None)
    n_hid, hid_sl = _zeroe_slices(hidden_ir)
    n_out = AOScalarFeedback(emb.idp).n_scalars

    reference = next(emb.parameters())
    emb.wm_node = ZeroInitWM(2 + n_out, n_hid).to(
        device=reference.device, dtype=reference.dtype
    )
    emb.wm_edge = ZeroInitWM(2 + n_out, n_hid).to(
        device=reference.device, dtype=reference.dtype
    )
    emb._wm_hid_slices = hid_sl
    emb._wm_hidden_dim = _irreps(hidden_ir).dim
    emb._wm_n_hid = n_hid
    emb._wm_n_out = n_out
    print(
        "[WM-TrueDiag] adapters on %s: in_dim=%d (2 charge + %d 0e-residual) -> hid_0e=%d"
        % (type(emb).__name__, 2 + n_out, n_out, n_hid),
        flush=True,
    )


def _inject(
    emb: nn.Module, ctx: dict, node_h: torch.Tensor, edge_h: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    wm_n = ctx.get("wm_n")
    wm_e = ctx.get("wm_e")
    if wm_n is None or wm_e is None:
        return node_h, edge_h
    expected_dim = getattr(emb, "_wm_hidden_dim", node_h.shape[-1])
    if node_h.shape[-1] != expected_dim or edge_h.shape[-1] != expected_dim:
        raise RuntimeError(
            "feedback injection irreps do not match hidden feature width"
        )
    n_use = node_h.shape[0]
    if wm_n.shape[0] != n_use:
        raise RuntimeError("node WM count does not match hidden nodes")
    node_h = _add_0e(node_h, emb.wm_node(wm_n), emb._wm_hid_slices)

    ae = ctx.get("active_edges")
    if ae is not None:
        if ae.ndim != 1 or ae.numel() != edge_h.shape[0]:
            raise RuntimeError("active edge mapping does not match hidden edges")
        # The hidden rows follow this order even when every edge is active.
        wm_e = wm_e.index_select(0, ae)
    elif wm_e.shape[0] != edge_h.shape[0]:
        raise RuntimeError("missing active edge mapping for feedback")
    edge_h = _add_0e(edge_h, emb.wm_edge(wm_e), emb._wm_hid_slices)
    return node_h, edge_h


def _wrap_embedding(emb: nn.Module, mode: str) -> None:
    orig_init = emb.init_layer.forward
    orig_heads = emb._apply_rme_output_heads

    def init_fwd(self, *args, **kwargs):
        out = orig_init(*args, **kwargs)
        latents, node_features, edge_features, cutoff_coeffs, active_edges = out
        ctx = getattr(emb, "_wm_ctx", None)
        if ctx is not None:
            ctx["active_edges"] = active_edges
            if mode == "moe":
                node_features, edge_features = _inject(
                    emb, ctx, node_features, edge_features
                )
        return latents, node_features, edge_features, cutoff_coeffs, active_edges

    def heads_fwd(node_features, edge_features, node_one_hot, edge_one_hot):
        ctx = getattr(emb, "_wm_ctx", None)
        if ctx is not None and mode == "head":
            node_features, edge_features = _inject(
                emb, ctx, node_features, edge_features
            )
        return orig_heads(node_features, edge_features, node_one_hot, edge_one_hot)

    emb.init_layer.forward = types.MethodType(init_fwd, emb.init_layer)
    emb._apply_rme_output_heads = heads_fwd


def freeze_by_patterns(
    model: nn.Module, patterns: Sequence[str], extra: Sequence[str] = ()
) -> Tuple[int, int]:
    pats = tuple(patterns) + tuple(extra)
    n_train = n_frozen = 0
    names = []
    for name, p in model.named_parameters():
        if any(pat in name for pat in pats):
            p.requires_grad_(True)
            n_train += p.numel()
            names.append(name)
        else:
            p.requires_grad_(False)
            n_frozen += p.numel()
    total = n_train + n_frozen
    print("=" * 70, flush=True)
    print(
        "FREEZE: trainable %d / %d (%.3f%%), frozen %d, n_tensors=%d"
        % (n_train, total, 100.0 * n_train / max(total, 1), n_frozen, len(names)),
        flush=True,
    )
    for nm in names[:16]:
        print("   trainable:", nm, flush=True)
    if len(names) > 16:
        print("   ... and %d more" % (len(names) - 16), flush=True)
    print("=" * 70, flush=True)
    if n_train == 0:
        raise SystemExit("freeze left zero trainable parameters, patterns=%r" % (pats,))
    if n_frozen == 0:
        raise SystemExit(
            "freeze left nothing frozen, refusing accidental full finetune"
        )
    return n_train, n_frozen
