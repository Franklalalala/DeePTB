"""UniTB atomic charge-equilibration head and its local-only physical control.

The state tree remains ``shift_head.response_net.*``. The response accepts
neutral QEq and the local-readout control, with no label-dependent forward path.
"""
from __future__ import annotations

import torch

from dptb.nn.response_shift_head import ResponseShiftHead
from dptb.nn.shift_head import normalize_shift_options


def normalize_charge_options(options):
    """Production defaults with explicit rejection of active archive options.

    ``qeq_local=True`` gives local readout + kappa Gamma q. Setting it false
    retains the original Gamma q form. ``kind=context, local_only=True`` is the
    local-only control with the checkpoint's graph-context channels zeroed.
    """
    if options is None:
        return normalize_shift_options(None)
    if not isinstance(options, dict):
        raise ValueError("shift_head must be a dictionary or None")
    cfg = dict(options)
    cfg.setdefault("mode", "atom")
    if cfg["mode"] == "off":
        return normalize_shift_options(cfg)
    if cfg["mode"] != "atom":
        raise ValueError("UniTB charge head supports mode=atom only; shell heads use the archived model")
    response = cfg.get("response", {})
    if not isinstance(response, dict):
        raise ValueError("UniTB charge head requires response options; additive heads use the archived model")
    response = dict(response)
    response.setdefault("kind", "qeq")
    response.setdefault("qeq_local", response["kind"] == "qeq")
    response.setdefault("canonical_onsite", False)
    response.setdefault("auxiliary_weight", 0.0)
    response.setdefault("output_scale", 0.1)
    cfg["response"] = response
    return normalize_shift_options(cfg)


class ChargeHead(ResponseShiftHead):
    """Checkpoint-compatible neutral QEq or local-only overlap correction.

    Inheriting the common adapter keeps parameter names, initialization draws,
    float64 charge solves, weighted gauge removal and overlap assembly intact.
    """

    def __init__(self, idp, irreps, options, *, dtype=torch.float32, device="cpu"):
        options = normalize_charge_options(options)
        if options["mode"] == "off":
            raise ValueError("Do not instantiate ChargeHead when shift_head.mode=off")
        super().__init__(idp, irreps, options, dtype=dtype, device=device)
