from __future__ import annotations

"""Offline prior-calibration artifacts for the overlap-Hueckel / basis-onsite priors.

An artifact is produced by ``tools/calibrate_huckel_scales.py`` from a training
LMDB that carries overlap + target features.  It packages, in the RME layout the
dataset was fit in:

* ``edge_scale``  -- ``[num_bond_types, rme_dim]`` multiplicative corrections for
  the Hueckel edge prior (one signed scalar per bond type per orbpair slice,
  constant inside each slice, so the correction commutes with rotations), and
* ``node_table``  -- ``[num_atom_types, rme_dim]`` absolute onsite prior rows
  (per-type mean of the training onsite RME; on PP/NAO crystal data this is a
  near-H0-quality onsite prior whereas the free-atom table is not).

Artifacts are verified fail-closed against the consuming ``idp``: a basis /
SOC-layout mismatch raises instead of silently rescaling the wrong channels.
"""

import hashlib
import json
from typing import Any, Dict, Optional

import torch

CALIBRATION_VERSION = 1
_REQUIRED_KEYS = ("version", "signature")


def basis_fingerprint(idp: Any) -> Optional[str]:
    """Stable fingerprint of the idp's basis + SOC layout knobs (hex sha256)."""
    basis = getattr(idp, "basis", None)
    if not isinstance(basis, dict):
        return None
    payload = {
        "basis": {
            str(k): list(v) if isinstance(v, (list, tuple)) else str(v)
            for k, v in sorted(basis.items())
        },
        "has_soc": bool(getattr(idp, "has_soc", False)),
        "nextham_uureal_mask": bool(getattr(idp, "nextham_uureal_mask", False)),
        "method": str(getattr(idp, "method", "")),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def make_signature(idp: Any, *, rme_dim: int) -> Dict[str, Any]:
    return {
        "basis_fingerprint": basis_fingerprint(idp),
        "rme_dim": int(rme_dim),
        "num_atom_types": int(len(getattr(idp, "chemical_symbol_to_type", {}) or {})),
        "num_bond_types": int(len(getattr(idp, "bond_to_type", {}) or {})),
    }


def load_calibration(path: str, *, idp: Any = None, strict: bool = True) -> Dict[str, Any]:
    """Load and (optionally) verify a calibration artifact.

    Under ``strict`` (default) a version mismatch or a basis-fingerprint
    mismatch against ``idp`` raises ``ValueError`` -- a calibration fit for a
    different basis/SOC layout must never be applied silently.
    """
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or any(k not in artifact for k in _REQUIRED_KEYS):
        raise ValueError(
            f"prior_calibration file {path!r} is not a calibration artifact "
            f"(expected a dict with keys {list(_REQUIRED_KEYS)})."
        )
    version = int(artifact.get("version", -1))
    if version != CALIBRATION_VERSION:
        raise ValueError(
            f"prior_calibration {path!r} has version {version}; this build "
            f"reads version {CALIBRATION_VERSION}."
        )
    if idp is not None and strict:
        expected = basis_fingerprint(idp)
        stored = (artifact.get("signature") or {}).get("basis_fingerprint")
        if expected is not None and stored is not None and stored != expected:
            raise ValueError(
                f"prior_calibration {path!r} was fit for a different basis/SOC "
                "layout (basis_fingerprint mismatch). Refusing to apply per-slice "
                "scales to mismatched channels; regenerate the calibration with "
                "tools/calibrate_huckel_scales.py against this basis."
            )
    for key in ("edge_scale", "node_table"):
        value = artifact.get(key)
        if value is not None and not torch.is_tensor(value):
            artifact[key] = torch.as_tensor(value)
    return artifact
