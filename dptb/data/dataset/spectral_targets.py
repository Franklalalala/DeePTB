"""Decode optional spectral labels without changing graph-aligned H/S fields."""

import torch
from dptb.data import AtomicDataDict, register_fields

register_fields(graph_fields=[AtomicDataDict.NELEC_KEY])


def attach_spectral_targets(dataset, record, atomicdata):
    """Preserve explicit electron counts; never infer a pseudopotential table.

    H/S decoding and graph validation belong to TargetDecoder. In particular,
    do not overwrite its overlap with raw compact/unpermuted record arrays.
    """
    value = record.get("nelec")
    if value is not None:
        nelec = torch.as_tensor(value, dtype=torch.get_default_dtype()).reshape(-1)
        if (
            nelec.numel() != 1
            or not bool(torch.isfinite(nelec).all())
            or bool((nelec < 0).any())
        ):
            raise ValueError(
                "A record's nelec must be one finite nonnegative electron count"
            )
        atomicdata["nelec"] = nelec

    if getattr(dataset, "get_overlap", False):
        for key in (AtomicDataDict.NODE_OVERLAP_KEY, AtomicDataDict.EDGE_OVERLAP_KEY):
            if key not in atomicdata or atomicdata[key] is None:
                raise KeyError(f"get_overlap=True but decoded record has no {key!r}")

    if not getattr(dataset, "get_eigenvalues", False):
        return
    if record.get("kpoint") is None or record.get("eigenvalue") is None:
        raise KeyError("get_eigenvalues=True requires record kpoint and eigenvalue")
    kpoint = torch.as_tensor(record["kpoint"], dtype=torch.get_default_dtype())
    eigenvalue = torch.as_tensor(record["eigenvalue"], dtype=torch.get_default_dtype())
    if kpoint.ndim == 3 and kpoint.shape[0] == 1:
        kpoint = kpoint[0]
    if eigenvalue.ndim == 3 and eigenvalue.shape[0] == 1:
        eigenvalue = eigenvalue[0]
    if kpoint.ndim != 2 or kpoint.shape[-1] != 3 or kpoint.shape[0] == 0:
        raise ValueError("record kpoint must have nonempty shape (nk, 3)")
    if (
        eigenvalue.ndim != 2
        or eigenvalue.shape[0] != kpoint.shape[0]
        or eigenvalue.shape[1] == 0
    ):
        raise ValueError("record eigenvalue must have shape (nk, nb), matching kpoint")
    if not bool(torch.isfinite(kpoint).all()) or not bool(
        torch.isfinite(eigenvalue).all()
    ):
        raise ValueError("record kpoint/eigenvalue must be finite")
    atomicdata[AtomicDataDict.KPOINT_KEY] = kpoint
    atomicdata[AtomicDataDict.ENERGY_EIGENVALUE_KEY] = eigenvalue.unsqueeze(0)
