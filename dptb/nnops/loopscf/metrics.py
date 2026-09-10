"""Evaluation-only chemical-potential alignment; energies and smearing are eV."""

import math
import torch
from .occupations import _global_occupations


def fermi_level(eigenvalues_bz, nelec, *, k_weights=None, smearing=0.0):
    """Solve one chemical potential on an explicit BZ integration sample.

    Do not pass a band plotting path. Zero temperature uses the partially
    occupied shell energy or the midgap convention; positive smearing uses
    Fermi-Dirac occupations. Empty/full finite bases have no unique finite mu.
    Projected spectra must not include artificial padding bands.
    """
    ev = eigenvalues_bz.detach().to(torch.float64)
    if ev.ndim != 2 or not ev.numel() or not bool(torch.isfinite(ev).all()):
        raise ValueError("finite rank-2 BZ eigenvalues required")
    if not math.isfinite(smearing) or smearing < 0:
        raise ValueError("smearing must be finite and nonnegative")
    if not 0 < float(nelec) < 2 * ev.shape[1]:
        raise ValueError("finite mu requires 0 < nelec < basis capacity")
    if k_weights is None:
        k_weights = torch.ones(ev.shape[0], device=ev.device)
    occ, weights = _global_occupations(
        ev, nelec, k_weights, torch.ones_like(ev, dtype=torch.bool)
    )
    active = (weights[:, None] > 0).expand_as(ev)
    if smearing == 0:
        partial = active & (occ > 0) & (occ < 2)
        if bool(partial.any()):
            return ev[partial].mean()
        return (ev[active & (occ > 0)].max() + ev[active & (occ == 0)].min()) / 2
    margin = smearing * (
        40 + abs(math.log(float(nelec) / (2 * ev.shape[1] - float(nelec))))
    )
    lo, hi = ev[active].min() - margin, ev[active].max() + margin
    for _ in range(100):
        mid = (lo + hi) / 2
        electrons = (2 * torch.sigmoid((mid - ev) / smearing) * weights[:, None]).sum()
        if electrons < nelec:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def mu_aligned_band_error(
    predicted_path, reference_path, *, mu_pred, mu_ref, window=10.0
):
    """MAE on a reference-defined path window, using independently BZ-solved mu.

    Separate BZ and path inputs prevent estimating metal occupations from a
    plotting path. Equal path/band ordering is required; no silent truncation.
    Return the error and the selected-state denominator.
    """
    if predicted_path.shape != reference_path.shape or predicted_path.ndim != 2:
        raise ValueError("matching rank-2 path spectra required")
    if not math.isfinite(window) or window <= 0:
        raise ValueError("window must be finite and positive")
    pred = predicted_path - mu_pred
    ref = reference_path.to(predicted_path) - mu_ref
    if not bool(torch.isfinite(pred).all() and torch.isfinite(ref).all()):
        raise ValueError("nonfinite path spectrum or chemical potential")
    selected = ref.abs() <= window
    count = int(selected.sum())
    if count == 0:
        raise ValueError("chemical-potential window contains no states")
    return (pred[selected] - ref[selected]).abs().mean(), count
