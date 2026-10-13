from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from .models import OrbitalBasis, OrbitalChannel
from .radial_quadrature import simpson_rab

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def _number_after(label: str, text: str, cast=float):
    pattern = re.compile(re.escape(label) + r"\s*[:=]?\s*(" + _FLOAT + r")", re.I)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Missing {label!r} in .orb header")
    raw = match.group(1).replace("D", "E").replace("d", "e")
    return cast(float(raw)) if cast is int else cast(raw)


def read_abacus_orb(
    path: str | Path,
    *,
    normalize: bool = True,
    channel_order: str = "file",
) -> OrbitalBasis:
    """Read the text NAO format used by ABACUS.

    The radial values are ``R_l(r)`` on the uniform mesh ``r_i=i*dr`` and
    each channel is labelled by ``L,N``.  The default preserves file order.
    Exact AO order is part of the Hamiltonian contract; sorting by ``(l,zeta)``
    is available only as an explicit diagnostic option.  Radials are normalized
    to ``integral r^2 |R_l(r)|^2 dr = 1`` by default.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    element_match = re.search(r"^\s*Element\s+([A-Za-z][A-Za-z0-9_]*)", text, re.M | re.I)
    element = element_match.group(1) if element_match else path.stem.split("_")[0]

    ecut_match = re.search(r"(?:Energy\s+)?Cutoff\(Ry\)\s+(" + _FLOAT + r")", text, re.I)
    if not ecut_match:
        raise ValueError(f"Could not find Energy Cutoff(Ry) in {path}")
    ecut = float(ecut_match.group(1).replace("D", "E").replace("d", "e"))

    mesh = _number_after("Mesh", text, int)
    dr = _number_after("dr", text, float)
    if mesh < 2 or dr <= 0:
        raise ValueError(f"Invalid mesh/dr in {path}: mesh={mesh}, dr={dr}")

    marker_re = re.compile(r"^\s*Type\s+L\s+N\s*$", re.I)
    channels: list[OrbitalChannel] = []
    idx = 0
    while idx < len(lines):
        if not marker_re.match(lines[idx]):
            idx += 1
            continue
        idx += 1
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        if idx >= len(lines):
            raise ValueError(f"Truncated channel label in {path}")
        ints = re.findall(r"[-+]?\d+", lines[idx])
        if len(ints) < 3:
            raise ValueError(f"Malformed Type/L/N line in {path}: {lines[idx]!r}")
        _, l_raw, zeta_raw = map(int, ints[:3])
        idx += 1
        values: list[float] = []
        while idx < len(lines) and len(values) < mesh:
            if marker_re.match(lines[idx]):
                break
            for token in lines[idx].split():
                try:
                    values.append(float(token.replace("D", "E").replace("d", "e")))
                except ValueError:
                    pass
                if len(values) == mesh:
                    break
            idx += 1
        if len(values) != mesh:
            raise ValueError(
                f"Channel (l={l_raw}, zeta={zeta_raw}) has {len(values)} values; expected {mesh}"
            )
        channels.append(OrbitalChannel(l=l_raw, zeta=zeta_raw, radial=np.asarray(values), source_index=len(channels)))

    if not channels:
        raise ValueError(f"No orbital channels found in {path}")
    labels = [(channel.l, channel.zeta) for channel in channels]
    if len(set(labels)) != len(labels):
        duplicate = next(label for label in labels if labels.count(label) > 1)
        raise ValueError(f"Duplicate orbital channel {duplicate} in {path}")
    order = str(channel_order).lower().replace("-", "_")
    if order == "file":
        pass
    elif order in {"l_zeta", "sorted"}:
        channels.sort(key=lambda c: (c.l, c.zeta, c.source_index))
    else:
        raise ValueError("channel_order must be 'file' or 'l_zeta'")
    input_mesh = mesh
    padded_even_mesh = (mesh % 2 == 0)
    if padded_even_mesh:
        # ABACUS ORB_read.cpp increments an even radial mesh and leaves the
        # appended orbital value at zero before applying Simpson integration.
        mesh += 1
        channels = [
            OrbitalChannel(
                channel.l,
                channel.zeta,
                np.pad(channel.radial, (0, 1)),
                channel.source_index,
            )
            for channel in channels
        ]
    r = np.arange(mesh, dtype=float) * dr
    rab = np.full(mesh, dr, dtype=float)
    input_norms: list[float] = []
    normalized_channels: list[OrbitalChannel] = []
    for channel in channels:
        norm2 = float(simpson_rab(r * r * channel.radial * channel.radial, rab))
        if not np.isfinite(norm2) or norm2 <= 0.0:
            raise ValueError(
                f"Orbital channel (l={channel.l}, zeta={channel.zeta}) in {path} has invalid norm {norm2}"
            )
        input_norms.append(norm2)
        radial = channel.radial / np.sqrt(norm2) if normalize else channel.radial.copy()
        normalized_channels.append(OrbitalChannel(channel.l, channel.zeta, radial, channel.source_index))
    return OrbitalBasis(
        element=element,
        ecut_ry=ecut,
        r=r,
        dr=dr,
        channels=normalized_channels,
        source=path,
        metadata={
            "abacus_radial_normalization_applied": bool(normalize),
            "input_radial_norm2": input_norms,
            "channel_order": order,
            "input_mesh": int(input_mesh),
            "abacus_even_mesh_zero_padding": bool(padded_even_mesh),
            "output_mesh": int(mesh),
            "normalization_quadrature": "ABACUS composite Simpson with uniform rab=dr",
            "channel_labels_in_output": [
                [int(channel.l), int(channel.zeta), int(channel.source_index)]
                for channel in normalized_channels
            ],
        },
    )
