"""ABACUS numerical atomic orbitals (.orb) and norm-conserving UPF pseudopotentials: readers and radial quadrature.

Shared by the H0 and GF priors and their table builders.

    .orb   R_l(r) on the uniform mesh r_i = i dr per channel (L, N), in file order; an even mesh is padded with one zero
           sample (ABACUS ORB_read), and every channel is normalised to int r^2 R_l(r)^2 dr = 1 with the ABACUS
           composite Simpson rule (rab = dr).
    UPF    PP_R, PP_RAB, PP_LOCAL (Ry), PP_RHOATOM q(r) = 4 pi r^2 rho(r), PP_NLCC rho_c(r), PP_BETA u_l(r) = r beta_l(r)
           with the support index kkbeta (cutoff_radius_index), PP_DIJ (Ry, symmetrised), j of PP_RELBETA for fully
           relativistic files; ultrasoft/PAW files are rejected.
    quadrature  int f(r) dr = sum_i c_i f_i PP_RAB_i with Simpson coefficients 1, 4, 2, ..., 4, 1 / 3 (odd meshes); an
           even mesh closes its last interval with the trapezoid rule (``simpson_rab_weights``).

Units: bohr and Rydberg, as in the files.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


# ------------------------------------------------------------------------------------------------ radial quadrature
def simpson_rab_weights(rab: np.ndarray) -> np.ndarray:
    """Return ABACUS/QE composite-Simpson weights for a UPF radial mesh.

    ``PP_RAB(i)`` is the Jacobian ``dr/dx`` (including the uniform mesh step in
    the UPF convention).  ABACUS ``Integral::Simpson_Integral`` evaluates

    ``integral f(r) dr = sum_i c_i f_i PP_RAB_i``

    with coefficients ``1,4,2,...,4,1`` divided by three.

    Odd meshes (an even number of intervals) use the exact ABACUS composite
    Simpson rule.  Production norm-conserving UPFs (e.g. ONCV Si/Ca) instead
    ship an *even* number of mesh points, for which a single composite-Simpson
    rule does not exist.  Rather than fail closed (the v0.4.1 behaviour, which
    rejected every real UPF), an even mesh is integrated by applying composite
    Simpson to the first ``n-1`` samples (an even number of intervals) and
    closing the final interval ``[n-2, n-1]`` with the trapezoid rule.  This
    keeps every tabulated ``PP_R``/``PP_RAB`` sample untouched -- no resampling,
    padding, or truncation -- and reduces to pure composite Simpson for odd
    meshes.  The trailing trapezoid is second order on a single interval; on the
    thousand-point production meshes where the integrand tail is already
    negligible its contribution is far below the acceptance tolerance.
    """
    rab = np.asarray(rab, dtype=float)
    if rab.ndim != 1:
        raise ValueError(f"PP_RAB must be one-dimensional, got shape {rab.shape}")
    if rab.size < 3:
        raise ValueError(
            "ABACUS-compatible radial integration requires at least three "
            f"mesh points, got {rab.size}."
        )
    if not np.isfinite(rab).all() or np.any(rab <= 0.0):
        raise ValueError("PP_RAB must contain finite positive radial Jacobians")
    n = rab.size
    if n % 2 == 1:
        # Odd mesh (even number of intervals): exact ABACUS composite Simpson.
        coeff = np.ones(n, dtype=float)
        coeff[1:-1:2] = 4.0
        coeff[2:-1:2] = 2.0
        return coeff * rab / 3.0
    # Even mesh (odd number of intervals): composite Simpson on samples
    # 0..n-2 (weights 1,4,2,...,4,1 / 3) plus a trapezoid on [n-2, n-1].
    coeff = np.zeros(n, dtype=float)
    coeff[0] += 1.0 / 3.0
    coeff[1:n - 2:2] += 4.0 / 3.0
    coeff[2:n - 2:2] += 2.0 / 3.0
    coeff[n - 2] += 1.0 / 3.0
    # Closing interval via the trapezoid rule (unit x-step, PP_RAB Jacobian).
    coeff[n - 2] += 0.5
    coeff[n - 1] += 0.5
    return coeff * rab


def simpson_rab(values: np.ndarray, rab: np.ndarray, *, axis: int = -1) -> np.ndarray:
    """Integrate tabulated values with ABACUS's exact ``PP_RAB`` convention."""
    array = np.asarray(values)
    moved = np.moveaxis(array, axis, -1)
    weights = simpson_rab_weights(rab)
    if moved.shape[-1] != weights.size:
        raise ValueError(
            f"Radial data length {moved.shape[-1]} does not match PP_RAB length {weights.size}"
        )
    return np.sum(moved * weights, axis=-1)


def cumulative_simpson_rab(values: np.ndarray, rab: np.ndarray) -> np.ndarray:
    """ABACUS ``Simpson_Integral_0toall`` cumulative radial integral.

    Even-index endpoints use composite Simpson; the intervening odd point uses
    the local trapezoid exactly as in the ABACUS implementation.  For an even
    mesh (odd number of intervals) the composite-Simpson pairs cover samples
    ``0..n-2`` and the final interval ``[n-2, n-1]`` is closed with the
    trapezoid rule, so the cumulative endpoint matches :func:`simpson_rab`
    exactly and no sample is dropped.
    """
    values = np.asarray(values)
    weights = np.asarray(rab, dtype=float)
    # Reuse all mesh validation from the full-integral weight builder.
    simpson_rab_weights(weights)
    if values.ndim != 1 or values.shape != weights.shape:
        raise ValueError("cumulative radial data and PP_RAB must be matching 1-D arrays")
    n = values.size
    out = np.zeros(n, dtype=np.result_type(values, float))
    # Largest even index reached by complete Simpson pairs: n-1 (odd mesh) or
    # n-2 (even mesh).  range(1, last_pair_end, 2) is identical to the original
    # range(1, n, 2) for odd meshes, so odd-mesh output is unchanged.
    last_pair_end = n - 1 if (n % 2 == 1) else n - 2
    f3 = values[0] * weights[0]
    for i in range(1, last_pair_end, 2):
        f1 = f3
        f2 = values[i] * weights[i]
        f3 = values[i + 1] * weights[i + 1]
        out[i] = out[i - 1] + 0.5 * (f1 + f2)
        out[i + 1] = out[i - 1] + (f1 + 4.0 * f2 + f3) / 3.0
    if n % 2 == 0:
        # Close the final odd interval with the trapezoid rule.
        fa = values[n - 2] * weights[n - 2]
        fb = values[n - 1] * weights[n - 1]
        out[n - 1] = out[n - 2] + 0.5 * (fa + fb)
    return out


# ------------------------------------------------------------------------------------------------------ data model
@dataclass(frozen=True)
class OrbitalChannel:
    l: int
    zeta: int
    radial: np.ndarray
    # Position of this channel in the .orb file.  Keeping it makes basis-order
    # provenance explicit and prevents an apparently harmless sort from silently
    # permuting AO/RME channels.
    source_index: int = -1


@dataclass
class OrbitalBasis:
    element: str
    ecut_ry: float
    r: np.ndarray
    dr: float
    channels: list[OrbitalChannel]
    source: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def rcut(self) -> float:
        return float(self.r[-1])

    @property
    def lmax(self) -> int:
        return max((c.l for c in self.channels), default=-1)

    @property
    def norb(self) -> int:
        return sum(2 * c.l + 1 for c in self.channels)

    def descriptors(self) -> list["OrbitalDescriptor"]:
        out: list[OrbitalDescriptor] = []
        for ic, channel in enumerate(self.channels):
            for m in abacus_m_order(channel.l):
                out.append(OrbitalDescriptor(ic, channel.l, channel.zeta, m))
        return out


@dataclass(frozen=True)
class OrbitalDescriptor:
    channel_index: int
    l: int
    zeta: int
    m: int


@dataclass(frozen=True)
class Projector:
    index: int
    l: int
    j: float | None
    radial_u: np.ndarray  # UPF PP_BETA stores u_l(r)=r*beta_l(r)
    cutoff_index: int
    cutoff_radius: float


@dataclass
class UPFData:
    element: str
    z_valence: float
    r: np.ndarray
    rab: np.ndarray
    vloc_ry: np.ndarray
    rhoatom_q: np.ndarray  # q(r)=4*pi*r^2*rho_atom(r)
    dij_ry: np.ndarray
    projectors: list[Projector]
    has_so: bool
    functional: str = ""
    nlcc: np.ndarray | None = None
    nlcc_is_radial_density: bool = True
    source: Path | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def max_projector_cutoff(self) -> float:
        return max((p.cutoff_radius for p in self.projectors), default=0.0)


@dataclass
class SpeciesData:
    orb: OrbitalBasis
    upf: UPFData


def abacus_m_order(l: int) -> list[int]:
    """ABACUS real-harmonic order: 0,+1,-1,+2,-2,..."""
    order = [0]
    for m in range(1, l + 1):
        order.extend((m, -m))
    return order


# ------------------------------------------------------------------------------------------------------ ORB reader
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


# ------------------------------------------------------------------------------------------------------ UPF reader
def _strip_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find(root: ET.Element, name: str) -> ET.Element | None:
    target = name.upper()
    for elem in root.iter():
        if _strip_tag(elem.tag).upper() == target:
            return elem
    return None


def _attr(elem: ET.Element | None, names: Iterable[str], default=None):
    if elem is None:
        return default
    attrs = {k.lower(): v for k, v in elem.attrib.items()}
    for name in names:
        if name.lower() in attrs:
            return attrs[name.lower()]
    return default


def _as_bool(value, default=False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"t", "true", ".true.", "1", "yes"}


def _numbers(elem: ET.Element | None) -> np.ndarray:
    if elem is None or not elem.text:
        return np.empty(0, dtype=float)
    tokens = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?", elem.text
    )
    return np.asarray([float(t.replace("D", "E").replace("d", "e")) for t in tokens])


def _parse_xml(path: Path) -> ET.Element:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        return ET.fromstring(raw)
    except ET.ParseError:
        # Some UPF writers leave a declaration/DOCTYPE before the UPF root or
        # use bare ampersands in comments.  Keep this recovery narrow.
        start = raw.find("<UPF")
        end = raw.rfind("</UPF>")
        if start < 0 or end < 0:
            raise
        cleaned = raw[start : end + len("</UPF>")]
        cleaned = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", cleaned)
        return ET.fromstring(cleaned)


def _indexed_elements(
    root: ET.Element,
    prefix: str,
) -> tuple[list[tuple[int, ET.Element]], list[dict[str, int]]]:
    """Return UPF repeated elements in the same slots used by QE/ABACUS.

    UPF-v2 readers address ``PP_BETA.1``, ``PP_BETA.2``, ... directly, while
    schema-style repeated ``pp_beta`` elements are consumed in document order.
    The optional ``index`` attribute is validation metadata only: several old
    UPFs contain a wrong attribute, and using it to sort would silently permute
    projectors relative to ``PP_DIJ``.
    """
    pattern = re.compile(rf"^{re.escape(prefix.upper())}(?:[._](\d+))?$")
    raw: list[tuple[int | None, ET.Element]] = []
    for elem in root.iter():
        match = pattern.match(_strip_tag(elem.tag).upper())
        if match:
            suffix = None if match.group(1) is None else int(match.group(1)) - 1
            raw.append((suffix, elem))
    if not raw:
        return [], []
    has_suffix = [slot is not None for slot, _ in raw]
    if any(has_suffix) and not all(has_suffix):
        raise ValueError(f"Mixed indexed and unindexed {prefix} tags are ambiguous")
    if all(has_suffix):
        slots = [int(slot) for slot, _ in raw]
        expected = list(range(len(raw)))
        if sorted(slots) != expected:
            raise ValueError(
                f"{prefix} tag suffixes must be unique and contiguous 1..{len(raw)}, "
                f"got {[slot + 1 for slot in slots]}"
            )
        ordered = sorted(((int(slot), elem) for slot, elem in raw), key=lambda item: item[0])
    else:
        ordered = [(slot, elem) for slot, (_, elem) in enumerate(raw)]

    mismatches: list[dict[str, int]] = []
    for slot, elem in ordered:
        attr = _attr(elem, ["index"], None)
        if attr is None:
            continue
        try:
            if not re.fullmatch(r"[+]?[0-9]+", str(attr).strip()):
                raise ValueError("index must be a decimal integer")
            attr_index = int(attr)
        except (TypeError, ValueError):
            # Legacy Fortran I1 output overflows at projector 10 (index="*").
            # Numbered tags, already checked above, remain the authoritative slots.
            if str(attr).strip() == '*' and all(has_suffix) and slot + 1 >= 10:
                mismatches.append({'slot': int(slot + 1), 'index_attribute': str(attr)})
                continue
            raise ValueError(f"Non-numeric index attribute on {prefix} slot {slot + 1}: {attr!r}")
        if attr_index != slot + 1:
            mismatches.append(
                {"slot": int(slot + 1), "index_attribute": int(attr_index)}
            )
    return ordered, mismatches


def read_upf(
    path: str | Path,
    *,
    allow_rab_fallback: bool = False,
    strict_projector_index_attributes: bool = False,
) -> UPFData:
    """Read a norm-conserving UPF and preserve its exact radial/projector contract.

    Ultrasoft/PAW augmentation is deliberately rejected.  By default a missing
    or malformed ``PP_RAB`` is fatal because ABACUS radial transforms integrate
    with that Jacobian, not with a newly inferred ``PP_R`` spacing.
    """
    path = Path(path)
    root = _parse_xml(path)
    header = _find(root, "PP_HEADER")
    if header is None:
        raise ValueError(f"{path} has no PP_HEADER")

    element = str(_attr(header, ["element"], path.stem.split(".")[0])).strip()
    z_valence = float(_attr(header, ["z_valence", "z_valence_charge"], 0.0))
    has_so = _as_bool(_attr(header, ["has_so", "has_spin_orbit"], False))
    is_ultrasoft = _as_bool(_attr(header, ["is_ultrasoft"], False))
    is_paw = _as_bool(_attr(header, ["is_paw"], False))
    if is_ultrasoft or is_paw:
        raise NotImplementedError(
            "This reference supports norm-conserving KB UPFs only; USPP/PAW augmentation is not implemented."
        )
    functional = str(_attr(header, ["functional", "dft"], ""))

    mesh = _find(root, "PP_MESH")
    mesh_root = mesh if mesh is not None else root
    r = _numbers(_find(mesh_root, "PP_R"))
    rab = _numbers(_find(mesh_root, "PP_RAB"))
    if r.ndim != 1 or r.size < 3 or not np.isfinite(r).all():
        raise ValueError(f"{path} has no usable PP_R mesh")
    if np.any(np.diff(r) <= 0.0):
        raise ValueError(f"PP_R must be strictly increasing in {path}")
    rab_fallback = False
    if rab.shape != r.shape:
        if not allow_rab_fallback:
            raise ValueError(
                f"{path} has PP_RAB shape {rab.shape}, expected {r.shape}; "
                "set allow_rab_fallback=True only for non-reproduction diagnostics."
            )
        rab = np.gradient(r)
        rab_fallback = True
    # Validate the exact production quadrature contract immediately.
    simpson_rab(np.ones_like(r), rab)

    def exact_radial(tag: str, *, required: bool) -> np.ndarray | None:
        elem = _find(root, tag)
        if elem is None:
            if required:
                raise ValueError(f"{path} has no {tag}")
            return None
        arr = _numbers(elem)
        if arr.shape != r.shape:
            raise ValueError(f"{tag} in {path} has shape {arr.shape}, expected {r.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{tag} in {path} contains non-finite values")
        return arr

    vloc = exact_radial("PP_LOCAL", required=True)
    rhoatom = exact_radial("PP_RHOATOM", required=True)
    nlcc = exact_radial("PP_NLCC", required=False)
    assert vloc is not None and rhoatom is not None

    beta_slots, beta_index_mismatches = _indexed_elements(root, "PP_BETA")
    relbeta_slots, relbeta_index_mismatches = _indexed_elements(root, "PP_RELBETA")
    if strict_projector_index_attributes and (beta_index_mismatches or relbeta_index_mismatches):
        raise ValueError(
            f"Projector index attributes disagree with UPF slots in {path}: "
            f"beta={beta_index_mismatches}, relbeta={relbeta_index_mismatches}"
        )

    relbeta: dict[int, tuple[int | None, float | None]] = {}
    for slot, elem in relbeta_slots:
        lval = _attr(elem, ["angular_momentum", "lll", "l"], None)
        jval = _attr(elem, ["total_angular_momentum", "tot_ang_mom", "jjj", "j"], None)
        relbeta[slot] = (
            None if lval is None else int(float(lval)),
            None if jval is None else float(jval),
        )

    projectors: list[Projector] = []
    projector_cutoff_records: list[dict[str, float | int | None]] = []
    for slot, elem in beta_slots:
        l = int(float(_attr(elem, ["angular_momentum", "lll", "l"], 0)))
        j_attr = _attr(elem, ["total_angular_momentum", "tot_ang_mom", "jjj", "j"], None)
        j: float | None = None if j_attr is None else float(j_attr)
        if slot in relbeta:
            l_rel, j_rel = relbeta[slot]
            if l_rel is not None and l_rel != l:
                raise ValueError(f"Inconsistent l for PP_BETA slot {slot + 1} in {path}")
            if j_rel is not None:
                j = j_rel
        cutoff_index = int(
            float(_attr(elem, ["cutoff_radius_index", "kkbeta", "size"], r.size))
        )
        cutoff_index = max(1, min(cutoff_index, r.size))
        declared_cutoff_radius = _attr(
            elem, ["cutoff_radius", "beta_cutoff_radius"], None
        )
        # ABACUS/QE use the 1-based kkbeta/cutoff_radius_index as the support
        # bound.  The optional floating radius is retained only as provenance;
        # old UPFs sometimes write zero or a rounded value there.
        cutoff_radius = float(r[cutoff_index - 1])
        u = _numbers(elem)
        if u.size > r.size:
            raise ValueError(
                f"PP_BETA slot {slot + 1} in {path} has {u.size} values for mesh {r.size}"
            )
        if u.size < r.size:
            u = np.pad(u, (0, r.size - u.size))
        if not np.isfinite(u).all():
            raise ValueError(f"PP_BETA slot {slot + 1} in {path} contains non-finite values")
        projector_cutoff_records.append(
            {
                "slot": int(slot + 1),
                "cutoff_radius_index": int(cutoff_index),
                "mesh_cutoff_radius_bohr": float(cutoff_radius),
                "declared_cutoff_radius_bohr": (
                    None if declared_cutoff_radius is None else float(declared_cutoff_radius)
                ),
            }
        )
        projectors.append(
            Projector(
                index=slot,
                l=l,
                j=j,
                radial_u=u,
                cutoff_index=cutoff_index,
                cutoff_radius=cutoff_radius,
            )
        )

    nbeta = len(projectors)
    declared_nbeta = _attr(header, ["number_of_proj", "number_of_projectors"], None)
    if declared_nbeta is not None and int(float(declared_nbeta)) != nbeta:
        raise ValueError(
            f"PP_HEADER declares {declared_nbeta} projectors but {nbeta} PP_BETA slots were read"
        )
    dij_arr = _numbers(_find(root, "PP_DIJ"))
    if nbeta == 0:
        if dij_arr.size not in {0, 1} or (dij_arr.size == 1 and abs(dij_arr[0]) > 0.0):
            raise ValueError(f"PP_DIJ is nonempty in projector-free UPF {path}")
        dij = np.zeros((0, 0), dtype=float)
    elif dij_arr.size == nbeta * nbeta:
        # QE/ABACUS reads this directly into dion(nb,nb) in PP_BETA slot order.
        dij_raw = dij_arr.reshape(nbeta, nbeta)
        scale = max(1.0, float(np.max(np.abs(dij_raw))))
        dij_antisymmetry = float(np.max(np.abs(dij_raw - dij_raw.T)))
        if dij_antisymmetry > 1.0e-10 * scale:
            raise ValueError(
                f"PP_DIJ in {path} is not symmetric: max residual {dij_antisymmetry:.3e}"
            )
        dij = 0.5 * (dij_raw + dij_raw.T)
    else:
        raise ValueError(
            f"PP_DIJ in {path} has {dij_arr.size} entries for {nbeta} projectors"
        )
    if nbeta == 0:
        dij_antisymmetry = 0.0

    if has_so and any(p.j is None for p in projectors):
        missing = [p.index + 1 for p in projectors if p.j is None]
        raise ValueError(f"Fully relativistic UPF {path} lacks j for PP_BETA slots {missing}")
    scale = max(1.0, float(np.max(np.abs(dij))) if dij.size else 0.0)
    for ip, p in enumerate(projectors):
        gp = (p.l, p.j if has_so else None)
        for iq, q in enumerate(projectors):
            gq = (q.l, q.j if has_so else None)
            if gp != gq and abs(dij[ip, iq]) > 1.0e-12 * scale:
                raise NotImplementedError(
                    f"PP_DIJ couples projector groups {gp} and {gq} in {path}; "
                    "the reference evaluator refuses to discard this term."
                )

    raw_rhoatom_charge = float(simpson_rab(rhoatom, rab))
    metadata = {
        "pseudo_type": _attr(header, ["pseudo_type"], ""),
        "relativistic": _attr(header, ["relativistic"], ""),
        "suggested_ecutwfc_ry": _attr(header, ["wfc_cutoff", "ecutwfc"], None),
        "suggested_ecutrho_ry": _attr(header, ["rho_cutoff", "ecutrho"], None),
        "rhoatom_raw_integral_electrons": raw_rhoatom_charge,
        "rhoatom_declared_z_valence": z_valence,
        "radial_quadrature": "ABACUS composite Simpson with PP_RAB",
        "pp_rab_fallback_used": rab_fallback,
        "projector_order": "UPF tag suffix for v2; document order for repeated schema tags",
        "projector_index_attribute_mismatches": beta_index_mismatches,
        "relbeta_index_attribute_mismatches": relbeta_index_mismatches,
        "dij_max_antisymmetry_ry": dij_antisymmetry,
        "dij_block_structure_validated": True,
        "projector_cutoffs": projector_cutoff_records,
        "projector_support_convention": "PP_BETA cutoff_radius_index/kkbeta on PP_R",
    }
    return UPFData(
        element=element,
        z_valence=z_valence,
        r=r,
        rab=rab,
        vloc_ry=vloc,
        rhoatom_q=rhoatom,
        dij_ry=dij,
        projectors=projectors,
        has_so=has_so,
        functional=functional,
        nlcc=nlcc,
        # PP_NLCC is the radial core density rho_c(r), unlike PP_RHOATOM,
        # which is q(r)=4*pi*r^2*rho(r).
        nlcc_is_radial_density=True,
        source=path,
        metadata=metadata,
    )


def align_soc_projectors(species_data):
    """Projector samples cut with the ABACUS setupNonlocal convention (setup_nonlocal.cpp: ``cut_mesh`` is the last
    index with |beta| > 1e-10, rounded up to odd, samples ``ir < cut_mesh`` kept; clamped to the mesh). cutoff_radius
    keeps the conservative support envelope. Inputs are never modified."""
    from dataclasses import replace
    aligned = {}
    for symbol, data in species_data.items():
        if len(data.upf.r) == 0:
            raise ValueError(f"{symbol}: SOC projector alignment requires a nonempty radial mesh")
        projectors = []
        for projector in data.upf.projectors:
            nz = np.flatnonzero(np.abs(projector.radial_u) > 1e-10)
            cut = int(nz[-1]) if len(nz) else len(data.upf.r)
            if cut % 2 == 0:
                cut += 1
            cut = min(cut, len(data.upf.r))
            radial_u = projector.radial_u.copy()
            radial_u[cut:] = 0
            projectors.append(replace(projector, radial_u=radial_u, cutoff_index=cut))
        aligned[symbol] = replace(data, upf=replace(data.upf, projectors=projectors))
    return aligned


def neutral_atom_potential_radial(upf: UPFData, *, normalize_atomic_charge: bool = True,
                                  tail_gauge: str = "none") -> tuple[np.ndarray, np.ndarray]:
    """V_NA(r) = Vloc(r) + V_H[rho_atom](r) in Rydberg on the UPF mesh, with q = 4 pi r^2 rho (PP_RHOATOM) and
    V_H(r) = e^2 [(1/r) int_0^r q + int_r^inf q / s] (e^2 = 2 Ry bohr); q is scaled to z_valence electrons when
    ``normalize_atomic_charge``. tail_gauge 'last_point_zero' shifts V_NA to vanish at the last mesh point."""
    r = np.asarray(upf.r, dtype=float)
    q = np.asarray(upf.rhoatom_q, dtype=float).copy()
    if normalize_atomic_charge:
        raw_charge = float(cumulative_simpson_rab(q, upf.rab)[-1])
        if raw_charge <= 0.0:
            raise ValueError("PP_RHOATOM has non-positive radial integral")
        q *= float(upf.z_valence) / raw_charge
    enclosed = cumulative_simpson_rab(q, upf.rab)
    q_over_r = np.zeros_like(q)
    mask = r > 1e-14
    q_over_r[mask] = q[mask] / r[mask]
    outer_from_zero = cumulative_simpson_rab(q_over_r, upf.rab)
    outer = outer_from_zero[-1] - outer_from_zero
    vh = 2.0 * (outer + np.divide(enclosed, r, out=np.zeros_like(r), where=mask))
    if len(r) > 1:
        vh[0] = 2.0 * outer[0]
    vna = upf.vloc_ry + vh
    gauge = str(tail_gauge).lower()
    if gauge == "last_point_zero":
        vna = vna - float(vna[-1])
    elif gauge != "none":
        raise ValueError("tail_gauge must be 'none' or 'last_point_zero'")
    return r, vna


def projector_beta_values(rmesh: np.ndarray, radial_u: np.ndarray, cutoff: float, radii: np.ndarray) -> np.ndarray:
    """beta_l(r) = u_l(r) / r from the UPF u_l by a not-a-knot cubic spline, zero outside [r_0, cutoff]; near r = 0
    the slope u'(r_s) at r_s = max(1e-4 r_1, 1e-8)."""
    from scipy.interpolate import CubicSpline
    rmesh = np.asarray(rmesh, dtype=float)
    spline = CubicSpline(rmesh, np.asarray(radial_u, dtype=float), bc_type="not-a-knot", extrapolate=False)

    def ev(x, nu=0):
        out = np.zeros_like(x, dtype=float)
        m = (x >= rmesh[0]) & (x <= cutoff)
        if np.any(m):
            out[m] = np.nan_to_num(spline(x[m], nu=nu))
        return out
    radii = np.asarray(radii, dtype=float)
    u = ev(radii)
    out = np.zeros_like(radii)
    mask = radii > max(1e-10, rmesh[1] * 1e-5)
    out[mask] = u[mask] / radii[mask]
    if np.any(~mask):
        out[~mask] = ev(np.full(np.count_nonzero(~mask), max(rmesh[1] * 1e-4, 1e-8)), 1)
    return out
