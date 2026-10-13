from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import Projector, UPFData
from .radial_quadrature import simpson_rab


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
