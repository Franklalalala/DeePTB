from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .grid_collocation import FFTGridAOCache
from .spatial import PeriodicAtomIndex, iter_pair_images
from .models import BlockKey, SpeciesData, Structure
from .nonlocal_kb import nonlocal_pair_block
from .quadrature import scalar_pair_integrals
from .quadrature import local_pair_integral_fft_grid, local_pair_integral_fft_grid_cached
from .radial import OrbitalEvaluator
from .reciprocal import PeriodicField, build_periodic_field
from .provenance import input_file_manifest, structure_fingerprint
from .scalar_upf import scalarize_upf


@dataclass
class AssemblyResult:
    h_blocks_ry: dict[BlockKey, np.ndarray]
    s_blocks: dict[BlockKey, np.ndarray]
    components_ry: dict[str, dict[BlockKey, np.ndarray]]
    field: PeriodicField
    orbital_counts: list[int]
    metadata: dict[str, object]


def hermiticity_report(blocks: Mapping[BlockKey, np.ndarray]) -> dict[str, object]:
    """Measure, but do not repair, the real-space Hermiticity contract.

    The pair ``(i,j,R)`` must equal the conjugate transpose of
    ``(j,i,-R)``.  Reporting the pre-repair residual is important: automatic
    symmetrization otherwise hides edge-order, phase, or quadrature mistakes.
    """
    seen: set[BlockKey] = set()
    missing: list[tuple[int, int, int, int, int]] = []
    max_abs = 0.0
    sum_sq = 0.0
    count = 0
    for key, block in blocks.items():
        if key in seen:
            continue
        mate = BlockKey(key.j, key.i, tuple(-x for x in key.R))
        if mate not in blocks:
            missing.append(key.as_tuple())
            seen.add(key)
            continue
        residual = np.asarray(block) - np.asarray(blocks[mate]).conj().T
        if residual.size:
            max_abs = max(max_abs, float(np.max(np.abs(residual))))
            sum_sq += float(np.vdot(residual.ravel(), residual.ravel()).real)
            count += int(residual.size)
        seen.add(key)
        seen.add(mate)
    return {
        "max_abs": max_abs,
        "rms": float(np.sqrt(sum_sq / count)) if count else 0.0,
        "missing_counterparts": missing,
        "missing_counterpart_count": len(missing),
    }


def _spin_duplicate(block: np.ndarray) -> np.ndarray:
    ni, nj = block.shape
    out = np.zeros((2 * ni, 2 * nj), dtype=complex)
    out[:ni, :nj] = block
    out[ni:, nj:] = block
    return out


def _hermitize(blocks: dict[BlockKey, np.ndarray]) -> None:
    visited: set[BlockKey] = set()
    for key in list(blocks):
        if key in visited:
            continue
        counterpart = BlockKey(key.j, key.i, tuple(-x for x in key.R))
        if counterpart in blocks:
            avg = 0.5 * (blocks[key] + blocks[counterpart].conj().T)
            blocks[key] = avg
            blocks[counterpart] = avg.conj().T
            visited.add(counterpart)
        elif key.i == key.j and key.R == (0, 0, 0):
            blocks[key] = 0.5 * (blocks[key] + blocks[key].conj().T)
        visited.add(key)


def assemble_h0(
    structure: Structure,
    species_data: Mapping[str, SpeciesData],
    *,
    ecutrho_ry: float | None = None,
    fft_shape: tuple[int, int, int] | None = None,
    xc: str = "LDA_PZ81",
    pair_grid_step_bohr: float = 0.30,
    local_integration: str = "fft_grid",
    local_cache_max_mb: float | None = 512.0,
    local_cache_chunk_size: int = 100_000,
    local_cache_key_decimals: int | None = None,
    projector_grid_step_bohr: float | None = None,
    include_nonlocal: bool = True,
    include_nlcc: bool = True,
    hartree_g0: str | float = "zero",
    local_g0: str = "alpha",
    total_electrons: float | None = None,
    radial_g_round_decimals: int | None = None,
    pbe_density_threshold: float = 1.0e-6,
    prune_frobenius_ry: float = 0.0,
    strict_reproduction: bool = True,
    hermitize: bool = True,
    spatial_backend: str = "reference",
    structure_factor_backend: str = "direct",
    structure_factor_eps: float = 1.0e-12,
    structure_factor_max_work_mb: float | None = 512.0,
    pair_support: str = "orbital_overlap",
    structure_factor_nthreads: int = 1,
    field_backend: str = "numpy",
    compute_device: str = "cpu",
    field_max_work_mb: float | None = 1024.0,
    nspin: int = 4,
    initial_moments_z=None,
    two_center_backend: str = 'midpoint',
    two_center_dr_bohr: float = .01,
    two_center_cache_dir: str | None = None,
    output_atom_cell_shifts=None,
    cuda_precompiled: bool = False,
    offline_table_dir: str | None = None,
    pseudo_rcut_bohr: float = 15.0,
) -> AssemblyResult:
    """Assemble H[rho0]=T+Vnl+Vloc+VH[rho0]+Vxc[rho0].

    All local-potential contributions are evaluated together from the complete
    superposed density. ``nspin=1`` returns scalar spatial blocks after the
    ABACUS non-SOC UPF reduction; ``nspin=4`` returns spin-block-major blocks.
    ``two_center_backend='pyabacus'`` uses official ABACUS S/T/KB integrals.
    ``local_integration='fft_grid_periodic'`` reuses explicit integer lattice
    images and contracts on ``compute_device``. Under ``strict_reproduction=True``
    both ``ecutrho_ry`` and the exact
    ``fft_shape`` are mandatory; the orbital-file cutoff is not a valid silent
    substitute for the charge-density grid.

    ``output_atom_cell_shifts`` specifies integer shifts q_i of each output
    atom's home cell. H0, S, components and geometry metadata are transformed
    together: R_out = R_in + q_i - q_j. None preserves the input convention.
    """
    if output_atom_cell_shifts is not None:
        from .cell_gauge import validate_cell_shifts
        output_atom_cell_shifts = validate_cell_shifts(output_atom_cell_shifts, len(structure.atoms))
    if nspin not in (1, 4):
        raise ValueError('Only nonmagnetic nspin=1 and spinor nspin=4 are supported')
    if cuda_precompiled and (two_center_backend != 'pyabacus' or local_integration != 'fft_grid_periodic' or not str(compute_device).startswith('cuda')):
        raise ValueError('cuda_precompiled requires pyabacus table semantics, fft_grid_periodic and a CUDA device')
    if cuda_precompiled and offline_table_dir is None:
        import os
        offline_table_dir = os.environ.get('H0_OFFLINE_TABLE_DIR')
        if offline_table_dir is None:
            raise ValueError('Precompiled runtime requires offline_table_dir or H0_OFFLINE_TABLE_DIR; prepare tables once before inference.')
    if initial_moments_z is not None and (nspin!=4 or local_integration!='fft_grid_periodic' or two_center_backend!='pyabacus'):
        raise ValueError('Explicit initial magnetic moments require spinor pyabacus with fft_grid_periodic')
    original_inputs = species_data
    if nspin == 1:
        species_data = {symbol: SpeciesData(data.orb, scalarize_upf(data.upf))
                        for symbol, data in species_data.items()}
    if two_center_backend not in ('midpoint','pyabacus'):
        raise ValueError('two_center_backend must be midpoint or pyabacus')
    two_center = None
    if two_center_backend == 'pyabacus':
        if cuda_precompiled:
            if offline_table_dir is not None:
                from .table_contract import verify_contract
                verify_contract(offline_table_dir)
                from .offline import prepared_two_center
                two_center = prepared_two_center(species_data, store=offline_table_dir, dr_bohr=two_center_dr_bohr, nspin=nspin, device=compute_device)
            else:
                from .cuda_two_center import CUDATwoCenter
                two_center = CUDATwoCenter(species_data,dr_bohr=two_center_dr_bohr,cache_dir=two_center_cache_dir,nspin=nspin,device=compute_device)
        else:
            from .pyabacus_integrals import PyAbacusTwoCenter
            two_center = PyAbacusTwoCenter(species_data,dr_bohr=two_center_dr_bohr,cache_dir=two_center_cache_dir,nspin=nspin)
    spatial_mode = str(spatial_backend).lower()
    sf_mode = str(structure_factor_backend).lower()
    support_mode = str(pair_support).lower()
    if spatial_mode not in {"reference", "indexed"}:
        raise ValueError("spatial_backend must be 'reference' or 'indexed'")
    if sf_mode not in {"direct", "gaussian_nufft", "finufft", "cufinufft", "torch_gaussian"}:
        raise ValueError("Unknown structure_factor_backend")
    if support_mode not in {"orbital_overlap", "nonlocal_complete"}:
        raise ValueError("pair_support must be 'orbital_overlap' or 'nonlocal_complete'")
    if strict_reproduction and sf_mode != "direct":
        raise ValueError(
            "strict_reproduction=True requires structure_factor_backend='direct'; "
            "NUFFT is an explicit controlled approximation. Set "
            "strict_reproduction=False and still supply the exact FFT grid/cutoff "
            "to test that approximation in isolation."
        )
    if sf_mode != "direct" and (ecutrho_ry is None or fft_shape is None):
        raise ValueError(
            "NUFFT assembly requires explicit ecutrho_ry and fft_shape; "
            "do not change the physical grid while evaluating this approximation."
        )
    if not structure.atoms:
        raise ValueError("assemble_h0 requires at least one atom")
    missing = {a.species for a in structure.atoms} - set(species_data)
    if missing:
        raise KeyError(f"Missing species inputs for: {sorted(missing)}")
    if strict_reproduction and ecutrho_ry is None:
        raise ValueError(
            "strict_reproduction=True requires explicit ecutrho_ry; "
            "the .orb Energy Cutoff is not the charge-density cutoff."
        )
    if strict_reproduction and fft_shape is None:
        raise ValueError(
            "strict_reproduction=True requires the exact FFT shape used by the "
            "reference H0 calculation."
        )
    local_mode = str(local_integration).lower().replace("-", "_")
    local_mode = {
        "fft": "fft_grid",
        "exact_fft": "fft_grid",
        "cached_fft": "fft_grid_cached",
        "exact_fft_cached": "fft_grid_cached",
        "collocation": "fft_grid_cached",
        "midpoint": "midpoint_interpolated",
        "interpolated": "midpoint_interpolated",
    }.get(local_mode, local_mode)
    if local_mode not in {"fft_grid", "fft_grid_cached", "fft_grid_periodic", "midpoint_interpolated"}:
        raise ValueError(
            "local_integration must be 'fft_grid', 'fft_grid_cached', "
            "or 'midpoint_interpolated'"
        )
    if strict_reproduction and local_mode not in {"fft_grid", "fft_grid_cached", "fft_grid_periodic"}:
        raise ValueError(
            "strict_reproduction=True requires local_integration='fft_grid' or "
            "'fft_grid_cached'; interpolated midpoint "
            "contraction does not reproduce ABACUS's FFT grid."
        )
    if local_cache_max_mb is not None and local_cache_max_mb < 0.0:
        raise ValueError("local_cache_max_mb must be non-negative or None")
    if strict_reproduction and local_cache_key_decimals is not None:
        raise ValueError(
            "strict_reproduction=True requires local_cache_key_decimals=None; "
            "rounded cache identities can merge distinct AO centres."
        )
    if ecutrho_ry is None:
        ecutrho_ry = max(data.orb.ecut_ry for data in species_data.values())
    if projector_grid_step_bohr is None:
        projector_grid_step_bohr = pair_grid_step_bohr

    from .field_inputs import prepare_field_upf
    field_species_data = {s: SpeciesData(d.orb, prepare_field_upf(d.upf, pseudo_rcut_bohr)) for s, d in species_data.items()}
    field = build_periodic_field(
        structure,
        field_species_data,
        ecutrho_ry=float(ecutrho_ry),
        fft_shape=fft_shape,
        xc=xc,
        hartree_g0=hartree_g0,
        local_g0=local_g0,
        include_nlcc=include_nlcc,
        total_electrons=total_electrons,
        radial_g_round_decimals=radial_g_round_decimals,
        pbe_density_threshold=pbe_density_threshold,
        structure_factor_backend=sf_mode,
        structure_factor_eps=structure_factor_eps,
        structure_factor_max_work_mb=structure_factor_max_work_mb,
        structure_factor_nthreads=structure_factor_nthreads,
        field_backend=field_backend, compute_device=compute_device,
        field_max_work_mb=field_max_work_mb,
    )
    spin_z_field=None
    if initial_moments_z is not None:
        from dataclasses import replace
        from .spin_fields import add_collinear_spin_field
        field,spin_z=add_collinear_spin_field(field,structure,field_species_data,initial_moments_z,
            backend=sf_mode,device=compute_device,eps=structure_factor_eps,nthreads=structure_factor_nthreads,
            max_work_mb=structure_factor_max_work_mb,threshold=pbe_density_threshold)
        spin_z_field=replace(field,values_ry=spin_z)
    evaluators = {symbol: OrbitalEvaluator(data.orb) for symbol, data in species_data.items()}
    orbital_cutoffs = np.asarray([evaluators[a.species].basis.rcut for a in structure.atoms])
    max_orbital_cutoff = float(np.max(orbital_cutoffs))
    max_projector_cutoff = max(
        (species_data[a.species].upf.max_projector_cutoff for a in structure.atoms),
        default=0.0,
    ) if include_nonlocal else 0.0
    extra_cutoff = 2 * max_projector_cutoff if support_mode == "nonlocal_complete" else 0.0
    atom_index = None
    if spatial_mode == "indexed":
        atom_index = PeriodicAtomIndex(
            structure, max(2 * (max_orbital_cutoff + max_projector_cutoff), 1e-12)
        )
    local_cache = None
    periodic_cache = None
    spin_z_cache = None
    if local_mode == 'fft_grid_periodic':
        if two_center is None:
            raise ValueError('fft_grid_periodic currently requires the pyabacus two-center backend')
        from .periodic_collocation import PeriodicFFTGridAOCache
        if cuda_precompiled:
            from .cuda_local_grid import CudaPeriodicFFTGridAOCache as PeriodicFFTGridAOCache
        periodic_cache = PeriodicFFTGridAOCache(field,
            max_bytes=None if local_cache_max_mb is None else int(local_cache_max_mb*1024**2),
            chunk_size=local_cache_chunk_size,device=compute_device)
        home_positions=structure.cart_positions
        if spin_z_field is not None:
            spin_z_cache=PeriodicFFTGridAOCache(spin_z_field,
                max_bytes=None if local_cache_max_mb is None else int(local_cache_max_mb*1024**2),
                chunk_size=local_cache_chunk_size,device=compute_device)
    if local_mode == "fft_grid_cached":
        max_bytes = (
            None
            if local_cache_max_mb is None
            else int(float(local_cache_max_mb) * 1024**2)
        )
        local_cache = FFTGridAOCache(
            field,
            chunk_size=int(local_cache_chunk_size),
            max_bytes=max_bytes,
            key_decimals=local_cache_key_decimals,
        )

    h_blocks: dict[BlockKey, np.ndarray] = {}
    s_blocks: dict[BlockKey, np.ndarray] = {}
    components: dict[str, dict[BlockKey, np.ndarray]] = {
        "kinetic": {},
        "local": {},
        "nonlocal": {},
    }

    for i, j, R, ci, cj in iter_pair_images(
        structure, orbital_cutoffs, index=atom_index, extra_cutoff_bohr=extra_cutoff
    ):
        eval_i = evaluators[structure.atoms[i].species]
        eval_j = evaluators[structure.atoms[j].species]
        if two_center is None:
            s, t, v = scalar_pair_integrals(
                eval_i,
                eval_j,
                ci,
                cj,
                field,
                pair_grid_step_bohr,
                chunk_size=int(local_cache_chunk_size),
                local_integration=local_mode,
                local_cache=local_cache,
            )
        else:
            s,t = two_center.scalar_pair(structure.atoms[i].species,structure.atoms[j].species,ci,cj)
            if local_mode == 'fft_grid_periodic':
                v = periodic_cache.contract_pair(eval_i,eval_j,ci,home_positions[j],R)
            elif local_mode == 'fft_grid_cached':
                v = local_pair_integral_fft_grid_cached(eval_i,eval_j,ci,cj,field,cache=local_cache)
            elif local_mode == 'fft_grid':
                v = local_pair_integral_fft_grid(eval_i,eval_j,ci,cj,field)
            else:
                raise ValueError('pyabacus backend requires FFT local integration')
        key = BlockKey(i, j, R)
        s_spin = _spin_duplicate(s) if nspin == 4 else s
        t_spin = _spin_duplicate(t) if nspin == 4 else t
        v_spin = _spin_duplicate(v) if nspin == 4 else v
        if spin_z_cache is not None:
            vz=spin_z_cache.contract_pair(eval_i,eval_j,ci,home_positions[j],R)
            ni,nj=v.shape
            v_spin[:ni,:nj]+=vz
            v_spin[ni:,nj:]-=vz
        if include_nonlocal:
            candidates = None
            if atom_index is not None:
                # Query a conservative midpoint ball once. There is no
                # per-pair Structure.cart_positions/all-atom traversal here.
                radius = (
                    max_projector_cutoff + max(eval_i.basis.rcut, eval_j.basis.rcut)
                    + 0.5 * np.linalg.norm(ci - cj)
                )
                candidates = (
                    (hit.atom_index, hit.center_bohr)
                    for hit in atom_index.query(0.5 * (ci + cj), radius)
                )
            if two_center is not None:
                if candidates is None:
                    from .nonlocal_kb import _reference_projector_centers
                    candidates = _reference_projector_centers(structure,species_data,ci,cj,max(eval_i.basis.rcut,eval_j.basis.rcut))
                vnl = two_center.nonlocal_block(structure,structure.atoms[i].species,structure.atoms[j].species,ci,cj,candidates)
            else:
                vnl = nonlocal_pair_block(
                    structure,
                    species_data,
                    eval_i,
                    eval_j,
                    ci,
                    cj,
                    projector_grid_step_bohr,
                    _candidate_centers=candidates,
                    nspin=nspin,
                )
        else:
            vnl = np.zeros_like(t_spin)
        h = t_spin + v_spin + vnl
        if prune_frobenius_ry > 0.0 and np.linalg.norm(h) < prune_frobenius_ry:
            continue
        s_blocks[key] = s_spin
        h_blocks[key] = h
        components["kinetic"][key] = t_spin
        components["local"][key] = v_spin
        components["nonlocal"][key] = vnl

    pre_repair = {
        "overlap": hermiticity_report(s_blocks),
        "hamiltonian": hermiticity_report(h_blocks),
        "components": {name: hermiticity_report(value) for name, value in components.items()},
    }
    if hermitize:
        _hermitize(s_blocks)
        _hermitize(h_blocks)
        for comp in components.values():
            _hermitize(comp)

    counts = [evaluators[a.species].norb for a in structure.atoms]
    result = AssemblyResult(
        h_blocks_ry=h_blocks,
        s_blocks=s_blocks,
        components_ry=components,
        field=field,
        orbital_counts=counts,
        metadata={
            "energy_unit_internal": "Ry",
            "length_unit_internal": "bohr",
            "nspin": nspin,
            "spin_z_collocation_cache": spin_z_cache.stats() if spin_z_cache is not None else None,
            "two_center_backend": 'cuda_precompiled' if cuda_precompiled else two_center_backend,
            "cuda_precompiled": bool(cuda_precompiled),
            "two_center_metadata": ({**two_center.metadata,**getattr(two_center, 'stats', getattr(two_center, 'timing', {}))} if two_center is not None else None),
            "spin_order": ("spin-block-major-per-atom:[up_spatial,down_spatial]" if nspin == 4 else "scalar-spatial-per-atom"),
            "scalar_reductions": {s:d.upf.metadata.get('scalar_reduction') for s,d in species_data.items()},
            "real_harmonic_order": "ABACUS: m=0,+1,-1,+2,-2,...",
            "strict_reproduction": bool(strict_reproduction),
            "spatial_backend": spatial_mode,
            "spatial_index_stats": atom_index.stats() if atom_index is not None else None,
            "pair_support": support_mode,
            "pair_extra_cutoff_bohr": float(extra_cutoff),
            "hermitized_output": bool(hermitize),
            "pre_hermitization_residual": pre_repair,
            "structure_fingerprint": structure_fingerprint(
                structure.cell_bohr,
                [atom.species for atom in structure.atoms],
                np.asarray([atom.frac for atom in structure.atoms], dtype=float),
            ),
            "structure": {
                "cell_bohr": np.asarray(structure.cell_bohr, dtype=float).tolist(),
                "species": [atom.species for atom in structure.atoms],
                "fractional_coordinates": np.asarray(
                    [atom.frac for atom in structure.atoms], dtype=float
                ).tolist(),
            },
            "input_files": input_file_manifest(original_inputs),
            "pair_grid_step_bohr": float(pair_grid_step_bohr),
            "local_integration": local_mode,
            "local_collocation_cache": (
                periodic_cache.stats() if periodic_cache is not None else (local_cache.stats() if local_cache is not None else None)
            ),
            "projector_grid_step_bohr": float(projector_grid_step_bohr),
            "include_nonlocal": include_nonlocal,
            "include_nlcc": include_nlcc,
            "local_terms_assembled_together": True,
            **field.metadata,
        },
    )
    if output_atom_cell_shifts is not None:
        from .cell_gauge import rebase_result
        result = rebase_result(result, output_atom_cell_shifts)
    return result
