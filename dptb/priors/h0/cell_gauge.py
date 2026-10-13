"""Exact change of atom home cells for periodic AO matrices.

If r'_i = r_i + q_i @ cell, the same block has R' = R + q_i - q_j.
This changes labels, not matrix elements, energies, or a physical approximation.
"""
from dataclasses import replace
import numpy as np
from .models import BlockKey
from .provenance import structure_fingerprint


def validate_cell_shifts(shifts, natoms):
    values = np.asarray(shifts)
    if values.shape != (natoms, 3) or not np.isfinite(values).all():
        raise ValueError('atom cell shifts must be finite [natoms,3] integers')
    if not np.equal(values, np.rint(values)).all() or np.max(np.abs(values), initial=0) >= 2**52:
        raise ValueError('atom cell shifts must be exact, bounded integers')
    return values.astype(np.int64)


def shifts_between_coordinates(cell, input_frac, output_cart_bohr, *, tolerance_bohr=1e-7):
    """Infer only integer lattice shifts; reject real displacement or reordering.

    Output positions must retain atom order. Never infer shifts from H or S.
    Explicit positions avoid ambiguous modulo at floating point cell boundaries.
    """
    cell = np.asarray(cell, dtype=float)
    frac = np.asarray(input_frac, dtype=float)
    output = np.asarray(output_cart_bohr, dtype=float)
    if frac.ndim != 2 or frac.shape[1] != 3 or output.shape != frac.shape:
        raise ValueError('coordinate arrays must have matching [natoms,3] shape')
    if not np.isfinite(output).all() or not np.isfinite(frac).all():
        raise ValueError('coordinates must be finite')
    delta = output - frac @ cell
    q = np.rint(delta @ np.linalg.inv(cell))
    if np.max(np.abs(delta - q @ cell), initial=0) > tolerance_bohr:
        raise ValueError('output atom positions are not integer-cell equivalents in the same order')
    return validate_cell_shifts(q, len(frac))


def rebase_blocks(blocks, shifts):
    out = {}
    for key, value in blocks.items():
        r = np.asarray(key.R) + shifts[key.i] - shifts[key.j]
        newkey = BlockKey(key.i, key.j, tuple(int(x) for x in r))
        if newkey in out:
            raise ValueError('duplicate key during atom-cell rebasing')
        out[newkey] = value
    return out


def rebase_result(result, atom_cell_shifts):
    """Return H0/S/components and geometry metadata in a common cell gauge.

    The original result remains unchanged. Values are reused without rounding;
    the periodic scalar field is unchanged by integer cell shifts.
    """
    geometry = result.metadata['structure']
    frac = np.asarray(geometry['fractional_coordinates'], dtype=float)
    shifts = validate_cell_shifts(atom_cell_shifts, len(frac))
    output_frac = frac + shifts
    meta = dict(result.metadata)
    meta['structure'] = {**geometry, 'fractional_coordinates': output_frac.tolist()}
    meta['structure_fingerprint'] = structure_fingerprint(geometry['cell_bohr'], geometry['species'], output_frac)
    meta['cell_gauge'] = {'schema': 'h0.atom-cell-gauge/v1', 'input_fractional_coordinates': frac.tolist(),
        'output_minus_input_integer_shifts': shifts.tolist(), 'block_rule': 'R_out = R_in + shift_i - shift_j',
        'matrix_values_unchanged': True, 'periodic_field_unchanged': True}
    h = rebase_blocks(result.h_blocks_ry, shifts)
    s = rebase_blocks(result.s_blocks, shifts)
    components = {k: rebase_blocks(v, shifts) for k, v in result.components_ry.items()}
    # A diagnostic's missing-counterpart key list also belongs to a cell gauge.
    from .assemble import hermiticity_report
    meta['output_hermiticity'] = {'hamiltonian': hermiticity_report(h), 'overlap': hermiticity_report(s)}
    return replace(result, h_blocks_ry=h, s_blocks=s, components_ry=components, metadata=meta)
