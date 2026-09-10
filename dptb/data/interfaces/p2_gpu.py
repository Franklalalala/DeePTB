"""Device-resident evaluation of the qualified P2/P23 radial tables.

Table compilation is a one-time CPU operation. Forward uses only torch tensors:
no SciPy, NumPy, file reads, tensor-to-host copies, or per-displacement loops.
It preserves the source cubic spline (including its boundary conditions), rather
than refitting a different interpolant. Units and the ABACUS real harmonic gauge
are inherited unchanged from :class:`RadialBlockTable`.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.linalg import qr
from torch import nn

from .p2_table import RadialBlockTable


def _harmonics(l: int, vectors: torch.Tensor) -> torch.Tensor:
    """Real Y_lm in ABACUS order, without angular pole singularities."""
    xyz = vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True).clamp_min(1e-30)
    x, y, z = xyz.unbind(-1)
    real, imag = torch.ones_like(x), torch.zeros_like(x)
    result = []
    for m in range(l + 1):
        if m:
            real, imag = real * x - imag * y, imag * x + real * y
        q = torch.ones_like(z) * ((-1) ** m * math.prod(range(1, 2 * m, 2)))
        if l > m:
            prev, q = q, (2 * m + 1) * z * q
            for degree in range(m + 2, l + 1):
                prev, q = q, ((2 * degree - 1) * z * q - (degree + m - 1) * prev) / (degree - m)
        norm = math.sqrt((2 * l + 1) / (4 * math.pi) * math.factorial(l - m) / math.factorial(l + m))
        if m == 0:
            result.append(norm * q)
        else:
            result.extend((math.sqrt(2) * norm * q * real, math.sqrt(2) * norm * q * imag))
    return torch.stack(result, dim=-1)


def _rotation_z_to(vectors: torch.Tensor) -> torch.Tensor:
    radius = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    n = vectors / radius.clamp_min(1e-30)
    x, y, z = n.unbind(-1)
    zero = torch.zeros_like(z)
    skew = torch.stack((zero, zero, x, zero, zero, y, -x, -y, zero), -1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=vectors.device, dtype=vectors.dtype)
    # Avoid cancellation of 1+z near the south pole, especially in float32.
    factor = torch.where(z < 0, (1 - z) / (x*x + y*y).clamp_min(1e-30),
                         1 / (1 + z).clamp_min(1e-30))
    rotation = eye + skew + (skew @ skew) * factor[:, None, None]
    south = torch.diag(vectors.new_tensor([1., -1., -1.]))
    rotation = torch.where((z <= -1 + 1e-14)[:, None, None], south, rotation)
    return torch.where(((z >= 1 - 1e-14) | (radius[:, 0] <= 1e-14))[:, None, None], eye, rotation)


class TorchRadialBlockTable(nn.Module):
    """Compile one immutable radial table into serializable device buffers.

    ``forward(displacements_bohr)`` returns ``[queries, left_AO, right_AO]``.
    Construct in float64 for oracle comparison; opt into float32 only after
    measuring the error on the actual table and geometry distribution. Caller
    input must be finite, floating point, and on the module's device/dtype.
    Empty batches, zero displacements, nonuniform knots and exact support
    boundaries are supported. Derivatives away from cutoffs are autograd-ready;
    this class does not promise smoothness across a source table's hard cutoff.
    """

    def __init__(self, table: RadialBlockTable, *, device=None, dtype=torch.float64):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('radial evaluation requires float32 or float64')
        if not np.isfinite(table.distances).all() or not np.isfinite(table.support_bohr):
            raise ValueError('radial knots and support must be finite')
        if np.iscomplexobj(table.values):
            raise ValueError('only real non-SOC radial tables are supported')
        self.left_shells = table.left_shells
        self.right_shells = table.right_shells
        self.support_bohr = float(table.support_bohr)
        self.shape = table.values.shape[1:]
        self.angular_degrees = tuple(sorted(set(self.left_shells + self.right_shells)))

        def buffer(name, array):
            self.register_buffer(name, torch.as_tensor(np.array(array, copy=True), device=device, dtype=dtype))

        buffer('knots', table.distances)
        if table._spline is not None:
            coefficients = table._spline.c
        else:
            coefficients = np.zeros((4, len(table.distances) - 1, *self.shape))
            coefficients[2] = np.diff(table.values, axis=0) / np.diff(table.distances)[:, None, None]
            coefficients[3] = table.values[:-1]
        # Zero AO entries are structural zeros of the canonical SK block. Avoid
        # storing/reading four dense AO matrices for each spline interval.
        coefficients = coefficients.reshape(4, len(table.distances) - 1, -1)
        active = np.flatnonzero(np.any(coefficients != 0, axis=(0, 1)))
        self.register_buffer('active_columns', torch.as_tensor(active, dtype=torch.long, device=device))
        buffer('coefficients', coefficients[:, :, active].transpose(1, 0, 2))
        for l in self.angular_degrees:
            base = table._rotator._base[l]
            # Select a well-conditioned square collocation grid once. Harmonic
            # rotations are recovered algebraically; no per-query SVD is needed.
            _, _, pivots = qr(base.T, pivoting=True)
            rows = pivots[:2 * l + 1]
            buffer(f'directions_{l}', table._rotator.directions[rows])
            buffer(f'inverse_{l}', np.linalg.inv(base[rows]))

    def _check(self, vectors):
        if vectors.ndim != 2 or vectors.shape[-1] != 3:
            raise ValueError('displacements must have shape [queries,3]')
        if vectors.dtype != self.knots.dtype or vectors.device != self.knots.device:
            raise ValueError('displacements must match radial table device and dtype')

    def canonical(self, distances: torch.Tensor) -> torch.Tensor:
        """Evaluate source spline with Horner's rule and exact support masking."""
        index = torch.searchsorted(self.knots, distances.contiguous(), right=True)
        index = index.clamp(1, self.knots.numel() - 1) - 1
        delta = (distances - self.knots[index])[:, None]
        c = self.coefficients[index]
        values = ((c[:, 0] * delta + c[:, 1]) * delta + c[:, 2]) * delta + c[:, 3]
        values = torch.where((distances < self.support_bohr - 1e-12)[:, None], values, 0.)
        dense = values.new_zeros((distances.shape[0], math.prod(self.shape)))
        return dense.index_copy(1, self.active_columns, values).reshape(-1, *self.shape)

    def forward(self, displacements_bohr: torch.Tensor) -> torch.Tensor:
        self._check(displacements_bohr)
        if displacements_bohr.shape[0] == 0:
            return displacements_bohr.new_empty((0, *self.shape))
        distances = torch.linalg.vector_norm(displacements_bohr, dim=-1)
        canonical = self.canonical(distances)
        cartesian = _rotation_z_to(displacements_bohr)
        rotations = {}
        for l in self.angular_degrees:
            directions = getattr(self, f'directions_{l}') @ cartesian.transpose(-1, -2)
            values = _harmonics(l, directions)
            rotations[l] = (getattr(self, f'inverse_{l}') @ values).transpose(-1, -2)
        output_rows = []
        left_offset = 0
        for left_l in self.left_shells:
            left_end = left_offset + 2 * left_l + 1
            blocks = []
            right_offset = 0
            for right_l in self.right_shells:
                right_end = right_offset + 2 * right_l + 1
                block = canonical[:, left_offset:left_end, right_offset:right_end]
                blocks.append(rotations[left_l] @ block @ rotations[right_l].transpose(-1, -2))
                right_offset = right_end
            output_rows.append(torch.cat(blocks, dim=-1))
            left_offset = left_end
        rotated = torch.cat(output_rows, dim=-2)
        return torch.where((distances <= 1e-14)[:, None, None], canonical, rotated)


__all__ = ['TorchRadialBlockTable']
