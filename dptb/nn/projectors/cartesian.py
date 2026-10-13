"""Small Cartesian projector utilities for late DeePTB output heads.

The Cartesian basis used here is the symmetric monomial basis of rank ``l``.
Its irreducible (harmonic/STF) subspace is obtained from e3nn real spherical
harmonics by a deterministic change of basis.  Cartesian-3j tensors are then
constructed by transforming the e3nn Wigner-3j intertwiner into that basis.

No learnable parameter lives in this file.  The generated matrices are fixed
intertwiners and are registered as non-persistent buffers by the output heads.
"""

from __future__ import annotations

import functools
import math
import re
from typing import List, Sequence, Tuple, Union

import torch
from e3nn import o3


_ANGULAR_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}


def shell_l(shell: str) -> int:
    """Return angular momentum from an OrbitalMapper shell label."""
    labels = re.findall(r"[A-Za-z]", str(shell))
    if len(labels) != 1 or labels[0].lower() not in _ANGULAR_L:
        raise ValueError("Unsupported AO shell label {!r}.".format(shell))
    return _ANGULAR_L[labels[0].lower()]


def ao_shell_layout(full_basis: Sequence[str]) -> Tuple[Tuple[int, int, int], ...]:
    """Return ``(start, stop, l)`` entries in OrbitalMapper full-basis order."""
    layout: List[Tuple[int, int, int]] = []
    start = 0
    for shell in full_basis:
        ell = shell_l(str(shell))
        stop = start + 2 * ell + 1
        layout.append((start, stop, ell))
        start = stop
    return tuple(layout)


def _symmetric_exponents(rank: int) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(
        (nx, ny, rank - nx - ny)
        for nx in range(rank + 1)
        for ny in range(rank - nx + 1)
    )


def _symmetric_monomials(points: torch.Tensor, rank: int) -> torch.Tensor:
    """Evaluate an orthonormalized symmetric Cartesian monomial basis."""
    columns = []
    rank_factorial = math.factorial(rank)
    for nx, ny, nz in _symmetric_exponents(rank):
        multiplicity = rank_factorial / (
            math.factorial(nx) * math.factorial(ny) * math.factorial(nz)
        )
        columns.append(
            math.sqrt(multiplicity)
            * points[:, 0].pow(nx)
            * points[:, 1].pow(ny)
            * points[:, 2].pow(nz)
        )
    return torch.stack(columns, dim=-1)


@functools.lru_cache(maxsize=None)
def _cartesian_basis_cpu(rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build deterministic spherical <-> irreducible Cartesian maps in fp64."""
    if rank < 0:
        raise ValueError("rank must be non-negative, got {}.".format(rank))

    n_cart = (rank + 1) * (rank + 2) // 2
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0xC0A7E51 + rank)
    points = torch.randn(
        max(6 * n_cart, 64),
        3,
        generator=generator,
        dtype=torch.float64,
        device="cpu",
    )
    monomials = _symmetric_monomials(points, rank)
    spherical = o3.spherical_harmonics(
        rank,
        points,
        normalize=False,
        normalization="component",
    )
    to_cartesian = torch.linalg.lstsq(monomials, spherical).solution.contiguous()
    from_cartesian = torch.linalg.pinv(
        to_cartesian,
        rtol=1.0e-12,
        atol=1.0e-14,
    ).contiguous()

    relative_residual = torch.linalg.vector_norm(
        monomials.matmul(to_cartesian) - spherical
    ) / torch.linalg.vector_norm(spherical).clamp_min(torch.finfo(torch.float64).tiny)
    roundtrip = from_cartesian.matmul(to_cartesian)
    identity = torch.eye(2 * rank + 1, dtype=torch.float64)
    if float(relative_residual) > 5.0e-11 or not torch.allclose(
        roundtrip, identity, atol=5.0e-11, rtol=5.0e-11
    ):
        raise RuntimeError(
            "Failed to construct a stable Cartesian irrep basis for l={}: "
            "residual={:.3e}, roundtrip={:.3e}.".format(
                rank,
                float(relative_residual),
                float((roundtrip - identity).abs().max()),
            )
        )
    return to_cartesian, from_cartesian


def cartesian_irrep_basis(
    rank: int,
    dtype: torch.dtype,
    device: Union[str, torch.device],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``spherical -> Cartesian`` and its left inverse."""
    to_cartesian, from_cartesian = _cartesian_basis_cpu(int(rank))
    return (
        to_cartesian.to(dtype=dtype, device=device),
        from_cartesian.to(dtype=dtype, device=device),
    )


class CartesianShellPairCoupling(torch.nn.Module):
    """Fixed ICT projector from one irrep to an ordered AO shell-pair block.

    ``forward`` uses a compact projector compiled from the explicit Cartesian
    path.  ``forward_explicit`` retains the full spherical -> ICT ->
    Cartesian-3j -> AO-spherical sequence for convention and equivariance
    tests.  Both paths are numerically identical.
    """

    def __init__(
        self,
        l_row: int,
        l_col: int,
        l_in: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.l_row = int(l_row)
        self.l_col = int(l_col)
        self.l_in = int(l_in)
        if not abs(self.l_row - self.l_col) <= self.l_in <= self.l_row + self.l_col:
            raise ValueError(
                "Invalid angular path ({}, {}) -> {}.".format(
                    self.l_row, self.l_col, self.l_in
                )
            )

        row_to_cart, row_from_cart = cartesian_irrep_basis(
            self.l_row, dtype=dtype, device=device
        )
        col_to_cart, col_from_cart = cartesian_irrep_basis(
            self.l_col, dtype=dtype, device=device
        )
        in_to_cart, in_from_cart = cartesian_irrep_basis(
            self.l_in, dtype=dtype, device=device
        )

        # Match DeePTB E3Hamiltonian's real-CG normalization convention.
        spherical_3j = o3.wigner_3j(
            self.l_row,
            self.l_col,
            self.l_in,
            dtype=dtype,
            device=device,
        ) * math.sqrt(2 * self.l_in + 1)

        cartesian_3j = torch.einsum(
            "am,bn,mnk,kc->abc",
            row_to_cart,
            col_to_cart,
            spherical_3j,
            in_from_cart,
        ).contiguous()
        compiled = torch.einsum(
            "ma,nb,abc,ck->mnk",
            row_from_cart,
            col_from_cart,
            cartesian_3j,
            in_to_cart,
        ).contiguous()

        self.register_buffer("row_to_cart", row_to_cart, persistent=False)
        self.register_buffer("row_from_cart", row_from_cart, persistent=False)
        self.register_buffer("col_to_cart", col_to_cart, persistent=False)
        self.register_buffer("col_from_cart", col_from_cart, persistent=False)
        self.register_buffer("in_to_cart", in_to_cart, persistent=False)
        self.register_buffer("cartesian_3j", cartesian_3j, persistent=False)
        self.register_buffer("compiled_projector", compiled, persistent=False)

    @property
    def row_dim(self) -> int:
        return 2 * self.l_row + 1

    @property
    def col_dim(self) -> int:
        return 2 * self.l_col + 1

    @property
    def in_dim(self) -> int:
        return 2 * self.l_in + 1

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the precompiled Cartesian projector."""
        if features.shape[-1] != self.in_dim:
            raise ValueError(
                "Expected input dim {}, got {}.".format(
                    self.in_dim, features.shape[-1]
                )
            )
        leading = features.shape[:-1]
        flat = features.reshape(-1, self.in_dim)
        out = torch.einsum("ijk,nk->nij", self.compiled_projector, flat)
        return out.reshape(*leading, self.row_dim, self.col_dim)

    def forward_explicit(self, features: torch.Tensor) -> torch.Tensor:
        """Apply the explicit irreducible Cartesian tensor route."""
        if features.shape[-1] != self.in_dim:
            raise ValueError(
                "Expected input dim {}, got {}.".format(
                    self.in_dim, features.shape[-1]
                )
            )
        leading = features.shape[:-1]
        flat = features.reshape(-1, self.in_dim)
        cart_in = torch.einsum("ck,nk->nc", self.in_to_cart, flat)
        cart_pair = torch.einsum("abc,nc->nab", self.cartesian_3j, cart_in)
        out = torch.einsum(
            "ia,nab,jb->nij",
            self.row_from_cart,
            cart_pair,
            self.col_from_cart,
        )
        return out.reshape(*leading, self.row_dim, self.col_dim)


class CartesianIrrepProduct(torch.nn.Module):
    """Fixed Cartesian-3j product for two irreps with shared leading axes."""

    def __init__(
        self,
        l_in1: int,
        l_in2: int,
        l_out: int,
        *,
        dtype: torch.dtype = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.l_in1 = int(l_in1)
        self.l_in2 = int(l_in2)
        self.l_out = int(l_out)
        if not abs(self.l_in1 - self.l_in2) <= self.l_out <= self.l_in1 + self.l_in2:
            raise ValueError(
                "Invalid angular path ({}, {}) -> {}.".format(
                    self.l_in1, self.l_in2, self.l_out
                )
            )

        in1_to_cart, in1_from_cart = cartesian_irrep_basis(
            self.l_in1, dtype=dtype, device=device
        )
        in2_to_cart, in2_from_cart = cartesian_irrep_basis(
            self.l_in2, dtype=dtype, device=device
        )
        out_to_cart, out_from_cart = cartesian_irrep_basis(
            self.l_out, dtype=dtype, device=device
        )
        spherical_3j = o3.wigner_3j(
            self.l_in1,
            self.l_in2,
            self.l_out,
            dtype=dtype,
            device=device,
        ) * math.sqrt(2 * self.l_out + 1)
        cartesian_3j = torch.einsum(
            "ck,ijk,ia,jb->cab",
            out_to_cart,
            spherical_3j,
            in1_from_cart,
            in2_from_cart,
        ).contiguous()
        compiled = torch.einsum(
            "kc,cab,ai,bj->ijk",
            out_from_cart,
            cartesian_3j,
            in1_to_cart,
            in2_to_cart,
        ).contiguous()

        self.register_buffer("in1_to_cart", in1_to_cart, persistent=False)
        self.register_buffer("in2_to_cart", in2_to_cart, persistent=False)
        self.register_buffer("out_from_cart", out_from_cart, persistent=False)
        self.register_buffer("cartesian_3j", cartesian_3j, persistent=False)
        self.register_buffer("compiled_projector", compiled, persistent=False)

    @property
    def in1_dim(self) -> int:
        return 2 * self.l_in1 + 1

    @property
    def in2_dim(self) -> int:
        return 2 * self.l_in2 + 1

    @property
    def out_dim(self) -> int:
        return 2 * self.l_out + 1

    def _validate_inputs(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if (
            x.shape[:-1] != y.shape[:-1]
            or x.shape[-1] != self.in1_dim
            or y.shape[-1] != self.in2_dim
        ):
            raise ValueError(
                "CartesianIrrepProduct expected matching leading shapes and "
                "last dims ({}, {}), got {} and {}.".format(
                    self.in1_dim, self.in2_dim, tuple(x.shape), tuple(y.shape)
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(x, y)
        return torch.einsum("...i,...j,ijk->...k", x, y, self.compiled_projector)

    def forward_explicit(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(x, y)
        x_cart = torch.einsum("ai,...i->...a", self.in1_to_cart, x)
        y_cart = torch.einsum("bj,...j->...b", self.in2_to_cart, y)
        out_cart = torch.einsum(
            "cab,...a,...b->...c", self.cartesian_3j, x_cart, y_cart
        )
        return torch.einsum("kc,...c->...k", self.out_from_cart, out_cart)
