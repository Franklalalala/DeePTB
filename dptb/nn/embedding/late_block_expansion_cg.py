"""QHFlow-style late Expansion decoder that writes padded AO blocks directly.

This is the only production output path in this patch that performs explicit
angular coupling. The coupling maps hidden irreps straight into AO shell-pair
blocks; no OrbitalMapper reduced-matrix-element vector is created or consumed.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from typing import Optional, Sequence, Union

import torch
from e3nn import o3


_ANGULAR_L = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4, "h": 5}

# Forward-dispatch switches (design A.7 / A.8; superseded by project decision
# 2026-07-22 P1-3: COMPATIBILITY-DEFAULT).  The DEFAULT forward path is the
# reference per-path loop (``_forward_legacy``): fixed Python-order accumulation,
# bit-stable with every pre-fusion (0715-refactor) checkpoint/config.  The fused
# grouped-einsum implementation (``_forward_fused``) is an opt-in performance path
# selected per module instance via the ``cg_head_impl`` constructor argument (the
# config field of the same name on the ``late_block_expansion_cg`` / h_b0 output
# route); it is never chosen implicitly.
#
# History: 2026-07-20 shipped fused as the silent default -- no config flag, no
# checkpoint metadata, rollback only via a process env var.  2026-07-22 (P1-3)
# reverted the default to legacy because that silent flip changed the numerical
# path under UNCHANGED config/checkpoints: an old (0715-refactor) checkpoint
# resumed on this module took a different floating-point route with no way to
# detect it from the checkpoint or config alone.  Reaching the old numerics now
# requires nothing (it IS the default again); reaching fused requires an
# explicit, config-recorded opt-in.
#
# The accepted trade-off when fused IS selected: it REASSOCIATES the
# floating-point reduction (grouped batched einsums + one scatter instead of the
# per-path accumulation), and fp addition is non-associative, so fused output
# equals the legacy order only up to reassociation -- NOT elementwise-tight.
# Absent cancellation the drift is pure roundoff (fp64 <= ~4e-16, fp32 <= ~2e-7).
# Under fp32 CANCELLATION with independent per-tensor scales the two paths evaluate
# the SAME real expression under different reduction orders, so there is NO
# guaranteed elementwise accuracy ordering between them: the relative drift can
# exceed any tight rtol, and EITHER path may sit further from the fp64 truth than
# the other (a constructed seed-1135-style cancellation case puts the fused output
# ~2-3 orders of magnitude further from truth than legacy -- see the cancellation
# characterization test).  What holds is the weaker REASSOCIATION property: in the
# tested non-cancellation regime both paths stay within the certified tolerance of
# each other and of the fp64 truth.  A run that opts in to fused accepts this
# reassociation-level training-trajectory divergence -- the same class of
# nondeterminism as cudnn autotune -- in exchange for a substantial head speedup
# (external benchmark, NOT reproduced by the in-tree test suite: ~4-5x on the head,
# ~-3.2% wall-clock per training iteration).  Fresh (non-resumed) runs that do not
# need bit-continuity with pre-fusion history are the intended fused audience.
#
# The certified domain is fp32/fp64 EAGER only -- autocast, low-precision input
# dtype, and deterministic mode always fall back to legacy regardless of the
# ``cg_head_impl`` selection or any env switch (see ``forward``).
#
# Dispatch precedence inside ``forward``, highest first:
#   (a) ``DPTB_LEGACY_CG_HEAD=1`` -- absolute manual override; forces the
#       reference loop (emergency rollback).  Only the literal string ``"1"``
#       activates it (unset / ``"0"`` / ``"true"`` / ``""`` are ignored).  This is
#       the ONLY environment switch ``forward`` reads.
#   (b) certified-domain guard -- forces legacy under autocast, a non-fp32/64
#       input dtype, or ``use_deterministic_algorithms(True)``, regardless of (c).
#   (c) ``self.cg_head_impl == "fused"`` (constructor / config opt-in) -- fused.
#   (d) DEFAULT -- legacy.
#
# ``DPTB_FUSED_CG_HEAD`` is accepted (does not error) for compatibility with
# configs/scripts written against the interim env-var-opt-in scheme, but it is NOT
# read and has NO runtime effect: the opt-in mechanism is the ``cg_head_impl``
# config field, not an environment variable.  ``_FUSED_CG_HEAD_ENV`` is retained
# only to document that reserved name.
_LEGACY_CG_HEAD_ENV = "DPTB_LEGACY_CG_HEAD"
_FUSED_CG_HEAD_ENV = "DPTB_FUSED_CG_HEAD"

# ``cg_head_impl`` config/constructor choices (P1-3).  ``LEGACY`` is the default in
# both this module's constructor and the ``argcheck`` schema, so an omitted field
# reproduces the pre-fusion (0715-refactor) numerics unconditionally.
_CG_HEAD_IMPL_LEGACY = "legacy"
_CG_HEAD_IMPL_FUSED = "fused"
_CG_HEAD_IMPL_CHOICES = (_CG_HEAD_IMPL_LEGACY, _CG_HEAD_IMPL_FUSED)


def _shell_l(shell: str) -> int:
    labels = re.findall(r"[A-Za-z]", str(shell))
    if len(labels) != 1 or labels[0].lower() not in _ANGULAR_L:
        raise ValueError(f"Unsupported AO shell label {shell!r}.")
    return _ANGULAR_L[labels[0].lower()]


def _cpu_autocast_enabled() -> bool:
    """CPU-autocast state, robust across torch versions.

    ``torch.is_autocast_cpu_enabled`` is the historical predicate (deprecated on
    current torch in favour of the device-agnostic ``is_autocast_enabled('cpu')``
    and slated for removal); prefer it when present but silence its
    ``DeprecationWarning`` so this per-forward guard never spams production logs,
    and fall back to the device-agnostic form -- guarded because very old builds
    reject the positional device argument with ``TypeError``.
    """
    cpu_check = getattr(torch, "is_autocast_cpu_enabled", None)
    if cpu_check is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return bool(cpu_check())
    try:
        return bool(torch.is_autocast_enabled("cpu"))
    except TypeError:
        return False


def _autocast_is_active() -> bool:
    """True when any autocast context is active.

    ``torch.is_autocast_enabled()`` (no argument) reports the CUDA / new
    device-agnostic default flag; CPU autocast is a separate flag handled by
    ``_cpu_autocast_enabled``.  Either being set means the caller is inside a
    low-precision autocast region.
    """
    return bool(torch.is_autocast_enabled()) or _cpu_autocast_enabled()


class LateBlockExpansionCGHead(torch.nn.Module):
    """Decode hidden irreps directly into ``[max_norb, max_norb]`` AO blocks."""

    performs_angular_coupling = True
    output_contract = "ao_block"
    bypasses_rme = True
    bypasses_e3hamiltonian = True
    uses_ict = False

    def __init__(
        self,
        irreps_in: Union[str, o3.Irreps],
        full_basis: Sequence[str],
        *,
        symmetrize: bool,
        rank: int = 16,
        init: float = 0.0,
        condition: str = "scalar_0e",
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        cg_head_impl: str = _CG_HEAD_IMPL_LEGACY,
    ) -> None:
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.full_basis = tuple(str(shell) for shell in full_basis)
        self.symmetrize = bool(symmetrize)
        self.rank = int(rank)
        self.dynamic_init = float(init)
        self.condition = str(condition).strip().lower()
        self.cg_head_impl = str(cg_head_impl).strip().lower()

        if not self.full_basis:
            raise ValueError("full_basis must contain at least one AO shell.")
        if self.rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}.")
        if self.dynamic_init < 0.0:
            raise ValueError(f"init must be non-negative, got {init}.")
        if self.condition != "scalar_0e":
            raise ValueError(
                "late_block_expansion_cg supports only condition='scalar_0e', "
                f"got {condition!r}."
            )
        if self.cg_head_impl not in _CG_HEAD_IMPL_CHOICES:
            raise ValueError(
                f"cg_head_impl must be one of {_CG_HEAD_IMPL_CHOICES}, got "
                f"{cg_head_impl!r}."
            )

        factory_kwargs = {}
        if dtype is not None:
            factory_kwargs["dtype"] = dtype
        if device is not None:
            factory_kwargs["device"] = device

        self._shell_specs = []
        offset = 0
        ao_terms = []
        for shell in self.full_basis:
            ell = _shell_l(shell)
            dim = 2 * ell + 1
            self._shell_specs.append((offset, offset + dim, ell))
            ao_terms.append((1, (ell, (-1) ** ell)))
            offset += dim
        self.max_norb = offset
        self.output_shape = (self.max_norb, self.max_norb)
        self.ao_irreps = o3.Irreps(ao_terms)

        scalar_indices = []
        for term_slice, (_, ir) in zip(self.irreps_in.slices(), self.irreps_in):
            if ir.l == 0 and ir.p == 1:
                scalar_indices.extend(range(term_slice.start, term_slice.stop))
        if not scalar_indices:
            raise ValueError(
                "late_block_expansion_cg requires at least one 0e input "
                f"multiplicity; irreps_in={self.irreps_in}."
            )
        self.register_buffer(
            "_scalar_indices",
            torch.tensor(scalar_indices, dtype=torch.long, device=device),
            persistent=False,
        )

        self._paths = []
        weight_offset = 0
        static_weights = []
        for input_index, (mul_in, ir_in) in enumerate(self.irreps_in):
            for row_index, (_, _, row_l) in enumerate(self._shell_specs):
                row_ir = o3.Irrep(row_l, (-1) ** row_l)
                for col_index, (_, _, col_l) in enumerate(self._shell_specs):
                    col_ir = o3.Irrep(col_l, (-1) ** col_l)
                    if ir_in not in row_ir * col_ir:
                        continue

                    buffer_name = f"_path_coefficient_{len(self._paths)}"
                    coefficient = o3.wigner_3j(
                        row_l, col_l, ir_in.l, dtype=dtype, device=device
                    ) * math.sqrt(2 * ir_in.l + 1)
                    self.register_buffer(buffer_name, coefficient, persistent=False)

                    next_offset = weight_offset + mul_in
                    self._paths.append(
                        (
                            input_index,
                            row_index,
                            col_index,
                            weight_offset,
                            next_offset,
                            buffer_name,
                        )
                    )
                    weight = torch.empty(mul_in, **factory_kwargs)
                    torch.nn.init.normal_(
                        weight, mean=0.0, std=1.0 / math.sqrt(mul_in)
                    )
                    static_weights.append(weight)
                    weight_offset = next_offset

        if not self._paths:
            raise ValueError(
                "No compatible hidden-irrep to AO-shell expansion paths were "
                f"found for irreps_in={self.irreps_in}, full_basis={self.full_basis}."
            )

        self.num_path_weights = weight_offset
        self.static_weights = torch.nn.Parameter(torch.cat(static_weights, dim=0))
        self.condition_down = torch.nn.Linear(
            len(scalar_indices), self.rank, bias=True, **factory_kwargs
        )
        self.dynamic_up = torch.nn.Linear(
            self.rank, self.num_path_weights, bias=True, **factory_kwargs
        )
        self.reset_dynamic_parameters()

        # Fused-forward acceleration structures (design A.2).  Derived purely from
        # the per-path data above; every tensor is a NON-persistent buffer so the
        # state_dict is byte-identical to the legacy module (the 5 learnable
        # tensors only) and old checkpoints load strict=True.
        self._build_fused_plan(device=device)

    def reset_dynamic_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.condition_down.weight, a=math.sqrt(5.0))
        if self.condition_down.bias is not None:
            torch.nn.init.zeros_(self.condition_down.bias)
        if self.dynamic_init == 0.0:
            torch.nn.init.zeros_(self.dynamic_up.weight)
        else:
            torch.nn.init.normal_(
                self.dynamic_up.weight, mean=0.0, std=self.dynamic_init
            )
        if self.dynamic_up.bias is not None:
            torch.nn.init.zeros_(self.dynamic_up.bias)

    def _build_fused_plan(self, device: Optional[Union[str, torch.device]]) -> None:
        """Precompute the constant group structures for the fused forward.

        Paths are partitioned by ``(input_index, row_l, col_l)`` (design A.1).
        Every path in a group shares the same input slice, ``mul_in``, ``ir_in.dim``
        and -- verified bit-identical -- Wigner tensor, so the group collapses to a
        single batched pair of einsums.  Only three constant-integer facts differ
        per path within a group: the ``[weight_start:weight_stop]`` slice it reads
        and the ``(row, col)`` canvas block it writes.  Both are precomputed here.
        """
        input_slices = self.irreps_in.slices()

        groups: "dict[tuple, list[int]]" = {}
        for path_index, path in enumerate(self._paths):
            input_index, row_index, col_index = path[0], path[1], path[2]
            row_l = self._shell_specs[row_index][2]
            col_l = self._shell_specs[col_index][2]
            groups.setdefault((input_index, row_l, col_l), []).append(path_index)

        # Per-group non-tensor metadata (plain ints -- no device move needed).
        self._fused_groups = []
        scatter_index_parts = []
        for group_index, ((input_index, row_l, col_l), path_list) in enumerate(
            groups.items()
        ):
            mul_in, ir_in = self.irreps_in[input_index]
            in_slice = input_slices[input_index]
            row_dim = 2 * row_l + 1
            col_dim = 2 * col_l + 1

            # (a) weight-gather index [P, u]: each path reads its own original-order
            # ``arange(weight_start, weight_stop)`` slice.  This permutes indices at
            # READ time only; the stored ``static_weights`` / ``dynamic_up`` output
            # layout is never reordered (design A.3, R1), so the state_dict stays
            # byte-identical.
            weight_index = torch.stack(
                [
                    torch.arange(
                        self._paths[p][3],
                        self._paths[p][4],
                        dtype=torch.long,
                        device=device,
                    )
                    for p in path_list
                ],
                dim=0,
            )
            self.register_buffer(
                f"_fused_weight_index_{group_index}", weight_index, persistent=False
            )

            # (b) shared Wigner [i, j, k] * sqrt(2*l_in+1), verified bit-identical
            # within the group; kept as a non-persistent buffer so it moves with the
            # module under ``.to()`` -- this pre-casts once and removes the per-path
            # ``.to()`` cast the legacy loop pays every forward (design A.2 / A.6).
            shared_wigner = getattr(self, self._paths[path_list[0]][5]).clone()
            self.register_buffer(
                f"_fused_wigner_{group_index}", shared_wigner, persistent=False
            )

            # (c) flattened canvas scatter positions, ordered (path, i, j) to match
            # ``block.reshape(*batch, P*i*j)`` in the fused forward.
            for p in path_list:
                row_start = self._shell_specs[self._paths[p][1]][0]
                col_start = self._shell_specs[self._paths[p][2]][0]
                rows = row_start + torch.arange(row_dim, dtype=torch.long, device=device)
                cols = col_start + torch.arange(col_dim, dtype=torch.long, device=device)
                scatter_index_parts.append(
                    (rows[:, None] * self.max_norb + cols[None, :]).reshape(-1)
                )

            self._fused_groups.append(
                (
                    int(in_slice.start),
                    int(in_slice.stop),
                    int(mul_in),
                    int(ir_in.dim),
                    len(path_list),
                    int(row_dim),
                    int(col_dim),
                )
            )

        self._fused_num_groups = len(self._fused_groups)
        # One head-level scatter index (concatenated in the SAME group/path order
        # the fused forward concatenates its blocks) -> a single ``index_add``.
        scatter_index = torch.cat(scatter_index_parts, dim=0)
        self.register_buffer("_fused_scatter_index", scatter_index, persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.irreps_in.dim:
            raise ValueError(
                f"Expected last dimension {self.irreps_in.dim}, got "
                f"{features.shape[-1]}."
            )
        # -------------------------------------------------------------------
        # Dispatch (design A.7 / A.8; project decision 2026-07-22 P1-3:
        # COMPATIBILITY-DEFAULT).  The reference per-path loop is the DEFAULT;
        # the fused grouped-einsum path is an explicit opt-in via
        # ``self.cg_head_impl`` (see the module-level comment for the full
        # rationale).  Precedence, highest first:
        #
        #   (a) ``DPTB_LEGACY_CG_HEAD=1`` -> legacy.  Absolute manual override
        #       (bit-continuity with pre-fusion history / emergency rollback).
        #   (b) certified-domain guard -> legacy.  The fused reassociation is only a
        #       safe substitute in EAGER fp32/fp64 -- the only dtypes the block-ODE
        #       contract admits and production uses.  Outside that domain we ALWAYS
        #       fall back to the per-path loop, regardless of ``cg_head_impl``, when
        #       ANY of the following holds:
        #         * autocast is active -- under bf16/fp16 autocast the fused grouped
        #           einsums and the single scatter reduce in a DIFFERENT order than
        #           the legacy per-path loop, and at low precision that reorder is
        #           materially divergent, not roundoff.  Both the CUDA/device-agnostic
        #           and the CPU autocast flags are checked (see ``_autocast_is_active``).
        #         * the input dtype is neither fp32 nor fp64 -- same low-precision
        #           reduction-order divergence for an explicitly half/bf16 feature
        #           tensor, with no autocast context in play.
        #         * deterministic algorithms are enabled -- the fused head's single
        #           ``index_add`` scatter writes duplicate destination indices
        #           (overlapping shell-pair blocks), a documented nondeterministic
        #           CUDA reduction (atomicAdd).  Under ``use_deterministic_algorithms(
        #           True)`` we route to the per-path loop, which assembles each block by
        #           explicit ``cat`` and never scatters onto shared cells.
        #   (c) ``self.cg_head_impl == "fused"`` -> fused.  Explicit constructor/
        #       config opt-in (external benchmark, not reproduced in-tree: ~4-5x
        #       head / ~-3.2% iter); its output equals the legacy loop only up to fp
        #       reassociation -- NOT elementwise-tight under cancellation, and under
        #       cancellation neither path is guaranteed closer to the fp64 truth
        #       (see the module comment).  ``DPTB_FUSED_CG_HEAD`` is NOT read here;
        #       the name is inert (the opt-in is the config field, not an env var).
        #   (d) DEFAULT -> legacy.  No opt-in selects fused, so ``forward`` stays
        #       bit-stable with pre-fusion (0715-refactor) checkpoints/configs.
        # -------------------------------------------------------------------
        if os.environ.get(_LEGACY_CG_HEAD_ENV) == "1":
            return self._forward_legacy(features)
        if (
            _autocast_is_active()
            or features.dtype not in (torch.float32, torch.float64)
            or torch.are_deterministic_algorithms_enabled()
        ):
            return self._forward_legacy(features)
        if self.cg_head_impl == _CG_HEAD_IMPL_FUSED:
            return self._forward_fused(features)
        return self._forward_legacy(features)

    def _forward_fused(self, features: torch.Tensor) -> torch.Tensor:
        """Grouped batched-einsum forward + a single scatter (design A.2).

        The opt-in performance path, selected via ``cg_head_impl="fused"`` inside
        the certified eager fp32/fp64 domain (project decision 2026-07-22 P1-3
        COMPATIBILITY-DEFAULT, superseding the 2026-07-20 PERFORMANCE-DEFAULT);
        ``_forward_legacy`` is the DEFAULT and remains the reduction-order oracle.
        Equivalent to the legacy loop only up to floating-point REASSOCIATION -- the
        grouped einsums + single scatter reduce in a different order.  Absent
        cancellation the drift is pure roundoff
        (fp64 <= ~4e-16, fp32 <= ~2e-7; design A.4); under fp32 cancellation with
        independent per-tensor scales the relative elementwise drift can exceed any
        tight rtol and NEITHER path is guaranteed closer to the fp64 truth, so this
        path does NOT claim elementwise parity with the legacy accumulation order.  A
        failure above the certified reassociation tolerances (in the non-cancellation
        regime the parity tests exercise) signals a real semantic divergence, not
        precision.
        """
        condition = features.index_select(-1, self._scalar_indices)
        invariant = torch.nn.functional.silu(self.condition_down(condition))
        dynamic_weights = self.dynamic_up(invariant)      # [..., num_path_weights]
        static = self.static_weights                      # [num_path_weights]

        batch_shape = features.shape[:-1]
        canvas = features.new_zeros(*batch_shape, self.max_norb * self.max_norb)

        group_blocks = []
        for group_index, (
            in_start,
            in_stop,
            mul_in,
            ir_dim,
            num_paths,
            row_dim,
            col_dim,
        ) in enumerate(self._fused_groups):
            input_block = features[..., in_start:in_stop].reshape(
                *batch_shape, mul_in, ir_dim
            )
            weight_index = getattr(self, f"_fused_weight_index_{group_index}")  # [P,u]
            # Pre-cast buffer; ``.to`` is an identity (no copy/launch) when the
            # module and features already share dtype/device -- the common path.
            wigner = getattr(self, f"_fused_wigner_{group_index}").to(
                dtype=features.dtype, device=features.device
            )  # [i, j, k]
            mixed_weights = static[weight_index] + dynamic_weights[..., weight_index]
            mixed = torch.einsum(
                "...pu,...uk->...pk", mixed_weights, input_block
            ) / float(mul_in)                                       # [..., P, k]
            block = torch.einsum("ijk,...pk->...pij", wigner, mixed)  # [..., P, i, j]
            group_blocks.append(
                block.reshape(*batch_shape, num_paths * row_dim * col_dim)
            )

        # One concatenation + one scatter for the whole head.  ``index_add``
        # accumulates so a canvas block fed by several groups (e.g. p-p receiving
        # l_in in {0e,1e,2e}) sums exactly as the legacy contribution list did;
        # untouched blocks are never indexed and stay bit-zero (design A.2, R2).
        flat_blocks = torch.cat(group_blocks, dim=-1)
        canvas = canvas.index_add(-1, self._fused_scatter_index, flat_blocks)
        output = canvas.reshape(*batch_shape, self.max_norb, self.max_norb)

        if self.symmetrize:
            output = 0.5 * (output + output.transpose(-1, -2))
        return output

    def _forward_legacy(self, features: torch.Tensor) -> torch.Tensor:
        """Reference per-path loop -- the reduction-order-stable ORACLE the fused
        equivalence tests compare against and the DEFAULT forward path (design A.7;
        project decision 2026-07-22 P1-3 COMPATIBILITY-DEFAULT).  ``forward`` routes
        here whenever ``DPTB_LEGACY_CG_HEAD=1`` is set, the certified eager fp32/fp64
        domain guard fires (autocast, low-precision dtype, deterministic mode), or
        ``cg_head_impl`` is not explicitly set to ``"fused"`` (the default)."""
        condition = features.index_select(-1, self._scalar_indices)
        invariant = torch.nn.functional.silu(self.condition_down(condition))
        dynamic_weights = self.dynamic_up(invariant)
        input_slices = self.irreps_in.slices()

        contributions = {
            (row_index, col_index): []
            for row_index in range(len(self._shell_specs))
            for col_index in range(len(self._shell_specs))
        }
        for (
            input_index,
            row_index,
            col_index,
            weight_start,
            weight_stop,
            buffer_name,
        ) in self._paths:
            mul_in, ir_in = self.irreps_in[input_index]
            input_block = features[..., input_slices[input_index]].reshape(
                *features.shape[:-1], mul_in, ir_in.dim
            )
            static = self.static_weights[weight_start:weight_stop]
            dynamic = dynamic_weights[..., weight_start:weight_stop]
            mixed = torch.einsum(
                "...u,...uk->...k", static + dynamic, input_block
            ) / float(mul_in)
            coefficient = getattr(self, buffer_name).to(
                dtype=features.dtype, device=features.device
            )
            block = torch.einsum("ijk,...k->...ij", coefficient, mixed)
            contributions[(row_index, col_index)].append(block)

        rows = []
        for row_index, (row_start, row_stop, _) in enumerate(self._shell_specs):
            row = []
            row_dim = row_stop - row_start
            for col_index, (col_start, col_stop, _) in enumerate(self._shell_specs):
                values = contributions[(row_index, col_index)]
                if values:
                    block = values[0]
                    for value in values[1:]:
                        block = block + value
                else:
                    block = features.new_zeros(
                        *features.shape[:-1], row_dim, col_stop - col_start
                    )
                row.append(block)
            rows.append(torch.cat(row, dim=-1))
        output = torch.cat(rows, dim=-2)

        if self.symmetrize:
            output = 0.5 * (output + output.transpose(-1, -2))
        return output
