import math
import os
import torch
from torch_runstats.scatter import scatter
from dptb.data import _keys
import logging
from typing import Optional, List, Union, Tuple
import torch.nn.functional
from e3nn.o3 import Linear
from e3nn.util.jit import compile_mode
from dptb.data import AtomicDataDict
import e3nn.o3 as o3

class PerSpeciesScaleShift(torch.nn.Module):
    """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

    Args:
        field: the per-atom field to scale/shift.
        num_types: the number of types in the model.
        shifts: the initial shifts to use, one per atom type.
        scales: the initial scales to use, one per atom type.
        arguments_in_dataset_units: if ``True``, says that the provided shifts/scales are in dataset
            units (in which case they will be rescaled appropriately by any global rescaling later
            applied to the model); if ``False``, the provided shifts/scales will be used without modification.

            For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
            But if scales/shifts computed from the training data are used, and are thus in dataset units,
            this should be ``True``.
        out_field: the output field; defaults to ``field``.
    """

    field: str
    out_field: str
    scales_trainble: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool

    def __init__(
        self,
        field: str,
        num_types: int,
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_types = num_types
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field

        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types) * shifts
            assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types) * scales
            assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY]
        in_field = data[self.field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"
        if self.has_scales:
            in_field = self.scales[species_idx].view(-1, 1) * in_field
        if self.has_shifts:
            in_field = self.shifts[species_idx].view(-1, 1) + in_field
        data[self.out_field] = in_field
        return data

    # def update_for_rescale(self, rescale_module):
    #     if hasattr(rescale_module, "related_scale_keys"):
    #         if self.out_field not in rescale_module.related_scale_keys:
    #             return
    #     if self.arguments_in_dataset_units and rescale_module.has_scale:
    #         logging.debug(
    #             f"PerSpeciesScaleShift's arguments were in dataset units; rescaling:\n  "
    #             f"Original scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
    #             f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
    #         )
    #         with torch.no_grad():
    #             if self.has_scales:
    #                 self.scales.div_(rescale_module.scale_by)
    #             if self.has_shifts:
    #                 self.shifts.div_(rescale_module.scale_by)
    #         logging.debug(
    #             f"  New scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
    #             f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
    #         )

class PerEdgeSpeciesScaleShift(torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """

    field: str
    out_field: str
    scales_trainble: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool

    def __init__(
        self,
        field: str,
        num_types: int,
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        **kwargs,
    ):
        """Sum edges into nodes."""
        super(PerEdgeSpeciesScaleShift, self).__init__()
        self.num_types = num_types
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field

        self.has_shifts = shifts is not None
        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types, num_types) * scales
            assert scales.shape == (num_types, num_types,), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types, num_types) * shifts
            assert shifts.shape == (num_types, num_types,), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)



    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        center_species = species_idx[edge_center]
        neighbor_species = species_idx[edge_neighbor]
        in_field = data[self.field]

        assert len(in_field) == len(
            edge_center
        ), "in_field doesnt seem to have correct per-edge shape"


        if self.has_scales:
            in_field = self.scales[center_species, neighbor_species].view(-1, 1) * in_field
        if self.has_shifts:
            in_field = self.shifts[center_species, neighbor_species].view(-1, 1) + in_field

        data[self.out_field] = in_field

        return data


class E3PerEdgeSpeciesScaleShift(torch.nn.Module):
    def __init__(
        self,
        field: str,
        num_types: int,
        irreps_in,
        shifts: Optional[torch.Tensor],
        scales: Optional[torch.Tensor],
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        scale_type: str = 'scale_w_back_grad',
        **kwargs,
    ):
        super(E3PerEdgeSpeciesScaleShift, self).__init__()
        self.num_types = num_types
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self.irreps_in = irreps_in
        self.num_scalar = 0
        self.device = device
        self.dtype = dtype
        shift_indices = []
        scale_indices = []
        self.scale_type = scale_type
        self.scales_trainable = scales_trainable

        start = 0
        start_scalar = 0
        for mul, ir in irreps_in:
            if getattr(ir, "l", None) == 0:  # 0e + 0o 都算
                self.num_scalar += mul
                shift_indices += list(range(start_scalar, start_scalar + mul))
                start_scalar += mul
            else:
                shift_indices += [-1] * (mul * ir.dim)

            for _ in range(mul):
                scale_indices += [start] * ir.dim
                start += 1

        self.shift_index = torch.as_tensor(shift_indices, dtype=torch.long, device=device)
        self.scale_index = torch.as_tensor(scale_indices, dtype=torch.long, device=device)

        self.has_shifts = shifts is not None
        self.has_scales = (scales is not None) and (scale_type != "no_scale")

        if scales is not None:
            scales = torch.as_tensor(scales, dtype=self.dtype, device=device)
            if len(scales.reshape(-1)) == 1:
                scales = scales * torch.ones(num_types * num_types, self.irreps_in.num_irreps, dtype=self.dtype, device=self.device)
            assert scales.shape == (num_types * num_types, self.irreps_in.num_irreps), \
                f"Invalid shape of scales {scales.shape}, expect {(num_types * num_types, self.irreps_in.num_irreps)}"
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=self.dtype, device=device)
            if len(shifts.reshape(-1)) == 1:
                shifts = shifts * torch.ones(num_types * num_types, self.num_scalar, dtype=self.dtype, device=self.device)
            assert shifts.shape == (num_types * num_types, self.num_scalar), \
                f"Invalid shape of shifts {shifts.shape}, expect {(num_types * num_types, self.num_scalar)}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

    def set_scale_shift(self, scales: torch.Tensor = None, shifts: torch.Tensor = None):
        self.has_scales = scales is not None or self.has_scales
        if scales is not None:
            assert scales.shape == (self.num_types * self.num_types, self.irreps_in.num_irreps), f"Invalid shape of scales {scales.shape}"
            if self.scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

        self.has_shifts = shifts is not None or self.has_shifts
        if shifts is not None:
            assert shifts.shape == (self.num_types * self.num_types, self.num_scalar), \
                f"Invalid shape of shifts {shifts.shape}, expect {(self.num_types * self.num_types, self.num_scalar)}"
            if self.shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not (self.has_scales or self.has_shifts):
            return data

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        mask = data[self.field][:, 0] != 0  # strictly zero valued point must come from masked edges
        in_field = data[self.field][mask]
        species_idx = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()[mask]

        assert len(in_field) == len(edge_center[mask]), "in_field doesnt seem to have correct per-edge shape"

        if self.has_scales:
            scales = self.scales[species_idx][:, self.scale_index].view(-1, self.irreps_in.dim)
            if self.scale_type == 'scale_w_back_grad':
                in_field = scales * in_field
            elif self.scale_type == 'scale_wo_back_grad':
                if self.scales_trainable:
                    in_field = in_field + in_field.detach() * (scales - 1.0)
                else:
                    in_field = in_field + (in_field * (scales - 1.0)).detach()
            else:
                raise NotImplementedError

        if self.has_shifts:
            shifts = self.shifts[species_idx][:, self.shift_index[self.shift_index >= 0]].view(-1, self.num_scalar)
            in_field[:, self.shift_index >= 0] = shifts + in_field[:, self.shift_index >= 0]

        # out_field 通常等于 field（你在 NNENV 里就是这样传的），所以这里是安全的
        data[self.out_field][mask] = in_field
        return data


class E3PerSpeciesScaleShift(torch.nn.Module):
    def __init__(
        self,
        field: str,
        num_types: int,
        irreps_in,
        shifts: Optional[torch.Tensor],
        scales: Optional[torch.Tensor],
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        scale_type: str = 'scale_w_back_grad',
        **kwargs,
    ):
        super().__init__()
        self.num_types = num_types
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self.irreps_in = irreps_in
        self.num_scalar = 0
        shift_indices = []
        scale_indices = []
        self.dtype = dtype
        self.device = device
        self.scale_type = scale_type
        self.scales_trainable = scales_trainable

        start = 0
        start_scalar = 0
        for mul, ir in irreps_in:
            # SOC 下既可能有 0e 也可能有 0o；它们都是 l==0 的标量通道
            if getattr(ir, "l", None) == 0:
                self.num_scalar += mul
                shift_indices += list(range(start_scalar, start_scalar + mul))
                start_scalar += mul
            else:
                shift_indices += [-1] * (mul * ir.dim)

            for _ in range(mul):
                scale_indices += [start] * ir.dim
                start += 1

        self.shift_index = torch.as_tensor(shift_indices, dtype=torch.long, device=device)
        self.scale_index = torch.as_tensor(scale_indices, dtype=torch.long, device=device)

        # ---- shifts ----
        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=self.dtype, device=device)
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types, self.num_scalar, dtype=self.dtype, device=device) * shifts
            assert shifts.shape == (num_types, self.num_scalar), f"Invalid shape of shifts {shifts.shape}, expect {(num_types, self.num_scalar)}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

        # ---- scales ----
        self.has_scales = (scales is not None) and (scale_type != "no_scale")
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=self.dtype, device=device)
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types, self.irreps_in.num_irreps, dtype=self.dtype, device=device) * scales
            assert scales.shape == (num_types, self.irreps_in.num_irreps), f"Invalid shape of scales {scales.shape}, expect {(num_types, self.irreps_in.num_irreps)}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

    def set_scale_shift(self, scales: torch.Tensor = None, shifts: torch.Tensor = None):
        self.has_scales = scales is not None or self.has_scales
        if scales is not None:
            assert scales.shape == (self.num_types, self.irreps_in.num_irreps), f"Invalid shape of scales {scales.shape}"
            if self.scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

        self.has_shifts = shifts is not None or self.has_shifts
        if shifts is not None:
            assert shifts.shape == (self.num_types, self.num_scalar), f"Invalid shape of shifts {shifts.shape}, expect {(self.num_types, self.num_scalar)}"
            if self.shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        if not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        in_field = data[self.field]
        assert len(in_field) == len(species_idx), "in_field doesnt seem to have correct per-atom shape"

        if self.has_scales:
            scales = self.scales[species_idx][:, self.scale_index].view(-1, self.irreps_in.dim)
            if self.scale_type == 'scale_w_back_grad':
                in_field = scales * in_field
            elif self.scale_type == 'scale_wo_back_grad':
                if self.scales_trainable:
                    in_field = in_field + in_field.detach() * (scales - 1.0)
                else:
                    in_field = in_field + (in_field * (scales - 1.0)).detach()
            else:
                raise NotImplementedError

        if self.has_shifts:
            shifts = self.shifts[species_idx][:, self.shift_index[self.shift_index >= 0]].view(-1, self.num_scalar)
            in_field[:, self.shift_index >= 0] = shifts + in_field[:, self.shift_index >= 0]

        data[self.out_field] = in_field
        return data


@compile_mode("script")
class E3ElementLinear(torch.nn.Module):
    """Per-channel scale of every irrep channel, plus a shift of every 0e channel.

    ``weights`` holds, per row, one scale per irrep channel (``num_scales``) followed by
    one shift per 0e channel (``num_shifts``); without weights the input is returned.
    The input and the weights are split into their irrep blocks once and the output is
    concatenated once, so the backward writes one gradient per input.
    """

    weight_numel: int
    _muls: List[int]
    _dims: List[int]
    _shifted: List[bool]
    _widths: List[int]
    _scale_widths: List[int]
    _shift_widths: List[int]

    def __init__(
        self,
        irreps_in: o3.Irreps,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        super(E3ElementLinear, self).__init__()
        self.irreps_in = irreps_in
        self.num_scalar = 0
        self.device = device
        self.dtype = dtype
        self._muls = []
        self._dims = []
        self._shifted = []
        for mul, ir in irreps_in:
            shifted = str(ir) == "0e"
            if shifted:
                self.num_scalar += mul
            self._muls.append(mul)
            self._dims.append(ir.dim)
            self._shifted.append(shifted)
        self.num_scales = irreps_in.num_irreps
        self.num_shifts = self.num_scalar
        self.weight_numel = self.num_scales + self.num_shifts
        self._widths = [mul * dim for mul, dim in zip(self._muls, self._dims)]
        self._scale_widths = list(self._muls)
        self._shift_widths = [mul for mul, shifted in zip(self._muls, self._shifted) if shifted]

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None):
        if weights is None:
            return x
        assert len(weights) == len(x), "in_field doesnt seem to have correct shape as scales"
        n = x.shape[0]
        extra = weights.shape[1] - self.weight_numel
        shifts: Optional[torch.Tensor] = None
        if weights.shape[1] == self.num_scales:
            scales = weights
        elif extra >= 0:
            pieces = torch.split(weights, [self.num_scales, self.num_shifts, extra], dim=1)
            scales = pieces[0]
            shifts = pieces[1]
        else:
            raise ValueError(
                "E3ElementLinear expects " + str(self.num_scales) + " scales and " + str(self.num_shifts)
                + " shifts per row, got " + str(weights.shape[1]) + " weights"
            )
        x_blocks = torch.split(x, self._widths, dim=1)
        scale_blocks = torch.split(scales, self._scale_widths, dim=1)
        shift_blocks: List[torch.Tensor] = []
        if shifts is not None:
            shift_blocks = list(torch.split(shifts, self._shift_widths, dim=1))
        parts: List[torch.Tensor] = []
        k = 0
        for i in range(len(self._muls)):
            mul = self._muls[i]
            dim = self._dims[i]
            if dim == 1:
                part = x_blocks[i] * scale_blocks[i]
            else:
                part = (x_blocks[i].reshape(n, mul, dim) * scale_blocks[i].unsqueeze(-1)).reshape(n, mul * dim)
            if self._shifted[i] and shifts is not None:
                part = shift_blocks[k] + part
                k += 1
            parts.append(part)
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)
