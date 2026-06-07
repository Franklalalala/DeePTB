from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RouteLayout:
    graph_index: torch.Tensor
    num_routes: int
    ptr_cpu: torch.Tensor
    order: torch.Tensor | None = None
    unorder: torch.Tensor | None = None


@dataclass(frozen=True)
class SO2MProblem:
    m: int
    in_base: torch.Tensor
    in_l: torch.Tensor
    out_base: torch.Tensor
    out_l: torch.Tensor
    offsets: torch.Tensor
    cin: int
    cout: int
    rotate_in: bool
    rotate_out: bool
    radial_on_input: bool

    @property
    def is_pair(self) -> bool:
        return self.m > 0


@dataclass(frozen=True)
class SO2ProblemBatch:
    route_layout: RouteLayout
    problems: tuple[SO2MProblem, ...]
    wigner_mode: int
    wigner_stride: int

    @property
    def m_values(self) -> tuple[int, ...]:
        return tuple(problem.m for problem in self.problems)
