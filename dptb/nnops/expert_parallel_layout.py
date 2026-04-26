from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class ExpertParallelLayout:
    num_experts: int
    expert_data_parallel_size: int
    expected_world_size: int


@dataclass(frozen=True)
class ExpertRankInfo:
    rank: int
    local_expert_idx: int
    expert_dp_rank: int
    expert_group_ranks: List[int]


def _parse_positive_int(value: Any, *, name: str) -> int:
    out = int(value)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {out}")
    return out


def get_expert_data_parallel_size(train_options: Dict[str, Any]) -> int:
    if train_options is None:
        return 1
    value = train_options.get("expert_data_parallel_size", 1)
    alias_value = train_options.get("expert_dp_size", 1)
    if int(value) != 1 and int(alias_value) != 1 and int(value) != int(alias_value):
        raise ValueError(
            "conflicting expert data parallel settings: "
            f"expert_data_parallel_size={value}, expert_dp_size={alias_value}"
        )
    if int(value) == 1 and int(alias_value) != 1:
        value = alias_value
    return _parse_positive_int(value, name="expert_data_parallel_size")


def resolve_expert_parallel_layout(
    *,
    num_experts: int,
    world_size: int,
    train_options: Dict[str, Any],
) -> ExpertParallelLayout:
    num_experts = _parse_positive_int(num_experts, name="num_experts")
    world_size = _parse_positive_int(world_size, name="world_size")
    expert_data_parallel_size = get_expert_data_parallel_size(train_options)
    expected_world_size = num_experts * expert_data_parallel_size

    if world_size != expected_world_size:
        raise ValueError(
            "In distributed_expert mode, world_size must equal num_experts * "
            "expert_data_parallel_size. "
            f"Got world_size={world_size}, num_experts={num_experts}, "
            f"expert_data_parallel_size={expert_data_parallel_size}, "
            f"expected_world_size={expected_world_size}."
        )

    return ExpertParallelLayout(
        num_experts=num_experts,
        expert_data_parallel_size=expert_data_parallel_size,
        expected_world_size=expected_world_size,
    )


def rank_to_expert_parallel(
    *,
    rank: int,
    num_experts: int,
    expert_data_parallel_size: int,
) -> ExpertRankInfo:
    rank = int(rank)
    num_experts = _parse_positive_int(num_experts, name="num_experts")
    expert_data_parallel_size = _parse_positive_int(
        expert_data_parallel_size,
        name="expert_data_parallel_size",
    )
    world_size = num_experts * expert_data_parallel_size
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    local_expert_idx = rank // expert_data_parallel_size
    expert_dp_rank = rank % expert_data_parallel_size
    group_start = local_expert_idx * expert_data_parallel_size
    expert_group_ranks = list(range(group_start, group_start + expert_data_parallel_size))

    return ExpertRankInfo(
        rank=rank,
        local_expert_idx=local_expert_idx,
        expert_dp_rank=expert_dp_rank,
        expert_group_ranks=expert_group_ranks,
    )
