import pytest


def test_expert_parallel_layout_defaults_to_one_rank_per_expert():
    from dptb.nnops.expert_parallel_layout import (
        rank_to_expert_parallel,
        resolve_expert_parallel_layout,
    )

    layout = resolve_expert_parallel_layout(
        num_experts=2,
        world_size=2,
        train_options={},
    )

    assert layout.expert_data_parallel_size == 1
    assert layout.expected_world_size == 2

    rank0 = rank_to_expert_parallel(
        rank=0,
        num_experts=layout.num_experts,
        expert_data_parallel_size=layout.expert_data_parallel_size,
    )
    rank1 = rank_to_expert_parallel(
        rank=1,
        num_experts=layout.num_experts,
        expert_data_parallel_size=layout.expert_data_parallel_size,
    )

    assert rank0.local_expert_idx == 0
    assert rank0.expert_dp_rank == 0
    assert rank0.expert_group_ranks == [0]
    assert rank1.local_expert_idx == 1
    assert rank1.expert_dp_rank == 0
    assert rank1.expert_group_ranks == [1]


def test_expert_parallel_layout_maps_contiguous_replicas_to_same_expert():
    from dptb.nnops.expert_parallel_layout import (
        rank_to_expert_parallel,
        resolve_expert_parallel_layout,
    )

    layout = resolve_expert_parallel_layout(
        num_experts=2,
        world_size=4,
        train_options={"expert_data_parallel_size": 2},
    )

    assert layout.expert_data_parallel_size == 2
    assert layout.expected_world_size == 4

    rank0 = rank_to_expert_parallel(
        rank=0,
        num_experts=2,
        expert_data_parallel_size=2,
    )
    rank1 = rank_to_expert_parallel(
        rank=1,
        num_experts=2,
        expert_data_parallel_size=2,
    )
    rank2 = rank_to_expert_parallel(
        rank=2,
        num_experts=2,
        expert_data_parallel_size=2,
    )
    rank3 = rank_to_expert_parallel(
        rank=3,
        num_experts=2,
        expert_data_parallel_size=2,
    )

    assert rank0.local_expert_idx == 0
    assert rank0.expert_dp_rank == 0
    assert rank0.expert_group_ranks == [0, 1]
    assert rank1.local_expert_idx == 0
    assert rank1.expert_dp_rank == 1
    assert rank1.expert_group_ranks == [0, 1]
    assert rank2.local_expert_idx == 1
    assert rank2.expert_dp_rank == 0
    assert rank2.expert_group_ranks == [2, 3]
    assert rank3.local_expert_idx == 1
    assert rank3.expert_dp_rank == 1
    assert rank3.expert_group_ranks == [2, 3]


def test_expert_parallel_layout_rejects_world_size_mismatch():
    from dptb.nnops.expert_parallel_layout import resolve_expert_parallel_layout

    with pytest.raises(ValueError, match="world_size must equal num_experts \\* expert_data_parallel_size"):
        resolve_expert_parallel_layout(
            num_experts=2,
            world_size=3,
            train_options={"expert_data_parallel_size": 2},
        )


def test_expert_parallel_layout_accepts_short_expert_dp_alias():
    from dptb.nnops.expert_parallel_layout import resolve_expert_parallel_layout

    layout = resolve_expert_parallel_layout(
        num_experts=3,
        world_size=6,
        train_options={"expert_dp_size": 2},
    )

    assert layout.expert_data_parallel_size == 2
    assert layout.expected_world_size == 6


def test_expert_parallel_layout_rejects_conflicting_aliases():
    from dptb.nnops.expert_parallel_layout import resolve_expert_parallel_layout

    with pytest.raises(ValueError, match="conflicting expert data parallel settings"):
        resolve_expert_parallel_layout(
            num_experts=2,
            world_size=4,
            train_options={"expert_data_parallel_size": 2, "expert_dp_size": 3},
        )


def test_distributed_restart_rejects_state_list_length_mismatch():
    from dptb.nnops.multi_trainer import MultiTrainer

    with pytest.raises(RuntimeError, match="holds 1 entries.*2 experts"):
        MultiTrainer._validate_distributed_resume_state_list(
            [{}], expected_count=2, local_idx=0,
            label="optimizers_state_dict",
        )
    with pytest.raises(RuntimeError, match="outside checkpoint.*bounds"):
        MultiTrainer._validate_distributed_resume_state_list(
            [{}, {}], expected_count=2, local_idx=2,
            label="lr_schedulers_state_dict",
        )
