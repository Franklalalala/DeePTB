import torch

from dptb.nnops.distance_expert_mask import (
    edge_mask_for_distance_expert,
    node_mask_for_distance_expert,
)


def _dist():
    return torch.tensor([0.0, 5e-7, 1e-6, 1.5, 12.0])


def test_two_expert_default_matches_historical():
    dist = _dist()
    onsite = edge_mask_for_distance_expert(
        dist, 0.0, 1e-6, is_last_expert=False, clip_last_expert_range=False
    )
    hopping = edge_mask_for_distance_expert(
        dist, 1e-6, 10.0, is_last_expert=True, clip_last_expert_range=False
    )
    assert onsite.tolist() == [True, True, False, False, False]
    assert hopping.tolist() == [False, False, True, True, True]


def test_onsite_one_gpu_without_clip_eats_all_edges():
    dist = _dist()
    bad = edge_mask_for_distance_expert(
        dist, 0.0, 1e-6, is_last_expert=True, clip_last_expert_range=False
    )
    assert bad.tolist() == [True, True, True, True, True]


def test_onsite_one_gpu_with_clip_keeps_onsite_only():
    dist = _dist()
    good = edge_mask_for_distance_expert(
        dist, 0.0, 1e-6, is_last_expert=True, clip_last_expert_range=True
    )
    assert good.tolist() == [True, True, False, False, False]


def test_hopping_one_gpu_default_matches_original_last_expert():
    dist = _dist()
    hop = edge_mask_for_distance_expert(
        dist, 1e-6, 10.0, is_last_expert=True, clip_last_expert_range=False
    )
    assert hop.tolist() == [False, False, True, True, True]


def test_node_mask_onsite_vs_hopping():
    onsite = node_mask_for_distance_expert(4, 0.0, device="cpu")
    hopping = node_mask_for_distance_expert(4, 1e-6, device="cpu")
    assert onsite.tolist() == [True, True, True, True]
    assert hopping.tolist() == [False, False, False, False]
