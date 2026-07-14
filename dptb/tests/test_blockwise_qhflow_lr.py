import pytest


def test_hamil_blockwise_nextham_loss_is_registered():
    from dptb.nnops.loss import Loss

    assert "hamil_blockwise_nextham" in Loss._register.keys()
    assert "hamil_block_abs" in Loss._register.keys()


def test_hamil_blockwise_nextham_argcheck_accepts_rop_config():
    from dptb.utils.argcheck import train_options

    cfg = {
        "num_epoch": 1,
        "batch_size": 1,
        "optimizer": {"type": "AdamW", "lr": 1e-3},
        "lr_scheduler": {"type": "rop"},
        "loss_options": {
            "train": {
                "method": "hamil_blockwise_nextham",
                "optimization": "block_mae",
                "block_reduction": "global",
                "complex_reduction": "modulus",
                "log_feature_compatible": True,
                "feature_log_no_grad": True,
                "distributed_log_reduce": True,
                "loss_weight": 10.0,
            },
        },
    }

    normalized = train_options().normalize_value(cfg)
    train_options().check_value(normalized, strict=True)
    assert normalized["loss_options"]["train"]["method"] == "hamil_blockwise_nextham"
    assert normalized["loss_options"]["train"]["loss_weight"] == 10.0


def test_blockwise_default_endpoint_stays_in_block_space_without_rme_walk(monkeypatch):
    import torch

    import dptb.nnops.blockwise_nextham_loss as blockwise_mod

    monkeypatch.setattr(
        blockwise_mod,
        "feature_components_from_blocks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("RME-compatible slice walk must be opt-in")
        ),
    )
    loss_fn = blockwise_mod.HamilBlockwiseNexTHamLoss(
        basis={"H": "1s"},
        optimization="block_l1_rmse",
        block_reduction="equal_onsite_hopping",
        loss_weight=10.0,
    )
    data = {
        "node_hamil_blocks": torch.zeros(2, 1, 1),
        "edge_hamil_blocks": torch.zeros(2, 1, 1),
        "node_delta_hamil_blocks": torch.tensor([[[1.0]], [[3.0]]]),
        "edge_delta_hamil_blocks": torch.tensor([[[2.0]], [[4.0]]]),
        "node_delta_hamil_block_shape": torch.ones(2, 2, dtype=torch.long),
        "edge_delta_hamil_block_shape": torch.ones(2, 2, dtype=torch.long),
        "atomic_numbers": torch.ones(2, dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
    }

    opt_loss = loss_fn(data, data)
    onsite = 0.5 * (2.0 + 5.0**0.5)
    hopping = 0.5 * (3.0 + 10.0**0.5)
    endpoint = 0.5 * (onsite + hopping)

    assert opt_loss.item() == pytest.approx(10.0 * endpoint)
    assert loss_fn.last_endpoint_loss.item() == pytest.approx(endpoint)
    assert loss_fn.last_onsite_loss.item() == pytest.approx(onsite)
    assert loss_fn.last_hopping_loss.item() == pytest.approx(hopping)
    assert loss_fn.last_endpoint_metric_space == "block"
    assert loss_fn.last_feature_compat_loss is None

    reduced = loss_fn.compatible_loss_from_stats(
        onsite_l1_sum=loss_fn.last_onsite_l1_sum,
        onsite_mse_sum=loss_fn.last_onsite_mse_sum,
        onsite_count=loss_fn.last_onsite_count,
        hopping_l1_sum=loss_fn.last_hopping_l1_sum,
        hopping_mse_sum=loss_fn.last_hopping_mse_sum,
        hopping_count=loss_fn.last_hopping_count,
    )
    assert tuple(value.item() for value in reduced) == pytest.approx(
        (endpoint, onsite, hopping)
    )


def test_feature_optimization_keeps_components_differentiable_with_distributed_log_reduce(monkeypatch):
    import torch

    import dptb.nnops.blockwise_nextham_loss as blockwise_mod

    monkeypatch.setattr(
        blockwise_mod,
        "maybe_all_reduce_components",
        lambda comp: (_ for _ in ()).throw(
            AssertionError("feature optimization components must keep gradients")
        ),
    )
    loss_fn = blockwise_mod.HamilBlockwiseNexTHamLoss(
        basis={"H": "1s"},
        optimization="feature",
        distributed_log_reduce=True,
    )
    pred_node = torch.zeros(2, 1, 1, requires_grad=True)
    pred_edge = torch.zeros(2, 1, 1, requires_grad=True)
    data = {
        "node_hamil_blocks": pred_node,
        "edge_hamil_blocks": pred_edge,
        "node_delta_hamil_blocks": torch.tensor([[[1.0]], [[3.0]]]),
        "edge_delta_hamil_blocks": torch.tensor([[[2.0]], [[4.0]]]),
        "node_delta_hamil_block_shape": torch.ones(2, 2, dtype=torch.long),
        "edge_delta_hamil_block_shape": torch.ones(2, 2, dtype=torch.long),
        "atomic_numbers": torch.ones(2, dtype=torch.long),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
    }

    loss = loss_fn(data, data)
    loss.backward()

    assert pred_node.grad is not None
    assert pred_edge.grad is not None
    assert pred_node.grad.abs().sum().item() > 0.0
    assert pred_edge.grad.abs().sum().item() > 0.0
    assert loss_fn.last_endpoint_metric_space == "rme"


def test_hamil_blockwise_mae_mse_optimization_mode():
    import torch

    from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss

    basis = {"H": "1s", "O": "1s1p"}
    loss_fn = HamilBlockwiseNexTHamLoss(
        basis=basis,
        optimization="block_mae_mse",
        block_reduction="global",
        log_feature_compatible=False,
    )
    max_norb = 4  # 1s1p union
    pred_node = torch.zeros(2, max_norb, max_norb)
    target_node = torch.zeros(2, max_norb, max_norb)
    target_node[0, 0, 0] = 0.5  # H onsite 1x1
    target_node[1, :4, :4] = 0.25  # O onsite 4x4
    pred_edge = torch.zeros(2, max_norb, max_norb)
    target_edge = torch.zeros(2, max_norb, max_norb)
    target_edge[0, :1, :4] = 1.0  # H->O 1x4
    target_edge[1, :4, :1] = -1.0  # O->H 4x1

    data = {
        "node_hamil_blocks": pred_node,
        "edge_hamil_blocks": pred_edge,
        "atom_types": torch.tensor([[0], [1]]),
        "atomic_numbers": torch.tensor([1, 8]),
        "edge_index": torch.tensor([[0, 1], [1, 0]]),
        "node_delta_hamil_blocks": target_node,
        "edge_delta_hamil_blocks": target_edge,
        "node_delta_hamil_block_shape": torch.tensor([[1, 1], [4, 4]]),
        "edge_delta_hamil_block_shape": torch.tensor([[1, 4], [4, 1]]),
    }
    out = loss_fn(data, data)
    # Active entries: node 1 + 16, edge 4 + 4 = 25 total.
    abs_sum = 0.5 + 16 * 0.25 + 4 * 1.0 + 4 * 1.0
    sq_sum = 0.25 + 16 * 0.0625 + 4 * 1.0 + 4 * 1.0
    expected = abs_sum / 25.0 + sq_sum / 25.0
    assert torch.isfinite(out)
    assert abs(out.item() - expected) < 1e-6
    endpoint_onsite = 0.5 * (4.5 / 17.0 + (1.25 / 17.0 + loss_fn.eps) ** 0.5)
    endpoint_hopping = 0.5 * (1.0 + (1.0 + loss_fn.eps) ** 0.5)
    assert loss_fn.last_endpoint_loss.item() == pytest.approx(
        0.5 * (endpoint_onsite + endpoint_hopping)
    )
    assert loss_fn.last_endpoint_loss.item() != pytest.approx(out.item())
    assert loss_fn.last_endpoint_metric_space == "block"

    weighted_loss_fn = HamilBlockwiseNexTHamLoss(
        basis=basis,
        optimization="block_mae_mse",
        block_reduction="global",
        log_feature_compatible=False,
        loss_weight=10.0,
    )
    weighted = weighted_loss_fn(data, data)
    assert abs(weighted.item() - 10.0 * expected) < 1e-6
    assert abs(weighted_loss_fn.last_block_loss.item() - expected) < 1e-6
    assert abs(weighted_loss_fn.last_opt_loss.item() - 10.0 * expected) < 1e-6


def test_blockwise_stats_reducer_matches_feature_endpoint_triplet():
    import torch

    from dptb.nnops.blockwise_nextham_loss import HamilBlockwiseNexTHamLoss

    loss_fn = HamilBlockwiseNexTHamLoss(
        basis={"H": "1s"},
        log_feature_compatible=True,
    )
    total, onsite, hopping = loss_fn.compatible_loss_from_stats(
        onsite_l1_sum=torch.tensor(4.0),
        onsite_mse_sum=torch.tensor(10.0),
        onsite_count=torch.tensor(2.0),
        hopping_l1_sum=torch.tensor(3.0),
        hopping_mse_sum=torch.tensor(9.0),
        hopping_count=torch.tensor(3.0),
    )

    expected_onsite = 0.5 * (2.0 + (5.0 + loss_fn.eps) ** 0.5)
    expected_hopping = 0.5 * (1.0 + (3.0 + loss_fn.eps) ** 0.5)
    assert onsite.item() == pytest.approx(expected_onsite)
    assert hopping.item() == pytest.approx(expected_hopping)
    assert total.item() == pytest.approx(0.5 * (expected_onsite + expected_hopping))
