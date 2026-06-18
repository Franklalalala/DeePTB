import math

import torch


def test_hamil_blockwise_nextham_loss_is_registered():
    from dptb.nnops.loss import Loss

    assert "hamil_blockwise_nextham" in Loss._register.keys()


def test_qhflow_poly_scheduler_matches_warmup_and_decay_shape():
    from dptb.utils.tools import get_lr_scheduler

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([param], lr=5.0e-4)
    scheduler = get_lr_scheduler(
        type="qhflow_poly",
        optimizer=optimizer,
        warmup_step=1000,
        num_training_steps=200000,
        end_lr=1.0e-9,
        scheduler_power=1.0,
    )

    lrs = []
    for _ in range(1000):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    assert math.isclose(lrs[-1], 5.0e-4, rel_tol=1e-3)

    for _ in range(99000):
        optimizer.step()
        scheduler.step()
    mid_lr = optimizer.param_groups[0]["lr"]
    assert 2.0e-4 < mid_lr < 3.0e-4

    for _ in range(100000):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] <= 1.1e-9
