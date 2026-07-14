"""Shared registration for the stable core training monitors."""

from dptb.plugins.monitor import (
    ExpertLoadCVMonitor,
    LearningRateMonitor,
    ScalarFieldMonitor,
    TrainHoppingLossMonitor,
    TrainLossMonitor,
    TrainOnsiteLossMonitor,
    TrainZLossMonitor,
)


_EVERY_STEP_AND_EPOCH = [(1, "iteration"), (1, "epoch")]


def register_core_training_monitors(
    trainer,
    *,
    train_endpoint_capable: bool,
    sliding_win_size: int,
    avg_per_iter: bool,
) -> None:
    """Register metrics that must share the same per-iteration population."""

    trainer.register_plugin(
        TrainLossMonitor(
            sliding_win_size=sliding_win_size,
            avg_per_iter=avg_per_iter,
        )
    )
    trainer.register_plugin(LearningRateMonitor())
    trainer.register_plugin(
        ScalarFieldMonitor(
            stat_name="train_loss_opt",
            interval=list(_EVERY_STEP_AND_EPOCH),
            sliding_win_size=sliding_win_size,
            avg_per_iter=avg_per_iter,
        )
    )
    trainer.register_plugin(
        ScalarFieldMonitor(
            stat_name="total_grad_norm",
            interval=list(_EVERY_STEP_AND_EPOCH),
        )
    )
    if train_endpoint_capable:
        trainer.register_plugin(
            TrainOnsiteLossMonitor(interval=list(_EVERY_STEP_AND_EPOCH))
        )
        trainer.register_plugin(
            TrainHoppingLossMonitor(interval=list(_EVERY_STEP_AND_EPOCH))
        )
    trainer.register_plugin(TrainZLossMonitor(interval=list(_EVERY_STEP_AND_EPOCH)))
    trainer.register_plugin(
        ExpertLoadCVMonitor(interval=list(_EVERY_STEP_AND_EPOCH))
    )
