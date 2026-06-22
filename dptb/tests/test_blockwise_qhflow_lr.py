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
            },
        },
    }

    normalized = train_options().normalize_value(cfg)
    train_options().check_value(normalized, strict=True)
    assert normalized["loss_options"]["train"]["method"] == "hamil_blockwise_nextham"
