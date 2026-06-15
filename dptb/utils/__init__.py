from dptb.utils.auto_init import *  # noqa: F401,F403


def _install_rmf_argcheck_fields() -> None:
    """Allow RMF flow_options keys while preserving existing defaults."""

    try:
        from dptb.utils import argcheck as _argcheck
        Argument = _argcheck.Argument
    except Exception:
        # Some lightweight imports do not need the dargs-based config checker.
        # Do not make ordinary DeePTB imports fail because the RMF extension is
        # present but unused.
        return

    if getattr(_argcheck.flow_options, "__name__", "") == "_flow_options_with_rmf":
        return

    def _flow_options_with_rmf():
        doc = (
            "Trainer-side conditional flow matching for Hamiltonian prediction. "
            "When enabled, DeePTB replaces node_h0/edge_h0 by an interpolated "
            "Hamiltonian state H_t and trains the existing model to predict the "
            "clean target Hamiltonian.  Set type/objective='rmf' to opt into the "
            "PyTorch Riemannian MeanFlow path."
        )
        args = [
            Argument("enabled", bool, optional=True, default=False),
            Argument("type", str, optional=True, default="cfm"),
            Argument("objective", str, optional=True, default="cfm"),
            Argument("manifold", str, optional=True, default="euclidean"),
            Argument("time_sampler", dict, optional=True, default={}),
            Argument("rmf_options", dict, optional=True, default={}),
            Argument("mode", str, optional=True, default="residual"),
            Argument("prior", str, optional=True, default="zero"),
            Argument("node_h0_key", str, optional=True, default="node_h0"),
            Argument("edge_h0_key", str, optional=True, default="edge_h0"),
            Argument("node_target_key", str, optional=True, default="node_features"),
            Argument("edge_target_key", str, optional=True, default="edge_features"),
            Argument("flow_time_key", str, optional=True, default="flow_time"),
            Argument("flow_time_r_key", str, optional=True, default="flow_time_r"),
            Argument("flow_time_t_key", str, optional=True, default="flow_time_t"),
            Argument("flow_time_h_key", str, optional=True, default="flow_time_h"),
            Argument("meanflow", dict, optional=True, default={}),
            Argument("time_sampling", str, optional=True, default="uniform"),
            Argument("t_min", (int, float), optional=True, default=0.0),
            Argument("t_max", (int, float), optional=True, default=0.999),
            Argument("t0_probability", (int, float), optional=True, default=0.0),
            Argument("t_eps", (int, float), optional=True, default=1.0e-3),
            Argument("time_logit_mean", (int, float), optional=True, default=-0.4),
            Argument("time_logit_std", (int, float), optional=True, default=1.0),
            Argument("node_sigma", (int, float), optional=True, default=1.0),
            Argument("edge_sigma", (int, float), optional=True, default=1.0),
            Argument("residual_sigma_floor", (int, float), optional=True, default=1.0e-6),
            Argument("loss_type", str, optional=True, default="mse"),
            Argument("node_weight", (int, float), optional=True, default=1.0),
            Argument("edge_weight", (int, float), optional=True, default=1.0),
            Argument("z_loss_coef", (int, float), optional=True, default=0.0),
            Argument("omit_time_scaling", bool, optional=True, default=True),
            Argument("endpoint_weight_power", (int, float), optional=True, default=0.0),
            Argument("endpoint_weight_cap", (int, float), optional=True, default=100.0),
            Argument("component_reduction", str, optional=True, default="global_elements"),
            Argument("validation_ode_steps", list, optional=True, default=[1, 3]),
            Argument("apply_to_reference", bool, optional=True, default=False),
            Argument("log_compatible_loss", bool, optional=True, default=False),
            Argument("log_train_compatible_loss", bool, optional=True, default=False),
            Argument("log_validation_compatible_loss", bool, optional=True, default=False),
            Argument("compatible_loss_to_legacy_keys", bool, optional=True, default=True),
            Argument("overwrite_feature_keys", bool, optional=True, default=True),
            Argument("detach_interpolated_h0", bool, optional=True, default=True),
            Argument("warn_missing_h0", bool, optional=True, default=True),
            Argument("strict_h0", bool, optional=True, default=True),
        ]
        return Argument(
            "flow_options",
            dict,
            optional=True,
            default={"enabled": False},
            sub_fields=args,
            sub_variants=[],
            doc=doc,
        )

    _argcheck.flow_options = _flow_options_with_rmf


_install_rmf_argcheck_fields()
