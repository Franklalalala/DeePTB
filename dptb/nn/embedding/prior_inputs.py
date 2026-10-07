"""Shared H0/P input projection for the LEM and SLEM prior baselines.

Inputs use the mapper's packed AO-product layout, or coupled RME when the
batch sets ``_h0_coupled_rme``. P priors use the same layout and configurable
node/edge keys. Geometry initialization and its parameter names stay intact.
"""
import warnings

import torch
from e3nn.o3 import Linear

from dptb.configuration import resolve_init_scope
from dptb.data import _keys
from .prior_common import (
    _ao_product_to_sorted_irreps,
    _build_uureal_cg_change_of_basis,
    _get_feature_source_with_key,
    _h0_is_coupled_rme,
    _sorted_irrep_coordinate_index,
)


PRIOR_INPUT_KEYS = frozenset({
    "h0_init_scope", "use_h0_init", "use_h0_node_init", "use_h0_edge_init",
    "h0_node_key", "h0_edge_key", "h0_node_mode", "h0_merge_mode",
    "h0_self_edge_tol", "h0_ao_cg", "fallback_to_hamiltonian",
    "h0_fallback_to_hamiltonian", "fallback_node_key", "fallback_edge_key",
    "allow_target_fallback_in_training",
})


def resolve_legacy_prior_method(method, options):
    """Make legacy prior options explicit before dispatch or strict validation."""
    if method in {"lem", "slem"} and PRIOR_INPUT_KEYS.intersection(options):
        replacement = method + "_prior"
        warnings.warn(
            f"Embedding method {method!r} with prior options is mapped to "
            f"{replacement!r}. Set method explicitly in new configurations. "
            "Use h0_init_scope='none' to load a geometry-only legacy checkpoint; "
            "legacy LEM ignored prior options.",
            FutureWarning,
            stacklevel=3,
        )
        return replacement
    return method


class PriorInputs(torch.nn.Module):
    """Project selected prior fields and replace or add initial node/edge state.

    An enabled edge input must resolve before either prior is applied. Target
    fallback defaults to enabled, with the production H0 training guard.
    ``self_edge`` selects projected active zero-length self edges or direct
    node initialization, evaluating the guarded direct path in either case.
    """

    @classmethod
    def from_options(cls, base_init, options):
        selected = {key: value for key, value in options.items() if key in PRIOR_INPUT_KEYS}
        # Merely choosing the prior-capable class preserves the geometry model.
        # Once configured, the adapter uses the production default scope, both.
        if not selected:
            return None
        scope, enabled, node, edge = resolve_init_scope(
            selected.pop("h0_init_scope", None),
            enabled=selected.pop("use_h0_init", None),
            node=selected.pop("use_h0_node_init", None),
            edge=selected.pop("use_h0_edge_init", None),
            option_name="h0_init_scope",
        )
        if not enabled:
            return None
        return cls(base_init, use_node=node, use_edge=edge, **selected)

    def __init__(
        self, base_init, *, use_node=True, use_edge=True,
        h0_node_key=_keys.NODE_H0_KEY, h0_edge_key=_keys.EDGE_H0_KEY,
        h0_node_mode="direct", h0_merge_mode="replace", h0_self_edge_tol=1e-8,
        h0_ao_cg=True, fallback_to_hamiltonian=None,
        h0_fallback_to_hamiltonian=None,
        fallback_node_key=_keys.NODE_FEATURES_KEY,
        fallback_edge_key=_keys.EDGE_FEATURES_KEY,
        allow_target_fallback_in_training=False,
    ):
        super().__init__()
        if h0_node_mode not in {"direct", "self_edge"}:
            raise ValueError(f"Unsupported h0_node_mode={h0_node_mode!r}")
        if h0_merge_mode not in {"replace", "add"}:
            raise ValueError(f"Unsupported h0_merge_mode={h0_merge_mode!r}")
        if use_node and h0_node_mode == "self_edge" and not use_edge:
            raise ValueError("h0_node_mode='self_edge' requires edge initialization")
        if fallback_to_hamiltonian is None:
            fallback_to_hamiltonian = (
                True if h0_fallback_to_hamiltonian is None
                else bool(h0_fallback_to_hamiltonian)
            )
        elif (h0_fallback_to_hamiltonian is not None and
              bool(fallback_to_hamiltonian) != bool(h0_fallback_to_hamiltonian)):
            raise ValueError("Conflicting Hamiltonian fallback options")
        self.idp = base_init.idp
        irreps, index = _sorted_irrep_coordinate_index(self.idp, device=base_init.device)
        self.register_buffer("sort_index", index, persistent=False)
        self.register_buffer(
            "cg_change_of_basis",
            _build_uureal_cg_change_of_basis(
                self.idp, dtype=base_init.dtype, device=base_init.device,
            ) if h0_ao_cg else torch.empty(0, dtype=base_init.dtype, device=base_init.device),
            persistent=False,
        )
        self.register_buffer("h0_ao_cg_version", torch.tensor([int(h0_ao_cg)], device=base_init.device))
        self.h0_ao_cg = bool(h0_ao_cg)
        self.use_node, self.use_edge = use_node, use_edge
        self.h0_node_key, self.h0_edge_key = h0_node_key, h0_edge_key
        self.h0_node_mode, self.h0_merge_mode = h0_node_mode, h0_merge_mode
        self.h0_self_edge_tol = h0_self_edge_tol
        self.fallback_to_hamiltonian = bool(fallback_to_hamiltonian)
        self.fallback_node_key, self.fallback_edge_key = fallback_node_key, fallback_edge_key
        self.allow_target_fallback_in_training = bool(allow_target_fallback_in_training)
        self.node_projector = Linear(irreps, base_init.irreps_out, biases=True)
        self.edge_projector = Linear(irreps, base_init.irreps_out, biases=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        marker = state_dict.get(prefix + "h0_ao_cg_version")
        if marker is not None and not torch.equal(marker.cpu(), self.h0_ao_cg_version.cpu()):
            error_msgs.append(
                f"{prefix}h0_ao_cg_version disagrees with h0_ao_cg; "
                "load with the checkpoint's AO/RME input convention."
            )
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def _read(self, data, *, node, reference, types):
        primary = self.h0_node_key if node else self.h0_edge_key
        keys = [primary]
        if self.fallback_to_hamiltonian:
            keys.extend([
                _keys.NODE_HAMILTONIAN_KEY if node else _keys.EDGE_HAMILTONIAN_KEY,
                self.fallback_node_key if node else self.fallback_edge_key,
            ])
        source, key = _get_feature_source_with_key(
            data, [key for key in keys if key], self.idp.reduced_matrix_element,
            reference.dtype, reference.device, "node prior" if node else "edge prior",
        )
        if source is None:
            return None
        if (self.training and key != primary and
                not self.allow_target_fallback_in_training):
            raise RuntimeError(
                f"Prior input resolved target field {key!r} because {primary!r} is missing. "
                "Provide an explicit prior or set allow_target_fallback_in_training=true."
            )
        if source.ndim != 2 or source.shape[0] != types.numel():
            raise ValueError(f"Prior field {key!r} must have one row per {'node' if node else 'edge'}")
        mask = self.idp.mask_to_nrme if node else self.idp.mask_to_erme
        source = source * mask.to(source.device)[types.flatten()].to(source.dtype)
        return _ao_product_to_sorted_irreps(
            source, self.sort_index, self.cg_change_of_basis,
            coupled=not self.h0_ao_cg or (key == primary and _h0_is_coupled_rme(data)),
        )

    def _merge(self, base, prior):
        return prior if self.h0_merge_mode == "replace" else base + prior

    def forward(self, data, node_features, edge_features, atom_type, bond_type,
                edge_index, edge_length, active_edges):
        edge_prior = None
        if self.use_edge:
            source = self._read(data, node=False, reference=edge_features, types=bond_type)
            if source is None:
                return node_features, edge_features
            edge_prior = self.edge_projector(source[active_edges])
            edge_features = self._merge(edge_features, edge_prior)
        if self.use_node:
            source = self._read(data, node=True, reference=node_features, types=atom_type)
            direct_node_features = node_features
            if source is not None:
                direct_node_features = self._merge(node_features, self.node_projector(source))
            if self.h0_node_mode == "self_edge" and edge_prior is not None:
                src, dst = edge_index[:, active_edges]
                self_mask = (src == dst) & (edge_length[active_edges].reshape(-1) <= self.h0_self_edge_tol)
                node_prior = node_features.new_zeros((atom_type.numel(), node_features.shape[-1]))
                node_prior = node_prior.index_add(0, src[self_mask], edge_prior[self_mask])
                node_features = torch.where(
                    self_mask.any().reshape(1, 1),
                    self._merge(node_features, node_prior), direct_node_features,
                )
            else:
                node_features = direct_node_features
        return node_features, edge_features
