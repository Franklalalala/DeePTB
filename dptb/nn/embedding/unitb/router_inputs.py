"""Stateless router-input strategies; no parameters or persistent buffers."""
from abc import ABC, abstractmethod
import torch


class PDQMoERouterInput(ABC):
    @abstractmethod
    def __call__(self, owner, data, bond_type, active_edges, onehot, vectors):
        """Produce router features without reading a target Hamiltonian."""


class BondTypeInput(PDQMoERouterInput):
    def __call__(self, owner, data, bond_type, active_edges, onehot, vectors):
        return onehot


class BondDistanceInput(PDQMoERouterInput):
    def __call__(self, owner, data, bond_type, active_edges, onehot, vectors):
        length = vectors[active_edges].norm(dim=-1, keepdim=True).to(onehot.dtype)
        centers = owner._router_rbf_centers.to(dtype=length.dtype, device=length.device)
        rbf = torch.exp(-0.5 * ((length - centers) / owner._router_rbf_width) ** 2)
        return torch.cat([onehot, rbf], dim=-1)


class PriorGramInput(PDQMoERouterInput):
    def __call__(self, owner, data, bond_type, active_edges, onehot, vectors):
        descriptor = owner._gram_descriptor(owner._raw_prior_source(data, bond_type, active_edges))
        return torch.cat([onehot, descriptor.to(onehot.dtype)], dim=-1)


def router_input_strategy(name, *, per_edge):
    """Select once at construction, keeping alternatives out of the main forward."""
    if not per_edge:
        return BondTypeInput()
    return {"onehot": BondTypeInput, "onehot_r": BondDistanceInput,
            "onehot_prior": PriorGramInput}[name]()
