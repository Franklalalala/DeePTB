"""Invariant feedback from real AO blocks, after the Hamiltonian CG readout."""

import math
import re

import torch
from torch import nn
from dptb.utils.constants import anglrMId


class AOPriorToRME(nn.Module):
    """Separate physical AO H0 from the legacy embedding's RME input basis."""

    def __init__(self, idp, *, dtype, device):
        super().__init__()
        from dptb.nn.hamiltonian import E3Hamiltonian
        from dptb.data import AtomicDataDict as A

        self.node_key, self.edge_key = A.NODE_H0_KEY, A.EDGE_H0_KEY
        # The legacy CG adjoint is not generally its inverse; use the explicit
        # inverse contract so reassembling the RME recovers the physical H0.
        self.decompose = E3Hamiltonian(
            idp=idp,
            decompose=True,
            enable_inverse_cg=True,
            dtype=dtype,
            device=device,
            node_field=self.node_key,
            edge_field=self.edge_key,
        )

    def forward(self, data):
        copied = dict(data)
        copied[self.node_key] = data[self.node_key].clone()
        copied[self.edge_key] = data[self.edge_key].clone()
        result = self.decompose(copied)
        return result[self.node_key], result[self.edge_key]


class AOScalarFeedback(nn.Module):
    """One trace/sqrt(2l+1) per equal-l radial shell pair.

    OrbitalMapper calls the packed width ``reduced_matrix_element`` even after
    E3Hamiltonian converts it to AO coordinates. Irreps slices therefore cannot
    index these model outputs. Traces use the shared real spherical-harmonic
    basis on both axes; unlike diagonal coordinates they are SO(3) invariants.
    """

    def __init__(self, idp):
        super().__init__()
        idp.get_orbpair_maps()
        self.width = int(idp.reduced_matrix_element)
        self.blocks = []
        for pair, sl in idp.orbpair_maps.items():
            left, right = pair.split("-")
            li, lj = [anglrMId[re.findall(r"[a-zA-Z]", x)[0]] for x in (left, right)]
            if li == lj:
                dim = 2 * li + 1
                if sl.stop - sl.start != dim * dim:
                    raise ValueError("AO shell-pair width mismatch: " + pair)
                self.blocks.append((sl, dim))
        self.n_scalars = len(self.blocks)
        if not self.blocks:
            raise ValueError("AO feedback requires at least one equal-l shell pair")

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != self.width:
            raise ValueError("feedback requires rank-2 packed AO features")
        return torch.stack(
            [
                features[:, sl].reshape(-1, dim, dim).diagonal(dim1=-2, dim2=-1).sum(-1)
                / math.sqrt(dim)
                for sl, dim in self.blocks
            ],
            dim=-1,
        )
