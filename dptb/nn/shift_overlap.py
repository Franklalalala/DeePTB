"""Fail-closed physical-overlap ingress, independent of e3nn and data loaders.

A packet is an explicit assertion that overlap is in the checkpoint mapper's
compact physical AO layout. Geometry equality plus directed periodic edge keys
prevents reusing S for another geometry or silently changing row order. This
module does NOT compute overlap and never substitutes identity for missing S.
"""
from __future__ import annotations

import torch
from torch import nn


def _finite_real_tensor(x, shape, name):
    if not torch.is_tensor(x) or x.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not x.is_floating_point() or x.is_complex() or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must be real finite floating-point physical AO data")


def prepare_shift_overlap(data, width, *, standard=False, dtype=None):
    """Opt-in snapshot before LEM repurposes edge_overlap. Does not reorder rows.

    Existing PHYS_* always wins; partial physical fields are rejected. Standard
    input use is an explicit caller assertion of compact physical AO semantics.
    For independently built S, use attach_overlap_packet instead.
    """
    pkeys = ("phys_node_overlap", "phys_edge_overlap")
    present = [key in data for key in pkeys]
    if any(present) and not all(present):
        raise ValueError("partial physical S input")
    source = pkeys
    if not all(present):
        if not standard:
            raise ValueError("physical S is required; supply phys_node_overlap and phys_edge_overlap or an overlap packet")
        source = ("node_overlap", "edge_overlap")
        if not all(key in data for key in source):
            raise ValueError("standard overlap input requested, but node/edge overlap is missing")
    n = len(data["pos"])
    e = data["edge_index"].shape[1]
    for key, shape in zip(source, ((n, width), (e, width))):
        _finite_real_tensor(data[key], shape, key)
    for dst, src in zip(pkeys, source):
        data[dst] = data[src] if dtype is None else data[src].to(dtype=dtype)
        # Alias is safe for LEM's dictionary REPLACEMENT when dtype already matches.
    return data


def _edge_keys(edge, shift, n):
    if edge.ndim != 2 or edge.shape[0] != 2 or shift.shape != (edge.shape[1], 3):
        raise ValueError("invalid directed periodic graph shape")
    if edge.is_floating_point() and not torch.equal(edge, edge.round()):
        raise ValueError("edge indices must be integers")
    if not bool(torch.isfinite(shift).all()) or not torch.allclose(shift, shift.round(), atol=1e-6, rtol=0):
        raise ValueError("periodic image shifts must be integer triplets")
    if bool((edge < 0).any()) or bool((edge >= n).any()):
        raise ValueError("edge endpoint outside atom range")
    rows = torch.cat((edge.T.long().cpu(), shift.round().long().cpu()), 1).tolist()
    keys = [tuple(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate directed periodic edge keys")
    return keys


def _geometry_equal(data, packet, name, *, integral=False):
    if name not in data or name not in packet:
        raise ValueError(f"overlap packet and model input both require {name}")
    a, b = torch.as_tensor(data[name]).detach().cpu(), torch.as_tensor(packet[name]).detach().cpu()
    if name == "atomic_numbers":
        a, b = a.reshape(-1), b.reshape(-1)
    if name == "cell":
        a, b = a.reshape(-1, 3, 3), b.reshape(-1, 3, 3)
    if name == "pbc":
        a, b = a.reshape(-1, 3), b.reshape(-1, 3)
        if not bool(((a == 0) | (a == 1)).all()) or not bool(((b == 0) | (b == 1)).all()):
            raise ValueError("pbc must be boolean flags")
    if a.shape != b.shape or (not torch.equal(a, b) if integral else not torch.allclose(a.double(), b.double(), atol=1e-6, rtol=0)):
        raise ValueError(f"overlap packet {name} differs from model geometry")


def attach_overlap_packet(data, packet, *, width, basis_token):
    """Join exact directed (i,j,R) keys; rejects missing, extra or duplicate edges.

    The packet MUST carry atomic_numbers,pos,cell,pbc,edge_index,edge_cell_shift,
    basis_token,layout,phys_node_overlap,phys_edge_overlap. For batches it must
    also carry batch with exactly matching atom-to-graph assignments. No
    implicit reverse-edge transposition or atom reordering is performed.
    """
    if not basis_token or packet.get("basis_token") != basis_token:
        raise ValueError("physical S basis token does not match caller's checkpoint basis")
    if packet.get("layout") != "compact_uureal_physical_ao":
        raise ValueError("overlap packet must explicitly declare compact physical AO layout")
    for name in ("atomic_numbers", "pos", "cell", "pbc"):
        _geometry_equal(data, packet, name, integral=name in {"atomic_numbers", "pbc"})
    if "batch" in data or "batch" in packet:
        _geometry_equal(data, packet, "batch", integral=True)
    n = len(data["pos"])
    if "edge_cell_shift" not in data or "edge_cell_shift" not in packet:
        raise ValueError("explicit periodic image shifts required, including all-zero shifts")
    wanted = _edge_keys(data["edge_index"], data["edge_cell_shift"], n)
    got = _edge_keys(packet["edge_index"], packet["edge_cell_shift"], n)
    if len(wanted) != len(got) or set(wanted) != set(got):
        raise ValueError("physical S and model directed periodic graphs differ")
    _finite_real_tensor(packet["phys_node_overlap"], (n, width), "packet node S")
    _finite_real_tensor(packet["phys_edge_overlap"], (len(got), width), "packet edge S")
    lookup = {key: row for row, key in enumerate(got)}
    src = packet["phys_edge_overlap"]
    order = torch.tensor([lookup[key] for key in wanted], device=src.device, dtype=torch.long)
    data["phys_node_overlap"] = packet["phys_node_overlap"].to(device=data["pos"].device)
    data["phys_edge_overlap"] = src.index_select(0, order).to(device=data["pos"].device)
    return data


class PhysicalOverlapAdapter(nn.Module):
    """Inference wrapper around an existing H0/S provider callable.

    provider(data) -> a packet in this module's contract. Use SAME physical S
    for the potential correction and generalized band problem. Set wrapper.eval()
    before band evaluation. Provider may read H0/radial tables, NEVER H labels.
    """
    def __init__(self, model, provider, *, basis_token):
        super().__init__()
        self.model = model
        self.provider = provider
        self.basis_token = basis_token

    def forward(self, data):
        prepared = dict(data)
        packet = self.provider(dict(data))
        attach_overlap_packet(prepared, packet,
                              width=self.model.idp.reduced_matrix_element,
                              basis_token=self.basis_token)
        for key in ("phys_node_overlap", "phys_edge_overlap"):
            prepared[key] = prepared[key].to(dtype=self.model.dtype)
        return self.model(prepared)
