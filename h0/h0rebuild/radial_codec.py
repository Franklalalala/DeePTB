"""Lossless structural codec for uniform cubic Hermite radial coefficients.

Keep each segment's value and derivative at its left node. Interior curvature
coefficients follow from the next node using the original extraction arithmetic;
the final segment keeps its two curvatures explicitly. Sparse bit corrections
preserve signed zeros and encoding-environment rounding differences. Decoding
must reproduce the original byte hash; other arithmetic environments may reject.
This codec
is experimental and is not silently substituted for the frozen offline format.
"""
import math
import hashlib
import numpy as np


def _derive(nodes, tail, dr):
    rows, segments, _ = nodes.shape
    out = np.empty((rows, segments, 4), dtype=np.float64)
    out[:, :, :2] = nodes
    inv_dr = 1.0 / dr
    inv_dr2 = inv_dr * inv_dr
    dd = (nodes[:, 1:, 0] - nodes[:, :-1, 0]) * inv_dr
    c1 = nodes[:, :-1, 1]
    c3 = (c1 + nodes[:, 1:, 1] - 2.0 * dd) * inv_dr2
    out[:, :-1, 3] = c3
    out[:, :-1, 2] = (dd - c1) * inv_dr - c3 * dr
    out[:, -1, 2:] = tail
    return out


def pack(coeffs, dr):
    c = np.asarray(coeffs)
    dr = float(dr)
    if c.dtype != np.float64 or c.ndim != 3 or c.shape[1] < 1 or c.shape[2] != 4:
        raise ValueError('Expected float64 [curves, segments>=1, 4]')
    if not math.isfinite(dr) or dr <= 0 or not np.isfinite(c).all():
        raise ValueError('Coefficients and positive grid step must be finite')
    c = np.ascontiguousarray(c)
    nodes = c[:, :, :2].copy()
    tail = c[:, -1, 2:].copy()
    with np.errstate(all='ignore'):
        restored = _derive(nodes, tail, dr)
    bits = c.view(np.uint64).reshape(-1)
    expected = restored.view(np.uint64).reshape(-1)
    indices = np.flatnonzero(bits != expected).astype(np.int64)
    return dict(schema=2, dr=dr, nodes=nodes, tail=tail,
                checksum=hashlib.sha256(c.tobytes()).hexdigest(),
                correction_indices=indices, correction_bits=bits[indices].copy())


def unpack(record):
    if record['schema'] != 2:
        raise ValueError('Unsupported Hermite codec schema')
    with np.errstate(all='ignore'):
        out = _derive(record['nodes'], record['tail'], record['dr'])
    out.view(np.uint64).reshape(-1)[record['correction_indices']] = record['correction_bits']
    if hashlib.sha256(out.tobytes()).hexdigest() != record['checksum']:
        raise ValueError('Hermite decoded checksum mismatch: incompatible arithmetic or corrupt data')
    return out


def payload_bytes(record):
    return sum(value.nbytes for value in record.values() if isinstance(value, np.ndarray))
