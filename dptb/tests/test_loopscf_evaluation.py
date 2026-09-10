"""Chunked spectra must preserve full-path alignment and nonorthogonal physics."""

import importlib.util
from pathlib import Path

import torch
from dptb.data import AtomicDataDict as A
from dptb.data.transforms import OrbitalMapper
from dptb.nnops.loopscf.spectral import _fw10_one_graph


def test_chunked_nonorthogonal_path_keeps_global_vbm():
    path = Path(__file__).resolve().parents[2] / "examples/loopscf/evaluate.py"
    spec = importlib.util.spec_from_file_location("loopscf_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    idp = OrbitalMapper({"H": "1s"}, method="e3tb")
    ref = {
        A.ATOM_TYPE_KEY: torch.tensor([0, 0]),
        A.EDGE_INDEX_KEY: torch.tensor([[0, 1], [1, 0]]),
        A.EDGE_CELL_SHIFT_KEY: torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64
        ),
        A.NODE_OVERLAP_KEY: torch.tensor([[1.0], [1.3]], dtype=torch.float64),
        A.EDGE_OVERLAP_KEY: torch.tensor([[0.1], [0.1]], dtype=torch.float64),
        A.NODE_H0_KEY: torch.tensor([[-1.0], [2.0]], dtype=torch.float64),
        A.EDGE_H0_KEY: torch.tensor([[0.2], [0.2]], dtype=torch.float64),
    }
    kp = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.13, 0.0, 0.0],
            [0.3, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    zero = (
        torch.zeros(2, 1, dtype=torch.float64),
        torch.zeros(2, 1, dtype=torch.float64),
    )
    output = {
        "label": zero,
        "prediction": (
            torch.tensor([[0.1], [0.3]], dtype=torch.float64),
            torch.tensor([[0.2], [0.2]], dtype=torch.float64),
        ),
    }
    full, full_diag = module.path_bands(ref, output, idp, kp, 5)
    chunks, chunk_diag = module.path_bands(ref, output, idp, kp, 2)
    assert full_diag == chunk_diag
    for name in output:
        torch.testing.assert_close(chunks[name], full[name], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        _fw10_one_graph(chunks["prediction"], chunks["label"], 2, 10)[0],
        _fw10_one_graph(full["prediction"], full["label"], 2, 10)[0],
    )
    # A k-dependent common energy shift exposes erroneous per-chunk alignment:
    # full-path alignment retains relative dispersion across the chunks.
    labels = torch.tensor([[-2.0, 2.0], [-1.0, 3.0], [0.0, 4.0]], dtype=torch.float64)
    prediction = labels + torch.tensor([[0.0], [1.0], [3.0]], dtype=torch.float64)
    global_error = _fw10_one_graph(prediction, labels, 2, 10)[0]
    local_error = torch.stack(
        [
            _fw10_one_graph(p[None], r[None], 2, 10)[0]
            for p, r in zip(prediction, labels)
        ]
    ).mean()
    assert global_error > 1 and local_error == 0
